"""
Code Blue ECG — 6-Experiment Scientific Diagnostic Suite on 360-Patient DEV Set
- Strictly evaluates on the 360-patient Fold-10 DEV cohort (SEED=20261126, 1 ECG/patient_id).
- NEVER tunes thresholds on the DEV set: Fold-9 thresholds (Exp 5) are chosen strictly on Fold 9.
- Prints one structured table per experiment with n, 95% Wilson CI, and exact paired McNemar p-values.

Usage (Colab / Kaggle):
  !python3 -u run_dev360_experiments.py --ptb_dir /content/ptbxl --ckpt /path/to/ecg_densenet121_v3.pt
"""
import argparse
import ast
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import wfdb
from sklearn.metrics import roc_auc_score

SEED = 20261126
CRITICAL_CLASSES = [
    "ANTERIOR_SEPTAL_MI",
    "INFERIOR_POSTERIOR_MI",
    "LATERAL_MI",
    "HIGH_GRADE_AV_BLOCK",
]
MI_CLASSES = ["ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI", "NSTEMI_ISCHEMIA"]
ACUTE_IMI_CODES = {"INJIN", "INJIL"}
OLD_IMI_CODES = {"IMI", "PMI", "ILMI", "IPMI"}

SCP_MAP = {
    "NORMAL_SINUS_RHYTHM": {"NORM"},
    "ANTERIOR_SEPTAL_MI": {"AMI", "ASMI", "ALMI", "INJAS", "INJAP"},
    "INFERIOR_POSTERIOR_MI": {"IMI", "PMI", "ILMI", "IPMI", "INJIN", "INJIL"},
    "LATERAL_MI": {"LMI", "ALMI", "ILMI", "INJLA"},
    "NSTEMI_ISCHEMIA": {"ISC_", "ISCAL", "ISCAN", "ISCAS", "ISCIL", "ISCIN", "ISCLA", "NST_", "STD_"},
    "ATRIAL_FIBRILLATION": {"AFIB"},
    "ATRIAL_FLUTTER": {"AFLT"},
    "SVT_TACHYCARDIA": {"SVTAC", "PSVT", "STACH"},
    "SINUS_BRADYCARDIA": {"SBRAD"},
    "FIRST_DEGREE_AV_BLOCK": {"1AVB"},
    "HIGH_GRADE_AV_BLOCK": {"2AVB", "3AVB"},
    "RBBB": {"CRBBB", "IRBBB"},
    "LBBB": {"CLBBB", "ILBBB"},
    "LAFB_LPFB": {"LAFB", "LPFB"},
    "LVH_RVH": {"LVH", "RVH"},
    "PACEMAKER_RHYTHM": {"PACE"},
    "WPW_PREEXCITATION": {"WPW"},
    "LONG_QT_OR_ECTOPY": {"LNGQT", "PVC", "BIGU", "TRIGU"},
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt_ci(k: int, n: int) -> str:
    p, lo, hi = wilson_ci(k, n)
    return f"{k}/{n} ({p*100:.1f}% [{lo*100:.1f},{hi*100:.1f}])"


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    """Exact two-sided paired McNemar test comparing boolean arrays b (new) vs a (base)."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return n01, n10, p


def load_cohort(ptb_dir: str, fold: int, target_n: int, seed: int) -> List[Dict]:
    df = pd.read_csv(os.path.join(ptb_dir, "ptbxl_database.csv"))
    sub = df[df["strat_fold"] == fold].copy()
    rng = random.Random(seed)
    rows = list(sub.iterrows())
    rng.shuffle(rows)

    by_bucket = defaultdict(list)
    seen_pid: Set[int] = set()
    for _, row in rows:
        pid = int(row["patient_id"])
        if pid in seen_pid:
            continue
        scp = {k: float(v) for k, v in ast.literal_eval(row["scp_codes"]).items() if float(v) >= 50.0}
        matched = sorted([cls for cls, sset in SCP_MAP.items() if set(scp.keys()) & sset])
        if not matched:
            continue
        seen_pid.add(pid)
        crit = [c for c in matched if c in CRITICAL_CLASSES]
        bucket = crit[0] if crit else matched[0]
        by_bucket[bucket].append({
            "rec_path": os.path.join(ptb_dir, str(row["filename_lr"])),
            "patient_id": pid,
            "gt": matched,
            "scp": scp,
        })

    selected: List[Dict] = []
    per_cls = max(12, target_n // max(1, len(by_bucket)))
    for b in sorted(by_bucket.keys()):
        selected.extend(by_bucket[b][:per_cls])
    used = {x["patient_id"] for x in selected}
    for b in sorted(by_bucket.keys()):
        for item in by_bucket[b]:
            if len(selected) >= target_n:
                break
            if item["patient_id"] not in used:
                selected.append(item)
                used.add(item["patient_id"])
    return selected[:target_n]


def render_ecg(sig: np.ndarray, rng: random.Random, h: int = 384, w: int = 512) -> np.ndarray:
    paper = rng.choice(["pink", "cream", "gray"])
    bg = (242, 236, 252) if paper == "pink" else ((232, 242, 248) if paper == "cream" else (244, 244, 244))
    gc = (210, 195, 240) if paper == "pink" else ((205, 220, 230) if paper == "cream" else (215, 215, 215))
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    step = max(12, int(round(16 * w / 512)))
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), gc, 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), gc, 1)

    layout = rng.choice(["3x4_1", "3x4_3"])
    n_rows = 4 if layout == "3x4_1" else 6
    row_h, col_w, T = h // n_rows, w // 4, sig.shape[0]
    thick = max(1, int(round(w / 512)))
    for r in range(3):
        for c in range(4):
            li = c * 3 + r
            if li >= sig.shape[1]:
                continue
            seg = sig[int(c * T / 4):int((c + 1) * T / 4), li]
            xs = np.linspace(c * col_w + 6, (c + 1) * col_w - 6, len(seg)).astype(np.int32)
            cy = int((r + 0.5) * row_h)
            ys = np.clip(cy - (seg * (row_h * 0.28)).astype(np.int32), 2, h - 3)
            cv2.polylines(img, [np.stack([xs, ys], axis=1).reshape(-1, 1, 2)], False, (25, 25, 25), thick, cv2.LINE_AA)
    r_leads = [1] if layout == "3x4_1" else [1, 6, 10]
    for ri, li in enumerate(r_leads):
        seg = sig[:, min(li, sig.shape[1] - 1)]
        xs = np.linspace(8, w - 8, len(seg)).astype(np.int32)
        cy = int((3 + ri + 0.5) * row_h)
        ys = np.clip(cy - (seg * (row_h * 0.28)).astype(np.int32), 2, h - 3)
        cv2.polylines(img, [np.stack([xs, ys], axis=1).reshape(-1, 1, 2)], False, (25, 25, 25), thick, cv2.LINE_AA)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img


def preprocess(img_bgr: np.ndarray, h: int, w: int, dilate_1px: bool = False) -> torch.Tensor:
    img = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (31, 31), 0)
    diff = np.clip(bg.astype(np.float32) - gray.astype(np.float32), 0, 255)
    p99 = max(12.0, float(np.percentile(diff, 99.5)))
    ink = np.clip(diff / p99, 0.0, 1.0)
    if dilate_1px:
        k = np.ones((2, 2), np.uint8)
        ink = cv2.dilate(ink, k, iterations=1)
    rgb = np.stack([1.0 - ink] * 3, axis=-1).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def infer_probs(model, img_bgr: np.ndarray, device, T_cal: float, b_cal: float, h: int = 384, w: int = 512, dilate: bool = False) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    x = preprocess(img_bgr, h, w, dilate).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).cpu().numpy()[0]
    probs = 1.0 / (1.0 + np.exp(-np.clip((logits - b_cal) / T_cal, -30, 30)))
    return probs, (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptb_dir", default="/content/ptbxl")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_dev", type=int, default=360)
    ap.add_argument("--n_fold9", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes: List[str] = list(ckpt["classes"])
    c2i = {c: i for i, c in enumerate(classes)}
    T_cal = max(1.15, float(ckpt.get("calib_temp", 1.1721)))
    b_cal = max(1.50, float(ckpt.get("calib_bias", 3.0927)))

    model = timm.create_model("densenet121", pretrained=False, num_classes=len(classes), drop_rate=0.2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Device={device} | Ckpt={os.path.basename(args.ckpt)} | T={T_cal:.3f}, b={b_cal:.3f}")
    dev_cohort = load_cohort(args.ptb_dir, fold=10, target_n=args.n_dev, seed=SEED)
    rng_dev = random.Random(SEED)
    # Render high-res base images once (768x1024) so all resolution ablations see identical ECGs
    dev_imgs = []
    for rec in dev_cohort:
        sig, _ = wfdb.rdsamp(rec["rec_path"])
        dev_imgs.append(render_ecg(np.nan_to_num(sig), rng_dev, h=768, w=1024))

    Y_dev = np.zeros((len(dev_cohort), len(classes)), dtype=int)
    for idx, rec in enumerate(dev_cohort):
        for c in rec["gt"]:
            if c in c2i:
                Y_dev[idx, c2i[c]] = 1

    # Baseline inference at 384x512
    P_base = np.zeros((len(dev_cohort), len(classes)), dtype=np.float32)
    lat_base = []
    for i, img in enumerate(dev_imgs):
        P_base[i], ms = infer_probs(model, img, device, T_cal, b_cal, 384, 512, False)
        lat_base.append(ms)

    # =========================================================================
    # EXP 1: Multi-Label Truth Re-Scoring + MI-vs-AFIB Confusion
    # =========================================================================
    print("\n================ EXP 1: MULTI-LABEL TRUTH RE-SCORING (DEV N=360, op=0.55) ================")
    print(f"{'Class':24s} | {'n_pos':>5s} | {'Sens@0.55 (95% CI)':28s} | {'AUROC':>6s}")
    for c in MI_CLASSES + ["ATRIAL_FIBRILLATION", "HIGH_GRADE_AV_BLOCK", "NORMAL_SINUS_RHYTHM"]:
        j = c2i[c]
        yt = Y_dev[:, j]
        yp = (P_base[:, j] >= 0.55).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        npos = int(yt.sum())
        auc = roc_auc_score(yt, P_base[:, j]) if 0 < npos < len(yt) else float("nan")
        print(f"{c:24s} | {npos:5d} | {fmt_ci(tp, npos):28s} | {auc:6.4f}")

    # MI-vs-AFIB confusion under Single-Label Top-1 vs Multi-Label
    mi_idxs = [c2i[c] for c in ["ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI"]]
    af_idx = c2i["ATRIAL_FIBRILLATION"]
    has_mi = Y_dev[:, mi_idxs].sum(axis=1) > 0
    has_af = Y_dev[:, af_idx] == 1
    top1_af = np.argmax(P_base, axis=1) == af_idx
    mi_pred_af_total = int((has_mi & top1_af).sum())
    mi_pred_af_true_coexist = int((has_mi & top1_af & has_af).sum())
    mi_pred_af_false_af = int((has_mi & top1_af & ~has_af).sum())
    mi_pred_af_mi_also_pos = int((has_mi & top1_af & (P_base[:, mi_idxs].max(axis=1) >= 0.55)).sum())
    print(f"\nMI-vs-AFIB Confusion (N_MI={int(has_mi.sum())}):")
    print(f"  Single-label Top-1=AFIB on MI patients: {mi_pred_af_total}/{int(has_mi.sum())}")
    print(f"    -> True co-existing AFIB in PTB-XL truth: {mi_pred_af_true_coexist}/{mi_pred_af_total}")
    print(f"    -> True false-positive AFIB (no AFIB in truth): {mi_pred_af_false_af}/{mi_pred_af_total}")
    print(f"    -> Multi-label MI also >=0.55 alongside AFIB: {mi_pred_af_mi_also_pos}/{mi_pred_af_total}")

    # =========================================================================
    # EXP 2: Stratify IMI & IMI->NSR Misses by SCP Likelihood (>=80) & Acute vs Old
    # =========================================================================
    print("\n================ EXP 2: INFERIOR_POSTERIOR_MI STRATIFICATION ================")
    imi_idx, nsr_idx = c2i["INFERIOR_POSTERIOR_MI"], c2i["NORMAL_SINUS_RHYTHM"]
    strata = defaultdict(lambda: [0, 0, 0])  # [n, detected_imi, missed_as_nsr]
    for i, rec in enumerate(dev_cohort):
        if Y_dev[i, imi_idx] == 0:
            continue
        scp = rec["scp"]
        imi_scores = [scp[k] for k in scp if k in (ACUTE_IMI_CODES | OLD_IMI_CODES)]
        max_lik = max(imi_scores) if imi_scores else 50.0
        lik_bin = "lik>=80" if max_lik >= 80.0 else "lik_50_79"
        is_acute = any(k in ACUTE_IMI_CODES for k in scp)
        age_bin = "acute_INJ" if is_acute else "old_chronic_MI"
        det = int(P_base[i, imi_idx] >= 0.55)
        nsr_miss = int(np.argmax(P_base[i]) == nsr_idx and det == 0)
        for key in (lik_bin, age_bin, f"{lik_bin}+{age_bin}"):
            strata[key][0] += 1
            strata[key][1] += det
            strata[key][2] += nsr_miss
    print(f"{'Stratum':26s} | {'n':>4s} | {'IMI Sens@0.55 (95% CI)':28s} | {'IMI->NSR Miss':>13s}")
    for k in sorted(strata.keys()):
        n_s, det_s, nsr_s = strata[k]
        print(f"{k:26s} | {n_s:4d} | {fmt_ci(det_s, n_s):28s} | {nsr_s:13d}")

    # =========================================================================
    # EXP 3: Resolution & Preprocessing Ablation for IMI Sensitivity
    # =========================================================================
    print("\n================ EXP 3: RESOLUTION & DILATION ABLATION (IMI Sens / NORM FPR / Latency) ================")
    imi_mask = Y_dev[:, imi_idx] == 1
    pure_norm_mask = (Y_dev.sum(axis=1) == 1) & (Y_dev[:, nsr_idx] == 1)
    base_imi_hit = (P_base[imi_mask, imi_idx] >= 0.55).astype(int)

    configs = [
        ("384x512_baseline", 384, 512, False),
        ("512x768_area", 512, 768, False),
        ("768x1024_area", 768, 1024, False),
        ("384x512_area+1px_dilate", 384, 512, True),
    ]
    P_res = {}
    lat_res = {}
    print(f"{'Config':25s} | {'IMI Sens (95% CI)':28s} | {'NORM FPR (95% CI)':26s} | {'Lat(ms)':>7s} | {'McNemar p':>9s}")
    for name, hh, ww, dil in configs:
        if name == "384x512_baseline":
            P_cur, lats = P_base, lat_base
        else:
            P_cur = np.zeros_like(P_base)
            lats = []
            for i, img in enumerate(dev_imgs):
                P_cur[i], ms = infer_probs(model, img, device, T_cal, b_cal, hh, ww, dil)
                lats.append(ms)
        P_res[name], lat_res[name] = P_cur, lats
        imi_hit = (P_cur[imi_mask, imi_idx] >= 0.55).astype(int)
        norm_fp = int((P_cur[pure_norm_mask, imi_idx] >= 0.55).sum())
        _, _, pval = mcnemar_exact(base_imi_hit, imi_hit)
        print(f"{name:25s} | {fmt_ci(int(imi_hit.sum()), int(imi_mask.sum())):28s} | {fmt_ci(norm_fp, int(pure_norm_mask.sum())):26s} | {np.mean(lats):7.1f} | {pval:9.4f}")

    # =========================================================================
    # EXP 4: Two-Stage Cascade Prototype (Low-Res -> High-Res + Inferior Crop on Normal/Uncertain)
    # =========================================================================
    print("\n================ EXP 4: TWO-STAGE CASCADE PROTOTYPE ================")
    P_hi = P_res["768x1024_area"]
    casc_imi = np.zeros(len(dev_cohort), dtype=int)
    casc_lats = []
    triggered = 0
    for i, img in enumerate(dev_imgs):
        top1_c = int(np.argmax(P_base[i]))
        max_p = float(P_base[i, top1_c])
        is_norm_or_unc = (top1_c == nsr_idx) or (max_p < 0.55)
        if not is_norm_or_unc:
            casc_imi[i] = int(P_base[i, imi_idx] >= 0.55)
            casc_lats.append(lat_base[i])
        else:
            triggered += 1
            # Inferior-lead crop (left 50% limb leads II/III/aVF + bottom rhythm strip)
            H0, W0 = img.shape[:2]
            inf_crop = np.vstack([img[:int(H0 * 0.55), :int(W0 * 0.52)], img[int(H0 * 0.55):, :int(W0 * 0.52)]])
            p_crop, ms_c = infer_probs(model, inf_crop, device, T_cal, b_cal, 384, 512, False)
            p_comb = max(float(P_base[i, imi_idx]), float(P_hi[i, imi_idx]), float(p_crop[imi_idx]))
            casc_imi[i] = int(p_comb >= 0.55)
            casc_lats.append(lat_base[i] + lat_res["768x1024_area"][i] + ms_c)

    casc_hit = casc_imi[imi_mask]
    n01, n10, p_casc = mcnemar_exact(base_imi_hit, casc_hit)
    extra_norm_fp = int(casc_imi[pure_norm_mask].sum()) - int((P_base[pure_norm_mask, imi_idx] >= 0.55).sum())
    print(f"Triggered Stage-2 on: {triggered}/{len(dev_cohort)} ({triggered*100/len(dev_cohort):.1f}%)")
    print(f"Baseline IMI Sens: {fmt_ci(int(base_imi_hit.sum()), int(imi_mask.sum()))}")
    print(f"Cascade  IMI Sens: {fmt_ci(int(casc_hit.sum()), int(imi_mask.sum()))} | gained={n01}, lost={n10}, McNemar p={p_casc:.4f}")
    print(f"Extra NORM False Alarms: +{extra_norm_fp}/{int(pure_norm_mask.sum())} | Mean Latency: {np.mean(casc_lats):.1f} ms (base {np.mean(lat_base):.1f} ms)")

    # =========================================================================
    # EXP 5: Multi-Label Per-Class Thresholds Chosen Strictly on Fold 9 (Sens >= 0.90)
    # =========================================================================
    print("\n================ EXP 5: FOLD-9 PER-CLASS THRESHOLDS (Crit Sens >= 0.90) -> DEV EVAL ================")
    f9_cohort = load_cohort(args.ptb_dir, fold=9, target_n=args.n_fold9, seed=SEED + 9)
    rng_f9 = random.Random(SEED + 9)
    P_f9 = np.zeros((len(f9_cohort), len(classes)), dtype=np.float32)
    Y_f9 = np.zeros((len(f9_cohort), len(classes)), dtype=int)
    for i, rec in enumerate(f9_cohort):
        sig, _ = wfdb.rdsamp(rec["rec_path"])
        img9 = render_ecg(np.nan_to_num(sig), rng_f9, h=384, w=512)
        P_f9[i], _ = infer_probs(model, img9, device, T_cal, b_cal, 384, 512, False)
        for c in rec["gt"]:
            if c in c2i:
                Y_f9[i, c2i[c]] = 1

    print(f"{'Critical Class':24s} | {'F9_Th':>5s} | {'DEV Sens (95% CI)':28s} | {'DEV PPV (95% CI)':26s} | {'McNemar p':>9s}")
    for c in CRITICAL_CLASSES:
        j = c2i[c]
        pos_probs = np.sort(P_f9[Y_f9[:, j] == 1, j])
        th = float(np.quantile(pos_probs, 0.10)) if len(pos_probs) > 0 else 0.55
        th = max(0.05, min(0.90, th))
        yt = Y_dev[:, j]
        yp_base = (P_base[:, j] >= 0.55).astype(int)
        yp_new = (P_base[:, j] >= th).astype(int)
        tp = int(((yt == 1) & (yp_new == 1)).sum())
        pred_pos = int(yp_new.sum())
        _, _, p_th = mcnemar_exact(yp_base[yt == 1], yp_new[yt == 1])
        print(f"{c:24s} | {th:5.3f} | {fmt_ci(tp, int(yt.sum())):28s} | {fmt_ci(tp, pred_pos):26s} | {p_th:9.4f}")

    # =========================================================================
    # EXP 6: Full Displayed Output with Safety Floor (Tier 1 CRITICAL + Multi-Diagnosis List <= 4)
    # =========================================================================
    print("\n================ EXP 6: FULL DISPLAYED OUTPUT WITH SAFETY FLOOR (DEV N=360) ================")
    crit_set = set(CRITICAL_CLASSES)
    crit_indices = [i for i, rec in enumerate(dev_cohort) if set(rec["gt"]) & crit_set]
    top1_hits = []
    disp_hits = []
    disp_refused = 0
    for i in crit_indices:
        gt_crit = set(dev_cohort[i]["gt"]) & crit_set
        probs = P_base[i]
        # Top-1 raw at 0.55
        top1_c = classes[int(np.argmax(probs))]
        top1_hits.append(1 if (probs.max() >= 0.55 and top1_c in gt_crit) else 0)
        # Displayed multi-diagnosis list with Safety Floor: all confirmed >=0.55, Tier 1 CRITICAL first, max 4
        confirmed = [(classes[k], float(probs[k])) for k in range(len(classes)) if probs[k] >= 0.55]
        confirmed.sort(key=lambda x: (0 if x[0] in crit_set else 1, -x[1]))
        displayed = [x[0] for x in confirmed[:4]]
        if not displayed:
            disp_refused += 1
        disp_hits.append(1 if any(c in gt_crit for c in displayed) else 0)

    a1, a2 = np.array(top1_hits), np.array(disp_hits)
    n01, n10, p_sf = mcnemar_exact(a1, a2)
    print(f"True Critical Cases (n={len(crit_indices)}):")
    print(f"  Single-Label Top-1@0.55 Hit:     {fmt_ci(int(a1.sum()), len(crit_indices))}")
    print(f"  Full Displayed List (SafetyFlr): {fmt_ci(int(a2.sum()), len(crit_indices))} | gained={n01}, lost={n10}, McNemar p={p_sf:.4f}")
    print(f"  Critical Refusal (Empty List):   {fmt_ci(disp_refused, len(crit_indices))}")


if __name__ == "__main__":
    main()
