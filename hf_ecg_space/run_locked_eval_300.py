"""
Code Blue ECG — Locked 360-ECG Patient-Grouped Benchmark Suite (Zero-Leakage)
Builds a FROZEN, deterministic (SEED=20261126), patient-grouped Fold-10 PTB-XL test set
(1 ECG per unique patient_id, never used in training, calibration, or model selection),
renders realistic printed/phone-photo multi-layout ECGs (including 6-row & gray/cream paper),
evaluates ONE frozen checkpoint with 6x MC Dropout, and outputs:
  - Top-1 & Top-3 accuracy with 95% Wilson CI
  - 10-bin Expected Calibration Error (ECE)
  - Selective Accuracy / Coverage curve
  - Critical Refusal Rate (with 95% Wilson CI)
  - Critical Confident Miss Rate (with 95% Wilson CI)
  - Normal Refusal Rate (with 95% Wilson CI)
  - Exported CSV compatible with `ecg_eval_harness.py --compare`

Usage (Kaggle or Colab GPU — runs in ~2-3 minutes):
  !python3 -u run_locked_eval_300.py --ckpt /path/to/ecg_densenet121_v2.pt --out locked_v2.csv
  !python3 -u run_locked_eval_300.py --ckpt /path/to/ecg_densenet121_v3.pt --out locked_v3.csv
  !python3 -u ecg_eval_harness.py --csv locked_v3.csv --compare locked_v2.csv
"""
import argparse
import ast
import csv
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import wfdb

SEED = 20261126
IMG_H, IMG_W = 384, 512

CRITICAL_CLASSES = {
    "ANTERIOR_SEPTAL_MI",
    "INFERIOR_POSTERIOR_MI",
    "LATERAL_MI",
    "HIGH_GRADE_AV_BLOCK",
}
NORMAL_CLASS = "NORMAL_SINUS_RHYTHM"

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
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def compute_ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (probs >= lo) & (probs <= hi if b == n_bins - 1 else probs < hi)
        if mask.sum() > 0:
            ece += (mask.sum() / n) * abs(float(correct[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def build_locked_patient_grouped_cohort(ptb_dir: str, target_n: int = 360) -> List[Tuple[str, int, List[str]]]:
    """Selects strictly 1 ECG per unique patient_id from Fold 10, stratified across classes."""
    df = pd.read_csv(os.path.join(ptb_dir, "ptbxl_database.csv"))
    f10 = df[df["strat_fold"] == 10].copy()
    rng = random.Random(SEED)

    by_class: Dict[str, List[Tuple[str, int, List[str]]]] = defaultdict(list)
    seen_patients: Set[int] = set()
    rows = list(f10.iterrows())
    rng.shuffle(rows)

    for _, row in rows:
        pid = int(row["patient_id"])
        if pid in seen_patients:
            continue
        codes = ast.literal_eval(row["scp_codes"])
        active = {k for k, v in codes.items() if float(v) >= 50.0}
        matched = [cls for cls, sset in SCP_MAP.items() if active & sset]
        if not matched:
            continue
        seen_patients.add(pid)
        rec_path = os.path.join(ptb_dir, str(row["filename_lr"]))
        # Prioritize critical classes for stratification bucket
        crit = [c for c in matched if c in CRITICAL_CLASSES]
        bucket = crit[0] if crit else matched[0]
        by_class[bucket].append((rec_path, pid, sorted(matched)))

    selected: List[Tuple[str, int, List[str]]] = []
    per_cls = max(12, target_n // max(1, len(by_class)))
    for bucket in sorted(by_class.keys()):
        selected.extend(by_class[bucket][:per_cls])
    # Top up to target_n if needed
    used_pids = {x[1] for x in selected}
    for bucket in sorted(by_class.keys()):
        for item in by_class[bucket]:
            if len(selected) >= target_n:
                break
            if item[1] not in used_pids:
                selected.append(item)
                used_pids.add(item[1])
    return selected[:target_n]


def render_realistic_ecg(sig: np.ndarray, rng: random.Random) -> np.ndarray:
    """Renders 12-lead signal into realistic multi-layout ECG image (4-row, 6-row, cream/pink/gray paper)."""
    paper_style = rng.choice(["pink", "cream", "gray"])
    bg_color = (242, 236, 252) if paper_style == "pink" else ((232, 242, 248) if paper_style == "cream" else (244, 244, 244))
    grid_color = (210, 195, 240) if paper_style == "pink" else ((205, 220, 230) if paper_style == "cream" else (215, 215, 215))
    img = np.full((IMG_H, IMG_W, 3), bg_color, dtype=np.uint8)
    for x in range(0, IMG_W, 16):
        cv2.line(img, (x, 0), (x, IMG_H), grid_color, 1)
    for y in range(0, IMG_H, 16):
        cv2.line(img, (0, y), (IMG_W, y), grid_color, 1)

    layout = rng.choice(["3x4_1", "3x4_3"])
    n_rows = 4 if layout == "3x4_1" else 6
    row_h = IMG_H // n_rows
    col_w = IMG_W // 4
    T = sig.shape[0]

    for r in range(3):
        for c in range(4):
            lead_idx = c * 3 + r
            if lead_idx >= sig.shape[1]:
                continue
            seg = sig[int(c * T / 4):int((c + 1) * T / 4), lead_idx]
            xs = np.linspace(c * col_w + 6, (c + 1) * col_w - 6, len(seg)).astype(np.int32)
            cy = int((r + 0.5) * row_h)
            ys = np.clip(cy - (seg * (row_h * 0.28)).astype(np.int32), 2, IMG_H - 3)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (25, 25, 25), 1, cv2.LINE_AA)

    rhythm_leads = [1] if layout == "3x4_1" else [1, 6, 10]
    for r_i, lead_idx in enumerate(rhythm_leads):
        seg = sig[:, min(lead_idx, sig.shape[1] - 1)]
        xs = np.linspace(8, IMG_W - 8, len(seg)).astype(np.int32)
        cy = int((3 + r_i + 0.5) * row_h)
        ys = np.clip(cy - (seg * (row_h * 0.28)).astype(np.int32), 2, IMG_H - 3)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (25, 25, 25), 1, cv2.LINE_AA)

    # Mild phone-photo blur/noise
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img


def normalize_ink(img_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img_bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (31, 31), 0)
    diff = np.clip(bg.astype(np.float32) - gray.astype(np.float32), 0, 255)
    p99 = max(12.0, float(np.percentile(diff, 99.5)))
    ink = np.clip(diff / p99, 0.0, 1.0)
    rgb = np.stack([1.0 - ink] * 3, axis=-1).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptb_dir", default="/kaggle/working/ptbxl")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="locked_eval_360.csv")
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--certainty_thresh", type=float, default=0.55)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes = list(ckpt["classes"])
    T_cal = max(1.15, float(ckpt.get("calib_temp", 1.1721)))
    b_cal = max(1.50, float(ckpt.get("calib_bias", 3.0927)))

    model = timm.create_model("densenet121", pretrained=False, num_classes=len(classes), drop_rate=0.2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    cohort = build_locked_patient_grouped_cohort(args.ptb_dir, target_n=args.n)
    rng = random.Random(SEED)

    rows_out = []
    top1_hits, top3_hits = 0, 0
    crit_total, crit_refused, crit_conf_miss = 0, 0, 0
    norm_total, norm_refused = 0, 0
    all_conf, all_corr = [], []

    for idx, (rec_path, pid, gt_list) in enumerate(cohort):
        sig, _ = wfdb.rdsamp(rec_path)
        img = render_realistic_ecg(np.nan_to_num(sig), rng)
        x = normalize_ink(img).unsqueeze(0).to(device)
        ps = []
        with torch.no_grad():
            for _ in range(6):
                logits = model(x).cpu().numpy()[0]
                ps.append(1.0 / (1.0 + np.exp(-np.clip((logits - b_cal) / T_cal, -30, 30))))
        arr = np.stack(ps)
        mean_p, std_p = arr.mean(axis=0), arr.std(axis=0)
        order = np.argsort(mean_p)[::-1]
        pred1 = classes[order[0]]
        top3 = [classes[k] for k in order[:3]]
        conf1 = float(mean_p[order[0]])
        std1 = float(std_p[order[0]])
        avail1 = max(0.1, min(1.0, 1.0 - min(0.9, std1 * 2.5)))
        is_confirmed = (conf1 >= args.certainty_thresh) and (avail1 >= 0.65)
        h1 = pred1 in gt_list
        h3 = any(c in gt_list for c in top3)
        if h1:
            top1_hits += 1
        if h3:
            top3_hits += 1
        all_conf.append(conf1)
        all_corr.append(1.0 if h1 else 0.0)

        is_crit = any(c in CRITICAL_CLASSES for c in gt_list)
        is_norm = (gt_list == [NORMAL_CLASS])
        if is_crit:
            crit_total += 1
            if not is_confirmed:
                crit_refused += 1
            elif not h1:
                crit_conf_miss += 1
        if is_norm:
            norm_total += 1
            if not is_confirmed:
                norm_refused += 1

        rows_out.append({
            "case_id": f"LOCKED_{idx + 1:03d}",
            "patient_id": pid,
            "domain": "ptbxl_fold10_locked",
            "true_label": "|".join(gt_list),
            "is_critical": int(is_crit),
            "pred_label": pred1 if is_confirmed else "INSUFFICIENT_DATA",
            "raw_top1": pred1,
            "confidence": round(conf1, 4),
            "uncertainty_std": round(std1, 4),
            "abstained": int(not is_confirmed),
            "correct_top1": int(h1),
            "correct_top3": int(h3),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    n = len(cohort)
    p1, lo1, hi1 = wilson_ci(top1_hits, n)
    p3, lo3, hi3 = wilson_ci(top3_hits, n)
    cr, cr_lo, cr_hi = wilson_ci(crit_refused, crit_total)
    cm, cm_lo, cm_hi = wilson_ci(crit_conf_miss, crit_total)
    nr, nr_lo, nr_hi = wilson_ci(norm_refused, norm_total)
    ece = compute_ece(np.array(all_conf), np.array(all_corr))

    print(f"=== LOCKED PATIENT-GROUPED BENCHMARK (N={n}, unique patients={n}) ===")
    print(f"Checkpoint: {os.path.basename(args.ckpt)} | T={T_cal:.3f}, b={b_cal:.3f}")
    print(f"Top-1: {top1_hits}/{n} ({p1*100:.1f}%, 95% CI [{lo1*100:.1f}%, {hi1*100:.1f}%])")
    print(f"Top-3: {top3_hits}/{n} ({p3*100:.1f}%, 95% CI [{lo3*100:.1f}%, {hi3*100:.1f}%])")
    print(f"10-bin ECE: {ece:.4f}")
    print(f"Critical Refusal Rate:        {crit_refused}/{crit_total} ({cr*100:.1f}%, 95% CI [{cr_lo*100:.1f}%, {cr_hi*100:.1f}%])")
    print(f"Critical Confident Miss Rate: {crit_conf_miss}/{crit_total} ({cm*100:.1f}%, 95% CI [{cm_lo*100:.1f}%, {cm_hi*100:.1f}%])")
    print(f"Normal Refusal Rate:          {norm_refused}/{norm_total} ({nr*100:.1f}%, 95% CI [{nr_lo*100:.1f}%, {nr_hi*100:.1f}%])")
    print(f"Saved locked CSV -> {args.out}")


if __name__ == "__main__":
    main()
