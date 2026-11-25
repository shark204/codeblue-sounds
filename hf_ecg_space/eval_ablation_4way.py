#!/usr/bin/env python3
"""
Code Blue ECG — Automated 4-Way Ablation & Calibration Benchmark (v2 — Exact Training Preprocessing)
Compares 4 system configurations on the exact same ECG images:
  V1: Visual AI Only (DenseNet121 v2 + MC Dropout)
  V2: Deterministic Rules + Estimated Header/OCR Features Only (no Visual AI)
  V3: Hybrid Fusion WITHOUT KB (Visual AI + Rules via Noisy-OR + Hierarchy + 3-Tier)
  V4: Full Fusion WITH KB (V3 + KB inside Noisy-OR — directly measures double-counting impact)

Uses the EXACT `CLASSES`, `REAL_FOLDER_TO_CLASSES`, zero-leakage 60/40 split, and
`normalize_ecg_to_clean_ink` preprocessing from `train_ecg_densenet_v2.py`.
"""

import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
IMG_H, IMG_W = 384, 512

CLASSES = [
    "NORMAL_SINUS_RHYTHM",
    "ANTERIOR_SEPTAL_MI",
    "INFERIOR_POSTERIOR_MI",
    "LATERAL_MI",
    "NSTEMI_ISCHEMIA",
    "ATRIAL_FIBRILLATION",
    "ATRIAL_FLUTTER",
    "SVT_TACHYCARDIA",
    "SINUS_BRADYCARDIA",
    "FIRST_DEGREE_AV_BLOCK",
    "HIGH_GRADE_AV_BLOCK",
    "RBBB",
    "LBBB",
    "LAFB_LPFB",
    "LVH_RVH",
    "PACEMAKER_RHYTHM",
    "WPW_PREEXCITATION",
    "LONG_QT_OR_ECTOPY",
]

REAL_FOLDER_TO_CLASSES = {
    "01_normal_sinus_rhythm": {"NORMAL_SINUS_RHYTHM"},
    "02_stemi_acute_mi": {"ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI"},
    "03_wellens_and_ischemia": {"NSTEMI_ISCHEMIA", "ANTERIOR_SEPTAL_MI"},
    "04_atrial_fibrillation": {"ATRIAL_FIBRILLATION"},
    "05_atrial_flutter": {"ATRIAL_FLUTTER"},
    "06_svt_and_mat": {"SVT_TACHYCARDIA"},
    "08_av_block_1st_2nd": {"FIRST_DEGREE_AV_BLOCK", "HIGH_GRADE_AV_BLOCK", "SINUS_BRADYCARDIA"},
    "09_av_block_3rd_complete": {"HIGH_GRADE_AV_BLOCK", "SINUS_BRADYCARDIA"},
    "10_rbbb_and_fascicular": {"RBBB", "LAFB_LPFB"},
    "11_lbbb": {"LBBB"},
    "12_wpw_preexcitation": {"WPW_PREEXCITATION"},
    "13_long_short_qt": {"LONG_QT_OR_ECTOPY"},
    "15_lvh_rvh_enlargement": {"LVH_RVH"},
    "20_pacemaker_toxicity_reversal": {"PACEMAKER_RHYTHM"},
}

CRITICAL_CLASSES = {
    "ANTERIOR_SEPTAL_MI",
    "INFERIOR_POSTERIOR_MI",
    "LATERAL_MI",
    "STEMI",
    "HIGH_GRADE_AV_BLOCK",
    "COMPLETE_HEART_BLOCK",
    "VENTRICULAR_TACHYCARDIA",
}

CANONICAL_PARENT = {
    "ANTERIOR_SEPTAL_MI": "STEMI",
    "INFERIOR_POSTERIOR_MI": "STEMI",
    "LATERAL_MI": "STEMI",
    "STEMI": "STEMI",
    "NSTEMI_ISCHEMIA": "ISCHEMIA_ACS",
    "ATRIAL_FIBRILLATION": "ATRIAL_FIBRILLATION",
    "ATRIAL_FLUTTER": "ATRIAL_FLUTTER",
    "SVT_TACHYCARDIA": "SVT",
    "SINUS_BRADYCARDIA": "BRADYCARDIA",
    "FIRST_DEGREE_AV_BLOCK": "FIRST_DEGREE_AVB",
    "HIGH_GRADE_AV_BLOCK": "HIGH_GRADE_AVB",
    "RBBB": "RBBB",
    "LBBB": "LBBB",
    "LAFB_LPFB": "LAFB_LPFB",
    "LVH_RVH": "LVH",
    "PACEMAKER_RHYTHM": "PACEMAKER",
    "WPW_PREEXCITATION": "WPW",
    "LONG_QT_OR_ECTOPY": "LONG_QT",
    "NORMAL_SINUS_RHYTHM": "NORMAL_SINUS_RHYTHM",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STRONG_EFF_THRESHOLD = 0.40
CONSENSUS_BOOST = 0.15
AI_CRITICAL_TIER1_THRESHOLD = 0.65
CALIB_TEMP = 1.1721
CALIB_BIAS = 3.0927


def find_first(patterns):
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None


def normalize_ecg_to_clean_ink(img_bgr: np.ndarray, H: int = IMG_H, W: int = IMG_W) -> torch.Tensor:
    """EXACT grid-removal preprocessing from train_ecg_densenet_v2.py."""
    if img_bgr.shape[0] != H or img_bgr.shape[1] != W:
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    pink_score = np.clip((r - np.minimum(g, b)) - 12.0, 0.0, 255.0)
    luma = 0.55 * r + 0.35 * g + 0.10 * b
    luma_no_grid = np.clip(luma + pink_score * 0.95, 0.0, 255.0)
    bg = cv2.GaussianBlur(luma_no_grid, (31, 31), 0)
    ink = np.clip(bg - luma_no_grid, 0.0, 255.0)
    hi = max(25.0, float(np.percentile(ink, 99.5)))
    clean_gray = np.clip(255.0 - (ink / hi) * 255.0, 0.0, 255.0).astype(np.uint8)
    clean_rgb = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2RGB)
    x = (clean_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def load_densenet_v2(weights_path: str):
    model = timm.create_model("densenet121", pretrained=False, num_classes=len(CLASSES), drop_rate=0.2)
    ckpt = torch.load(weights_path, map_location=DEVICE)
    sd = (ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    return model


def enable_mc_dropout(model: nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict_densenet_mc(model, img_bgr: np.ndarray, passes: int = 4):
    x = normalize_ecg_to_clean_ink(img_bgr).unsqueeze(0).to(DEVICE)
    enable_mc_dropout(model)
    probs_list = [torch.sigmoid((model(x) - CALIB_BIAS) / CALIB_TEMP).cpu().numpy()[0] for _ in range(passes)]
    arr = np.stack(probs_list, axis=0)
    mean_p, std_p = arr.mean(axis=0), arr.std(axis=0)
    preds = []
    for i, cls in enumerate(CLASSES):
        p = float(mean_p[i])
        u = float(min(0.9, max(0.0, std_p[i] * 2.5)))
        if p >= 0.30:
            preds.append({"code": cls, "prob": p, "uncertainty": u})
    preds.sort(key=lambda d: d["prob"], reverse=True)
    return preds


OCR_ENGINE = None


def init_ocr():
    global OCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
        OCR_ENGINE = RapidOCR()
        print("RapidOCR initialized.")
    except Exception as e:
        print(f"RapidOCR unavailable ({e}); using waveform HR estimation only.")
        OCR_ENGINE = None


def extract_header_ocr(pil_img: Image.Image):
    res = {"hr_bpm": None, "pr_ms": None, "qrs_ms": None, "qt_ms": None, "axis_deg": None, "ocr_used": False}
    if OCR_ENGINE is None:
        return res
    w, h = pil_img.size
    crop = np.array(pil_img.crop((0, 0, w, int(h * 0.25))).convert("RGB"))
    ocr_out, _ = OCR_ENGINE(crop)
    if not ocr_out:
        return res
    text = " ".join([item[1] for item in ocr_out if len(item) >= 2])
    m_hr = re.search(r"(?:Vent\.?\s*rate|HR|Rate)\s*[:=]?\s*(\d{2,3})", text, re.I)
    m_pr = re.search(r"(?:PR\s*(?:int|interval)?)\s*[:=]?\s*(\d{2,3})", text, re.I)
    m_qrs = re.search(r"(?:QRS\s*(?:dur|duration)?)\s*[:=]?\s*(\d{2,3})", text, re.I)
    m_qt = re.search(r"(?:QT/QTc|QT)\s*[:=]?\s*(\d{3})", text, re.I)
    m_ax = re.search(r"(?:Axis|Axes)\s*[:=]?\s*[*\s]*(-?\d{1,3})", text, re.I)
    if m_hr: res["hr_bpm"] = int(m_hr.group(1))
    if m_pr: res["pr_ms"] = int(m_pr.group(1))
    if m_qrs: res["qrs_ms"] = int(m_qrs.group(1))
    if m_qt: res["qt_ms"] = int(m_qt.group(1))
    if m_ax: res["axis_deg"] = int(m_ax.group(1))
    res["ocr_used"] = any(res[k] is not None for k in ("hr_bpm", "pr_ms", "qrs_ms", "qt_ms", "axis_deg"))
    return res


def estimate_hr_from_waveform(pil_img: Image.Image):
    arr = np.array(pil_img.convert("L"))
    h, w = arr.shape
    strip = arr[int(h * 0.72):int(h * 0.96), int(w * 0.04):int(w * 0.96)]
    if strip.size == 0:
        return None
    inv = 255 - strip
    col_profile = inv.max(axis=0).astype(np.float32)
    thresh = np.percentile(col_profile, 92)
    peaks = []
    min_dist = max(8, int(strip.shape[1] * 0.035))
    for i in range(1, len(col_profile) - 1):
        if col_profile[i] >= thresh and col_profile[i] >= col_profile[i - 1] and col_profile[i] >= col_profile[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
    if len(peaks) < 3:
        return None
    rr_px = np.median(np.diff(peaks))
    if rr_px <= 0:
        return None
    hr = 60.0 / (rr_px * (10.0 / float(strip.shape[1])))
    return int(round(hr)) if 28 <= hr <= 240 else None


def run_deterministic_rules(meas: dict):
    findings = []
    hr, pr, qrs, qt, axis = meas.get("hr_bpm"), meas.get("pr_ms"), meas.get("qrs_ms"), meas.get("qt_ms"), meas.get("axis_deg")
    if hr is not None:
        if hr < 60:
            findings.append({"code": "SINUS_BRADYCARDIA", "strength": 0.85, "avail": 1.0, "urgency": "ABNORMAL"})
        elif hr > 150:
            findings.append({"code": "SVT_TACHYCARDIA", "strength": 0.70, "avail": 1.0, "urgency": "URGENT"})
    if pr is not None and pr > 200:
        findings.append({"code": "FIRST_DEGREE_AV_BLOCK", "strength": 0.85, "avail": 1.0, "urgency": "ABNORMAL"})
    if qrs is not None and qrs >= 120:
        findings.append({"code": "RBBB", "strength": 0.55, "avail": 1.0, "urgency": "ABNORMAL"})
        findings.append({"code": "LBBB", "strength": 0.55, "avail": 1.0, "urgency": "ABNORMAL"})
    if axis is not None and axis <= -45:
        findings.append({"code": "LAFB_LPFB", "strength": 0.75, "avail": 1.0, "urgency": "ABNORMAL"})
    if hr is not None and qt is not None:
        rr = 60.0 / max(20.0, float(hr))
        qt_eff = float(qt) - (0.485 * float(qrs) if (qrs is not None and qrs >= 120) else 0.0)
        qtc = qt_eff / (rr ** (1.0 / 3.0)) if (hr > 100 or hr < 50) else qt_eff / math.sqrt(rr)
        if qtc >= 470:
            findings.append({
                "code": "LONG_QT_OR_ECTOPY",
                "strength": 0.85 if qtc >= 500 else 0.65,
                "avail": 1.0,
                "urgency": "CRITICAL" if qtc >= 500 else "URGENT",
            })
    if not findings and hr is not None and 60 <= hr <= 100:
        findings.append({"code": "NORMAL_SINUS_RHYTHM", "strength": 0.55, "avail": 0.8, "urgency": "NORMAL"})
    return findings


def fuse_noisy_or(ai_preds, rule_preds, include_kb=False):
    groups = defaultdict(list)
    for a in ai_preds:
        p = CANONICAL_PARENT.get(a["code"], a["code"])
        groups[p].append(("ai", a["code"], a["prob"], max(0.1, 1.0 - a["uncertainty"]),
                          "CRITICAL" if a["code"] in CRITICAL_CLASSES else "URGENT"))
    for r in rule_preds:
        p = CANONICAL_PARENT.get(r["code"], r["code"])
        groups[p].append(("rules", r["code"], r["strength"], r["avail"], r["urgency"]))
    if include_kb:
        for r in rule_preds:
            p = CANONICAL_PARENT.get(r["code"], r["code"])
            groups[p].append(("kb", r["code"], min(0.80, r["strength"] * 0.85), 0.85, r["urgency"]))

    ranked = []
    for parent, items in groups.items():
        best = {}
        for src, code, s, a, urg in items:
            eff = max(0.0, min(1.0, s * a))
            if src not in best or eff > best[src][1]:
                best[src] = (code, eff, urg)
        comp, n_strong, is_crit, best_code, best_eff = 1.0, 0, False, parent, -1.0
        for src, (code, eff, urg) in best.items():
            comp *= (1.0 - eff)
            if eff >= STRONG_EFF_THRESHOLD:
                n_strong += 1
            if urg == "CRITICAL":
                is_crit = True
            if eff > best_eff:
                best_eff, best_code = eff, code
        conf = max(0.0, min(1.0, 1.0 - comp))
        rank = min(1.0, conf * (1.0 + (CONSENSUS_BOOST if n_strong >= 2 else 0.0)))
        has_non_ai = any(src in ("rules", "context") for src in best.keys())
        tier = 1 if (is_crit and (has_non_ai or conf >= AI_CRITICAL_TIER1_THRESHOLD)) else 2
        ranked.append({"code": best_code, "parent": parent, "conf": conf, "rank": rank, "tier": tier})

    ranked.sort(key=lambda d: (d["tier"], -d["rank"]))
    return ranked


def bootstrap_ci(binary_vec, n_boot=1000):
    if not binary_vec:
        return (0.0, 0.0, 0.0)
    arr = np.array(binary_vec, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(arr.mean()), float(lo), float(hi))


def compute_ece(confs, corrects, n_bins=5):
    bins, ece, n = [], 0.0, len(confs)
    if n == 0:
        return 0.0, []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idxs = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not idxs:
            bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0, "avg_conf": None, "emp_acc": None})
            continue
        avg_c = float(np.mean([confs[i] for i in idxs]))
        emp_a = float(np.mean([corrects[i] for i in idxs]))
        ece += (len(idxs) / n) * abs(avg_c - emp_a)
        bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": len(idxs), "avg_conf": round(avg_c, 4), "emp_acc": round(emp_a, 4)})
    return round(ece, 4), bins


def evaluate_version(records, version_key):
    top1_vec, top3_vec, confs, corrects = [], [], [], []
    normal_total, alarm_fatigue_cnt, crit_total, missed_crit_cnt = 0, 0, 0, 0

    for r in records:
        gt_set = r["gt_classes"]
        gt_parents = {CANONICAL_PARENT.get(g, g) for g in gt_set}
        preds = r[version_key]

        is_t1 = int(len(preds) > 0 and (preds[0]["code"] in gt_set or preds[0]["parent"] in gt_parents))
        is_t3 = int(any(p["code"] in gt_set or p["parent"] in gt_parents for p in preds[:3]))
        top1_vec.append(is_t1)
        top3_vec.append(is_t3)

        if len(preds) > 0:
            confs.append(preds[0]["conf"])
            corrects.append(is_t1)

        if "NORMAL_SINUS_RHYTHM" in gt_set:
            normal_total += 1
            if any(p["tier"] == 1 for p in preds[:2]):
                alarm_fatigue_cnt += 1
        if any(g in CRITICAL_CLASSES for g in gt_set):
            crit_total += 1
            if not any(p["code"] in gt_set or p["parent"] in gt_parents for p in preds[:2]):
                missed_crit_cnt += 1

    t1, t1_lo, t1_hi = bootstrap_ci(top1_vec)
    t3, t3_lo, t3_hi = bootstrap_ci(top3_vec)
    ece, rel_table = compute_ece(confs, corrects, n_bins=5)

    return {
        "n": len(records),
        "top1": round(t1 * 100, 1),
        "top1_ci95": [round(t1_lo * 100, 1), round(t1_hi * 100, 1)],
        "top3": round(t3 * 100, 1),
        "top3_ci95": [round(t3_lo * 100, 1), round(t3_hi * 100, 1)],
        "alarm_fatigue_pct": round((alarm_fatigue_cnt / max(1, normal_total)) * 100, 1),
        "missed_critical_pct": round((missed_crit_cnt / max(1, crit_total)) * 100, 1),
        "ece": ece,
        "reliability_table": rel_table,
    }


def main():
    weights_path = find_first([
        "/kaggle/input/**/ecg_densenet121_v2.pt",
        "/kaggle/working/**/ecg_densenet121_v2.pt",
        "/content/drive/MyDrive/**/ecg_densenet121_v2.pt",
        "hf_ecg_space/weights/ecg_densenet121_v2.pt",
    ])
    assert weights_path, "Could not locate ecg_densenet121_v2.pt!"

    tb_dir = find_first([
        "/kaggle/working/ecg_100_test",
        "/kaggle/input/**/01_normal_sinus_rhythm",
        "/content/drive/MyDrive/**/01_normal_sinus_rhythm",
    ])
    if tb_dir and tb_dir.endswith("01_normal_sinus_rhythm"):
        tb_dir = str(Path(tb_dir).parent)
    assert tb_dir and os.path.isdir(tb_dir), "Could not locate textbook images directory!"

    print(f"Device: {DEVICE} | Weights: {weights_path} | Textbook dir: {tb_dir}")
    init_ocr()
    model = load_densenet_v2(weights_path)

    # Replicate EXACT zero-leakage 60/40 split from train_ecg_densenet_v2.py
    samples = []
    for fld, cls_set in sorted(REAL_FOLDER_TO_CLASSES.items()):
        hits = sorted([
            p for p in glob.glob(f"{tb_dir}/**/{fld}/*.*", recursive=True)
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
        if not hits:
            continue
        split_k = max(1, int(len(hits) * 0.6))
        for i, p in enumerate(hits):
            samples.append({"path": p, "folder": fld, "gt_classes": cls_set, "is_heldout": i >= split_k})

    print(f"Evaluating {len(samples)} textbook ECGs ({sum(s['is_heldout'] for s in samples)} strictly held-out)...")
    records, ocr_hits, hr_est_hits = [], 0, 0

    for s in samples:
        img_bgr = cv2.imread(s["path"])
        if img_bgr is None:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ai_preds = predict_densenet_mc(model, img_bgr, passes=4)
        meas = extract_header_ocr(pil_img)
        if meas["ocr_used"]:
            ocr_hits += 1
        if meas["hr_bpm"] is None:
            meas["hr_bpm"] = estimate_hr_from_waveform(pil_img)
        if meas["hr_bpm"] is not None:
            hr_est_hits += 1
        rule_preds = run_deterministic_rules(meas)

        v1 = [{"code": a["code"], "parent": CANONICAL_PARENT.get(a["code"], a["code"]),
               "conf": a["prob"], "rank": a["prob"] * max(0.1, 1.0 - a["uncertainty"]),
               "tier": 1 if (a["code"] in CRITICAL_CLASSES and a["prob"] * max(0.1, 1.0 - a["uncertainty"]) >= AI_CRITICAL_TIER1_THRESHOLD) else 2} for a in ai_preds]
        v1.sort(key=lambda d: (d["tier"], -d["rank"]))

        records.append({
            "path": os.path.basename(s["path"]),
            "folder": s["folder"],
            "gt_classes": s["gt_classes"],
            "is_heldout": s["is_heldout"],
            "v1_ai_only": v1,
            "v2_rules_only": fuse_noisy_or([], rule_preds, include_kb=False),
            "v3_fusion_no_kb": fuse_noisy_or(ai_preds, rule_preds, include_kb=False),
            "v4_fusion_with_kb": fuse_noisy_or(ai_preds, rule_preds, include_kb=True),
        })

    report = {
        "ocr_capture_rate_pct": round((ocr_hits / max(1, len(records))) * 100, 1),
        "hr_populated_rate_pct": round((hr_est_hits / max(1, len(records))) * 100, 1),
        "heldout_only": {},
        "all_textbook": {},
    }

    heldout_recs = [r for r in records if r["is_heldout"]]
    versions = ["v1_ai_only", "v2_rules_only", "v3_fusion_no_kb", "v4_fusion_with_kb"]

    print(f"\nOCR capture rate: {report['ocr_capture_rate_pct']}% | HR populated rate: {report['hr_populated_rate_pct']}%")
    print(f"\n================ 4-WAY ABLATION REPORT (STRICTLY HELD-OUT, N={len(heldout_recs)}) ================")
    print(f"{'Version':<20} | {'Top-1 (95% CI)':<22} | {'Top-3 (95% CI)':<22} | {'AlarmFat%':<9} | {'MissCrit%':<9} | {'ECE':<6}")
    print("-" * 102)
    csv_rows = []
    for vk in versions:
        m_ho = evaluate_version(heldout_recs, vk)
        m_all = evaluate_version(records, vk)
        report["heldout_only"][vk] = m_ho
        report["all_textbook"][vk] = m_all
        print(
            f"{vk:<20} | {m_ho['top1']:5.1f}% [{m_ho['top1_ci95'][0]:4.1f},{m_ho['top1_ci95'][1]:4.1f}] | "
            f"{m_ho['top3']:5.1f}% [{m_ho['top3_ci95'][0]:4.1f},{m_ho['top3_ci95'][1]:4.1f}] | "
            f"{m_ho['alarm_fatigue_pct']:8.1f}% | {m_ho['missed_critical_pct']:8.1f}% | {m_ho['ece']:6.4f}"
        )
        csv_rows.append({"split": "heldout", "version": vk, **{k: v for k, v in m_ho.items() if k != "reliability_table"}})
        csv_rows.append({"split": "all_textbook", "version": vk, **{k: v for k, v in m_all.items() if k != "reliability_table"}})

    with open("ablation_4way_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open("ablation_4way_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print("\n--- Reliability Table (v3_fusion_no_kb, Held-Out) ---")
    for b in report["heldout_only"]["v3_fusion_no_kb"]["reliability_table"]:
        print(f"  Bin {b['bin']}: n={b['count']:2d} | avg_conf={b['avg_conf']} | emp_acc={b['emp_acc']}")
    # Export harness-compatible CSVs for ecg_eval_harness.py (McNemar, Selective Accuracy, Per-Class, n_needed)
    for vk in versions:
        csv_name = f"preds_{vk}.csv"
        with open(csv_name, "w", newline="", encoding="utf-8") as cf:
            cw = csv.writer(cf)
            cw.writerow(["filename", "domain", "split", "truth", "pred1", "conf1", "pred2", "conf2", "pred3", "conf3", "abstain"])
            for r in records:
                preds = r[vk]
                ab = 1 if len(preds) == 0 else 0
                p1, c1 = (preds[0]["code"], f"{preds[0]['conf']:.4f}") if len(preds) > 0 else ("", "")
                p2, c2 = (preds[1]["code"], f"{preds[1]['conf']:.4f}") if len(preds) > 1 else ("", "")
                p3, c3 = (preds[2]["code"], f"{preds[2]['conf']:.4f}") if len(preds) > 2 else ("", "")
                cw.writerow([
                    f"{r['folder']}/{r['path']}",
                    "textbook",
                    "test" if r["is_heldout"] else "val",
                    "|".join(sorted(r["gt_classes"])),
                    p1, c1, p2, c2, p3, c3, ab,
                ])
    print("\nSaved ablation_4way_report.json, ablation_4way_summary.csv, and preds_<version>.csv for ecg_eval_harness.py!")


if __name__ == "__main__":
    main()
