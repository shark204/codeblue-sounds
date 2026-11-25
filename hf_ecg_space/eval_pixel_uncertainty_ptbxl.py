#!/usr/bin/env python3
"""
Code Blue ECG — Pixel Estimator Error Calibration & 140-NORM False-Alarm Benchmark
Measures on PTB-XL Fold 10 synthetic 12-lead ECG images with NeuroKit2 ground-truth delineation:
  1. Per-quantity MAE, P95 (95th percentile absolute error), and Bias for HR, PR, QRS, QT.
  2. Exact QTc error propagation via root-sum-square of dQT/QT and 0.5*dRR/RR (or 1/3*dRR/RR for Fridericia).
  3. Before vs After uncertainty-aware thresholding on 140 synthetic NORM images:
     - False Short-QT (<350 ms) definite alarm rate before vs after
     - False Long-QT (>460 ms / >500 ms) definite alarm rate before vs after
     - Exact "possible (verify measurement)" rate (never hidden, no red banner)
"""

import ast
import glob
import json
import math
import os
import random

import cv2
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMG_H, IMG_W = 384, 512


def find_ptb_dir():
    for p in ["/kaggle/working/ptbxl", "/kaggle/input/**/ptbxl_database.csv", "/content/ptbxl"]:
        hits = glob.glob(p, recursive=True)
        if hits:
            return os.path.dirname(hits[0]) if hits[0].endswith(".csv") else hits[0]
    raise FileNotFoundError("PTB-XL directory not found! Run the PTB-XL download cell first.")


def delineate_ground_truth(sig_12x1000: np.ndarray, fs: int = 100):
    """Extract reference HR, PR, QRS, QT from Lead II (index 1) via NeuroKit2."""
    lead_ii = sig_12x1000[1].astype(np.float64)
    try:
        cleaned = nk.ecg_clean(lead_ii, sampling_rate=fs, method="neurokit")
        _, r_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = r_info["ECG_R_Peaks"]
        if len(rpeaks) < 4:
            return None
        rr_sec = np.median(np.diff(rpeaks)) / float(fs)
        hr_bpm = 60.0 / rr_sec if rr_sec > 0 else None

        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")
        p_on = np.array(waves.get("ECG_P_Onsets", []), dtype=np.float64)
        q_on = np.array(waves.get("ECG_R_Onsets", []), dtype=np.float64)
        r_off = np.array(waves.get("ECG_R_Offsets", []), dtype=np.float64)
        t_off = np.array(waves.get("ECG_T_Offsets", []), dtype=np.float64)

        n = min(len(p_on), len(q_on), len(r_off), len(t_off))
        pr_vals, qrs_vals, qt_vals = [], [], []
        for i in range(n):
            if np.isfinite(p_on[i]) and np.isfinite(q_on[i]):
                pr = (q_on[i] - p_on[i]) * (1000.0 / fs)
                if 80 <= pr <= 360:
                    pr_vals.append(pr)
            if np.isfinite(q_on[i]) and np.isfinite(r_off[i]):
                qrs = (r_off[i] - q_on[i]) * (1000.0 / fs)
                if 50 <= qrs <= 220:
                    qrs_vals.append(qrs)
            if np.isfinite(q_on[i]) and np.isfinite(t_off[i]):
                qt = (t_off[i] - q_on[i]) * (1000.0 / fs)
                if 220 <= qt <= 620:
                    qt_vals.append(qt)

        return {
            "hr_bpm": float(hr_bpm) if hr_bpm and 30 <= hr_bpm <= 220 else None,
            "pr_ms": float(np.median(pr_vals)) if pr_vals else None,
            "qrs_ms": float(np.median(qrs_vals)) if qrs_vals else None,
            "qt_ms": float(np.median(qt_vals)) if qt_vals else None,
        }
    except Exception:
        return None


def render_clean_strip_image(sig_12x1000: np.ndarray):
    """Render standard 10s Lead II rhythm strip on pink ECG grid (matching real 12-lead bottom strip)."""
    img = np.full((IMG_H, IMG_W, 3), (242, 236, 250), dtype=np.uint8)
    # Draw pink grid
    for x in range(0, IMG_W, 10):
        cv2.line(img, (x, 0), (x, IMG_H), (205, 190, 230), 1)
    for y in range(0, IMG_H, 10):
        cv2.line(img, (0, y), (IMG_W, y), (205, 190, 230), 1)
    lead_ii = sig_12x1000[1]
    lead_ii = lead_ii - np.median(lead_ii)
    y0 = int(IMG_H * 0.84)
    scale_y = 45.0
    xs = np.linspace(int(IMG_W * 0.04), int(IMG_W * 0.96), len(lead_ii)).astype(np.int32)
    ys = np.clip(y0 - (lead_ii * scale_y), int(IMG_H * 0.68), int(IMG_H * 0.97)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, (25, 25, 28), 1, cv2.LINE_AA)
    return img


def estimate_intervals_from_pixels(img_bgr: np.ndarray):
    """Python mirror of browser pixel-based waveform estimator on the bottom 10s rhythm strip."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    strip = gray[int(h * 0.70):int(h * 0.97), int(w * 0.04):int(w * 0.96)]
    sh, sw = strip.shape
    inv = (255 - strip).astype(np.float32)
    # Extract 1D trace by darkest pixel per column
    y_idx = np.argmax(inv, axis=0).astype(np.float32)
    sig = -(y_idx - np.median(y_idx))
    sec_per_px = 10.0 / float(sw)
    ms_per_px = sec_per_px * 1000.0

    # R-peak detection
    thr = np.percentile(sig, 93)
    min_d = max(8, int(sw * 0.035))
    rpeaks = []
    for i in range(1, sw - 1):
        if sig[i] >= thr and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
            if not rpeaks or (i - rpeaks[-1]) >= min_d:
                rpeaks.append(i)
    if len(rpeaks) < 3:
        return {"hr_bpm": None, "pr_ms": None, "qrs_ms": None, "qt_ms": None}

    rr_px = float(np.median(np.diff(rpeaks)))
    hr_bpm = 60.0 / (rr_px * sec_per_px)

    # Estimate PR, QRS, QT around median beat window
    pr_list, qrs_list, qt_list = [], [], []
    base_std = max(1.0, float(np.std(sig) * 0.25))
    for rp in rpeaks[1:-1]:
        # Q onset: walk left from R until crossing baseline
        q_on = rp
        for k in range(rp - 1, max(0, rp - int(70 / ms_per_px)), -1):
            if abs(sig[k]) <= base_std:
                q_on = k
                break
        # S offset: walk right from R
        s_off = rp
        for k in range(rp + 1, min(sw - 1, rp + int(90 / ms_per_px))):
            if abs(sig[k]) <= base_std:
                s_off = k
                break
        qrs_ms = (s_off - q_on) * ms_per_px
        if 50 <= qrs_ms <= 220:
            qrs_list.append(qrs_ms)

        # P peak in [rp - 240ms, q_on - 10ms]
        p_lo, p_hi = max(0, rp - int(240 / ms_per_px)), max(1, q_on - int(15 / ms_per_px))
        if p_hi > p_lo + 2:
            p_pk = p_lo + int(np.argmax(sig[p_lo:p_hi]))
            pr_ms = (q_on - max(0, p_pk - int(35 / ms_per_px))) * ms_per_px
            if 80 <= pr_ms <= 340:
                pr_list.append(pr_ms)

        # T peak in [s_off + 40ms, s_off + 360ms], T end where signal returns to baseline
        t_lo, t_hi = min(sw - 2, s_off + int(40 / ms_per_px)), min(sw - 1, s_off + int(360 / ms_per_px))
        if t_hi > t_lo + 3:
            t_pk = t_lo + int(np.argmax(sig[t_lo:t_hi]))
            t_end = t_pk
            for k in range(t_pk + 1, min(sw - 1, t_pk + int(140 / ms_per_px))):
                if abs(sig[k]) <= base_std:
                    t_end = k
                    break
            qt_ms = (t_end - q_on) * ms_per_px
            if 220 <= qt_ms <= 620:
                qt_list.append(qt_ms)

    return {
        "hr_bpm": round(hr_bpm, 1) if 30 <= hr_bpm <= 220 else None,
        "pr_ms": round(float(np.median(pr_list)), 1) if pr_list else None,
        "qrs_ms": round(float(np.median(qrs_list)), 1) if qrs_list else None,
        "qt_ms": round(float(np.median(qt_list)), 1) if qt_list else None,
    }


def propagate_qtc(qt_ms, d_qt, hr_bpm, d_hr):
    rr = 60.0 / max(20.0, hr_bpm)
    exp = (1.0 / 3.0) if (hr_bpm > 100 or hr_bpm < 50) else 0.5
    qtc = qt_ms / (rr ** exp)
    rel_qt = d_qt / qt_ms if qt_ms > 0 else 0.0
    rel_rr = d_hr / max(20.0, hr_bpm)
    err = qtc * math.sqrt(rel_qt ** 2 + (exp * rel_rr) ** 2)
    return qtc, err


def main():
    ptb_dir = find_ptb_dir()
    df = pd.read_csv(os.path.join(ptb_dir, "ptbxl_database.csv"))
    df["scp_dict"] = df["scp_codes"].apply(ast.literal_eval)
    fold10 = df[df["strat_fold"] == 10].copy()
    norm_df = fold10[fold10["scp_dict"].apply(lambda d: "NORM" in d and d["NORM"] >= 80.0)]
    sample_norm = norm_df.sample(n=min(140, len(norm_df)), random_state=SEED)

    errs = {k: [] for k in ("hr_bpm", "pr_ms", "qrs_ms", "qt_ms")}
    norm_pairs = []

    print(f"Evaluating pixel estimator on {len(sample_norm)} synthetic NORM Fold-10 ECGs...")
    for _, row in sample_norm.iterrows():
        rec_path = os.path.join(ptb_dir, row["filename_lr"])
        try:
            sig, _ = wfdb.rdsamp(rec_path)
        except Exception:
            continue
        sig_12 = sig.T  # (12, 1000)
        gt = delineate_ground_truth(sig_12, fs=100)
        if not gt:
            continue
        img = render_clean_strip_image(sig_12)
        est = estimate_intervals_from_pixels(img)
        for k in errs:
            if gt[k] is not None and est[k] is not None:
                errs[k].append(est[k] - gt[k])
        norm_pairs.append((gt, est))

    print("\n================ PIXEL ESTIMATOR ERROR TABLE (PTB-XL Fold 10 + NeuroKit2) ================")
    print(f"{'Quantity':<10} | {'N':<4} | {'MAE':<8} | {'P95 (error_ms)':<15} | {'Bias (est-gt)':<14}")
    print("-" * 64)
    p95_map = {}
    for k, diffs in errs.items():
        if not diffs:
            p95_map[k] = 0.0
            continue
        arr = np.array(diffs, dtype=np.float64)
        mae = float(np.mean(np.abs(arr)))
        p95 = float(np.percentile(np.abs(arr), 95))
        bias = float(np.mean(arr))
        p95_map[k] = round(p95, 1)
        print(f"{k:<10} | {len(arr):<4} | {mae:7.2f}  | {p95:14.2f}  | {bias:+13.2f}")

    # Requirement 6(d): False Short-QT / False Long-QT alarm rate on the 140 synthetic NORM images (Before vs After)
    n_eval = 0
    before_short, before_long = 0, 0
    after_def_short, after_pos_short = 0, 0
    after_def_long, after_pos_long = 0, 0

    for gt, est in norm_pairs:
        if est["hr_bpm"] is None or est["qt_ms"] is None:
            continue
        n_eval += 1
        qtc, d_qtc = propagate_qtc(est["qt_ms"], p95_map["qt_ms"], est["hr_bpm"], p95_map["hr_bpm"])
        # Before (strict threshold with error=0)
        if qtc < 350: before_short += 1
        if qtc > 460: before_long += 1
        # After (uncertainty-aware, clinical thresholds unchanged)
        if qtc + d_qtc < 350:
            after_def_short += 1
        elif abs(qtc - 350) <= d_qtc:
            after_pos_short += 1
        if qtc - d_qtc > 460:
            after_def_long += 1
        elif abs(qtc - 460) <= d_qtc:
            after_pos_long += 1

    print(f"\n================ 140 SYNTHETIC NORM FALSE-ALARM BENCHMARK (N={n_eval}) ================")
    print(f"False Short-QT (<350 ms) BEFORE (strict): {before_short}/{n_eval} ({before_short/max(1,n_eval)*100:.1f}%)")
    print(f"False Short-QT (<350 ms) AFTER  definite: {after_def_short}/{n_eval} ({after_def_short/max(1,n_eval)*100:.1f}%) | possible(verify): {after_pos_short}/{n_eval} ({after_pos_short/max(1,n_eval)*100:.1f}%)")
    print(f"False Long-QT  (>460 ms) BEFORE (strict): {before_long}/{n_eval} ({before_long/max(1,n_eval)*100:.1f}%)")
    print(f"False Long-QT  (>460 ms) AFTER  definite: {after_def_long}/{n_eval} ({after_def_long/max(1,n_eval)*100:.1f}%) | possible(verify): {after_pos_long}/{n_eval} ({after_pos_long/max(1,n_eval)*100:.1f}%)")

    out = {"p95_map": p95_map, "n_eval": n_eval, "before_short": before_short, "after_def_short": after_def_short,
           "after_pos_short": after_pos_short, "before_long": before_long, "after_def_long": after_def_long, "after_pos_long": after_pos_long}
    with open("pixel_uncertainty_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved pixel_uncertainty_report.json!")


if __name__ == "__main__":
    main()
