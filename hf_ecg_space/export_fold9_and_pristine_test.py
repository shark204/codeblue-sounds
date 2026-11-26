"""
export_fold9_and_pristine_test.py — Exports calibrated multi-label probability CSVs compatible with freeze_thresholds.py:
  1) fold9_probs.csv          : unique Fold-9 patients (VAL set for threshold selection)
  2) pristine_test_probs.csv  : ALL remaining Fold-10 unique patients STRICTLY EXCLUDING the 360 DEV patients!
Prints the exact SHA-256 hash of the checkpoint for freeze_thresholds.py --model-hash.
"""
import argparse
import ast
import csv
import hashlib
import os
import random
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import timm
import torch
import wfdb

from run_dev360_experiments import SCP_MAP, SEED, infer_probs, load_cohort, render_ecg


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pristine_fold10(ptb_dir: str, exclude_pids: Set[int], max_n: int = 1200) -> List[Dict]:
    df = pd.read_csv(os.path.join(ptb_dir, "ptbxl_database.csv"))
    f10 = df[df["strat_fold"] == 10].copy()
    rng = random.Random(SEED + 999)
    rows = list(f10.iterrows())
    rng.shuffle(rows)
    seen = set(exclude_pids)
    out = []
    for _, row in rows:
        pid = int(row["patient_id"])
        if pid in seen:
            continue
        scp = {k: float(v) for k, v in ast.literal_eval(row["scp_codes"]).items() if float(v) >= 50.0}
        matched = sorted([cls for cls, sset in SCP_MAP.items() if set(scp.keys()) & sset])
        if not matched:
            continue
        seen.add(pid)
        out.append({
            "rec_path": os.path.join(ptb_dir, str(row["filename_lr"])),
            "patient_id": pid,
            "gt": matched,
        })
        if len(out) >= max_n:
            break
    return out


def export_csv(cohort: List[Dict], out_csv: str, model, classes: List[str], device, T_cal: float, b_cal: float, seed: int):
    rng = random.Random(seed)
    fields = ["filename", "truth"] + [f"p_{c}" for c in classes]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in cohort:
            sig, _ = wfdb.rdsamp(rec["rec_path"])
            img = render_ecg(np.nan_to_num(sig), rng, h=384, w=512)
            probs, _ = infer_probs(model, img, device, T_cal, b_cal, 384, 512, False)
            row = {"filename": f"pid_{rec['patient_id']}", "truth": "|".join(rec["gt"])}
            for idx, c in enumerate(classes):
                row[f"p_{c}"] = f"{float(probs[idx]):.6f}"
            w.writerow(row)
    print(f"Exported {len(cohort)} unique patients -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptb_dir", default="/kaggle/working/ptbxl")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="/kaggle/working")
    ap.add_argument("--n_fold9", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes = list(ckpt["classes"])
    T_cal = max(1.15, float(ckpt.get("calib_temp", 1.1721)))
    b_cal = max(1.50, float(ckpt.get("calib_bias", 3.0927)))
    mhash = sha256_file(args.ckpt)[:16]
    print(f"Model: {os.path.basename(args.ckpt)} | sha256[:16]={mhash} | T={T_cal:.3f}, b={b_cal:.3f}")

    model = timm.create_model("densenet121", pretrained=False, num_classes=len(classes), drop_rate=0.2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1) Identify exact 360 DEV patient_ids so they are STRICTLY excluded from pristine test!
    dev_360 = load_cohort(args.ptb_dir, fold=10, target_n=360, seed=SEED)
    dev_pids = {r["patient_id"] for r in dev_360}

    # 2) Fold 9 validation set for threshold selection
    f9 = load_cohort(args.ptb_dir, fold=9, target_n=args.n_fold9, seed=SEED + 9)
    export_csv(f9, os.path.join(args.out_dir, "fold9_probs.csv"), model, classes, device, T_cal, b_cal, SEED + 9)

    # 3) Pristine untouched Fold-10 test set (ZERO overlap with 360 DEV!)
    pristine = load_pristine_fold10(args.ptb_dir, exclude_pids=dev_pids, max_n=args.n_test)
    assert len({r["patient_id"] for r in pristine} & dev_pids) == 0, "LEAKAGE DETECTED!"
    export_csv(pristine, os.path.join(args.out_dir, "pristine_test_probs.csv"), model, classes, device, T_cal, b_cal, SEED + 999)
    print(f"VERIFIED ZERO OVERLAP between 360 DEV and {len(pristine)} pristine Fold-10 test patients.")


if __name__ == "__main__":
    main()
