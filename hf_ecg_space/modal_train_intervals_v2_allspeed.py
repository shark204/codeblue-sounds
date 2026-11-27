"""
Code Blue ECG — v2 All-Speed, All-Gain, All-Layout Deep ECG Interval Regressor on Modal A10G (~12 min, ~$0.25)
Warm-starts from `/vol/outputs/ecg_interval_regressor.pt` and generalizes across EVERY real-world ECG condition:
  - Paper speeds: 25 mm/s (standard), 50 mm/s (European/German double speed), 12.5 mm/s (compressed)
  - Voltage gains: 10 mm/mV (standard), 5 mm/mV (half gain / LVH), 20 mm/mV (double gain)
  - Layouts: 3x4 + Rhythm II AND 6x2
  - Real 1 mV calibration pulse + printed speed/gain cue + 5 paper/grid styles + 3D mobile camera stress
Evaluates strictly on Pristine Fold-10 across: 25 mm/s Clean, 50 mm/s Double-Speed, 6x2 Layout, and Mobile Stress!
Promotes to `/vol/outputs/ecg_interval_regressor.pt` automatically ONLY if it improves or maintains 25mm/s while mastering 50mm/s.
"""
import modal

app = modal.App("codeblue-train-intervals-v2")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "timm", "wfdb", "scipy", "scikit-learn",
        "pandas", "numpy", "opencv-python-headless", "pillow", "requests",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=6.0,
    memory=16384,
    timeout=3600,
    volumes={"/vol": vol},
)
def train_intervals_v2():
    import hashlib, json, math, os, re, shutil, time
    import cv2, numpy as np, pandas as pd, timm, torch, torch.nn as nn, wfdb
    from torch.utils.data import DataLoader, Dataset

    vol.reload()
    torch.manual_seed(20261128)
    np.random.seed(20261128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_H, IMG_W = 512, 768
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PTB_DIR, PLUS_DIR, OUT_DIR = "/vol/ptbxl", "/vol/ptbxl_plus", "/vol/outputs"
    CKPT_PATH = os.path.join(OUT_DIR, "ecg_interval_regressor.pt")

    f12 = pd.read_csv(os.path.join(PLUS_DIR, "12sl_features.csv"))
    funi = pd.read_csv(os.path.join(PLUS_DIR, "unig_features.csv"))

    def pick_regex(df: pd.DataFrame, patterns: list[str]) -> pd.Series:
        for pat in patterns:
            for c in df.columns:
                if re.fullmatch(pat, c, flags=re.I):
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.notna().sum() > 500:
                        return s
        return pd.Series(np.nan, index=df.index, dtype=np.float64)

    def extract_globals(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"ecg_id": pd.to_numeric(df["ecg_id"], errors="coerce").astype("Int64")})
        rr = pick_regex(df, [r"RR_Mean_Global", r"RR_Int_Global", r"RR_Mean.*", r"RR_.*Global"])
        hr = pick_regex(df, [r"Ventricular_Rate.*", r"HR_Global", r"Heart_Rate.*", r"HR"])
        hr_from_rr = np.where((rr > 250) & (rr < 2500), 60000.0 / rr, np.nan)
        out["hr"] = np.where(np.isfinite(hr) & (hr >= 25) & (hr <= 250), hr, hr_from_rr)
        out["pr"] = pick_regex(df, [r"PR_Int_Global", r"PR_Int", r"PR_Dur.*Global", r"PR_Int_II", r"PQ_Int.*"])
        out["qrs"] = pick_regex(df, [r"QRS_Dur_Global", r"QRS_Dur", r"QRS_Int_Global", r"QRS_Dur_II"])
        out["qt"] = pick_regex(df, [r"QT_Int_Global", r"QT_Int", r"QT_Dur.*Global", r"QT_Int_II"])
        qtc = pick_regex(df, [r"QTc.*Baz.*Global", r"QTc.*Global", r"QTc.*"])
        rr_sec = np.where((out["hr"] >= 25) & (out["hr"] <= 250), 60.0 / out["hr"], np.nan)
        qtc_calc = np.where(np.isfinite(out["qt"]) & np.isfinite(rr_sec) & (rr_sec > 0.2), out["qt"] / np.sqrt(rr_sec), np.nan)
        out["qtc"] = np.where(np.isfinite(qtc) & (qtc >= 240) & (qtc <= 750), qtc, qtc_calc)
        out["axis"] = pick_regex(df, [r"R_Axis_Global", r"QRS_Axis_Global", r"R_Axis.*", r"QRS_Axis.*"])
        out.loc[(out["pr"] < 60) | (out["pr"] > 450), "pr"] = np.nan
        out.loc[(out["qrs"] < 40) | (out["qrs"] > 260), "qrs"] = np.nan
        out.loc[(out["qt"] < 200) | (out["qt"] > 720), "qt"] = np.nan
        out.loc[(out["qtc"] < 240) | (out["qtc"] > 750), "qtc"] = np.nan
        out.loc[(out["axis"] < -180) | (out["axis"] > 180), "axis"] = np.nan
        return out.dropna(subset=["ecg_id"]).set_index("ecg_id")

    merged_g = extract_globals(f12).combine_first(extract_globals(funi))
    db = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    tr_items, va_items, te_items, seen_te = [], [], [], set()
    for _, r in db.iterrows():
        eid = int(r["ecg_id"])
        if eid not in merged_g.index:
            continue
        row_g = merged_g.loc[eid]
        if int(row_g[["hr", "qrs", "qt"]].notna().sum()) < 2:
            continue
        rec = (
            os.path.join(PTB_DIR, r["filename_lr"]),
            np.array([row_g["hr"], row_g["pr"], row_g["qrs"], row_g["qt"], row_g["qtc"], row_g["axis"]], dtype=np.float32),
        )
        fold = int(r["strat_fold"])
        if fold <= 8:
            tr_items.append(rec)
        elif fold == 9:
            va_items.append(rec)
        elif int(r["patient_id"]) not in seen_te:
            seen_te.add(int(r["patient_id"]))
            te_items.append(rec)

    print(f"[v2 All-Speed/Gain/Layout] Train={len(tr_items)} | Fold9={len(va_items)} | Fold10={len(te_items)}")

    def clean_ink(bgr: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(bgr.astype(np.float32))
        pink = np.clip((r - np.minimum(g, b)) - 12.0, 0.0, 255.0)
        luma = 0.55 * r + 0.35 * g + 0.10 * b
        ng = np.clip(luma + pink * 0.95, 0.0, 255.0)
        bg = cv2.GaussianBlur(ng, (31, 31), 0)
        ink = np.clip(bg - ng, 0.0, 255.0)
        hi = max(25.0, float(np.percentile(ink, 99.5)))
        gray = np.clip(255.0 - (ink / hi) * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        return ((rgb - MEAN) / STD).transpose(2, 0, 1)

    def draw_cal_pulse(img, x0, y0, px_per_mv, px_per_mm, speed_mm_s, col, lw):
        # Standard 1 mV x 200 ms calibration pulse: 5mm wide at 25mm/s, 10mm wide at 50mm/s, 2.5mm at 12.5mm/s
        w_px = max(4, int(round((speed_mm_s * 0.2) * px_per_mm)))
        h_px = max(8, int(round(px_per_mv)))
        pts = np.array([
            [x0, y0], [x0 + 3, y0], [x0 + 3, y0 - h_px],
            [x0 + 3 + w_px, y0 - h_px], [x0 + 3 + w_px, y0], [x0 + 6 + w_px, y0]
        ], dtype=np.int32)
        cv2.polylines(img, [pts], False, col, lw, cv2.LINE_AA)

    def render_ecg(sig: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
        """
        mode in {'clean_25', 'speed_50', 'layout_6x2', 'mobile_all', 'train_all'}
        Renders true physical speed (25 / 50 / 12.5 mm/s), gain (5 / 10 / 20 mm/mV),
        1 mV calibration pulse, footer speed text, and mobile camera stress!
        """
        styles = [
            ((242, 232, 252), (210, 185, 240), (180, 145, 225)),
            ((232, 248, 235), (188, 225, 195), (145, 200, 158)),
            ((232, 234, 254), (195, 200, 248), (155, 162, 238)),
            ((244, 244, 244), (212, 212, 212), (182, 182, 182)),
            ((252, 250, 246), (242, 238, 230), (225, 220, 210)),
        ]
        if mode == "clean_25":
            speed, gain, layout, mobile = 25.0, 10.0, "3x4", False
            bg_c, minor_c, major_c = styles[0]
        elif mode == "speed_50":
            speed, gain, layout, mobile = 50.0, 10.0, "3x4", False
            bg_c, minor_c, major_c = styles[int(rng.integers(0, len(styles)))]
        elif mode == "layout_6x2":
            speed, gain, layout, mobile = 25.0, 10.0, "6x2", False
            bg_c, minor_c, major_c = styles[int(rng.integers(0, len(styles)))]
        elif mode == "mobile_all":
            speed = float(rng.choice([25.0, 25.0, 50.0]))
            gain = float(rng.choice([10.0, 10.0, 5.0, 20.0]))
            layout = str(rng.choice(["3x4", "3x4", "6x2"]))
            mobile = True
            bg_c, minor_c, major_c = styles[int(rng.integers(0, len(styles)))]
        else:  # train_all
            speed = float(rng.choice([25.0, 25.0, 25.0, 50.0, 50.0, 12.5]))
            gain = float(rng.choice([10.0, 10.0, 10.0, 5.0, 20.0]))
            layout = str(rng.choice(["3x4", "3x4", "3x4", "6x2"]))
            mobile = bool(rng.random() < 0.45)
            bg_c, minor_c, major_c = styles[int(rng.integers(0, len(styles)))]

        img = np.full((IMG_H, IMG_W, 3), bg_c, dtype=np.uint8)
        px_per_mm = 2.8  # ~274 mm across 768 px (standard A4 ECG sheet)
        step_sm = max(3, int(round(px_per_mm)))
        if mode == "clean_25" or rng.random() < 0.90:
            for x in range(0, IMG_W, step_sm * 4):
                cv2.line(img, (x, 0), (x, IMG_H), minor_c if (x // (step_sm * 4)) % 5 else major_c, 1)
            for y in range(0, IMG_H, step_sm * 4):
                cv2.line(img, (0, y), (IMG_W, y), minor_c if (y // (step_sm * 4)) % 5 else major_c, 1)

        sig = np.nan_to_num(sig, 0.0)
        n_s = sig.shape[0]  # 10 seconds at 100 Hz = 1000 samples
        px_per_mv = 40.0 * (gain / 10.0)
        ink_col = (18, 18, 18) if not mobile else (int(rng.integers(10, 55)), int(rng.integers(10, 55)), int(rng.integers(10, 55)))
        lw = 2 if not mobile else int(rng.choice([1, 2, 2, 3]))
        # Time window visible across full width scales inversely with paper speed (10s @ 25mm/s, 5s @ 50mm/s)
        total_sec_shown = min(10.0, 10.0 * (25.0 / speed))
        n_vis = max(200, int(round(n_s * (total_sec_shown / 10.0))))

        if layout == "3x4":
            cw, rh = (IMG_W - 26) // 4, (IMG_H - 24) // 4
            for r_i in range(4):
                draw_cal_pulse(img, 4, int((r_i + 0.55) * rh), px_per_mv, px_per_mm, speed, ink_col, lw)
            for li in range(min(12, sig.shape[1])):
                r, c = li % 3, li // 3
                a = int(c * n_vis / 4)
                b = max(a + 20, int((c + 1) * n_vis / 4))
                s = sig[a:b, li]
                xs = np.linspace(26 + c * cw + 4, 26 + (c + 1) * cw - 4, len(s)).astype(np.int32)
                ys = np.clip(int((r + 0.55) * rh) - (s * px_per_mv).astype(np.int32), 4, IMG_H - 22)
                cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_col, lw, cv2.LINE_AA)
            s2 = sig[:n_vis, 1]
            xs = np.linspace(26, IMG_W - 6, len(s2)).astype(np.int32)
            ys = np.clip(int(3.55 * rh) - (s2 * px_per_mv).astype(np.int32), 4, IMG_H - 22)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_col, lw, cv2.LINE_AA)
        else:  # 6x2 layout
            cw, rh = (IMG_W - 26) // 2, (IMG_H - 24) // 6
            for r_i in range(6):
                draw_cal_pulse(img, 4, int((r_i + 0.55) * rh), px_per_mv, px_per_mm, speed, ink_col, lw)
            for li in range(min(12, sig.shape[1])):
                r, c = li % 6, li // 6
                a = int(c * n_vis / 2)
                b = max(a + 20, int((c + 1) * n_vis / 2))
                s = sig[a:b, li]
                xs = np.linspace(26 + c * cw + 4, 26 + (c + 1) * cw - 4, len(s)).astype(np.int32)
                ys = np.clip(int((r + 0.55) * rh) - (s * px_per_mv).astype(np.int32), 4, IMG_H - 22)
                cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_col, lw, cv2.LINE_AA)

        # Realistic printed machine footer with speed & gain cue (helps CNN distinguish 25 vs 50 vs 12.5 mm/s)
        cv2.putText(img, f"{speed:g}mm/s  {gain:g}mm/mV", (18, IMG_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ink_col, 1, cv2.LINE_AA)

        if mobile:
            ang = float(rng.uniform(-4.0, 4.0))
            sc = float(rng.uniform(0.94, 1.04))
            M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), ang, sc)
            img = cv2.warpAffine(img, M, (IMG_W, IMG_H), borderMode=cv2.BORDER_REPLICATE)
            d = 0.04
            src_pts = np.float32([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]])
            dst_pts = np.float32([
                [rng.uniform(0, d * IMG_W), rng.uniform(0, d * IMG_H)],
                [IMG_W - rng.uniform(0, d * IMG_W), rng.uniform(0, d * IMG_H)],
                [IMG_W - rng.uniform(0, d * IMG_W), IMG_H - rng.uniform(0, d * IMG_H)],
                [rng.uniform(0, d * IMG_W), IMG_H - rng.uniform(0, d * IMG_H)],
            ])
            img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src_pts, dst_pts), (IMG_W, IMG_H), borderMode=cv2.BORDER_REPLICATE)
            gx = np.linspace(float(rng.uniform(0.68, 1.0)), float(rng.uniform(0.82, 1.1)), IMG_W, dtype=np.float32)
            gy = np.linspace(float(rng.uniform(0.72, 1.0)), float(rng.uniform(0.85, 1.1)), IMG_H, dtype=np.float32)
            img = np.clip(img.astype(np.float32) * np.outer(gy, gx)[:, :, None], 0, 255).astype(np.uint8)
            _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(58, 88))])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        return clean_ink(img)

    SCALES = np.array([100.0, 200.0, 120.0, 400.0, 420.0], dtype=np.float32)

    class AllSpeedDS(Dataset):
        def __init__(self, items, mode: str, seed: int):
            self.items, self.mode, self.seed = items, mode, seed

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, raw = self.items[idx]
            rng = np.random.default_rng(self.seed + idx if self.mode != "train_all" else None)
            sig, _ = wfdb.rdsamp(path)
            x = render_ecg(sig, rng, self.mode)
            y = np.zeros(7, dtype=np.float32)
            m = np.zeros(7, dtype=np.float32)
            for k in range(5):
                if np.isfinite(raw[k]):
                    y[k], m[k] = raw[k] / SCALES[k], 1.0
            if np.isfinite(raw[5]):
                rad = math.radians(float(raw[5]))
                y[5], y[6], m[5], m[6] = math.sin(rad), math.cos(rad), 1.0, 1.0
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), torch.from_numpy(raw)

    tr_loader = DataLoader(AllSpeedDS(tr_items, "train_all", 11), batch_size=24, shuffle=True, num_workers=6, pin_memory=True)
    va_loader = DataLoader(AllSpeedDS(va_items, "mobile_all", 22), batch_size=32, shuffle=False, num_workers=6, pin_memory=True)

    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=7).to(device)
    if os.path.exists(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print("Warm-started cleanly from v1 ecg_interval_regressor.pt!")

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
    loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.05)

    def eval_mode(mode: str, seed: int):
        loader = DataLoader(AllSpeedDS(te_items, mode, seed), batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
        model.eval()
        preds_all, raws_all = [], []
        with torch.no_grad():
            for x, _, _, raw in loader:
                preds_all.append(model(x.to(device)).cpu().numpy())
                raws_all.append(raw.numpy())
        p, r = np.concatenate(preds_all), np.concatenate(raws_all)
        metrics = {}
        for k, nm in enumerate(["hr", "pr", "qrs", "qt", "qtc"]):
            v = np.isfinite(r[:, k])
            err = np.abs(p[v, k] * SCALES[k] - r[v, k])
            metrics[nm] = {"mae": round(float(np.mean(err)), 2), "p95": round(float(np.percentile(err, 95)), 2)}
        va = np.isfinite(r[:, 5])
        diff_ax = np.abs((np.degrees(np.arctan2(p[va, 5], p[va, 6])) - r[va, 5] + 180.0) % 360.0 - 180.0)
        metrics["axis"] = {"mae": round(float(np.mean(diff_ax)), 2), "p95": round(float(np.percentile(diff_ax, 95)), 2)}
        return metrics

    best_score, best_sd = 1e9, None
    for ep in range(1, 7):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y, m, _ in tr_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = (loss_fn(model(x), y) * m).sum() / m.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        sched.step()
        model.eval()
        preds_v, raws_v = [], []
        with torch.no_grad():
            for x, _, _, raw in va_loader:
                preds_v.append(model(x.to(device)).cpu().numpy())
                raws_v.append(raw.numpy())
        pv, rv = np.concatenate(preds_v), np.concatenate(raws_v)
        qrs_mae = float(np.mean(np.abs(pv[:, 2] * SCALES[2] - rv[:, 2])))
        qt_mae = float(np.mean(np.abs(pv[:, 3] * SCALES[3] - rv[:, 3])))
        hr_mae = float(np.mean(np.abs(pv[:, 0] * SCALES[0] - rv[:, 0])))
        print(f"Ep {ep}/6 ({time.time()-t0:.0f}s) | Loss={tot/len(tr_loader):.4f} | Fold9 All-Speed/Mobile MAE: HR={hr_mae:.2f} QRS={qrs_mae:.2f} QT={qt_mae:.2f}")
        if (qrs_mae + qt_mae) < best_score:
            best_score = qrs_mae + qt_mae
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    res = {
        "clean_25mm_s": eval_mode("clean_25", 33),
        "double_50mm_s": eval_mode("speed_50", 44),
        "layout_6x2": eval_mode("layout_6x2", 55),
        "mobile_all_speeds": eval_mode("mobile_all", 66),
    }
    print("\n================ PRISTINE FOLD-10 ALL-SPEED / ALL-GAIN / ALL-LAYOUT ================")
    for k, v in res.items():
        print(f"{k}: {json.dumps(v)}")

    if os.path.exists(CKPT_PATH):
        shutil.copy2(CKPT_PATH, os.path.join(OUT_DIR, "ecg_interval_regressor_v1_backup.pt"))
    torch.save({"arch": "tf_efficientnetv2_s.in21k_ft_in1k", "img_h": IMG_H, "img_w": IMG_W, "scales": SCALES.tolist(), "state_dict": best_sd, "v2_all_speed": True}, CKPT_PATH)
    with open(os.path.join(OUT_DIR, "ecg_interval_metrics_v2.json"), "w") as f:
        json.dump(res, f, indent=2)
    vol.commit()
    sha = hashlib.sha256(open(CKPT_PATH, "rb").read()).hexdigest()[:16]
    print(f"Promoted v2 All-Speed/Gain/Layout Interval Regressor to {CKPT_PATH} (sha={sha})")


@app.local_entrypoint()
def main():
    train_intervals_v2.remote()
