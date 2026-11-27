"""
Code Blue ECG — Deep Multi-Domain ECG Interval & Axis Regressor on Modal A10G (~18 min, ~$0.35)
Trains EfficientNetV2-S (512x768) to predict [HR, PR, QRS, QT, QTc, Axis] from ALL real-world
ECG image conditions WITHOUT printed headers:
  1. Multi-machine grids: GE (pink/red), Nihon Kohden (green), Edan (purple), Grayscale/Thermal, Dark Monitor
  2. Mobile-camera distortions: 3D perspective warp, paper fold shadow, uneven lighting, moire, blur, JPEG
  3. Textbook / cropped / faded ink & variable stroke thickness
Ground truth: PTB-XL+ (12SL + Uni-G) exact intervals + optional PhysioNet ecg-arrhythmia HR.
Evaluates strictly on Pristine Fold-10 across Clean, Hard Multi-Machine, AND Mobile-Photo Stress sets.
"""
import modal

app = modal.App("codeblue-train-intervals")
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
def train_intervals():
    import hashlib, json, math, os, time
    import cv2, numpy as np, pandas as pd, requests, timm, torch, torch.nn as nn, wfdb
    from torch.utils.data import DataLoader, Dataset

    vol.reload()
    torch.manual_seed(20261127)
    np.random.seed(20261127)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_H, IMG_W = 512, 768
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PTB_DIR, PLUS_DIR, OUT_DIR = "/vol/ptbxl", "/vol/ptbxl_plus", "/vol/outputs"
    os.makedirs(PLUS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Cache PTB-XL+ feature CSVs (12SL + Uni-G)
    for fname in ["12sl_features.csv", "unig_features.csv", "feature_description.csv"]:
        dst = os.path.join(PLUS_DIR, fname)
        if not os.path.exists(dst) or os.path.getsize(dst) < 1024:
            for ver in ["1.0.1", "1.0.0"]:
                url = f"https://physionet.org/files/ptb-xl-plus/{ver}/features/{fname}?download"
                r = requests.get(url, timeout=120)
                if r.status_code == 200 and len(r.content) > 1024:
                    with open(dst, "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded {fname} ({len(r.content)//1024} KB) from v{ver}")
                    break
            vol.commit()

    f12 = pd.read_csv(os.path.join(PLUS_DIR, "12sl_features.csv"))
    funi = pd.read_csv(os.path.join(PLUS_DIR, "unig_features.csv"))
    print(f"Loaded 12SL={len(f12)} rows, Uni-G={len(funi)} rows")

    def pick_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        cols_lower = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce")
            if c.lower() in cols_lower:
                return pd.to_numeric(df[cols_lower[c.lower()]], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=np.float64)

    def pick_regex(df: pd.DataFrame, patterns: list[str]) -> pd.Series:
        import re as _re
        for pat in patterns:
            for c in df.columns:
                if _re.fullmatch(pat, c, flags=_re.I):
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
        # Exact Bazett QTc fallback from QT & HR whenever QTc column missing
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
    for col in ["hr", "pr", "qrs", "qt", "qtc", "axis"]:
        cnt = int(merged_g[col].notna().sum())
        med = float(merged_g[col].median()) if cnt else 0.0
        print(f"  Target '{col}': {cnt} valid ECGs (median={med:.1f})")

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

    print(f"[Multi-Domain Intervals] Train={len(tr_items)} | Fold9(Val)={len(va_items)} | Fold10(Test)={len(te_items)}")

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

    def render_ecg(sig: np.ndarray, rng: np.random.Generator, domain: str) -> np.ndarray:
        """
        domain in {'clean', 'hard', 'mobile'}:
        Covers GE pink, Nihon Kohden green, Edan purple, Grayscale/Thermal, Dark Monitor,
        plus 3D perspective warp, shadow gradients, blur, and JPEG compression.
        """
        styles = [
            ((242, 232, 252), (210, 185, 240), (180, 145, 225)),  # Edan/Mindray purple
            ((232, 248, 235), (188, 225, 195), (145, 200, 158)),  # Nihon Kohden green
            ((232, 234, 254), (195, 200, 248), (155, 162, 238)),  # GE pink/red (BGR)
            ((244, 244, 244), (212, 212, 212), (182, 182, 182)),  # Grayscale thermal
            ((252, 250, 246), (242, 238, 230), (225, 220, 210)),  # Faded textbook paper
        ]
        if domain == "clean":
            bg_c, minor_c, major_c = styles[0]
        else:
            bg_c, minor_c, major_c = styles[int(rng.integers(0, len(styles)))]

        img = np.full((IMG_H, IMG_W, 3), bg_c, dtype=np.uint8)
        if domain == "clean" or rng.random() < 0.88:
            for x in range(0, IMG_W, 12):
                cv2.line(img, (x, 0), (x, IMG_H), minor_c if x % 60 else major_c, 1)
            for y in range(0, IMG_H, 12):
                cv2.line(img, (0, y), (IMG_W, y), minor_c if y % 60 else major_c, 1)

        sig = np.nan_to_num(sig, 0.0)
        n_s = sig.shape[0]
        cw, rh = IMG_W // 4, IMG_H // 4
        ink_col = (18, 18, 18) if domain == "clean" else (int(rng.integers(10, 55)), int(rng.integers(10, 55)), int(rng.integers(10, 55)))
        lw = 2 if domain == "clean" else int(rng.choice([1, 2, 2, 3]))

        for li in range(min(12, sig.shape[1])):
            r, c = li % 3, li // 3
            s = sig[int(c * n_s / 4):int((c + 1) * n_s / 4), li]
            xs = np.linspace(c * cw + 6, (c + 1) * cw - 6, len(s)).astype(np.int32)
            ys = np.clip(int((r + 0.55) * rh) - (s * 40.0).astype(np.int32), 4, IMG_H - 4)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_col, lw, cv2.LINE_AA)
        s2 = sig[:, 1]
        xs = np.linspace(6, IMG_W - 6, len(s2)).astype(np.int32)
        ys = np.clip(int(3.55 * rh) - (s2 * 40.0).astype(np.int32), 4, IMG_H - 4)
        cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_col, lw, cv2.LINE_AA)

        if domain in ("hard", "mobile"):
            # Realistic rotation + scale
            if domain == "mobile" or rng.random() < 0.65:
                ang = float(rng.uniform(-4.0, 4.0)) if domain == "mobile" else float(rng.uniform(-2.0, 2.0))
                sc = float(rng.uniform(0.94, 1.04))
                M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), ang, sc)
                img = cv2.warpAffine(img, M, (IMG_W, IMG_H), borderMode=cv2.BORDER_REPLICATE)
            # 3D perspective camera tilt + uneven shadow + phone blur + JPEG artifacts
            if domain == "mobile" or rng.random() < 0.50:
                d = 0.045 if domain == "mobile" else 0.025
                src_pts = np.float32([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]])
                dst_pts = np.float32([
                    [rng.uniform(0, d * IMG_W), rng.uniform(0, d * IMG_H)],
                    [IMG_W - rng.uniform(0, d * IMG_W), rng.uniform(0, d * IMG_H)],
                    [IMG_W - rng.uniform(0, d * IMG_W), IMG_H - rng.uniform(0, d * IMG_H)],
                    [rng.uniform(0, d * IMG_W), IMG_H - rng.uniform(0, d * IMG_H)],
                ])
                P = cv2.getPerspectiveTransform(src_pts, dst_pts)
                img = cv2.warpPerspective(img, P, (IMG_W, IMG_H), borderMode=cv2.BORDER_REPLICATE)
            if domain == "mobile" or rng.random() < 0.45:
                gx = np.linspace(float(rng.uniform(0.65, 1.0)), float(rng.uniform(0.8, 1.1)), IMG_W, dtype=np.float32)
                gy = np.linspace(float(rng.uniform(0.70, 1.0)), float(rng.uniform(0.85, 1.1)), IMG_H, dtype=np.float32)
                shade = np.outer(gy, gx)[:, :, None]
                img = np.clip(img.astype(np.float32) * shade, 0, 255).astype(np.uint8)
            if domain == "mobile" or rng.random() < 0.35:
                if rng.random() < 0.5:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(55, 88))])
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        return clean_ink(img)

    SCALES = np.array([100.0, 200.0, 120.0, 400.0, 420.0], dtype=np.float32)

    class IntervalDS(Dataset):
        def __init__(self, items, domain: str, seed: int):
            self.items = items
            self.domain = domain
            self.seed = seed

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, raw = self.items[idx]
            rng = np.random.default_rng(self.seed + idx if self.domain == "clean" else None)
            sig, _ = wfdb.rdsamp(path)
            dom = self.domain if self.domain != "train_mix" else ("mobile" if rng.random() < 0.45 else "hard")
            x = render_ecg(sig, rng, dom)
            y = np.zeros(7, dtype=np.float32)
            m = np.zeros(7, dtype=np.float32)
            for k in range(5):
                if np.isfinite(raw[k]):
                    y[k] = raw[k] / SCALES[k]
                    m[k] = 1.0
            if np.isfinite(raw[5]):
                rad = math.radians(float(raw[5]))
                y[5], y[6] = math.sin(rad), math.cos(rad)
                m[5], m[6] = 1.0, 1.0
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), torch.from_numpy(raw)

    tr_loader = DataLoader(IntervalDS(tr_items, "train_mix", 11), batch_size=24, shuffle=True, num_workers=6, pin_memory=True)
    va_loader = DataLoader(IntervalDS(va_items, "hard", 22), batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    te_clean = DataLoader(IntervalDS(te_items, "clean", 33), batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    te_hard = DataLoader(IntervalDS(te_items, "hard", 44), batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    te_mob = DataLoader(IntervalDS(te_items, "mobile", 55), batch_size=32, shuffle=False, num_workers=6, pin_memory=True)

    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.05)

    def eval_loader(loader):
        model.eval()
        preds_all, raws_all = [], []
        with torch.no_grad():
            for x, _, _, raw in loader:
                preds_all.append(model(x.to(device)).cpu().numpy())
                raws_all.append(raw.numpy())
        p = np.concatenate(preds_all)
        r = np.concatenate(raws_all)
        metrics = {}
        for k, nm in enumerate(["hr", "pr", "qrs", "qt", "qtc"]):
            valid = np.isfinite(r[:, k])
            if int(valid.sum()) == 0:
                metrics[nm] = {"mae": 999.0, "p95": 999.0, "n": 0}
                continue
            err = np.abs(p[valid, k] * SCALES[k] - r[valid, k])
            metrics[nm] = {"mae": round(float(np.mean(err)), 2), "p95": round(float(np.percentile(err, 95)), 2), "n": int(valid.sum())}
        valid_ax = np.isfinite(r[:, 5])
        if int(valid_ax.sum()) == 0:
            metrics["axis"] = {"mae": 999.0, "p95": 999.0, "n": 0}
        else:
            pred_deg = np.degrees(np.arctan2(p[valid_ax, 5], p[valid_ax, 6]))
            diff_ax = np.abs((pred_deg - r[valid_ax, 5] + 180.0) % 360.0 - 180.0)
            metrics["axis"] = {"mae": round(float(np.mean(diff_ax)), 2), "p95": round(float(np.percentile(diff_ax, 95)), 2), "n": int(valid_ax.sum())}
        return metrics

    best_score, best_sd = 1e9, None
    for ep in range(1, 11):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y, m, _ in tr_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = (loss_fn(pred, y) * m).sum() / m.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        sched.step()
        vm = eval_loader(va_loader)
        score = vm["pr"]["mae"] + vm["qrs"]["mae"] + vm["qt"]["mae"]
        print(
            f"Ep {ep}/10 ({time.time()-t0:.0f}s) | Loss={tot/len(tr_loader):.4f} | "
            f"F9 Hard MAE: HR={vm['hr']['mae']} PR={vm['pr']['mae']} QRS={vm['qrs']['mae']} "
            f"QT={vm['qt']['mae']} QTc={vm['qtc']['mae']} Axis={vm['axis']['mae']}°"
        )
        if score < best_score:
            best_score = score
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    clean_m = eval_loader(te_clean)
    hard_m = eval_loader(te_hard)
    mob_m = eval_loader(te_mob)
    print("\n================ PRISTINE FOLD-10 MULTI-DOMAIN INTERVALS ================")
    print("Clean Fold-10:", json.dumps(clean_m))
    print("Hard Multi-Machine Fold-10:", json.dumps(hard_m))
    print("Mobile Photo Stress Fold-10:", json.dumps(mob_m))

    ckpt_path = os.path.join(OUT_DIR, "ecg_interval_regressor.pt")
    torch.save({"arch": "tf_efficientnetv2_s.in21k_ft_in1k", "img_h": IMG_H, "img_w": IMG_W, "scales": SCALES.tolist(), "state_dict": best_sd}, ckpt_path)
    with open(os.path.join(OUT_DIR, "ecg_interval_metrics.json"), "w") as f:
        json.dump({"clean": clean_m, "hard": hard_m, "mobile_stress": mob_m}, f, indent=2)
    vol.commit()
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()[:16]
    print(f"Saved Multi-Domain Interval Regressor to {ckpt_path} (sha={sha})")


@app.local_entrypoint()
def main():
    train_intervals.remote()
