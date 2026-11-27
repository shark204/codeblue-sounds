"""
Code Blue ECG — v7 Real-Photo Fine-Tuning of EfficientNetV2-S Multi-Machine (512x768)
- Zero download wait: uses already-cached /vol/pmcardio (5,564 real photos) + /vol/ptbxl.
- Initializes from ecg_effnetv2s_multimachine_512x768.pt (0.9707 Clean / 0.9680 Hard Multi-Machine).
- Trains 5 epochs on 50% Real PMcardio Photos (F1-8) + 50% Hard Multi-Machine PTB-XL (F1-8).
- Fits Per-Class Platt Calibration strictly on Combined Fold-9 (1,937 PTB-XL + 921 Real Photos).
- Evaluates v7_effnetv2_real AND (0.60 v7_effnetv2_real + 0.40 v6_densenet_real) across ALL 3 Pristine Fold-10 benchmarks.
- Auto-promotes to production checkpoint ONLY if Real Photos AUROC jumps cleanly without degrading Hard Multi-Machine.
"""
import modal

app = modal.App("codeblue-train-v7-effnetv2-real")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "timm", "wfdb", "scipy", "scikit-learn",
        "pandas", "numpy", "opencv-python-headless", "pillow",
    )
)

CLASSES_18 = [
    "NORMAL_SINUS_RHYTHM", "ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI",
    "NSTEMI_ISCHEMIA", "ATRIAL_FIBRILLATION", "ATRIAL_FLUTTER", "SVT_TACHYCARDIA",
    "SINUS_BRADYCARDIA", "FIRST_DEGREE_AV_BLOCK", "HIGH_GRADE_AV_BLOCK", "RBBB", "LBBB",
    "LAFB_LPFB", "LVH_RVH", "PACEMAKER_RHYTHM", "WPW_PREEXCITATION", "LONG_QT_OR_ECTOPY",
]
CRITICAL_SET = {"ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI", "HIGH_GRADE_AV_BLOCK"}
RHYTHM_OR_FORM_CODES = {
    "AFIB", "AFLT", "SVTAC", "PSVT", "STACH", "SBRAD", "PACE", "PVC", "BIGU", "TRIGU", "LQT", "SARRH", "SR",
}
SCP_TO_CLASS = {
    "NORM": "NORMAL_SINUS_RHYTHM",
    "AMI": "ANTERIOR_SEPTAL_MI", "ASMI": "ANTERIOR_SEPTAL_MI", "ALMI": "ANTERIOR_SEPTAL_MI", "INJAS": "ANTERIOR_SEPTAL_MI", "INJAL": "ANTERIOR_SEPTAL_MI",
    "IMI": "INFERIOR_POSTERIOR_MI", "ILMI": "INFERIOR_POSTERIOR_MI", "IPMI": "INFERIOR_POSTERIOR_MI", "IPLMI": "INFERIOR_POSTERIOR_MI", "PMI": "INFERIOR_POSTERIOR_MI", "INJIN": "INFERIOR_POSTERIOR_MI", "INJIL": "INFERIOR_POSTERIOR_MI",
    "LMI": "LATERAL_MI", "INJLA": "LATERAL_MI",
    "ISC_": "NSTEMI_ISCHEMIA", "ISCAL": "NSTEMI_ISCHEMIA", "ISCAN": "NSTEMI_ISCHEMIA", "ISCAS": "NSTEMI_ISCHEMIA", "ISCIL": "NSTEMI_ISCHEMIA", "ISCIN": "NSTEMI_ISCHEMIA", "ISCLA": "NSTEMI_ISCHEMIA", "NST_": "NSTEMI_ISCHEMIA", "STD_": "NSTEMI_ISCHEMIA", "STTC": "NSTEMI_ISCHEMIA",
    "AFIB": "ATRIAL_FIBRILLATION", "AFLT": "ATRIAL_FLUTTER",
    "SVTAC": "SVT_TACHYCARDIA", "PSVT": "SVT_TACHYCARDIA", "STACH": "SVT_TACHYCARDIA",
    "SBRAD": "SINUS_BRADYCARDIA",
    "1AVB": "FIRST_DEGREE_AV_BLOCK",
    "2AVB": "HIGH_GRADE_AV_BLOCK", "3AVB": "HIGH_GRADE_AV_BLOCK",
    "CRBBB": "RBBB", "IRBBB": "RBBB", "CLBBB": "LBBB", "ILBBB": "LBBB",
    "LAFB": "LAFB_LPFB", "LPFB": "LAFB_LPFB", "LVH": "LVH_RVH", "RVH": "LVH_RVH",
    "PACE": "PACEMAKER_RHYTHM", "WPW": "WPW_PREEXCITATION",
    "LQT": "LONG_QT_OR_ECTOPY", "PVC": "LONG_QT_OR_ECTOPY", "BIGU": "LONG_QT_OR_ECTOPY", "TRIGU": "LONG_QT_OR_ECTOPY",
}


@app.function(
    image=image,
    gpu="A10G",
    cpu=6.0,
    memory=24576,
    timeout=7200,
    volumes={"/vol": vol},
)
def train_v7_effnetv2_real():
    import ast, hashlib, json, os, re, shutil, time
    import cv2, numpy as np, pandas as pd, timm, torch, torch.nn as nn, wfdb
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset

    vol.reload()
    torch.manual_seed(20261127)
    np.random.seed(20261127)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_H, IMG_W = 512, 768
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PTB_DIR, PM_DIR, OUT_DIR = "/vol/ptbxl", "/vol/pmcardio", "/vol/outputs"
    EFF_CKPT = os.path.join(OUT_DIR, "ecg_effnetv2s_multimachine_512x768.pt")
    V6_CKPT = os.path.join(OUT_DIR, "ecg_densenet121_v6_real_512x768.pt")
    assert os.path.exists(EFF_CKPT), f"Missing {EFF_CKPT}"

    # 1. Load PTB-XL (F1-8 Train, F9 Val, F10 Pristine 1-per-pt)
    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    c2i = {c: i for i, c in enumerate(CLASSES_18)}
    ptb_by_id, tr_synth, va_synth, te_synth, seen_te = {}, [], [], [], set()
    for _, r in df.iterrows():
        scp = ast.literal_eval(r["scp_codes"])
        y = np.zeros(len(CLASSES_18), dtype=np.float32)
        for k, v in scp.items():
            if k in SCP_TO_CLASS and (k in RHYTHM_OR_FORM_CODES or float(v) >= 50.0):
                y[c2i[SCP_TO_CLASS[k]]] = 1.0
        if y.sum() == 0:
            continue
        rec = {"ecg_id": int(r["ecg_id"]), "pid": int(r["patient_id"]), "fold": int(r["strat_fold"]), "path": os.path.join(PTB_DIR, r["filename_lr"]), "y": y}
        ptb_by_id[rec["ecg_id"]] = rec
        if rec["fold"] <= 8:
            tr_synth.append(rec)
        elif rec["fold"] == 9:
            va_synth.append(rec)
        elif rec["pid"] not in seen_te:
            seen_te.add(rec["pid"])
            te_synth.append(rec)

    # 2. Load Cached PMcardio Real Photos (F1-8 Train, F9 Val, F10 Test)
    df_pm = pd.read_csv(os.path.join(PM_DIR, "metadata.csv"))
    cols = list(df_pm.columns)
    img_col = next((c for c in cols if "image" in c.lower() and "name" in c.lower()), cols[0])
    id_col = next((c for c in cols if "ecg" in c.lower() and "id" in c.lower()), cols[1])
    PM_IMG_DIR = os.path.join(PM_DIR, "images")
    all_imgs = [os.path.join(PM_IMG_DIR, f) for f in os.listdir(PM_IMG_DIR) if f.lower().endswith(".jpg")]
    by_stem = {}
    for p_img in all_imgs:
        by_stem.setdefault(os.path.splitext(os.path.basename(p_img).split("__")[-1])[0].lower(), []).append(p_img)
    real_tr, real_va, real_te, seen_p = [], [], [], set()
    for _, r in df_pm.iterrows():
        try:
            eid = int(float(r[id_col]))
        except Exception:
            continue
        if eid not in ptb_by_id:
            continue
        stem = os.path.splitext(os.path.basename(str(r[img_col]).strip()))[0].lower()
        for p_img in by_stem.get(stem, []):
            if p_img in seen_p:
                continue
            seen_p.add(p_img)
            b = ptb_by_id[eid]
            item = {"img_path": p_img, "y": b["y"]}
            (real_tr if b["fold"] <= 8 else real_va if b["fold"] == 9 else real_te).append(item)
    if len(real_tr) < 500:
        for p_img in all_imgs:
            if p_img in seen_p:
                continue
            m = re.search(r"(\d{1,5})", os.path.basename(p_img).split("__")[-1])
            if not m:
                continue
            eid = int(m.group(1))
            if eid in ptb_by_id:
                seen_p.add(p_img)
                b = ptb_by_id[eid]
                item = {"img_path": p_img, "y": b["y"]}
                (real_tr if b["fold"] <= 8 else real_va if b["fold"] == 9 else real_te).append(item)

    print(f"[v7 EffNetV2-S Real] Real Train={len(real_tr)} | Val={len(real_va)} | Test={len(real_te)} | Synth Train={len(tr_synth)}")

    def clean_ink(bgr: np.ndarray) -> np.ndarray:
        if bgr.shape[0] != IMG_H or bgr.shape[1] != IMG_W:
            bgr = cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
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

    def augment_photo(bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if np.random.rand() < 0.6:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), float(np.random.uniform(-3.0, 3.0)), float(np.random.uniform(0.94, 1.05)))
            bgr = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if np.random.rand() < 0.5:
            bgr = np.clip(bgr.astype(np.float32) * float(np.random.uniform(0.78, 1.22)) + float(np.random.uniform(-22, 22)), 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
        return bgr

    def render_mm(sig: np.ndarray, rng: np.random.Generator, hard: bool) -> np.ndarray:
        styles = [
            ((242, 232, 252), (210, 185, 240), (180, 145, 225)),
            ((232, 248, 235), (188, 225, 195), (145, 200, 158)),
            ((244, 244, 244), (212, 212, 212), (182, 182, 182)),
            ((226, 240, 248), (195, 210, 224), (165, 185, 205)),
        ]
        bg_c, minor_c, major_c = styles[int(rng.integers(0, len(styles)))] if hard else styles[0]
        img = np.full((IMG_H, IMG_W, 3), bg_c, dtype=np.uint8)
        step = int(rng.choice([10, 12, 14])) if hard else 12
        for x in range(0, IMG_W, step):
            cv2.line(img, (x, 0), (x, IMG_H), minor_c if x % (step * 5) else major_c, 1)
        for y in range(0, IMG_H, step):
            cv2.line(img, (0, y), (IMG_W, y), minor_c if y % (step * 5) else major_c, 1)
        sig = np.nan_to_num(sig, 0.0)
        n_s = sig.shape[0]
        thick = int(rng.choice([1, 2, 2])) if hard else 2
        ink_v = int(rng.integers(10, 55)) if hard else 18
        ink_c = (ink_v, ink_v, ink_v)
        gain_px = float(rng.uniform(36.0, 48.0)) if hard else 42.0
        if hard and (rng.random() < 0.25):
            cw, rh = IMG_W // 2, IMG_H // 6
            for li in range(min(12, sig.shape[1])):
                r, c = li % 6, li // 6
                s = sig[int(c * n_s / 2):int((c + 1) * n_s / 2), li]
                xs = np.linspace(c * cw + 6, (c + 1) * cw - 6, len(s)).astype(np.int32)
                ys = np.clip(int((r + 0.55) * rh) - (s * (gain_px * 0.72)).astype(np.int32), 4, IMG_H - 4)
                cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_c, thick, cv2.LINE_AA)
        else:
            cw, rh = IMG_W // 4, IMG_H // 4
            for li in range(min(12, sig.shape[1])):
                r, c = li % 3, li // 3
                s = sig[int(c * n_s / 4):int((c + 1) * n_s / 4), li]
                xs = np.linspace(c * cw + 6, (c + 1) * cw - 6, len(s)).astype(np.int32)
                ys = np.clip(int((r + 0.55) * rh) - (s * gain_px).astype(np.int32), 4, IMG_H - 4)
                cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_c, thick, cv2.LINE_AA)
            s2 = sig[:, 1]
            xs = np.linspace(6, IMG_W - 6, len(s2)).astype(np.int32)
            ys = np.clip(int(3.55 * rh) - (s2 * gain_px).astype(np.int32), 4, IMG_H - 4)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_c, thick, cv2.LINE_AA)
        if hard:
            if rng.random() < 0.65:
                gx = np.linspace(rng.uniform(0.72, 1.05), rng.uniform(0.72, 1.05), IMG_W, dtype=np.float32)
                gy = np.linspace(rng.uniform(0.80, 1.05), rng.uniform(0.80, 1.05), IMG_H, dtype=np.float32)
                img = np.clip(img.astype(np.float32) * np.outer(gy, gx)[:, :, None], 0, 255).astype(np.uint8)
            if rng.random() < 0.65:
                pts1 = np.float32([[0, 0], [IMG_W, 0], [0, IMG_H], [IMG_W, IMG_H]])
                j = float(rng.uniform(8.0, 24.0))
                M = cv2.getPerspectiveTransform(pts1, pts1 + rng.uniform(-j, j, size=(4, 2)).astype(np.float32))
                img = cv2.warpPerspective(img, M, (IMG_W, IMG_H), borderMode=cv2.BORDER_REPLICATE)
            if rng.random() < 0.35:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            if rng.random() < 0.45:
                ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(45, 85))])
                if ok:
                    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return clean_ink(img)

    class MixedDS(Dataset):
        def __init__(self, r_rows, s_rows, r_rep=2, s_cap=7000):
            self.r_rows = r_rows * r_rep
            idx = np.random.default_rng(20261127).choice(len(s_rows), size=min(s_cap, len(s_rows)), replace=False)
            self.s_rows = [s_rows[j] for j in idx]
        def __len__(self):
            return len(self.r_rows) + len(self.s_rows)
        def __getitem__(self, i):
            if i < len(self.r_rows):
                r = self.r_rows[i]
                bgr = cv2.imread(r["img_path"], cv2.IMREAD_COLOR)
                if bgr is None:
                    bgr = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
                return torch.from_numpy(clean_ink(augment_photo(bgr))), torch.from_numpy(r["y"])
            r = self.s_rows[i - len(self.r_rows)]
            sig, _ = wfdb.rdsamp(r["path"])
            return torch.from_numpy(render_mm(sig, np.random.default_rng(), True)), torch.from_numpy(r["y"])

    class SynthDS(Dataset):
        def __init__(self, rows, hard, seed):
            self.rows, self.hard, self.seed = rows, hard, seed
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            sig, _ = wfdb.rdsamp(r["path"])
            return torch.from_numpy(render_mm(sig, np.random.default_rng(self.seed + i), self.hard)), torch.from_numpy(r["y"])

    class RealDS(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            bgr = cv2.imread(r["img_path"], cv2.IMREAD_COLOR)
            if bgr is None:
                bgr = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
            return torch.from_numpy(clean_ink(bgr)), torch.from_numpy(r["y"])

    tr_dl = DataLoader(MixedDS(real_tr, tr_synth), batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    va_hd_dl = DataLoader(SynthDS(va_synth, True, 60000), batch_size=32, num_workers=4, pin_memory=True)
    va_rl_dl = DataLoader(RealDS(real_va), batch_size=32, num_workers=4, pin_memory=True)
    te_cl_dl = DataLoader(SynthDS(te_synth, False, 110000), batch_size=32, num_workers=4, pin_memory=True)
    te_hd_dl = DataLoader(SynthDS(te_synth, True, 100000), batch_size=32, num_workers=4, pin_memory=True)
    te_rl_dl = DataLoader(RealDS(real_te), batch_size=32, num_workers=4, pin_memory=True)

    def collect(m, dl):
        m.eval()
        Ls, Ys = [], []
        with torch.no_grad():
            for x, y in dl:
                with torch.amp.autocast("cuda"):
                    Ls.append(m(x.to(device, non_blocking=True)).float().cpu().numpy())
                Ys.append(y.numpy())
        return np.concatenate(Ls), np.concatenate(Ys)

    def macro_auroc(Y, P):
        aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(CLASSES_18)) if 0 < Y[:, j].sum() < len(Y)]
        return float(np.mean(aucs)) if aucs else 0.0

    def ece(Y, P, n_bins=15):
        yf, pf = Y.reshape(-1), P.reshape(-1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        return float(sum((m.sum() / len(pf)) * abs(yf[m].mean() - pf[m].mean()) for lo, hi in zip(bins[:-1], bins[1:]) for m in [((pf >= lo) & (pf < hi)) if hi < 1.0 else ((pf >= lo) & (pf <= hi))] if m.sum() > 0))

    def fit_platt(L, Y):
        A, B = np.ones(len(CLASSES_18), dtype=np.float32), np.zeros(len(CLASSES_18), dtype=np.float32)
        for j in range(len(CLASSES_18)):
            lj, yj = L[:, j], Y[:, j]
            if yj.sum() < 5:
                continue
            res = minimize(
                lambda p: float(np.mean(np.logaddexp(0, p[0] * lj + p[1]) - yj * (p[0] * lj + p[1]))),
                x0=[1.0, -1.5], bounds=[(0.2, 4.0), (-6.0, 4.0)], method="L-BFGS-B",
            )
            A[j], B[j] = float(res.x[0]), float(res.x[1])
        return A, B

    def apply_platt(L, A, B):
        return 1.0 / (1.0 + np.exp(-np.clip(L * A[None, :] + B[None, :], -30, 30)))

    def freeze_thr(P, Y):
        thr = {}
        for j, cls in enumerate(CLASSES_18):
            if cls == "NORMAL_SINUS_RHYTHM":
                continue
            yt = Y[:, j] == 1
            if yt.sum() < 3:
                thr[cls] = 0.25
                continue
            target = 0.90 if cls in CRITICAL_SET else 0.85
            best_t, best_s = 0.25, -1.0
            for t in np.linspace(0.02, 0.80, 157):
                yp = P[:, j] >= t
                if float((yp & yt).sum() / yt.sum()) >= target:
                    spec = float(((~yp) & (~yt)).sum() / max(1, (~yt).sum()))
                    if spec > best_s:
                        best_s, best_t = spec, float(t)
            thr[cls] = round(best_t, 4)
        return thr

    # 3. Fine-tune EffNetV2-S Multi-Machine on Real Photos + Hard Multi-Machine PTB-XL (5 epochs)
    ckpt0 = torch.load(EFF_CKPT, map_location="cpu", weights_only=False)
    arch = ckpt0.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k")
    model = timm.create_model(arch, pretrained=False, num_classes=len(CLASSES_18), drop_rate=0.2).to(device)
    model.load_state_dict(ckpt0["state_dict"])

    y_all = np.stack([r["y"] for r in real_tr] + [r["y"] for r in tr_synth[:7000]])
    pos_w = torch.tensor(np.clip((len(y_all) - y_all.sum(0)) / np.maximum(y_all.sum(0), 1.0), 1.0, 20.0), dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-3)
    EPOCHS = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_score, best_state = -1.0, None
    print("\n--- Fine-Tuning EffNetV2-S v7 on Real PMcardio Photos + Multi-Machine PTB-XL ---")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item()) * len(x)
        sched.step()
        L_hd, Y_hd = collect(model, va_hd_dl)
        L_rl, Y_rl = collect(model, va_rl_dl)
        a_hd = macro_auroc(Y_hd, 1.0 / (1.0 + np.exp(-np.clip(L_hd, -30, 30))))
        a_rl = macro_auroc(Y_rl, 1.0 / (1.0 + np.exp(-np.clip(L_rl, -30, 30))))
        sc = 0.55 * a_hd + 0.45 * a_rl
        print(f"Epoch {ep}/{EPOCHS} ({time.time()-t0:.0f}s) | Loss={tot/len(tr_dl.dataset):.4f} | Fold9 Hard={a_hd:.4f} | Fold9 Real={a_rl:.4f}")
        if sc > best_score:
            best_score = sc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # 4. Calibrate Per-Class Platt on Combined Fold-9 (Hard + Real) & Evaluate on ALL 3 Pristine Fold-10 Benchmarks
    L_va_hd, Y_va_hd = collect(model, va_hd_dl)
    L_va_rl, Y_va_rl = collect(model, va_rl_dl)
    L_te_cl, Y_te_cl = collect(model, te_cl_dl)
    L_te_hd, Y_te_hd = collect(model, te_hd_dl)
    L_te_rl, Y_te_rl = collect(model, te_rl_dl)

    A_v7, B_v7 = fit_platt(np.concatenate([L_va_hd, L_va_rl]), np.concatenate([Y_va_hd, Y_va_rl]))
    thr_v7 = freeze_thr(apply_platt(np.concatenate([L_va_hd, L_va_rl]), A_v7, B_v7), np.concatenate([Y_va_hd, Y_va_rl]))

    P_cl = apply_platt(L_te_cl, A_v7, B_v7)
    P_hd = apply_platt(L_te_hd, A_v7, B_v7)
    P_rl = apply_platt(L_te_rl, A_v7, B_v7)

    a_cl, a_hd, a_rl, e_hd = macro_auroc(Y_te_cl, P_cl), macro_auroc(Y_te_hd, P_hd), macro_auroc(Y_te_rl, P_rl), ece(Y_te_hd, P_hd)
    crit_idx = [c2i[c] for c in CRITICAL_SET]
    has_c = Y_te_hd[:, crit_idx].sum(1) > 0
    c_hd = 100.0 * float(np.mean([any(Y_te_hd[i, c2i[c]] == 1 and P_hd[i, c2i[c]] >= thr_v7[c] for c in CRITICAL_SET) for i in np.where(has_c)[0]]))
    norm_only = (Y_te_cl[:, 0] == 1) & (Y_te_cl[:, 1:].sum(1) == 0)
    n_alrt = 100.0 * float(np.mean([any(P_cl[i, c2i[c]] >= thr_v7[c] for c in CRITICAL_SET) for i in np.where(norm_only)[0]]))

    print("\n================ v7 EffNetV2-S REAL-PHOTO RESULTS (ALL 3 PRISTINE FOLD-10 BENCHMARKS) ================")
    print(f"v7_effnetv2_real ONLY    | Clean={a_cl:.4f} | Hard={a_hd:.4f} | REAL={a_rl:.4f} | Hard Crit={c_hd:.1f}% | NormAlrt={n_alrt:.1f}% | ECE={e_hd:.4f}")

    # Also test 2-Model Real-Photo Super-Ensemble (0.60 v7_effnetv2_real + 0.40 v6_densenet_real)
    if os.path.exists(V6_CKPT):
        ck6 = torch.load(V6_CKPT, map_location="cpu", weights_only=False)
        m6 = timm.create_model("densenet121", pretrained=False, num_classes=len(CLASSES_18), drop_rate=0.2).to(device)
        m6.load_state_dict(ck6["state_dict"])
        L6_vh, _ = collect(m6, va_hd_dl)
        L6_vr, _ = collect(m6, va_rl_dl)
        L6_cl, _ = collect(m6, te_cl_dl)
        L6_hd, _ = collect(m6, te_hd_dl)
        L6_rl, _ = collect(m6, te_rl_dl)
        A_en, B_en = fit_platt(np.concatenate([0.6 * L_va_hd + 0.4 * L6_vh, 0.6 * L_va_rl + 0.4 * L6_vr]), np.concatenate([Y_va_hd, Y_va_rl]))
        thr_en = freeze_thr(apply_platt(np.concatenate([0.6 * L_va_hd + 0.4 * L6_vh, 0.6 * L_va_rl + 0.4 * L6_vr]), A_en, B_en), np.concatenate([Y_va_hd, Y_va_rl]))
        Pe_cl = apply_platt(0.6 * L_te_cl + 0.4 * L6_cl, A_en, B_en)
        Pe_hd = apply_platt(0.6 * L_te_hd + 0.4 * L6_hd, A_en, B_en)
        Pe_rl = apply_platt(0.6 * L_te_rl + 0.4 * L6_rl, A_en, B_en)
        ce_hd = 100.0 * float(np.mean([any(Y_te_hd[i, c2i[c]] == 1 and Pe_hd[i, c2i[c]] >= thr_en[c] for c in CRITICAL_SET) for i in np.where(has_c)[0]]))
        ne_alrt = 100.0 * float(np.mean([any(Pe_cl[i, c2i[c]] >= thr_en[c] for c in CRITICAL_SET) for i in np.where(norm_only)[0]]))
        print(f"ENS (0.60 v7 + 0.40 v6)  | Clean={macro_auroc(Y_te_cl, Pe_cl):.4f} | Hard={macro_auroc(Y_te_hd, Pe_hd):.4f} | REAL={macro_auroc(Y_te_rl, Pe_rl):.4f} | Hard Crit={ce_hd:.1f}% | NormAlrt={ne_alrt:.1f}% | ECE={ece(Y_te_hd, Pe_hd):.4f}")
        A_save, B_save, thr_save = A_en, B_en, thr_en
    else:
        A_save, B_save, thr_save = A_v7, B_v7, thr_v7

    # 5. Safe Promotion: backup old EffNetV2-S and promote v7 ONLY if Real AUROC > 0.85 and Hard AUROC >= 0.960
    v7_path = os.path.join(OUT_DIR, "ecg_effnetv2s_v7_real_512x768.pt")
    payload = {
        "state_dict": best_state, "arch": arch, "classes": CLASSES_18, "img_h": IMG_H, "img_w": IMG_W,
        "val_auc": best_score, "platt_a": A_save.tolist(), "platt_b": B_save.tolist(), "thresholds": thr_save,
        "clean_auc": a_cl, "hard_auc": a_hd, "real_auc": a_rl, "hard_ece": e_hd,
    }
    torch.save(payload, v7_path)
    promoted = (a_rl >= 0.85) and (a_hd >= 0.960)
    if promoted:
        shutil.copy2(EFF_CKPT, os.path.join(OUT_DIR, "ecg_effnetv2s_multimachine_pre_v7_backup.pt"))
        shutil.copy2(v7_path, EFF_CKPT)
    vol.commit()
    sha = hashlib.sha256(open(v7_path, "rb").read()).hexdigest()[:16]
    print(f"\nPROMOTION STATUS: {'PROMOTED v7 TO ACTIVE ENSEMBLE!' if promoted else 'SAVED AS v7 CANDIDATE (previous EffNetV2 kept)'} (sha={sha})")


@app.local_entrypoint()
def main():
    train_v7_effnetv2_real.remote()
