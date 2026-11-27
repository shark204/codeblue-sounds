"""
Code Blue ECG — Parallel Experiment A+B on Modal A10G GPU:
1) Train ConvNeXt-Small natively at 512x768 (7x7 receptive field for multi-lead ST & rhythm context)
   on PTB-XL Folds 1-8 (same exact split as DenseNet121 v5).
2) Fit Per-Class Platt Calibration (a_c, b_c) strictly on Fold 9 (zero leakage) and measure Held-Out ECE
   on Pristine Fold-10 (N=1,705 unique patients) targeting ECE < 0.05.
3) Head-to-head Paired McNemar on Pristine Fold-10:
   - DenseNet121 v5 alone (baseline)
   - ConvNeXt-Small alone
   - Diverse Ensemble (DenseNet121 v5 + ConvNeXt-Small)
4) Saves ConvNeXt-Small + Ensemble calibration/thresholds to `/vol/outputs/`.
"""
import modal

app = modal.App("codeblue-exp-convnext-ensemble")
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
def run_convnext_and_ensemble():
    import ast, hashlib, json, math, os, time
    import cv2, numpy as np, pandas as pd, timm, torch, torch.nn as nn, wfdb
    from scipy.optimize import minimize
    from scipy.stats import binomtest
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset

    vol.reload()
    torch.manual_seed(20261126)
    np.random.seed(20261126)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_H, IMG_W = 512, 768
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PTB_DIR = "/vol/ptbxl"
    OUT_DIR = "/vol/outputs"
    V5_CKPT = os.path.join(OUT_DIR, "ecg_densenet121_v5_512x768.pt")
    os.makedirs(OUT_DIR, exist_ok=True)
    assert os.path.exists(V5_CKPT), f"Missing v5 at {V5_CKPT}"

    # 1. Load exact PTB-XL Folds 1-8 (Train), Fold 9 (Val/Calib), Fold 10 (Pristine Test, 1-per-patient)
    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    c2i = {c: i for i, c in enumerate(CLASSES_18)}
    tr_rows, va_rows, te_rows = [], [], []
    seen_te_pts = set()
    for _, r in df.iterrows():
        scp = ast.literal_eval(r["scp_codes"])
        y = np.zeros(len(CLASSES_18), dtype=np.float32)
        for k, v in scp.items():
            if k in SCP_TO_CLASS and (k in RHYTHM_OR_FORM_CODES or float(v) >= 50.0):
                y[c2i[SCP_TO_CLASS[k]]] = 1.0
        if y.sum() == 0:
            continue
        rec = {"path": os.path.join(PTB_DIR, r["filename_lr"]), "y": y, "pid": int(r["patient_id"])}
        fold = int(r["strat_fold"])
        if fold <= 8:
            tr_rows.append(rec)
        elif fold == 9:
            va_rows.append(rec)
        else:
            if rec["pid"] not in seen_te_pts:
                seen_te_pts.add(rec["pid"])
                te_rows.append(rec)

    print(f"[ConvNeXt-Small 512x768] Train={len(tr_rows)} | Fold9(Val)={len(va_rows)} | Fold10(Pristine 1-per-pt)={len(te_rows)}")

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

    def render_ecg(sig: np.ndarray, augment: bool) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), (242, 232, 252), dtype=np.uint8)
        for x in range(0, IMG_W, 12):
            cv2.line(img, (x, 0), (x, IMG_H), (210, 185, 240) if x % 60 else (180, 145, 225), 1)
        for y in range(0, IMG_H, 12):
            cv2.line(img, (0, y), (IMG_W, y), (210, 185, 240) if y % 60 else (180, 145, 225), 1)
        sig = np.nan_to_num(sig, 0.0)
        n_s = sig.shape[0]
        cw, rh = IMG_W // 4, IMG_H // 4
        for li in range(min(12, sig.shape[1])):
            r, c = li % 3, li // 3
            s = sig[int(c * n_s / 4):int((c + 1) * n_s / 4), li]
            if len(s) < 2:
                continue
            xs = np.linspace(c * cw + 6, (c + 1) * cw - 6, len(s)).astype(np.int32)
            ys = np.clip(int((r + 0.55) * rh) - (s * 42.0).astype(np.int32), 4, IMG_H - 4)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, (18, 18, 18), 2, cv2.LINE_AA)
        s2 = sig[:, 1]
        xs = np.linspace(6, IMG_W - 6, len(s2)).astype(np.int32)
        ys = np.clip(int(3.55 * rh) - (s2 * 42.0).astype(np.int32), 4, IMG_H - 4)
        cv2.polylines(img, [np.stack([xs, ys], 1)], False, (18, 18, 18), 2, cv2.LINE_AA)
        if augment:
            if np.random.rand() < 0.6:
                M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), float(np.random.uniform(-2.5, 2.5)), float(np.random.uniform(0.95, 1.04)))
                img = cv2.warpAffine(img, M, (IMG_W, IMG_H), borderMode=cv2.BORDER_REPLICATE)
            if np.random.rand() < 0.5:
                img = np.clip(img.astype(np.float32) * float(np.random.uniform(0.8, 1.2)) + float(np.random.uniform(-20, 20)), 0, 255).astype(np.uint8)
            if np.random.rand() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)
        return clean_ink(img)

    class EcgDS(Dataset):
        def __init__(self, rows, augment=False):
            self.rows, self.augment = rows, augment
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            sig, _ = wfdb.rdsamp(r["path"])
            return torch.from_numpy(render_ecg(sig, self.augment)), torch.from_numpy(r["y"])

    tr_dl = DataLoader(EcgDS(tr_rows, True), batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    va_dl = DataLoader(EcgDS(va_rows, False), batch_size=24, shuffle=False, num_workers=4, pin_memory=True)
    te_dl = DataLoader(EcgDS(te_rows, False), batch_size=24, shuffle=False, num_workers=4, pin_memory=True)

    def collect_logits(m, dl):
        m.eval()
        Ls, Ys = [], []
        with torch.no_grad():
            for x, y in dl:
                with torch.amp.autocast("cuda"):
                    Ls.append(m(x.to(device, non_blocking=True)).float().cpu().numpy())
                Ys.append(y.numpy())
        return np.concatenate(Ls), np.concatenate(Ys)

    def macro_auroc(Y, P):
        return float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(len(CLASSES_18)) if 0 < Y[:, j].sum() < len(Y)]))

    def compute_ece(Y, P, n_bins=15):
        """Multi-label expected calibration error across all (sample, class) predictions."""
        y_flat = Y.reshape(-1)
        p_flat = P.reshape(-1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_flat >= lo) & (p_flat < hi) if hi < 1.0 else (p_flat >= lo) & (p_flat <= hi)
            if mask.sum() > 0:
                ece += (mask.sum() / len(p_flat)) * abs(y_flat[mask].mean() - p_flat[mask].mean())
        return float(ece)

    def fit_per_class_platt(L_va, Y_va):
        """Fit per-class Platt scaling sigmoid(a_c * logit + b_c) strictly on Fold-9 (zero leakage)."""
        A, B = np.ones(len(CLASSES_18), dtype=np.float32), np.zeros(len(CLASSES_18), dtype=np.float32)
        for j in range(len(CLASSES_18)):
            lj, yj = L_va[:, j], Y_va[:, j]
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

    def freeze_thresholds(P_va, Y_va):
        thr = {}
        for j, cls in enumerate(CLASSES_18):
            if cls == "NORMAL_SINUS_RHYTHM":
                continue
            yt = Y_va[:, j] == 1
            if yt.sum() < 3:
                thr[cls] = 0.25
                continue
            target = 0.90 if cls in CRITICAL_SET else 0.85
            best_t, best_s = 0.25, -1.0
            for t in np.linspace(0.02, 0.80, 157):
                yp = P_va[:, j] >= t
                sens = float((yp & yt).sum() / yt.sum())
                spec = float(((~yp) & (~yt)).sum() / max(1, (~yt).sum()))
                if sens >= target and spec > best_s:
                    best_s, best_t = spec, float(t)
            thr[cls] = round(best_t, 4)
        return thr

    def wilson(k, n):
        if n == 0:
            return 0.0, 0.0, 0.0
        z = 1.95996398454
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / d
        return p, max(0.0, c - h), min(1.0, c + h)

    # 2. Train ConvNeXt-Small natively at 512x768 (7 epochs on A10G)
    model_cn = timm.create_model("convnext_small.fb_in22k_ft_in1k_384", pretrained=True, num_classes=len(CLASSES_18), drop_rate=0.2).to(device)
    y_tr = np.stack([r["y"] for r in tr_rows])
    pos_w = torch.tensor(np.clip((len(y_tr) - y_tr.sum(0)) / np.maximum(y_tr.sum(0), 1.0), 1.0, 20.0), dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model_cn.parameters(), lr=1.5e-4, weight_decay=1e-2)
    EPOCHS = 7
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_auc, best_state = -1.0, None
    print("\n--- Training ConvNeXt-Small Natively at 512x768 ---")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model_cn.train()
        tot_loss = 0.0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(model_cn(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot_loss += float(loss.item()) * len(x)
        sched.step()
        L_va, Y_va = collect_logits(model_cn, va_dl)
        auc = macro_auroc(Y_va, 1.0 / (1.0 + np.exp(-np.clip(L_va, -30, 30))))
        print(f"Epoch {ep}/{EPOCHS} ({time.time()-t0:.0f}s) | Loss={tot_loss/len(tr_rows):.4f} | Fold9 Macro AUROC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model_cn.state_dict().items()}

    model_cn.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # 3. Collect Fold-9 & Fold-10 logits for BOTH ConvNeXt-Small AND DenseNet121 v5
    L_cn_va, Y_va = collect_logits(model_cn, va_dl)
    L_cn_te, Y_te = collect_logits(model_cn, te_dl)

    ckpt_v5 = torch.load(V5_CKPT, map_location="cpu", weights_only=False)
    model_v5 = timm.create_model("densenet121", pretrained=False, num_classes=len(CLASSES_18), drop_rate=0.2).to(device)
    model_v5.load_state_dict(ckpt_v5["state_dict"])
    L_v5_va, _ = collect_logits(model_v5, va_dl)
    L_v5_te, _ = collect_logits(model_v5, te_dl)

    # Optimal Fold-9 ensemble weight w in {0.3, 0.4, 0.5, 0.6, 0.7}
    best_w, best_ens_va_auc = 0.5, -1.0
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        L_try = w * L_cn_va + (1.0 - w) * L_v5_va
        a_try = macro_auroc(Y_va, 1.0 / (1.0 + np.exp(-np.clip(L_try, -30, 30))))
        if a_try > best_ens_va_auc:
            best_ens_va_auc, best_w = a_try, w
    L_ens_va = best_w * L_cn_va + (1.0 - best_w) * L_v5_va
    L_ens_te = best_w * L_cn_te + (1.0 - best_w) * L_v5_te

    # 4. Per-Class Platt Calibration strictly on Fold 9 + Freeze Thresholds
    Av5, Bv5 = fit_per_class_platt(L_v5_va, Y_va)
    Acn, Bcn = fit_per_class_platt(L_cn_va, Y_va)
    Aen, Ben = fit_per_class_platt(L_ens_va, Y_va)

    Pv5_va, Pv5_te = apply_platt(L_v5_va, Av5, Bv5), apply_platt(L_v5_te, Av5, Bv5)
    Pcn_va, Pcn_te = apply_platt(L_cn_va, Acn, Bcn), apply_platt(L_cn_te, Acn, Bcn)
    Pen_va, Pen_te = apply_platt(L_ens_va, Aen, Ben), apply_platt(L_ens_te, Aen, Ben)

    thr_v5 = freeze_thresholds(Pv5_va, Y_va)
    thr_cn = freeze_thresholds(Pcn_va, Y_va)
    thr_en = freeze_thresholds(Pen_va, Y_va)

    # 5. Evaluate ALL THREE on Pristine Fold-10 (N=1,705 unique patients)
    crit_idx = [c2i[c] for c in CRITICAL_SET]
    norm_only = (Y_te[:, 0] == 1) & (Y_te[:, 1:].sum(1) == 0)

    def summarize(name, P, thr):
        auc = macro_auroc(Y_te, P)
        ece = compute_ece(Y_te, P)
        has_c = Y_te[:, crit_idx].sum(1) > 0
        hits = np.array([any(Y_te[i, c2i[c]] == 1 and P[i, c2i[c]] >= thr[c] for c in CRITICAL_SET) for i in range(len(Y_te))])
        cap_k, cap_n = int(hits[has_c].sum()), int(has_c.sum())
        cp, cl, cu = wilson(cap_k, cap_n)
        norm_alerts = np.array([sum(P[i, c2i[c]] >= thr[c] for c in CRITICAL_SET) for i in np.where(norm_only)[0]])
        ab_pct = 100.0 * float((norm_alerts >= 1).mean())
        ab_per100 = 100.0 * float(norm_alerts.mean())
        print(f"{name:28s} | Macro AUROC={auc:.4f} | Held-Out ECE={ece:.4f} | Crit Capture={cp*100:.1f}% [{cl*100:.1f}-{cu*100:.1f}] ({cap_k}/{cap_n}) | Norm Alert={ab_pct:.1f}% ({ab_per100:.1f}/100)")
        return auc, ece, has_c, hits

    print(f"\n================ PRISTINE FOLD-10 HEAD-TO-HEAD (N={len(Y_te)} unique patients, best_w_convnext={best_w}) ================")
    auc_v5, ece_v5, has_c, hits_v5 = summarize("DenseNet121 v5 (PerClass)", Pv5_te, thr_v5)
    auc_cn, ece_cn, _, hits_cn = summarize("ConvNeXt-Small (PerClass)", Pcn_te, thr_cn)
    auc_en, ece_en, _, hits_en = summarize("ENSEMBLE (v5 + ConvNeXt)", Pen_te, thr_en)

    for label, h_new in [("ConvNeXt vs v5", hits_cn), ("ENSEMBLE vs v5", hits_en)]:
        o, n = hits_v5[has_c], h_new[has_c]
        gained, lost = int(((~o) & n).sum()), int((o & (~n)).sum())
        pval = float(binomtest(gained, gained + lost, 0.5).pvalue) if (gained + lost) > 0 else 1.0
        print(f"Paired McNemar Crit ({label:14s}): gained={gained}, lost={lost}, exact p={pval:.4f}")

    print("\n--- Per-Class Fold-10 AUROC: v5 -> ConvNeXt -> ENSEMBLE ---")
    for j, cls in enumerate(CLASSES_18):
        if cls == "NORMAL_SINUS_RHYTHM":
            continue
        yt = Y_te[:, j] == 1
        a5 = roc_auc_score(yt, Pv5_te[:, j])
        ac = roc_auc_score(yt, Pcn_te[:, j])
        ae = roc_auc_score(yt, Pen_te[:, j])
        yp = Pen_te[:, j] >= thr_en[cls]
        sens = float((yp & yt).sum() / yt.sum())
        spec = float(((~yp) & (~yt)).sum() / (~yt).sum())
        ppv = float((yp & yt).sum() / max(1, yp.sum()))
        print(f"{cls:24s} (n={int(yt.sum()):3d}) AUROC {a5:.4f} | {ac:.4f} | {ae:.4f} (ENS) | Ens Sens {sens*100:5.1f}% Spec {spec*100:5.1f}% PPV {ppv*100:5.1f}%")

    # 6. Save ConvNeXt-Small + Ensemble Platt Calibration & Frozen Thresholds to Modal Volume
    cn_path = os.path.join(OUT_DIR, "ecg_convnext_small_512x768.pt")
    torch.save({
        "state_dict": best_state, "arch": "convnext_small.fb_in22k_ft_in1k_384",
        "classes": CLASSES_18, "img_h": IMG_H, "img_w": IMG_W, "val_auc": best_auc,
        "platt_a": Acn.tolist(), "platt_b": Bcn.tolist(),
    }, cn_path)
    sha_cn = hashlib.sha256(open(cn_path, "rb").read()).hexdigest()[:16]
    with open(os.path.join(OUT_DIR, "ensemble_v5_convnext_platt.json"), "w", encoding="utf-8") as f:
        json.dump({
            "convnext_sha256": sha_cn, "w_convnext": best_w, "w_densenet_v5": round(1.0 - best_w, 2),
            "classes": CLASSES_18,
            "v5_platt_a": Av5.tolist(), "v5_platt_b": Bv5.tolist(), "v5_thresholds": thr_v5, "v5_ece": ece_v5, "v5_auroc": auc_v5,
            "cn_platt_a": Acn.tolist(), "cn_platt_b": Bcn.tolist(), "cn_thresholds": thr_cn, "cn_ece": ece_cn, "cn_auroc": auc_cn,
            "ens_platt_a": Aen.tolist(), "ens_platt_b": Ben.tolist(), "ens_thresholds": thr_en, "ens_ece": ece_en, "ens_auroc": auc_en,
        }, f, indent=2)
    vol.commit()
    print(f"\nSaved to /vol/outputs/: ecg_convnext_small_512x768.pt (sha={sha_cn}) + ensemble_v5_convnext_platt.json")


@app.local_entrypoint()
def main():
    run_convnext_and_ensemble.remote()
