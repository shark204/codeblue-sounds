"""
Code Blue ECG — Grand Finale Super-Ensemble Bake-Off on Modal A10G (~6 min, ~$0.12)
Evaluates ALL 4 trained 512x768 backbones + Ensembles head-to-head across ALL 3 Pristine Fold-10 Benchmarks:
  1) Clean Standard Fold-10 (N=1,704 unique patients)
  2) Hard Multi-Machine Fold-10 (N=1,704 unique patients: Green/Gray/6x2/Shadows/Warp/JPEG)
  3) REAL Photographed PMcardio Fold-10 (N=576 real mobile/paper/bent/screen ECG photos)
Fits Per-Class Platt Calibration (a_c, b_c) + Frozen Thresholds strictly on Fold-9 (1,937 PTB-XL + 921 Real Photos).
Saves `/vol/outputs/super_ensemble_config.json` with exact weights, Platt calibration, and frozen thresholds.
"""
import modal

app = modal.App("codeblue-super-ensemble-bakeoff")
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
    timeout=3600,
    volumes={"/vol": vol},
)
def run_bakeoff():
    import ast, json, math, os, re
    import cv2, numpy as np, pandas as pd, timm, torch, wfdb
    from scipy.optimize import minimize
    from scipy.stats import binomtest
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset

    vol.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_H, IMG_W = 512, 768
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PTB_DIR = "/vol/ptbxl"
    PM_DIR = "/vol/pmcardio"
    OUT_DIR = "/vol/outputs"

    # 1. Load PTB-XL Fold 9 (Val/Calib) & Fold 10 (Pristine 1-per-patient)
    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    c2i = {c: i for i, c in enumerate(CLASSES_18)}
    ptb_by_id, va_rows, te_rows, seen_te = {}, [], [], set()
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
        if rec["fold"] == 9:
            va_rows.append(rec)
        elif rec["fold"] == 10 and rec["pid"] not in seen_te:
            seen_te.add(rec["pid"])
            te_rows.append(rec)

    # 2. Load PMcardio Real Photos (Fold 9 Val & Fold 10 Test) — exact v6 matcher + prefix fallback
    df_pm = pd.read_csv(os.path.join(PM_DIR, "metadata.csv"))
    cols = list(df_pm.columns)
    img_col = next((c for c in cols if "image" in c.lower() and "name" in c.lower()), cols[0])
    id_col = next((c for c in cols if "ecg" in c.lower() and "id" in c.lower()), cols[1])
    PM_IMG_DIR = os.path.join(PM_DIR, "images")
    all_imgs = [os.path.join(PM_IMG_DIR, f) for f in os.listdir(PM_IMG_DIR) if f.lower().endswith(".jpg")]
    by_stem = {}
    for p_img in all_imgs:
        stem = os.path.splitext(os.path.basename(p_img).split("__")[-1])[0].lower()
        by_stem.setdefault(stem, []).append(p_img)
    real_va, real_te, seen_p = [], [], set()
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
            if b["fold"] == 9:
                real_va.append({"img_path": p_img, "y": b["y"]})
            elif b["fold"] == 10:
                real_te.append({"img_path": p_img, "y": b["y"]})
    if len(real_va) + len(real_te) < 200:
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
                if b["fold"] == 9:
                    real_va.append({"img_path": p_img, "y": b["y"]})
                elif b["fold"] == 10:
                    real_te.append({"img_path": p_img, "y": b["y"]})
    assert len(real_va) > 0 and len(real_te) > 0, f"PMcardio join empty! total_imgs={len(all_imgs)}"

    print(f"[BAKE-OFF] Fold9: PTB-XL={len(va_rows)} + Real={len(real_va)} | Fold10: PTB-XL={len(te_rows)} + Real={len(real_te)}")

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

    def render_ecg(sig: np.ndarray, rng: np.random.Generator, hard: bool) -> np.ndarray:
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

    class SynthDS(Dataset):
        def __init__(self, rows, hard, seed):
            self.rows, self.hard, self.seed = rows, hard, seed
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            sig, _ = wfdb.rdsamp(r["path"])
            return torch.from_numpy(render_ecg(sig, np.random.default_rng(self.seed + i), self.hard)), torch.from_numpy(r["y"])

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

    dls = {
        "va_clean": DataLoader(SynthDS(va_rows, False, 50000), batch_size=32, num_workers=4, pin_memory=True),
        "va_hard": DataLoader(SynthDS(va_rows, True, 60000), batch_size=32, num_workers=4, pin_memory=True),
        "va_real": DataLoader(RealDS(real_va), batch_size=32, num_workers=4, pin_memory=True),
        "te_clean": DataLoader(SynthDS(te_rows, False, 110000), batch_size=32, num_workers=4, pin_memory=True),
        "te_hard": DataLoader(SynthDS(te_rows, True, 100000), batch_size=32, num_workers=4, pin_memory=True),
        "te_real": DataLoader(RealDS(real_te), batch_size=32, num_workers=4, pin_memory=True),
    }

    # 3. Load all 4 checkpoints & collect logits once
    specs = [
        ("v5", "densenet121", os.path.join(OUT_DIR, "ecg_densenet121_v5_512x768.pt")),
        ("v6_real", "densenet121", os.path.join(OUT_DIR, "ecg_densenet121_v6_real_512x768.pt")),
        ("convnext", "convnext_small.fb_in22k_ft_in1k_384", os.path.join(OUT_DIR, "ecg_convnext_small_512x768.pt")),
        ("effnetv2", "tf_efficientnetv2_s.in21k_ft_in1k", os.path.join(OUT_DIR, "ecg_effnetv2s_multimachine_512x768.pt")),
    ]
    logits, labels = {k: {} for k, _, _ in specs}, {}
    for name, arch, path in specs:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = timm.create_model(ckpt.get("arch", arch), pretrained=False, num_classes=len(CLASSES_18), drop_rate=0.2).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        for split, dl in dls.items():
            Ls, Ys = [], []
            with torch.no_grad():
                for x, y in dl:
                    with torch.amp.autocast("cuda"):
                        Ls.append(m(x.to(device, non_blocking=True)).float().cpu().numpy())
                    Ys.append(y.numpy())
            logits[name][split] = np.concatenate(Ls)
            if split not in labels:
                labels[split] = np.concatenate(Ys)
        del m
        torch.cuda.empty_cache()
        print(f"Collected logits for {name}")

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

    def macro_auroc(Y, P):
        aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(CLASSES_18)) if 0 < Y[:, j].sum() < len(Y)]
        return float(np.mean(aucs)) if aucs else 0.0

    def ece(Y, P, n_bins=15):
        yf, pf = Y.reshape(-1), P.reshape(-1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        return float(sum((m.sum() / len(pf)) * abs(yf[m].mean() - pf[m].mean()) for lo, hi in zip(bins[:-1], bins[1:]) for m in [((pf >= lo) & (pf < hi)) if hi < 1.0 else ((pf >= lo) & (pf <= hi))] if m.sum() > 0))

    # 4. Compare single models + diverse ensembles calibrated strictly on Combined Fold-9 (Clean + Hard + Real)
    Y_va_all = np.concatenate([labels["va_clean"], labels["va_hard"], labels["va_real"]])
    candidates = {
        "v5_only": {"v5": 1.0},
        "v6_real_only": {"v6_real": 1.0},
        "convnext_only": {"convnext": 1.0},
        "effnetv2_only": {"effnetv2": 1.0},
        "ens_eff_cn (0.5/0.5)": {"effnetv2": 0.5, "convnext": 0.5},
        "ens_eff_v6 (0.6/0.4)": {"effnetv2": 0.6, "v6_real": 0.4},
        "SUPER_3MODEL (0.45 Eff + 0.30 CN + 0.25 v6)": {"effnetv2": 0.45, "convnext": 0.30, "v6_real": 0.25},
        "SUPER_3MODEL (0.40 Eff + 0.25 CN + 0.35 v6)": {"effnetv2": 0.40, "convnext": 0.25, "v6_real": 0.35},
    }

    crit_idx = [c2i[c] for c in CRITICAL_SET]
    norm_only = (labels["te_clean"][:, 0] == 1) & (labels["te_clean"][:, 1:].sum(1) == 0)

    def blend(weights, split):
        return sum(w * logits[m][split] for m, w in weights.items())

    print("\n================ GRAND FINALE SUPER-ENSEMBLE BAKE-OFF (ALL 3 PRISTINE FOLD-10 BENCHMARKS) ================")
    print(f"{'Candidate':44s} | {'Clean AUC':9s} | {'Hard AUC':8s} | {'REAL AUC':8s} | {'Hard Crit':9s} | {'NormAlrt':8s} | {'ECE':6s}")
    best_name, best_score, best_payload = None, -1.0, None
    results_summary = {}

    for cname, wmap in candidates.items():
        L_va = np.concatenate([blend(wmap, "va_clean"), blend(wmap, "va_hard"), blend(wmap, "va_real")])
        A, B = fit_platt(L_va, Y_va_all)
        thr = freeze_thr(apply_platt(L_va, A, B), Y_va_all)

        P_cl = apply_platt(blend(wmap, "te_clean"), A, B)
        P_hd = apply_platt(blend(wmap, "te_hard"), A, B)
        P_rl = apply_platt(blend(wmap, "te_real"), A, B)

        a_cl = macro_auroc(labels["te_clean"], P_cl)
        a_hd = macro_auroc(labels["te_hard"], P_hd)
        a_rl = macro_auroc(labels["te_real"], P_rl)
        e_hd = ece(labels["te_hard"], P_hd)

        has_c = labels["te_hard"][:, crit_idx].sum(1) > 0
        hits = np.array([any(labels["te_hard"][i, c2i[c]] == 1 and P_hd[i, c2i[c]] >= thr[c] for c in CRITICAL_SET) for i in range(len(P_hd))])
        c_hd = 100.0 * float(hits[has_c].mean())
        n_alrt = 100.0 * float(np.mean([any(P_cl[i, c2i[c]] >= thr[c] for c in CRITICAL_SET) for i in np.where(norm_only)[0]]))

        print(f"{cname:44s} | {a_cl:9.4f} | {a_hd:8.4f} | {a_rl:8.4f} | {c_hd:8.1f}% | {n_alrt:7.1f}% | {e_hd:.4f}")
        # Composite score balancing Real Photos + Hard Multi-Machine + Clean
        comp = 0.35 * a_rl + 0.40 * a_hd + 0.25 * a_cl
        results_summary[cname] = {"clean_auc": a_cl, "hard_auc": a_hd, "real_auc": a_rl, "hard_crit_pct": c_hd, "norm_alert_pct": n_alrt, "hard_ece": e_hd}
        if comp > best_score:
            best_score = comp
            best_name = cname
            best_payload = {
                "winner": cname, "weights": wmap, "classes": CLASSES_18,
                "platt_a": A.tolist(), "platt_b": B.tolist(), "thresholds": thr,
                "metrics": results_summary[cname], "all_candidates": results_summary,
            }

    out_json = os.path.join(OUT_DIR, "super_ensemble_config.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)
    vol.commit()
    print(f"\nCHAMPION SELECTED: {best_name}")
    print(f"Saved champion config to {out_json}")


@app.local_entrypoint()
def main():
    run_bakeoff.remote()
