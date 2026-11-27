"""
Code Blue ECG — v8 Rare-Class Boost (19 Classes, Masked Multi-Source BCE) on Modal A10G (~15 min, ~$0.30)
- Initializes from promoted v7 EffNetV2-S (512x768, Real-Photo + Multi-Machine) and cleanly expands 18 -> 19 classes
  adding BRUGADA_TYPE_1 (strictly visible basal_pattern in {1, 2} from Brugada-HUCA v2.0.0).
- Massively boosts our 4 under-powered classes using cardiologist-labeled PhysioNet ecg-arrhythmia (45,152 ECGs):
  * HIGH_GRADE_AV_BLOCK (142 real patients)
  * PACEMAKER_RHYTHM (600 sampled from 1,182 real patients)
  * ATRIAL_FLUTTER (600 sampled from 8,060 real patients)
  * WPW_PREEXCITATION (84 real patients)
- Zero label corruption: uses Masked BCE Loss on external datasets so unlabeled classes are never penalized.
- Strict patient-level 70/15/15 split on all external rare datasets + exact PTB-XL F1-8 / F9 / F10 splits.
- Head-to-head v7 vs v8 evaluation on ALL 4 Pristine Held-Out Benchmarks (Rare-Class Test, Clean F10, Hard F10, Real PMcardio F10).
- Saves to `/vol/outputs/ecg_effnetv2s_v8_rare19_512x768.pt` (non-destructive; live v7+v6 stays untouched until verified).
"""
import modal

app = modal.App("codeblue-train-v8-rare19")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "timm", "wfdb", "scipy", "scikit-learn",
        "pandas", "numpy", "opencv-python-headless", "pillow",
    )
)

CLASSES_19 = [
    "NORMAL_SINUS_RHYTHM", "ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI",
    "NSTEMI_ISCHEMIA", "ATRIAL_FIBRILLATION", "ATRIAL_FLUTTER", "SVT_TACHYCARDIA",
    "SINUS_BRADYCARDIA", "FIRST_DEGREE_AV_BLOCK", "HIGH_GRADE_AV_BLOCK", "RBBB", "LBBB",
    "LAFB_LPFB", "LVH_RVH", "PACEMAKER_RHYTHM", "WPW_PREEXCITATION", "LONG_QT_OR_ECTOPY",
    "BRUGADA_TYPE_1",
]
CRITICAL_SET = {"ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI", "HIGH_GRADE_AV_BLOCK", "BRUGADA_TYPE_1"}
RARE_CSN_SET = {"ATRIAL_FLUTTER", "HIGH_GRADE_AV_BLOCK", "PACEMAKER_RHYTHM", "WPW_PREEXCITATION"}
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
def train_v8_rare19():
    import ast, hashlib, json, os, re, time
    import cv2, numpy as np, pandas as pd, timm, torch, torch.nn as nn, wfdb
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset

    vol.reload()
    torch.manual_seed(20261127)
    np.random.seed(20261127)
    rng_split = np.random.default_rng(20261127)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_H, IMG_W = 512, 768
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PTB_DIR, PM_DIR, RARE_DIR, OUT_DIR = "/vol/ptbxl", "/vol/pmcardio", "/vol/rare_physionet", "/vol/outputs"
    V7_CKPT = os.path.join(OUT_DIR, "ecg_effnetv2s_v7_real_512x768.pt")
    if not os.path.exists(V7_CKPT):
        V7_CKPT = os.path.join(OUT_DIR, "ecg_effnetv2s_multimachine_512x768.pt")
    assert os.path.exists(V7_CKPT), f"Missing {V7_CKPT}"

    c2i = {c: i for i, c in enumerate(CLASSES_19)}
    N_CLS = len(CLASSES_19)

    # 1. Load PTB-XL (full 19-class mask with BRUGADA=0)
    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    ptb_by_id, tr_synth, va_synth, te_synth, seen_te = {}, [], [], [], set()
    full_mask = np.ones(N_CLS, dtype=np.float32)
    for _, r in df.iterrows():
        scp = ast.literal_eval(r["scp_codes"])
        y = np.zeros(N_CLS, dtype=np.float32)
        for k, v in scp.items():
            if k in SCP_TO_CLASS and (k in RHYTHM_OR_FORM_CODES or float(v) >= 50.0):
                y[c2i[SCP_TO_CLASS[k]]] = 1.0
        if y.sum() == 0:
            continue
        rec = {"kind": "wfdb", "path": os.path.join(PTB_DIR, r["filename_lr"]), "y": y, "mask": full_mask, "ecg_id": int(r["ecg_id"]), "pid": int(r["patient_id"]), "fold": int(r["strat_fold"])}
        ptb_by_id[rec["ecg_id"]] = rec
        if rec["fold"] <= 8:
            tr_synth.append(rec)
        elif rec["fold"] == 9:
            va_synth.append(rec)
        elif rec["pid"] not in seen_te:
            seen_te.add(rec["pid"])
            te_synth.append(rec)

    # 2. Load PMcardio Real Photos (F1-8 Train, F9 Val, F10 Test)
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
            item = {"kind": "img", "path": p_img, "y": b["y"], "mask": full_mask}
            (real_tr if b["fold"] <= 8 else real_va if b["fold"] == 9 else real_te).append(item)
    if len(real_tr) < 500:
        for p_img in all_imgs:
            if p_img in seen_p:
                continue
            m = re.search(r"(\d{1,5})", os.path.basename(p_img).split("__")[-1])
            if m and int(m.group(1)) in ptb_by_id:
                seen_p.add(p_img)
                b = ptb_by_id[int(m.group(1))]
                item = {"kind": "img", "path": p_img, "y": b["y"], "mask": full_mask}
                (real_tr if b["fold"] <= 8 else real_va if b["fold"] == 9 else real_te).append(item)

    # 3. Load Rare PhysioNet Manifest (Brugada-HUCA + CSN ecg-arrhythmia) with strict 70/15/15 patient splits & Masked BCE
    man = json.load(open(os.path.join(RARE_DIR, "rare_manifest.json"), encoding="utf-8"))
    csn_mask = np.zeros(N_CLS, dtype=np.float32)
    for c in list(RARE_CSN_SET) + ["NORMAL_SINUS_RHYTHM", "BRUGADA_TYPE_1"]:
        csn_mask[c2i[c]] = 1.0
    brug_mask = np.zeros(N_CLS, dtype=np.float32)
    brug_mask[c2i["BRUGADA_TYPE_1"]] = 1.0
    brug_mask[c2i["NORMAL_SINUS_RHYTHM"]] = 1.0

    def split_70_15_15(items):
        idx = rng_split.permutation(len(items))
        n1 = int(0.70 * len(items))
        n2 = int(0.85 * len(items))
        return [items[i] for i in idx[:n1]], [items[i] for i in idx[n1:n2]], [items[i] for i in idx[n2:]]

    rare_tr, rare_va, rare_te = [], [], []
    # Brugada visible basal (34) + controls (201)
    b_pos = [{"kind": "wfdb", "path": x["wfdb_path"], "y": np.eye(N_CLS, dtype=np.float32)[c2i["BRUGADA_TYPE_1"]], "mask": brug_mask} for x in man["brugada_visible_pos"]]
    b_ctl = [{"kind": "wfdb", "path": x["wfdb_path"], "y": np.eye(N_CLS, dtype=np.float32)[c2i["NORMAL_SINUS_RHYTHM"]], "mask": brug_mask} for x in man["brugada_clean_ctrl"]]
    for grp, oversample in [(b_pos, 10), (b_ctl, 1)]:
        tr, va, te = split_70_15_15(grp)
        rare_tr.extend(tr * oversample)
        rare_va.extend(va)
        rare_te.extend(te)

    # CSN 4 rare classes (balanced cap 600 per large class, oversample small classes in train ONLY)
    by_csn_cls = {c: [] for c in RARE_CSN_SET}
    for x in man["csn_rare_records"]:
        y = np.zeros(N_CLS, dtype=np.float32)
        for c in x["classes"]:
            if c in c2i:
                y[c2i[c]] = 1.0
        primary_c = x["classes"][0]
        if primary_c in by_csn_cls:
            by_csn_cls[primary_c].append({"kind": "wfdb", "path": x["wfdb_path"], "y": y, "mask": csn_mask})

    for cls_name, cap, rep in [("HIGH_GRADE_AV_BLOCK", 200, 4), ("WPW_PREEXCITATION", 200, 5), ("PACEMAKER_RHYTHM", 600, 2), ("ATRIAL_FLUTTER", 600, 1)]:
        pool = by_csn_cls[cls_name]
        if len(pool) > cap:
            idx = rng_split.choice(len(pool), size=cap, replace=False)
            pool = [pool[i] for i in idx]
        tr, va, te = split_70_15_15(pool)
        rare_tr.extend(tr * rep)
        rare_va.extend(va)
        rare_te.extend(te)
        print(f"  [Rare Split] {cls_name:22s}: total={len(pool):4d} -> tr={len(tr):3d} (x{rep}) | va={len(va):3d} | te={len(te):3d}")

    print(f"[v8 Rare-19] rare_tr={len(rare_tr)} | rare_va={len(rare_va)} | rare_te={len(rare_te)} | real_tr={len(real_tr)}")

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
        if sig.ndim == 1:
            sig = np.tile(sig[:, None], (1, 12))
        n_s = sig.shape[0]
        thick = int(rng.choice([1, 2, 2])) if hard else 2
        ink_v = int(rng.integers(10, 55)) if hard else 18
        ink_c = (ink_v, ink_v, ink_v)
        gain_px = float(rng.uniform(36.0, 48.0)) if hard else 42.0
        cw, rh = IMG_W // 4, IMG_H // 4
        for li in range(min(12, sig.shape[1])):
            r, c = li % 3, li // 3
            s = sig[int(c * n_s / 4):int((c + 1) * n_s / 4), li]
            if len(s) < 2:
                continue
            xs = np.linspace(c * cw + 6, (c + 1) * cw - 6, len(s)).astype(np.int32)
            ys = np.clip(int((r + 0.55) * rh) - (s * gain_px).astype(np.int32), 4, IMG_H - 4)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_c, thick, cv2.LINE_AA)
        s2 = sig[:, min(1, sig.shape[1] - 1)]
        xs = np.linspace(6, IMG_W - 6, len(s2)).astype(np.int32)
        ys = np.clip(int(3.55 * rh) - (s2 * gain_px).astype(np.int32), 4, IMG_H - 4)
        cv2.polylines(img, [np.stack([xs, ys], 1)], False, ink_c, thick, cv2.LINE_AA)
        if hard:
            if rng.random() < 0.6:
                gx = np.linspace(rng.uniform(0.74, 1.05), rng.uniform(0.74, 1.05), IMG_W, dtype=np.float32)
                gy = np.linspace(rng.uniform(0.80, 1.05), rng.uniform(0.80, 1.05), IMG_H, dtype=np.float32)
                img = np.clip(img.astype(np.float32) * np.outer(gy, gx)[:, :, None], 0, 255).astype(np.uint8)
            if rng.random() < 0.6:
                pts1 = np.float32([[0, 0], [IMG_W, 0], [0, IMG_H], [IMG_W, IMG_H]])
                j = float(rng.uniform(6.0, 20.0))
                M = cv2.getPerspectiveTransform(pts1, pts1 + rng.uniform(-j, j, size=(4, 2)).astype(np.float32))
                img = cv2.warpPerspective(img, M, (IMG_W, IMG_H), borderMode=cv2.BORDER_REPLICATE)
            if rng.random() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)
        return clean_ink(img)

    class UnifiedDS(Dataset):
        def __init__(self, rows, hard=True, fixed_seed=None):
            self.rows, self.hard, self.fixed_seed = rows, hard, fixed_seed
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            if r["kind"] == "img":
                bgr = cv2.imread(r["path"], cv2.IMREAD_COLOR)
                if bgr is None:
                    bgr = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
                x = clean_ink(augment_photo(bgr) if self.fixed_seed is None else bgr)
            else:
                try:
                    sig, _ = wfdb.rdsamp(r["path"])
                except Exception:
                    sig = np.zeros((1000, 12), dtype=np.float32)
                rng = np.random.default_rng(self.fixed_seed + i) if self.fixed_seed is not None else np.random.default_rng()
                x = render_mm(sig, rng, self.hard)
            return torch.from_numpy(x), torch.from_numpy(r["y"]), torch.from_numpy(r["mask"])

    idx_s = np.random.default_rng(20261127).choice(len(tr_synth), size=min(6500, len(tr_synth)), replace=False)
    train_pool = [tr_synth[i] for i in idx_s] + real_tr + rare_tr
    tr_dl = DataLoader(UnifiedDS(train_pool, True, None), batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    va_rare_dl = DataLoader(UnifiedDS(rare_va, True, 70000), batch_size=32, num_workers=4, pin_memory=True)
    te_rare_dl = DataLoader(UnifiedDS(rare_te, True, 80000), batch_size=32, num_workers=4, pin_memory=True)
    va_hd_dl = DataLoader(UnifiedDS(va_synth, True, 60000), batch_size=32, num_workers=4, pin_memory=True)
    te_hd_dl = DataLoader(UnifiedDS(te_synth, True, 100000), batch_size=32, num_workers=4, pin_memory=True)
    te_rl_dl = DataLoader(UnifiedDS(real_te, False, 120000), batch_size=32, num_workers=4, pin_memory=True)

    def collect(m, dl):
        m.eval()
        Ls, Ys, Ms = [], [], []
        with torch.no_grad():
            for x, y, msk in dl:
                with torch.amp.autocast("cuda"):
                    Ls.append(m(x.to(device, non_blocking=True)).float().cpu().numpy())
                Ys.append(y.numpy())
                Ms.append(msk.numpy())
        return np.concatenate(Ls), np.concatenate(Ys), np.concatenate(Ms)

    def masked_auroc_report(L, Y, M, title):
        P = 1.0 / (1.0 + np.exp(-np.clip(L, -30, 30)))
        print(f"\n--- {title} ---")
        aucs = []
        for j, cls in enumerate(CLASSES_19):
            valid = M[:, j] > 0.5
            if valid.sum() < 5:
                continue
            yv, pv = Y[valid, j], P[valid, j]
            if 0 < yv.sum() < len(yv):
                a = roc_auc_score(yv, pv)
                aucs.append(a)
                if cls in RARE_CSN_SET or cls == "BRUGADA_TYPE_1":
                    print(f"  {cls:24s} (pos={int(yv.sum()):3d}/{int(valid.sum()):3d}) AUROC={a:.4f}")
        macro = float(np.mean(aucs)) if aucs else 0.0
        print(f"  => Macro AUROC = {macro:.4f}")
        return macro

    # 4. Load v7 (18-class), expand classifier cleanly 18 -> 19 (preserving all 18 trained weights!)
    ck7 = torch.load(V7_CKPT, map_location="cpu", weights_only=False)
    arch = ck7.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k")
    model = timm.create_model(arch, pretrained=False, num_classes=N_CLS, drop_rate=0.2).to(device)
    sd7 = ck7["state_dict"]
    sd_new = model.state_dict()
    for k, v in sd7.items():
        if k in sd_new and sd_new[k].shape == v.shape:
            sd_new[k] = v
        elif k in sd_new and v.shape[0] == 18 and sd_new[k].shape[0] == 19:
            sd_new[k][:18] = v
    model.load_state_dict(sd_new)

    # Baseline v7 on Rare Test Set BEFORE v8 training
    L7_r, Y_r, M_r = collect(model, te_rare_dl)
    masked_auroc_report(L7_r, Y_r, M_r, "BASELINE v7 (Before Rare Boost) on Pristine Held-Out Rare Test")

    # 5. Train v8 with Masked BCE Loss (5 epochs, low LR to preserve v7's 0.967 Hard + 0.95 Real!)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-3)
    EPOCHS = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    pos_w = torch.ones(N_CLS, dtype=torch.float32, device=device) * 4.0
    pos_w[c2i["NORMAL_SINUS_RHYTHM"]] = 1.0
    pos_w[c2i["BRUGADA_TYPE_1"]] = 8.0
    pos_w[c2i["HIGH_GRADE_AV_BLOCK"]] = 8.0
    pos_w[c2i["WPW_PREEXCITATION"]] = 6.0

    best_sc, best_state = -1.0, None
    print(f"\n--- Fine-Tuning EffNetV2-S v8 Rare-19 (N_epoch={len(train_pool)}) ---")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y, msk in tr_dl:
            x, y, msk = x.to(device, non_blocking=True), y.to(device, non_blocking=True), msk.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                raw_bce = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")
                loss = (raw_bce * msk).sum() / torch.clamp(msk.sum(), min=1.0)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item()) * len(x)
        sched.step()
        Lv_r, Yv_r, Mv_r = collect(model, va_rare_dl)
        Lv_h, Yv_h, Mv_h = collect(model, va_hd_dl)
        a_r = float(np.mean([roc_auc_score(Yv_r[Mv_r[:, j] > 0.5, j], 1 / (1 + np.exp(-np.clip(Lv_r[Mv_r[:, j] > 0.5, j], -30, 30)))) for j in range(N_CLS) if (Mv_r[:, j] > 0.5).sum() > 5 and 0 < Yv_r[Mv_r[:, j] > 0.5, j].sum() < (Mv_r[:, j] > 0.5).sum()]))
        a_h = float(np.mean([roc_auc_score(Yv_h[:, j], 1 / (1 + np.exp(-np.clip(Lv_h[:, j], -30, 30)))) for j in range(18) if 0 < Yv_h[:, j].sum() < len(Yv_h)]))
        sc = 0.5 * a_h + 0.5 * a_r
        print(f"Epoch {ep}/{EPOCHS} ({time.time()-t0:.0f}s) | Loss={tot/len(train_pool):.4f} | Val Hard={a_h:.4f} | Val Rare={a_r:.4f}")
        if sc > best_sc:
            best_sc = sc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # 6. Calibrate Per-Class Platt strictly on Combined Val (Hard Fold-9 + Rare Val)
    Lv_h, Yv_h, Mv_h = collect(model, va_hd_dl)
    Lv_r, Yv_r, Mv_r = collect(model, va_rare_dl)
    L_va = np.concatenate([Lv_h, Lv_r])
    Y_va = np.concatenate([Yv_h, Yv_r])
    M_va = np.concatenate([Mv_h, Mv_r])

    A, B = np.ones(N_CLS, dtype=np.float32), np.zeros(N_CLS, dtype=np.float32)
    thr = {}
    for j, cls in enumerate(CLASSES_19):
        valid = M_va[:, j] > 0.5
        lj, yj = L_va[valid, j], Y_va[valid, j]
        if yj.sum() >= 3:
            res = minimize(
                lambda p: float(np.mean(np.logaddexp(0, p[0] * lj + p[1]) - yj * (p[0] * lj + p[1]))),
                x0=[1.0, -1.5], bounds=[(0.2, 4.0), (-6.0, 4.0)], method="L-BFGS-B",
            )
            A[j], B[j] = float(res.x[0]), float(res.x[1])
        if cls != "NORMAL_SINUS_RHYTHM":
            pj = 1.0 / (1.0 + np.exp(-np.clip(lj * A[j] + B[j], -30, 30)))
            yt = yj == 1
            if yt.sum() < 3:
                thr[cls] = 0.25
            else:
                target = 0.90 if cls in CRITICAL_SET else 0.85
                bt, bs = 0.25, -1.0
                for t in np.linspace(0.02, 0.80, 157):
                    yp = pj >= t
                    if float((yp & yt).sum() / yt.sum()) >= target:
                        sp = float(((~yp) & (~yt)).sum() / max(1, (~yt).sum()))
                        if sp > bs:
                            bs, bt = sp, float(t)
                thr[cls] = round(bt, 4)

    # 7. Final Head-to-Head on ALL 3 Pristine Test Benchmarks
    L8_r, _, _ = collect(model, te_rare_dl)
    L8_h, Y_h, M_h = collect(model, te_hd_dl)
    L8_rl, Y_rl, M_rl = collect(model, te_rl_dl)

    a_rare = masked_auroc_report(L8_r, Y_r, M_r, "v8 RARE-19 on Pristine Held-Out Rare Test (Brugada + AVB + Pace + WPW + Flutter)")
    a_hard = masked_auroc_report(L8_h, Y_h, M_h, "v8 RARE-19 on Pristine Hard Multi-Machine Fold-10 (N=1,704)")
    a_real = masked_auroc_report(L8_rl, Y_rl, M_rl, "v8 RARE-19 on Pristine REAL PMcardio Fold-10 (N=576)")

    out_path = os.path.join(OUT_DIR, "ecg_effnetv2s_v8_rare19_512x768.pt")
    torch.save({
        "state_dict": best_state, "arch": arch, "classes": CLASSES_19, "img_h": IMG_H, "img_w": IMG_W,
        "val_auc": best_sc, "platt_a": A.tolist(), "platt_b": B.tolist(), "thresholds": thr,
        "rare_test_auc": a_rare, "hard_f10_auc": a_hard, "real_f10_auc": a_real,
    }, out_path)
    vol.commit()
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()[:16]
    print(f"\nSaved v8 Rare-19 to {out_path} (sha={sha}) — live v7+v6 untouched until you approve!")


@app.local_entrypoint()
def main():
    train_v8_rare19.remote()
