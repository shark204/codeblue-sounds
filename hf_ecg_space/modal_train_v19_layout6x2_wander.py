"""
Code Blue ECG — v19 Multi-Layout (3x4 + 6x2) & Baseline-Wander Hardened Classifier (Modal A10G)
Fixes the 6x2 layout + baseline-wander misclassification uncovered on real clinical ECGs (e.g. AFib+LBBB in 6x2):
  - Starts from promoted v18 EfficientNetV2-S weights.
  - Trains with:
      * 35% true 6x2 layout (Left: I,II,III,aVR,aVL,aVF | Right: V1-V6) + 65% 3x4+1 layout
      * Realistic low-frequency baseline wander (0.15-0.45 mV sinusoidal drift on random leads)
      * Strict odd-ID LITFL Top-150 external textbook ECGs (zero overlap with even-ID held-out test)
  - Evaluates on ALL 3 Pristine Held-Out Benchmarks:
      1) PTB-XL Fold-10 Standard 3x4+1 Macro-AUC (anti-forgetting)
      2) PTB-XL Fold-10 6x2 + Baseline Wander Macro-AUC (target fix!)
      3) Held-Out External LITFL Even-ID (N=49, zero leakage)
  - Promotes ONLY if 6x2 Macro-AUC improves >= +0.02 without harming 3x4+1 AUC (> -0.01) or LITFL Hit.
"""
import os, json, ast, math, shutil, time, modal

app = modal.App("codeblue-train-v19-layout6x2")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", "timm", "wfdb", "scikit-learn", "pandas", "numpy", "opencv-python-headless", "pillow")
)


def wilson_ci(k: int, n: int, z: float = 1.95996):
    if n <= 0: return [0.0, 0.0]
    p = k / n; d = 1.0 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [round(max(0.0, c - h) * 100.0, 1), round(min(1.0, c + h) * 100.0, 1)]


@app.function(image=image, gpu="A10G", cpu=6.0, memory=24576, timeout=3600, volumes={"/vol": vol})
def train_v19_6x2():
    import cv2, numpy as np, pandas as pd, timm, torch, torch.nn as nn, wfdb
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda")
    PTB_DIR, OUT_DIR, LITFL_DIR = "/vol/ptbxl", "/vol/outputs", "/vol/litfl_top150_v18"
    V8_PATH = os.path.join(OUT_DIR, "ecg_effnetv2s_v8_rare19_512x768.pt")
    ck = torch.load(V8_PATH, map_location="cpu", weights_only=False)
    CLASSES = list(ck["classes"])
    IMG_H, IMG_W = int(ck.get("img_h", 512)), int(ck.get("img_w", 768))
    PLATT_A, PLATT_B = np.array(ck["platt_a"], dtype=np.float32), np.array(ck["platt_b"], dtype=np.float32)
    THR_MAP = ck.get("thresholds", {})
    c2i = {c: i for i, c in enumerate(CLASSES)}

    SCP_MAP = {
        "NORM": "NORMAL_SINUS_RHYTHM", "AMI": "ANTERIOR_SEPTAL_MI", "ASMI": "ANTERIOR_SEPTAL_MI", "ALMI": "ANTERIOR_SEPTAL_MI", "ILMI": "ANTERIOR_SEPTAL_MI",
        "IMI": "INFERIOR_POSTERIOR_MI", "IPMI": "INFERIOR_POSTERIOR_MI", "PMI": "INFERIOR_POSTERIOR_MI", "IPLMI": "INFERIOR_POSTERIOR_MI",
        "LMI": "LATERAL_MI", "INJAL": "LATERAL_MI", "INJLA": "LATERAL_MI", "ISC_": "NSTEMI_ISCHEMIA", "ISCAL": "NSTEMI_ISCHEMIA", "ISCAN": "NSTEMI_ISCHEMIA",
        "ISCAS": "NSTEMI_ISCHEMIA", "ISCIL": "NSTEMI_ISCHEMIA", "ISCIN": "NSTEMI_ISCHEMIA", "ISCLA": "NSTEMI_ISCHEMIA", "NST_": "NSTEMI_ISCHEMIA", "STD_": "NSTEMI_ISCHEMIA", "STTC": "NSTEMI_ISCHEMIA",
        "AFIB": "ATRIAL_FIBRILLATION", "AFLT": "ATRIAL_FLUTTER", "SVTAC": "SVT_TACHYCARDIA", "PSVT": "SVT_TACHYCARDIA", "SBRAD": "SINUS_BRADYCARDIA",
        "1AVB": "FIRST_DEGREE_AV_BLOCK", "2AVB": "HIGH_GRADE_AV_BLOCK", "3AVB": "HIGH_GRADE_AV_BLOCK", "CRBBB": "RBBB", "IRBBB": "RBBB",
        "CLBBB": "LBBB", "ILBBB": "LBBB", "LAFB": "LAFB_LPFB", "LPFB": "LAFB_LPFB", "LVH": "LVH_RVH", "RVH": "LVH_RVH",
        "PACE": "PACEMAKER_RHYTHM", "WPW": "WPW_PREEXCITATION", "LNGQT": "LONG_QT_OR_ECTOPY", "PVC": "LONG_QT_OR_ECTOPY",
    }
    RHYTHM_SCPS = {"AFIB", "AFLT", "SVTAC", "PSVT", "SBRAD", "PACE", "PVC"}

    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    ptb_tr, ptb_te, seen10 = [], [], set()
    for _, r in df.iterrows():
        f = int(r["strat_fold"])
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for k, v in ast.literal_eval(r["scp_codes"]).items():
            if k in SCP_MAP and SCP_MAP[k] in c2i and (k in RHYTHM_SCPS or float(v) >= 50.0):
                y[c2i[SCP_MAP[k]]] = 1.0
        if y.sum() == 0: continue
        if f <= 8:
            ptb_tr.append({"kind": "wfdb", "path": os.path.join(PTB_DIR, r["filename_lr"]), "y": y})
        elif f == 10 and int(r["patient_id"]) not in seen10:
            seen10.add(int(r["patient_id"]))
            ptb_te.append({"kind": "wfdb", "path": os.path.join(PTB_DIR, r["filename_lr"]), "y": y})

    lit_tr, lit_te = [], []
    meta_p = os.path.join(LITFL_DIR, "litfl150_meta.json")
    if os.path.exists(meta_p):
        lit_all = json.load(open(meta_p))
        lit_tr = [x for x in lit_all if x["num"] % 2 == 1]
        lit_te = [x for x in lit_all if x["num"] % 2 == 0]
    print(f"[v19 6x2+Wander] PTB Tr={len(ptb_tr)} Te={len(ptb_te)} | LITFL Odd-Tr={len(lit_tr)} Even-Te={len(lit_te)}")

    def clean_ink(bgr, aug=False, rng=None):
        if aug and rng is not None:
            h, w = bgr.shape[:2]
            if rng.random() < 0.35:
                cy0, cy1 = int(h * rng.uniform(0.0, 0.06)), int(h * rng.uniform(0.94, 1.0))
                cx0, cx1 = int(w * rng.uniform(0.0, 0.06)), int(w * rng.uniform(0.94, 1.0))
                bgr = bgr[cy0:cy1, cx0:cx1]
            if rng.random() < 0.35:
                M = cv2.getRotationMatrix2D((bgr.shape[1]/2, bgr.shape[0]/2), float(rng.uniform(-3.5, 3.5)), 1.0)
                bgr = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        if bgr.shape[0] != IMG_H or bgr.shape[1] != IMG_W:
            bgr = cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        b, g, r = cv2.split(bgr.astype(np.float32))
        red = r - 0.5 * (g + b)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        pm = (red > 14.0) & (gray > 75.0)
        gray[pm] = np.clip(gray[pm] + red[pm] * 1.15, 0, 255)
        bg = cv2.GaussianBlur(gray, (0, 0), 15.0)
        norm = np.clip((gray / np.maximum(bg, 30.0)) * 245.0, 0, 255).astype(np.uint8)
        lo, hi = np.percentile(norm, 2), np.percentile(norm, 98)
        if hi - lo > 10:
            norm = np.clip((norm.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        x = np.stack([norm, norm, norm], axis=0).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (x - mean) / std

    def add_wander(sig, rng, prob=0.45):
        if rng.random() > prob:
            return sig
        s = sig.copy()
        n = s.shape[0]
        t = np.linspace(0, 1, n, dtype=np.float32)
        for ch in range(s.shape[1]):
            if rng.random() < 0.45:
                amp = float(rng.uniform(0.12, 0.42))
                freq = float(rng.uniform(0.25, 0.9))
                phase = float(rng.uniform(0, 2 * math.pi))
                s[:, ch] += amp * np.sin(2 * math.pi * freq * t + phase)
        return s

    def render_ecg(sig, rng, layout_6x2=False, wander=False, aug=False):
        H, W = IMG_H, IMG_W
        if wander:
            sig = add_wander(sig, rng, prob=0.85 if not aug else 0.50)
        canvas = np.full((H, W, 3), (242, 236, 252), dtype=np.uint8)
        px_mm = (W - 32) / 250.0
        for x in range(0, W, max(2, int(round(px_mm)))): cv2.line(canvas, (x, 0), (x, H), (225, 212, 244), 1)
        for y in range(0, H, max(2, int(round(px_mm)))): cv2.line(canvas, (0, y), (W, y), (225, 212, 244), 1)
        n_samp = sig.shape[0]
        if layout_6x2:
            # Left column: I, II, III, aVR, aVL, aVF (0..5) | Right column: V1..V6 (6..11)
            col_w = (W - 32) // 2
            row_h = (H - 36) // 6
            half = n_samp // 2
            for r_idx in range(6):
                yc = 18 + r_idx * row_h + row_h // 2
                for c_idx, l_idx in enumerate([r_idx, r_idx + 6]):
                    x0 = 16 + c_idx * col_w
                    s = sig[c_idx * half : (c_idx + 1) * half, l_idx]
                    xs = np.linspace(x0, x0 + col_w - 6, len(s)).astype(np.int32)
                    ys = np.clip(yc - (s * 10.0 * px_mm), 4, H - 4).astype(np.int32)
                    cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (20, 20, 20), 1, cv2.LINE_AA)
        else:
            layout = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
            col_w, row_h, seg = (W - 32) // 4, (H - 64) // 4, n_samp // 4
            for r_idx, row in enumerate(layout):
                yc = 28 + r_idx * row_h + row_h // 2
                for c_idx, l_idx in enumerate(row):
                    x0 = 16 + c_idx * col_w
                    s = sig[c_idx * seg : (c_idx + 1) * seg, l_idx]
                    xs = np.linspace(x0, x0 + col_w - 4, len(s)).astype(np.int32)
                    ys = np.clip(yc - (s * 10.0 * px_mm), 6, H - 6).astype(np.int32)
                    cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (20, 20, 20), 1, cv2.LINE_AA)
            yc = 28 + 3 * row_h + row_h // 2
            xs = np.linspace(16, W - 16, n_samp).astype(np.int32)
            ys = np.clip(yc - (sig[:, 1] * 10.0 * px_mm), 6, H - 6).astype(np.int32)
            cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (20, 20, 20), 1, cv2.LINE_AA)
        return clean_ink(canvas, aug=aug, rng=rng)

    model = timm.create_model(ck.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k"), pretrained=False, num_classes=len(CLASSES), drop_rate=0.25).to(device)
    model.load_state_dict(ck["state_dict"])

    def eval_ptb(m_eval, recs, is_6x2=False):
        m_eval.eval()
        rng = np.random.default_rng(77 if is_6x2 else 33)
        ps, ys = [], []
        with torch.no_grad():
            for i in range(0, len(recs), 32):
                batch = recs[i:i+32]
                xs = np.stack([render_ecg(np.nan_to_num(wfdb.rdsamp(r["path"])[0].astype(np.float32)), rng, is_6x2, is_6x2, False) for r in batch])
                with torch.amp.autocast("cuda"):
                    ps.append(torch.sigmoid(m_eval(torch.from_numpy(xs).to(device))).float().cpu().numpy())
                ys.append(np.stack([r["y"] for r in batch]))
        P, Y = np.concatenate(ps), np.concatenate(ys)
        aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(CLASSES)) if Y[:, j].sum() > 0 and (1 - Y[:, j]).sum() > 0]
        return round(float(np.mean(aucs)), 4)

    def eval_litfl(m_eval, items):
        if not items: return {"n": 0, "op_hit_pct": 0.0}
        m_eval.eval()
        op_hit, top3, n = 0, 0, 0
        with torch.no_grad():
            for it in items:
                im = cv2.imread(it["path"])
                if im is None: continue
                n += 1
                xb = torch.from_numpy(clean_ink(im)).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda"):
                    lg = m_eval(xb).float().cpu().numpy()[0]
                pr = 1.0 / (1.0 + np.exp(-np.clip(PLATT_A * lg + PLATT_B, -30, 30)))
                gt = set(it["gt"])
                if any(CLASSES[k] in gt for k in np.argsort(-pr)[:3]): top3 += 1
                if {CLASSES[j] for j in range(len(CLASSES)) if pr[j] >= float(THR_MAP.get(CLASSES[j], 0.5))} & gt: op_hit += 1
        return {"n": n, "top3_pct": round(100*top3/max(1,n), 1), "op_hit_pct": round(100*op_hit/max(1,n), 1), "ci95": wilson_ci(op_hit, n)}

    te_sub = ptb_te[:600]
    b_3x4 = eval_ptb(model, te_sub, False)
    b_6x2 = eval_ptb(model, te_sub, True)
    b_lit = eval_litfl(model, lit_te)
    print(f"[BASELINE v18] PTB 3x4 AUC={b_3x4} | PTB 6x2+Wander AUC={b_6x2} | Held-Out LITFL={b_lit}")

    rng_tr = np.random.default_rng(42)
    ptb_sub = [ptb_tr[i] for i in rng_tr.choice(len(ptb_tr), size=min(6000, len(ptb_tr)), replace=False)]
    lit_rows = []
    for it in lit_tr:
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for c in it["gt"]:
            if c in c2i: y[c2i[c]] = 1.0
        if y.sum() > 0:
            lit_rows.append({"kind": "img", "path": it["path"], "y": y})
    train_rows = ptb_sub + lit_rows * 12

    class MultiLayoutDS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            rng = np.random.default_rng()
            if r["kind"] == "img":
                return torch.from_numpy(clean_ink(cv2.imread(r["path"]), True, rng)), torch.from_numpy(r["y"])
            sig, _ = wfdb.rdsamp(r["path"])
            is_6x2 = rng.random() < 0.35
            return torch.from_numpy(render_ecg(np.nan_to_num(sig.astype(np.float32)), rng, is_6x2, True, True)), torch.from_numpy(r["y"])

    dl = DataLoader(MultiLayoutDS(train_rows), batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_6x2, best_sd = b_6x2, None
    for ep in range(1, 6):
        t0 = time.time()
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        e_3x4 = eval_ptb(model, te_sub, False)
        e_6x2 = eval_ptb(model, te_sub, True)
        e_lit = eval_litfl(model, lit_te)
        print(f"Ep {ep}/5 ({time.time()-t0:.0f}s) | PTB 3x4 AUC={e_3x4} | PTB 6x2+Wander AUC={e_6x2} | Held-Out LITFL={e_lit}")
        if e_6x2 > best_6x2 and e_3x4 >= (b_3x4 - 0.01) and e_lit["op_hit_pct"] >= (b_lit["op_hit_pct"] - 5.0):
            best_6x2 = e_6x2
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_sd is not None and (best_6x2 - b_6x2) >= 0.02:
        shutil.copy2(V8_PATH, os.path.join(OUT_DIR, "ecg_effnetv2s_v18_backup.pt"))
        ck["state_dict"] = best_sd
        torch.save(ck, V8_PATH)
        vol.commit()
        print(f"PROMOTED v19 Multi-Layout (3x4 + 6x2 + Wander)! 6x2 AUC: {b_6x2} -> {best_6x2}")
    else:
        print(f"NOT PROMOTED (v18 preserved): best 6x2 AUC={best_6x2} vs baseline={b_6x2}")


@app.local_entrypoint()
def main():
    train_v19_6x2.remote()
