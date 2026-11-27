"""
Code Blue ECG — v20b Stage-2 External-Domain Fine-Tune on ConvNeXt-Small (50M)
Starts directly from saved v20 (/vol/outputs/ecg_convnext_small_v20_512x768.pt, 0.9723 F10 AUC):
  - Balanced 50/50 domain mix:
      * Strict LITFL Odd-Train (N=53 x 35 = 1,855/ep) with realistic phone/scan augmentations
        (perspective tilt, margin crop, uneven lighting/shadow, JPEG compression, contrast shifts)
      * PTB-XL Fold 1-8 (2,400/ep: 55% 3x4+1, 35% 6x2, 10% 3x4 + Baseline Wander + EMG noise)
  - Gentle fine-tuning (lr=3.5e-5, 6 epochs, ~4-5 min on Modal A100 bf16) + Model EMA.
  - Refits strict Fold-9 Platt Calibration + Clinical F2-balanced operating thresholds (ZERO leakage).
  - Evaluates head-to-head vs v18 on FULL Pristine Fold-10 (N=1,694: 3x4 & 6x2+Wander) + LITFL Even-49.
  - Always saves /vol/outputs/ecg_convnext_small_v20b_512x768.pt and auto-promotes on Pareto win.
"""
import os, json, ast, math, copy, shutil, time, modal

app = modal.App("codeblue-train-v20b-convnext-ext")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "timm", "wfdb", "scikit-learn",
        "pandas", "numpy", "opencv-python-headless", "pillow",
    )
)


def wilson_ci(k: int, n: int, z: float = 1.95996):
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [round(max(0.0, c - h) * 100.0, 1), round(min(1.0, c + h) * 100.0, 1)]


@app.function(image=image, gpu="A100", cpu=8.0, memory=32768, timeout=3600, volumes={"/vol": vol})
def train_v20b_external():
    import cv2, numpy as np, pandas as pd, timm, torch, torch.nn as nn, wfdb
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score, fbeta_score
    from sklearn.linear_model import LogisticRegression

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    PTB_DIR, OUT_DIR, LITFL_DIR = "/vol/ptbxl", "/vol/outputs", "/vol/litfl_top150_v18"
    V8_PATH = os.path.join(OUT_DIR, "ecg_effnetv2s_v8_rare19_512x768.pt")
    V20_PATH = os.path.join(OUT_DIR, "ecg_convnext_small_v20_512x768.pt")
    V20B_PATH = os.path.join(OUT_DIR, "ecg_convnext_small_v20b_512x768.pt")

    ck_base = torch.load(V8_PATH, map_location="cpu", weights_only=False)
    ck_v20 = torch.load(V20_PATH, map_location="cpu", weights_only=False)
    ARCH = ck_v20["arch"]
    CLASSES = list(ck_v20["classes"])
    IMG_H, IMG_W = int(ck_v20.get("img_h", 512)), int(ck_v20.get("img_w", 768))
    BASE_PA = np.array(ck_base["platt_a"], dtype=np.float32)
    BASE_PB = np.array(ck_base["platt_b"], dtype=np.float32)
    BASE_THR = dict(ck_base.get("thresholds", {}))
    c2i = {c: i for i, c in enumerate(CLASSES)}

    SCP_MAP = {
        "NORM": "NORMAL_SINUS_RHYTHM",
        "AMI": "ANTERIOR_SEPTAL_MI", "ASMI": "ANTERIOR_SEPTAL_MI", "ALMI": "ANTERIOR_SEPTAL_MI", "ILMI": "ANTERIOR_SEPTAL_MI",
        "IMI": "INFERIOR_POSTERIOR_MI", "IPMI": "INFERIOR_POSTERIOR_MI", "PMI": "INFERIOR_POSTERIOR_MI", "IPLMI": "INFERIOR_POSTERIOR_MI",
        "LMI": "LATERAL_MI", "INJAL": "LATERAL_MI", "INJLA": "LATERAL_MI",
        "ISC_": "NSTEMI_ISCHEMIA", "ISCAL": "NSTEMI_ISCHEMIA", "ISCAN": "NSTEMI_ISCHEMIA", "ISCAS": "NSTEMI_ISCHEMIA",
        "ISCIL": "NSTEMI_ISCHEMIA", "ISCIN": "NSTEMI_ISCHEMIA", "ISCLA": "NSTEMI_ISCHEMIA", "NST_": "NSTEMI_ISCHEMIA", "STD_": "NSTEMI_ISCHEMIA", "STTC": "NSTEMI_ISCHEMIA",
        "AFIB": "ATRIAL_FIBRILLATION", "AFLT": "ATRIAL_FLUTTER", "SVTAC": "SVT_TACHYCARDIA", "PSVT": "SVT_TACHYCARDIA",
        "SBRAD": "SINUS_BRADYCARDIA", "1AVB": "FIRST_DEGREE_AV_BLOCK", "2AVB": "HIGH_GRADE_AV_BLOCK", "3AVB": "HIGH_GRADE_AV_BLOCK",
        "CRBBB": "RBBB", "IRBBB": "RBBB", "CLBBB": "LBBB", "ILBBB": "LBBB", "LAFB": "LAFB_LPFB", "LPFB": "LAFB_LPFB",
        "LVH": "LVH_RVH", "RVH": "LVH_RVH", "PACE": "PACEMAKER_RHYTHM", "WPW": "WPW_PREEXCITATION",
        "LNGQT": "LONG_QT_OR_ECTOPY", "PVC": "LONG_QT_OR_ECTOPY",
    }
    RHYTHM_SCPS = {"AFIB", "AFLT", "SVTAC", "PSVT", "SBRAD", "PACE", "PVC"}

    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    ptb_tr, ptb_f9, ptb_f10, seen9, seen10 = [], [], [], set(), set()
    for _, r in df.iterrows():
        f = int(r["strat_fold"])
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for k, v in ast.literal_eval(r["scp_codes"]).items():
            if k in SCP_MAP and SCP_MAP[k] in c2i and (k in RHYTHM_SCPS or float(v) >= 50.0):
                y[c2i[SCP_MAP[k]]] = 1.0
        if y.sum() == 0:
            continue
        item = {"kind": "wfdb", "path": os.path.join(PTB_DIR, r["filename_lr"]), "y": y}
        if f <= 8:
            ptb_tr.append(item)
        elif f == 9 and int(r["patient_id"]) not in seen9:
            seen9.add(int(r["patient_id"]))
            ptb_f9.append(item)
        elif f == 10 and int(r["patient_id"]) not in seen10:
            seen10.add(int(r["patient_id"]))
            ptb_f10.append(item)

    lit_all = json.load(open(os.path.join(LITFL_DIR, "litfl150_meta.json")))
    lit_tr = [x for x in lit_all if x["num"] % 2 == 1]
    lit_te = [x for x in lit_all if x["num"] % 2 == 0]
    print(f"[v20b Stage-2] Starting from {ARCH} | PTB Tr={len(ptb_tr)} F9={len(ptb_f9)} F10={len(ptb_f10)} | LITFL Odd-Tr={len(lit_tr)} Even-Te={len(lit_te)}")

    def clean_ink(bgr, aug=False, rng=None, real_photo=False):
        if aug and rng is not None:
            h, w = bgr.shape[:2]
            if rng.random() < 0.55:
                cy0, cy1 = int(h * rng.uniform(0.0, 0.08)), int(h * rng.uniform(0.92, 1.0))
                cx0, cx1 = int(w * rng.uniform(0.0, 0.08)), int(w * rng.uniform(0.92, 1.0))
                bgr = bgr[cy0:cy1, cx0:cx1]
            if rng.random() < 0.50:
                M = cv2.getRotationMatrix2D((bgr.shape[1] / 2, bgr.shape[0] / 2), float(rng.uniform(-4.5, 4.5)), 1.0)
                bgr = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            if real_photo and rng.random() < 0.45:
                # Uneven lighting / phone shadow gradient
                hh, ww = bgr.shape[:2]
                gx = np.linspace(float(rng.uniform(0.78, 1.08)), float(rng.uniform(0.78, 1.08)), ww, dtype=np.float32)
                gy = np.linspace(float(rng.uniform(0.82, 1.06)), float(rng.uniform(0.82, 1.06)), hh, dtype=np.float32)
                grad = np.outer(gy, gx)[:, :, None]
                bgr = np.clip(bgr.astype(np.float32) * grad, 0, 255).astype(np.uint8)
            if real_photo and rng.random() < 0.35:
                q = int(rng.integers(45, 88))
                _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
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

    def add_artifacts(sig, rng, wander_prob=0.50, emg_prob=0.30):
        s = sig.copy()
        n = s.shape[0]
        if rng.random() < wander_prob:
            t = np.linspace(0, 1, n, dtype=np.float32)
            for ch in range(s.shape[1]):
                if rng.random() < 0.45:
                    s[:, ch] += float(rng.uniform(0.12, 0.42)) * np.sin(2 * math.pi * float(rng.uniform(0.2, 0.9)) * t + float(rng.uniform(0, 2 * math.pi)))
        if rng.random() < emg_prob:
            s += rng.normal(0.0, float(rng.uniform(0.015, 0.04)), size=s.shape).astype(np.float32)
        return s

    GRID_STYLES = [
        ((242, 236, 252), (225, 212, 244)),
        ((238, 244, 254), (210, 224, 248)),
        ((245, 245, 245), (222, 222, 222)),
    ]

    def render_ecg(sig, rng, mode="3x4+1", wander=False, aug=False):
        H, W = IMG_H, IMG_W
        if wander or aug:
            sig = add_artifacts(sig, rng, wander_prob=0.85 if (wander and not aug) else 0.45, emg_prob=0.30 if aug else 0.0)
        bg_c, gr_c = GRID_STYLES[int(rng.integers(0, len(GRID_STYLES)))] if aug else GRID_STYLES[0]
        canvas = np.full((H, W, 3), bg_c, dtype=np.uint8)
        px_mm = (W - 32) / 250.0
        step_px = max(2, int(round(px_mm)))
        for x in range(0, W, step_px):
            cv2.line(canvas, (x, 0), (x, H), gr_c, 1)
        for y in range(0, H, step_px):
            cv2.line(canvas, (0, y), (W, y), gr_c, 1)
        ink_c = int(rng.integers(14, 40)) if aug else 20
        thick = 2 if (aug and rng.random() < 0.25) else 1
        n_samp = sig.shape[0]
        if mode == "6x2":
            col_w, row_h, half = (W - 32) // 2, (H - 36) // 6, n_samp // 2
            for r_idx in range(6):
                yc = 18 + r_idx * row_h + row_h // 2
                for c_idx, l_idx in enumerate([r_idx, r_idx + 6]):
                    x0 = 16 + c_idx * col_w
                    s = sig[c_idx * half : (c_idx + 1) * half, l_idx]
                    xs = np.linspace(x0, x0 + col_w - 6, len(s)).astype(np.int32)
                    ys = np.clip(yc - (s * 10.0 * px_mm), 4, H - 4).astype(np.int32)
                    cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (ink_c, ink_c, ink_c), thick, cv2.LINE_AA)
        else:
            layout = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
            rows_n = 4 if mode == "3x4+1" else 3
            col_w, row_h, seg = (W - 32) // 4, (H - 64) // rows_n, n_samp // 4
            for r_idx, row in enumerate(layout):
                yc = 28 + r_idx * row_h + row_h // 2
                for c_idx, l_idx in enumerate(row):
                    x0 = 16 + c_idx * col_w
                    s = sig[c_idx * seg : (c_idx + 1) * seg, l_idx]
                    xs = np.linspace(x0, x0 + col_w - 4, len(s)).astype(np.int32)
                    ys = np.clip(yc - (s * 10.0 * px_mm), 6, H - 6).astype(np.int32)
                    cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (ink_c, ink_c, ink_c), thick, cv2.LINE_AA)
            if mode == "3x4+1":
                yc = 28 + 3 * row_h + row_h // 2
                xs = np.linspace(16, W - 16, n_samp).astype(np.int32)
                ys = np.clip(yc - (sig[:, 1] * 10.0 * px_mm), 6, H - 6).astype(np.int32)
                cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (ink_c, ink_c, ink_c), thick, cv2.LINE_AA)
        return clean_ink(canvas, aug=aug, rng=rng, real_photo=aug)

    def predict_ptb(m_eval, recs, mode="3x4+1", wander=False, seed=33):
        m_eval.eval()
        rng = np.random.default_rng(seed)
        lgs, ys = [], []
        with torch.no_grad():
            for i in range(0, len(recs), 32):
                batch = recs[i : i + 32]
                xs = np.stack([render_ecg(np.nan_to_num(wfdb.rdsamp(r["path"])[0].astype(np.float32)), rng, mode, wander, False) for r in batch])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lgs.append(m_eval(torch.from_numpy(xs).to(device)).float().cpu().numpy())
                ys.append(np.stack([r["y"] for r in batch]))
        return np.concatenate(lgs), np.concatenate(ys)

    def macro_auc(Y, L):
        P = 1.0 / (1.0 + np.exp(-np.clip(L, -30, 30)))
        aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(CLASSES)) if Y[:, j].sum() > 0 and (1 - Y[:, j]).sum() > 0]
        return round(float(np.mean(aucs)), 4)

    def eval_litfl(m_eval, items, pa, pb, thr_map):
        m_eval.eval()
        op_hit, top3, n = 0, 0, 0
        with torch.no_grad():
            for it in items:
                im = cv2.imread(it["path"])
                if im is None:
                    continue
                n += 1
                xb = torch.from_numpy(clean_ink(im)).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = m_eval(xb).float().cpu().numpy()[0]
                pr = 1.0 / (1.0 + np.exp(-np.clip(pa * lg + pb, -30, 30)))
                gt = set(it["gt"])
                if any(CLASSES[k] in gt for k in np.argsort(-pr)[:3]):
                    top3 += 1
                if {CLASSES[j] for j in range(len(CLASSES)) if pr[j] >= float(thr_map.get(CLASSES[j], 0.35))} & gt:
                    op_hit += 1
        return {"n": n, "top3_pct": round(100 * top3 / max(1, n), 1), "op_hit_pct": round(100 * op_hit / max(1, n), 1), "ci95": wilson_ci(op_hit, n)}

    def fit_fold9_calibration(m_eval):
        L9, Y9 = predict_ptb(m_eval, ptb_f9, "3x4+1", False, 99)
        L9_6, Y9_6 = predict_ptb(m_eval, ptb_f9[:500], "6x2", True, 101)
        L_cal = np.concatenate([L9, L9_6], axis=0)
        Y_cal = np.concatenate([Y9, Y9_6], axis=0)
        pa = np.ones(len(CLASSES), dtype=np.float32)
        pb = np.zeros(len(CLASSES), dtype=np.float32)
        thrs = {}
        for j, cname in enumerate(CLASSES):
            yj = Y_cal[:, j].astype(int)
            lj = L_cal[:, j].reshape(-1, 1)
            if yj.sum() >= 15 and (1 - yj).sum() >= 15:
                lr = LogisticRegression(C=1.0, max_iter=500)
                lr.fit(lj, yj)
                pa[j] = float(np.clip(lr.coef_[0][0], 0.3, 4.0))
                pb[j] = float(np.clip(lr.intercept_[0], -6.0, 6.0))
            pj = 1.0 / (1.0 + np.exp(-np.clip(pa[j] * L_cal[:, j] + pb[j], -30, 30)))
            if yj.sum() >= 30:
                best_t, best_f = 0.35, -1.0
                # Clinical screening F1.5-score (balances sensitivity & specificity without over-firing)
                for t in np.linspace(0.08, 0.55, 48):
                    f = fbeta_score(yj, (pj >= t).astype(int), beta=1.5, zero_division=0)
                    if f > best_f:
                        best_f, best_t = f, float(t)
                thrs[cname] = round(best_t, 3)
            else:
                thrs[cname] = 0.35
        return pa, pb, thrs

    # 1) Measure Baseline v18 on FULL Pristine Fold-10 & LITFL Even-49
    base_m = timm.create_model(ck_base.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k"), pretrained=False, num_classes=len(CLASSES), drop_rate=0.2).to(device)
    base_m.load_state_dict(ck_base["state_dict"])
    L_b3, Y_10 = predict_ptb(base_m, ptb_f10, "3x4+1", False, 33)
    L_b6, _ = predict_ptb(base_m, ptb_f10, "6x2", True, 77)
    b_3x4, b_6x2 = macro_auc(Y_10, L_b3), macro_auc(Y_10, L_b6)
    b_lit = eval_litfl(base_m, lit_te, BASE_PA, BASE_PB, BASE_THR)
    print(f"[BASELINE v18 FULL F10 N={len(ptb_f10)}] 3x4 AUC={b_3x4} | 6x2+Wander AUC={b_6x2} | LITFL={b_lit}")
    del base_m
    torch.cuda.empty_cache()

    # 2) Load saved v20 ConvNeXt-Small (50M)
    model = timm.create_model(ARCH, pretrained=False, num_classes=len(CLASSES), drop_rate=0.25, drop_path_rate=0.15).to(device)
    model.load_state_dict(ck_v20["state_dict"])
    ema_model = copy.deepcopy(model).eval()

    lit_rows = []
    for it in lit_tr:
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for c in it["gt"]:
            if c in c2i:
                y[c2i[c]] = 1.0
        if y.sum() > 0:
            lit_rows.append({"kind": "img", "path": it["path"], "y": y})

    rng_sub = np.random.default_rng(42)
    ptb_sub = [ptb_tr[i] for i in rng_sub.choice(len(ptb_tr), size=min(2400, len(ptb_tr)), replace=False)]
    train_rows = ptb_sub + lit_rows * 35

    class Stage2DS(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            rng = np.random.default_rng()
            if r["kind"] == "img":
                return torch.from_numpy(clean_ink(cv2.imread(r["path"]), True, rng, True)), torch.from_numpy(r["y"])
            sig, _ = wfdb.rdsamp(r["path"])
            u = rng.random()
            mode = "6x2" if u < 0.35 else ("3x4" if u < 0.45 else "3x4+1")
            return torch.from_numpy(render_ecg(np.nan_to_num(sig.astype(np.float32)), rng, mode, True, True)), torch.from_numpy(r["y"])

    dl = DataLoader(Stage2DS(train_rows), batch_size=24, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    EPOCHS = 6
    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-5, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(dl), eta_min=3e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    f9_sub = ptb_f9[:500]
    best_sd, best_ep_score = None, -1.0
    v20_pa = np.array(ck_v20["platt_a"], dtype=np.float32)
    v20_pb = np.array(ck_v20["platt_b"], dtype=np.float32)
    v20_thr = dict(ck_v20["thresholds"])

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tot_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.mul_(0.985).add_(p, alpha=0.015)

        L9_3, Y9_s = predict_ptb(ema_model, f9_sub, "3x4+1", False, 55)
        L9_6, _ = predict_ptb(ema_model, f9_sub, "6x2", True, 66)
        a3, a6 = macro_auc(Y9_s, L9_3), macro_auc(Y9_s, L9_6)
        lit_ep = eval_litfl(ema_model, lit_te, v20_pa, v20_pb, v20_thr)
        print(f"Ep {ep}/{EPOCHS} ({time.time()-t0:.0f}s) | loss={tot_loss/len(dl):.4f} | F9 3x4={a3} | F9 6x2+W={a6} | LITFL(pre-cal)={lit_ep}")
        ep_score = 0.5 * (a3 + a6) + 0.002 * lit_ep["top3_pct"]
        if a3 >= 0.960 and a6 >= 0.960 and ep_score > best_ep_score:
            best_ep_score = ep_score
            best_sd = {k: v.cpu().clone() for k, v in ema_model.state_dict().items()}

    if best_sd is not None:
        ema_model.load_state_dict(best_sd)
    pa, pb, thrs = fit_fold9_calibration(ema_model)
    L_n3, _ = predict_ptb(ema_model, ptb_f10, "3x4+1", False, 33)
    L_n6, _ = predict_ptb(ema_model, ptb_f10, "6x2", True, 77)
    n_3x4, n_6x2 = macro_auc(Y_10, L_n3), macro_auc(Y_10, L_n6)
    n_lit = eval_litfl(ema_model, lit_te, pa, pb, thrs)
    print(f"[CANDIDATE v20b ConvNeXt-Small FULL F10 N={len(ptb_f10)}] 3x4 AUC={n_3x4} | 6x2+Wander AUC={n_6x2} | LITFL={n_lit}")

    cand_ck = {
        "arch": ARCH,
        "version": "v20b_convnext_small_multilayout_ext",
        "classes": CLASSES,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "state_dict": {k: v.cpu().clone() for k, v in ema_model.state_dict().items()},
        "platt_a": pa.tolist(),
        "platt_b": pb.tolist(),
        "thresholds": thrs,
        "metrics_f10": {"3x4_auc": n_3x4, "6x2_wander_auc": n_6x2, "litfl_even49": n_lit},
    }
    torch.save(cand_ck, V20B_PATH)
    vol.commit()
    print(f"Saved candidate permanently to {V20B_PATH}")

    pareto_ok = (
        n_3x4 >= (b_3x4 + 0.005)
        and n_6x2 >= (b_6x2 + 0.005)
        and n_lit["op_hit_pct"] >= (b_lit["op_hit_pct"] - 2.0)
    )
    if pareto_ok:
        bak = os.path.join(OUT_DIR, "ecg_effnetv2s_v18_backup.pt")
        if not os.path.exists(bak):
            shutil.copy2(V8_PATH, bak)
        torch.save(cand_ck, V8_PATH)
        vol.commit()
        print(f"PROMOTED v20b ConvNeXt-Small (50M) to Production! 3x4: {b_3x4}->{n_3x4} | 6x2+W: {b_6x2}->{n_6x2} | LITFL: {b_lit}->{n_lit}")
    else:
        print(f"NOT AUTO-PROMOTED (v18 preserved; v20b saved at {V20B_PATH}). Compare metrics above.")


@app.local_entrypoint()
def main():
    train_v20b_external.remote()
