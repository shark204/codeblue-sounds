"""
Code Blue ECG — 3-Model Diverse Super-Ensemble Benchmark (~79M params)
Evaluates head-to-head on FULL Pristine Fold-10 (N=1,694: 3x4+1 & 6x2+Wander) + Held-Out LITFL Even-49:
  1) ConvNeXt-Small v20 (50M params, ImageNet-22K, 0.9723 F10 6x2+Wander AUC)
  2) EfficientNetV2-S v18 (21M params, LITFL External-Domain Hardened)
  3) DenseNet121 v6_real (8M params, Current Production Partner)

Zero retraining — loads all 3 saved checkpoints from /vol/outputs/ and benchmarks
candidate ensemble weights cleanly in ~2-3 minutes on Modal A10G.
"""
import os, json, ast, math, time, modal

app = modal.App("codeblue-eval-3model-ensemble")
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


@app.function(image=image, gpu="A10G", cpu=6.0, memory=24576, timeout=1800, volumes={"/vol": vol})
def eval_3model_ensemble():
    import cv2, numpy as np, pandas as pd, timm, torch, wfdb
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    device = torch.device("cuda")
    PTB_DIR, OUT_DIR, LITFL_DIR = "/vol/ptbxl", "/vol/outputs", "/vol/litfl_top150_v18"
    EFF_PATH = os.path.join(OUT_DIR, "ecg_effnetv2s_v8_rare19_512x768.pt")
    CNX_PATH = os.path.join(OUT_DIR, "ecg_convnext_small_v20_512x768.pt")
    DN_PATH = os.path.join(OUT_DIR, "ecg_densenet121_v6_real_512x768.pt")

    ck_eff = torch.load(EFF_PATH, map_location="cpu", weights_only=False)
    ck_cnx = torch.load(CNX_PATH, map_location="cpu", weights_only=False)
    ck_dn = torch.load(DN_PATH, map_location="cpu", weights_only=False)

    CLASSES = list(ck_eff["classes"])
    DN_CLASSES = list(ck_dn["classes"])
    IMG_H, IMG_W = int(ck_eff.get("img_h", 512)), int(ck_eff.get("img_w", 768))
    c2i = {c: i for i, c in enumerate(CLASSES)}
    dn_idx = [DN_CLASSES.index(c) if c in DN_CLASSES else -1 for c in CLASSES]

    EFF_PA, EFF_PB = np.array(ck_eff["platt_a"], dtype=np.float32), np.array(ck_eff["platt_b"], dtype=np.float32)
    CNX_PA, CNX_PB = np.array(ck_cnx["platt_a"], dtype=np.float32), np.array(ck_cnx["platt_b"], dtype=np.float32)
    EFF_THR = dict(ck_eff.get("thresholds", {}))
    DN_TEMP = max(1.0, float(ck_dn.get("calib_temp", 1.1721)))
    DN_BIAS = float(ck_dn.get("calib_bias", 1.447))

    m_eff = timm.create_model(ck_eff.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k"), pretrained=False, num_classes=len(CLASSES)).to(device).eval()
    m_eff.load_state_dict(ck_eff["state_dict"])

    m_cnx = timm.create_model(ck_cnx["arch"], pretrained=False, num_classes=len(CLASSES)).to(device).eval()
    m_cnx.load_state_dict(ck_cnx["state_dict"])

    m_dn = timm.create_model("densenet121", pretrained=False, num_classes=len(DN_CLASSES)).to(device).eval()
    m_dn.load_state_dict(ck_dn["state_dict"])

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
    ptb_f10, seen10 = [], set()
    for _, r in df.iterrows():
        if int(r["strat_fold"]) != 10 or int(r["patient_id"]) in seen10:
            continue
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for k, v in ast.literal_eval(r["scp_codes"]).items():
            if k in SCP_MAP and SCP_MAP[k] in c2i and (k in RHYTHM_SCPS or float(v) >= 50.0):
                y[c2i[SCP_MAP[k]]] = 1.0
        if y.sum() > 0:
            seen10.add(int(r["patient_id"]))
            ptb_f10.append({"path": os.path.join(PTB_DIR, r["filename_lr"]), "y": y})

    lit_all = json.load(open(os.path.join(LITFL_DIR, "litfl150_meta.json")))
    lit_te = [x for x in lit_all if x["num"] % 2 == 0]
    print(f"[3-Model Ensemble Eval] FULL Fold-10 N={len(ptb_f10)} | Held-Out LITFL Even N={len(lit_te)}")

    def clean_ink(bgr):
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

    def render_ecg(sig, rng, layout_6x2=False):
        H, W = IMG_H, IMG_W
        if layout_6x2 and rng.random() < 0.85:
            sig = sig.copy()
            t = np.linspace(0, 1, sig.shape[0], dtype=np.float32)
            for ch in range(sig.shape[1]):
                if rng.random() < 0.45:
                    sig[:, ch] += float(rng.uniform(0.12, 0.42)) * np.sin(2 * math.pi * float(rng.uniform(0.25, 0.9)) * t + float(rng.uniform(0, 2 * math.pi)))
        canvas = np.full((H, W, 3), (242, 236, 252), dtype=np.uint8)
        px_mm = (W - 32) / 250.0
        step_px = max(2, int(round(px_mm)))
        for x in range(0, W, step_px): cv2.line(canvas, (x, 0), (x, H), (225, 212, 244), 1)
        for y in range(0, H, step_px): cv2.line(canvas, (0, y), (W, y), (225, 212, 244), 1)
        n_samp = sig.shape[0]
        if layout_6x2:
            col_w, row_h, half = (W - 32) // 2, (H - 36) // 6, n_samp // 2
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
        return clean_ink(canvas)

    def sigm(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def infer_batch(xs_np):
        xb = torch.from_numpy(xs_np).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            le = m_eff(xb).float().cpu().numpy()
            lc = m_cnx(xb).float().cpu().numpy()
            ld = m_dn(xb).float().cpu().numpy()
        pe = sigm(EFF_PA * le + EFF_PB)
        pc = sigm(CNX_PA * lc + CNX_PB)
        pd_raw = sigm((ld - DN_BIAS) / DN_TEMP)
        pd_full = np.full_like(pe, np.nan)
        for j, di in enumerate(dn_idx):
            if di >= 0:
                pd_full[:, j] = pd_raw[:, di]
        return pc, pe, pd_full

    def collect_ptb(is_6x2):
        rng = np.random.default_rng(77 if is_6x2 else 33)
        pcs, pes, pds, ys = [], [], [], []
        for i in range(0, len(ptb_f10), 32):
            batch = ptb_f10[i : i + 32]
            xs = np.stack([render_ecg(np.nan_to_num(wfdb.rdsamp(r["path"])[0].astype(np.float32)), rng, is_6x2) for r in batch])
            pc, pe, pd_f = infer_batch(xs)
            pcs.append(pc); pes.append(pe); pds.append(pd_f); ys.append(np.stack([r["y"] for r in batch]))
        return np.concatenate(pcs), np.concatenate(pes), np.concatenate(pds), np.concatenate(ys)

    pc3, pe3, pd3, Y10 = collect_ptb(False)
    pc6, pe6, pd6, _ = collect_ptb(True)

    # Collect LITFL Even-49
    lit_xs, lit_gts = [], []
    for it in lit_te:
        im = cv2.imread(it["path"])
        if im is not None:
            lit_xs.append(clean_ink(im))
            lit_gts.append(set(it["gt"]))
    pcL, peL, pdL = infer_batch(np.stack(lit_xs))

    def blend(pc, pe, pd_f, wc, we, wd):
        out = np.zeros_like(pe)
        for j in range(len(CLASSES)):
            if np.isnan(pd_f[0, j]) or wd == 0.0:
                s = max(1e-8, wc + we)
                out[:, j] = (wc * pc[:, j] + we * pe[:, j]) / s
            else:
                out[:, j] = wc * pc[:, j] + we * pe[:, j] + wd * pd_f[:, j]
        return out

    def m_auc(Y, P):
        return round(float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(len(CLASSES)) if Y[:, j].sum() > 0 and (1 - Y[:, j]).sum() > 0])), 4)

    def eval_lit_p(P):
        top3, op = 0, 0
        for i, gt in enumerate(lit_gts):
            if any(CLASSES[k] in gt for k in np.argsort(-P[i])[:3]):
                top3 += 1
            if {CLASSES[j] for j in range(len(CLASSES)) if P[i, j] >= float(EFF_THR.get(CLASSES[j], 0.45))} & gt:
                op += 1
        n = len(lit_gts)
        return round(100 * top3 / n, 1), round(100 * op / n, 1), wilson_ci(op, n)

    configs = [
        ("v18 EffNetV2-S alone",       0.00, 1.00, 0.00),
        ("PROD Current (0.60E+0.40D)", 0.00, 0.60, 0.40),
        ("v20 ConvNeXt-Small alone",   1.00, 0.00, 0.00),
        ("2-Model (0.50C+0.50E)",      0.50, 0.50, 0.00),
        ("3-Model (0.45C+0.40E+0.15D)",0.45, 0.40, 0.15),
        ("3-Model (0.50C+0.35E+0.15D)",0.50, 0.35, 0.15),
        ("3-Model (0.35C+0.50E+0.15D)",0.35, 0.50, 0.15),
        ("3-Model (0.40C+0.40E+0.20D)",0.40, 0.40, 0.20),
    ]
    print("\n=== HEAD-TO-HEAD 3-MODEL ENSEMBLE COMPARISON (FULL F10 N=1694 + LITFL N=49) ===")
    results = []
    for name, wc, we, wd in configs:
        a3 = m_auc(Y10, blend(pc3, pe3, pd3, wc, we, wd))
        a6 = m_auc(Y10, blend(pc6, pe6, pd6, wc, we, wd))
        t3, op, ci = eval_lit_p(blend(pcL, peL, pdL, wc, we, wd))
        row = {"name": name, "weights": [wc, we, wd], "f10_3x4_auc": a3, "f10_6x2_auc": a6, "litfl_top3": t3, "litfl_op_hit": op, "litfl_ci95": ci}
        results.append(row)
        print(f"{name:28s} | F10 3x4={a3:.4f} | F10 6x2+W={a6:.4f} | LITFL Top3={t3:4.1f}% | OpHit={op:4.1f}% {ci}")

    with open(os.path.join(OUT_DIR, "ensemble_3model_v20_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    print(f"\nDone in {time.time()-t0:.1f}s. Saved /vol/outputs/ensemble_3model_v20_eval.json")


@app.local_entrypoint()
def main():
    eval_3model_ensemble.remote()
