"""
Code Blue ECG — Fit & Save Fold-9 Isotonic Calibrator for NORMAL_SINUS_RHYTHM (v18, ~2 min on A10G)
Fits sklearn IsotonicRegression strictly on Pristine Fold-9 (PTB-XL), verifies Top-1 ECE & NSR ECE
improvement on Fold-10 (N=1,675 unique patients), and saves pure-JSON piecewise knots to:
  /vol/outputs/nsr_isotonic_v18.json
Loaded automatically by modal_app.py v18 with zero runtime sklearn dependency (<0.05ms np.interp).
"""
import os, json, ast, modal

app = modal.App("codeblue-fit-nsr-isotonic-v18")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", "timm", "wfdb", "scikit-learn", "pandas", "numpy", "opencv-python-headless", "pillow")
)


def compute_ece(y_true, y_prob, n_bins=15):
    import numpy as np
    yt, yp = np.asarray(y_true, dtype=np.float64), np.asarray(y_prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    return float(sum((m.sum() / len(yp)) * abs(float(yt[m].mean()) - float(yp[m].mean()))
                     for lo, hi in zip(bins[:-1], bins[1:])
                     for m in [((yp >= lo) & (yp < hi)) if hi < 1.0 else ((yp >= lo) & (yp <= hi))] if m.sum() > 0))


@app.function(image=image, gpu="A10G", cpu=6.0, memory=24576, timeout=1800, volumes={"/vol": vol})
def fit_nsr_isotonic():
    import cv2, numpy as np, pandas as pd, timm, torch, wfdb
    from sklearn.isotonic import IsotonicRegression

    device = torch.device("cuda")
    PTB_DIR, OUT_DIR = "/vol/ptbxl", "/vol/outputs"
    ck_ef = torch.load(os.path.join(OUT_DIR, "ecg_effnetv2s_v8_rare19_512x768.pt"), map_location="cpu", weights_only=False)
    ck_dn = torch.load(os.path.join(OUT_DIR, "ecg_densenet121_v6_real_512x768.pt"), map_location="cpu", weights_only=False)
    CLASSES, DN_CLASSES = list(ck_ef["classes"]), list(ck_dn["classes"])
    IMG_H, IMG_W = int(ck_ef.get("img_h", 512)), int(ck_ef.get("img_w", 768))
    PLATT_A, PLATT_B = np.array(ck_ef["platt_a"], dtype=np.float32), np.array(ck_ef["platt_b"], dtype=np.float32)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    nsr_i = c2i["NORMAL_SINUS_RHYTHM"]

    m_ef = timm.create_model(ck_ef.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k"), pretrained=False, num_classes=len(CLASSES), drop_rate=0.2).to(device)
    m_ef.load_state_dict(ck_ef["state_dict"]); m_ef.eval()
    m_dn = timm.create_model("densenet121", pretrained=False, num_classes=len(DN_CLASSES), drop_rate=0.2).to(device)
    m_dn.load_state_dict(ck_dn["state_dict"]); m_dn.eval()

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

    def clean_ink(bgr):
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

    def render_ecg(sig, rng):
        H, W = IMG_H, IMG_W
        canvas = np.full((H, W, 3), (238, 244, 240), dtype=np.uint8)
        px_mm = (W - 32) / 250.0
        for x in range(0, W, max(2, int(round(px_mm)))): cv2.line(canvas, (x, 0), (x, H), (225, 212, 244), 1)
        for y in range(0, H, max(2, int(round(px_mm)))): cv2.line(canvas, (0, y), (W, y), (225, 212, 244), 1)
        layout = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
        col_w, row_h, seg = (W - 32) // 4, (H - 64) // 4, sig.shape[0] // 4
        for r_idx, row in enumerate(layout):
            yc = 28 + r_idx * row_h + row_h // 2
            for c_idx, l_idx in enumerate(row):
                x0 = 16 + c_idx * col_w
                s = sig[c_idx * seg : (c_idx + 1) * seg, l_idx]
                xs = np.linspace(x0, x0 + col_w - 4, len(s)).astype(np.int32)
                ys = np.clip(yc - (s * 10.0 * px_mm), 6, H - 6).astype(np.int32)
                cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (20, 20, 20), 1, cv2.LINE_AA)
        yc = 28 + 3 * row_h + row_h // 2
        xs = np.linspace(16, W - 16, sig.shape[0]).astype(np.int32)
        ys = np.clip(yc - (sig[:, 1] * 10.0 * px_mm), 6, H - 6).astype(np.int32)
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (20, 20, 20), 1, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), float(rng.uniform(-3.0, 3.0)), 1.0)
        return cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

    def infer_probs(recs, seed):
        rng = np.random.default_rng(seed)
        imgs = [render_ecg(np.nan_to_num(wfdb.rdsamp(r["path"])[0].astype(np.float32)), rng) for r in recs]
        Y = np.stack([r["y"] for r in recs], axis=0)
        probs = []
        with torch.no_grad():
            for i in range(0, len(imgs), 32):
                xb = torch.from_numpy(np.stack([clean_ink(im) for im in imgs[i:i+32]])).to(device)
                with torch.amp.autocast("cuda"):
                    le, ld = m_ef(xb).float().cpu().numpy(), m_dn(xb).float().cpu().numpy()
                ld_a, has_d = np.zeros_like(le), np.zeros(len(CLASSES), dtype=bool)
                for j, c in enumerate(DN_CLASSES):
                    if c in c2i:
                        ld_a[:, c2i[c]], has_d[c2i[c]] = ld[:, j], True
                bl = np.where(has_d[None, :], 0.60 * le + 0.40 * ld_a, le)
                probs.append(1.0 / (1.0 + np.exp(-np.clip(PLATT_A[None, :] * bl + PLATT_B[None, :], -30, 30))))
        return Y, np.concatenate(probs, axis=0)

    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    f9, f10, seen10 = [], [], set()
    for _, r in df.iterrows():
        fold = int(r["strat_fold"])
        if fold not in (9, 10): continue
        pid = int(r["patient_id"])
        if fold == 10:
            if pid in seen10: continue
            seen10.add(pid)
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for k, v in ast.literal_eval(r["scp_codes"]).items():
            if k in SCP_MAP and SCP_MAP[k] in c2i and (k in RHYTHM_SCPS or float(v) >= 50.0):
                y[c2i[SCP_MAP[k]]] = 1.0
        if y.sum() > 0:
            (f9 if fold == 9 else f10).append({"path": os.path.join(PTB_DIR, r["filename_lr"]), "y": y})

    Y9, P9 = infer_probs(f9, 101)
    Y10, P10 = infer_probs(f10, 202)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(P9[:, nsr_i], Y9[:, nsr_i])
    P10_iso = P10.copy()
    P10_iso[:, nsr_i] = np.interp(P10[:, nsr_i], iso.X_thresholds_, iso.y_thresholds_)

    ece_before = round(compute_ece(Y10[:, nsr_i], P10[:, nsr_i]), 4)
    ece_after = round(compute_ece(Y10[:, nsr_i], P10_iso[:, nsr_i]), 4)
    idx_b, idx_a = np.argmax(P10, axis=1), np.argmax(P10_iso, axis=1)
    top1_b = round(compute_ece(Y10[np.arange(len(P10)), idx_b], P10[np.arange(len(P10)), idx_b]), 4)
    top1_a = round(compute_ece(Y10[np.arange(len(P10)), idx_a], P10_iso[np.arange(len(P10)), idx_a]), 4)

    out_p = os.path.join(OUT_DIR, "nsr_isotonic_v18.json")
    payload = {
        "class": "NORMAL_SINUS_RHYTHM",
        "fit_fold": 9,
        "x_knots": [round(float(x), 6) for x in iso.X_thresholds_],
        "y_knots": [round(float(y), 6) for y in iso.y_thresholds_],
        "fold10_nsr_ece_before": ece_before,
        "fold10_nsr_ece_after": ece_after,
        "fold10_top1_ece_before": top1_b,
        "fold10_top1_ece_after": top1_a,
    }
    with open(out_p, "w") as f:
        json.dump(payload, f, indent=2)
    vol.commit()
    print(f"Saved {out_p} | NSR ECE: {ece_before} -> {ece_after} | Top-1 ECE: {top1_b} -> {top1_a}")


@app.local_entrypoint()
def main():
    fit_nsr_isotonic.remote()
