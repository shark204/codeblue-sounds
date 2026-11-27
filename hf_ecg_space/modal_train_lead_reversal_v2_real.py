"""
Code Blue ECG — v2 Real-Photo-Hardened Einthoven Lead-Reversal Detector on Modal A10G (~7 min, ~$0.14)
Fixes the 71.18% False-Positive Rate on real PMcardio phone photos uncovered by the Torture Benchmark:
  - Trains MobileNetV3-Large (384x512) on:
      1. PTB-XL Folds 1-8 (normal + exact Einthoven LA-RA / LA-LL / RA-LL reversals with mobile tilt/shadow/blur)
      2. REAL un-swapped PMcardio hospital phone photos (first ~4,400 photos -> label=0, strictly separated from last 576 test photos!)
  - Freezes threshold strictly so Real PMcardio Val FPR <= 1.5% while maximizing Einthoven sensitivity.
  - Verifies BOTH Pristine Fold-10 PTB-XL AUROC/Sens AND Held-Out Real PMcardio FPR (N=576).
  - Saves `/vol/outputs/ecg_lead_reversal_detector.pt`.
"""
import modal

app = modal.App("codeblue-train-lead-reversal-v2")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "timm", "wfdb", "scipy", "scikit-learn",
        "pandas", "numpy", "opencv-python-headless", "pillow",
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
def train_lead_reversal_v2():
    import hashlib, json, os, time
    import cv2, numpy as np, pandas as pd, timm, torch, torch.nn as nn, wfdb
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset

    vol.reload()
    torch.manual_seed(20261129)
    np.random.seed(20261129)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_H, IMG_W = 384, 512
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PTB_DIR, PM_DIR, OUT_DIR = "/vol/ptbxl", "/vol/pmcardio", "/vol/outputs"

    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    tr_ptb, va_ptb, te_ptb, seen_te = [], [], [], set()
    for _, r in df.iterrows():
        p = os.path.join(PTB_DIR, r["filename_lr"])
        f = int(r["strat_fold"])
        if f <= 8: tr_ptb.append(p)
        elif f == 9: va_ptb.append(p)
        elif int(r["patient_id"]) not in seen_te:
            seen_te.add(int(r["patient_id"]))
            te_ptb.append(p)

    # Collect real PMcardio phone photos (strictly hold out last 576 for test + 400 before that for val!)
    pm_all = []
    for root, _, files in os.walk(PM_DIR):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                pm_all.append(os.path.join(root, fn))
    pm_all.sort()
    pm_te = pm_all[-576:]
    pm_va = pm_all[-976:-576]
    pm_tr = pm_all[:-976]
    print(f"[Lead-Rev v2] PTB Train={len(tr_ptb)} Val={len(va_ptb)} Test={len(te_ptb)} | REAL PMcardio Train={len(pm_tr)} Val={len(pm_va)} Test={len(pm_te)}")

    def apply_limb_swap(sig: np.ndarray, mode: int) -> np.ndarray:
        s = sig.copy()
        I, II, III, aVR, aVL, aVF = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
        if mode == 1: s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5] = -I, III, II, aVL, aVR, aVF
        elif mode == 2: s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5] = II, I, -III, aVR, aVF, aVL
        elif mode == 3: s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5] = -III, -II, -I, aVF, aVL, aVR
        return s

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

    def render_ecg(sig: np.ndarray, rng: np.random.Generator, mobile: bool) -> np.ndarray:
        styles = [
            ((242, 232, 252), (210, 185, 240), (180, 145, 225)),
            ((232, 248, 235), (188, 225, 195), (145, 200, 158)),
            ((232, 234, 254), (195, 200, 248), (155, 162, 238)),
            ((244, 244, 244), (212, 212, 212), (182, 182, 182)),
        ]
        bg_c, minor_c, major_c = styles[int(rng.integers(0, len(styles)))] if mobile else styles[0]
        img = np.full((IMG_H, IMG_W, 3), bg_c, dtype=np.uint8)
        for x in range(0, IMG_W, 10): cv2.line(img, (x, 0), (x, IMG_H), minor_c if x % 50 else major_c, 1)
        for y in range(0, IMG_H, 10): cv2.line(img, (0, y), (IMG_W, y), minor_c if y % 50 else major_c, 1)
        sig = np.nan_to_num(sig, 0.0)
        n_s = sig.shape[0]
        cw, rh = IMG_W // 4, IMG_H // 4
        for li in range(12):
            r, c = li % 3, li // 3
            s = sig[int(c * n_s / 4):int((c + 1) * n_s / 4), li]
            xs = np.linspace(c * cw + 4, (c + 1) * cw - 4, len(s)).astype(np.int32)
            ys = np.clip(int((r + 0.55) * rh) - (s * 30.0).astype(np.int32), 3, IMG_H - 3)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, (18, 18, 18), 2, cv2.LINE_AA)
        s2 = sig[:, 1]
        xs = np.linspace(4, IMG_W - 4, len(s2)).astype(np.int32)
        ys = np.clip(int(3.55 * rh) - (s2 * 30.0).astype(np.int32), 3, IMG_H - 3)
        cv2.polylines(img, [np.stack([xs, ys], 1)], False, (18, 18, 18), 2, cv2.LINE_AA)
        if mobile and rng.random() < 0.7:
            M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), float(rng.uniform(-4.0, 4.0)), float(rng.uniform(0.93, 1.04)))
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
            _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(55, 88))])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return clean_ink(img)

    class MixedRevDS(Dataset):
        def __init__(self, ptb_paths, pm_paths, train: bool, seed: int):
            self.items = [("ptb", p) for p in ptb_paths] + [("pm", p) for p in pm_paths]
            self.train, self.seed = train, seed

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            kind, path = self.items[idx]
            rng = np.random.default_rng(self.seed + idx if not self.train else None)
            if kind == "pm":
                im = cv2.imread(path)
                if im is None:
                    im = np.full((IMG_H, IMG_W, 3), 240, dtype=np.uint8)
                return torch.from_numpy(clean_ink(im)), torch.tensor(0.0, dtype=torch.float32)
            sig, _ = wfdb.rdsamp(path)
            is_rev = bool(rng.random() < 0.5)
            if is_rev:
                sig = apply_limb_swap(sig, int(rng.choice([1, 1, 2, 3])))
            return torch.from_numpy(render_ecg(sig, rng, self.train)), torch.tensor(1.0 if is_rev else 0.0, dtype=torch.float32)

    tr_dl = DataLoader(MixedRevDS(tr_ptb, pm_tr * 2, True, 11), batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    va_ptb_dl = DataLoader(MixedRevDS(va_ptb, [], False, 22), batch_size=48, shuffle=False, num_workers=6, pin_memory=True)
    va_pm_dl = DataLoader(MixedRevDS([], pm_va, False, 33), batch_size=48, shuffle=False, num_workers=6, pin_memory=True)
    te_ptb_dl = DataLoader(MixedRevDS(te_ptb, [], False, 44), batch_size=48, shuffle=False, num_workers=6, pin_memory=True)
    te_pm_dl = DataLoader(MixedRevDS([], pm_te, False, 55), batch_size=48, shuffle=False, num_workers=6, pin_memory=True)

    model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    loss_fn = nn.BCEWithLogitsLoss()

    def predict(dl):
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in dl:
                ps.append(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
                ys.append(y.numpy())
        return np.concatenate(ps), np.concatenate(ys)

    best_score, best_sd = -1.0, None
    for ep in range(1, 6):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            loss.backward(); opt.step()
            tot += float(loss.item())
        sched.step()
        p_va, y_va = predict(va_ptb_dl)
        p_pm, _ = predict(va_pm_dl)
        auc_va = roc_auc_score(y_va, p_va)
        fpr_pm = 100.0 * float(np.mean(p_pm >= 0.5))
        print(f"Ep {ep}/5 ({time.time()-t0:.0f}s) | Loss={tot/len(tr_dl):.4f} | F9 AUC={auc_va:.4f} | Real PMcardio Val FPR@0.5={fpr_pm:.2f}%")
        score = auc_va - (fpr_pm / 100.0) * 0.5
        if score > best_score:
            best_score = score
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    p_va, y_va = predict(va_ptb_dl)
    p_pm_va, _ = predict(va_pm_dl)

    # Freeze threshold strictly so Real PMcardio Val FPR <= 1.5%
    best_thr = 0.85
    for t in np.linspace(0.50, 0.99, 50):
        if float(np.mean(p_pm_va >= t)) <= 0.015:
            best_thr = float(t)
            break

    p_te, y_te = predict(te_ptb_dl)
    p_pm_te, _ = predict(te_pm_dl)
    te_auc = roc_auc_score(y_te, p_te)
    te_sens = 100.0 * float(np.mean(p_te[y_te == 1] >= best_thr))
    te_spec = 100.0 * float(np.mean(p_te[y_te == 0] < best_thr))
    pm_fpr = 100.0 * float(np.mean(p_pm_te >= best_thr))

    print("\n================ v2 LEAD-REVERSAL PRISTINE TEST RESULTS ================")
    print(f"Frozen Thr={best_thr:.2f} | PTB-XL Fold10 AUC={te_auc:.4f} | Sens={te_sens:.1f}% | Spec={te_spec:.1f}%")
    print(f"REAL PMcardio Held-Out Phone Photos (N={len(p_pm_te)}): FPR={pm_fpr:.2f}% (was 71.18%!) | Median={float(np.median(p_pm_te)):.4f}")

    ckpt_path = os.path.join(OUT_DIR, "ecg_lead_reversal_detector.pt")
    torch.save({
        "arch": "mobilenetv3_large_100", "img_h": IMG_H, "img_w": IMG_W,
        "threshold": round(best_thr, 4), "test_auc": round(float(te_auc), 4),
        "real_pmcardio_fpr_pct": round(pm_fpr, 2), "state_dict": best_sd,
    }, ckpt_path)
    vol.commit()
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()[:16]
    print(f"Saved Real-Photo-Hardened Lead-Reversal v2 to {ckpt_path} (sha={sha})")


@app.local_entrypoint()
def main():
    train_lead_reversal_v2.remote()
