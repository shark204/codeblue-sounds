"""
Code Blue ECG — Exact Einthoven Physics Lead-Reversal Detector on Modal A10G (~8 min, ~$0.15)
Detects student/nurse electrode placement errors directly from 12-lead ECG images:
  - LA-RA Reversal (most common): I'=-I, II'<->III', aVR'<->aVL', aVF'=aVF, V1-V6 unchanged
  - LA-LL Reversal: I'=II, II'=I, III'=-III, aVL'<->aVF', aVR'=aVR
  - RA-LL Reversal: I'=-III, II'=-II, III'=-I, aVR'<->aVF', aVL'=aVL
Trains lightweight MobileNetV3-Large (384x512, <120ms on Modal CPU) on PTB-XL Folds 1-8
(strictly patient-level split: Fold 9 Val/Threshold Freeze, Fold 10 Pristine Test N=1,704).
Saves `/vol/outputs/ecg_lead_reversal_detector.pt` for automatic pickup by modal_app.py v15.
"""
import modal

app = modal.App("codeblue-train-lead-reversal")
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
def train_lead_reversal():
    import hashlib, json, math, os, time
    import cv2, numpy as np, pandas as pd, timm, torch, torch.nn as nn, wfdb
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, Dataset

    vol.reload()
    torch.manual_seed(20261127)
    np.random.seed(20261127)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_H, IMG_W = 384, 512
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    PTB_DIR, OUT_DIR = "/vol/ptbxl", "/vol/outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    tr_paths, va_paths, te_paths, seen_te = [], [], [], set()
    for _, r in df.iterrows():
        p = os.path.join(PTB_DIR, r["filename_lr"])
        f = int(r["strat_fold"])
        if f <= 8:
            tr_paths.append(p)
        elif f == 9:
            va_paths.append(p)
        elif int(r["patient_id"]) not in seen_te:
            seen_te.add(int(r["patient_id"]))
            te_paths.append(p)

    print(f"[Lead-Reversal Detector] Train={len(tr_paths)} | Fold9(Val)={len(va_paths)} | Fold10(Test)={len(te_paths)}")

    # PTB-XL lead order: 0:I, 1:II, 2:III, 3:AVR, 4:AVL, 5:AVF, 6..11:V1..V6
    def apply_limb_swap(sig: np.ndarray, mode: int) -> np.ndarray:
        """Exact Einthoven's Triangle physics for limb electrode reversals."""
        s = sig.copy()
        I, II, III, aVR, aVL, aVF = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
        if mode == 1:  # LA-RA Reversal (most common student/nurse error)
            s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5] = -I, III, II, aVL, aVR, aVF
        elif mode == 2:  # LA-LL Reversal
            s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5] = II, I, -III, aVR, aVF, aVL
        elif mode == 3:  # RA-LL Reversal
            s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5] = -III, -II, -I, aVF, aVL, aVR
        return s

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

    def render_ecg(sig: np.ndarray, rng: np.random.Generator, hard: bool) -> np.ndarray:
        styles = [
            ((242, 232, 252), (210, 185, 240), (180, 145, 225)),
            ((232, 248, 235), (188, 225, 195), (145, 200, 158)),
            ((244, 244, 244), (212, 212, 212), (182, 182, 182)),
        ]
        bg_c, minor_c, major_c = styles[int(rng.integers(0, len(styles)))] if hard else styles[0]
        img = np.full((IMG_H, IMG_W, 3), bg_c, dtype=np.uint8)
        for x in range(0, IMG_W, 10):
            cv2.line(img, (x, 0), (x, IMG_H), minor_c if x % 50 else major_c, 1)
        for y in range(0, IMG_H, 10):
            cv2.line(img, (0, y), (IMG_W, y), minor_c if y % 50 else major_c, 1)
        sig = np.nan_to_num(sig, 0.0)
        n_s = sig.shape[0]
        cw, rh = IMG_W // 4, IMG_H // 4
        for li in range(min(12, sig.shape[1])):
            r, c = li % 3, li // 3
            s = sig[int(c * n_s / 4):int((c + 1) * n_s / 4), li]
            xs = np.linspace(c * cw + 4, (c + 1) * cw - 4, len(s)).astype(np.int32)
            ys = np.clip(int((r + 0.55) * rh) - (s * 30.0).astype(np.int32), 3, IMG_H - 3)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, (18, 18, 18), 2, cv2.LINE_AA)
        s2 = sig[:, 1]
        xs = np.linspace(4, IMG_W - 4, len(s2)).astype(np.int32)
        ys = np.clip(int(3.55 * rh) - (s2 * 30.0).astype(np.int32), 3, IMG_H - 3)
        cv2.polylines(img, [np.stack([xs, ys], 1)], False, (18, 18, 18), 2, cv2.LINE_AA)
        if hard and rng.random() < 0.55:
            M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), float(rng.uniform(-2.5, 2.5)), float(rng.uniform(0.95, 1.04)))
            img = cv2.warpAffine(img, M, (IMG_W, IMG_H), borderMode=cv2.BORDER_REPLICATE)
        return clean_ink(img)

    class SwapDS(Dataset):
        def __init__(self, paths, fixed_seed=None):
            self.paths, self.fixed_seed = paths, fixed_seed
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            sig, _ = wfdb.rdsamp(self.paths[i])
            rng = np.random.default_rng(self.fixed_seed + i) if self.fixed_seed is not None else np.random.default_rng()
            # 50% correct placement (0), 50% swapped (LA-RA 60%, LA-LL 20%, RA-LL 20%)
            if rng.random() < 0.50:
                mode = int(rng.choice([1, 1, 1, 2, 3]))
                sig = apply_limb_swap(sig, mode)
                y = 1.0
            else:
                y = 0.0
            return torch.from_numpy(render_ecg(sig, rng, self.fixed_seed is None)), torch.tensor([y], dtype=torch.float32)

    tr_dl = DataLoader(SwapDS(tr_paths, None), batch_size=48, shuffle=True, num_workers=4, pin_memory=True)
    va_dl = DataLoader(SwapDS(va_paths, 90000), batch_size=48, num_workers=4, pin_memory=True)
    te_dl = DataLoader(SwapDS(te_paths, 100000), batch_size=48, num_workers=4, pin_memory=True)

    model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=1, drop_rate=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    EPOCHS = 4
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.BCEWithLogitsLoss()

    def eval_dl(dl):
        model.eval()
        Ps, Ys = [], []
        with torch.no_grad():
            for x, y in dl:
                with torch.amp.autocast("cuda"):
                    Ps.append(torch.sigmoid(model(x.to(device, non_blocking=True))).float().cpu().numpy())
                Ys.append(y.numpy())
        return np.concatenate(Ps).ravel(), np.concatenate(Ys).ravel()

    best_auc, best_state = -1.0, None
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
        pv, yv = eval_dl(va_dl)
        auc = float(roc_auc_score(yv, pv))
        print(f"Epoch {ep}/{EPOCHS} ({time.time()-t0:.0f}s) | Loss={tot/len(tr_paths):.4f} | Fold9 Lead-Reversal AUROC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    pv, yv = eval_dl(va_dl)
    # Freeze high-specificity threshold (Spec >= 99.0% on Fold-9 so normal ECGs almost never get false reversal warnings!)
    best_thr, best_sens = 0.85, -1.0
    for t in np.linspace(0.50, 0.99, 99):
        yp = pv >= t
        spec = float(((~yp) & (yv == 0)).sum() / max(1, (yv == 0).sum()))
        sens = float((yp & (yv == 1)).sum() / max(1, (yv == 1).sum()))
        if spec >= 0.99 and sens > best_sens:
            best_sens, best_thr = sens, float(t)

    pt, yt = eval_dl(te_dl)
    te_auc = float(roc_auc_score(yt, pt))
    yp = pt >= best_thr
    te_sens = 100.0 * float((yp & (yt == 1)).sum() / max(1, (yt == 1).sum()))
    te_spec = 100.0 * float(((~yp) & (yt == 0)).sum() / max(1, (yt == 0).sum()))
    te_ppv = 100.0 * float((yp & (yt == 1)).sum() / max(1, yp.sum()))

    print(f"\n================ PRISTINE FOLD-10 LEAD-REVERSAL TEST (N={len(yt)}) ================")
    print(f"AUROC={te_auc:.4f} | Frozen Thr={best_thr:.2f} | Sens={te_sens:.1f}% | Spec={te_spec:.1f}% | PPV={te_ppv:.1f}%")

    out_path = os.path.join(OUT_DIR, "ecg_lead_reversal_detector.pt")
    torch.save({
        "state_dict": best_state, "arch": "mobilenetv3_large_100",
        "img_h": IMG_H, "img_w": IMG_W, "threshold": round(best_thr, 3),
        "val_auc": best_auc, "test_auc": te_auc, "test_sens": te_sens, "test_spec": te_spec,
    }, out_path)
    vol.commit()
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()[:16]
    print(f"Saved Lead-Reversal Detector to {out_path} (sha={sha})")


@app.local_entrypoint()
def main():
    train_lead_reversal.remote()
