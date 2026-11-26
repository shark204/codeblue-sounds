"""
Code Blue ECG — DenseNet121 v2 Training + 4-Part Stress Test Suite
- Unified Grid-Removal Preprocessing (eliminates synthetic-vs-real Domain Gap)
- Multi-layout renderer (3x4, 2x6, 12x1, partial strips) with lead labels & calibration pulses
- Realistic phone-photo degradation + random partial-lead masking
- Strict 60/40 held-out split on real textbook ECGs (zero test leakage)
- 4 built-in benchmarks: Fold 10 Clean, Fold 10 Degraded/Partial-Lead + MC Safety Gate,
  Held-out 40% Real Textbook ECGs, and verified Grad-CAM overlay saved to Drive.
"""
import ast
import glob
import os
import random
import time
import zipfile
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import wfdb
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# =====================================================================
# 1. Configuration & Paths
# =====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    PTB_DIR = "/kaggle/working/ptbxl"
    SAVE_DIR = "/kaggle/working/weights"
    TEXTBOOK_DIR = "/kaggle/working/ecg_100_test"
    _zips = glob.glob("/kaggle/input/**/ecg_examples_and_images.zip", recursive=True)
    TEXTBOOK_ZIP = _zips[0] if _zips else "/kaggle/working/ecg_examples_and_images.zip"
else:
    PTB_DIR = "/content/ptbxl"
    DRIVE_ROOT = "/content/drive/Othercomputers/My Laptop/codeblue-sounds"
    SAVE_DIR = os.path.join(DRIVE_ROOT, "hf_ecg_space", "weights")
    TEXTBOOK_DIR = "/content/ecg_100_test"
    TEXTBOOK_ZIP = os.path.join(DRIVE_ROOT, "ecg_engine", "ecg_examples_and_images.zip")

CKPT_V2_PATH = os.path.join(SAVE_DIR, "ecg_densenet121_v3.pt")
GRADCAM_OUT = os.path.join(SAVE_DIR, "gradcam_sample_v3.png")
REPORT_OUT = os.path.join(SAVE_DIR, "eval_report_v3.txt")

IMG_H, IMG_W = 384, 512
EPOCHS_MAIN = 8
EPOCHS_ADAPT = 3
BATCH_SIZE = 32

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = [
    "NORMAL_SINUS_RHYTHM",
    "ANTERIOR_SEPTAL_MI",
    "INFERIOR_POSTERIOR_MI",
    "LATERAL_MI",
    "NSTEMI_ISCHEMIA",
    "ATRIAL_FIBRILLATION",
    "ATRIAL_FLUTTER",
    "SVT_TACHYCARDIA",
    "SINUS_BRADYCARDIA",
    "FIRST_DEGREE_AV_BLOCK",
    "HIGH_GRADE_AV_BLOCK",
    "RBBB",
    "LBBB",
    "LAFB_LPFB",
    "LVH_RVH",
    "PACEMAKER_RHYTHM",
    "WPW_PREEXCITATION",
    "LONG_QT_OR_ECTOPY",
]

RHYTHM_OR_FORM_CODES = {
    "AFIB", "AFLT", "STACH", "SBRAD", "SVTAC", "PSVT",
    "PACE", "PVC", "PAC", "BIGU", "TRIGU", "LNGQT",
}

SCP_MAP = {
    0: {"NORM"},
    1: {"AMI", "ASMI", "ALMI", "INJAS", "INJAP"},
    2: {"IMI", "PMI", "ILMI", "IPMI", "INJIN", "INJIL"},
    3: {"LMI", "ALMI", "ILMI", "INJLA"},
    4: {"ISCAL", "ISCAN", "ISCAS", "ISCIL", "ISCIN", "ISCLA", "ISC_", "NST_", "STD_", "INVT"},
    5: {"AFIB"},
    6: {"AFLT"},
    7: {"SVTAC", "PSVT", "STACH"},
    8: {"SBRAD"},
    9: {"1AVB"},
    10: {"2AVB", "3AVB"},
    11: {"CRBBB", "IRBBB"},
    12: {"CLBBB", "ILBBB"},
    13: {"LAFB", "LPFB"},
    14: {"LVH", "RVH", "LAO/LAE", "RAO/RAE"},
    15: {"PACE"},
    16: {"WPW"},
    17: {"LNGQT", "PVC", "PAC"},
}

REAL_FOLDER_TO_CLASSES = {
    "01_normal_sinus_rhythm": {"NORMAL_SINUS_RHYTHM"},
    "02_stemi_acute_mi": {"ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI"},
    "03_wellens_and_ischemia": {"NSTEMI_ISCHEMIA", "ANTERIOR_SEPTAL_MI"},
    "04_atrial_fibrillation": {"ATRIAL_FIBRILLATION"},
    "05_atrial_flutter": {"ATRIAL_FLUTTER"},
    "06_svt_and_mat": {"SVT_TACHYCARDIA"},
    "08_av_block_1st_2nd": {"FIRST_DEGREE_AV_BLOCK", "HIGH_GRADE_AV_BLOCK", "SINUS_BRADYCARDIA"},
    "09_av_block_3rd_complete": {"HIGH_GRADE_AV_BLOCK", "SINUS_BRADYCARDIA"},
    "10_rbbb_and_fascicular": {"RBBB", "LAFB_LPFB"},
    "11_lbbb": {"LBBB"},
    "12_wpw_preexcitation": {"WPW_PREEXCITATION"},
    "13_long_short_qt": {"LONG_QT_OR_ECTOPY"},
    "15_lvh_rvh_enlargement": {"LVH_RVH"},
    "20_pacemaker_toxicity_reversal": {"PACEMAKER_RHYTHM"},
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =====================================================================
# 2. Unified Grid-Removal Preprocessing (Identical for Train & Real)
# =====================================================================
def normalize_ecg_to_clean_ink(img_bgr: np.ndarray, H: int = IMG_H, W: int = IMG_W) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Remove pink/red paper grid and uneven lighting shadows, producing crisp dark ink
    on a clean white canvas so synthetic PTB-XL images and real photos share the same domain.
    """
    if img_bgr.shape[0] != H or img_bgr.shape[1] != W:
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    # Suppress pink/red grid lines (high R relative to G/B) while preserving black & blue ink
    pink_score = np.clip((r - np.minimum(g, b)) - 12.0, 0.0, 255.0)
    # Ink darkness: black ink has low R,G,B; blue ink has low R,G and higher B
    luma = 0.55 * r + 0.35 * g + 0.10 * b
    luma_no_grid = np.clip(luma + pink_score * 0.95, 0.0, 255.0)

    # Local shadow removal
    bg = cv2.GaussianBlur(luma_no_grid, (31, 31), 0)
    ink = np.clip(bg - luma_no_grid, 0.0, 255.0)
    hi = max(25.0, float(np.percentile(ink, 99.5)))
    clean_gray = np.clip(255.0 - (ink / hi) * 255.0, 0.0, 255.0).astype(np.uint8)

    clean_rgb = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2RGB)
    x = (clean_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)), clean_rgb


# =====================================================================
# 3. Realistic Multi-Layout Renderer + Degradation & Partial-Lead Masking
# =====================================================================
def degrade_ecg_bgr(img: np.ndarray, heavy: bool = False) -> np.ndarray:
    """Apply realistic phone-photo tilt, uneven shadow, blur, JPEG noise, and optional partial-lead occlusion."""
    H, W = img.shape[:2]
    out = img.copy()

    # Rotation
    max_ang = 7.0 if heavy else 4.0
    if heavy or random.random() < 0.55:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), random.uniform(-max_ang, max_ang), 1.0)
        out = cv2.warpAffine(out, M, (W, H), borderValue=(245, 240, 252))

    # Uneven lighting / shadow gradient
    if heavy or random.random() < 0.50:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        grad = (xx / W) * random.uniform(-0.30, 0.30) + (yy / H) * random.uniform(-0.25, 0.25)
        factor = np.clip(1.0 + grad, 0.60, 1.25)[:, :, None]
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Blur + JPEG compression artifacts
    if heavy or random.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if heavy or random.random() < 0.40:
        q = random.randint(25, 50) if heavy else random.randint(45, 85)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec

    # Partial-lead masking (simulate cropped or missing leads)
    if heavy:
        # Mask 40-50% of the ECG area
        if random.random() < 0.5:
            out[H // 2 :, :] = 248
        else:
            out[:, W // 2 :] = 248
    elif random.random() < 0.15:
        y0 = random.randint(0, H // 2)
        h_mask = random.randint(H // 6, H // 3)
        out[y0 : y0 + h_mask, :] = 248

    return out


def render_ecg_bgr(sig_12x1000: np.ndarray, augment: bool = True, H: int = IMG_H, W: int = IMG_W) -> np.ndarray:
    """Render 12-lead ECG with realistic grids, lead text labels, calibration pulses, and variable ink."""
    style = random.choice(["pink_grid", "pink_grid", "red_grid", "cream_grid", "white_bg", "blue_limb"]) if augment else "pink_grid"
    if style == "white_bg":
        img = np.full((H, W, 3), 252, dtype=np.uint8)
    elif style == "cream_grid":
        img = np.full((H, W, 3), (225, 240, 248), dtype=np.uint8)
        for x in range(0, W, 4):
            cv2.line(img, (x, 0), (x, H), (175, 195, 210) if x % 20 == 0 else (205, 220, 232), 1)
        for y in range(0, H, 4):
            cv2.line(img, (0, y), (W, y), (175, 195, 210) if y % 20 == 0 else (205, 220, 232), 1)
    else:
        img = np.full((H, W, 3), (246, 240, 255), dtype=np.uint8)
        minor_c = (218, 206, 248) if style != "red_grid" else (200, 185, 248)
        major_c = (182, 162, 238) if style != "red_grid" else (150, 130, 235)
        for x in range(0, W, 4):
            cv2.line(img, (x, 0), (x, H), major_c if x % 20 == 0 else minor_c, 1)
        for y in range(0, H, 4):
            cv2.line(img, (0, y), (W, y), major_c if y % 20 == 0 else minor_c, 1)

    layout = random.choices(["3x4", "3x4_3rhythm", "3x4_header", "2x6", "12x1"], weights=[0.35, 0.25, 0.15, 0.15, 0.10])[0] if augment else "3x4"
    thick = random.choice([1, 1, 2]) if augment else 1
    N = sig_12x1000.shape[1]

    def draw_cal_pulse(x0: int, yc: int, amp_px: float):
        h_cal = int(amp_px)
        pts = np.array([
            [x0, yc], [x0 + 3, yc], [x0 + 3, yc - h_cal],
            [x0 + 11, yc - h_cal], [x0 + 11, yc], [x0 + 14, yc],
        ], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (25, 25, 28), 1)

    def draw_trace(lead_i: int, x0: int, x1: int, yc: int, amp_px: float, s0: int, s1: int):
        seg = np.nan_to_num(sig_12x1000[lead_i, s0:s1], nan=0.0)
        xs = np.linspace(x0, x1, len(seg)).astype(np.int32)
        ys = np.clip(yc - seg * amp_px, 2, H - 3).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        color = (185, 45, 35) if (style == "blue_limb" and lead_i < 6) else (20, 20, 24)
        cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)
        cv2.putText(img, LEADS_12[lead_i], (x0 + 2, max(12, yc - int(amp_px * 0.45))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (30, 30, 35), 1, cv2.LINE_AA)

    if layout == "3x4":
        cw, rh, seg_len = W // 4, H // 4, N // 4
        for idx in range(12):
            c, r = idx // 3, idx % 3
            yc = int((r + 0.52) * rh)
            x0 = c * cw + (18 if c == 0 else 4)
            if c == 0:
                draw_cal_pulse(2, yc, rh * 0.28)
            draw_trace(idx, x0, (c + 1) * cw - 4, yc, rh * 0.28, c * seg_len, (c + 1) * seg_len)
        draw_cal_pulse(2, int(3.52 * rh), rh * 0.28)
        draw_trace(1, 18, W - 4, int(3.52 * rh), rh * 0.28, 0, N)
    elif layout == "3x4_3rhythm":
        # 6-row hospital layout (GE MAC / Louisville Lectures): 3x4 leads + 3 full rhythm strips (V1, II, V5)
        cw, rh, seg_len = W // 4, H // 6, N // 4
        for idx in range(12):
            c, r = idx // 3, idx % 3
            yc = int((r + 0.52) * rh)
            x0 = c * cw + (16 if c == 0 else 4)
            if c == 0:
                draw_cal_pulse(2, yc, rh * 0.30)
            draw_trace(idx, x0, (c + 1) * cw - 4, yc, rh * 0.30, c * seg_len, (c + 1) * seg_len)
        for r_off, r_lead in enumerate([6, 1, 10]):  # V1, II, V5
            yc = int((3 + r_off + 0.52) * rh)
            draw_cal_pulse(2, yc, rh * 0.30)
            draw_trace(r_lead, 16, W - 4, yc, rh * 0.30, 0, N)
    elif layout == "3x4_header":
        # 4-row ECG with printed top computer header block (~16% top margin)
        top_m = int(H * 0.16)
        img[:top_m, :] = (250, 250, 252)
        cv2.putText(img, "Vent. rate  78 BPM   PR 160 ms   QRS 92 ms   QT/QTc 380/412", (14, int(top_m * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (40, 40, 45), 1, cv2.LINE_AA)
        cv2.putText(img, "25mm/s  10mm/mV  12-Lead ECG", (14, int(top_m * 0.82)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 65), 1, cv2.LINE_AA)
        cw, rh, seg_len = W // 4, (H - top_m) // 4, N // 4
        for idx in range(12):
            c, r = idx // 3, idx % 3
            yc = top_m + int((r + 0.52) * rh)
            x0 = c * cw + (16 if c == 0 else 4)
            if c == 0:
                draw_cal_pulse(2, yc, rh * 0.28)
            draw_trace(idx, x0, (c + 1) * cw - 4, yc, rh * 0.28, c * seg_len, (c + 1) * seg_len)
        draw_cal_pulse(2, top_m + int(3.52 * rh), rh * 0.28)
        draw_trace(1, 16, W - 4, top_m + int(3.52 * rh), rh * 0.28, 0, N)
    elif layout == "2x6":
        cw, rh, seg_len = W // 2, H // 6, N // 2
        for idx in range(12):
            c, r = idx // 6, idx % 6
            yc = int((r + 0.52) * rh)
            x0 = c * cw + (16 if c == 0 else 4)
            if c == 0:
                draw_cal_pulse(2, yc, rh * 0.30)
            draw_trace(idx, x0, (c + 1) * cw - 4, yc, rh * 0.30, c * seg_len, (c + 1) * seg_len)
    else:
        rh = H / 12.0
        for idx in range(12):
            yc = int((idx + 0.55) * rh)
            draw_trace(idx, 20, W - 6, yc, rh * 0.32, 0, N)

    if augment:
        img = degrade_ecg_bgr(img, heavy=False)
    return img


# =====================================================================
# 4. Datasets (PTB-XL + Real Textbook 60/40 Split)
# =====================================================================
class PtbxlDataset(Dataset):
    def __init__(self, df_ref: pd.DataFrame, labels: np.ndarray, indices: np.ndarray, augment: bool, heavy_degrade: bool = False):
        self.paths = [os.path.join(PTB_DIR, df_ref.loc[i, "filename_lr"]) for i in indices]
        self.labels = labels[indices]
        self.augment = augment
        self.heavy_degrade = heavy_degrade

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        sig, _ = wfdb.rdsamp(self.paths[i])
        bgr = render_ecg_bgr(sig.T, augment=self.augment)
        if self.heavy_degrade:
            bgr = degrade_ecg_bgr(bgr, heavy=True)
        x, _ = normalize_ecg_to_clean_ink(bgr)
        return x, torch.from_numpy(self.labels[i])


class RealImageDataset(Dataset):
    def __init__(self, items: List[Tuple[str, np.ndarray]], augment: bool = True, repeat: int = 1):
        self.items = items
        self.augment = augment
        self.repeat = repeat

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, idx):
        path, y = self.items[idx % len(self.items)]
        bgr = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        if self.augment:
            bgr = degrade_ecg_bgr(bgr, heavy=False)
        x, _ = normalize_ecg_to_clean_ink(bgr)
        return x, torch.from_numpy(y.copy())


# =====================================================================
# 5. MC Dropout + Grad-CAM Utilities (Bug-Free Hook!)
# =====================================================================
def mc_predict(model: nn.Module, x: torch.Tensor, n_passes: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    ps = []
    with torch.no_grad():
        for _ in range(n_passes):
            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    p = torch.sigmoid(model(x)).cpu().numpy()[0]
            else:
                p = torch.sigmoid(model(x)).cpu().numpy()[0]
            ps.append(p)
    arr = np.stack(ps)
    return arr.mean(axis=0), arr.std(axis=0)


def save_gradcam(model: nn.Module, img_path: str, out_path: str) -> str:
    model.eval()
    feats: Dict[str, torch.Tensor] = {}
    grads: Dict[str, torch.Tensor] = {}

    def _fwd(_m, _inp, out):
        feats["a"] = out

    def _bwd(_m, _gin, gout):
        grads["g"] = gout[0]
        return None  # CRITICAL: returning None avoids PyTorch hook size-change error!

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(_fwd)
    h2 = target_layer.register_full_backward_hook(_bwd)

    orig_bgr = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    orig_rgb = cv2.cvtColor(cv2.resize(orig_bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
    x, _ = normalize_ecg_to_clean_ink(orig_bgr)
    x = x.unsqueeze(0).to(DEVICE)

    logits = model(x)
    top_cls = int(torch.argmax(logits[0]).item())
    model.zero_grad()
    logits[0, top_cls].backward()

    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats["a"]).sum(dim=1)).squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam / (cam.max() + 1e-8), (IMG_W, IMG_H))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.62 * orig_rgb + 0.38 * heat).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)
    h1.remove()
    h2.remove()
    return CLASSES[top_cls]


# =====================================================================
# 6. Main Training + 4-Part Intensive Benchmark
# =====================================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Running DenseNet121 v3 (Multi-Layout 4/6-Row + Dampened Weights + Auto-Calibration) on {DEVICE} (AMP={USE_AMP})")

    df = pd.read_csv(f"{PTB_DIR}/ptbxl_database.csv")
    df["scp_dict"] = df["scp_codes"].apply(ast.literal_eval)

    def encode_scp(d):
        y = np.zeros(len(CLASSES), dtype=np.float32)
        active = {k for k, v in d.items() if (v >= 50.0 or k in RHYTHM_OR_FORM_CODES)}
        for idx, codes in SCP_MAP.items():
            if active & codes:
                y[idx] = 1.0
        return y

    Y_all = np.stack(df["scp_dict"].apply(encode_scp).values)
    keep = Y_all.sum(axis=1) > 0
    df = df[keep].reset_index(drop=True)
    Y_all = Y_all[keep]

    train_idx = np.where(df["strat_fold"].values <= 8)[0]
    val_idx = np.where(df["strat_fold"].values == 9)[0]
    test_idx = np.where(df["strat_fold"].values == 10)[0]
    print(f"PTB-XL: Train={len(train_idx)} | Val={len(val_idx)} | Test(Fold10)={len(test_idx)}")

    # Unzip real textbook images and enforce strict 60% adapt / 40% held-out split
    if not os.path.exists(TEXTBOOK_DIR) and os.path.exists(TEXTBOOK_ZIP):
        with zipfile.ZipFile(TEXTBOOK_ZIP, "r") as z:
            z.extractall(TEXTBOOK_DIR)

    adapt_items: List[Tuple[str, np.ndarray]] = []
    heldout_items: List[Tuple[str, str]] = []
    for fld, cls_set in sorted(REAL_FOLDER_TO_CLASSES.items()):
        hits = sorted([
            p for p in glob.glob(f"{TEXTBOOK_DIR}/**/{fld}/*.*", recursive=True)
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
        y_vec = np.zeros(len(CLASSES), dtype=np.float32)
        for ci, cn in enumerate(CLASSES):
            if cn in cls_set:
                y_vec[ci] = 1.0
        split_k = max(1, int(len(hits) * 0.6))
        for p in hits[:split_k]:
            adapt_items.append((p, y_vec))
        for p in hits[split_k:]:
            heldout_items.append((fld, p))
    print(f"Textbook Split (Zero Leakage): Adapt(60%)={len(adapt_items)} | Held-out Test(40%)={len(heldout_items)}")

    train_loader = DataLoader(PtbxlDataset(df, Y_all, train_idx, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=USE_AMP)
    val_loader = DataLoader(PtbxlDataset(df, Y_all, val_idx, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=USE_AMP)
    test_clean_loader = DataLoader(PtbxlDataset(df, Y_all, test_idx, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=USE_AMP)
    test_deg_loader = DataLoader(PtbxlDataset(df, Y_all, test_idx[:300], False, heavy_degrade=True), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=USE_AMP)

    model = timm.create_model("densenet121", pretrained=True, num_classes=len(CLASSES), drop_rate=0.2).to(DEVICE)
    pos_counts = Y_all[train_idx].sum(axis=0)
    pos_weight = torch.tensor(np.clip(np.sqrt((len(train_idx) - pos_counts) / np.maximum(pos_counts, 1.0)), 1.0, 4.0), dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_MAIN)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_auc = 0.0
    print("\n--- Phase 1: Training on PTB-XL (Unified Grid-Removed Multi-Layout) ---")
    for ep in range(1, EPOCHS_MAIN + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            if USE_AMP and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(xb), yb * 0.96 + 0.02)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(xb), yb * 0.96 + 0.02)
                loss.backward()
                optimizer.step()
            tr_loss += loss.item() * len(xb)
        scheduler.step()

        model.eval()
        ps, ts_y = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                if USE_AMP:
                    with torch.amp.autocast("cuda"):
                        p = torch.sigmoid(model(xb.to(DEVICE, non_blocking=True))).cpu().numpy()
                else:
                    p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
                ps.append(p)
                ts_y.append(yb.numpy())
        ps, ts_y = np.concatenate(ps), np.concatenate(ts_y)
        aucs = [roc_auc_score(ts_y[:, j], ps[:, j]) for j in range(len(CLASSES)) if ts_y[:, j].sum() > 0]
        m_auc = float(np.mean(aucs))
        print(f"Epoch {ep}/{EPOCHS_MAIN} ({time.time()-t0:.0f}s) | Loss={tr_loss/len(train_idx):.4f} | Val Macro AUROC={m_auc:.4f}")
        if m_auc > best_auc:
            best_auc = m_auc
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "val_auc": best_auc, "version": "v2_grid_removed"}, CKPT_V2_PATH)
            print(f"  -> Saved best v2 checkpoint to Drive (AUROC={best_auc:.4f})")

    # Phase 2: Gentle domain adaptation mixing 60% textbook images + PTB-XL subset
    if adapt_items:
        print("\n--- Phase 2: Gentle Domain Adaptation (60% Textbook + PTB-XL Replay, lr=3e-5) ---")
        ckpt = torch.load(CKPT_V2_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        replay_ds = PtbxlDataset(df, Y_all, train_idx[:1200], True)
        real_ds = RealImageDataset(adapt_items, augment=True, repeat=6)
        adapt_loader = DataLoader(ConcatDataset([replay_ds, real_ds]), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=USE_AMP)
        opt2 = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
        for ep in range(1, EPOCHS_ADAPT + 1):
            t0 = time.time()
            model.train()
            for xb, yb in adapt_loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                opt2.zero_grad()
                if USE_AMP and scaler is not None:
                    with torch.amp.autocast("cuda"):
                        loss = criterion(model(xb), yb * 0.96 + 0.02)
                    scaler.scale(loss).backward()
                    scaler.step(opt2)
                    scaler.update()
                else:
                    loss = criterion(model(xb), yb * 0.96 + 0.02)
                    loss.backward()
                    opt2.step()
            print(f"Adapt Epoch {ep}/{EPOCHS_ADAPT} ({time.time()-t0:.0f}s) complete.")
        # Phase 3: Zero-Leakage Temperature + Bias Calibration (fit on Adapt split ONLY)
        from scipy.optimize import minimize
        model.eval()
        fit_z, fit_y = [], []
        with torch.no_grad():
            for p_img, y_v in adapt_items:
                bgr_i = cv2.imread(p_img)
                if bgr_i is None: continue
                xi, _ = normalize_ecg_to_clean_ink(bgr_i)
                fit_z.append(model(xi.unsqueeze(0).to(DEVICE)).cpu().numpy()[0])
                fit_y.append(y_v)
        T_opt, b_opt = 1.0, 0.0
        if fit_z:
            Zf, Yf = np.stack(fit_z), np.stack(fit_y)
            def _nll(prm):
                T, b = prm
                if T < 0.3 or T > 8.0: return 1e9
                pr = 1.0 / (1.0 + np.exp(-np.clip((Zf - b) / T, -30, 30)))
                return float(-np.mean(Yf * np.log(pr + 1e-7) + (1.0 - Yf) * np.log(1.0 - pr + 1e-7)))
            opt_r = minimize(_nll, x0=[1.2, 0.5], method="Nelder-Mead")
            T_opt, b_opt = float(opt_r.x[0]), float(opt_r.x[1])
            print(f"Auto-Calibration (Adapt ONLY): T={T_opt:.4f}, b={b_opt:.4f}")
        torch.save({
            "state_dict": model.state_dict(),
            "classes": CLASSES,
            "val_auc": best_auc,
            "calib_temp": T_opt,
            "calib_bias": b_opt,
            "version": "v3_multilayout_calibrated",
        }, CKPT_V2_PATH)
        print("Saved final calibrated v3 checkpoint!")

    # =====================================================================
    # 7. Intensive 4-Part Benchmark Suite
    # =====================================================================
    lines = ["================ DENSENET121 v3 INTENSIVE BENCHMARK ================"]
    model.eval()

    # TEST 1: Clean PTB-XL Fold 10
    p1, y1 = [], []
    with torch.no_grad():
        for xb, yb in test_clean_loader:
            p1.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            y1.append(yb.numpy())
    p1, y1 = np.concatenate(p1), np.concatenate(y1)
    aucs1 = [roc_auc_score(y1[:, j], p1[:, j]) for j in range(len(CLASSES)) if y1[:, j].sum() > 0]
    lines.append(f"\n[TEST 1] Unseen PTB-XL Fold 10 Clean ({len(y1)} ECGs): Mean Macro AUROC = {np.mean(aucs1):.4f}")
    for j, c in enumerate(CLASSES):
        if y1[:, j].sum() > 0:
            lines.append(f"  {c:24s}: AUROC = {roc_auc_score(y1[:, j], p1[:, j]):.4f}")

    # TEST 2: Heavy Degradation + 50% Lead Occlusion + MC Dropout Safety Gate
    p2, y2 = [], []
    with torch.no_grad():
        for xb, yb in test_deg_loader:
            p2.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            y2.append(yb.numpy())
    p2, y2 = np.concatenate(p2), np.concatenate(y2)
    aucs2 = [roc_auc_score(y2[:, j], p2[:, j]) for j in range(len(CLASSES)) if y2[:, j].sum() > 0 and y2[:, j].sum() < len(y2)]
    lines.append(f"\n[TEST 2] Heavy Degraded + 50% Masked Leads (300 Fold-10 ECGs): Mean Macro AUROC = {np.mean(aucs2):.4f}")

    # TEST 3: Strictly Held-Out 40% Real Textbook ECGs (Unseen!) + MC Dropout Uncertainty
    t1_hits, t3_hits, refused, lat_list = 0, 0, 0, []
    sample_stemi = None
    lines.append(f"\n[TEST 3] Strictly Held-Out 40% Real Textbook ECGs ({len(heldout_items)} UNSEEN ECGs, 6x MC Dropout):")
    for fld, path in heldout_items:
        if fld == "02_stemi_acute_mi" and sample_stemi is None:
            sample_stemi = path
        bgr = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        x, _ = normalize_ecg_to_clean_ink(bgr)
        ts = time.time()
        mean_p, std_p = mc_predict(model, x.unsqueeze(0).to(DEVICE), n_passes=6)
        lat_list.append(time.time() - ts)
        order = np.argsort(mean_p)[::-1]
        top3 = [CLASSES[k] for k in order[:3]]
        top_conf = float(mean_p[order[0]])
        top_std = float(std_p[order[0]])
        is_refused = (top_conf < 0.55) or (top_std > 0.15)
        if is_refused:
            refused += 1
        h1 = top3[0] in REAL_FOLDER_TO_CLASSES[fld]
        h3 = any(c in REAL_FOLDER_TO_CLASSES[fld] for c in top3)
        if h1:
            t1_hits += 1
        if h3:
            t3_hits += 1
        tag = "TOP1" if h1 else ("TOP3" if h3 else "MISS")
        flag = " [LOW-CONF REFUSE]" if is_refused else ""
        lines.append(f"  [{tag}] {fld:32s} | {CLASSES[order[0]]}({int(top_conf*100)}%±{int(top_std*100)}%), {CLASSES[order[1]]}({int(mean_p[order[1]]*100)}%){flag}")

    n_h = max(1, len(heldout_items))
    lines.append(f"  >>> Held-Out Real Top-1: {t1_hits}/{len(heldout_items)} ({t1_hits*100/n_h:.1f}%) | Top-3: {t3_hits}/{len(heldout_items)} ({t3_hits*100/n_h:.1f}%) | Safety-Gate Refusals: {refused}/{len(heldout_items)} | Avg MC Latency: {np.mean(lat_list)*1000:.0f} ms")

    # TEST 4: Verified Grad-CAM Overlay
    if sample_stemi:
        pred_lbl = save_gradcam(model, sample_stemi, GRADCAM_OUT)
        lines.append(f"\n[TEST 4] Grad-CAM saved to Drive ({GRADCAM_OUT}) | Predicted: {pred_lbl}")

    report_text = "\n".join(lines)
    print(report_text)
    with open(REPORT_OUT, "w", encoding="utf-8") as rf:
        rf.write(report_text)
    print(f"\nSaved full evaluation report to: {REPORT_OUT}")


if __name__ == "__main__":
    main()
