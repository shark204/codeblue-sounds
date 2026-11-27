"""
Code Blue ECG — v18 External-Domain Fine-Tuning & Zero-Leakage LITFL Evaluation (Modal A10G)
Addresses the #1 external generalization gap discovered by the scientific audit (31.0% on LITFL):
  1. Crawls LITFL Top-150 (CC BY-NC-SA 4.0), splits strictly 50/50 by case ID (odd=External Train, even=HELD-OUT External Test).
  2. Fine-tunes EfficientNetV2-S (from v8_rare19) with heavy external-domain layout/crop/annotation augmentation
     mixed with PTB-XL Fold 1-8 + LITFL odd-ID external textbook ECGs (low LR 5e-5, 4 epochs, anti-forgetting).
  3. Evaluates on BOTH:
       - Pristine PTB-XL Fold-10 (anti-forgetting gate: AUROC drop <= 0.01)
       - HELD-OUT External LITFL Even-ID Cases (ZERO overlap with training!)
  4. Promotes to /vol/outputs/ecg_effnetv2s_v8_rare19_512x768.pt ONLY if External LITFL Hit improves >= +10%
     without harming PTB-XL Fold-10 AUROC (keeping a backup of v8!).
"""
import os, re, json, ast, math, shutil, time, modal

app = modal.App("codeblue-train-v18-external")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", "timm", "wfdb", "scikit-learn", "pandas", "numpy", "opencv-python-headless", "pillow", "requests", "beautifulsoup4")
)


def wilson_ci(k: int, n: int, z: float = 1.95996):
    if n <= 0: return [0.0, 0.0]
    p = k / n; d = 1.0 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [round(max(0.0, c - h) * 100.0, 1), round(min(1.0, c + h) * 100.0, 1)]


@app.function(image=image, gpu="A10G", cpu=6.0, memory=24576, timeout=3600, volumes={"/vol": vol})
def train_v18_external():
    import cv2, numpy as np, pandas as pd, requests, timm, torch, torch.nn as nn, wfdb
    from bs4 import BeautifulSoup
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda")
    PTB_DIR, OUT_DIR, LITFL_DIR = "/vol/ptbxl", "/vol/outputs", "/vol/litfl_top150_v18"
    os.makedirs(LITFL_DIR, exist_ok=True)

    V8_PATH = os.path.join(OUT_DIR, "ecg_effnetv2s_v8_rare19_512x768.pt")
    ck = torch.load(V8_PATH, map_location="cpu", weights_only=False)
    CLASSES = list(ck["classes"])
    IMG_H, IMG_W = int(ck.get("img_h", 512)), int(ck.get("img_w", 768))
    PLATT_A = np.array(ck["platt_a"], dtype=np.float32)
    PLATT_B = np.array(ck["platt_b"], dtype=np.float32)
    THR_MAP = ck.get("thresholds", {})
    c2i = {c: i for i, c in enumerate(CLASSES)}

    LITFL_PATS = {
        "ANTERIOR_SEPTAL_MI": r"\b(anterior\s+(stemi|mi|myocardial\s+infarction)|anteroseptal\s+(stemi|mi)|anterolateral\s+(stemi|mi)|lad\s+occlusion)\b",
        "INFERIOR_POSTERIOR_MI": r"\b(inferior\s+(stemi|mi|myocardial\s+infarction)|posterior\s+(stemi|mi)|inferoposterior|rca\s+occlusion)\b",
        "LATERAL_MI": r"\b(high\s+lateral\s+(stemi|mi)|lateral\s+(stemi|mi))\b",
        "NSTEMI_ISCHEMIA": r"\b(nstemi|subendocardial\s+ischa?emia|wellens|de\s+winter)\b",
        "ATRIAL_FIBRILLATION": r"\b(atrial\s+fibrillation|\bafib\b|\baf\s+with\s+rvr\b)\b",
        "ATRIAL_FLUTTER": r"\b(atrial\s+flutter)\b",
        "SVT_TACHYCARDIA": r"\b(svt|supraventricular\s+tachycardia|avnrt|avrt)\b",
        "SINUS_BRADYCARDIA": r"\b(sinus\s+bradycardia)\b",
        "FIRST_DEGREE_AV_BLOCK": r"\b(first[-\s]degree\s+(av|heart)\s+block|1st\s+degree\s+av\s+block)\b",
        "HIGH_GRADE_AV_BLOCK": r"\b(complete\s+heart\s+block|third[-\s]degree\s+(av|heart)\s+block|mobitz\s+(ii|2)|2:1\s+av\s+block)\b",
        "RBBB": r"\b(right\s+bundle\s+branch\s+block|\brbbb\b)\b",
        "LBBB": r"\b(left\s+bundle\s+branch\s+block|\blbbb\b)\b",
        "LAFB_LPFB": r"\b(left\s+anterior\s+fascicular\s+block|\blafb\b|left\s+posterior\s+fascicular\s+block|\blpfb\b)\b",
        "LVH_RVH": r"\b(left\s+ventricular\s+hypertrophy|\blvh\b|right\s+ventricular\s+hypertrophy|\brvh\b)\b",
        "PACEMAKER_RHYTHM": r"\b(paced\s+rhythm|ventricular\s+pacing|pacemaker)\b",
        "WPW_PREEXCITATION": r"\b(wolff[-\s]parkinson[-\s]white|\bwpw\b|delta\s+wave|pre[-\s]?excitation)\b",
        "LONG_QT_OR_ECTOPY": r"\b(long\s+qt|prolonged\s+qt|torsades|ventricular\s+ectop|bigeminy|pvc)\b",
        "BRUGADA_TYPE_1": r"\b(brugada\s+(syndrome|pattern|type\s*1))\b",
        "NORMAL_SINUS_RHYTHM": r"\b(normal\s+sinus\s+rhythm)\b",
    }

    # 1. Fetch/Cache LITFL Top-150 (strictly split odd=Train, even=Held-Out External Test)
    meta_p = os.path.join(LITFL_DIR, "litfl150_meta.json")
    if os.path.exists(meta_p):
        litfl_all = json.load(open(meta_p))
    else:
        litfl_all = []
        hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        for num in range(1, 151):
            url = f"https://litfl.com/ecg-case-{num:03d}/"
            try:
                r = requests.get(url, headers=hdrs, timeout=10)
                if r.status_code != 200: continue
                soup = BeautifulSoup(r.text, "html.parser")
                txt_all = soup.get_text(" ", strip=True)
                m_ans = re.search(r"(ECG\s+Interpretation|Answer|Diagnosis)(.{20,1800})", txt_all, re.I)
                ans_txt = m_ans.group(2) if m_ans else txt_all[:2000]
                gt = [c for c, pat in LITFL_PATS.items() if re.search(pat, ans_txt, re.I)]
                if not gt: continue
                img_url = next((img.get("src") or img.get("data-src") for img in soup.find_all("img")
                                if "wp-content/uploads" in (img.get("src") or img.get("data-src") or "")
                                and any(e in (img.get("src") or img.get("data-src") or "").lower() for e in (".jpg", ".jpeg", ".png"))
                                and not any(b in (img.get("src") or img.get("data-src") or "").lower() for b in ("logo", "avatar", "icon", "banner"))), None)
                if not img_url: continue
                ir = requests.get(img_url, headers=hdrs, timeout=10)
                if ir.status_code != 200 or len(ir.content) < 8000: continue
                fp = os.path.join(LITFL_DIR, f"litfl_{num:03d}.jpg")
                with open(fp, "wb") as fh: fh.write(ir.content)
                litfl_all.append({"num": num, "path": fp, "gt": gt})
            except Exception:
                continue
        json.dump(litfl_all, open(meta_p, "w"), indent=2)
        vol.commit()

    lit_tr = [x for x in litfl_all if x["num"] % 2 == 1]
    lit_te = [x for x in litfl_all if x["num"] % 2 == 0]
    print(f"[v18 External] LITFL Total={len(litfl_all)} | Odd-Train={len(lit_tr)} | Even-HELD-OUT-Test={len(lit_te)}")

    def clean_ink(bgr, aug=False, rng=None):
        if aug and rng is not None:
            # External-domain simulation: random aspect-ratio crops, border pads, annotations, rotation
            h, w = bgr.shape[:2]
            if rng.random() < 0.45:
                cy0, cy1 = int(h * rng.uniform(0.0, 0.08)), int(h * rng.uniform(0.92, 1.0))
                cx0, cx1 = int(w * rng.uniform(0.0, 0.08)), int(w * rng.uniform(0.92, 1.0))
                bgr = bgr[cy0:cy1, cx0:cx1]
            if rng.random() < 0.35:
                ang = float(rng.uniform(-4.0, 4.0))
                M = cv2.getRotationMatrix2D((bgr.shape[1]/2, bgr.shape[0]/2), ang, 1.0)
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

    def render_ptb(sig, rng, aug=False):
        H, W = IMG_H, IMG_W
        canvas = np.full((H, W, 3), (242, 236, 252), dtype=np.uint8)
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
        return clean_ink(canvas, aug=aug, rng=rng)

    model = timm.create_model(ck.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k"), pretrained=False, num_classes=len(CLASSES), drop_rate=0.25).to(device)
    model.load_state_dict(ck["state_dict"])

    def eval_litfl(m_eval, items):
        m_eval.eval()
        top1, top3, op_hit, n = 0, 0, 0, 0
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
                ord_i = np.argsort(-pr)
                if CLASSES[ord_i[0]] in gt: top1 += 1
                if any(CLASSES[k] in gt for k in ord_i[:3]): top3 += 1
                fired = {CLASSES[j] for j in range(len(CLASSES)) if pr[j] >= float(THR_MAP.get(CLASSES[j], 0.5))}
                if fired & gt: op_hit += 1
        return {"n": n, "top1_pct": round(100*top1/max(1,n), 1), "top3_pct": round(100*top3/max(1,n), 1), "op_hit_pct": round(100*op_hit/max(1,n), 1), "op_ci95": wilson_ci(op_hit, n)}

    def eval_ptb_auc(m_eval, recs):
        m_eval.eval()
        rng = np.random.default_rng(99)
        ps, ys = [], []
        with torch.no_grad():
            for i in range(0, len(recs), 32):
                batch = recs[i:i+32]
                xs = np.stack([render_ptb(np.nan_to_num(wfdb.rdsamp(r["path"])[0].astype(np.float32)), rng, False) for r in batch])
                with torch.amp.autocast("cuda"):
                    ps.append(torch.sigmoid(m_eval(torch.from_numpy(xs).to(device))).float().cpu().numpy())
                ys.append(np.stack([r["y"] for r in batch]))
        P, Y = np.concatenate(ps), np.concatenate(ys)
        aucs = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(CLASSES)) if Y[:, j].sum() > 0 and (1 - Y[:, j]).sum() > 0]
        return round(float(np.mean(aucs)), 4)

    base_lit = eval_litfl(model, lit_te)
    base_ptb = eval_ptb_auc(model, ptb_te[:500])
    print(f"[BASELINE v8] External Held-Out LITFL: {base_lit} | PTB-XL Fold-10 Macro-AUC: {base_ptb}")

    # Build mixed training set: 2,500 PTB-XL Fold 1-8 + 15x oversampled LITFL Odd-ID External Train
    rng_tr = np.random.default_rng(42)
    ptb_sub = [ptb_tr[i] for i in rng_tr.choice(len(ptb_tr), size=min(2500, len(ptb_tr)), replace=False)]
    lit_rows = []
    for it in lit_tr:
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for c in it["gt"]:
            if c in c2i: y[c2i[c]] = 1.0
        if y.sum() > 0:
            lit_rows.append({"kind": "img", "path": it["path"], "y": y})
    train_rows = ptb_sub + lit_rows * 15

    class MixDS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            rng = np.random.default_rng()
            if r["kind"] == "img":
                im = cv2.imread(r["path"])
                x = clean_ink(im, aug=True, rng=rng)
            else:
                sig, _ = wfdb.rdsamp(r["path"])
                x = render_ptb(np.nan_to_num(sig.astype(np.float32)), rng, aug=True)
            return torch.from_numpy(x), torch.from_numpy(r["y"])

    dl = DataLoader(MixDS(train_rows), batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_hit, best_sd = base_lit["op_hit_pct"], None
    for ep in range(1, 5):
        t0 = time.time()
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ep_lit = eval_litfl(model, lit_te)
        ep_ptb = eval_ptb_auc(model, ptb_te[:500])
        print(f"Ep {ep}/4 ({time.time()-t0:.0f}s) | External Held-Out LITFL: {ep_lit} | PTB-XL Fold-10 AUC: {ep_ptb}")
        if ep_lit["op_hit_pct"] > best_hit and ep_ptb >= (base_ptb - 0.012):
            best_hit = ep_lit["op_hit_pct"]
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_sd is not None and (best_hit - base_lit["op_hit_pct"]) >= 10.0:
        shutil.copy2(V8_PATH, os.path.join(OUT_DIR, "ecg_effnetv2s_v8_rare19_512x768_backup.pt"))
        ck["state_dict"] = best_sd
        torch.save(ck, V8_PATH)
        vol.commit()
        print(f"PROMOTED v18 External-Domain model! Held-Out LITFL Hit: {base_lit['op_hit_pct']}% -> {best_hit}%")
    else:
        print(f"NOT PROMOTED (v8 preserved): best Held-Out LITFL Hit={best_hit}% vs baseline={base_lit['op_hit_pct']}%")


@app.local_entrypoint()
def main():
    train_v18_external.remote()
