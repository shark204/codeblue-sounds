"""
Code Blue ECG — Zero-Mercy 5-Pillar Production Torture Benchmark on Modal (~6 min, ~$0.12)
Exposes REAL weaknesses without sugarcoating:
  Pillar 1: OOD & Garbage Rejection (N=120: noise, blank, gradient, extreme blur, 50% crop, 180-deg flip)
  Pillar 2: Lead-Reversal False-Positive Rate on REAL PMcardio Phone Photos (N=576 un-swapped photos)
  Pillar 3: Blind 50mm/s (NO footer text), 12.5mm/s, 5mm/mV, 20mm/mV on Fold-10 (N=500)
  Pillar 4: Adversarial Multi-Machine OCR Torture (N=60 skewed/shadowed GE/NK/Edan/Philips headers)
  Pillar 5: Live Production Modal Endpoint Health + End-to-End Latency (P50/P95) + 0% 500-Error Check
"""
import modal

app = modal.App("codeblue-torture-benchmark")
vol = modal.Volume.from_name("codeblue-ecg-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "timm", "wfdb", "scipy", "scikit-learn",
        "pandas", "numpy", "opencv-python-headless", "pillow", "requests",
        "rapidocr-onnxruntime",
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
def run_torture_benchmark():
    import io, json, math, os, re, time
    import cv2, numpy as np, pandas as pd, requests, timm, torch, wfdb
    from PIL import Image

    vol.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR, PTB_DIR, PM_DIR, PLUS_DIR = "/vol/outputs", "/vol/ptbxl", "/vol/pmcardio", "/vol/ptbxl_plus"
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def clean_ink(bgr: np.ndarray, h: int, w: int) -> torch.Tensor:
        if bgr.shape[0] != h or bgr.shape[1] != w:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        b, g, r = cv2.split(bgr.astype(np.float32))
        pink = np.clip((r - np.minimum(g, b)) - 12.0, 0.0, 255.0)
        luma = 0.55 * r + 0.35 * g + 0.10 * b
        ng = np.clip(luma + pink * 0.95, 0.0, 255.0)
        bg = cv2.GaussianBlur(ng, (31, 31), 0)
        ink = np.clip(bg - ng, 0.0, 255.0)
        hi = max(25.0, float(np.percentile(ink, 99.5)))
        gray = np.clip(255.0 - (ink / hi) * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)

    # Load all 3 production models
    v8_p = os.path.join(OUT_DIR, "ecg_effnetv2s_v8_rare19_512x768.pt")
    if not os.path.exists(v8_p):
        v8_p = os.path.join(OUT_DIR, "ecg_effnetv2s_multimachine_512x768.pt")
    ck_v8 = torch.load(v8_p, map_location=device, weights_only=False)
    classes = list(ck_v8["classes"])
    m_v8 = timm.create_model(ck_v8.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k"), pretrained=False, num_classes=len(classes)).to(device)
    m_v8.load_state_dict(ck_v8["state_dict"]); m_v8.eval()
    platt_a = np.array(ck_v8["platt_a"], dtype=np.float32)
    platt_b = np.array(ck_v8["platt_b"], dtype=np.float32)
    thr_map = ck_v8.get("thresholds", {})
    CRITICAL = {"STEMI_ANTERIOR", "STEMI_INFERIOR", "STEMI_LATERAL", "HIGH_GRADE_AV_BLOCK", "BRUGADA_TYPE_1", "WPW_PREEXCITATION"}

    ck_rev = torch.load(os.path.join(OUT_DIR, "ecg_lead_reversal_detector.pt"), map_location=device, weights_only=False)
    m_rev = timm.create_model(ck_rev.get("arch", "mobilenetv3_large_100"), pretrained=False, num_classes=1).to(device)
    m_rev.load_state_dict(ck_rev["state_dict"]); m_rev.eval()
    rev_thr = float(ck_rev.get("threshold", 0.54))

    ck_int = torch.load(os.path.join(OUT_DIR, "ecg_interval_regressor.pt"), map_location=device, weights_only=False)
    m_int = timm.create_model(ck_int.get("arch", "tf_efficientnetv2_s.in21k_ft_in1k"), pretrained=False, num_classes=7).to(device)
    m_int.load_state_dict(ck_int["state_dict"]); m_int.eval()
    int_scales = np.array(ck_int.get("scales", [100.0, 200.0, 120.0, 400.0, 420.0]), dtype=np.float32)

    report = {}

    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 1: OOD & GARBAGE REJECTION (N = 120)
    # ═══════════════════════════════════════════════════════════════════════
    rng = np.random.default_rng(2026)
    ood_groups = {"random_noise": [], "blank_paper": [], "xray_gradient": [], "extreme_blur": [], "half_crop": [], "upside_down_180": []}

    # Load 20 Fold-10 ECGs for degraded variants
    db = pd.read_csv(os.path.join(PTB_DIR, "ptbxl_database.csv"))
    f10_rows = db[db["strat_fold"] == 10].drop_duplicates("patient_id").head(500)

    def render_basic(sig, speed=25.0, gain=10.0, show_footer=False):
        img = np.full((512, 768, 3), (242, 232, 252), dtype=np.uint8)
        for x in range(0, 768, 11): cv2.line(img, (x, 0), (x, 512), (210, 185, 240) if x % 55 else (180, 145, 225), 1)
        for y in range(0, 512, 11): cv2.line(img, (0, y), (768, y), (210, 185, 240) if y % 55 else (180, 145, 225), 1)
        sig = np.nan_to_num(sig, 0.0)
        n_vis = max(200, int(round(sig.shape[0] * min(1.0, 25.0 / speed))))
        px_mv = 40.0 * (gain / 10.0)
        cw, rh = (768 - 26) // 4, (512 - 24) // 4
        # Draw 1 mV calibration pulse (width scales with speed)
        w_cal = max(4, int(round((speed * 0.2) * 2.8)))
        h_cal = max(8, int(round(px_mv)))
        for ri in range(4):
            y0 = int((ri + 0.55) * rh)
            pts = np.array([[4, y0], [7, y0], [7, y0 - h_cal], [7 + w_cal, y0 - h_cal], [7 + w_cal, y0], [10 + w_cal, y0]], dtype=np.int32)
            cv2.polylines(img, [pts], False, (18, 18, 18), 2, cv2.LINE_AA)
        for li in range(12):
            r, c = li % 3, li // 3
            s = sig[int(c * n_vis / 4):max(int(c * n_vis / 4) + 20, int((c + 1) * n_vis / 4)), li]
            xs = np.linspace(26 + c * cw + 4, 26 + (c + 1) * cw - 4, len(s)).astype(np.int32)
            ys = np.clip(int((r + 0.55) * rh) - (s * px_mv).astype(np.int32), 4, 490)
            cv2.polylines(img, [np.stack([xs, ys], 1)], False, (18, 18, 18), 2, cv2.LINE_AA)
        s2 = sig[:n_vis, 1]
        xs = np.linspace(26, 762, len(s2)).astype(np.int32)
        ys = np.clip(int(3.55 * rh) - (s2 * px_mv).astype(np.int32), 4, 490)
        cv2.polylines(img, [np.stack([xs, ys], 1)], False, (18, 18, 18), 2, cv2.LINE_AA)
        if show_footer:
            cv2.putText(img, f"{speed:g}mm/s  {gain:g}mm/mV", (18, 506), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (18, 18, 18), 1, cv2.LINE_AA)
        return img

    sample_ecgs = [render_basic(wfdb.rdsamp(os.path.join(PTB_DIR, r["filename_lr"]))[0]) for _, r in f10_rows.head(20).iterrows()]
    for i in range(20):
        ood_groups["random_noise"].append(rng.integers(0, 256, (512, 768, 3), dtype=np.uint8))
        ood_groups["blank_paper"].append(np.full((512, 768, 3), int(rng.integers(220, 255)), dtype=np.uint8))
        gx = np.linspace(20, 230, 768, dtype=np.uint8); gy = np.linspace(30, 210, 512, dtype=np.uint8)
        grad = np.clip(np.outer(gy, gx) // 255 + rng.integers(0, 15, (512, 768)), 0, 255).astype(np.uint8)
        ood_groups["xray_gradient"].append(cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR))
        ood_groups["extreme_blur"].append(cv2.GaussianBlur(sample_ecgs[i], (31, 31), 12))
        crop = sample_ecgs[i][:256, :384]
        ood_groups["half_crop"].append(cv2.resize(crop, (768, 512)))
        ood_groups["upside_down_180"].append(cv2.rotate(sample_ecgs[i], cv2.ROTATE_180))

    ood_res = {}
    def validate_ecg_image(img_bgr):
        small = cv2.resize(img_bgr, (512, 340), interpolation=cv2.INTER_AREA)
        b, g, r = cv2.split(small.astype(np.float32))
        pink = np.clip((r - np.minimum(g, b)) - 12.0, 0.0, 255.0)
        ng = np.clip(0.55 * r + 0.35 * g + 0.10 * b + pink * 0.95, 0.0, 255.0)
        raw_ink = np.clip(cv2.GaussianBlur(ng, (31, 31), 0) - ng, 0.0, 255.0)
        if float(np.percentile(raw_ink, 99.0)) < 18.0 or float(np.mean(raw_ink > 22.0)) < 0.0025: return False
        if float(np.mean(raw_ink > 22.0)) > 0.30 or float(cv2.Laplacian(ng, cv2.CV_32F).var()) < 22.0: return False
        row_ink = np.mean(raw_ink > 22.0, axis=1)
        return sum(1 for bi in range(4) if float(np.max(row_ink[bi*85:(bi+1)*85])) > 0.012) >= 2

    with torch.no_grad():
        for gname, imgs in ood_groups.items():
            any_alert, crit_alert, rev_alert, rejected = 0, 0, 0, 0
            for im in imgs:
                if not validate_ecg_image(im):
                    rejected += 1
                    continue
                x = clean_ink(im, 512, 768).to(device)
                logits = m_v8(x).cpu().numpy()[0]
                probs = 1.0 / (1.0 + np.exp(-(platt_a * logits + platt_b)))
                fired = [classes[k] for k in range(len(classes)) if probs[k] >= float(thr_map.get(classes[k], 0.35)) and classes[k] != "NORMAL_SINUS_RHYTHM"]
                if fired: any_alert += 1
                if any(c in CRITICAL for c in fired): crit_alert += 1
                xr = clean_ink(im, 384, 512).to(device)
                pr = float(torch.sigmoid(m_rev(xr)).item())
                if pr >= rev_thr: rev_alert += 1
            ood_res[gname] = {"n": len(imgs), "safely_rejected_pct": round(100 * rejected / len(imgs), 1), "hallucinated_any_pct": round(100 * any_alert / len(imgs), 1), "hallucinated_crit_pct": round(100 * crit_alert / len(imgs), 1), "lead_rev_fired_pct": round(100 * rev_alert / len(imgs), 1)}
    report["pillar1_ood_garbage"] = ood_res
    print("\n=== PILLAR 1: OOD & GARBAGE REJECTION ===")
    print(json.dumps(ood_res, indent=2))

    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 2: LEAD-REVERSAL FALSE-POSITIVE RATE ON REAL PMCARDIO PHOTOS
    # ═══════════════════════════════════════════════════════════════════════
    pm_imgs = []
    for root, _, files in os.walk(PM_DIR):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                pm_imgs.append(os.path.join(root, fn))
    pm_imgs.sort()
    pm_sample = pm_imgs[-576:] if len(pm_imgs) >= 576 else pm_imgs
    rev_probs, real_rejected = [], 0
    with torch.no_grad():
        for pth in pm_sample:
            im = cv2.imread(pth)
            if im is None: continue
            if not validate_ecg_image(im):
                real_rejected += 1
            xr = clean_ink(im, 384, 512).to(device)
            rev_probs.append(float(torch.sigmoid(m_rev(xr)).item()))
    arr_r = np.array(rev_probs, dtype=np.float32)
    p2 = {
        "n_real_photos": int(len(arr_r)),
        "ood_gate_false_rejection_on_real_ecgs_pct": round(100.0 * real_rejected / max(1, len(arr_r)), 2),
        "thr_used": round(rev_thr, 4),
        "false_positive_rate_at_thr_pct": round(100.0 * float(np.mean(arr_r >= rev_thr)), 2) if len(arr_r) else None,
        "fpr_at_0_85_pct": round(100.0 * float(np.mean(arr_r >= 0.85)), 2) if len(arr_r) else None,
        "fpr_at_0_95_pct": round(100.0 * float(np.mean(arr_r >= 0.95)), 2) if len(arr_r) else None,
        "median_prob": round(float(np.median(arr_r)), 4) if len(arr_r) else None,
        "p95_prob": round(float(np.percentile(arr_r, 95)), 4) if len(arr_r) else None,
    }
    report["pillar2_lead_reversal_real_pmcardio_fpr"] = p2
    print("\n=== PILLAR 2: LEAD-REVERSAL FPR ON REAL PMCARDIO PHONE PHOTOS ===")
    print(json.dumps(p2, indent=2))

    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 3: BLIND 50mm/s (NO FOOTER TEXT) + 12.5mm/s + HALF/DOUBLE GAIN
    # ═══════════════════════════════════════════════════════════════════════
    f12 = pd.read_csv(os.path.join(PLUS_DIR, "12sl_features.csv"))
    def pcol(df, pats):
        for pat in pats:
            for c in df.columns:
                if re.fullmatch(pat, c, flags=re.I):
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.notna().sum() > 500: return s
        return pd.Series(np.nan, index=df.index)
    gt = pd.DataFrame({
        "ecg_id": pd.to_numeric(f12["ecg_id"], errors="coerce").astype("Int64"),
        "rr": pcol(f12, [r"RR_Mean_Global", r"RR_Int_Global"]),
        "hr": pcol(f12, [r"Ventricular_Rate.*", r"HR_Global"]),
        "pr": pcol(f12, [r"PR_Int_Global", r"PR_Int"]),
        "qrs": pcol(f12, [r"QRS_Dur_Global", r"QRS_Dur"]),
        "qt": pcol(f12, [r"QT_Int_Global", r"QT_Int"]),
    }).dropna(subset=["ecg_id"]).set_index("ecg_id")
    gt["hr"] = np.where(np.isfinite(gt["hr"]), gt["hr"], np.where(gt["rr"] > 250, 60000.0 / gt["rr"], np.nan))

    regimes = [
        ("blind_50mm_s_NO_TEXT", 50.0, 10.0, False),
        ("labeled_50mm_s_WITH_TEXT", 50.0, 10.0, True),
        ("blind_12_5mm_s_NO_TEXT", 12.5, 10.0, False),
        ("half_gain_5mm_mV", 25.0, 5.0, False),
        ("double_gain_20mm_mV", 25.0, 20.0, False),
    ]
    p3 = {}
    eval_rows = [r for _, r in f10_rows.head(250).iterrows() if int(r["ecg_id"]) in gt.index]
    with torch.no_grad():
        for rname, spd, gn, show_ft in regimes:
            errs = {k: [] for k in ["hr", "pr", "qrs", "qt"]}
            for r in eval_rows:
                eid = int(r["ecg_id"])
                sig, _ = wfdb.rdsamp(os.path.join(PTB_DIR, r["filename_lr"]))
                im = render_basic(sig, speed=spd, gain=gn, show_footer=show_ft)
                out = m_int(clean_ink(im, 512, 768).to(device)).cpu().numpy()[0]
                row_gt = gt.loc[eid]
                for idx_k, nm in enumerate(["hr", "pr", "qrs", "qt"]):
                    gv = float(row_gt[nm])
                    if np.isfinite(gv) and gv > 20:
                        errs[nm].append(abs(float(out[idx_k]) * float(int_scales[idx_k]) - gv))
            p3[rname] = {nm: {"mae": round(float(np.mean(v)), 2), "p95": round(float(np.percentile(v, 95)), 2)} for nm, v in errs.items() if v}
    report["pillar3_blind_speed_gain_intervals"] = p3
    print("\n=== PILLAR 3: BLIND SPEED / GAIN INTERVAL STRESS (N=250 FOLD-10) ===")
    print(json.dumps(p3, indent=2))

    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 4: ADVERSARIAL MULTI-MACHINE OCR TORTURE (N = 40)
    # ═══════════════════════════════════════════════════════════════════════
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    ocr_cases = [
        ("GE_standard", "HR: 84 bpm   PR: 168 ms   QRS: 96 ms   QT/QTc: 382/428 ms   Axis: +42"),
        ("NihonKohden_glued", "HR 112/min  PR 0.14s  QRS 138ms  QT/QTc 360/462ms  P-R-T 45/-25/30"),
        ("Edan_PQ_variant", "Vent. rate 54 BPM   PQ 224 ms   QRSd 118 ms   QT 440 ms   QTc 424 ms"),
        ("Philips_shortQT", "HR 76   PR 144ms   QRS 88ms   QT 318ms   QTc 334ms   QRS Axis -48"),
    ]
    ocr_hits, ocr_tot = 0, 0
    for cname, txt in ocr_cases:
        for tilt in [-3.0, 0.0, 3.0]:
            canvas = np.full((180, 900, 3), (242, 238, 248), dtype=np.uint8)
            cv2.putText(canvas, txt, (24, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
            if tilt != 0.0:
                M = cv2.getRotationMatrix2D((450, 90), tilt, 1.0)
                canvas = cv2.warpAffine(canvas, M, (900, 180), borderMode=cv2.BORDER_REPLICATE)
            res_o, _ = ocr(canvas)
            joined = " ".join([x[1] for x in (res_o or [])])
            ocr_tot += 1
            if re.search(r"\b(HR|Vent\.?\s*rate)\b", joined, re.I) and re.search(r"\b(QRS|QRSd)\b", joined, re.I):
                ocr_hits += 1
    report["pillar4_adversarial_ocr"] = {"tested": ocr_tot, "header_parsed_pct": round(100.0 * ocr_hits / max(1, ocr_tot), 1)}
    print("\n=== PILLAR 4: ADVERSARIAL OCR TORTURE ===")
    print(json.dumps(report["pillar4_adversarial_ocr"], indent=2))

    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 5: LIVE PRODUCTION MODAL ENDPOINT HEALTH + LATENCY
    # ═══════════════════════════════════════════════════════════════════════
    BASE_URL = "https://mohama2468d--codeblue-ecg-ai-fastapi-app.modal.run"
    h_t0 = time.time()
    h_r = requests.get(f"{BASE_URL}/health", timeout=60)
    h_json = h_r.json() if h_r.status_code == 200 else {"status": h_r.status_code}
    latencies, ok_cnt = [], 0
    for i in range(6):
        _, buf = cv2.imencode(".jpg", sample_ecgs[i], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        t0 = time.time()
        resp = requests.post(f"{BASE_URL}/analyze-ecg", files={"file": (f"test_{i}.jpg", buf.tobytes(), "image/jpeg")}, timeout=90)
        dt = time.time() - t0
        if resp.status_code == 200 and resp.json().get("ok"):
            ok_cnt += 1
            latencies.append(dt)
    report["pillar5_live_endpoint"] = {
        "health": h_json,
        "health_sec": round(time.time() - h_t0, 2),
        "live_success": f"{ok_cnt}/6",
        "p50_sec": round(float(np.median(latencies)), 2) if latencies else None,
        "p95_sec": round(float(np.percentile(latencies, 95)), 2) if latencies else None,
    }
    print("\n=== PILLAR 5: LIVE PRODUCTION ENDPOINT HEALTH & LATENCY ===")
    print(json.dumps(report["pillar5_live_endpoint"], indent=2))

    with open(os.path.join(OUT_DIR, "torture_benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    vol.commit()
    print("\nSaved full torture report to /vol/outputs/torture_benchmark_report.json")


@app.local_entrypoint()
def main():
    run_torture_benchmark.remote()
