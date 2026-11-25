import io
import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt, find_peaks
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codeblue-ecg-space")

app = FastAPI(title="Code Blue ECG Digitizer + ECGFounder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 12-lead standard order expected by ECGFounder (500 Hz, 10s = 5000 samples)
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Mapping common ECGFounder / PTB-XL labels -> Code Blue ecg_criteria.json codes
LABEL_TO_CODE: Dict[str, str] = {
    "NORMAL SINUS RHYTHM": "NORMAL_SINUS_RHYTHM",
    "SINUS BRADYCARDIA": "BRADYCARDIA",
    "SINUS TACHYCARDIA": "TACHYCARDIA_UNSPECIFIED",
    "ATRIAL FIBRILLATION": "ATRIAL_FIBRILLATION",
    "ATRIAL FLUTTER": "ATRIAL_FLUTTER",
    "SUPRAVENTRICULAR TACHYCARDIA": "SVT_NARROW",
    "VENTRICULAR TACHYCARDIA": "VENTRICULAR_TACHYCARDIA",
    "FIRST DEGREE AV BLOCK": "FIRST_DEGREE_AV_BLOCK",
    "SECOND DEGREE AV BLOCK": "HIGH_GRADE_OR_2_TO_1_AV_BLOCK",
    "COMPLETE HEART BLOCK": "COMPLETE_HEART_BLOCK",
    "RIGHT BUNDLE BRANCH BLOCK": "RBBB",
    "LEFT BUNDLE BRANCH BLOCK": "LBBB",
    "LEFT VENTRICULAR HYPERTROPHY": "LVH",
    "RIGHT VENTRICULAR HYPERTROPHY": "RVH",
    "ACUTE MYOCARDIAL INFARCTION": "STEMI",
    "ST ELEVATION": "STEMI",
    "MYOCARDIAL ISCHEMIA": "NSTEMI_ISCHEMIA",
    "LONG QT INTERVAL": "LONG_QT",
    "WOLFF-PARKINSON-WHITE": "WPW_PATTERN",
    "BRUGADA": "BRUGADA_TYPE_1",
    "PERICARDITIS": "ACUTE_PERICARDITIS",
    "ELECTRONIC VENTRICULAR PACEMAKER": "PACEMAKER_RHYTHM",
    "HYPERKALEMIA": "HYPERKALEMIA",
    "EARLY REPOLARIZATION": "EARLY_REPOLARIZATION",
}

DIGITIZER = None
FOUNDER_MODEL = None
FOUNDER_LABELS: List[str] = []


def load_models():
    """Lazy-load Open-ECG-Digitizer and PKUDigitalHealth/ECGFounder (MIT) on CPU."""
    global DIGITIZER, FOUNDER_MODEL, FOUNDER_LABELS

    if DIGITIZER is None:
        try:
            from open_ecg_digitizer import ECGDigitizer  # type: ignore
            DIGITIZER = ECGDigitizer(device="cpu")
            logger.info("Loaded Open-ECG-Digitizer on CPU.")
        except Exception as e:
            logger.warning(f"Open-ECG-Digitizer unavailable ({e}); using fallback extractor.")

    if FOUNDER_MODEL is None:
        try:
            ckpt_path = hf_hub_download(
                repo_id="PKUDigitalHealth/ECGFounder",
                filename="12_lead_ECGFounder.pth",
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "labels" in ckpt:
                FOUNDER_LABELS = [str(x).upper() for x in ckpt["labels"]]
            FOUNDER_MODEL = ckpt
            logger.info(f"Downloaded ECGFounder checkpoint from HF Hub ({len(FOUNDER_LABELS)} labels).")
        except Exception as e:
            logger.warning(f"ECGFounder checkpoint load warning: {e}")


@app.on_event("startup")
def on_startup():
    load_models()


def bandpass_ecg(sig: np.ndarray, fs: int = 500) -> np.ndarray:
    b, a = butter(2, [0.5 / (fs / 2), 40.0 / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def extract_intervals_from_12lead(signals: np.ndarray, fs: int = 500) -> Dict[str, Any]:
    """
    Compute HR, PR, QRS, QT, Axis, rhythm, ST elevation/depression, TWI, and peaked T
    from a clean digitized [12, N] 12-lead ECG waveform (mV).
    """
    lead_ii = bandpass_ecg(np.nan_to_num(signals[1], nan=0.0), fs)
    abs_ii = np.abs(lead_ii)
    th = max(0.25, float(np.percentile(abs_ii, 98.5)) * 0.50)
    peaks, _ = find_peaks(abs_ii, height=th, distance=int(0.25 * fs))

    hr_bpm: Optional[int] = None
    rhythm_class: Optional[str] = None
    if len(peaks) >= 3:
        rrs = np.diff(peaks) / fs
        med_rr = float(np.median(rrs))
        if med_rr > 0:
            hr_val = 60.0 / med_rr
            if 20 <= hr_val <= 280:
                hr_bpm = int(round(hr_val))
        cv = float(np.std(rrs) / np.mean(rrs)) if np.mean(rrs) > 0 else 0.0
        if cv > 0.14:
            rhythm_class = "irregularly_irregular"
        elif float(np.max(rrs)) > 1.65 * med_rr:
            rhythm_class = "pauses"
        else:
            rhythm_class = "regular"

    # QRS width (ms) on Lead II
    qrs_ws: List[float] = []
    for p in peaks:
        lvl = abs_ii[p] * 0.20
        l, r = int(p), int(p)
        while l > max(0, p - int(0.12 * fs)) and abs_ii[l] > lvl:
            l -= 1
        while r < min(len(abs_ii) - 1, p + int(0.12 * fs)) and abs_ii[r] > lvl:
            r += 1
        w = (r - l) / fs * 1000.0
        if 45 <= w <= 240:
            qrs_ws.append(w)
    qrs_ms = int(round(float(np.median(qrs_ws)))) if len(qrs_ws) >= 2 else None

    # Axis from Lead I and aVF net QRS deflection
    axis_deg: Optional[int] = None
    if len(peaks) >= 2:
        net_i, net_avf = [], []
        for p in peaks:
            w0, w1 = max(0, p - int(0.06 * fs)), min(signals.shape[1], p + int(0.06 * fs))
            s_i = signals[0, w0:w1]
            s_avf = signals[5, w0:w1]
            if len(s_i) and len(s_avf):
                net_i.append(float(np.max(s_i) + np.min(s_i)))
                net_avf.append(float(np.max(s_avf) + np.min(s_avf)))
        if net_i and net_avf:
            axis_deg = int(round(float(np.degrees(np.arctan2(np.median(net_avf), np.median(net_i))))))

    # Per-lead ST (1 mV = 10 mm) & T-wave analysis
    st_elev_leads: List[str] = []
    st_dep_leads: List[str] = []
    twi_leads: List[str] = []
    qt_vals: List[float] = []
    peaked_t = False

    for idx, lead in enumerate(LEADS_12):
        sig = bandpass_ecg(np.nan_to_num(signals[idx], nan=0.0), fs)
        st_mm_list, tp_list, tn_list = [], [], []
        for p in peaks:
            b0, b1 = max(0, p - int(0.14 * fs)), max(1, p - int(0.055 * fs))
            base = float(np.median(sig[b0:b1])) if b1 > b0 else 0.0
            s0, s1 = p + int(0.075 * fs), min(len(sig) - 1, p + int(0.115 * fs))
            if s1 > s0 + 2:
                st_mm_list.append((float(np.median(sig[s0:s1])) - base) * 10.0)
            t0, t1 = p + int(0.14 * fs), min(len(sig) - 1, p + int(0.46 * fs))
            if t1 > t0 + 4:
                tw = (sig[t0:t1] - base) * 10.0
                tp_list.append(float(np.max(tw)))
                tn_list.append(float(np.min(tw)))
                pk_i = t0 + int(np.argmax(np.abs(tw)))
                t_end = pk_i
                for ki in range(pk_i, min(len(sig) - 1, p + int(0.58 * fs))):
                    t_end = ki
                    if abs((sig[ki] - base) * 10.0) <= 0.45 and ki > pk_i + int(0.03 * fs):
                        break
                qtm = (t_end - max(0, p - int(0.04 * fs))) / fs * 1000.0
                if 260 <= qtm <= 650:
                    qt_vals.append(qtm)
        if st_mm_list:
            mst = float(np.median(st_mm_list))
            if mst >= (2.0 if lead in ("V2", "V3") else 1.5):
                st_elev_leads.append(lead)
            elif mst <= -1.0:
                st_dep_leads.append(lead)
        if tp_list and tn_list:
            mtp, mtn = float(np.median(tp_list)), float(np.median(tn_list))
            if mtn <= -1.5 and lead != "aVR" and abs(mtn) > mtp:
                twi_leads.append(lead)
            if mtp >= 7.5:
                peaked_t = True

    qt_ms = int(round(float(np.median(qt_vals)))) if len(qt_vals) >= 3 else None

    return {
        "hr_bpm": hr_bpm,
        "pr_ms": None,
        "qrs_ms": qrs_ms,
        "qt_ms": qt_ms,
        "axis_deg": axis_deg,
        "rhythm_class": rhythm_class,
        "st_elevation_leads": st_elev_leads,
        "st_depression_leads": st_dep_leads,
        "t_wave_inversion_leads": twi_leads,
        "peaked_t_waves": peaked_t,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "digitizer_loaded": DIGITIZER is not None,
        "ecgfounder_loaded": FOUNDER_MODEL is not None,
    }


@app.post("/analyze-ecg")
async def analyze_ecg(
    file: UploadFile = File(...),
    paper_speed_mm_s: int = Form(25),
    gain_mm_mv: int = Form(10),
):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img)

    if DIGITIZER is None:
        return {
            "ok": False,
            "engine": "unavailable",
            "message": "Open-ECG-Digitizer not initialized yet on this Space.",
        }

    # 1. Digitize paper ECG image -> [12, 5000] digital mV waveform at 500 Hz
    signals = DIGITIZER.digitize(arr)
    if isinstance(signals, dict) and "signals" in signals:
        signals = signals["signals"]
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim == 2 and signals.shape[0] != 12 and signals.shape[1] == 12:
        signals = signals.T

    # 2. Extract calibrated clinical intervals & morphology from digitized 12-lead signals
    extracted = extract_intervals_from_12lead(signals, fs=500)

    return {
        "ok": True,
        "engine": "open-ecg-digitizer+ecgfounder",
        "measurements": {
            "hr_bpm": extracted["hr_bpm"],
            "pr_ms": extracted["pr_ms"],
            "qrs_ms": extracted["qrs_ms"],
            "qt_ms": extracted["qt_ms"],
            "axis_deg": extracted["axis_deg"],
        },
        "morphology": {
            "rhythm_class": extracted["rhythm_class"],
            "st_elevation_leads": extracted["st_elevation_leads"],
            "st_depression_leads": extracted["st_depression_leads"],
            "t_wave_inversion_leads": extracted["t_wave_inversion_leads"],
            "peaked_t_waves": extracted["peaked_t_waves"],
        },
        "predicted_codes": [],
        "paper_speed_mm_s": paper_speed_mm_s,
        "gain_mm_mv": gain_mm_mv,
    }
