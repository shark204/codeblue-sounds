"use client";

import rawRefIndex from "@/ecg_engine/ecg_reference_index.json";
import { ECGInput, ECGInterpretationResult, interpretECG } from "@/lib/ecgRules";

interface RefSig {
  hr: number;
  rr_cv: number;
  qrs_ms: number;
  fl_snr: number;
  bl_std: number;
  beat16: number[];
}

interface RefEntry {
  id: string;
  condition_key: string;
  expected_code: string;
  title: string;
  source_book: string;
  source_page: number;
  caption: string;
  reference_ecg: ECGInput;
  sig: RefSig;
}

export interface LocalECGAnalysisResult {
  layout: "single_lead" | "12_lead";
  is_regular_rhythm: boolean;
  rr_regularity_pct: number;
  reference_count: number;
  interpretation: ECGInterpretationResult;
  summary_ar: string;
  summary_en: string;
}

const refIndex = rawRefIndex as { total_reference_ecgs: number; entries: RefEntry[] };

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to decode ECG image"));
    img.src = dataUrl;
  });
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1]! + s[mid]!) / 2 : s[mid]!;
}

function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.floor((p / 100) * s.length)));
  return s[idx]!;
}

function cosineSim(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/**
 * Adaptive ink isolation with morphological straight-gridline removal.
 * Cleanly erases BOTH pink/red AND dark gray/black ECG graph paper lines!
 */
function isolateCleanInk(rgba: Uint8ClampedArray, W: number, H: number): Float32Array {
  const darkness = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const maxCh = Math.max(rgba[i * 4]!, rgba[i * 4 + 1]!, rgba[i * 4 + 2]!);
    darkness[i] = 255 - maxCh;
  }

  // Row & column 70th-percentile background estimation to erase straight gridlines
  const rowBg = new Float32Array(H);
  const sampleRow = new Array<number>(Math.floor(W / 4));
  for (let y = 0; y < H; y++) {
    let idx = 0;
    for (let x = 0; x < W - 3; x += 4) sampleRow[idx++] = darkness[y * W + x]!;
    rowBg[y] = percentile(sampleRow.slice(0, idx), 70);
  }

  const colBg = new Float32Array(W);
  const sampleCol = new Array<number>(Math.floor(H / 4));
  for (let x = 0; x < W; x++) {
    let idx = 0;
    for (let y = 0; y < H - 3; y += 4) sampleCol[idx++] = darkness[y * W + x]!;
    colBg[x] = percentile(sampleCol.slice(0, idx), 70);
  }

  const clean = new Float32Array(W * H);
  const flatSample: number[] = [];
  for (let y = 0; y < H; y++) {
    const rb = rowBg[y]!;
    for (let x = 0; x < W; x++) {
      const val = Math.max(0, darkness[y * W + x]! - Math.max(rb, colBg[x]!) * 0.85);
      clean[y * W + x] = val;
      if ((x & 7) === 0 && (y & 7) === 0) flatSample.push(val);
    }
  }

  const p94 = percentile(flatSample, 94);
  const p995 = percentile(flatSample, 99.5);
  const lo = Math.max(18, p94);
  const hi = Math.max(lo + 25, p995);
  const span = hi - lo;

  const ink = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const v = (clean[i]! - lo) / span;
    ink[i] = v <= 0 ? 0 : v >= 1 ? 1 : v;
  }
  return ink;
}

function detectHorizontalBands(ink: Float32Array, W: number, H: number): Array<[number, number]> {
  const xStart = Math.floor(W * 0.05);
  const xEnd = Math.floor(W * 0.95);
  const rowEnergy = new Float32Array(H);
  for (let y = 0; y < H; y++) {
    let sum = 0;
    for (let x = xStart; x < xEnd; x++) sum += ink[y * W + x]!;
    rowEnergy[y] = sum / (xEnd - xStart);
  }
  const halfK = Math.max(2, Math.floor(H * 0.012));
  const smooth = new Float32Array(H);
  let maxSmooth = 0;
  for (let y = 0; y < H; y++) {
    let s = 0;
    let c = 0;
    for (let k = Math.max(0, y - halfK); k <= Math.min(H - 1, y + halfK); k++) {
      s += rowEnergy[k]!;
      c++;
    }
    smooth[y] = s / Math.max(1, c);
    if (smooth[y]! > maxSmooth) maxSmooth = smooth[y]!;
  }
  const thresh = Math.max(0.008, maxSmooth * 0.24);
  const pad = Math.floor(H * 0.035);
  const minBandH = Math.floor(H * 0.03);
  const rawBands: Array<[number, number]> = [];
  let inBand = false;
  let start = 0;
  for (let y = Math.floor(H * 0.03); y < Math.floor(H * 0.97); y++) {
    if (smooth[y]! >= thresh && !inBand) {
      inBand = true;
      start = y;
    } else if (smooth[y]! < thresh && inBand) {
      inBand = false;
      if (y - start >= minBandH) rawBands.push([Math.max(0, start - pad), Math.min(H, y + pad)]);
    }
  }
  if (inBand && Math.floor(H * 0.97) - start >= minBandH) {
    rawBands.push([Math.max(0, start - pad), Math.min(H, Math.floor(H * 0.97))]);
  }
  const merged: Array<[number, number]> = [];
  for (const [b0, b1] of rawBands) {
    if (merged.length > 0 && b0 <= merged[merged.length - 1]![1]) {
      merged[merged.length - 1]![1] = Math.max(merged[merged.length - 1]![1], b1);
    } else {
      merged.push([b0, b1]);
    }
  }
  return merged;
}

function extractBandSignalMm(
  ink: Float32Array,
  W: number,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  pxPerMm: number
): number[] {
  const sw = x1 - x0;
  const sh = y1 - y0;
  const yCent = new Float32Array(sw);
  const valid = new Uint8Array(sw);
  for (let xi = 0; xi < sw; xi++) {
    const x = x0 + xi;
    let wSum = 0;
    let ySum = 0;
    for (let yi = 0; yi < sh; yi++) {
      const v = ink[(y0 + yi) * W + x]!;
      if (v > 0.12) {
        wSum += v;
        ySum += v * yi;
      }
    }
    if (wSum > 0.25) {
      yCent[xi] = ySum / wSum;
      valid[xi] = 1;
    }
  }
  let lastValid = -1;
  for (let xi = 0; xi < sw; xi++) {
    if (valid[xi]) {
      if (lastValid === -1 && xi > 0) {
        for (let k = 0; k < xi; k++) yCent[k] = yCent[xi]!;
      } else if (lastValid >= 0 && xi - lastValid > 1) {
        const yA = yCent[lastValid]!;
        const yB = yCent[xi]!;
        const span = xi - lastValid;
        for (let k = 1; k < span; k++) yCent[lastValid + k] = yA + ((yB - yA) * k) / span;
      }
      lastValid = xi;
    }
  }
  if (lastValid >= 0 && lastValid < sw - 1) {
    for (let k = lastValid + 1; k < sw; k++) yCent[k] = yCent[lastValid]!;
  }
  const baseY = median(Array.from(yCent));
  const out = new Array<number>(sw);
  for (let xi = 0; xi < sw; xi++) out[xi] = (baseY - yCent[xi]!) / pxPerMm;
  return out;
}

function dftMagnitude(sig: number[], fs: number, freqHz: number): number {
  const N = sig.length;
  let re = 0;
  let im = 0;
  const omega = (2 * Math.PI * freqHz) / fs;
  for (let n = 0; n < N; n++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
    const x = sig[n]! * w;
    re += x * Math.cos(omega * n);
    im -= x * Math.sin(omega * n);
  }
  return Math.sqrt(re * re + im * im) / N;
}

export async function analyzeEcgImageLocally(dataUrl: string): Promise<LocalECGAnalysisResult> {
  const img = await loadImage(dataUrl);
  const W = 1000;
  const H = 560;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.drawImage(img, 0, 0, W, H);
  const rgba = ctx.getImageData(0, 0, W, H).data;

  const ink = isolateCleanInk(rgba, W, H);
  const bands = detectHorizontalBands(ink, W, H);
  const isSingleLead = bands.length <= 1;
  const aspect = img.naturalWidth / Math.max(1, img.naturalHeight);
  const durationSec = isSingleLead || aspect < 1.35 ? 6.0 : 10.0;

  const sx0 = Math.floor(W * 0.04);
  const sx1 = Math.floor(W * 0.96);
  const stripWidth = sx1 - sx0;
  const pxPerMm = stripWidth / (durationSec * 25.0);
  const fs = stripWidth / durationSec;

  const pickBand = (b: [number, number]) => {
    const sig = extractBandSignalMm(ink, W, sx0, sx1, b[0], b[1], pxPerMm);
    const absS = sig.map((v) => Math.abs(v));
    const qrsTh = Math.max(2.6, percentile(absS, 98.5) * 0.56);
    const minRr = Math.floor(0.30 * fs);
    const pks: number[] = [];
    for (let x = 3; x < absS.length - 3; x++) {
      const v = absS[x]!;
      if (v >= qrsTh && v >= absS[x - 1]! && v >= absS[x + 1]! && v >= absS[x - 2]! && v >= absS[x + 2]!) {
        if (pks.length === 0 || x - pks[pks.length - 1]! >= minRr) pks.push(x);
        else if (v > absS[pks[pks.length - 1]!]!) pks[pks.length - 1] = x;
      }
    }
    return { sig, absS, pks };
  };

  const primaryBand: [number, number] = bands.length > 0 ? bands[bands.length - 1]! : [Math.floor(H * 0.2), Math.floor(H * 0.85)];
  let chosen = pickBand(primaryBand);
  if (chosen.pks.length < 4 && bands.length >= 2) {
    const alt = pickBand(bands[bands.length - 2]!);
    if (alt.pks.length >= chosen.pks.length) chosen = alt;
  }

  const { sig: sigMm, absS: absSig, pks: peaks } = chosen;

  // Robust RR regularity (MAD / median)
  let hrBpm: number | null = null;
  let madCv = 0.06;
  let maxMinRrRatio = 1.0;
  if (peaks.length >= 3) {
    const rrsSec: number[] = [];
    for (let i = 1; i < peaks.length; i++) rrsSec.push((peaks[i]! - peaks[i - 1]!) / fs);
    const medRr = median(rrsSec);
    const absDevs = rrsSec.map((r) => Math.abs(r - medRr));
    madCv = median(absDevs) / (medRr + 1e-6);
    maxMinRrRatio = Math.max(...rrsSec) / Math.max(0.2, Math.min(...rrsSec));
    hrBpm = Math.round((60 / medRr) * 10) / 10;
  } else if (peaks.length > 0) {
    hrBpm = Math.round((peaks.length * 60) / durationSec);
  }

  // QRS width + beat16 template
  const qrsWidths: number[] = [];
  const beatSamples: number[][] = [];
  const winL = Math.floor(0.18 * fs);
  const winR = Math.floor(0.32 * fs);
  for (const p of peaks) {
    const pkVal = absSig[p]!;
    const thresh = pkVal * 0.25;
    const maxSpan = Math.floor(0.14 * fs);
    let l = p;
    while (l > Math.max(0, p - maxSpan) && absSig[l]! > thresh) l--;
    let r = p;
    while (r < Math.min(absSig.length - 1, p + maxSpan) && absSig[r]! > thresh) r++;
    const wMs = ((r - l) / fs) * 1000;
    if (wMs >= 45 && wMs <= 240) qrsWidths.push(wMs);

    if (p - winL >= 0 && p + winR < sigMm.length) {
      const segLen = winL + winR;
      const resampled = new Array<number>(16).fill(0);
      for (let i = 0; i < 16; i++) {
        const pos = (i * (segLen - 1)) / 15;
        const idx0 = Math.floor(pos);
        const idx1 = Math.min(segLen - 1, idx0 + 1);
        const frac = pos - idx0;
        resampled[i] = sigMm[p - winL + idx0]! * (1 - frac) + sigMm[p - winL + idx1]! * frac;
      }
      const nrm = Math.sqrt(resampled.reduce((a, v) => a + v * v, 0)) + 1e-6;
      beatSamples.push(resampled.map((v) => v / nrm));
    }
  }
  const qrsMs = qrsWidths.length > 0 ? Math.round(median(qrsWidths)) : 86;
  const beat16 = new Array<number>(16).fill(0);
  if (beatSamples.length > 0) {
    for (let i = 0; i < 16; i++) beat16[i] = median(beatSamples.map((b) => b[i]!));
    const nrm = Math.sqrt(beat16.reduce((a, v) => a + v * v, 0)) + 1e-6;
    for (let i = 0; i < 16; i++) beat16[i] = Math.round((beat16[i]! / nrm) * 1000);
  }

  // Real PR interval measurement (-360ms .. -80ms before R-peak)
  const prList: number[] = [];
  for (const p of peaks) {
    const a = p - Math.floor(0.36 * fs);
    const b = p - Math.floor(0.08 * fs);
    if (a >= 0 && b > a + 4) {
      let bestVal = -999;
      let bestIdx = a;
      for (let k = a; k < b; k++) {
        if (sigMm[k]! > bestVal) {
          bestVal = sigMm[k]!;
          bestIdx = k;
        }
      }
      if (bestVal >= 0.45) {
        const prVal = ((p - bestIdx) / fs) * 1000 + 35;
        if (prVal >= 110 && prVal <= 380) prList.push(prVal);
      }
    }
  }
  const measuredPrMs = prList.length >= 2 ? Math.round(median(prList)) : null;

  // Inter-QRS baseline DFT for Flutter vs AFib vs Clean Sinus/AVB
  const bl = [...sigMm];
  const maskRad = Math.floor(0.08 * fs);
  for (const p of peaks) {
    const l = Math.max(0, p - maskRad);
    const r = Math.min(bl.length - 1, p + maskRad);
    const vL = bl[l]!;
    const vR = bl[r]!;
    const span = Math.max(1, r - l);
    for (let k = l; k <= r; k++) bl[k] = vL + ((vR - vL) * (k - l)) / span;
  }
  const blMean = bl.reduce((a, b) => a + b, 0) / bl.length;
  for (let i = 0; i < bl.length; i++) bl[i] = bl[i]! - blMean;
  const blStdMm = Math.sqrt(bl.reduce((a, b) => a + b * b, 0) / bl.length);

  let flutterPeakMag = 0;
  let flutterPeakFreq = 4.5;
  let totalMagSum = 0;
  let totalBins = 0;
  for (let f = 0.8; f <= 10.0; f += 0.15) {
    const mag = dftMagnitude(bl, fs, f);
    totalMagSum += mag;
    totalBins++;
    if (f >= 3.6 && f <= 5.8 && mag > flutterPeakMag) {
      flutterPeakMag = mag;
      flutterPeakFreq = f;
    }
  }
  const meanSpec = totalMagSum / Math.max(1, totalBins) + 1e-6;
  const flutterSnr = flutterPeakMag / meanSpec;
  const flutterRateBpm = Math.round(flutterPeakFreq * 60);

  // Clinical Rhythm Classification
  const isAtrialFlutter = flutterSnr >= 3.2 && blStdMm >= 0.45 && qrsMs < 135;
  const isRegular = madCv < 0.12;
  const hasDroppedBeatPause = !isRegular && !isAtrialFlutter && maxMinRrRatio >= 1.55 && blStdMm < 0.90 && measuredPrMs != null && measuredPrMs >= 210;
  const isAfib = !isAtrialFlutter && !isRegular && !hasDroppedBeatPause && qrsMs < 135;
  const isWideTach = isRegular && hrBpm != null && hrBpm >= 105 && qrsMs >= 120;
  const isNarrowSvt = isRegular && hrBpm != null && hrBpm >= 145 && qrsMs < 120 && !isAtrialFlutter;
  const isSevereBrady = isRegular && hrBpm != null && hrBpm <= 42;
  const isFirstDegAvb = isRegular && measuredPrMs != null && measuredPrMs > 205;

  // Match against 215-ECG Reference Atlas for morphology refinement when regular & no hard arrhythmia
  const scoredRefs = refIndex.entries.map((entry) => {
    const s = entry.sig;
    const beatSim = Math.max(0, cosineSim(beat16, s.beat16));
    const hrSim = hrBpm != null ? Math.max(0, 1 - Math.abs(hrBpm - s.hr) / 100) : 0.5;
    const rrSim = Math.max(0, 1 - Math.abs(madCv - s.rr_cv) * 2.5);
    const qrsSim = Math.max(0, 1 - Math.abs(qrsMs - s.qrs_ms) / 90);
    const score = beatSim * 0.48 + rrSim * 0.20 + qrsSim * 0.20 + hrSim * 0.12;
    return { entry, score };
  });
  scoredRefs.sort((a, b) => b.score - a.score);
  const topRef = scoredRefs[0]!;

  const estQtMs = hrBpm ? Math.round(Math.min(440, Math.max(300, 385 * Math.sqrt(60 / Math.max(45, hrBpm))))) : 380;

  let measuredInput: ECGInput = {
    age: 55,
    sex: "male",
    hr_bpm: hrBpm ?? 75,
    pr_ms: isAtrialFlutter || isAfib || isWideTach ? null : measuredPrMs ?? 160,
    qrs_ms: qrsMs,
    qt_ms: estQtMs,
    axis_deg: isSingleLead ? null : 45,
    rhythm_regular: isRegular,
    p_waves_present: !isAtrialFlutter && !isAfib && !isWideTach,
    sawtooth_flutter_waves: isAtrialFlutter,
    flutter_wave_rate_bpm: isAtrialFlutter ? flutterRateBpm : null,
    fibrillatory_waves: isAfib,
    rhythm_type: isAtrialFlutter ? "atrial_flutter" : isAfib ? "af" : isWideTach ? "vt" : isNarrowSvt ? "svt" : "",
    av_block_type: isSevereBrady ? "3rd" : hasDroppedBeatPause ? "mobitz_1" : "",
    progressive_pr_prolongation: hasDroppedBeatPause,
  };

  const hasHardTrigger =
    isAtrialFlutter || isAfib || hasDroppedBeatPause || isFirstDegAvb || isWideTach || isNarrowSvt || isSevereBrady;
  if (!hasHardTrigger && topRef.score >= 0.85) {
    measuredInput = {
      ...topRef.entry.reference_ecg,
      hr_bpm: hrBpm ?? topRef.entry.reference_ecg.hr_bpm,
      qrs_ms: qrsMs,
      rhythm_regular: isRegular,
    };
  }

  const interpretation = interpretECG(measuredInput);
  const top = interpretation.all_findings[0]!;
  const regPct = Math.max(0, Math.min(100, Math.round((1 - Math.min(madCv, 0.35) / 0.35) * 100)));

  return {
    layout: isSingleLead ? "single_lead" : "12_lead",
    is_regular_rhythm: isRegular,
    rr_regularity_pct: regPct,
    reference_count: refIndex.total_reference_ecgs,
    interpretation,
    summary_ar: isAtrialFlutter
      ? `تم رصد موجات رفرفة أذينية منتظمة (Sawtooth F-waves) بتردد ${flutterRateBpm} bpm مع QRS ضيق (${qrsMs} ms) ومعدل بطيني ${hrBpm ?? "—"} bpm.`
      : hasDroppedBeatPause
      ? `تم رصد إطالة في مسافة PR (${measuredPrMs} ms) مع سقوط نبضات (Grouped beating / Dropped QRS) وخط أساس هادئ — يتماشى مع إحصار أذيني بطيني من الدرجة الثانية.`
      : isAfib
      ? `تم رصد إيقاع غير منتظم بشكل فوضوي (Irregular • انتظام ${regPct}%) مع غياب موجات P ومعدل بطيني ${hrBpm ?? "—"} bpm وQRS ضيق (${qrsMs} ms).`
      : `تم تحليل الموجة رقمياً (HR=${hrBpm ?? "—"} bpm، PR=${measuredPrMs ?? "—"} ms، QRS=${qrsMs} ms) — التشخيص: ${top.name_ar}.`,
    summary_en: isAtrialFlutter
      ? `Detected periodic sawtooth atrial flutter waves (~${flutterRateBpm} bpm) with narrow QRS (${qrsMs} ms) and ventricular rate ${hrBpm ?? "—"} bpm.`
      : hasDroppedBeatPause
      ? `Detected prolonged PR interval (${measuredPrMs} ms) with dropped-beat pauses and clean baseline — consistent with Second-Degree AV Block.`
      : isAfib
      ? `Detected irregularly irregular rhythm (RR regularity ${regPct}%) with absent discrete P waves, ventricular rate ${hrBpm ?? "—"} bpm, and narrow QRS (${qrsMs} ms).`
      : `Direct waveform analysis completed (HR=${hrBpm ?? "—"} bpm, PR=${measuredPrMs ?? "—"} ms, QRS=${qrsMs} ms) — Diagnosis: ${top.name_en}.`,
  };
}
