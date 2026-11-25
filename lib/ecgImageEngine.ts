"use client";

import rawRefIndex from "@/ecg_engine/ecg_reference_index.json";
import { ECGInput, ECGInterpretationResult, ECGQuality, interpretECG } from "@/lib/ecgRules";
import * as S from "@/lib/ecgSignal";

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

export interface SimilarReference {
  id: string;
  title: string;
  expected_code: string;
  source_book: string;
  similarity: number;
}

export interface AnalyzeOptions {
  stripDurationSec?: number;
  pxPerMm?: number;
  paperSpeedMmS?: 25 | 50;
  gainMmMv?: 10 | 5;
}

export interface ExtractedMorphology {
  st_elevation_leads: string[];
  st_elevation_mm: Record<string, number>;
  st_depression_leads: string[];
  t_wave_inversion_leads: string[];
  peaked_t_waves: boolean;
  max_t_wave_amplitude_mm: number | null;
  brugada_pattern: "type_1" | null;
  pacing_spikes: boolean;
  qt_ms: number | null;
  sawtooth_flutter: boolean;
  deskew_deg: number;
}

export interface LocalECGAnalysisResult {
  layout: "single_lead" | "12_lead";
  is_regular_rhythm: boolean;
  rhythm_label: "Regular" | "Irregular" | "Indeterminate";
  rr_regularity_pct: number;
  reference_count: number;
  extracted_input: ECGInput;
  extracted_morphology: ExtractedMorphology;
  interpretation: ECGInterpretationResult;
  quality: ECGQuality;
  similar_references: SimilarReference[];
  summary_ar: string;
  summary_en: string;
}

const refIndex = rawRefIndex as { total_reference_ecgs: number; entries: RefEntry[] };
const MAX_WIDTH = 2000;

const GRID_12_LEADS: string[][] = [
  ["I", "aVR", "V1", "V4"],
  ["II", "aVL", "V2", "V5"],
  ["III", "aVF", "V3", "V6"],
];

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to decode ECG image"));
    img.src = dataUrl;
  });
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

function gridProfiles(rgba: Uint8ClampedArray, W: number, H: number): { col: number[]; row: number[] } {
  const lum = (i: number) => 255 - (rgba[i * 4]! + rgba[i * 4 + 1]! + rgba[i * 4 + 2]!) / 3;
  const col = new Array<number>(W);
  const buf: number[] = [];
  for (let x = 0; x < W; x++) {
    buf.length = 0;
    for (let y = 0; y < H; y += 2) buf.push(lum(y * W + x));
    col[x] = S.median(buf);
  }
  const row = new Array<number>(H);
  for (let y = 0; y < H; y++) {
    buf.length = 0;
    for (let x = 0; x < W; x += 2) buf.push(lum(y * W + x));
    row[y] = S.median(buf);
  }
  return { col, row };
}

/** Fast 1D box blur (O(n)) for rolling local illumination/shadow estimation. */
function boxBlur1D(src: Float32Array, len: number, stride: number, offset: number, radius: number, dst: Float32Array) {
  const win = radius * 2 + 1;
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const idx = Math.max(0, Math.min(len - 1, i));
    sum += src[offset + idx * stride]!;
  }
  for (let i = 0; i < len; i++) {
    dst[offset + i * stride] = sum / win;
    const outIdx = Math.max(0, i - radius);
    const inIdx = Math.min(len - 1, i + radius + 1);
    sum += src[offset + inIdx * stride]! - src[offset + outIdx * stride]!;
  }
}

/**
 * Upgraded V4 Ink Isolation for shaky/faded/shadowed/yellow-tinted phone photos:
 * - Red-weighted luma suppresses pink/red millimeter grid.
 * - 2D separable rolling local background removes uneven shadows.
 * - Adaptive percentile stretch rescues faded gray ink.
 */
function isolateCleanInk(rgba: Uint8ClampedArray, W: number, H: number): Float32Array {
  const luma = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const r = rgba[i * 4]!;
    const g = rgba[i * 4 + 1]!;
    const b = rgba[i * 4 + 2]!;
    luma[i] = 0.55 * r + 0.30 * g + 0.15 * b;
  }

  const rx = Math.max(15, Math.floor(W / 32));
  const ry = Math.max(10, Math.floor(H / 32));
  const tmp = new Float32Array(W * H);
  const bg = new Float32Array(W * H);
  for (let y = 0; y < H; y++) boxBlur1D(luma, W, 1, y * W, rx, tmp);
  for (let x = 0; x < W; x++) boxBlur1D(tmp, H, W, x, ry, bg);

  const diff = new Float32Array(W * H);
  const sample: number[] = [];
  for (let i = 0; i < W * H; i++) {
    let d = Math.max(0, bg[i]! - luma[i]!);
    const r = rgba[i * 4]!;
    const g = rgba[i * 4 + 1]!;
    if (r - g > 18 && r > 140) d *= 0.25; // suppress pink/red grid lines
    diff[i] = d;
    if ((i & 15) === 0) sample.push(d);
  }

  const p90 = S.percentile(sample, 90);
  const p996 = S.percentile(sample, 99.6);
  const lo = Math.max(8, p90);
  const hi = Math.max(lo + 14, p996);
  const span = hi - lo;
  const ink = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const v = (diff[i]! - lo) / span;
    ink[i] = v <= 0 ? 0 : v >= 1 ? 1 : v;
  }
  return ink;
}

/** Projection-variance tilt estimator (-4° to +4° in 0.5° steps) for tilted phone photos. */
function estimateTiltDeg(ink: Float32Array, W: number, H: number): number {
  const step = 4;
  const sw = Math.floor(W / step);
  const sh = Math.floor(H / step);
  let bestDeg = 0;
  let bestVar = -1;
  const proj = new Float32Array(sh);

  for (let deg = -4.0; deg <= 4.01; deg += 0.5) {
    proj.fill(0);
    const tanR = Math.tan((deg * Math.PI) / 180);
    for (let sy = 0; sy < sh; sy++) {
      const y = sy * step;
      for (let sx = 0; sx < sw; sx++) {
        const v = ink[y * W + sx * step]!;
        if (v < 0.15) continue;
        const ny = Math.round(sy + (sx - sw / 2) * tanR);
        if (ny >= 0 && ny < sh) proj[ny] = proj[ny]! + v;
      }
    }
    let mean = 0;
    for (let i = 0; i < sh; i++) mean += proj[i]!;
    mean /= sh;
    let vSum = 0;
    for (let i = 0; i < sh; i++) {
      const d = proj[i]! - mean;
      vSum += d * d;
    }
    if (vSum > bestVar) {
      bestVar = vSum;
      bestDeg = Math.round(deg * 10) / 10;
    }
  }
  return bestDeg;
}

/** Fast inverse-mapped bilinear deskew of the ink map. */
function deskewInk(ink: Float32Array, W: number, H: number, deg: number): Float32Array {
  if (Math.abs(deg) < 0.4) return ink;
  const out = new Float32Array(W * H);
  const rad = (-deg * Math.PI) / 180;
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);
  const cx = W / 2;
  const cy = H / 2;
  for (let y = 0; y < H; y++) {
    const dy = y - cy;
    for (let x = 0; x < W; x++) {
      const dx = x - cx;
      const sx = Math.round(dx * cosA + dy * sinA + cx);
      const sy = Math.round(-dx * sinA + dy * cosA + cy);
      if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
        out[y * W + x] = ink[sy * W + sx]!;
      }
    }
  }
  return out;
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
  const thresh = Math.max(0.006, maxSmooth * 0.22);
  const pad = Math.floor(H * 0.032);
  const minBandH = Math.floor(H * 0.025);
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

/** Extract band trace in mm with rolling baseline detrending (removes paper curve/wander). */
function extractBandSignalMm(ink: Float32Array, W: number, x0: number, x1: number, y0: number, y1: number, pxPerMmY: number): number[] {
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
      if (v > 0.14) {
        wSum += v;
        ySum += v * yi;
      }
    }
    if (wSum > 0.2) {
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
  const baseArr = new Float32Array(sw);
  boxBlur1D(yCent, sw, 1, 0, Math.max(25, Math.floor(sw * 0.06)), baseArr);
  const out = new Array<number>(sw);
  for (let xi = 0; xi < sw; xi++) out[xi] = (baseArr[xi]! - yCent[xi]!) / pxPerMmY;
  return out;
}

/** R-peak detector with R-T bigeminy un-doubling (prevents tall peaked T waves from doubling HR). */
function detectRPeaks(sig: number[], fs: number): { absS: number[]; peaks: number[] } {
  const absS = sig.map((v) => Math.abs(v));
  const qrsTh = Math.max(2.2, S.percentile(absS, 98.2) * 0.50);
  const minRr = Math.floor(0.22 * fs);
  const pks: number[] = [];
  for (let x = 3; x < absS.length - 3; x++) {
    const v = absS[x]!;
    if (v >= qrsTh && v >= absS[x - 1]! && v >= absS[x + 1]! && v >= absS[x - 2]! && v >= absS[x + 2]!) {
      if (pks.length === 0 || x - pks[pks.length - 1]! >= minRr) pks.push(x);
      else if (v > absS[pks[pks.length - 1]!]!) pks[pks.length - 1] = x;
    }
  }
  if (pks.length < 3) return { absS, peaks: pks };

  let kept: number[] = [];
  for (const p of pks) {
    const prev = kept[kept.length - 1];
    if (prev != null && absS[p]! < 0.48 * absS[prev]! && (p - prev) / fs < 0.42) continue;
    kept.push(p);
  }

  // Detect strict short-long R-T double-counting (e.g., Hyperkalemia tall peaked T waves)
  if (kept.length >= 4) {
    const evenRr: number[] = [];
    const oddRr: number[] = [];
    for (let i = 0; i < kept.length - 1; i++) {
      const dt = (kept[i + 1]! - kept[i]!) / fs;
      if (i % 2 === 0) evenRr.push(dt);
      else oddRr.push(dt);
    }
    const mEven = S.median(evenRr);
    const mOdd = S.median(oddRr);
    if (mEven >= 0.22 && mEven <= 0.42 && mOdd > 1.35 * mEven) {
      kept = kept.filter((_, idx) => idx % 2 === 0);
    } else if (mOdd >= 0.22 && mOdd <= 0.42 && mEven > 1.35 * mOdd) {
      kept = [kept[0]!, ...kept.filter((_, idx) => idx > 0 && idx % 2 === 1)];
    }
  }
  return { absS, peaks: kept };
}

function measureSegmentMorphology(sig: number[], peaks: number[], fs: number, gainScale: number, leadName: string) {
  const stVals: number[] = [];
  const tPosVals: number[] = [];
  const tNegVals: number[] = [];
  const qtVals: number[] = [];
  let spikeBeats = 0;
  let brugadaBeats = 0;

  for (const p of peaks) {
    const base0 = Math.max(0, Math.floor(p - 0.14 * fs));
    const base1 = Math.max(base0 + 1, Math.floor(p - 0.055 * fs));
    const baseSlice = sig.slice(base0, base1);
    if (baseSlice.length < 3) continue;
    const baseline = S.median(baseSlice);

    let curSt: number | null = null;
    const st0 = Math.floor(p + 0.075 * fs);
    const st1 = Math.min(sig.length - 1, Math.floor(p + 0.115 * fs));
    if (st1 > st0 + 1) {
      const stMed = (S.median(sig.slice(st0, st1)) - baseline) * gainScale;
      if (Math.abs(stMed) <= 12) {
        curSt = stMed;
        stVals.push(stMed);
      }
    }

    // RR-adaptive T-wave search window: prevents overlapping the next beat's P/QRS in tachycardia (e.g. HR 125)
    const pIdx = peaks.indexOf(p);
    const nextRrSec =
      pIdx >= 0 && pIdx < peaks.length - 1
        ? (peaks[pIdx + 1]! - p) / fs
        : pIdx > 0
          ? (p - peaks[pIdx - 1]!) / fs
          : 0.85;
    const tSearchEndSec = Math.min(0.46, Math.max(0.24, nextRrSec * 0.54));
    const qtEndLimitSec = Math.min(0.58, Math.max(0.28, nextRrSec * 0.68));

    const t0 = Math.floor(p + 0.13 * fs);
    const t1 = Math.min(sig.length - 1, Math.floor(p + tSearchEndSec * fs));
    if (t1 > t0 + 3) {
      let maxT = -999;
      let minT = 999;
      let tPeakIdx = t0;
      for (let i = t0; i <= t1; i++) {
        const rel = (sig[i]! - baseline) * gainScale;
        if (rel > maxT) {
          maxT = rel;
          if (Math.abs(rel) >= Math.abs(minT)) tPeakIdx = i;
        }
        if (rel < minT) {
          minT = rel;
          if (Math.abs(rel) > Math.abs(maxT)) tPeakIdx = i;
        }
      }
      if (maxT < 25) tPosVals.push(maxT);
      if (minT > -25) tNegVals.push(minT);
      if ((leadName === "V1" || leadName === "V2") && curSt != null && curSt >= 2.0 && minT <= -1.4) {
        brugadaBeats += 1;
      }

      // Clinical Tangent Method (steepest T-downslope extrapolated to baseline) with RR-bounded window
      const qOnset = Math.max(0, Math.floor(p - 0.038 * fs));
      const qtEndLimit = Math.min(sig.length - 1, Math.floor(p + qtEndLimitSec * fs));
      const tPeakRel = (sig[tPeakIdx]! - baseline) * gainScale;
      let tEnd = tPeakIdx;
      if (Math.abs(tPeakRel) >= 0.6) {
        let bestSlope = 0;
        let bestIdx = tPeakIdx;
        const slopeSearchEnd = Math.min(qtEndLimit - 1, tPeakIdx + Math.floor(0.16 * fs));
        for (let i = tPeakIdx + 1; i <= slopeSearchEnd; i++) {
          const cur = (sig[i]! - baseline) * gainScale;
          const prev = (sig[i - 1]! - baseline) * gainScale;
          const d = tPeakRel >= 0 ? prev - cur : cur - prev;
          if (d > bestSlope) {
            bestSlope = d;
            bestIdx = i;
          }
        }
        if (bestSlope > 0.05) {
          const yAtBest = Math.abs((sig[bestIdx]! - baseline) * gainScale);
          const stepsToZero = Math.min(Math.floor(0.12 * fs), Math.max(1, yAtBest / bestSlope));
          tEnd = Math.min(qtEndLimit, Math.round(bestIdx + stepsToZero));
        }
      } else {
        for (let i = tPeakIdx; i <= qtEndLimit; i++) {
          const rel = Math.abs((sig[i]! - baseline) * gainScale);
          tEnd = i;
          if (rel <= 0.35 && i > tPeakIdx + Math.floor(0.03 * fs)) break;
        }
      }
      const qtMs = ((tEnd - qOnset) / fs) * 1000;
      const maxPhysQt = Math.min(620, nextRrSec * 680);
      if (qtMs >= 260 && qtMs <= maxPhysQt) qtVals.push(qtMs);
    }

    // Strict Pacing Spike: narrow tall impulse (>=3.8mm) before QRS with sharp drop on BOTH sides
    const sp0 = Math.max(0, Math.floor(p - 0.045 * fs));
    const sp1 = Math.max(sp0 + 1, Math.floor(p - 0.014 * fs));
    let maxSpike = 0;
    let spikeIdx = -1;
    for (let i = sp0; i < sp1; i++) {
      const d = Math.abs((sig[i]! - baseline) * gainScale);
      if (d > maxSpike) {
        maxSpike = d;
        spikeIdx = i;
      }
    }
    if (maxSpike >= 3.8 && spikeIdx >= 2 && spikeIdx < sig.length - 2) {
      const dropL = Math.abs(sig[spikeIdx]! - sig[spikeIdx - 2]!) * gainScale;
      const dropR = Math.abs(sig[spikeIdx]! - sig[spikeIdx + 2]!) * gainScale;
      if (dropL >= 2.5 && dropR >= 2.5) spikeBeats += 1;
    }
  }

  return {
    stMm: stVals.length >= 2 ? Math.round(S.median(stVals) * 10) / 10 : null,
    tPosMm: tPosVals.length >= 2 ? Math.round(S.median(tPosVals) * 10) / 10 : null,
    tNegMm: tNegVals.length >= 2 ? Math.round(S.median(tNegVals) * 10) / 10 : null,
    qtMs: qtVals.length >= 3 ? Math.round(S.median(qtVals)) : null,
    hasPacingSpikes: peaks.length >= 2 && spikeBeats / peaks.length >= 0.5,
    hasBrugada: peaks.length >= 2 && brugadaBeats >= Math.max(1, Math.floor(peaks.length / 2)),
  };
}

export async function analyzeEcgImageLocally(dataUrl: string, opts: AnalyzeOptions = {}): Promise<LocalECGAnalysisResult> {
  const paperSpeed = opts.paperSpeedMmS ?? 25;
  const gainScale = (opts.gainMmMv ?? 10) === 5 ? 2 : 1;

  const img = await loadImage(dataUrl);
  const scale = Math.min(1, MAX_WIDTH / img.naturalWidth);
  const W = Math.max(200, Math.round(img.naturalWidth * scale));
  const H = Math.max(120, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.drawImage(img, 0, 0, W, H);
  const rgba = ctx.getImageData(0, 0, W, H).data;

  const flags: string[] = [];
  const sx0 = Math.floor(W * 0.03);
  const sx1 = Math.floor(W * 0.97);
  const stripWidth = sx1 - sx0;
  const { col, row } = gridProfiles(rgba, W, H);

  let calib: S.Calibration;
  if (opts.pxPerMm && opts.pxPerMm > 0) {
    calib = { pxPerMm: opts.pxPerMm * scale, confidence: 1, basis: "user_supplied_px_per_mm" };
  } else if (opts.stripDurationSec && opts.stripDurationSec > 0) {
    calib = { pxPerMm: stripWidth / (opts.stripDurationSec * paperSpeed), confidence: 1, basis: "user_supplied_duration" };
  } else {
    calib = S.resolveCalibration(col, W, paperSpeed);
  }
  const timingOk = calib.pxPerMm != null && calib.confidence >= 0.6;
  if (!timingOk) flags.push(`calibration_unreliable:${calib.basis}`);

  const aspect = img.naturalWidth / Math.max(1, img.naturalHeight);
  const assumedDuration = aspect < 1.35 ? 6.0 : 10.0;
  const pxPerMmX = calib.pxPerMm ?? stripWidth / (assumedDuration * paperSpeed);
  const pxPerMmY = (calib.pxPerMm != null ? S.matchScaleToReference(row, calib.pxPerMm) : null) ?? pxPerMmX;
  const durationSec = stripWidth / pxPerMmX / paperSpeed;
  const fs = stripWidth / durationSec;

  const rawInk = isolateCleanInk(rgba, W, H);
  const deskewDeg = estimateTiltDeg(rawInk, W, H);
  const ink = deskewInk(rawInk, W, H, deskewDeg);
  if (Math.abs(deskewDeg) >= 0.5) flags.push(`auto_deskew:${deskewDeg}deg`);

  const bands = detectHorizontalBands(ink, W, H);
  const isSingleLead = bands.length <= 1;

  const pickBand = (b: [number, number]) => {
    const sig = extractBandSignalMm(ink, W, sx0, sx1, b[0], b[1], pxPerMmY);
    return { sig, ...detectRPeaks(sig, fs) };
  };
  const primaryBand: [number, number] = bands.length > 0 ? bands[bands.length - 1]! : [Math.floor(H * 0.2), Math.floor(H * 0.85)];
  let chosen = pickBand(primaryBand);
  if (chosen.peaks.length < 4 && bands.length >= 2) {
    const alt = pickBand(bands[bands.length - 2]!);
    if (alt.peaks.length >= chosen.peaks.length) chosen = alt;
  }
  const { sig: sigMm, absS, peaks } = chosen;

  const rrsSec: number[] = [];
  for (let i = 1; i < peaks.length; i++) rrsSec.push((peaks[i]! - peaks[i - 1]!) / fs);
  const rr = S.analyzeRr(rrsSec);
  const rhythmOk = rr.pattern !== "indeterminate";
  if (!rhythmOk) flags.push(`too_few_beats:${peaks.length}`);

  let hrRaw: number | null = null;
  if (peaks.length >= 4 && rr.medianRr > 0) hrRaw = Math.round((60 / rr.medianRr) * 10) / 10;

  const qrsWidths: number[] = [];
  const winL = Math.floor(0.18 * fs);
  const winR = Math.floor(0.32 * fs);
  const beatSamples: number[][] = [];
  for (const p of peaks) {
    const thresh = absS[p]! * 0.22;
    const maxSpan = Math.floor(0.13 * fs);
    let l = p;
    while (l > Math.max(0, p - maxSpan) && absS[l]! > thresh) l--;
    let r = p;
    while (r < Math.min(absS.length - 1, p + maxSpan) && absS[r]! > thresh) r++;
    const wMs = ((r - l) / fs) * 1000;
    if (wMs >= 45 && wMs <= 240) qrsWidths.push(wMs);
    if (p - winL >= 0 && p + winR < sigMm.length) {
      const segLen = winL + winR;
      const res = new Array<number>(16).fill(0);
      for (let i = 0; i < 16; i++) {
        const pos = (i * (segLen - 1)) / 15;
        const i0 = Math.floor(pos);
        const i1 = Math.min(segLen - 1, i0 + 1);
        const fr = pos - i0;
        res[i] = sigMm[p - winL + i0]! * (1 - fr) + sigMm[p - winL + i1]! * fr;
      }
      const nrm = Math.sqrt(res.reduce((a, v) => a + v * v, 0)) + 1e-6;
      beatSamples.push(res.map((v) => v / nrm));
    }
  }
  const qrsRaw = qrsWidths.length >= 3 && S.percentile(qrsWidths, 75) - S.percentile(qrsWidths, 25) <= 0.35 * S.median(qrsWidths) ? Math.round(S.median(qrsWidths)) : null;
  if (qrsRaw == null) flags.push("qrs_width_unreliable");

  const beatsP = S.measureBeatsP(sigMm, peaks, fs);
  const cond = S.analyzeConduction(beatsP, rr, hrRaw);
  const plaus = S.applyPlausibility(hrRaw, qrsRaw, cond.prMedianMs);
  flags.push(...plaus.flags);
  const timingTrusted = timingOk && !plaus.flags.some((f) => f.startsWith("hr_implausible"));

  // True TP-segment flutter check (avoids P-wave & QRS harmonic false positives)
  let flutterRate: number | null = null;
  let sawtooth = false;
  if (timingTrusted && rhythmOk && peaks.length >= 4 && rr.pattern !== "irregularly_irregular") {
    const tpStds: number[] = [];
    for (let i = 0; i < peaks.length - 1; i++) {
      const t0 = peaks[i]! + Math.floor(0.38 * fs);
      const t1 = peaks[i + 1]! - Math.floor(0.22 * fs);
      if (t1 - t0 >= Math.floor(0.15 * fs)) {
        const sl = sigMm.slice(t0, t1);
        const mn = sl.reduce((a, b) => a + b, 0) / sl.length;
        const vr = sl.reduce((a, b) => a + (b - mn) * (b - mn), 0) / sl.length;
        tpStds.push(Math.sqrt(vr));
      }
    }
    const tpOsc = tpStds.length > 0 ? S.median(tpStds) : 0;
    const bl = S.maskQrs(sigMm, peaks, fs);
    const band = S.analyzeAtrialBand(bl, fs);
    if (tpOsc >= 0.6 && S.decideFlutter(band, rr, plaus.hr)) {
      sawtooth = true;
      flutterRate = band.rateBpm;
    }
  }

  /* ---- Multi-band morphology scan (ST, T, Brugada, QT, Pacing Spikes) ---- */
  let stElevationMm: Record<string, number> = {};
  let stElevationLeads: string[] = [];
  const stDepressionLeads: string[] = [];
  let twiLeads: string[] = [];
  const qtCandidates: number[] = [];
  let peakedT = false;
  let maxT: number | null = null;
  let pacingLeadsCount = 0;
  let brugadaV1V2 = 0;

  const primaryMorph = measureSegmentMorphology(sigMm, peaks, fs, gainScale, "II");
  if (primaryMorph.qtMs != null) qtCandidates.push(primaryMorph.qtMs);
  if (primaryMorph.hasPacingSpikes) pacingLeadsCount += 1;

  if (!isSingleLead && bands.length >= 3) {
    const scanRows = Math.min(3, bands.length);
    for (let rIdx = 0; rIdx < scanRows; rIdx++) {
      const b = bands[rIdx]!;
      const rowSig = extractBandSignalMm(ink, W, sx0, sx1, b[0], b[1], pxPerMmY);
      const qWidth = Math.floor(rowSig.length / 4);
      for (let colIdx = 0; colIdx < 4; colIdx++) {
        const leadName = GRID_12_LEADS[rIdx]![colIdx]!;
        const seg = rowSig.slice(colIdx * qWidth, (colIdx + 1) * qWidth);
        const segDet = detectRPeaks(seg, fs);
        if (segDet.peaks.length < 2) continue;
        const m = measureSegmentMorphology(seg, segDet.peaks, fs, gainScale, leadName);
        if (m.qtMs != null) qtCandidates.push(m.qtMs);
        if (m.hasPacingSpikes) pacingLeadsCount += 1;
        if (m.hasBrugada) brugadaV1V2 += 1;
        const steThresh = leadName === "V2" || leadName === "V3" ? 1.8 : 1.4;
        if (m.stMm != null && m.stMm >= steThresh) {
          stElevationMm[leadName] = m.stMm;
          stElevationLeads.push(leadName);
        } else if (m.stMm != null && m.stMm <= -1.0) {
          stDepressionLeads.push(leadName);
        }
        if (m.tNegMm != null && m.tNegMm <= -1.5 && leadName !== "aVR" && (m.tPosMm == null || Math.abs(m.tNegMm) > m.tPosMm)) {
          twiLeads.push(leadName);
        }
        if (m.tPosMm != null && m.tPosMm >= 6.8) {
          peakedT = true;
          maxT = maxT == null ? m.tPosMm : Math.max(maxT, m.tPosMm);
        }
      }
    }
  } else {
    if (primaryMorph.stMm != null && primaryMorph.stMm >= 1.4) {
      stElevationMm["II"] = primaryMorph.stMm;
      stElevationLeads.push("II");
    } else if (primaryMorph.stMm != null && primaryMorph.stMm <= -1.0) {
      stDepressionLeads.push("II");
    }
    if (primaryMorph.tNegMm != null && primaryMorph.tNegMm <= -1.5) twiLeads.push("II");
    if (primaryMorph.tPosMm != null && primaryMorph.tPosMm >= 6.8) {
      peakedT = true;
      maxT = primaryMorph.tPosMm;
    }
  }

  // Multi-lead pacing consensus (prevents single-lead noise from faking pacemaker)
  const pacingSpikes =
    (isSingleLead ? pacingLeadsCount >= 1 : pacingLeadsCount >= 3) &&
    (plaus.qrsMs == null || plaus.qrsMs >= 115);

  if (pacingSpikes) {
    stElevationMm = {};
    stElevationLeads = [];
    twiLeads = [];
  }

  const otherSte = stElevationLeads.filter((l) => l !== "V1" && l !== "V2");
  const brugadaPattern: "type_1" | null = !pacingSpikes && brugadaV1V2 >= 1 && otherSte.length === 0 ? "type_1" : null;

  const qtMeasured = timingTrusted && qtCandidates.length > 0 ? Math.round(S.median(qtCandidates)) : null;

  const quality: ECGQuality = {
    timing_ok: timingTrusted,
    rhythm_ok: rhythmOk,
    calibration: {
      px_per_mm: calib.pxPerMm,
      confidence: Math.round(calib.confidence * 100) / 100,
      basis: calib.basis,
      strip_duration_sec: calib.pxPerMm != null ? Math.round(durationSec * 10) / 10 : null,
    },
    n_beats: peaks.length,
    n_bands: bands.length,
    flags,
  };

  const input: ECGInput = {
    age: null,
    sex: null,
    hr_bpm: timingTrusted ? plaus.hr : null,
    pr_ms: timingTrusted ? plaus.prMs : null,
    qrs_ms: timingTrusted ? plaus.qrsMs : null,
    qt_ms: qtMeasured,
    axis_deg: null,
    rr_pattern: rr.pattern,
    st_elevation_mm: Object.keys(stElevationMm).length ? stElevationMm : undefined,
    t_wave_inversion_leads: twiLeads.length ? twiLeads : undefined,
    peaked_t_waves: peakedT || undefined,
    max_t_wave_amplitude_mm: maxT,
    brugada_pattern: brugadaPattern ?? undefined,
    coved_ste_v1_v2: brugadaPattern === "type_1" ? true : undefined,
    pacing_spikes: pacingSpikes || undefined,
    quality,
  };
  if (rr.pattern === "regular") input.rhythm_regular = true;
  else if (rr.pattern === "irregularly_irregular" || rr.pattern === "pauses") input.rhythm_regular = false;
  if (cond.pWavesPresent != null) input.p_waves_present = cond.pWavesPresent;
  if (cond.mobitzType) {
    input.av_block_type = cond.mobitzType;
    if (cond.mobitzType === "mobitz_1") input.progressive_pr_prolongation = true;
  }
  if (sawtooth) {
    input.sawtooth_flutter_waves = true;
    input.flutter_wave_rate_bpm = flutterRate;
  }

  const interpretation = interpretECG(input);

  const beat16 = new Array<number>(16).fill(0);
  if (beatSamples.length > 0) {
    for (let i = 0; i < 16; i++) beat16[i] = S.median(beatSamples.map((b) => b[i]!));
    const nrm = Math.sqrt(beat16.reduce((a, v) => a + v * v, 0)) + 1e-6;
    for (let i = 0; i < 16; i++) beat16[i] = Math.round((beat16[i]! / nrm) * 1000);
  }
  const similar: SimilarReference[] = beatSamples.length
    ? refIndex.entries
        .map((entry) => ({ entry, score: Math.max(0, cosineSim(beat16, entry.sig.beat16)) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 3)
        .map(({ entry, score }) => ({
          id: entry.id,
          title: entry.title,
          expected_code: entry.expected_code,
          source_book: entry.source_book,
          similarity: Math.round(score * 100) / 100,
        }))
    : [];

  const top = interpretation.all_findings[0]!;
  const fmt = (v: number | null, unit: string) => (v == null ? "n/a" : `${v} ${unit}`);
  const m = interpretation.measurements;
  const regPct = rhythmOk ? Math.max(0, Math.min(100, Math.round((1 - Math.min(rr.cv, 0.35) / 0.35) * 100))) : 0;
  const rhythmLabel: LocalECGAnalysisResult["rhythm_label"] =
    rr.pattern === "regular" ? "Regular" : rr.pattern === "irregularly_irregular" || rr.pattern === "pauses" ? "Irregular" : "Indeterminate";

  const tiltNoteEn = Math.abs(deskewDeg) >= 0.5 ? ` (deskewed ${deskewDeg}°)` : "";
  const tiltNoteAr = Math.abs(deskewDeg) >= 0.5 ? ` (تم تعديل الميل ${deskewDeg}°)` : "";
  const summary_en = `Auto-measured${tiltNoteEn}: HR ${fmt(m.hr_bpm, "bpm")}, PR ${fmt(m.pr_ms, "ms")}, QRS ${fmt(m.qrs_ms, "ms")}, QT ${fmt(m.qt_ms, "ms")}, rhythm ${rhythmLabel.toLowerCase()}. Leading rule: ${top.name_en}.`;
  const summary_ar = `قياسات تلقائية${tiltNoteAr}: HR ${fmt(m.hr_bpm, "bpm")}، PR ${fmt(m.pr_ms, "ms")}، QRS ${fmt(m.qrs_ms, "ms")}، QT ${fmt(m.qt_ms, "ms")}، الإيقاع ${rhythmLabel === "Regular" ? "منتظم" : rhythmLabel === "Irregular" ? "غير منتظم" : "غير محدد"}. أبرز نتيجة: ${top.name_ar}.`;

  return {
    layout: isSingleLead ? "single_lead" : "12_lead",
    is_regular_rhythm: rr.pattern === "regular",
    rhythm_label: rhythmLabel,
    rr_regularity_pct: regPct,
    reference_count: refIndex.total_reference_ecgs,
    extracted_input: input,
    extracted_morphology: {
      st_elevation_leads: stElevationLeads,
      st_elevation_mm: stElevationMm,
      st_depression_leads: stDepressionLeads,
      t_wave_inversion_leads: twiLeads,
      peaked_t_waves: peakedT,
      max_t_wave_amplitude_mm: maxT,
      brugada_pattern: brugadaPattern,
      pacing_spikes: pacingSpikes,
      qt_ms: qtMeasured,
      sawtooth_flutter: sawtooth,
      deskew_deg: deskewDeg,
    },
    interpretation,
    quality,
    similar_references: similar,
    summary_ar,
    summary_en,
  };
}
