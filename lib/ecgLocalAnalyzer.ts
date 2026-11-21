"use client";

import rawRefIndex from "@/ecg_engine/ecg_reference_index.json";
import { ECGInput, ECGInterpretationResult, ECGQuality, interpretECG } from "@/lib/ecgRules";
import * as S from "@/lib/ecgSignal";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */
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

/** Atlas matches are HINTS only. They never feed the diagnosis. */
export interface SimilarReference {
  id: string;
  title: string;
  expected_code: string;
  source_book: string;
  source_page: number;
  similarity: number; // 0..1, weak visual similarity — not diagnostic
}

export interface AnalyzeOptions {
  /** Real duration (seconds) of the strip, if the user knows it. Overrides grid calibration. */
  stripDurationSec?: number;
  /** Real pixels-per-mm of the (natural-size) image, if known. */
  pxPerMm?: number;
}

export interface LocalECGAnalysisResult {
  layout: "single_lead" | "12_lead";
  /** true ONLY when the rhythm was positively classified as regular */
  is_regular_rhythm: boolean;
  /** "Regular" | "Irregular" | "Indeterminate" — show this instead of a binary label */
  rhythm_label: "Regular" | "Irregular" | "Indeterminate";
  rr_regularity_pct: number;
  reference_count: number;
  interpretation: ECGInterpretationResult;
  quality: ECGQuality;
  similar_references: SimilarReference[];
  summary_ar: string;
  summary_en: string;
}

const refIndex = rawRefIndex as { total_reference_ecgs: number; entries: RefEntry[] };
const MAX_WIDTH = 2000;
const PAPER_SPEED_MM_S = 25;

/* ------------------------------------------------------------------ */
/* Image helpers                                                       */
/* ------------------------------------------------------------------ */
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

/** Column / row profiles of the paper grid (median across the other axis removes the ECG trace). */
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

/** Adaptive ink isolation: removes pink/red AND gray/black grid lines. */
function isolateCleanInk(rgba: Uint8ClampedArray, W: number, H: number): Float32Array {
  const darkness = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const maxCh = Math.max(rgba[i * 4]!, rgba[i * 4 + 1]!, rgba[i * 4 + 2]!);
    darkness[i] = 255 - maxCh;
  }
  const rowBg = new Float32Array(H);
  const sampleRow = new Array<number>(Math.floor(W / 4));
  for (let y = 0; y < H; y++) {
    let idx = 0;
    for (let x = 0; x < W - 3; x += 4) sampleRow[idx++] = darkness[y * W + x]!;
    rowBg[y] = S.percentile(sampleRow.slice(0, idx), 70);
  }
  const colBg = new Float32Array(W);
  const sampleCol = new Array<number>(Math.floor(H / 4));
  for (let x = 0; x < W; x++) {
    let idx = 0;
    for (let y = 0; y < H - 3; y += 4) sampleCol[idx++] = darkness[y * W + x]!;
    colBg[x] = S.percentile(sampleCol.slice(0, idx), 70);
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
  const p94 = S.percentile(flatSample, 94);
  const p995 = S.percentile(flatSample, 99.5);
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

/** Trace → signal in mm (vertical scale supplied separately from the horizontal one). */
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
  const baseY = S.median(Array.from(yCent));
  const out = new Array<number>(sw);
  for (let xi = 0; xi < sw; xi++) out[xi] = (baseY - yCent[xi]!) / pxPerMmY;
  return out;
}

/** Peak detection + removal of T-wave-like secondary peaks. */
function detectRPeaks(sig: number[], fs: number): { absS: number[]; peaks: number[] } {
  const absS = sig.map((v) => Math.abs(v));
  const qrsTh = Math.max(2.6, S.percentile(absS, 98.5) * 0.56);
  const minRr = Math.floor(0.3 * fs);
  const pks: number[] = [];
  for (let x = 3; x < absS.length - 3; x++) {
    const v = absS[x]!;
    if (v >= qrsTh && v >= absS[x - 1]! && v >= absS[x + 1]! && v >= absS[x - 2]! && v >= absS[x + 2]!) {
      if (pks.length === 0 || x - pks[pks.length - 1]! >= minRr) pks.push(x);
      else if (v > absS[pks[pks.length - 1]!]!) pks[pks.length - 1] = x;
    }
  }
  if (pks.length < 3) return { absS, peaks: pks };
  const medAmp = S.median(pks.map((p) => absS[p]!));
  const kept: number[] = [];
  for (const p of pks) {
    const prev = kept[kept.length - 1];
    const tLike = prev != null && absS[p]! < 0.45 * medAmp && (p - prev) / fs < 0.45 && absS[prev]! > 1.5 * absS[p]!;
    if (!tLike) kept.push(p);
  }
  return { absS, peaks: kept };
}

/* ------------------------------------------------------------------ */
/* Main entry                                                          */
/* ------------------------------------------------------------------ */
export async function analyzeEcgImageLocally(dataUrl: string, opts: AnalyzeOptions = {}): Promise<LocalECGAnalysisResult> {
  const img = await loadImage(dataUrl);

  // Preserve aspect ratio (the old code stretched every image to 1000x560 and distorted both axes).
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

  /* ---- 1. calibration (measured from the paper grid; never guessed) ---- */
  const sx0 = Math.floor(W * 0.04);
  const sx1 = Math.floor(W * 0.96);
  const stripWidth = sx1 - sx0;
  const { col, row } = gridProfiles(rgba, W, H);

  let calib: S.Calibration;
  if (opts.pxPerMm && opts.pxPerMm > 0) {
    calib = { pxPerMm: opts.pxPerMm * scale, confidence: 1, basis: "user_supplied_px_per_mm" };
  } else if (opts.stripDurationSec && opts.stripDurationSec > 0) {
    calib = { pxPerMm: stripWidth / (opts.stripDurationSec * PAPER_SPEED_MM_S), confidence: 1, basis: "user_supplied_duration" };
  } else {
    calib = S.resolveCalibration(col, W, PAPER_SPEED_MM_S);
  }
  const timingOk = calib.pxPerMm != null && calib.confidence >= 0.6;
  if (!timingOk) flags.push(`calibration_unreliable:${calib.basis}`);

  // Fallback scale is used ONLY for internal peak detection; absolute values are withheld when !timingOk.
  const aspect = img.naturalWidth / Math.max(1, img.naturalHeight);
  const assumedDuration = aspect < 1.35 ? 6.0 : 10.0;
  const pxPerMmX = calib.pxPerMm ?? stripWidth / (assumedDuration * PAPER_SPEED_MM_S);
  const pxPerMmY = (calib.pxPerMm != null ? S.matchScaleToReference(row, calib.pxPerMm) : null) ?? pxPerMmX;
  const durationSec = stripWidth / pxPerMmX / PAPER_SPEED_MM_S;
  const fs = stripWidth / durationSec; // samples (px) per second

  /* ---- 2. trace extraction ---- */
  const ink = isolateCleanInk(rgba, W, H);
  const bands = detectHorizontalBands(ink, W, H);
  const isSingleLead = bands.length <= 1;
  if (bands.length >= 3) flags.push("multi_lead_page: rhythm taken from one band, cross-lead segments may affect RR");

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

  /* ---- 3. rhythm (scale-free) ---- */
  const rrsSec: number[] = [];
  for (let i = 1; i < peaks.length; i++) rrsSec.push((peaks[i]! - peaks[i - 1]!) / fs);
  const rr = S.analyzeRr(rrsSec);
  const rhythmOk = rr.pattern !== "indeterminate";
  if (!rhythmOk) flags.push(`too_few_beats:${peaks.length}`);

  /* ---- 4. absolute measurements (only with calibrated time axis) ---- */
  let hrRaw: number | null = null;
  if (peaks.length >= 4 && rr.medianRr > 0) hrRaw = Math.round((60 / rr.medianRr) * 10) / 10;

  const qrsWidths: number[] = [];
  const winL = Math.floor(0.18 * fs);
  const winR = Math.floor(0.32 * fs);
  const beatSamples: number[][] = [];
  for (const p of peaks) {
    const thresh = absS[p]! * 0.25;
    const maxSpan = Math.floor(0.14 * fs);
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
  if (!timingTrusted && timingOk) flags.push("calibration_suspect: implausible heart rate");

  /* ---- 5. atrial activity band (only when timing is trusted) ---- */
  let flutterRate: number | null = null;
  let sawtooth = false;
  if (timingTrusted && rhythmOk && peaks.length >= 4) {
    const bl = S.maskQrs(sigMm, peaks, fs);
    const band = S.analyzeAtrialBand(bl, fs);
    sawtooth = S.decideFlutter(band, rr, plaus.hr);
    if (sawtooth) flutterRate = band.rateBpm;
  }

  /* ---- 6. quality ---- */
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

  /* ---- 7. build the input: ONLY measured fields, everything else stays undefined/null ---- */
  const input: ECGInput = {
    age: null, // unknown — never guessed
    sex: null,
    hr_bpm: timingTrusted ? plaus.hr : null,
    pr_ms: timingTrusted ? plaus.prMs : null,
    qrs_ms: timingTrusted ? plaus.qrsMs : null,
    qt_ms: null, // not measurable from this pipeline
    axis_deg: null, // needs limb-lead data
    rr_pattern: rr.pattern,
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

  /* ---- 8. atlas: similarity hints only (never overrides measurements) ---- */
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
          source_page: entry.source_page,
          similarity: Math.round(score * 100) / 100,
        }))
    : [];

  /* ---- 9. summaries are built from what was actually measured ---- */
  const top = interpretation.all_findings[0]!;
  const fmt = (v: number | null, unit: string) => (v == null ? "n/a" : `${v} ${unit}`);
  const m = interpretation.measurements;
  const regPct = rhythmOk ? Math.max(0, Math.min(100, Math.round((1 - Math.min(rr.cv, 0.35) / 0.35) * 100))) : 0;
  const rhythmLabel: LocalECGAnalysisResult["rhythm_label"] =
    rr.pattern === "regular" ? "Regular" : rr.pattern === "irregularly_irregular" || rr.pattern === "pauses" ? "Irregular" : "Indeterminate";

  const calText = timingTrusted ? `grid-calibrated (${calib.basis})` : "time scale NOT verified";
  const summary_en =
    `Measured from image (${calText}): HR ${fmt(m.hr_bpm, "bpm")}, PR ${fmt(m.pr_ms, "ms")}, QRS ${fmt(m.qrs_ms, "ms")}, ` +
    `rhythm ${rhythmLabel.toLowerCase()}, ${peaks.length} beats. Leading finding: ${top.name_en} (confidence: ${top.confidence}).`;
  const summary_ar =
    `القياسات من الصورة (${timingTrusted ? "معايرة من الشبكة" : "مقياس الزمن غير مُتحقَّق منه"}): ` +
    `HR ${fmt(m.hr_bpm, "bpm")}، PR ${fmt(m.pr_ms, "ms")}، QRS ${fmt(m.qrs_ms, "ms")}، ` +
    `الإيقاع ${rhythmLabel === "Regular" ? "منتظم" : rhythmLabel === "Irregular" ? "غير منتظم" : "غير محدد"}، عدد النبضات ${peaks.length}. ` +
    `أبرز نتيجة: ${top.name_ar} (الثقة: ${top.confidence}).`;

  return {
    layout: isSingleLead ? "single_lead" : "12_lead",
    is_regular_rhythm: rr.pattern === "regular",
    rhythm_label: rhythmLabel,
    rr_regularity_pct: regPct,
    reference_count: refIndex.total_reference_ecgs,
    interpretation,
    quality,
    similar_references: similar,
    summary_ar,
    summary_en,
  };
}
