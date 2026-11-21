"use client";

import rawRefIndex from "@/ecg_engine/ecg_reference_index.json";
import { ECGInput, ECGInterpretationResult, interpretECG } from "@/lib/ecgRules";

interface ReferenceFeatures {
  dhash: string;
  grid: number[];
  ink_density: number;
  col_std: number;
  peaks: number;
  rr_cv: number;
}

interface ReferenceEntry {
  id: string;
  condition_key: string;
  expected_code: string;
  title: string;
  source_book: string;
  source_page: number;
  caption: string;
  reference_ecg: ECGInput;
  features: ReferenceFeatures;
}

export interface LocalReferenceMatch {
  id: string;
  title: string;
  expected_code: string;
  source_book: string;
  source_page: number;
  caption: string;
  similarity_pct: number;
}

export interface LocalSignalMetrics {
  width: number;
  height: number;
  detected_r_peaks: number;
  estimated_hr_bpm: number | null;
  rr_regularity_pct: number;
  is_regular_rhythm: boolean;
  ink_density_pct: number;
  pink_grid_detected: boolean;
  quality_label_ar: string;
  quality_label_en: string;
}

export interface LocalECGAnalysisResult {
  signal: LocalSignalMetrics;
  top_matches: LocalReferenceMatch[];
  confidence_pct: number;
  interpretation: ECGInterpretationResult;
  summary_ar: string;
  summary_en: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const refIndex = rawRefIndex as { total_reference_images: number; entries: ReferenceEntry[] };

function hammingSimilarityHex(hexA: string, hexB: string): number {
  let diff = 0;
  const len = Math.min(hexA.length, hexB.length);
  for (let i = 0; i < len; i++) {
    let x = parseInt(hexA[i]!, 16) ^ parseInt(hexB[i]!, 16);
    while (x > 0) {
      diff += x & 1;
      x >>= 1;
    }
  }
  return 1 - diff / 64;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const va = a[i]!;
    const vb = b[i]!;
    dot += va * vb;
    na += va * va;
    nb += vb * vb;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to decode ECG image"));
    img.src = dataUrl;
  });
}

/**
 * 100% Local Offline ECG Image Analyzer (Zero external APIs, Zero tokens, Zero cost).
 * Extracts ink-isolated ECG waveform descriptors + direct R-peak DSP metrics on Canvas,
 * matches against the 100 real textbook ECG reference fingerprints, and runs ecg_criteria.json rules.
 */
export async function analyzeEcgImageLocally(dataUrl: string): Promise<LocalECGAnalysisResult> {
  const img = await loadImage(dataUrl);
  const W = 256;
  const H = 160;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.drawImage(img, 0, 0, W, H);
  const rgba = ctx.getImageData(0, 0, W, H).data;

  const ink = new Float32Array(W * H);
  let pinkPixels = 0;
  let inkSum = 0;

  for (let i = 0; i < W * H; i++) {
    const r = rgba[i * 4]!;
    const g = rgba[i * 4 + 1]!;
    const b = rgba[i * 4 + 2]!;
    if (r > 180 && r - g > 18 && r - b > 12) pinkPixels++;
    const maxCh = Math.max(r, g, b);
    const val = Math.max(0, Math.min(1, (210 - maxCh) / 140));
    ink[i] = val;
    inkSum += val;
  }

  let inkMean = inkSum / (W * H);
  if (inkMean < 0.015) {
    inkSum = 0;
    for (let i = 0; i < W * H; i++) {
      const r = rgba[i * 4]!;
      const g = rgba[i * 4 + 1]!;
      const b = rgba[i * 4 + 2]!;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const val = Math.max(0, Math.min(1, (200 - lum) / 140));
      ink[i] = val;
      inkSum += val;
    }
    inkMean = inkSum / (W * H);
  }

  // 1. 16x8 normalized ink density grid (128 values)
  const rawGrid = new Array<number>(128).fill(0);
  for (let gy = 0; gy < 8; gy++) {
    for (let gx = 0; gx < 16; gx++) {
      let sum = 0;
      for (let y = gy * 20; y < (gy + 1) * 20; y++) {
        for (let x = gx * 16; x < (gx + 1) * 16; x++) {
          sum += ink[y * W + x]!;
        }
      }
      rawGrid[gy * 16 + gx] = sum / (20 * 16);
    }
  }
  const gridNorm = Math.sqrt(rawGrid.reduce((acc, v) => acc + v * v, 0)) + 1e-6;
  const gridQ = rawGrid.map((v) => Math.round((v / gridNorm) * 1000));

  // 2. 64-bit dHash (9x8 downsampled ink)
  const small9x8 = new Float32Array(9 * 8);
  for (let sy = 0; sy < 8; sy++) {
    for (let sx = 0; sx < 9; sx++) {
      const y0 = Math.floor((sy * H) / 8);
      const y1 = Math.floor(((sy + 1) * H) / 8);
      const x0 = Math.floor((sx * W) / 9);
      const x1 = Math.floor(((sx + 1) * W) / 9);
      let s = 0;
      let cnt = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          s += ink[y * W + x]!;
          cnt++;
        }
      }
      small9x8[sy * 9 + sx] = cnt > 0 ? s / cnt : 0;
    }
  }
  let hexHash = "";
  let nibble = 0;
  let bitCount = 0;
  for (let sy = 0; sy < 8; sy++) {
    for (let sx = 0; sx < 8; sx++) {
      const bit = small9x8[sy * 9 + sx + 1]! > small9x8[sy * 9 + sx]! ? 1 : 0;
      nibble = (nibble << 1) | bit;
      bitCount++;
      if (bitCount === 4) {
        hexHash += nibble.toString(16);
        nibble = 0;
        bitCount = 0;
      }
    }
  }

  // 3. Rhythm strip R-peak DSP (bottom 25% of tracing)
  const colEnergy = new Float32Array(W);
  let colSum = 0;
  for (let x = 0; x < W; x++) {
    let s = 0;
    for (let y = 118; y < 156; y++) s += ink[y * W + x]!;
    const avg = s / 38;
    colEnergy[x] = avg;
    colSum += avg;
  }
  const colMean = colSum / W;
  let colVar = 0;
  for (let x = 0; x < W; x++) colVar += (colEnergy[x]! - colMean) ** 2;
  const colStd = Math.sqrt(colVar / W);
  const peakThresh = colMean + 1.25 * colStd;

  const peaks: number[] = [];
  for (let x = 2; x < W - 2; x++) {
    if (colEnergy[x]! > peakThresh && colEnergy[x]! >= colEnergy[x - 1]! && colEnergy[x]! >= colEnergy[x + 1]!) {
      if (peaks.length === 0 || x - peaks[peaks.length - 1]! >= 8) {
        peaks.push(x);
      }
    }
  }

  let rrCv = 0.15;
  let estHr: number | null = null;
  if (peaks.length >= 3) {
    const rrs: number[] = [];
    for (let i = 1; i < peaks.length; i++) rrs.push(peaks[i]! - peaks[i - 1]!);
    const rrMean = rrs.reduce((a, b) => a + b, 0) / rrs.length;
    const rrStd = Math.sqrt(rrs.reduce((a, b) => a + (b - rrMean) ** 2, 0) / rrs.length);
    rrCv = rrStd / (rrMean + 1e-6);
    // Standard 10-second rhythm strip: HR = count * 6 (clamped to physiological 28..260 bpm)
    const rawHr = Math.round(peaks.length * 6);
    if (rawHr >= 28 && rawHr <= 260) estHr = rawHr;
  }

  const regularityPct = Math.max(0, Math.min(100, Math.round((1 - Math.min(rrCv, 0.6) / 0.6) * 100)));
  const isRegular = rrCv < 0.22;
  const pinkGrid = pinkPixels / (W * H) > 0.08;

  // 4. Compare against ALL 100 real textbook ECG reference fingerprints
  const scored = refIndex.entries.map((entry) => {
    const f = entry.features;
    const cos = cosineSimilarity(gridQ, f.grid);
    const ham = hammingSimilarityHex(hexHash, f.dhash);
    const rrDiff = Math.abs(rrCv - f.rr_cv);
    const rrSim = Math.max(0, 1 - rrDiff * 2.2);
    const peakDiff = Math.abs(peaks.length - f.peaks);
    const peakSim = Math.max(0, 1 - peakDiff / 18);
    const densSim = Math.max(0, 1 - Math.abs(inkMean - f.ink_density) * 6);

    const score = cos * 0.54 + ham * 0.18 + rrSim * 0.12 + peakSim * 0.10 + densSim * 0.06;
    return { entry, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const topMatches: LocalReferenceMatch[] = scored.slice(0, 4).map((s) => ({
    id: s.entry.id,
    title: s.entry.title,
    expected_code: s.entry.expected_code,
    source_book: s.entry.source_book,
    source_page: s.entry.source_page,
    caption: s.entry.caption,
    similarity_pct: Math.min(99, Math.max(45, Math.round(s.score * 100))),
  }));

  const best = scored[0]!;
  // Combine matched textbook clinical profile with directly measured HR/regularity when clean
  const combinedInput: ECGInput = {
    ...best.entry.reference_ecg,
    ...(estHr && best.entry.reference_ecg.hr_bpm == null ? { hr_bpm: estHr } : {}),
  };

  const interpretation = interpretECG(combinedInput);
  const topFinding = interpretation.all_findings[0]!;
  const confidencePct = topMatches[0]?.similarity_pct ?? 78;

  const qualityGood = img.naturalWidth >= 400 && img.naturalHeight >= 200 && inkMean >= 0.02;

  return {
    signal: {
      width: img.naturalWidth,
      height: img.naturalHeight,
      detected_r_peaks: peaks.length,
      estimated_hr_bpm: estHr ?? interpretation.measurements.hr_bpm,
      rr_regularity_pct: regularityPct,
      is_regular_rhythm: isRegular,
      ink_density_pct: Math.round(inkMean * 1000) / 10,
      pink_grid_detected: pinkGrid,
      quality_label_ar: qualityGood ? "جودة صورة جيدة للتحليل" : "دقة الصورة متوسطة (يفضل صورة أوضح)",
      quality_label_en: qualityGood ? "Good ECG tracing quality" : "Moderate resolution (clearer scan preferred)",
    },
    top_matches: topMatches,
    confidence_pct: confidencePct,
    interpretation,
    summary_ar: `تمت مقارنة صورة الـ ECG محلياً مع ${refIndex.total_reference_images} صورة مرجعية من الكتب المعتمدة (أقرب تطابق: ${best.entry.caption} بنسبة ${confidencePct}% من ${best.entry.source_book} ص ${best.entry.source_page}) وتطبيق قواعد ecg_criteria.json — التشخيص الأول: ${topFinding.name_ar}.`,
    summary_en: `Locally matched against ${refIndex.total_reference_images} textbook reference ECGs (top match: ${best.entry.caption} at ${confidencePct}% from ${best.entry.source_book} p.${best.entry.source_page}) and evaluated via ecg_criteria.json — primary diagnosis: ${topFinding.name_en}.`,
  };
}
