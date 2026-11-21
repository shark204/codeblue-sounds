/**
 * ecgSignal.ts — pure signal helpers (no DOM, no React).
 *
 * Design policy for the whole engine:
 *   1. Never fabricate a value. If a quantity cannot be measured, return null / "unknown".
 *   2. Absolute time/amplitude values are only trusted when the paper grid was calibrated.
 *   3. Two independent measurements must agree before a strong claim is made.
 */

/* ------------------------------------------------------------------ */
/* Basic stats                                                         */
/* ------------------------------------------------------------------ */
export function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[m - 1]! + s[m]!) / 2 : s[m]!;
}

export function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.floor((p / 100) * s.length)));
  return s[idx]!;
}

export function mean(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

export function stdev(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a, v) => a + (v - m) * (v - m), 0) / arr.length);
}

export function coefVar(arr: number[]): number {
  const m = mean(arr);
  return m > 0 ? stdev(arr) / m : 0;
}

/* ------------------------------------------------------------------ */
/* 1. Paper-grid calibration (pixels per mm measured from the image)   */
/* ------------------------------------------------------------------ */
function boxcarMean(x: number[], win: number): number[] {
  const n = x.length;
  const half = Math.floor(win / 2);
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i]! + x[i]!;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - half);
    const b = Math.min(n - 1, i + half);
    out[i] = (pre[b + 1]! - pre[a]!) / (b - a + 1);
  }
  return out;
}

function autocorr(x: number[], maxLag: number): number[] {
  const n = x.length;
  const m = mean(x);
  const xc = x.map((v) => v - m);
  let denom = 0;
  for (const v of xc) denom += v * v;
  const ac = new Array<number>(maxLag + 1).fill(0);
  if (denom <= 1e-12) return ac;
  for (let lag = 0; lag <= maxLag; lag++) {
    let s = 0;
    for (let i = 0; i + lag < n; i++) s += xc[i]! * xc[i + lag]!;
    ac[lag] = (s / denom) * (n / (n - lag));
  }
  return ac;
}

export interface GridPeriod {
  /** period of the finest visible grid line, in pixels */
  period: number;
  /** autocorrelation strength at that period (0..1) */
  strength: number;
  /** true when a second, 5x-longer periodicity (large boxes) was also found */
  hasFiveMultiple: boolean;
}

/** Find the dominant periodicity in a 1-D grid profile (column or row means). */
export function findGridPeriod(profile: number[], minLag = 3, maxLagCap = 320): GridPeriod | null {
  const n = profile.length;
  const maxLag = Math.min(maxLagCap, Math.floor(n / 3));
  if (maxLag <= minLag + 2) return null;
  const win = Math.min(n - 1, maxLag * 2) | 1;
  const ma = boxcarMean(profile, win);
  const hp = profile.map((v, i) => v - ma[i]!);
  const ac = autocorr(hp, maxLag);

  const peaks: { lag: number; v: number }[] = [];
  for (let l = minLag; l < maxLag; l++) {
    if (ac[l]! > ac[l - 1]! && ac[l]! >= ac[l + 1]! && ac[l]! > 0.05) {
      const y0 = ac[l - 1]!;
      const y1 = ac[l]!;
      const y2 = ac[l + 1]!;
      const denom = y0 - 2 * y1 + y2;
      const delta = Math.abs(denom) > 1e-9 ? (0.5 * (y0 - y2)) / denom : 0;
      peaks.push({ lag: l + delta, v: y1 });
    }
  }
  if (peaks.length === 0) return null;
  const maxV = Math.max(...peaks.map((p) => p.v));
  const first = peaks.find((p) => p.v >= Math.max(0.12, 0.25 * maxV)) ?? peaks[0]!;

  // Two-level grid test: with small AND large boxes, the autocorrelation at 5x the small period
  // is clearly higher than at 1x..4x (large lines coincide). A single-level grid is flat across multiples.
  const acAt = (lag: number): number => {
    const c = Math.round(lag);
    let m = -1;
    for (let k = c - 2; k <= c + 2; k++) if (k >= 0 && k <= maxLag) m = Math.max(m, ac[k]!);
    return m;
  };
  let five = false;
  if (Math.round(first.lag * 5) + 2 <= maxLag) {
    const smalls = [1, 2, 3, 4].map((k) => acAt(first.lag * k));
    five = acAt(first.lag * 5) >= 1.8 * mean(smalls) && mean(smalls) > 0.05;
  }
  const period = five ? first.lag : first.lag;
  return { period, strength: first.v, hasFiveMultiple: five };
}

export interface Calibration {
  pxPerMm: number | null;
  confidence: number; // 0..1
  basis: string;
}

/**
 * Turn a grid period into pixels-per-mm.
 * A strip that is 2.6–13 s wide at 25 mm/s is considered plausible; this disambiguates
 * "small box" vs "large box" periods when only one grid level is visible.
 */
export function resolveCalibration(profile: number[], imageWidthPx: number, paperSpeedMmS = 25): Calibration {
  const g = findGridPeriod(profile);
  if (!g || g.strength < 0.15) return { pxPerMm: null, confidence: 0, basis: "no_grid_detected" };
  const plausible = (pxPerMm: number) => {
    const sec = imageWidthPx / pxPerMm / paperSpeedMmS;
    return sec >= 2.6 && sec <= 13;
  };
  if (g.hasFiveMultiple) {
    const ok = plausible(g.period);
    return {
      pxPerMm: ok ? g.period : null,
      confidence: ok ? Math.min(1, 0.6 + g.strength) : 0.3,
      basis: ok ? "grid_small_and_large_boxes" : "grid_found_but_implausible_width",
    };
  }
  const a = g.period; // assume 1 mm
  const b = g.period / 5; // assume 5 mm
  const pa = plausible(a);
  const pb = plausible(b);
  if (pa && !pb) return { pxPerMm: a, confidence: 0.5 + Math.min(0.4, g.strength), basis: "grid_single_level_assumed_1mm" };
  if (pb && !pa) return { pxPerMm: b, confidence: 0.5 + Math.min(0.4, g.strength), basis: "grid_single_level_assumed_5mm" };
  return { pxPerMm: null, confidence: 0.2, basis: "grid_ambiguous" };
}

/** Pick the vertical scale that agrees with the horizontal one (images can be stretched). */
export function matchScaleToReference(rowProfile: number[], refPxPerMm: number): number | null {
  const g = findGridPeriod(rowProfile);
  if (!g || g.strength < 0.15) return null;
  for (const c of [g.period, g.period / 5, g.period * 5]) {
    if (Math.abs(c / refPxPerMm - 1) < 0.12) return c;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* 2. RR-interval pattern (scale-free)                                 */
/* ------------------------------------------------------------------ */
export type RrPattern = "regular" | "irregularly_irregular" | "pauses" | "mildly_irregular" | "indeterminate";

export interface RrAnalysis {
  pattern: RrPattern;
  cv: number;
  medianRr: number;
  fracNearMedian: number;
  /** indices i where rr[i] = peaks[i+1]-peaks[i] is a ~2x pause */
  pauseIdx: number[];
  n: number;
}

export const MIN_RR_INTERVALS = 4;

export function analyzeRr(rrs: number[]): RrAnalysis {
  const n = rrs.length;
  const med = median(rrs);
  const base = { cv: coefVar(rrs), medianRr: med, n, pauseIdx: [] as number[], fracNearMedian: 0 };
  if (n < MIN_RR_INTERVALS || med <= 0) return { ...base, pattern: "indeterminate" };

  const near = rrs.filter((r) => Math.abs(r / med - 1) <= 0.12).length / n;
  const pauseIdx: number[] = [];
  rrs.forEach((r, i) => {
    const ratio = r / med;
    if (ratio >= 1.6 && ratio <= 2.5) pauseIdx.push(i);
  });
  const nonPause = rrs.filter((_, i) => !pauseIdx.includes(i));
  const out = { ...base, fracNearMedian: near, pauseIdx };

  if (pauseIdx.length >= 1 && pauseIdx.length <= Math.floor(n / 2) && nonPause.length >= 3 && coefVar(nonPause) < 0.08) {
    return { ...out, pattern: "pauses" };
  }
  if (base.cv < 0.08) return { ...out, pattern: "regular", pauseIdx: [] };
  if (base.cv >= 0.15 && near < 0.5) return { ...out, pattern: "irregularly_irregular", pauseIdx: [] };
  return { ...out, pattern: "mildly_irregular", pauseIdx: [] };
}

/* ------------------------------------------------------------------ */
/* 3. Per-beat P-wave / PR measurement and conduction analysis         */
/* ------------------------------------------------------------------ */
export interface BeatP {
  evaluable: boolean;
  found: boolean;
  prMs: number | null;
  amp: number;
}

/** Approx. offset between (P-peak → R-peak) distance and the clinical PR interval. */
const PR_PEAK_TO_PEAK_OFFSET_MS = 20;

export function measureBeatsP(sig: number[], peaks: number[], fs: number, minAmpMm = 0.3): BeatP[] {
  const out: BeatP[] = [];
  for (let k = 0; k < peaks.length; k++) {
    const R = peaks[k]!;
    const hi = R - Math.floor(0.07 * fs);
    let lo = R - Math.floor(0.34 * fs);
    if (k > 0) lo = Math.max(lo, peaks[k - 1]! + Math.floor(0.3 * fs)); // skip previous T wave
    lo = Math.max(lo, 1);
    const hiClamped = Math.min(hi, sig.length - 2);
    if (hiClamped - lo < Math.floor(0.08 * fs)) {
      out.push({ evaluable: false, found: false, prMs: null, amp: 0 });
      continue;
    }
    const seg: number[] = [];
    for (let i = lo; i <= hiClamped; i++) seg.push((sig[i - 1]! + sig[i]! + sig[i + 1]!) / 3);
    const base = median(seg);
    let best = 0;
    let bi = -1;
    for (let j = 0; j < seg.length; j++) {
      const d = Math.abs(seg[j]! - base);
      if (d > best) {
        best = d;
        bi = j;
      }
    }
    const edge = Math.floor(0.025 * fs);
    const interior = bi >= edge && bi <= seg.length - 1 - edge;
    const found = best >= minAmpMm && interior;
    out.push({
      evaluable: true,
      found,
      prMs: found ? ((R - (lo + bi)) / fs) * 1000 + PR_PEAK_TO_PEAK_OFFSET_MS : null,
      amp: best,
    });
  }
  return out;
}

export interface ConductionAnalysis {
  nEvaluable: number;
  pFraction: number | null;
  prMedianMs: number | null;
  prCv: number | null;
  /** true/false only when the evidence is clear, otherwise null (= not established) */
  pWavesPresent: boolean | null;
  mobitzType: "mobitz_1" | "mobitz_2" | null;
}

export function analyzeConduction(beats: BeatP[], rr: RrAnalysis, hrBpm: number | null): ConductionAnalysis {
  const ev = beats.filter((b) => b.evaluable);
  const empty: ConductionAnalysis = {
    nEvaluable: ev.length,
    pFraction: null,
    prMedianMs: null,
    prCv: null,
    pWavesPresent: null,
    mobitzType: null,
  };
  if (ev.length < 4) return empty;

  const found = ev.filter((b) => b.found);
  const pFraction = found.length / ev.length;
  const prs = found.map((b) => b.prMs!);
  const prCv = prs.length >= 3 ? coefVar(prs) : null;
  const consistent = prCv != null && prCv <= 0.2;

  let pWavesPresent: boolean | null = null;
  if (pFraction >= 0.7 && consistent) pWavesPresent = true;
  else if (pFraction <= 0.25) pWavesPresent = false;

  const prMedianMs = consistent && pFraction >= 0.7 ? Math.round(median(prs)) : null;

  // Mobitz I vs II from the PR trend in the beats that precede each pause.
  let mobitzType: ConductionAnalysis["mobitzType"] = null;
  if (rr.pattern === "pauses" && pFraction >= 0.7) {
    const pr = (idx: number): number | null => (idx >= 0 && idx < beats.length && beats[idx]!.found ? beats[idx]!.prMs! : null);
    let t1 = 0;
    let t2 = 0;
    for (const i of rr.pauseIdx) {
      const a = pr(i - 2);
      const b = pr(i - 1);
      const c = pr(i);
      const after = pr(i + 1);
      if (b == null || c == null) continue;
      const progressive = (a != null ? a < b && b < c && c - a >= 20 : c - b >= 20);
      const vals = [a, b, c, after].filter((x): x is number => x != null);
      const constant = Math.max(...vals) - Math.min(...vals) <= 20;
      if (progressive) t1++;
      else if (constant) t2++;
    }
    const total = t1 + t2;
    if (total > 0) {
      if (t1 > total / 2) mobitzType = "mobitz_1";
      else if (t2 > total / 2) mobitzType = "mobitz_2";
    }
  }

  return { nEvaluable: ev.length, pFraction, prMedianMs, prCv, pWavesPresent, mobitzType };
}

/* ------------------------------------------------------------------ */
/* 4. Physiological plausibility gate                                  */
/* ------------------------------------------------------------------ */
export interface Plausible {
  hr: number | null;
  qrsMs: number | null;
  prMs: number | null;
  flags: string[];
}

export function applyPlausibility(hr: number | null, qrsMs: number | null, prMs: number | null): Plausible {
  const flags: string[] = [];
  let h = hr;
  let q = qrsMs;
  let p = prMs;
  if (h != null && (h < 25 || h > 250)) {
    flags.push(`hr_implausible:${Math.round(h)}`);
    h = null;
  }
  const rrSec = h != null ? 60 / h : null;
  if (q != null && (q < 50 || q > 220 || (rrSec != null && q / 1000 > 0.5 * rrSec))) {
    flags.push(`qrs_implausible:${Math.round(q)}`);
    q = null;
  }
  if (p != null && (p < 80 || p > 400 || (rrSec != null && p / 1000 > 0.6 * rrSec))) {
    flags.push(`pr_implausible:${Math.round(p)}`);
    p = null;
  }
  return { hr: h, qrsMs: q, prMs: p, flags };
}

/* ------------------------------------------------------------------ */
/* 5. Atrial-activity band (flutter-like periodicity)                  */
/* ------------------------------------------------------------------ */
export function dftMagnitude(sig: number[], fs: number, freqHz: number): number {
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

/** Remove QRS complexes (linear bridge) and the mean, leaving atrial/baseline activity. */
export function maskQrs(sig: number[], peaks: number[], fs: number): number[] {
  const bl = [...sig];
  const rad = Math.floor(0.08 * fs);
  for (const p of peaks) {
    const l = Math.max(0, p - rad);
    const r = Math.min(bl.length - 1, p + rad);
    const vL = bl[l]!;
    const vR = bl[r]!;
    const span = Math.max(1, r - l);
    for (let k = l; k <= r; k++) bl[k] = vL + ((vR - vL) * (k - l)) / span;
  }
  const m = mean(bl);
  return bl.map((v) => v - m);
}

export interface AtrialBand {
  snr: number;
  freqHz: number;
  rateBpm: number;
  baselineStdMm: number;
}

export function analyzeAtrialBand(bl: number[], fs: number): AtrialBand {
  let peakMag = 0;
  let peakFreq = 4.5;
  let sum = 0;
  let bins = 0;
  for (let f = 0.8; f <= 10.0; f += 0.15) {
    const mag = dftMagnitude(bl, fs, f);
    sum += mag;
    bins++;
    if (f >= 3.8 && f <= 5.8 && mag > peakMag) {
      peakMag = mag;
      peakFreq = f;
    }
  }
  return {
    snr: peakMag / (sum / Math.max(1, bins) + 1e-6),
    freqHz: peakFreq,
    rateBpm: Math.round(peakFreq * 60),
    baselineStdMm: stdev(bl),
  };
}

/**
 * Conservative flutter decision. Irregularly-irregular rhythm always defers to AF,
 * and for a regular rhythm the atrial rate must be an integer multiple of the ventricular rate.
 */
export function decideFlutter(a: AtrialBand, rr: RrAnalysis, hrBpm: number | null): boolean {
  if (rr.pattern === "irregularly_irregular" || rr.pattern === "indeterminate") return false;
  const needSnr = rr.pattern === "regular" ? 4.0 : 5.0;
  if (a.snr < needSnr || a.baselineStdMm < 0.35) return false;
  if (a.rateBpm < 228 || a.rateBpm > 348) return false;
  if (rr.pattern === "regular") {
    if (hrBpm == null) return false;
    const ratio = a.rateBpm / hrBpm;
    const k = Math.round(ratio);
    if (k < 2 || k > 6 || Math.abs(ratio - k) > 0.1 * k) return false;
  }
  return true;
}
