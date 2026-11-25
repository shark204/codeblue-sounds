/**
 * Uncertainty-Aware ECG Measurement Evaluation (Provenance + Error Propagation + 3-State Thresholding)
 *
 * Implements:
 *  1. Per-measurement provenance: { value, source: "manual" | "device_ocr" | "pixel_estimate", error_ms }
 *     - manual: error = 0
 *     - device_ocr: error = 0, explicitly labeled "device-reported" (devices can err in AF/AV blocks)
 *     - pixel_estimate: error = empirical P95 absolute error half-width (measured on PTB-XL via NeuroKit2)
 *  2. Exact first-order root-sum-square error propagation for QTc:
 *     - Bazett:     dQTc / QTc = sqrt((dQT / QT)^2 + (0.5 * dRR / RR)^2)
 *     - Fridericia: dQTc / QTc = sqrt((dQT / QT)^2 + ((1/3) * dRR / RR)^2)
 *  3. Strict 3-state threshold evaluation WITHOUT shifting clinical thresholds:
 *     - ">" rule: definite if (val - err) > thr; possible if |val - thr| <= err; else none
 *     - "<" rule: definite if (val + err) < thr; possible if |val - thr| <= err; else none
 *  4. Dangerous "possible" findings (QTc > 500, wide-QRS tachycardia, high-grade AVB) are NEVER suppressed
 *     and surface as "possible (unverified urgent)" without triggering a false red CRITICAL banner.
 */

export type MeasurementSource = "manual" | "device_ocr" | "pixel_estimate";

export interface ProvenancedMeasurement {
  value: number | null;
  source: MeasurementSource;
  /** 95% absolute error half-width (P95); 0 for manual and device_ocr */
  error_ms: number;
}

export interface MeasurementProvenanceMap {
  hr_bpm: ProvenancedMeasurement;
  pr_ms: ProvenancedMeasurement;
  qrs_ms: ProvenancedMeasurement;
  qt_ms: ProvenancedMeasurement;
  axis_deg: ProvenancedMeasurement;
}

/**
 * Measured P95 absolute error half-widths for the pixel waveform estimator
 * (PTB-XL Fold 10, N=140 synthetic NORM ECGs vs NeuroKit2 delineation):
 *  - hr_bpm: MAE 3.68 bpm,  P95 27.61 bpm,  Bias +3.22 bpm   (N=140)
 *  - pr_ms:  MAE 36.16 ms,  P95 79.90 ms,   Bias +35.40 ms   (N=134)
 *  - qrs_ms: MAE 80.06 ms,  P95 134.82 ms,  Bias -80.06 ms   (N=125)
 *  - qt_ms:  MAE 104.80 ms, P95 172.69 ms,  Bias -104.76 ms  (N=140)
 */
export const PIXEL_ESTIMATOR_P95 = {
  hr_bpm: 27.6, // bpm (measured P95)
  pr_ms: 79.9, // ms (measured P95)
  qrs_ms: 134.8, // ms (measured P95)
  qt_ms: 172.7, // ms (measured P95)
  axis_deg: 25, // deg (not pixel-estimated; OCR/manual only)
} as const;

export type ThresholdStatus = "definite" | "possible" | "none";

export function makeProvenanced(
  value: number | null,
  source: MeasurementSource,
  key: keyof typeof PIXEL_ESTIMATOR_P95,
  customPixelError?: number
): ProvenancedMeasurement {
  if (value == null || !Number.isFinite(value)) {
    return { value: null, source, error_ms: 0 };
  }
  const err = source === "pixel_estimate" ? (customPixelError ?? PIXEL_ESTIMATOR_P95[key]) : 0;
  return { value, source, error_ms: err };
}

/**
 * First-order root-sum-square QTc error propagation from QT and HR uncertainties.
 * Since RR = 60 / HR, |dRR / RR| = |dHR / HR|.
 */
export function propagateQtcErrorMs(
  qtMs: number,
  dQtMs: number,
  hrBpm: number,
  dHrBpm: number,
  formula: "bazett" | "fridericia" = "bazett"
): { qtcMs: number; errorMs: number } {
  const safeHr = Math.max(20, hrBpm);
  const rrSec = 60 / safeHr;
  const exp = formula === "fridericia" ? 1 / 3 : 0.5;
  const qtcMs = qtMs / Math.pow(rrSec, exp);

  if (dQtMs <= 0 && dHrBpm <= 0) {
    return { qtcMs: Number(qtcMs.toFixed(1)), errorMs: 0 };
  }
  const relQt = qtMs > 0 ? dQtMs / qtMs : 0;
  const relRr = dHrBpm / safeHr;
  const relQtc = Math.sqrt(relQt * relQt + Math.pow(exp * relRr, 2));
  const errorMs = qtcMs * relQtc;
  return {
    qtcMs: Number(qtcMs.toFixed(1)),
    errorMs: Number(errorMs.toFixed(1)),
  };
}

/**
 * Uncertainty-aware ">" threshold test (clinical threshold is NEVER shifted):
 *  - definite: (value - error) > threshold
 *  - possible: |value - threshold| <= error
 *  - none:     otherwise
 */
export function evalGreaterThan(value: number, error: number, threshold: number): ThresholdStatus {
  const e = Math.max(0, error);
  if (value - e > threshold) return "definite";
  if (Math.abs(value - threshold) <= e) return "possible";
  return "none";
}

/**
 * Uncertainty-aware "<" threshold test (clinical threshold is NEVER shifted):
 *  - definite: (value + error) < threshold
 *  - possible: |value - threshold| <= error
 *  - none:     otherwise
 */
export function evalLessThan(value: number, error: number, threshold: number): ThresholdStatus {
  const e = Math.max(0, error);
  if (value + e < threshold) return "definite";
  if (Math.abs(value - threshold) <= e) return "possible";
  return "none";
}

/**
 * Evaluates a numeric finding (e.g. Short QTc < 350, Long QTc > 460/500, PR > 200, QRS >= 120)
 * against its propagated uncertainty, returning downgraded urgency/labels for "possible" items
 * while ensuring dangerous "possible" conditions remain visible as "unverified urgent".
 */
export interface UncertaintyAdjustedRule {
  status: ThresholdStatus;
  isDangerous: boolean;
  /** True when a pixel_estimate finding is in the uncertainty band and needs manual verification */
  needsVerification: boolean;
  /** True when dangerous condition is "possible" — must appear as URGENT (not hidden, not red CRITICAL) */
  unverifiedUrgent: boolean;
  note_en: string | null;
  note_ar: string | null;
}

const DANGEROUS_RULE_CODES = new Set([
  "MARKEDLY_PROLONGED_QTC",
  "LONG_QT_CRITICAL",
  "WIDE_QRS_TACHYCARDIA",
  "VENTRICULAR_TACHYCARDIA",
  "HIGH_GRADE_AV_BLOCK",
  "HIGH_GRADE_OR_2_TO_1_AV_BLOCK",
  "COMPLETE_HEART_BLOCK",
]);

export function classifyMeasurementRule(
  code: string,
  status: ThresholdStatus,
  source: MeasurementSource,
  errorMs: number
): UncertaintyAdjustedRule {
  const isDangerous = DANGEROUS_RULE_CODES.has(code);
  if (status === "none") {
    return { status: "none", isDangerous, needsVerification: false, unverifiedUrgent: false, note_en: null, note_ar: null };
  }
  if (status === "possible") {
    if (isDangerous) {
      return {
        status: "possible",
        isDangerous: true,
        needsVerification: true,
        unverifiedUrgent: true,
        note_en: `Possible (Unverified Urgent — ±${Math.round(errorMs)} ms uncertainty): verify measurement manually before acting.`,
        note_ar: `احتمال طارئ غير مؤكد (هامش خطأ ±${Math.round(errorMs)} مللي ثانية): تحقق من القياس يدوياً فوراً.`,
      };
    }
    return {
      status: "possible",
      isDangerous: false,
      needsVerification: true,
      unverifiedUrgent: false,
      note_en: `Possible — verify measurement (±${Math.round(errorMs)} ms pixel-estimate uncertainty).`,
      note_ar: `محتمل — يرجى التحقق من القياس (هامش خطأ التقدير الآلي ±${Math.round(errorMs)} مللي ثانية).`,
    };
  }
  // Definite
  if (source === "device_ocr") {
    return {
      status: "definite",
      isDangerous,
      needsVerification: false,
      unverifiedUrgent: false,
      note_en: "Based on device-reported header measurement (verify in AFib or AV blocks).",
      note_ar: "مبني على قراءة الجهاز المطبوعة (يُراجع في حالات الرجفان الأذيني أو إحصار التوصيل).",
    };
  }
  return { status: "definite", isDangerous, needsVerification: false, unverifiedUrgent: false, note_en: null, note_ar: null };
}

export interface SourceConflictWarning {
  ruleCode: string;
  ruleLabelEn: string;
  ruleLabelAr: string;
  aiCode: string;
  aiLabelEn: string;
  aiLabelAr: string;
  messageEn: string;
  messageAr: string;
}

/**
 * Requirement 5: Do NOT use magic weights to silently outvote conflicting sources.
 * When a rule computed from pixel_estimate conflicts with a strong visual-AI prediction,
 * surface an explicit "Conflict — verify measurement" flag.
 */
export function detectPixelRuleVsAiConflict(
  pixelRuleFindings: { code: string; name_en: string; name_ar: string }[],
  strongAiPredictions: { code: string; label: string; prob: number }[]
): SourceConflictWarning | null {
  if (pixelRuleFindings.length === 0 || strongAiPredictions.length === 0) return null;
  const topAi = strongAiPredictions[0];
  if (!topAi || topAi.prob < 0.65) return null;

  for (const rf of pixelRuleFindings) {
    const aiIsNormal = topAi.code === "NORMAL_SINUS_RHYTHM";
    const ruleIsAbnormal = rf.code !== "NORMAL_SINUS_RHYTHM" && rf.code !== "NO_ABNORMALITY_IN_ASSESSED_PARAMETERS" && rf.code !== "INSUFFICIENT_DATA";
    const rateConflict =
      (rf.code.includes("BRADY") && (topAi.code.includes("SVT") || topAi.code.includes("TACHY") || topAi.code.includes("FLUTTER"))) ||
      (rf.code.includes("TACHY") && topAi.code.includes("BRADY"));

    if ((aiIsNormal && ruleIsAbnormal) || rateConflict) {
      return {
        ruleCode: rf.code,
        ruleLabelEn: rf.name_en,
        ruleLabelAr: rf.name_ar,
        aiCode: topAi.code,
        aiLabelEn: topAi.label,
        aiLabelAr: topAi.label,
        messageEn: `Conflict — verify measurement: Pixel-estimated rule flagged "${rf.name_en}" while Visual AI strongly predicts "${topAi.label}". Verify ECG intervals manually.`,
        messageAr: `تعارض — تحقق من القياس: التقدير الآلي أشار إلى "${rf.name_ar}" بينما الذكاء البصري يرجّح "${topAi.label}". يرجى مراجعة القياسات يدوياً.`,
      };
    }
  }
  return null;
}

export function evaluateRuleWithProvenance(
  code: string,
  prov: Partial<MeasurementProvenanceMap>,
  raw: { hr_bpm: number | null; pr_ms: number | null; qrs_ms: number | null; qt_ms: number | null }
): { uRes: UncertaintyAdjustedRule | null; usedPixelEstimate: boolean } {
  const pHr = prov.hr_bpm ?? makeProvenanced(raw.hr_bpm, "manual", "hr_bpm");
  const pPr = prov.pr_ms ?? makeProvenanced(raw.pr_ms, "manual", "pr_ms");
  const pQrs = prov.qrs_ms ?? makeProvenanced(raw.qrs_ms, "manual", "qrs_ms");
  const pQt = prov.qt_ms ?? makeProvenanced(raw.qt_ms, "manual", "qt_ms");

  if ((code === "SHORT_QT" || code === "PROLONGED_QTC" || code === "MARKEDLY_PROLONGED_QTC" || code === "LONG_QT_CRITICAL" || code === "LONG_QT") && pQt.value != null && pHr.value != null) {
    const useFrid = pHr.value > 100 || pHr.value < 50;
    const { qtcMs, errorMs } = propagateQtcErrorMs(pQt.value, pQt.error_ms, pHr.value, pHr.error_ms, useFrid ? "fridericia" : "bazett");
    const src: MeasurementSource = pQt.source === "pixel_estimate" || pHr.source === "pixel_estimate" ? "pixel_estimate" : pQt.source;
    const st =
      code === "SHORT_QT"
        ? evalLessThan(qtcMs, errorMs, 350)
        : code === "MARKEDLY_PROLONGED_QTC" || code === "LONG_QT_CRITICAL"
          ? evalGreaterThan(qtcMs, errorMs, 500)
          : evalGreaterThan(qtcMs, errorMs, 460);
    return { uRes: classifyMeasurementRule(code, st, src, errorMs), usedPixelEstimate: src === "pixel_estimate" };
  }
  if (code === "FIRST_DEGREE_AV_BLOCK" && pPr.value != null) {
    return { uRes: classifyMeasurementRule(code, evalGreaterThan(pPr.value, pPr.error_ms, 200), pPr.source, pPr.error_ms), usedPixelEstimate: pPr.source === "pixel_estimate" };
  }
  if ((code === "WIDE_QRS_TACHYCARDIA" || code === "VENTRICULAR_TACHYCARDIA") && pQrs.value != null) {
    return { uRes: classifyMeasurementRule(code, evalGreaterThan(pQrs.value, pQrs.error_ms, 120), pQrs.source, pQrs.error_ms), usedPixelEstimate: pQrs.source === "pixel_estimate" };
  }
  if (code === "MARKED_BRADYCARDIA" && pHr.value != null) {
    return { uRes: classifyMeasurementRule(code, evalLessThan(pHr.value, pHr.error_ms, 40), pHr.source, pHr.error_ms), usedPixelEstimate: pHr.source === "pixel_estimate" };
  }
  if ((code === "BRADYCARDIA" || code === "SINUS_BRADYCARDIA") && pHr.value != null) {
    return { uRes: classifyMeasurementRule(code, evalLessThan(pHr.value, pHr.error_ms, 60), pHr.source, pHr.error_ms), usedPixelEstimate: pHr.source === "pixel_estimate" };
  }
  if (code === "TACHYCARDIA_UNSPECIFIED" && pHr.value != null) {
    return { uRes: classifyMeasurementRule(code, evalGreaterThan(pHr.value, pHr.error_ms, 100), pHr.source, pHr.error_ms), usedPixelEstimate: pHr.source === "pixel_estimate" };
  }
  return { uRes: null, usedPixelEstimate: false };
}
