import rawCriteria from "@/ecg_engine/ecg_criteria.json";

export type UrgencyLevel = "CRITICAL" | "URGENT" | "ABNORMAL" | "NORMAL";
export type Confidence = "high" | "moderate" | "low";
export type RrPatternInput = "regular" | "irregularly_irregular" | "pauses" | "mildly_irregular" | "indeterminate";

/**
 * Describes how trustworthy the numbers are. Present only for image-derived input.
 * Manual / device-supplied measurements omit it (treated as clinician-verified).
 */
export interface ECGQuality {
  timing_ok: boolean; // absolute ms values (HR/PR/QRS) come from a calibrated grid
  rhythm_ok: boolean; // enough beats were found to classify the rhythm
  calibration: { px_per_mm: number | null; confidence: number; basis: string; strip_duration_sec: number | null };
  n_beats: number;
  n_bands: number;
  flags: string[];
}

/**
 * IMPORTANT CONVENTION: `undefined` / `null` = NOT MEASURED. It is never interpreted as "normal".
 * A rule only fires when the field it needs was actually supplied.
 */
export interface ECGInput {
  age?: number | null;
  sex?: "male" | "female" | "" | null;
  hr_bpm?: number | null;
  rr_ms?: number | null;
  rr_sec?: number | null;
  rr_small_boxes?: number | null;
  rr_large_boxes?: number | null;
  qrs_count_10s?: number | null;
  pr_ms?: number | null;
  qrs_ms?: number | null;
  qt_ms?: number | null;
  axis_deg?: number | null;
  rhythm_regular?: boolean;
  rr_pattern?: RrPatternInput;
  p_waves_present?: boolean;
  rhythm_type?: string;
  fibrillatory_waves?: boolean;
  sawtooth_flutter_waves?: boolean;
  flutter_wave_rate_bpm?: number | null;
  distinct_p_morphologies?: number | null;
  retrograde_p_waves?: boolean;
  rp_interval_ms?: number | null;
  delta_wave?: boolean;
  varying_qrs_morphology?: boolean;
  shortest_rr_ms?: number | null;
  polymorphic_vt?: boolean;
  av_dissociation?: boolean;
  capture_or_fusion_beats?: boolean;
  precordial_concordance?: string;
  rs_interval_ms?: number | null;
  avr_initial_r_wave?: boolean;
  avr_vi_vt_ratio?: number | null;
  av_block_type?: string;
  atrial_rate_bpm?: number | null;
  progressive_pr_prolongation?: boolean;
  qrs_morphology?: string;
  rsr_prime_v1?: boolean;
  wide_s_i_v6?: boolean;
  broad_monophasic_r_lateral?: boolean;
  deep_qs_v1?: boolean;
  lafb?: boolean;
  lpfb?: boolean;
  st_elevation_mm?: Record<string, number>;
  t_wave_inversion_leads?: string[];
  t_wave_amplitude_mm?: Record<string, number>;
  wellens_pattern?: string;
  tall_r_v1_v2?: boolean;
  upright_t_v1_v3?: boolean;
  concordant_ste_mm?: number | null;
  concordant_std_v1_v3_mm?: number | null;
  discordant_ste_mm?: number | null;
  modified_smith_st_s_ratio?: number | null;
  st_morphology_concave?: boolean;
  pr_depression?: boolean;
  v6_st_t_ratio?: number | null;
  brugada_pattern?: string;
  coved_ste_v1_v2?: boolean;
  saddleback_ste_v1_v2?: boolean;
  peaked_t_waves?: boolean;
  max_t_wave_amplitude_mm?: number | null;
  flattened_or_absent_p_hyperk?: boolean;
  sine_wave_pattern?: boolean;
  prominent_u_waves?: boolean;
  u_wave_amplitude_mm?: number | null;
  osborn_j_waves?: boolean;
  low_voltage?: boolean;
  max_limb_qrs_mm?: number | null;
  electrical_alternans?: boolean;
  pericardial_effusion_suspected?: boolean;
  s1q3t3?: boolean;
  rv_strain_twi_v1_v4?: boolean;
  avr_terminal_r_mm?: number | null;
  avr_r_to_s_ratio?: number | null;
  s_v1_mm?: number | null;
  r_v5_mm?: number | null;
  r_v6_mm?: number | null;
  r_avl_mm?: number | null;
  s_v3_mm?: number | null;
  r_v1_mm?: number | null;
  v1_r_to_s_ratio?: number | null;
  pacing_spikes?: boolean;
  negative_p_qrs_lead_i?: boolean;
  normal_precordial_r_progression?: boolean;
  quality?: ECGQuality;
}

export interface ECGFinding {
  code: string;
  urgency: UrgencyLevel;
  priority: number;
  urgency_label_en: string;
  urgency_label_ar: string;
  name_en: string;
  name_ar: string;
  evidence: string[];
  explanation_ar: string;
  explanation_en: string;
  immediate_action_ar: string;
  immediate_action_en: string;
  confidence: Confidence;
}

export interface ECGInterpretationResult {
  overall_urgency: UrgencyLevel;
  /** true only for CRITICAL/URGENT findings whose confidence is not "low" */
  has_urgent_findings: boolean;
  urgent_count: number;
  measurements: {
    hr_bpm: number | null;
    hr_method: string;
    rr_ms: number | null;
    pr_ms: number | null;
    qrs_ms: number | null;
    qt_ms: number | null;
    qtc_bazett_ms: number | null;
    qtc_fridericia_ms: number | null;
    qtc_hodges_ms: number | null;
    axis_deg: number | null;
    axis_classification: string | null;
  };
  urgent_findings: ECGFinding[];
  /** CRITICAL/URGENT candidates that could not be confirmed (poor calibration / few beats) */
  unverified_urgent_findings: ECGFinding[];
  all_findings: ECGFinding[];
  overall_confidence: Confidence;
  /** parameters that were NOT assessed, so their absence must not be read as "normal" */
  not_assessed: string[];
  quality?: ECGQuality;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const criteria: any = rawCriteria;

/** Findings produced by the generic (data-quality / rate / rhythm-pattern) layer. */
const EXTRA_KB: Record<
  string,
  {
    urgency: UrgencyLevel;
    name_en: string;
    name_ar: string;
    explanation_en: string;
    explanation_ar: string;
    immediate_action_en: string;
    immediate_action_ar: string;
  }
> = {
  INSUFFICIENT_DATA: {
    urgency: "ABNORMAL",
    name_en: "Insufficient data for a reliable interpretation",
    name_ar: "بيانات غير كافية لقراءة موثوقة",
    explanation_en: "Too few measurements could be obtained from this tracing to support any diagnosis. This is NOT a normal result.",
    explanation_ar: "لم يمكن استخراج قياسات كافية من هذا التخطيط لدعم أي تشخيص. هذه ليست نتيجة طبيعية.",
    immediate_action_en: "Use a clearer/full-length tracing, or enter the measurements manually.",
    immediate_action_ar: "استخدم تخطيطاً أوضح وكاملاً، أو أدخل القياسات يدوياً.",
  },
  NO_ABNORMALITY_IN_ASSESSED_PARAMETERS: {
    urgency: "NORMAL",
    name_en: "No abnormality in the parameters that could be assessed",
    name_ar: "لا توجد شذوذات في العناصر التي أمكن تقييمها",
    explanation_en: "Only the listed parameters were evaluated; everything else was not assessed. This does not exclude ST/T, axis, QT or morphology abnormalities.",
    explanation_ar: "تم تقييم العناصر المذكورة فقط، وباقي العناصر لم تُقيَّم. هذا لا يستبعد شذوذات ST/T أو المحور أو QT أو شكل QRS.",
    immediate_action_en: "Interpret clinically; review the full 12-lead tracing.",
    immediate_action_ar: "فسّر سريرياً وراجع التخطيط الكامل 12 lead.",
  },
  IRREGULAR_RHYTHM_UNSPECIFIED: {
    urgency: "ABNORMAL",
    name_en: "Irregularly irregular rhythm (mechanism not established)",
    name_ar: "إيقاع غير منتظم بشكل عشوائي (الآلية غير محددة)",
    explanation_en: "RR intervals vary without a repeating pattern. Atrial fibrillation is the usual cause, but P waves were not assessed with enough certainty to confirm it (also consider MAT, flutter with variable block, frequent ectopy).",
    explanation_ar: "فواصل RR تتغير دون نمط متكرر. الرجفان الأذيني هو السبب الأشيع، لكن لم يتم تقييم موجات P بدرجة تكفي للتأكيد (فكّر أيضاً في MAT أو flutter بتوصيل متغير أو ectopics متكررة).",
    immediate_action_en: "Confirm on a full 12-lead/rhythm strip: look for P waves and flutter waves.",
    immediate_action_ar: "أكّد على تخطيط 12 lead/شريط إيقاع كامل: ابحث عن موجات P وموجات flutter.",
  },
  PAUSES_POSSIBLE_AV_BLOCK: {
    urgency: "ABNORMAL",
    name_en: "Regular rhythm with intermittent pauses (possible dropped beats)",
    name_ar: "إيقاع منتظم مع توقفات متقطعة (احتمال سقوط نبضات)",
    explanation_en: "Pauses of about twice the basic RR were found. This pattern fits 2nd-degree AV block, blocked premature atrial beats or sinus pauses, but the PR trend before the pauses was not conclusive.",
    explanation_ar: "تم رصد توقفات بحوالي ضعف الـ RR الأساسي. النمط يتماشى مع إحصار أذيني بطيني من الدرجة الثانية أو نبضات أذينية مبكرة محجوبة أو توقف جيبي، لكن اتجاه PR قبل التوقفات لم يكن حاسماً.",
    immediate_action_en: "Check the PR behaviour before/after each pause (Wenckebach vs constant PR).",
    immediate_action_ar: "افحص سلوك PR قبل وبعد كل توقف (Wenckebach مقابل PR ثابت).",
  },
  BRADYCARDIA: {
    urgency: "ABNORMAL",
    name_en: "Bradycardia (rhythm origin not determined)",
    name_ar: "بطء القلب (مصدر الإيقاع غير محدد)",
    explanation_en: "Ventricular rate below the normal range. Sinus origin, AV block or escape rhythm cannot be distinguished from the measurements available.",
    explanation_ar: "المعدل البطيني أقل من الطبيعي. لا يمكن التفريق بين منشأ جيبي أو إحصار أو إيقاع هروب من القياسات المتاحة.",
    immediate_action_en: "Assess P-QRS relationship, symptoms and medications.",
    immediate_action_ar: "قيّم علاقة P بـ QRS والأعراض والأدوية.",
  },
  MARKED_BRADYCARDIA: {
    urgency: "URGENT",
    name_en: "Marked bradycardia (rhythm origin not determined)",
    name_ar: "بطء قلب شديد (مصدر الإيقاع غير محدد)",
    explanation_en: "Ventricular rate below 40 bpm. High-grade/complete AV block, 2:1 block, escape rhythm and severe sinus bradycardia are all possible and cannot be separated from the measurements available.",
    explanation_ar: "المعدل البطيني أقل من 40. احتمالات: إحصار عالي الدرجة/كامل، إحصار 2:1، إيقاع هروب، أو بطء جيبي شديد، ولا يمكن التفريق بينها من القياسات المتاحة.",
    immediate_action_en: "Check haemodynamics now; look for AV dissociation / non-conducted P waves.",
    immediate_action_ar: "افحص الحالة الدورانية فوراً وابحث عن انفصال AV أو موجات P غير موصَّلة.",
  },
  TACHYCARDIA_UNSPECIFIED: {
    urgency: "ABNORMAL",
    name_en: "Tachycardia (mechanism not established)",
    name_ar: "تسارع القلب (الآلية غير محددة)",
    explanation_en: "Rate above the normal range without enough evidence to name the rhythm (sinus tachycardia, SVT, flutter with 2:1 block and others remain possible).",
    explanation_ar: "المعدل أعلى من الطبيعي دون دليل كافٍ لتسمية الإيقاع (تسارع جيبي، SVT، flutter بتوصيل 2:1 وغيرها محتملة).",
    immediate_action_en: "Identify P waves/atrial activity on a full tracing; treat according to stability.",
    immediate_action_ar: "حدد موجات P/النشاط الأذيني على تخطيط كامل، والعلاج حسب استقرار الحالة.",
  },
};

function round1(val: number): number {
  return Math.round(val * 10) / 10;
}

function makeFinding(code: string, evidence: string[]): ECGFinding {
  const kbEntry = criteria.conditions_knowledge_base?.[code] || EXTRA_KB[code] || {};
  const urgency: UrgencyLevel = kbEntry.urgency || "ABNORMAL";
  const urgMeta = criteria.urgency_levels[urgency] || criteria.urgency_levels.ABNORMAL;
  return {
    code,
    urgency,
    priority: urgMeta.priority,
    urgency_label_en: urgMeta.label_en,
    urgency_label_ar: urgMeta.label_ar,
    name_en: kbEntry.name_en || code,
    name_ar: kbEntry.name_ar || code,
    evidence,
    explanation_ar: kbEntry.explanation_ar || "",
    explanation_en: kbEntry.explanation_en || "",
    immediate_action_ar: kbEntry.immediate_action_ar || "",
    immediate_action_en: kbEntry.immediate_action_en || "",
    confidence: "moderate", // graded after all findings are known
  };
}

function computeHeartRate(ecg: ECGInput): { hr: number | null; rrMs: number | null; method: string } {
  const units = criteria.units_and_calibration;
  if (ecg.hr_bpm != null && ecg.hr_bpm > 0) {
    return { hr: round1(ecg.hr_bpm), rrMs: round1(60000 / ecg.hr_bpm), method: "direct_hr_bpm" };
  }
  if (ecg.rr_ms != null && ecg.rr_ms > 0) {
    return { hr: round1(60000 / ecg.rr_ms), rrMs: round1(ecg.rr_ms), method: "60000 / rr_ms" };
  }
  if (ecg.rr_sec != null && ecg.rr_sec > 0) {
    return { hr: round1(60 / ecg.rr_sec), rrMs: round1(ecg.rr_sec * 1000), method: "60 / rr_sec" };
  }
  if (ecg.rr_small_boxes != null && ecg.rr_small_boxes > 0) {
    const rrMs = ecg.rr_small_boxes * units.small_box_ms;
    return { hr: round1(60000 / rrMs), rrMs: round1(rrMs), method: "1500 / rr_small_boxes" };
  }
  if (ecg.rr_large_boxes != null && ecg.rr_large_boxes > 0) {
    const rrMs = ecg.rr_large_boxes * units.large_box_ms;
    return { hr: round1(60000 / rrMs), rrMs: round1(rrMs), method: "300 / rr_large_boxes" };
  }
  if (ecg.qrs_count_10s != null && ecg.qrs_count_10s > 0) {
    const hr = ecg.qrs_count_10s * (60 / units.standard_strip_duration_sec);
    return { hr: round1(hr), rrMs: round1(60000 / hr), method: "qrs_count_10s * 6" };
  }
  return { hr: null, rrMs: null, method: "unavailable" };
}

function computeQTc(qtMs: number | null, rrMs: number | null, hrBpm: number | null) {
  if (!qtMs || qtMs <= 0) {
    return { bazett: null, fridericia: null, hodges: null, primary: null, formula: "bazett" };
  }
  const effRr = rrMs && rrMs > 0 ? rrMs : hrBpm && hrBpm > 0 ? 60000 / hrBpm : null;
  if (!effRr || effRr <= 0) {
    return { bazett: null, fridericia: null, hodges: null, primary: null, formula: "bazett" };
  }
  const rrSec = effRr / 1000;
  const qtSec = qtMs / 1000;
  const effHr = hrBpm || 60 / rrSec;
  const bazett = round1((qtSec / Math.sqrt(rrSec)) * 1000);
  const fridericia = round1((qtSec / Math.cbrt(rrSec)) * 1000);
  const hodges = round1(qtMs + 1.75 * (effHr - 60));
  const formula: string = criteria.intervals_ms.qt_qtc.default_formula || "bazett";
  const map: Record<string, number> = { bazett, fridericia, hodges };
  return { bazett, fridericia, hodges, primary: map[formula] ?? bazett, formula };
}

function classifyAxis(axisDeg: number | null): string | null {
  if (axisDeg == null) return null;
  const cfg = criteria.axis_degrees;
  if (axisDeg >= cfg.normal_min && axisDeg <= cfg.normal_max) return "Normal Axis";
  if (axisDeg > cfg.normal_max && axisDeg <= cfg.normal_extended_max) return "Borderline Right Axis";
  if (axisDeg >= cfg.extreme_nw_min && axisDeg <= cfg.extreme_nw_max) return "Extreme Axis (Northwest)";
  if (axisDeg < cfg.normal_min) return "Left Axis Deviation (LAD)";
  if (axisDeg > cfg.normal_extended_max) return "Right Axis Deviation (RAD)";
  return "Indeterminate Axis";
}

function stThreshold(lead: string, age?: number | null, sex?: string | null): number {
  const ste = criteria.st_and_ischemia_criteria.stemi_j_point_elevation_mm;
  if (["V7", "V8", "V9"].includes(lead)) return ste.posterior_v7_v9;
  if (["V3R", "V4R", "V5R", "V6R"].includes(lead)) return ste.rv_v4r;
  if (lead === "V2" || lead === "V3") {
    const s = (sex || "").toLowerCase();
    if (s === "female" || s === "f") return ste.v2_v3_female_any_age;
    if (s === "male" || s === "m") {
      if (age != null && age < ste.age_cutoff_male_v2_v3) return ste.v2_v3_male_under_40;
      return ste.v2_v3_male_40_and_over;
    }
    return ste.v2_v3_unknown_demographics;
  }
  return ste.default_all_other_leads;
}

const TACHY_CODES = [
  "VENTRICULAR_TACHYCARDIA",
  "SVT_NARROW",
  "ATRIAL_FLUTTER",
  "ATRIAL_FIBRILLATION_RVR",
  "MAT",
  "PREEXCITED_AF",
  "TORSADES_DE_POINTES",
  "VF_OR_ASYSTOLE",
];

export function interpretECG(ecg: ECGInput): ECGInterpretationResult {
  const { hr, rrMs, method } = computeHeartRate(ecg);
  const prMs = ecg.pr_ms ?? null;
  const qrsMs = ecg.qrs_ms ?? null;
  const qtMs = ecg.qt_ms ?? null;
  const axisDeg = ecg.axis_deg ?? null;
  const qtc = computeQTc(qtMs, rrMs, hr);
  const axisLabel = classifyAxis(axisDeg);
  const isPaced = Boolean(ecg.pacing_spikes);
  const findings: ECGFinding[] = [];

  const intCfg = criteria.intervals_ms;
  const stCfg = criteria.st_and_ischemia_criteria;
  const arrCfg = criteria.arrhythmia_and_conduction_criteria;
  const hypCfg = criteria.hypertrophy_and_voltage_criteria;
  const axisCfg = criteria.axis_degrees;
  const hrCfg = criteria.heart_rate;

  // ---- AV conduction ------------------------------------------------------
  let hasFirstDeg = false;
  const avb = (ecg.av_block_type || "").toLowerCase();
  if (["3rd", "third", "complete", "chb"].includes(avb) || (ecg.av_dissociation && hr != null && hr < 60)) {
    const ev = ["Complete AV dissociation: independent P waves & slower escape rhythm"];
    if (ecg.atrial_rate_bpm) ev.push(`Atrial rate ${ecg.atrial_rate_bpm} bpm > Ventricular rate ${hr} bpm`);
    findings.push(makeFinding("COMPLETE_HEART_BLOCK", ev));
  } else if (["mobitz_2", "mobitz_ii", "type_2"].includes(avb)) {
    findings.push(makeFinding("MOBITZ_II_AV_BLOCK", ["Constant PR intervals with abrupt dropped QRS (infranodal)"]));
  } else if (["2:1", "2_to_1", "high_grade"].includes(avb)) {
    findings.push(makeFinding("HIGH_GRADE_OR_2_TO_1_AV_BLOCK", [`AV block pattern: ${avb}`]));
  } else if (["mobitz_1", "mobitz_i", "wenckebach", "type_1"].includes(avb) || ecg.progressive_pr_prolongation) {
    findings.push(makeFinding("MOBITZ_I_AV_BLOCK", ["Progressive PR lengthening followed by dropped QRS"]));
  } else if (prMs != null && prMs >= intCfg.pr.first_degree_avb_min_ms) {
    hasFirstDeg = true;
    findings.push(makeFinding("FIRST_DEGREE_AV_BLOCK", [`PR = ${prMs} ms (> ${intCfg.pr.normal_max_ms} ms)`]));
  }
  const hasSpecificAvBlock = findings.some((f) =>
    ["COMPLETE_HEART_BLOCK", "MOBITZ_II_AV_BLOCK", "HIGH_GRADE_OR_2_TO_1_AV_BLOCK", "MOBITZ_I_AV_BLOCK"].includes(f.code)
  );

  // ---- Bundle branch / fascicular ----------------------------------------
  const wideQrs = qrsMs != null && qrsMs >= intCfg.qrs.wide_qrs_min_ms;
  const morph = (ecg.qrs_morphology || "").toLowerCase();
  let hasRbbb = false;
  let hasLbbb = false;
  if (wideQrs && (morph === "rbbb" || (ecg.rsr_prime_v1 && ecg.wide_s_i_v6))) {
    hasRbbb = true;
    findings.push(makeFinding("RBBB", [`QRS = ${qrsMs} ms`, "rSR' in V1-V2 & wide S in I/V6"]));
  } else if (wideQrs && (morph === "lbbb" || (ecg.broad_monophasic_r_lateral && ecg.deep_qs_v1))) {
    hasLbbb = true;
    findings.push(makeFinding("LBBB", [`QRS = ${qrsMs} ms`, "Broad notched R in I/aVL/V5-V6 & deep QS in V1"]));
  }
  const hasLafb = Boolean(ecg.lafb) || (axisDeg != null && axisDeg >= axisCfg.lafb_min && axisDeg <= axisCfg.lafb_max);
  const hasLpfb = Boolean(ecg.lpfb);
  if (hasRbbb && (hasLafb || hasLpfb)) {
    const fName = hasLafb ? "LAFB" : "LPFB";
    findings.push(
      makeFinding(hasFirstDeg ? "TRIFASCICULAR_BLOCK" : "BIFASCICULAR_BLOCK", [`RBBB + ${fName}${hasFirstDeg ? " + 1st-degree AVB" : ""}`])
    );
  }

  // ---- Rhythm -------------------------------------------------------------
  const rhythmHint = (ecg.rhythm_type || "").toLowerCase();
  const regular: boolean | undefined = ecg.rhythm_regular; // undefined = NOT assessed (never "regular by default")
  const pPresent: boolean | undefined = ecg.p_waves_present; // undefined = NOT assessed
  const isRegular = regular === true;
  const isIrregular = regular === false;

  if (["vf", "ventricular_fibrillation", "asystole"].includes(rhythmHint)) {
    findings.push(makeFinding("VF_OR_ASYSTOLE", [`Arrest rhythm: ${rhythmHint.toUpperCase()}`]));
  } else if (
    ["torsades", "torsades_de_pointes", "tdp"].includes(rhythmHint) ||
    (ecg.polymorphic_vt && qtc.primary != null && qtc.primary >= intCfg.qt_qtc.normal_upper_female_ms)
  ) {
    findings.push(makeFinding("TORSADES_DE_POINTES", ["Polymorphic VT twisting around baseline", `QTc = ${qtc.primary ?? "prolonged"} ms`]));
  } else if (
    rhythmHint === "preexcited_af" ||
    (isIrregular &&
      wideQrs &&
      (ecg.delta_wave || ecg.varying_qrs_morphology || (ecg.shortest_rr_ms != null && ecg.shortest_rr_ms <= arrCfg.af.preexcited_af_shortest_rr_ms)))
  ) {
    const ev = ["Irregular wide-complex tachycardia with pre-excitation"];
    if (ecg.shortest_rr_ms) ev.push(`Shortest RR = ${ecg.shortest_rr_ms} ms`);
    findings.push(makeFinding("PREEXCITED_AF", ev));
  } else {
    if ((hr != null && hr >= arrCfg.wct_vt_criteria.min_rate_bpm && wideQrs && isRegular) || ["vt", "ventricular_tachycardia", "monomorphic_vt"].includes(rhythmHint)) {
      const vtEv = [`Regular WCT: HR = ${hr} bpm, QRS = ${qrsMs} ms`];
      if (ecg.av_dissociation) vtEv.push("AV dissociation present");
      if (ecg.capture_or_fusion_beats) vtEv.push("Capture/fusion beats present");
      if (ecg.precordial_concordance) vtEv.push(`Precordial concordance (${ecg.precordial_concordance})`);
      if (ecg.rs_interval_ms && ecg.rs_interval_ms > arrCfg.wct_vt_criteria.brugada_rs_interval_ms) {
        vtEv.push(`RS interval = ${ecg.rs_interval_ms} ms (>100 ms)`);
      }
      if (ecg.avr_initial_r_wave) vtEv.push("Vereckei aVR initial R wave");
      if (axisDeg != null && axisDeg >= axisCfg.extreme_nw_min && axisDeg <= axisCfg.extreme_nw_max) {
        vtEv.push(`Extreme NW axis (${axisDeg}°)`);
      }
      findings.push(makeFinding("VENTRICULAR_TACHYCARDIA", vtEv));
    }

    const flutterRateOk =
      ecg.flutter_wave_rate_bpm != null &&
      ecg.flutter_wave_rate_bpm >= arrCfg.atrial_flutter.flutter_wave_rate_min_bpm &&
      ecg.flutter_wave_rate_bpm <= arrCfg.atrial_flutter.flutter_wave_rate_max_bpm;

    if (ecg.sawtooth_flutter_waves || rhythmHint === "atrial_flutter" || flutterRateOk) {
      findings.push(
        makeFinding("ATRIAL_FLUTTER", [
          "Periodic atrial (F-wave) activity",
          `Flutter rate = ${ecg.flutter_wave_rate_bpm ?? "n/a"} bpm, Ventricular = ${hr ?? "?"} bpm`,
        ])
      );
    } else if ((isIrregular && pPresent === false) || ecg.fibrillatory_waves || ["af", "afib", "atrial_fibrillation"].includes(rhythmHint)) {
      const ev = ["Irregularly irregular RR"];
      ev.push(pPresent === false ? "No consistent P waves before QRS" : "Fibrillatory baseline");
      ev.push(`Ventricular rate = ${hr ?? "?"} bpm`);
      findings.push(makeFinding(hr != null && hr > arrCfg.af.rvr_min_bpm ? "ATRIAL_FIBRILLATION_RVR" : "ATRIAL_FIBRILLATION", ev));
    } else if (
      rhythmHint === "mat" ||
      ((ecg.distinct_p_morphologies ?? 1) >= arrCfg.mat.min_distinct_p_morphologies && hr != null && hr > arrCfg.mat.min_rate_bpm)
    ) {
      findings.push(makeFinding("MAT", [`HR = ${hr} bpm`, `${ecg.distinct_p_morphologies ?? 3} distinct P morphologies`]));
    } else if (
      isRegular &&
      hr != null &&
      hr >= arrCfg.svt.min_rate_bpm &&
      qrsMs != null &&
      qrsMs <= arrCfg.svt.max_qrs_ms &&
      (pPresent === false || ecg.retrograde_p_waves || ["svt", "avnrt", "avrt", "psvt"].includes(rhythmHint))
    ) {
      const ev = [`Regular narrow-complex tachycardia: HR = ${hr} bpm, QRS = ${qrsMs} ms`];
      if (ecg.rp_interval_ms != null) {
        ev.push(`RP = ${ecg.rp_interval_ms} ms (${ecg.rp_interval_ms < arrCfg.svt.avnrt_short_rp_max_ms ? "AVNRT likely" : "AVRT/AT likely"})`);
      }
      findings.push(makeFinding("SVT_NARROW", ev));
    } else if (isIrregular && ecg.rr_pattern === "irregularly_irregular" && pPresent !== true) {
      findings.push(
        makeFinding("IRREGULAR_RHYTHM_UNSPECIFIED", [
          "RR intervals irregularly irregular (no repeating pattern)",
          pPresent === undefined ? "P-wave presence not established" : "No consistent P waves",
        ])
      );
    }

    if (ecg.rr_pattern === "pauses" && !hasSpecificAvBlock) {
      findings.push(makeFinding("PAUSES_POSSIBLE_AV_BLOCK", ["Pauses of ~2x the basic RR interval", "PR trend before the pauses inconclusive"]));
    }

    if (ecg.delta_wave || (prMs != null && prMs <= intCfg.pr.short_pr_max_ms && wideQrs)) {
      const ev: string[] = [];
      if (prMs != null) ev.push(`PR = ${prMs} ms`);
      if (qrsMs != null) ev.push(`QRS = ${qrsMs} ms`);
      if (ecg.delta_wave) ev.push("Delta wave present");
      findings.push(makeFinding("WPW_PATTERN", ev));
    }
  }

  // ---- Generic rate layer: any abnormal measured rate is reported ---------
  if (hr != null) {
    if (hr < 40) findings.push(makeFinding("MARKED_BRADYCARDIA", [`Ventricular rate = ${hr} bpm`]));
    else if (hr < (hrCfg?.normal_min_bpm ?? 60)) findings.push(makeFinding("BRADYCARDIA", [`Ventricular rate = ${hr} bpm`]));
    else if (hr > (hrCfg?.normal_max_bpm ?? 100) && !findings.some((f) => TACHY_CODES.includes(f.code))) {
      findings.push(makeFinding("TACHYCARDIA_UNSPECIFIED", [`Ventricular rate = ${hr} bpm`]));
    }
  }

  // ---- ST / ischemia (only when per-lead ST values were supplied) ---------
  const stMap = ecg.st_elevation_mm || {};
  const hasStData = Object.keys(stMap).length > 0;
  if (hasLbbb || isPaced) {
    const sg = stCfg.sgarbossa_lbbb_or_paced;
    let score = 0;
    const sgEv: string[] = [];
    if ((ecg.concordant_ste_mm ?? 0) >= sg.concordant_ste_mm) {
      score += sg.concordant_ste_points;
      sgEv.push(`Concordant STE ${ecg.concordant_ste_mm} mm (+${sg.concordant_ste_points} pts)`);
    }
    if (ecg.concordant_std_v1_v3_mm != null && ecg.concordant_std_v1_v3_mm <= sg.concordant_std_v1_v3_mm) {
      score += sg.concordant_std_v1_v3_points;
      sgEv.push(`Concordant STD V1-V3 ${ecg.concordant_std_v1_v3_mm} mm (+${sg.concordant_std_v1_v3_points} pts)`);
    }
    if ((ecg.discordant_ste_mm ?? 0) >= sg.discordant_ste_mm) {
      score += sg.discordant_ste_points;
      sgEv.push(`Discordant STE ${ecg.discordant_ste_mm} mm (+${sg.discordant_ste_points} pts)`);
    }
    const smithPos = ecg.modified_smith_st_s_ratio != null && Math.abs(ecg.modified_smith_st_s_ratio) >= sg.modified_smith_st_s_ratio;
    if (score >= sg.diagnostic_score_min || smithPos) {
      findings.push(makeFinding("SGARBOSSA_POSITIVE", sgEv));
    }
  } else if (hasStData) {
    const stdLeadsAll = criteria.lead_groups_and_territories.all_standard_leads as string[];
    const elevatedStd = stdLeadsAll.filter((ld) => ld !== "aVR" && (stMap[ld] ?? 0) >= stThreshold(ld, ecg.age, ecg.sex));
    const depressedNonAvrV1 = Object.entries(stMap).filter(([ld, v]) => ld !== "aVR" && ld !== "V1" && v <= stCfg.nstemi_ischemia_st_depression_mm);
    const peric = arrCfg.pericarditis;
    const isPericarditis =
      elevatedStd.length >= peric.min_diffuse_ste_leads &&
      depressedNonAvrV1.length === 0 &&
      (Boolean(ecg.st_morphology_concave) || Boolean(ecg.pr_depression) || (ecg.v6_st_t_ratio ?? 0) > peric.st_t_ratio_v6_min);

    if (isPericarditis) {
      findings.push(
        makeFinding("ACUTE_PERICARDITIS", [
          `Diffuse concave STE in ${elevatedStd.length} leads (${elevatedStd.join(", ")}) without reciprocal STD`,
          ...(ecg.pr_depression ? ["PR-segment depression present"] : []),
        ])
      );
    } else {
      const avrSte = stMap["aVR"] ?? 0;
      const depLeads = Object.entries(stMap)
        .filter(([ld, v]) => ld !== "aVR" && v <= stCfg.nstemi_ischemia_st_depression_mm)
        .map(([ld]) => ld);
      if (avrSte >= stCfg.lmca_avr_ste_min_mm && depLeads.length >= stCfg.lmca_diffuse_std_min_leads) {
        findings.push(makeFinding("LMCA_OCCLUSION", [`aVR STE +${avrSte} mm`, `Diffuse STD in ${depLeads.length} leads (${depLeads.join(", ")})`]));
      }

      const territories = criteria.lead_groups_and_territories.contiguous_territories;
      const stEv: string[] = [];
      for (const tName of ["septal", "anterior", "lateral_low_apical", "lateral_high", "inferior"]) {
        const t = territories[tName];
        const qual = (t.leads as string[]).filter((ld) => (stMap[ld] ?? 0) >= stThreshold(ld, ecg.age, ecg.sex));
        if (qual.length >= t.min_contiguous_leads) {
          const recip = (t.reciprocal_leads as string[]).filter((rl) => (stMap[rl] ?? 0) <= stCfg.nstemi_ischemia_st_depression_mm);
          let artery = t.artery;
          if (tName === "inferior") {
            artery = (stMap["III"] ?? 0) > (stMap["II"] ?? 0) ? "RCA (STE III > II)" : "LCx likely (STE II >= III)";
          }
          stEv.push(
            `[${tName.toUpperCase()}] ${qual.map((ld) => `${ld} (+${stMap[ld]}mm)`).join(", ")} | Artery: ${artery}${
              recip.length ? ` | Reciprocal STD: ${recip.join(", ")}` : ""
            }`
          );
        }
      }
      if (stEv.length > 0) findings.push(makeFinding("STEMI", stEv));

      const postLeads = ["V7", "V8", "V9"].filter((ld) => (stMap[ld] ?? 0) >= stThreshold(ld, ecg.age, ecg.sex));
      const v1v3Std = ["V1", "V2", "V3"].filter((ld) => stMap[ld] != null && stMap[ld]! <= stCfg.posterior_mi_v1_v3_st_depression_mm);
      if (postLeads.length >= 2 || (v1v3Std.length >= 2 && (ecg.tall_r_v1_v2 || ecg.upright_t_v1_v3))) {
        findings.push(
          makeFinding("POSTERIOR_STEMI", [
            ...(postLeads.length ? [`Posterior STE in ${postLeads.join(", ")}`] : []),
            ...(v1v3Std.length ? [`Mirror STD in ${v1v3Std.join(", ")}`] : []),
          ])
        );
      }

      const rvLeads = ["V3R", "V4R", "V5R", "V6R"].filter((ld) => (stMap[ld] ?? 0) >= stThreshold(ld, ecg.age, ecg.sex));
      if (rvLeads.length > 0) {
        findings.push(makeFinding("RV_STEMI", [`Right precordial STE in ${rvLeads.map((l) => `${l} (+${stMap[l]}mm)`).join(", ")}`]));
      }

      const wType = (ecg.wellens_pattern || "").toLowerCase();
      if (["type_a", "type_b", "biphasic", "deep_symmetric"].includes(wType)) {
        findings.push(makeFinding("WELLENS_SYNDROME", [`Wellens ${wType.toUpperCase()} T-wave pattern in V2-V3/V4`]));
      } else if (stEv.length === 0 && !findings.some((f) => ["LMCA_OCCLUSION", "POSTERIOR_STEMI"].includes(f.code))) {
        const tInv = ecg.t_wave_inversion_leads || [];
        if (depLeads.length >= 2 || tInv.length >= 2) {
          findings.push(
            makeFinding("NSTEMI_ISCHEMIA", [
              ...(depLeads.length >= 2 ? [`ST depression in ${depLeads.join(", ")}`] : []),
              ...(tInv.length >= 2 ? [`T-wave inversion in ${tInv.join(", ")}`] : []),
            ])
          );
        }
      }
    }
  }

  // ---- QT -----------------------------------------------------------------
  if (qtc.primary != null) {
    const qtCfg = intCfg.qt_qtc;
    const isFemale = (ecg.sex || "").toLowerCase() === "female";
    const upper = isFemale ? qtCfg.normal_upper_female_ms : qtCfg.normal_upper_male_ms;
    if (qtc.primary >= qtCfg.critical_torsades_risk_ms) {
      findings.push(makeFinding("LONG_QT_CRITICAL", [`QTc (${qtc.formula}) = ${qtc.primary} ms (>= ${qtCfg.critical_torsades_risk_ms} ms)`]));
    } else if (qtc.primary > upper) {
      findings.push(makeFinding("LONG_QT", [`QTc (${qtc.formula}) = ${qtc.primary} ms (upper normal ${upper} ms)`]));
    } else if (qtc.primary <= qtCfg.short_qt_max_ms) {
      findings.push(makeFinding("SHORT_QT", [`QTc (${qtc.formula}) = ${qtc.primary} ms (<= ${qtCfg.short_qt_max_ms} ms)`]));
    }
  }

  // ---- Morphology / metabolic / structural (fire only on supplied data) ----
  const v1v2Ste = Math.max(stMap["V1"] ?? 0, stMap["V2"] ?? 0);
  const brug = (ecg.brugada_pattern || "").toLowerCase();
  if (brug === "type_1" || (ecg.coved_ste_v1_v2 && v1v2Ste >= arrCfg.brugada.type1_coved_ste_min_mm)) {
    findings.push(makeFinding("BRUGADA_TYPE_1", [`Coved STE ${v1v2Ste} mm in V1-V2`]));
  } else if (brug === "type_2" || (ecg.saddleback_ste_v1_v2 && v1v2Ste >= arrCfg.brugada.type2_saddleback_ste_min_mm)) {
    findings.push(makeFinding("BRUGADA_TYPE_2", [`Saddleback STE ${v1v2Ste} mm in V1-V2`]));
  }

  const peakedT = Boolean(ecg.peaked_t_waves) || (ecg.max_t_wave_amplitude_mm ?? 0) >= arrCfg.hyperkalemia.peaked_t_wave_amplitude_precordial_mm;
  if (ecg.sine_wave_pattern || (peakedT && ((qrsMs != null && qrsMs >= arrCfg.hyperkalemia.severe_qrs_widening_ms) || ecg.flattened_or_absent_p_hyperk))) {
    findings.push(makeFinding("HYPERKALEMIA_SEVERE", ["Peaked T waves with wide QRS / absent P / sine-wave"]));
  } else if (peakedT) {
    findings.push(makeFinding("HYPERKALEMIA", ["Symmetrically peaked tented T waves"]));
  }

  if (ecg.prominent_u_waves || (ecg.u_wave_amplitude_mm ?? 0) >= arrCfg.hypokalemia.prominent_u_wave_mm) {
    findings.push(makeFinding("HYPOKALEMIA", [`Prominent U waves (${ecg.u_wave_amplitude_mm ?? ">=1.5"} mm)`]));
  }
  if (ecg.osborn_j_waves) findings.push(makeFinding("HYPOTHERMIA_OSBORN", ["Osborn (J) waves detected"]));

  const lowVolt = Boolean(ecg.low_voltage) || (ecg.max_limb_qrs_mm != null && ecg.max_limb_qrs_mm < hypCfg.low_voltage.limb_leads_max_qrs_mm);
  if (lowVolt && (ecg.electrical_alternans || (hr != null && hr > criteria.heart_rate.normal_max_bpm && ecg.pericardial_effusion_suspected))) {
    findings.push(makeFinding("CARDIAC_TAMPONADE_SUSPECTED", ["Low QRS voltage + Tachycardia / Electrical alternans"]));
  }
  if (ecg.s1q3t3 || ecg.rv_strain_twi_v1_v4) {
    const peEv: string[] = [];
    if (hr && hr > 100) peEv.push(`Sinus tachycardia (${hr} bpm)`);
    if (ecg.s1q3t3) peEv.push("S1Q3T3 pattern");
    if (ecg.rv_strain_twi_v1_v4) peEv.push("T-wave inversions V1-V4 (RV strain)");
    findings.push(makeFinding("PULMONARY_EMBOLISM", peEv));
  }
  if (
    qrsMs != null &&
    qrsMs >= arrCfg.tca_toxicity.qrs_min_ms &&
    ((ecg.avr_terminal_r_mm ?? 0) >= arrCfg.tca_toxicity.avr_terminal_r_min_mm || (ecg.avr_r_to_s_ratio ?? 0) > arrCfg.tca_toxicity.avr_r_to_s_ratio_min)
  ) {
    findings.push(makeFinding("TCA_TOXICITY", [`QRS = ${qrsMs} ms`, `aVR terminal R = ${ecg.avr_terminal_r_mm ?? "n/a"} mm`]));
  }

  const sokolow = (ecg.s_v1_mm ?? 0) + Math.max(ecg.r_v5_mm ?? 0, ecg.r_v6_mm ?? 0);
  const cornell = (ecg.r_avl_mm ?? 0) + (ecg.s_v3_mm ?? 0);
  const cornellCut = (ecg.sex || "").toLowerCase() === "female" ? hypCfg.lvh.cornell_female_ravl_plus_sv3_mm : hypCfg.lvh.cornell_male_ravl_plus_sv3_mm;
  if (sokolow >= hypCfg.lvh.sokolow_lyon_sv1_plus_rv5_or_rv6_mm || cornell > cornellCut || (ecg.r_avl_mm ?? 0) >= hypCfg.lvh.r_avl_min_mm) {
    findings.push(makeFinding("LVH", [`Sokolow-Lyon = ${sokolow} mm, Cornell = ${cornell} mm`]));
  }
  if (((ecg.v1_r_to_s_ratio ?? 0) > hypCfg.rvh.v1_r_to_s_ratio_min || (ecg.r_v1_mm ?? 0) >= hypCfg.rvh.v1_r_amplitude_min_mm) && axisDeg != null && axisDeg >= hypCfg.rvh.rad_min_degrees) {
    findings.push(makeFinding("RVH", [`Dominant R in V1 (${ecg.r_v1_mm ?? "?"} mm), Axis = ${axisDeg}°`]));
  }
  if (isPaced) findings.push(makeFinding("PACEMAKER_RHYTHM", ["Pacing spikes present"]));
  if (ecg.negative_p_qrs_lead_i && ecg.normal_precordial_r_progression !== false) {
    findings.push(makeFinding("LEAD_REVERSAL_RA_LA", ["Global negative P/QRS in Lead I with normal precordial R progression"]));
  }

  // ---- What was (not) assessed ----------------------------------------------
  const notAssessed: string[] = [];
  if (hr == null) notAssessed.push("Heart rate");
  if (ecg.rhythm_regular === undefined) notAssessed.push("Rhythm regularity");
  if (ecg.p_waves_present === undefined) notAssessed.push("P waves");
  if (prMs == null) notAssessed.push("PR interval");
  if (qrsMs == null) notAssessed.push("QRS duration");
  if (qtMs == null) notAssessed.push("QT/QTc");
  if (axisDeg == null) notAssessed.push("Electrical axis");
  if (!hasStData) notAssessed.push("ST segment (per lead)");
  if (!ecg.qrs_morphology && ecg.rsr_prime_v1 === undefined && ecg.broad_monophasic_r_lateral === undefined) notAssessed.push("QRS morphology (BBB/fascicular)");

  // ---- Empty result: never silently "normal" ----------------------------------
  if (findings.length === 0) {
    const nsrAssessable =
      hr != null && hr >= (hrCfg?.normal_min_bpm ?? 60) && hr <= (hrCfg?.normal_max_bpm ?? 100) && isRegular && pPresent === true && prMs != null && qrsMs != null;
    if (nsrAssessable) {
      findings.push(
        makeFinding("NORMAL_SINUS_RHYTHM", [`HR = ${hr} bpm`, `PR = ${prMs} ms`, `QRS = ${qrsMs} ms`, ...(qtc.primary != null ? [`QTc = ${qtc.primary} ms`] : [])])
      );
    } else if (notAssessed.length >= 4) {
      findings.push(makeFinding("INSUFFICIENT_DATA", [`Not assessed: ${notAssessed.join(", ")}`]));
    } else {
      findings.push(makeFinding("NO_ABNORMALITY_IN_ASSESSED_PARAMETERS", [`Not assessed: ${notAssessed.join(", ") || "none"}`]));
    }
  }

  // ---- Dedupe, grade confidence, sort -----------------------------------------
  const map = new Map<string, ECGFinding>();
  for (const f of findings) if (!map.has(f.code)) map.set(f.code, f);

  const q = ecg.quality;
  let confidence: Confidence = "high"; // manual / device-supplied measurements
  if (q) confidence = !q.timing_ok || !q.rhythm_ok ? "low" : "moderate"; // image-derived is never "high"

  const sorted = Array.from(map.values())
    .map((f) => ({ ...f, confidence }))
    .sort((a, b) => a.priority - b.priority);

  const isUrgent = (f: ECGFinding) => f.urgency === "CRITICAL" || f.urgency === "URGENT";
  const urgent = sorted.filter((f) => isUrgent(f) && f.confidence !== "low");
  const unverified = sorted.filter((f) => isUrgent(f) && f.confidence === "low");

  return {
    overall_urgency: sorted[0]!.urgency,
    has_urgent_findings: urgent.length > 0,
    urgent_count: urgent.length,
    measurements: {
      hr_bpm: hr,
      hr_method: method,
      rr_ms: rrMs,
      pr_ms: prMs,
      qrs_ms: qrsMs,
      qt_ms: qtMs,
      qtc_bazett_ms: qtc.bazett,
      qtc_fridericia_ms: qtc.fridericia,
      qtc_hodges_ms: qtc.hodges,
      axis_deg: axisDeg,
      axis_classification: axisLabel,
    },
    urgent_findings: urgent,
    unverified_urgent_findings: unverified,
    all_findings: sorted,
    overall_confidence: confidence,
    not_assessed: notAssessed,
    quality: q,
  };
}

export const ECG_CRITERIA_META = criteria.metadata;
export const ECG_UNITS = criteria.units_and_calibration;
