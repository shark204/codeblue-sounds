import rawCasesKb from "@/ecg_engine/ecg_cases_kb_v2.json";
import {
  ECGInput,
  ECGFinding,
  ECGInterpretationResult,
  UrgencyLevel,
  interpretECG,
} from "@/lib/ecgRules";
import { evaluateVitalsContext } from "@/lib/ecgVitalsRules";
import {
  MeasurementProvenanceMap,
  SourceConflictWarning,
  detectPixelRuleVsAiConflict,
  evaluateRuleWithProvenance,
} from "@/lib/ecgMeasurementUncertainty";
import { detectCompositeConductionSyndrome, lookupClinicalEntry } from "@/lib/ecgClinicalLibrary";
import {
  FUSION_CONSTANTS,
  FusionCandidate,
  FusedECGFinding,
  computeAiAvailability,
  computeContextAvailability,
  computeRuleAvailability,
  fuseHybridFindings,
} from "@/lib/ecgFusionEngine";

export interface WizardFeatures {
  // Step 1: Rate & Rhythm
  rhythm_class: "regular" | "irregularly_irregular" | "pauses" | null;
  rate_class: "normal" | "bradycardia" | "tachycardia" | "marked_brady" | null;
  // Step 2: P Waves & PR
  p_wave_status: "present" | "absent" | "variable" | null;
  sawtooth_flutter: boolean;
  fibrillatory_waves: boolean;
  pr_class: "normal" | "prolonged" | "short" | "variable" | null;
  delta_wave: boolean;
  av_dissociation: boolean;
  // Step 3: QRS Width & Morphology
  qrs_width_class: "narrow" | "wide" | null;
  qrs_morphology: "normal" | "rbbb" | "lbbb" | "atypical_rbbb" | "bizarre_wide" | "paced" | null;
  pacing_spikes: boolean;
  low_voltage: boolean;
  lvh_voltage: boolean;
  // Step 4: ST Segment & J-Point
  st_elevation_leads: string[];
  st_depression_leads: string[];
  st_morphology_concave: boolean;
  pr_depression: boolean;
  early_repolarization: boolean;
  brugada_pattern: "type_1" | "type_2" | null;
  // Step 5: T Wave, Q Wave & QT
  t_wave_inversion_leads: string[];
  peaked_t_waves: boolean;
  q_wave_leads: string[];
  prolonged_qt_visual: boolean;
  prominent_u_waves: boolean;
  polymorphic_vt: boolean;
  // Step 6: Clinical Context Flags (1-tap chips)
  is_competitive_athlete: boolean;
  asymptomatic_nocturnal: boolean;
  chest_pain_or_shock: boolean;
  fever_or_cocaine_or_sodium_blocker: boolean;
  dka_ckd_or_high_k: boolean;
  tremor_or_nebulizer_or_loose_lead: boolean;
  has_pacemaker: boolean;
  syncope_or_family_scd: boolean;
  post_emotional_stress: boolean;
  // Vitals & Hemodynamic Stability (Card 1)
  bp_status: "normal" | "high" | "low";
  glucose_status: "normal" | "high" | "low";
  spo2_status: "normal" | "low";
  temp_status: "normal" | "fever" | "hypothermia";
  hemodynamic_unstable: boolean;
  pleuritic_or_dehydration_or_bleed: boolean;
  // Optional Printed Header Measurements & Paper Calibration
  hr_bpm: number | null;
  pr_ms: number | null;
  qrs_ms: number | null;
  qt_ms: number | null;
  axis_deg: number | null;
  paper_speed_mm_s: 25 | 50;
  gain_mm_mv: 10 | 5;
  early_repolarization_leads?: string[];
  av_block_type?: string | null;
  wellens_pattern?: string | null;
  osborn_j_waves?: boolean;
  s1q3t3?: boolean;
  rv_strain_twi_v1_v4?: boolean;
  lafb?: boolean;
  lpfb?: boolean;
  ai_predictions?: { label: string; code: string | null; prob: number; uncertainty?: number }[];
  measurement_provenance?: Partial<MeasurementProvenanceMap>;
}

export const INITIAL_WIZARD_FEATURES: WizardFeatures = {
  rhythm_class: null,
  rate_class: null,
  p_wave_status: null,
  sawtooth_flutter: false,
  fibrillatory_waves: false,
  pr_class: null,
  delta_wave: false,
  av_dissociation: false,
  qrs_width_class: null,
  qrs_morphology: null,
  pacing_spikes: false,
  low_voltage: false,
  lvh_voltage: false,
  st_elevation_leads: [],
  st_depression_leads: [],
  st_morphology_concave: false,
  pr_depression: false,
  early_repolarization: false,
  brugada_pattern: null,
  t_wave_inversion_leads: [],
  peaked_t_waves: false,
  q_wave_leads: [],
  prolonged_qt_visual: false,
  prominent_u_waves: false,
  polymorphic_vt: false,
  is_competitive_athlete: false,
  asymptomatic_nocturnal: false,
  chest_pain_or_shock: false,
  fever_or_cocaine_or_sodium_blocker: false,
  dka_ckd_or_high_k: false,
  tremor_or_nebulizer_or_loose_lead: false,
  has_pacemaker: false,
  syncope_or_family_scd: false,
  post_emotional_stress: false,
  bp_status: "normal",
  glucose_status: "normal",
  spo2_status: "normal",
  temp_status: "normal",
  hemodynamic_unstable: false,
  pleuritic_or_dehydration_or_bleed: false,
  hr_bpm: null,
  pr_ms: null,
  qrs_ms: null,
  qt_ms: null,
  axis_deg: null,
  paper_speed_mm_s: 25,
  gain_mm_mv: 10,
  ai_predictions: [],
};

export interface CasePitfall {
  trap: string;
  why_it_misleads: string;
  how_to_avoid: string;
  mechanism_flag?: boolean;
}

export interface MatchedCaseResult {
  case_id: string;
  similarity: number;
  matched_features: string[];
  final_diagnosis: string;
  diagnosis_codes: string[];
  urgency: string;
  presentation: string;
  age: number | null;
  sex: string | null;
  ecg_findings: string[];
  pitfalls: CasePitfall[];
  teaching_points: string[];
  next_steps: string[];
}

export interface ConditionalPitfallCheck {
  id: string;
  flagKey: keyof WizardFeatures;
  question_ar: string;
  question_en: string;
  why_ar: string;
  why_en: string;
  severity: "critical" | "warning" | "info";
}

export interface ContextAdjustment {
  code: string;
  urgency: UrgencyLevel;
  title_en: string;
  title_ar: string;
  reason_en: string;
  reason_ar: string;
  action_en: string;
  action_ar: string;
  demotesCodes?: string[];
  referenceCaseId?: string;
}

export interface HybridDiagnosisResult {
  rulesResult: ECGInterpretationResult;
  finalUrgency: UrgencyLevel;
  primaryFindings: FusedECGFinding[];
  contextAdjustments: ContextAdjustment[];
  conditionalChecks: ConditionalPitfallCheck[];
  matchedCases: MatchedCaseResult[];
  aggregatedPitfalls: (CasePitfall & { from_diagnosis: string })[];
  lookAlikeWarnings: { code: string; label_en: string; label_ar: string }[];
  sourceConflict?: SourceConflictWarning | null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const KB_CASES: any[] = Array.isArray((rawCasesKb as any)?.cases) ? (rawCasesKb as any).cases : [];

const URGENCY_PRIORITY: Record<UrgencyLevel, number> = {
  CRITICAL: 1,
  URGENT: 2,
  ABNORMAL: 3,
  NORMAL: 4,
};

const URGENCY_LABELS: Record<UrgencyLevel, { en: string; ar: string }> = {
  CRITICAL: { en: "CRITICAL — Immediate Action", ar: "حرج — تدخل فوري" },
  URGENT: { en: "URGENT — Prompt Evaluation", ar: "عاجل — تقييم سريع" },
  ABNORMAL: { en: "ABNORMAL — Clinical Review", ar: "غير طبيعي — مراجعة سريرية" },
  NORMAL: { en: "ROUTINE / BENIGN PATTERN", ar: "نمط حميد / روتيني" },
};

function jaccard(a: string[], b: string[]): number {
  if (!a.length && !b.length) return 0;
  const setA = new Set(a.map((x) => x.toUpperCase()));
  const setB = new Set(b.map((x) => x.toUpperCase()));
  let inter = 0;
  setA.forEach((v) => {
    if (setB.has(v)) inter += 1;
  });
  const union = setA.size + setB.size - inter;
  return union === 0 ? 0 : inter / union;
}

/**
 * Build ECGInput for deterministic ecgRules.ts from WizardFeatures.
 * Applies paper-speed calibration correction if user recorded at 50 mm/s.
 */
export function buildEcgInputFromWizard(f: WizardFeatures): ECGInput {
  const speedScale = f.paper_speed_mm_s === 50 ? 0.5 : 1;

  const hr =
    f.hr_bpm != null
      ? Math.round(f.hr_bpm * (f.paper_speed_mm_s === 50 ? 2 : 1))
      : undefined;
  const pr = f.pr_ms != null ? Math.round(f.pr_ms * speedScale) : undefined;
  const qrs = f.qrs_ms != null ? Math.round(f.qrs_ms * speedScale) : undefined;
  const qt = f.qt_ms != null ? Math.round(f.qt_ms * speedScale) : undefined;

  const erLeads =
    f.early_repolarization_leads && f.early_repolarization_leads.length > 0
      ? f.early_repolarization_leads
      : f.early_repolarization && f.st_elevation_leads.length >= 3
        ? f.st_elevation_leads
        : undefined;

  return {
    hr_bpm: hr,
    pr_ms: pr,
    qrs_ms: qrs,
    qt_ms: qt,
    axis_deg: f.axis_deg ?? undefined,
    rhythm_class: f.rhythm_class ?? undefined,
    rr_pattern: f.rhythm_class ?? undefined,
    p_wave_status: f.p_wave_status ?? undefined,
    pr_class: f.pr_class ?? undefined,
    qrs_width_class: f.qrs_width_class ?? undefined,
    fibrillatory_waves: f.fibrillatory_waves || undefined,
    sawtooth_flutter_waves: f.sawtooth_flutter || undefined,
    delta_wave: f.delta_wave || undefined,
    av_dissociation: f.av_dissociation || undefined,
    av_block_type: f.av_block_type ?? undefined,
    qrs_morphology:
      f.qrs_morphology === "rbbb"
        ? "RBBB"
        : f.qrs_morphology === "lbbb"
          ? "LBBB"
          : f.qrs_morphology === "atypical_rbbb"
            ? "atypical_RBBB"
            : undefined,
    pacing_spikes: f.pacing_spikes || f.qrs_morphology === "paced" ? true : undefined,
    low_voltage: f.low_voltage || undefined,
    lvh_voltage: f.lvh_voltage || undefined,
    lafb: f.lafb || undefined,
    lpfb: f.lpfb || undefined,
    st_elevation_leads: f.st_elevation_leads.length ? f.st_elevation_leads : undefined,
    st_depression_leads: f.st_depression_leads.length ? f.st_depression_leads : undefined,
    q_wave_leads: f.q_wave_leads.length ? f.q_wave_leads : undefined,
    early_repolarization_leads: erLeads,
    st_morphology_concave: f.st_morphology_concave || undefined,
    pr_depression: f.pr_depression || undefined,
    brugada_pattern: f.brugada_pattern ?? undefined,
    wellens_pattern: f.wellens_pattern ?? undefined,
    t_wave_inversion_leads: f.t_wave_inversion_leads.length ? f.t_wave_inversion_leads : undefined,
    peaked_t_waves: f.peaked_t_waves || undefined,
    prominent_u_waves: f.prominent_u_waves || undefined,
    osborn_j_waves: f.osborn_j_waves || undefined,
    polymorphic_vt: f.polymorphic_vt || undefined,
    s1q3t3: f.s1q3t3 || undefined,
    rv_strain_twi_v1_v4: f.rv_strain_twi_v1_v4 || undefined,
  };
}

/**
 * Weighted Case-Based Reasoning matcher against ecg_cases_kb_v2.json (40 verified cases).
 * Never exposes textbook page numbers.
 */
export function matchCases(f: WizardFeatures, k = 3): MatchedCaseResult[] {
  const scored: MatchedCaseResult[] = [];

  for (const c of KB_CASES) {
    const inp = c.ecg_input || {};
    let score = 0;
    let maxPossible = 0;
    const matched: string[] = [];

    // Categorical ECG features (high weight)
    if (f.rhythm_class && inp.rhythm_class) {
      maxPossible += 2.0;
      if (f.rhythm_class === inp.rhythm_class) {
        score += 2.0;
        matched.push(`Rhythm: ${f.rhythm_class}`);
      }
    }
    if (f.qrs_width_class && inp.qrs_width_class) {
      maxPossible += 1.5;
      if (f.qrs_width_class === inp.qrs_width_class) {
        score += 1.5;
        matched.push(`QRS: ${f.qrs_width_class}`);
      }
    }
    if (f.pr_class && inp.pr_class) {
      maxPossible += 1.5;
      if (f.pr_class === inp.pr_class) {
        score += 1.5;
        matched.push(`PR: ${f.pr_class}`);
      }
    }
    if (f.p_wave_status && inp.p_wave_status) {
      maxPossible += 1.5;
      if (f.p_wave_status === inp.p_wave_status) {
        score += 1.5;
        matched.push(`P waves: ${f.p_wave_status}`);
      }
    }
    if (f.brugada_pattern || inp.brugada_pattern) {
      maxPossible += 3.0;
      if (f.brugada_pattern && inp.brugada_pattern === f.brugada_pattern) {
        score += 3.0;
        matched.push(`Brugada ${f.brugada_pattern}`);
      }
    }
    if (f.delta_wave || inp.delta_wave) {
      maxPossible += 2.5;
      if (f.delta_wave && inp.delta_wave) {
        score += 2.5;
        matched.push("Delta wave");
      }
    }
    if (f.pacing_spikes || inp.pacing_spikes) {
      maxPossible += 2.5;
      if (f.pacing_spikes && inp.pacing_spikes) {
        score += 2.5;
        matched.push("Pacing spikes");
      }
    }
    if (f.polymorphic_vt || inp.polymorphic_vt) {
      maxPossible += 3.0;
      if (f.polymorphic_vt && inp.polymorphic_vt) {
        score += 3.0;
        matched.push("Polymorphic VT / TdP");
      }
    }
    if (f.lvh_voltage || inp.lvh_voltage) {
      maxPossible += 2.0;
      if (f.lvh_voltage && inp.lvh_voltage) {
        score += 2.0;
        matched.push("LVH voltage");
      }
    }
    if (f.early_repolarization || (inp.early_repolarization_leads && inp.early_repolarization_leads.length)) {
      maxPossible += 2.5;
      if (f.early_repolarization && inp.early_repolarization_leads?.length) {
        score += 2.5;
        matched.push("Early repolarization");
      }
    }

    // Lead-set overlaps
    const caseSte = Array.isArray(inp.st_elevation_leads) ? inp.st_elevation_leads : [];
    if (f.st_elevation_leads.length || caseSte.length) {
      maxPossible += 2.5;
      const j = jaccard(f.st_elevation_leads, caseSte);
      if (j > 0) {
        score += 2.5 * j;
        matched.push(`ST↑ leads overlap`);
      }
    }
    const caseTwi = Array.isArray(inp.t_wave_inversion_leads) ? inp.t_wave_inversion_leads : [];
    if (f.t_wave_inversion_leads.length || caseTwi.length) {
      maxPossible += 2.0;
      const j = jaccard(f.t_wave_inversion_leads, caseTwi);
      if (j > 0) {
        score += 2.0 * j;
        matched.push(`TWI leads overlap`);
      }
    }
    const caseQ = Array.isArray(inp.q_wave_leads) ? inp.q_wave_leads : [];
    if (f.q_wave_leads.length || caseQ.length) {
      maxPossible += 1.5;
      const j = jaccard(f.q_wave_leads, caseQ);
      if (j > 0) {
        score += 1.5 * j;
        matched.push(`Q-wave leads overlap`);
      }
    }

    // Context-boost signals (crucial for mimics!)
    const codes: string[] = Array.isArray(c.diagnosis_codes) ? c.diagnosis_codes : [];
    if (f.is_competitive_athlete && (codes.includes("ATHLETE_HEART") || codes.includes("HCM_PATTERN") || codes.includes("VAGAL_AV_BLOCK_BENIGN"))) {
      score += 2.2;
      maxPossible += 2.2;
      matched.push("Competitive Athlete context");
    }
    if (f.asymptomatic_nocturnal && codes.includes("VAGAL_AV_BLOCK_BENIGN")) {
      score += 2.5;
      maxPossible += 2.5;
      matched.push("Nocturnal asymptomatic context");
    }
    if ((f.fever_or_cocaine_or_sodium_blocker || f.temp_status === "fever") && codes.includes("BRUGADA_PATTERN_DRUG_INDUCED")) {
      score += 2.5;
      maxPossible += 2.5;
      matched.push("Drug/Fever trigger");
    }
    if ((f.dka_ckd_or_high_k || f.glucose_status === "high") && (codes.includes("HYPERKALEMIA") || codes.includes("HYPERKALEMIA_SEVERE"))) {
      score += 2.5;
      maxPossible += 2.5;
      matched.push("Hyperkalemia / CKD / DKA context");
    }
    if (f.tremor_or_nebulizer_or_loose_lead && codes.includes("ARTIFACT_LEAD_FAULT")) {
      score += 3.0;
      maxPossible += 3.0;
      matched.push("Artifact / Lead fault context");
    }
    if (f.has_pacemaker && (codes.includes("PACEMAKER_DEMAND_INHIBITION") || codes.includes("PACEMAKER_RHYTHM"))) {
      score += 2.0;
      maxPossible += 2.0;
      matched.push("Pacemaker history");
    }
    if (f.post_emotional_stress && codes.includes("TAKOTSUBO_CARDIOMYOPATHY")) {
      score += 2.5;
      maxPossible += 2.5;
      matched.push("Emotional/Physical stress trigger");
    }

    const sim = maxPossible > 0 ? Math.min(1, score / maxPossible) : 0;
    if (sim >= 0.25 && matched.length > 0) {
      scored.push({
        case_id: c.case_id,
        similarity: Math.round(sim * 100) / 100,
        matched_features: matched,
        final_diagnosis: c.final_diagnosis,
        diagnosis_codes: codes,
        urgency: c.urgency || "unknown",
        presentation: c.clinical_context?.presentation || "",
        age: c.clinical_context?.age ?? null,
        sex: c.clinical_context?.sex ?? null,
        ecg_findings: Array.isArray(c.ecg_findings) ? c.ecg_findings : [],
        pitfalls: Array.isArray(c.pitfalls) ? c.pitfalls : [],
        teaching_points: Array.isArray(c.teaching_points) ? c.teaching_points : [],
        next_steps: Array.isArray(c.next_steps) ? c.next_steps : [],
      });
    }
  }

  return scored.sort((a, b) => b.similarity - a.similarity).slice(0, k);
}

/**
 * Conditional 1-tap pitfall checks that surface dynamically based on selected ECG pattern.
 */
function buildConditionalChecks(f: WizardFeatures, ruleCodes: Set<string>): ConditionalPitfallCheck[] {
  const checks: ConditionalPitfallCheck[] = [];

  if ((f.st_elevation_leads.length > 0 || ruleCodes.has("STEMI")) && !f.dka_ckd_or_high_k) {
    checks.push({
      id: "chk_hyperk",
      flagKey: "dka_ckd_or_high_k",
      question_ar: "فحص فخ سريري: هل المريض لديه فشل كلوي (CKD) أو DKA أو اشتباه ارتفاع بوتاسيوم؟",
      question_en: "Pitfall Check: Does the patient have CKD, DKA, or suspected hyperkalemia?",
      why_ar: "ارتفاع البوتاسيوم الشديد (K+ > 7) قد يحاكي جلطة STEMI تماماً ويحتاج كالسيوم وريدي فوراً.",
      why_en: "Severe hyperkalemia can mimic acute anteroseptal STEMI and requires immediate IV calcium.",
      severity: "critical",
    });
  }

  if (
    (f.rhythm_class === "pauses" ||
      ruleCodes.has("COMPLETE_HEART_BLOCK") ||
      ruleCodes.has("HIGH_GRADE_OR_2_TO_1_AV_BLOCK") ||
      ruleCodes.has("PAUSES_POSSIBLE_AV_BLOCK")) &&
    !(f.is_competitive_athlete && f.asymptomatic_nocturnal)
  ) {
    checks.push({
      id: "chk_athlete_vagal",
      flagKey: "asymptomatic_nocturnal",
      question_ar: "فحص فخ سريري: هل حدث التوقف أثناء النوم بدون أعراض في رياضي محترف (تمرين شاق يومياً)؟",
      question_en: "Pitfall Check: Did pauses occur asymptomatically during sleep in a competitive athlete?",
      why_ar: "الإحصار العصبى الليلي الحميد لدى الرياضيين المحترفين قد يبدو كـ Complete Heart Block ولا يحتاج منظم ضربات.",
      why_en: "Benign nocturnal vagal AV block in competitive athletes mimics CHB and does NOT need a pacemaker.",
      severity: "warning",
    });
  }

  if (f.brugada_pattern && !f.fever_or_cocaine_or_sodium_blocker) {
    checks.push({
      id: "chk_brugada_drug",
      flagKey: "fever_or_cocaine_or_sodium_blocker",
      question_ar: "فحص فخ سريري: هل توجد حمى أو تعاطي كوكايين أو دواء حاصر لقنوات الصوديوم؟",
      question_en: "Pitfall Check: Is there fever, cocaine use, or a sodium-channel blocker?",
      why_ar: "قد يكون نمط بروجادا مكتسباً وعابراً يزول بعلاج السبب أو خفض الحرارة.",
      why_en: "Transient acquired/drug-induced Brugada pattern resolves once fever or offending drug is treated.",
      severity: "warning",
    });
  }

  if (
    (f.sawtooth_flutter || f.rhythm_class === "irregularly_irregular" || f.polymorphic_vt) &&
    !f.tremor_or_nebulizer_or_loose_lead
  ) {
    checks.push({
      id: "chk_artifact",
      flagKey: "tremor_or_nebulizer_or_loose_lead",
      question_ar: "فحص فخ سريري: هل المريض يرتعش، يستخدم نيبولايزر، أو يوجد Lead واحد طبيعي تماماً؟",
      question_en: "Pitfall Check: Is there tremor, nebulizer use, or a completely normal single lead?",
      why_ar: "تشويش الأقطاب أو النيبولايزر قد يحاكي Flutter أو VT؛ تأكد من الكابلات قبل إعطاء أدوية.",
      why_en: "Electrode/nebulizer artifact mimics atrial flutter or VT; verify leads before antiarrhythmics.",
      severity: "info",
    });
  }

  if (f.t_wave_inversion_leads.length >= 2 && !f.is_competitive_athlete && !f.chest_pain_or_shock) {
    checks.push({
      id: "chk_hcm_athlete",
      flagKey: "is_competitive_athlete",
      question_ar: "فحص فخ سريري: هل المريض شاب أو رياضي محترف (تمرين شاق يومياً) بدون ألم صدري؟",
      question_en: "Pitfall Check: Is the patient young or a competitive athlete without chest pain?",
      why_ar: "انقلاب موجة T العميق في الرياضي الشاب قد يعني اعتلال عضلة القلب الضخامي (HCM) وليس مجرد NSTEMI.",
      why_en: "Deep T-wave inversion in a young competitive athlete strongly suggests HCM rather than NSTEMI.",
      severity: "warning",
    });
  }

  return checks;
}

/**
 * Full Hybrid Diagnostic Fusion:
 * Combines deterministic ecgRules.ts + ecg_cases_kb_v2.json Case-Based Reasoning + Context Overrides.
 */
export function runHybridEcgDiagnosis(f: WizardFeatures): HybridDiagnosisResult {
  const ecgInput = buildEcgInputFromWizard(f);
  const rulesResult = interpretECG(ecgInput);
  const matchedCases = matchCases(f, 3);

  const ruleCodes = new Set(rulesResult.all_findings.map((x) => x.code));
  const adjustments: ContextAdjustment[] = [];
  const demotedCodes = new Set<string>();

  // Build AI code set BEFORE context rules (Stage 1 Bug #3 fix: all 8 context rules read both ruleCodes & aiCodes)
  const aiPreds = Array.isArray(f.ai_predictions) ? f.ai_predictions : [];
  const aiCodes = new Set<string>(
    aiPreds
      .filter((p) => p && typeof p.prob === "number" && p.prob >= FUSION_CONSTANTS.MIN_AI_PROB && p.code)
      .map((p) => String(p.code))
  );
  const hasAnyCode = (...codes: string[]) => codes.some((c) => ruleCodes.has(c) || aiCodes.has(c));

  // Hard clinical safety guard: never demote potentially lethal arrhythmias when syncope/shock/instability is present
  const hasDangerousSymptomsOrInstability =
    f.syncope_or_family_scd || f.chest_pain_or_shock || f.hemodynamic_unstable || f.bp_status === "low";

  // 1. Benign Nocturnal Vagal AV Block in Competitive Athlete (v2_case_2_2) — WITH syncope/instability safety guard
  if (
    f.is_competitive_athlete &&
    f.asymptomatic_nocturnal &&
    !hasDangerousSymptomsOrInstability &&
    (f.rhythm_class === "pauses" ||
      hasAnyCode("COMPLETE_HEART_BLOCK", "HIGH_GRADE_OR_2_TO_1_AV_BLOCK", "HIGH_GRADE_AV_BLOCK", "PAUSES_POSSIBLE_AV_BLOCK", "BRADYCARDIA", "SINUS_BRADYCARDIA"))
  ) {
    demotedCodes.add("COMPLETE_HEART_BLOCK");
    demotedCodes.add("HIGH_GRADE_OR_2_TO_1_AV_BLOCK");
    demotedCodes.add("HIGH_GRADE_AV_BLOCK");
    demotedCodes.add("PAUSES_POSSIBLE_AV_BLOCK");
    adjustments.push({
      code: "VAGAL_AV_BLOCK_BENIGN",
      urgency: "NORMAL",
      title_en: "Benign Nocturnal Vagal AV Block (Competitive Athlete)",
      title_ar: "إحصار أذيني بطيني ليلي حميد (فرط العصب الحائر لدى رياضي محترف)",
      reason_en:
        "High-grade AV block or pauses occurring asymptomatically at night in a competitive athlete (with no syncope or hemodynamic instability) reflect high vagal tone rather than structural conduction disease.",
      reason_ar:
        "حدوث الإحصار أو التوقف أثناء النوم بدون أعراض أو إغماء في رياضي محترف يعكس ارتفاع نشاط العصب الحائر وليس مرضاً عضوياً في التوصيل.",
      action_en: "Reassure; confirm normal conduction during wakefulness/exercise. Permanent pacemaker is NOT indicated.",
      action_ar: "طمأنة المريض والتأكد من انتظام التوصيل أثناء اليقظة والمجهود؛ لا حاجة لتركيب منظم ضربات قلب.",
      demotesCodes: ["COMPLETE_HEART_BLOCK", "HIGH_GRADE_OR_2_TO_1_AV_BLOCK", "HIGH_GRADE_AV_BLOCK"],
      referenceCaseId: "v2_case_2_2",
    });
  }

  // 2. Drug / Fever-Induced Brugada Pattern (v2_case_7_2)
  if (
    (f.brugada_pattern === "type_1" || hasAnyCode("BRUGADA_TYPE_1")) &&
    (f.fever_or_cocaine_or_sodium_blocker || f.temp_status === "fever")
  ) {
    demotedCodes.add("BRUGADA_TYPE_1");
    adjustments.push({
      code: "BRUGADA_PATTERN_DRUG_INDUCED",
      urgency: "URGENT",
      title_en: "Acquired / Drug- or Fever-Induced Type 1 Brugada Pattern",
      title_ar: "نمط بروجادا من النوع الأول مُحفَّز بالأدوية أو الكوكايين أو الحمى",
      reason_en:
        "Type 1 coved ST elevation in V1–V2 occurred with a reversible trigger (fever, cocaine, or sodium-channel blocker) rather than purely baseline congenital Brugada.",
      reason_ar:
        "ارتفاع ST المقوس في V1–V2 ارتبط بمحفّز قابل للعكس (حمى، كوكايين، أو حاصر قنوات صوديوم) وليس مجرد بروجادا خلقي ثابت.",
      action_en: "Treat fever aggressively / stop offending agent, monitor telemetry, avoid Class I antiarrhythmics, and repeat ECG after clearance.",
      action_ar: "علاج الحمى فوراً وإيقاف المادة المسببة، مراقبة التخطيط، تجنب مضادات اضطراب النظم Class I، وإعادة ECG بعد التعافي.",
      demotesCodes: ["BRUGADA_TYPE_1"],
      referenceCaseId: "v2_case_7_2",
    });
  }

  // 3. Severe Hyperkalemia Mimicking STEMI (v1_case_11_1) — reads both ruleCodes & aiCodes
  if (
    (f.dka_ckd_or_high_k || f.glucose_status === "high") &&
    (f.st_elevation_leads.length > 0 ||
      f.peaked_t_waves ||
      f.qrs_width_class === "wide" ||
      (f.qrs_ms != null && f.qrs_ms >= 120) ||
      hasAnyCode("STEMI", "ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI", "LBBB", "RBBB", "HIGH_GRADE_AV_BLOCK"))
  ) {
    adjustments.push({
      code: "HYPERKALEMIA_SEVERE",
      urgency: "CRITICAL",
      title_en: "Severe Hyperkalemia Mimicking Acute STEMI (Pseudo-STEMI)",
      title_ar: "ارتفاع بوتاسيوم شديد يحاكي جلطة STEMI (فخ سريري حرج)",
      reason_en:
        "In CKD, DKA, or suspected hyperkalemia, dramatic ST elevation or conduction widening can represent severe hyperkalemia rather than coronary occlusion.",
      reason_ar:
        "في حالات الفشل الكلوي أو DKA، ارتفاع ST أو اتساع QRS قد يكون ارتفاع بوتاسيوم حاد يحاكي الجلطة تماماً.",
      action_en: "Check stat serum K+/VBG immediately; if hyperkalemia confirmed or suspected with QRS widening, give IV Calcium Gluconate BEFORE fibrinolysis.",
      action_ar: "قياس البوتاسيوم فوراً (VBG/Labs) وإعطاء كالسيوم جلوكونات وريدياً فوراً لحماية القلب قبل أي مذيب للجلطة.",
      referenceCaseId: "v1_case_11_1",
    });
  }

  // 4. Artifact / Lead Fault (v2_case_16_1) — WITH hemodynamic instability safety guard
  if (f.tremor_or_nebulizer_or_loose_lead) {
    if (!hasDangerousSymptomsOrInstability) {
      demotedCodes.add("ATRIAL_FLUTTER");
      demotedCodes.add("VENTRICULAR_TACHYCARDIA");
      demotedCodes.add("INSUFFICIENT_DATA");
    }
    adjustments.push({
      code: "ARTIFACT_LEAD_FAULT",
      urgency: hasDangerousSymptomsOrInstability ? "URGENT" : "NORMAL",
      title_en: "Suspected ECG Artifact / Disconnected Lead Fault",
      title_ar: "اشتباه تشويش بالتخطيط أو انفصال أحد الأقطاب (Artifact)",
      reason_en: hasDangerousSymptomsOrInstability
        ? "Tremor or loose lead reported, but because the patient has shock/syncope/instability, life-threatening arrhythmias were NOT demoted — verify pulse and repeat ECG immediately."
        : "Tremor, nebulizer vibration, or a loose limb electrode can mimic atrial flutter, VT, or flatline while sparing an uninvolved lead.",
      reason_ar: hasDangerousSymptomsOrInstability
        ? "تم الإبلاغ عن رعشة أو ارتخاء قطب، لكن لوجود عدم استقرار أو إغماء لم يتم استبعاد اضطرابات النظم الخطيرة — افحص النبض وأعد التخطيط فوراً."
        : "الرعشة أو النيبولايزر أو ارتخاء قطب طرفي قد يحاكي الرفرفة الأذينية أو تسارع البطين بينما يظهر Lead آخر طبيعياً تماماً.",
      action_en: "Inspect skin-electrode contact, check patient pulse, and repeat 12-lead ECG before initiating antiarrhythmics or cardioversion.",
      action_ar: "افحص النبض وتثبيت الأقطاب والكابلات وأعد التخطيط قبل إعطاء أي أدوية أو صدمات كهربائية.",
      referenceCaseId: "v2_case_16_1",
    });
  }

  // 5. HCM Pattern in Competitive Athlete (v2_case_15_11) — reads both ruleCodes & aiCodes
  if (
    f.is_competitive_athlete &&
    (f.t_wave_inversion_leads.length >= 2 || f.lvh_voltage || hasAnyCode("LVH_STRAIN", "LVH_VOLTAGE", "NSTEMI_ISCHEMIA"))
  ) {
    adjustments.push({
      code: "HCM_PATTERN",
      urgency: "URGENT",
      title_en: "Hypertrophic Cardiomyopathy (HCM) Pattern in Athlete",
      title_ar: "نمط اعتلال عضلة القلب الضخامي (HCM) لدى رياضي محترف",
      reason_en:
        "Deep lateral/inferolateral T-wave inversions or marked LVH/strain in a competitive athlete are pathological (HCM) — not benign athlete's heart and often mislabeled as NSTEMI by pure rules.",
      reason_ar:
        "انقلاب موجات T العميق أو التضخم البطيني الواضح لدى رياضي محترف علامة مرضية تشير لـ HCM وليست تغيراً طبيعياً للرياضي.",
      action_en: "Restrict competitive sports immediately and arrange urgent transthoracic echocardiography / cardiac MRI.",
      action_ar: "إيقاف التمرين التنافسي مؤقتاً وطلب إيكو قلب (Echocardiography) وتقييم قلب عاجل.",
      referenceCaseId: "v2_case_15_11",
    });
  }

  // 6. Early Repolarization (v2_case_5_1 / v2_case_5_2)
  if (f.early_repolarization) {
    demotedCodes.add("INSUFFICIENT_DATA");
    const malignant = f.syncope_or_family_scd;
    adjustments.push({
      code: "EARLY_REPOLARIZATION",
      urgency: malignant ? "URGENT" : "NORMAL",
      title_en: malignant
        ? "Malignant Early Repolarization Syndrome (High-Risk)"
        : "Benign Early Repolarization (BER) Pattern",
      title_ar: malignant
        ? "متلازمة إعادة الاستقطاب المبكر الخبيثة (خطورة عالية)"
        : "نمط إعادة الاستقطاب المبكر الحميد (BER)",
      reason_en: malignant
        ? "J-point elevation/notching combined with syncope or family history of SCD carries risk of idiopathic VF."
        : "Concave J-point elevation/notching without ischemic chest pain or reciprocal ST depression fits benign early repolarization.",
      reason_ar: malignant
        ? "ارتفاع نقطة J مع تاريخ إغماء أو موت مفاجئ عائلي يرفع خطر الرجفان البطيني."
        : "ارتفاع نقطة J المقعّر بدون ألم صدري أو انخفاض ST مقابل يتماشى مع إعادة الاستقطاب المبكر الحميد.",
      action_en: malignant
        ? "Urgent cardiology/EP evaluation and telemetry monitoring."
        : "Compare with prior ECGs; no acute reperfusion needed if asymptomatic.",
      action_ar: malignant
        ? "تقييم عاجل لكهرباء القلب ومراقبة مستمرة."
        : "مقارنة بتخطيطات سابقة؛ لا حاجة لتدخل قسطرة طالما لا توجد أعراض جلطة.",
      referenceCaseId: malignant ? "v2_case_5_2" : "v2_case_5_1",
    });
  }

  // 7. Pacemaker Normal Demand Inhibition (v2_case_12_7) — checks aiCodes too
  if (f.has_pacemaker && !f.pacing_spikes && !hasAnyCode("PACEMAKER_RHYTHM", "PACED_RHYTHM") && (f.hr_bpm == null || f.hr_bpm >= 65)) {
    adjustments.push({
      code: "PACEMAKER_DEMAND_INHIBITION",
      urgency: "NORMAL",
      title_en: "Normal Pacemaker Demand Inhibition (Not Malfunction)",
      title_ar: "تثبيط طبيعي لمنظم ضربات القلب عند الطلب (وليس عطلاً)",
      reason_en:
        "Absence of pacing spikes when the patient's intrinsic heart rate exceeds the programmed lower rate limit represents normal demand sensing.",
      reason_ar:
        "غياب إشارات المنظم (Spikes) عندما يكون نبض المريض الذاتي أسرع من الحد المضبوط هو عمل طبيعي تماماً وليس عطلاً.",
      action_en: "Verify programmed lower rate limit; do not diagnose pacemaker failure based on absent spikes alone.",
      action_ar: "تأكد من الحد الأدنى المضبوط للمنظم ولا تشخّص عطلاً لمجرد عدم رؤية Spikes.",
      referenceCaseId: "v2_case_12_7",
    });
  }

  // 8. Takotsubo Cardiomyopathy (v1_case_13e_1) — reads both ruleCodes & aiCodes
  if (
    f.post_emotional_stress &&
    (f.st_elevation_leads.length > 0 ||
      f.t_wave_inversion_leads.length >= 2 ||
      hasAnyCode("STEMI", "ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI", "NSTEMI_ISCHEMIA"))
  ) {
    adjustments.push({
      code: "TAKOTSUBO_CARDIOMYOPATHY",
      urgency: "URGENT",
      title_en: "Takotsubo (Stress) Cardiomyopathy Differential",
      title_ar: "اعتلال عضلة القلب الإجهادي (تاكوتسوبو) ضمن التشخيص التفريقي",
      reason_en:
        "Acute ST elevation or deep diffuse T-wave inversions with QT prolongation following severe emotional/physical stress can represent Takotsubo cardiomyopathy.",
      reason_ar:
        "ارتفاع ST أو انقلاب T العميق بعد صدمة عاطفية أو ضغط حاد قد يمثل متلازمة تاكوتسوبو التي تحاكي الجلطة.",
      action_en: "Manage initially as ACS until obstructive CAD is excluded by angiography/echocardiography.",
      action_ar: "تعامل معها مبدئياً كمتلازمة شريان تاجي حادة حتى استبعاد الانسداد بالقسطرة والإيكو.",
      referenceCaseId: "v1_case_13e_1",
    });
  }

  // 8b. Composite Conduction Syndromes (Bifascicular & "Trifascicular" Block Pattern per AHA/ACCF/HRS)
  const comp = detectCompositeConductionSyndrome({
    hasRbbb: hasAnyCode("RBBB", "COMPLETE_RBBB") || f.qrs_morphology === "rbbb",
    hasFascicular: hasAnyCode("FASCICULAR_BLOCK", "LAFB", "LPFB") || Boolean(f.lafb || f.lpfb),
    hasFirstDegAvb: hasAnyCode("FIRST_DEGREE_AV_BLOCK", "PR_PROLONGATION") || (f.pr_ms != null && f.pr_ms > 200),
    syncopeOrUnstable: f.syncope_or_family_scd || f.hemodynamic_unstable,
    chestPainOrShock: f.chest_pain_or_shock,
  });
  if (comp) {
    adjustments.push({
      code: comp.code, urgency: comp.urgency,
      title_en: comp.entry.title_en, title_ar: comp.entry.title_ar,
      reason_en: comp.entry.explanation_en, reason_ar: comp.entry.explanation_ar,
      action_en: comp.entry.action_en, action_ar: comp.entry.action_ar,
    });
  }

  // 9. Exhaustive Vitals x ECG Clinical Matrix (Zero Dropped Combinations)
  for (const vAdj of evaluateVitalsContext(f, ruleCodes, aiCodes)) {
    if (!adjustments.some((x) => x.code === vAdj.code)) {
      adjustments.push(vAdj);
    }
  }

  // === STAGES 1, 2 & 3: UNIFIED NOISY-OR HIERARCHICAL FUSION ===
  // All 4 source streams feed into fuseHybridFindings() — zero duplicate skipping!
  const candidates: FusionCandidate[] = [];
  const ctxAvailability = computeContextAvailability(f);

  // Stream A: Patient Vitals & Clinical Context Adjustments
  for (const adj of adjustments) {
    const urgMeta = URGENCY_LABELS[adj.urgency]; const ctxStrength = adj.urgency === "CRITICAL" ? 0.85 : adj.urgency === "URGENT" ? 0.72 : 0.65;
    candidates.push({
      code: adj.code,
      urgency: adj.urgency,
      priority: URGENCY_PRIORITY[adj.urgency],
      urgency_label_en: urgMeta.en,
      urgency_label_ar: urgMeta.ar,
      name_en: adj.title_en,
      name_ar: adj.title_ar,
      evidence: [adj.referenceCaseId ? `Matched Clinical Pattern (${adj.referenceCaseId})` : "Context-Aware Clinical Rule"],
      explanation_en: adj.reason_en,
      explanation_ar: adj.reason_ar,
      immediate_action_en: adj.action_en,
      immediate_action_ar: adj.action_ar,
      source: "context_override",
      strength: ctxStrength,
      availability: ctxAvailability,
    });
  }

  // Stream B: Deterministic ECG Rules Engine (ecgRules.ts) with per-diagnosis OCR Availability Mask
  const hasRealOtherSignal =
    adjustments.length > 0 ||
    aiPreds.some((p) => p && typeof p.prob === "number" && p.prob >= FUSION_CONSTANTS.MIN_AI_PROB) ||
    false;

  const prov = f.measurement_provenance || {};
  const pixelRuleFindings: { code: string; name_en: string; name_ar: string }[] = [];

  for (const rf of rulesResult.all_findings) {
    if (
      (rf.code === "INSUFFICIENT_DATA" || rf.code === "NO_ABNORMALITY_IN_ASSESSED_PARAMETERS") &&
      hasRealOtherSignal
    ) {
      continue;
    }
    let adjUrgency: UrgencyLevel = rf.urgency;
    let adjStrength = rf.confidence === "high" ? 0.85 : rf.confidence === "moderate" ? 0.65 : 0.45;
    let adjNameEn = rf.name_en;
    let adjNameAr = rf.name_ar;
    let adjExpEn = rf.explanation_en;
    let adjExpAr = rf.explanation_ar;

    const { uRes, usedPixelEstimate } = evaluateRuleWithProvenance(rf.code, prov, f);
    if (usedPixelEstimate) pixelRuleFindings.push({ code: rf.code, name_en: rf.name_en, name_ar: rf.name_ar });

    if (uRes && uRes.status === "possible") {
      adjUrgency = uRes.unverifiedUrgent ? "URGENT" : "ABNORMAL";
      adjStrength = uRes.unverifiedUrgent ? 0.50 : 0.35;
      adjNameEn = uRes.unverifiedUrgent ? `Possible ${rf.name_en} (Unverified Urgent)` : `Possible ${rf.name_en}`;
      adjNameAr = uRes.unverifiedUrgent ? `احتمال طارئ غير مؤكد: ${rf.name_ar}` : `محتمل: ${rf.name_ar}`;
      if (uRes.note_en) adjExpEn = `${rf.explanation_en} [${uRes.note_en}]`;
      if (uRes.note_ar) adjExpAr = `${rf.explanation_ar} [${uRes.note_ar}]`;
    } else if (usedPixelEstimate) {
      // Even if (value - P95) crosses a threshold, a pixel_estimate alone is never verified clinical truth:
      // cap strength at 0.42 (Preliminary Finding) and cap urgency at URGENT (never red CRITICAL without manual/OCR/AI)
      if (adjUrgency === "CRITICAL") adjUrgency = "URGENT";
      adjStrength = Math.min(adjStrength, 0.42);
      adjNameEn = `${rf.name_en} (Auto Estimate — Verify)`;
      adjNameAr = `${rf.name_ar} (تقدير آلي — يُراجع)`;
      if (uRes?.note_en) adjExpEn = `${rf.explanation_en} [${uRes.note_en}]`;
      if (uRes?.note_ar) adjExpAr = `${rf.explanation_ar} [${uRes.note_ar}]`;
    } else if (uRes && uRes.note_en) {
      adjExpEn = `${rf.explanation_en} [${uRes.note_en}]`;
      adjExpAr = `${rf.explanation_ar} [${uRes.note_ar}]`;
    }

    const urgMeta = URGENCY_LABELS[adjUrgency];
    const ruleAvailability = computeRuleAvailability(rf.code, f);
    candidates.push({
      code: rf.code,
      urgency: adjUrgency,
      priority: URGENCY_PRIORITY[adjUrgency],
      urgency_label_en: urgMeta.en,
      urgency_label_ar: urgMeta.ar,
      name_en: adjNameEn,
      name_ar: adjNameAr,
      evidence: rf.evidence,
      explanation_en: adjExpEn,
      explanation_ar: adjExpAr,
      immediate_action_en: rf.immediate_action_en,
      immediate_action_ar: rf.immediate_action_ar,
      source: "rules",
      strength: adjStrength,
      availability: ruleAvailability,
    });
  }

  // Stream C: DenseNet121 v2 Visual AI Predictions with MC Dropout Availability (A_ai = 1 - u)
  const criticalAiCodes = new Set(["STEMI", "ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI", "VENTRICULAR_TACHYCARDIA", "VENTRICULAR_FIBRILLATION", "COMPLETE_HEART_BLOCK", "HIGH_GRADE_AV_BLOCK", "HYPERKALEMIA", "TORSADES_DE_POINTES"]);
  const urgentAiCodes = new Set(["NSTEMI_ISCHEMIA", "ATRIAL_FIBRILLATION", "ATRIAL_FLUTTER", "SVT_TACHYCARDIA", "SVT_NARROW", "HIGH_GRADE_OR_2_TO_1_AV_BLOCK", "LONG_QT", "BRUGADA_TYPE_1", "ACUTE_PERICARDITIS", "WPW_PATTERN"]);
  const normalAiCodes = new Set(["NORMAL_SINUS_RHYTHM"]);

  for (const pred of aiPreds) {
    if (!pred || typeof pred.prob !== "number" || pred.prob < FUSION_CONSTANTS.MIN_AI_PROB) continue;
    const code = pred.code || `AI_${pred.label.replace(/[^A-Z0-9]+/gi, "_").toUpperCase()}`;
    const pct = Math.round(pred.prob * 100);
    const urg: UrgencyLevel = criticalAiCodes.has(code)
      ? "CRITICAL"
      : urgentAiCodes.has(code)
        ? "URGENT"
        : normalAiCodes.has(code)
          ? "NORMAL"
          : "ABNORMAL";
    const urgMeta = URGENCY_LABELS[urg];
    const aiAvailability = computeAiAvailability(pred.uncertainty);
    const lib = lookupClinicalEntry(code, pred.label);
    candidates.push({
      code,
      urgency: urg,
      priority: URGENCY_PRIORITY[urg],
      urgency_label_en: urgMeta.en,
      urgency_label_ar: urgMeta.ar,
      name_en: lib ? lib.title_en : pred.label,
      name_ar: lib ? lib.title_ar : pred.label,
      evidence: [],
      explanation_en: lib
        ? lib.explanation_en
        : `Waveform morphology matches ${pred.label} (${pct}% visual confidence).`,
      explanation_ar: lib
        ? lib.explanation_ar
        : `تتوافق مورفولوجية التخطيط مع ${pred.label} (ثقة بصرية ${pct}%).`,
      immediate_action_en: lib
        ? lib.action_en
        : urg === "CRITICAL"
          ? "Verify immediately on full 12-lead ECG and assess patient hemodynamics."
          : "Correlate with patient symptoms, vitals, and serial ECGs.",
      immediate_action_ar: lib
        ? lib.action_ar
        : urg === "CRITICAL"
          ? "تحقق فوراً من التخطيط الكامل 12-lead وقيّم استقرار المريض الحركي."
          : "اربط النتيجة بأعراض المريض والعلامات الحيوية وأعد التخطيط للمتابعة.",
      source: "ai_model",
      strength: pred.prob,
      availability: aiAvailability,
    });
  }

  // NOTE (Anti-Double-Counting): The 40-Case Reference KB is intentionally EXCLUDED from Noisy-OR fusion
  // because it matches on overlapping ECG features. It is surfaced strictly in `matchedCases` & `aggregatedPitfalls`
  // as separate educational reference cases and clinical pitfalls.

  // Fallback if all streams empty
  if (candidates.length === 0) {
    for (const rf of rulesResult.all_findings) {
      candidates.push({
        ...rf,
        source: "rules",
        strength: 0.5,
        availability: computeRuleAvailability(rf.code, f),
      });
    }
  }

  const combined = fuseHybridFindings(candidates, demotedCodes);

  // Aggregate unique clinical pitfalls from top matched cases
  const aggregatedPitfalls: (CasePitfall & { from_diagnosis: string })[] = [];
  const seenTraps = new Set<string>();
  for (const mc of matchedCases) {
    for (const p of mc.pitfalls) {
      if (!seenTraps.has(p.trap)) {
        seenTraps.add(p.trap);
        aggregatedPitfalls.push({ ...p, from_diagnosis: mc.final_diagnosis });
      }
    }
  }

  // Look-alike warnings from demoted findings (Tier 3)
  const lookAlikeWarnings: { code: string; label_en: string; label_ar: string }[] = [];
  for (const item of combined) {
    if (item.isDemoted) {
      lookAlikeWarnings.push({
        code: item.code,
        label_en: `Look-alike ruled down by context (Tier 3): ${item.name_en}`,
        label_ar: `تشخيص مشابه تم استبعاده بالسياق السريري (Tier 3): ${item.name_ar}`,
      });
    }
  }

  const conditionalChecks = buildConditionalChecks(f, ruleCodes);
  const topActive = combined.find((x) => !x.isDemoted) ?? combined[0];

  const sourceConflict = detectPixelRuleVsAiConflict(
    pixelRuleFindings,
    aiPreds.filter((p) => p && typeof p.prob === "number" && p.code).map((p) => ({ code: String(p.code), label: p.label, prob: p.prob }))
  );

  return {
    rulesResult,
    finalUrgency: topActive?.urgency || rulesResult.overall_urgency,
    primaryFindings: combined,
    contextAdjustments: adjustments,
    conditionalChecks,
    matchedCases,
    aggregatedPitfalls: aggregatedPitfalls.slice(0, 5),
    lookAlikeWarnings,
    sourceConflict,
  };
}
