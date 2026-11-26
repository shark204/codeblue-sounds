import type { ECGFinding, UrgencyLevel } from "@/lib/ecgRules";
import type { WizardFeatures } from "@/lib/ecgHybridEngine";

/**
 * Code Blue ECG Noisy-OR Hybrid Fusion Engine (v1)
 *
 * Implements:
 *  1. Hierarchical Parent/Child Canonical Code Map (consensus at Parent, display at most-specific Child)
 *  2. Per-Diagnosis OCR/Header Availability Mask (A_rules computed per rule, not globally)
 *  3. MC Dropout Uncertainty-Weighted AI Availability (A_ai = clamp(1 - u, 0.1, 1.0))
 *  4. Noisy-OR Confidence + Gated Consensus RankScore (Confidence strictly separated from RankScore)
 *  5. 3-Tier Clinical Safety Floor (Tier 1 CRITICAL, Tier 2 Active, Tier 3 Demoted / Look-alikes)
 *
 * TODO: replace hand-tuned weights/thresholds with logistic-regression-fitted parameters once >=500 labeled cases exist.
 * TODO: measure empirical source correlation between DenseNet121 v2 and ecgRules.ts once >=500 labeled cases exist,
 *       and apply a copula or correlation-discounted OR if over-confidence is observed.
 */

export const FUSION_CONSTANTS = {
  /** Minimum effective source evidence (S_s * A_s) required to count as a "strong" agreeing source */
  STRONG_EFF_THRESHOLD: 0.4,
  /** Fixed multiplicative consensus boost applied ONLY when n_strong >= 2 */
  CONSENSUS_BOOST: 0.15,
  /** Minimum raw AI probability to enter the fusion pool */
  MIN_AI_PROB: 0.3,
  /** Fallback MC Dropout uncertainty when not provided by legacy callers */
  DEFAULT_AI_UNCERTAINTY: 0.1,
  /** Minimum per-rule availability floor when OCR misses required numeric header fields */
  MIN_OCR_AVAILABILITY_FLOOR: 0.2,
  /** Minimum effective confidence for an AI-only CRITICAL prediction to enter Tier 1 Safety Floor (prevents pos_weight alarm fatigue) */
  AI_CRITICAL_TIER1_THRESHOLD: 0.65,
  /** Upper bound for Confidence and RankScore */
  MAX_SCORE: 1.0,
  /** Strict Certainty Gate: minimum calibrated AI prob & availability to count as CONFIRMED without corroboration */
  CERTAINTY_MIN_AI_PROB: 0.55,
  CERTAINTY_MIN_AI_AVAIL: 0.65,
  CERTAINTY_MIN_RULE_EFF: 0.50,
} as const;

export type FusionSourceKind = "rules" | "ai_model" | "context_override" | "case_kb";

export interface SourceContribution {
  source: FusionSourceKind;
  code: string;
  /** Raw source strength S_s in [0, 1] */
  strength: number;
  /** Source availability/reliability A_s in [0, 1] */
  availability: number;
  /** Effective evidence eff_s = S_s * A_s */
  effective: number;
  /** Human-readable evidence summary for educational Evidence List */
  evidence_en: string;
  evidence_ar: string;
}

export interface FusionCandidate {
  code: string;
  urgency: UrgencyLevel;
  priority: number;
  urgency_label_en: string;
  urgency_label_ar: string;
  name_en: string;
  name_ar: string;
  explanation_en: string;
  explanation_ar: string;
  immediate_action_en: string;
  immediate_action_ar: string;
  evidence: string[];
  source: FusionSourceKind;
  /** Raw source strength S_s in [0, 1] */
  strength: number;
  /** Source availability A_s in [0, 1] */
  availability: number;
  /** True when derived from unverified pixel_estimate or 'possible' uncertainty band — cannot pass certainty gate alone */
  isUncertainEstimate?: boolean;
  /** Optional evidence label for the unified Evidence List */
  evidence_summary_en?: string;
  evidence_summary_ar?: string;
}

export interface FusedECGFinding extends ECGFinding {
  source: FusionSourceKind;
  /** Canonical Parent code used for cross-source consensus */
  canonicalParent: string;
  /** Calibrated Noisy-OR confidence in [0, 1] — displayed to the student (unmodified by demotion/boost) */
  fusionConfidence: number;
  /** Ranking score in [0, 1] — used ONLY for sorting inside each clinical Tier */
  rankScore: number;
  /** Tier 1 = CRITICAL Safety Floor, Tier 2 = Active Findings, Tier 3 = Demoted / Look-alikes */
  tier: 1 | 2 | 3;
  /** Number of independent sources with eff_s >= STRONG_EFF_THRESHOLD */
  nStrongSources: number;
  /** Whether this finding was clinically demoted into Tier 3 */
  isDemoted: boolean;
  /** Clinical analytical concordance level displayed to students instead of uncalibrated percentages */
  concordanceLevel: "strong" | "moderate" | "preliminary";
  concordanceLabelEn: string;
  concordanceLabelAr: string;
  /** Strict clinical certainty gate: only 'confirmed' findings may be displayed as active diagnoses */
  isConfirmed: boolean;
  certaintyTier: "confirmed" | "unverified";
  /** Full per-source breakdown for transparent medical education */
  sourceBreakdown: SourceContribution[];
}

export interface CanonicalNode {
  parent: string;
  /** Higher specificity wins display title/code when merging sources under the same parent */
  specificity: number;
}

/**
 * True Hierarchical Parent/Child Canonical Code Map.
 * Consensus is computed at `parent`; display keeps the highest-`specificity` child.
 */
export const CANONICAL_HIERARCHY: Record<string, CanonicalNode> = {
  // STEMI / Acute Transmural MI family
  STEMI: { parent: "STEMI", specificity: 1 },
  ANTERIOR_SEPTAL_MI: { parent: "STEMI", specificity: 3 },
  INFERIOR_POSTERIOR_MI: { parent: "STEMI", specificity: 3 },
  LATERAL_MI: { parent: "STEMI", specificity: 3 },
  STEMI_ANTERIOR: { parent: "STEMI", specificity: 3 },
  STEMI_INFERIOR: { parent: "STEMI", specificity: 3 },
  STEMI_LATERAL: { parent: "STEMI", specificity: 3 },
  STEMI_POSTERIOR: { parent: "STEMI", specificity: 3 },
  DE_WINTER_STEMI: { parent: "STEMI", specificity: 3 },
  CARDIOGENIC_SHOCK_RV_MI: { parent: "STEMI", specificity: 3 },

  // NSTEMI / Subendocardial Ischemia family
  NSTEMI_ISCHEMIA: { parent: "ISCHEMIA_ACS", specificity: 2 },
  WELLENS_SYNDROME: { parent: "ISCHEMIA_ACS", specificity: 3 },
  ST_DEPRESSION_ISCHEMIA: { parent: "ISCHEMIA_ACS", specificity: 2 },
  HYPERTENSIVE_ACS_DISSECTION: { parent: "ISCHEMIA_ACS", specificity: 3 },

  // Atrial Fibrillation family
  ATRIAL_FIBRILLATION: { parent: "ATRIAL_FIBRILLATION", specificity: 1 },
  ATRIAL_FIBRILLATION_RVR: { parent: "ATRIAL_FIBRILLATION", specificity: 2 },
  PREEXCITED_AFIB: { parent: "ATRIAL_FIBRILLATION", specificity: 3 },
  SECONDARY_AFIB_SEPSIS_FEVER: { parent: "ATRIAL_FIBRILLATION", specificity: 3 },

  // Atrial Flutter family
  ATRIAL_FLUTTER: { parent: "ATRIAL_FLUTTER", specificity: 2 },

  // SVT family
  SVT: { parent: "SVT", specificity: 1 },
  SVT_TACHYCARDIA: { parent: "SVT", specificity: 2 },
  SVT_NARROW: { parent: "SVT", specificity: 3 },
  AVNRT: { parent: "SVT", specificity: 3 },
  AVRT: { parent: "SVT", specificity: 3 },

  // High-Grade / Complete AV Block family
  HIGH_GRADE_AVB: { parent: "HIGH_GRADE_AVB", specificity: 1 },
  PAUSES_POSSIBLE_AV_BLOCK: { parent: "HIGH_GRADE_AVB", specificity: 1 },
  HIGH_GRADE_AV_BLOCK: { parent: "HIGH_GRADE_AVB", specificity: 2 },
  HIGH_GRADE_OR_2_TO_1_AV_BLOCK: { parent: "HIGH_GRADE_AVB", specificity: 2 },
  MOBITZ_TYPE_2: { parent: "HIGH_GRADE_AVB", specificity: 3 },
  COMPLETE_HEART_BLOCK: { parent: "HIGH_GRADE_AVB", specificity: 3 },

  // First-Degree AV Block family
  FIRST_DEGREE_AVB: { parent: "FIRST_DEGREE_AVB", specificity: 1 },
  FIRST_DEGREE_AV_BLOCK: { parent: "FIRST_DEGREE_AVB", specificity: 2 },
  PR_PROLONGATION: { parent: "FIRST_DEGREE_AVB", specificity: 1 },

  // Bradycardia family
  BRADYCARDIA: { parent: "BRADYCARDIA", specificity: 1 },
  SINUS_BRADYCARDIA: { parent: "BRADYCARDIA", specificity: 2 },
  MARKED_BRADYCARDIA: { parent: "BRADYCARDIA", specificity: 2 },
  CUSHING_REFLEX_BRADYCARDIA: { parent: "BRADYCARDIA", specificity: 3 },

  // Bundle Branch Block families
  RBBB: { parent: "RBBB", specificity: 2 },
  COMPLETE_RBBB: { parent: "RBBB", specificity: 3 },
  INCOMPLETE_RBBB: { parent: "RBBB", specificity: 2 },
  LBBB: { parent: "LBBB", specificity: 2 },
  COMPLETE_LBBB: { parent: "LBBB", specificity: 3 },

  // LVH / Hypertrophy family
  LVH: { parent: "LVH", specificity: 1 },
  LVH_VOLTAGE: { parent: "LVH", specificity: 2 },
  LVH_STRAIN: { parent: "LVH", specificity: 3 },
  HYPERTENSIVE_LVH_STRAIN: { parent: "LVH", specificity: 3 },

  // Long QT / Torsades family
  LONG_QT: { parent: "LONG_QT", specificity: 1 },
  PROLONGED_QTC: { parent: "LONG_QT", specificity: 2 },
  MARKEDLY_PROLONGED_QTC: { parent: "LONG_QT", specificity: 3 },
  HYPOGLYCEMIA_QT_ARRHYTHMIA: { parent: "LONG_QT", specificity: 3 },

  // Ventricular Arrhythmia families
  VENTRICULAR_TACHYCARDIA: { parent: "VENTRICULAR_TACHYCARDIA", specificity: 2 },
  POLYMORPHIC_VT: { parent: "VENTRICULAR_TACHYCARDIA", specificity: 3 },
  TORSADES_DE_POINTES: { parent: "VENTRICULAR_TACHYCARDIA", specificity: 3 },
  VENTRICULAR_FIBRILLATION: { parent: "VENTRICULAR_FIBRILLATION", specificity: 3 },
  PVC_BIGEMINY: { parent: "VENTRICULAR_ECTOPY", specificity: 2 },
  VENTRICULAR_ECTOPY: { parent: "VENTRICULAR_ECTOPY", specificity: 1 },

  // Brugada family
  BRUGADA_TYPE_1: { parent: "BRUGADA", specificity: 2 },
  BRUGADA_PATTERN_DRUG_INDUCED: { parent: "BRUGADA", specificity: 3 },
  FEVER_UNMASKED_BRUGADA: { parent: "BRUGADA", specificity: 3 },

  // Hyperkalemia family
  HYPERKALEMIA: { parent: "HYPERKALEMIA", specificity: 1 },
  HYPERKALEMIA_SEVERE: { parent: "HYPERKALEMIA", specificity: 3 },
  DKA_HYPERKALEMIA_PSEUDO_STEMI: { parent: "HYPERKALEMIA", specificity: 3 },

  // Pacemaker family
  PACEMAKER_RHYTHM: { parent: "PACEMAKER", specificity: 2 },
  PACED_RHYTHM: { parent: "PACEMAKER", specificity: 2 },
  PACEMAKER_DEMAND_INHIBITION: { parent: "PACEMAKER", specificity: 3 },

  // Normal family
  NORMAL_SINUS_RHYTHM: { parent: "NORMAL_SINUS_RHYTHM", specificity: 2 },
  NO_ABNORMALITY_IN_ASSESSED_PARAMETERS: { parent: "NORMAL_SINUS_RHYTHM", specificity: 1 },
};

export function getCanonicalNode(code: string): CanonicalNode {
  const key = String(code || "").toUpperCase();
  return CANONICAL_HIERARCHY[key] ?? { parent: key, specificity: 2 };
}

/**
 * Per-diagnosis OCR/Header field requirements for computing A_rules.
 * Morphology-driven diagnoses (e.g. STEMI, Brugada, Pericarditis) do NOT depend on OCR numbers
 * and therefore are NOT penalized when header OCR fails.
 */
type HeaderFieldKey = "hr_bpm" | "pr_ms" | "qrs_ms" | "qt_ms" | "axis_deg";

const RULE_REQUIRED_OCR_FIELDS: Record<string, HeaderFieldKey[]> = {
  BRADYCARDIA: ["hr_bpm"],
  SINUS_BRADYCARDIA: ["hr_bpm"],
  MARKED_BRADYCARDIA: ["hr_bpm"],
  TACHYCARDIA: ["hr_bpm"],
  SINUS_TACHYCARDIA: ["hr_bpm"],
  ATRIAL_FIBRILLATION_RVR: ["hr_bpm"],
  SVT: ["hr_bpm", "qrs_ms"],
  SVT_NARROW: ["hr_bpm", "qrs_ms"],
  SVT_TACHYCARDIA: ["hr_bpm", "qrs_ms"],
  FIRST_DEGREE_AVB: ["pr_ms"],
  FIRST_DEGREE_AV_BLOCK: ["pr_ms"],
  PR_PROLONGATION: ["pr_ms"],
  WPW_PATTERN: ["pr_ms", "qrs_ms"],
  RBBB: ["qrs_ms"],
  LBBB: ["qrs_ms"],
  WIDE_QRS_TACHYCARDIA: ["hr_bpm", "qrs_ms"],
  LONG_QT: ["hr_bpm", "qt_ms"],
  PROLONGED_QTC: ["hr_bpm", "qt_ms"],
  MARKEDLY_PROLONGED_QTC: ["hr_bpm", "qt_ms"],
  LEFT_AXIS_DEVIATION: ["axis_deg"],
  RIGHT_AXIS_DEVIATION: ["axis_deg"],
  EXTREME_AXIS_DEVIATION: ["axis_deg"],
  LAFB: ["axis_deg", "qrs_ms"],
  LPFB: ["axis_deg", "qrs_ms"],
};

export function computeRuleAvailability(code: string, f: WizardFeatures): number {
  const parent = getCanonicalNode(code).parent;
  const required = RULE_REQUIRED_OCR_FIELDS[code] ?? RULE_REQUIRED_OCR_FIELDS[parent];
  if (!required || required.length === 0) {
    // Morphology- or rhythm-class-driven rule (e.g., STEMI, Brugada, Pericarditis): independent of header OCR
    return 1.0;
  }
  let captured = 0;
  for (const field of required) {
    const val = f[field];
    if (typeof val === "number" && Number.isFinite(val)) {
      captured += 1;
    }
  }
  if (captured === 0) {
    return FUSION_CONSTANTS.MIN_OCR_AVAILABILITY_FLOOR;
  }
  return Math.max(FUSION_CONSTANTS.MIN_OCR_AVAILABILITY_FLOOR, captured / required.length);
}

export function computeAiAvailability(uncertainty?: number): number {
  const u =
    typeof uncertainty === "number" && Number.isFinite(uncertainty)
      ? Math.min(0.9, Math.max(0.0, uncertainty))
      : FUSION_CONSTANTS.DEFAULT_AI_UNCERTAINTY;
  return Math.max(0.1, Math.min(1.0, 1 - u));
}

export function computeContextAvailability(f: WizardFeatures): number {
  // Fraction of patient context dimensions actively provided by the user
  let filled = 0;
  const total = 4;
  if (
    f.bp_status !== "normal" ||
    f.glucose_status !== "normal" ||
    f.spo2_status !== "normal" ||
    f.temp_status !== "normal" ||
    f.hemodynamic_unstable
  ) {
    filled += 1;
  }
  if (f.chest_pain_or_shock || f.syncope_or_family_scd || f.pleuritic_or_dehydration_or_bleed || f.asymptomatic_nocturnal) {
    filled += 1;
  }
  if (f.is_competitive_athlete) filled += 1;
  if (
    f.dka_ckd_or_high_k ||
    f.fever_or_cocaine_or_sodium_blocker ||
    f.has_pacemaker ||
    f.tremor_or_nebulizer_or_loose_lead ||
    f.post_emotional_stress
  ) {
    filled += 1;
  }
  return filled === 0 ? 0.5 : Math.min(1.0, 0.7 + 0.3 * (filled / total));
}

/**
 * Pure Noisy-OR + Gated Consensus calculation.
 * Exported independently so automated acceptance tests can verify exact mathematical behavior.
 */
export function computeNoisyOrScore(sources: { strength: number; availability: number }[]): {
  confidence: number;
  rankScore: number;
  nStrong: number;
} {
  let complementProduct = 1.0;
  let nStrong = 0;
  for (const s of sources) {
    const eff = Math.max(0, Math.min(1, s.strength * s.availability));
    complementProduct *= 1 - eff;
    if (eff >= FUSION_CONSTANTS.STRONG_EFF_THRESHOLD) {
      nStrong += 1;
    }
  }
  const confidence = Math.max(0, Math.min(FUSION_CONSTANTS.MAX_SCORE, 1 - complementProduct));
  const boostMultiplier = nStrong >= 2 ? 1 + FUSION_CONSTANTS.CONSENSUS_BOOST : 1.0;
  const rankScore = Math.min(FUSION_CONSTANTS.MAX_SCORE, confidence * boostMultiplier);
  return { confidence, rankScore, nStrong };
}

export function assignClinicalTier(
  urgency: UrgencyLevel,
  isDemoted: boolean,
  hasNonAiSupport = true,
  fusionConfidence = 1.0
): 1 | 2 | 3 {
  if (isDemoted) return 3; // Tier 3: Look-alike warnings (never competes with Tier 1 or Tier 2)
  if (urgency === "CRITICAL") {
    // Deterministic rules or Vitals emergencies enter Tier 1 immediately;
    // AI-only critical predictions must exceed AI_CRITICAL_TIER1_THRESHOLD to prevent pos_weight alarm fatigue
    if (hasNonAiSupport || fusionConfidence >= FUSION_CONSTANTS.AI_CRITICAL_TIER1_THRESHOLD) {
      return 1;
    }
  }
  return 2; // Tier 2: Active URGENT / ABNORMAL / NORMAL findings (and weak uncorroborated AI criticals)
}

const SOURCE_TEXT_PRIORITY: Record<FusionSourceKind, number> = {
  context_override: 4,
  rules: 3,
  case_kb: 2,
  ai_model: 1,
};

const URGENCY_SEVERITY: Record<UrgencyLevel, number> = {
  CRITICAL: 4,
  URGENT: 3,
  ABNORMAL: 2,
  NORMAL: 1,
};

/**
 * Fuses multi-source diagnostic candidates under the Hierarchical Canonical Code Map:
 *  - Groups by Canonical Parent for Noisy-OR consensus
 *  - Preserves the most specific Child code/title for student display
 *  - Prefers deterministic Rules / Context clinical explanations over generic AI text
 *  - Builds a transparent per-source Evidence List
 *  - Sorts strictly by (tier ASC, rankScore DESC)
 */
export function fuseHybridFindings(
  candidates: FusionCandidate[],
  demotedCodes: Set<string>
): FusedECGFinding[] {
  // Check demotion at both exact code and parent level
  const demotedParents = new Set<string>();
  for (const dc of demotedCodes) {
    demotedParents.add(getCanonicalNode(dc).parent);
  }

  // Group candidates by Canonical Parent
  const groups = new Map<string, FusionCandidate[]>();
  for (const cand of candidates) {
    const parent = getCanonicalNode(cand.code).parent;
    const list = groups.get(parent) ?? [];
    list.push(cand);
    groups.set(parent, list);
  }

  const fused: FusedECGFinding[] = [];

  for (const [parent, group] of groups.entries()) {
    // Deduplicate per source kind inside a parent group (keep strongest contribution per source)
    const bestPerSource = new Map<FusionSourceKind, FusionCandidate>();
    for (const c of group) {
      const prev = bestPerSource.get(c.source);
      if (!prev || c.strength * c.availability > prev.strength * prev.availability) {
        bestPerSource.set(c.source, c);
      }
    }

    const contributions: SourceContribution[] = [];
    for (const c of bestPerSource.values()) {
      const eff = Math.max(0, Math.min(1, c.strength * c.availability));
      const pct = Math.round(eff * 100);
      const defaultEn =
        c.source === "ai_model"
          ? `DenseNet121 v2 Visual AI (${Math.round(c.strength * 100)}% prob, A=${c.availability.toFixed(2)} -> ${pct}%)`
          : c.source === "rules"
            ? `Deterministic ECG Rules (${pct}% effective)`
            : c.source === "context_override"
              ? `Vitals & Clinical Context (${pct}% effective)`
              : `40-Case Clinical KB (${pct}% match)`;
      const defaultAr =
        c.source === "ai_model"
          ? `الذكاء الاصطناعي DenseNet121 v2 (احتمال ${Math.round(c.strength * 100)}%، موثوقية ${c.availability.toFixed(2)} ← ${pct}%)`
          : c.source === "rules"
            ? `محرك القواعد الحتمي ecgRules (${pct}%)`
            : c.source === "context_override"
              ? `العلامات الحيوية والسياق السريري (${pct}%)`
              : `قاعدة الحالات السريرية الموثّقة (${pct}%)`;
      contributions.push({
        source: c.source,
        code: c.code,
        strength: c.strength,
        availability: c.availability,
        effective: eff,
        evidence_en: c.evidence_summary_en || defaultEn,
        evidence_ar: c.evidence_summary_ar || defaultAr,
      });
    }

    const { confidence, rankScore, nStrong } = computeNoisyOrScore(contributions);

    // Pick most-specific Child for display code/title
    const displayLeader = [...group].sort((a, b) => {
      const specDiff = getCanonicalNode(b.code).specificity - getCanonicalNode(a.code).specificity;
      if (specDiff !== 0) return specDiff;
      return SOURCE_TEXT_PRIORITY[b.source] - SOURCE_TEXT_PRIORITY[a.source];
    })[0];

    // Pick richest clinical explanation/action (context_override > rules > case_kb > ai_model)
    const explanationLeader = [...group].sort(
      (a, b) => SOURCE_TEXT_PRIORITY[b.source] - SOURCE_TEXT_PRIORITY[a.source]
    )[0];

    // Highest urgency among non-demoted members in the group
    const urgencyLeader = [...group].sort(
      (a, b) => URGENCY_SEVERITY[b.urgency] - URGENCY_SEVERITY[a.urgency]
    )[0];

    // A group is demoted if any exact code in the group is explicitly demoted (or its parent is demoted without a context_override rescuing it)
    const hasContextRescue = group.some((c) => c.source === "context_override" && !demotedCodes.has(c.code));
    const isDemoted =
      !hasContextRescue &&
      (group.some((c) => demotedCodes.has(c.code)) || demotedParents.has(parent));

    const hasNonAiSupport = group.some((c) => c.source === "rules" || c.source === "context_override");
    const tier = assignClinicalTier(urgencyLeader.urgency, isDemoted, hasNonAiSupport, confidence);

    // Merge transparent Evidence List (source breakdown + unique raw evidence strings)
    const evidenceSet = new Set<string>();
    for (const sc of contributions) {
      evidenceSet.add(sc.evidence_en);
    }
    for (const c of group) {
      for (const ev of c.evidence) {
        if (ev && !evidenceSet.has(ev)) evidenceSet.add(ev);
      }
    }

    // Strip any stale "(AI XX%)" suffix; display clinical concordance badge rather than uncalibrated %
    const cleanNameEn = displayLeader.name_en.replace(/\s*\(AI\s+\d+%\)\s*$/i, "");
    const cleanNameAr = displayLeader.name_ar.replace(/\s*\(توقع الذكاء الاصطناعي\s+\d+%\)\s*$/i, "");

    // Strict Certainty Gate (Zero Unverified Pollution):
    // Confirmed ONLY via: (a) >=2 non-uncertain corroborating sources, (b) confirmed context_override,
    // (c) deterministic manual/OCR rule (!isUncertainEstimate), or (d) high-certainty calibrated Visual AI.
    const verifiedGroup = group.filter((c) => !c.isUncertainEstimate);
    const verifiedStrongCount = verifiedGroup.filter(
      (c) => c.strength * c.availability >= FUSION_CONSTANTS.STRONG_EFF_THRESHOLD
    ).length;
    const hasConfirmedContext = verifiedGroup.some(
      (c) => c.source === "context_override" && c.strength * c.availability >= FUSION_CONSTANTS.CERTAINTY_MIN_RULE_EFF
    );
    const hasConfirmedRule = verifiedGroup.some(
      (c) => c.source === "rules" && c.strength * c.availability >= FUSION_CONSTANTS.CERTAINTY_MIN_RULE_EFF
    );
    const hasConfirmedAi = verifiedGroup.some(
      (c) =>
        c.source === "ai_model" &&
        c.strength >= FUSION_CONSTANTS.CERTAINTY_MIN_AI_PROB &&
        c.availability >= FUSION_CONSTANTS.CERTAINTY_MIN_AI_AVAIL
    );
    const isConfirmed =
      verifiedStrongCount >= 2 || hasConfirmedContext || hasConfirmedRule || hasConfirmedAi;
    const certaintyTier: "confirmed" | "unverified" = isConfirmed ? "confirmed" : "unverified";

    // Unverified/possible estimates are strictly capped at 'preliminary' (never 'moderate' or 'strong')
    const rawConcordance: "strong" | "moderate" | "preliminary" =
      confidence >= 0.75 || verifiedStrongCount >= 2 ? "strong" : confidence >= 0.45 ? "moderate" : "preliminary";
    const concordanceLevel: "strong" | "moderate" | "preliminary" = isConfirmed ? rawConcordance : "preliminary";
    const concordanceLabelEn =
      concordanceLevel === "strong"
        ? "Strong Concordance"
        : concordanceLevel === "moderate"
          ? "Moderate Concordance"
          : "Preliminary Finding";
    const concordanceLabelAr =
      concordanceLevel === "strong"
        ? "توافق تشخيصي قوي"
        : concordanceLevel === "moderate"
          ? "توافق تشخيصي متوسط"
          : "مؤشر تشخيصي مبدئي";

    fused.push({
      code: displayLeader.code,
      urgency: urgencyLeader.urgency,
      priority: tier,
      urgency_label_en: urgencyLeader.urgency_label_en,
      urgency_label_ar: urgencyLeader.urgency_label_ar,
      name_en: cleanNameEn,
      name_ar: cleanNameAr,
      evidence: Array.from(evidenceSet),
      explanation_en: explanationLeader.explanation_en,
      explanation_ar: explanationLeader.explanation_ar,
      immediate_action_en: explanationLeader.immediate_action_en,
      immediate_action_ar: explanationLeader.immediate_action_ar,
      confidence: concordanceLevel === "strong" ? "high" : concordanceLevel === "moderate" ? "moderate" : "low",
      source: explanationLeader.source,
      canonicalParent: parent,
      fusionConfidence: Number(confidence.toFixed(4)),
      rankScore: Number(rankScore.toFixed(4)),
      tier,
      nStrongSources: nStrong,
      isDemoted,
      isConfirmed,
      certaintyTier,
      concordanceLevel,
      concordanceLabelEn,
      concordanceLabelAr,
      sourceBreakdown: contributions,
    });
  }

  // Strictly sort by (confirmed first, tier ASC, rankScore DESC, urgency severity DESC)
  fused.sort((a, b) => {
    if (a.isConfirmed !== b.isConfirmed) return a.isConfirmed ? -1 : 1;
    if (a.tier !== b.tier) return a.tier - b.tier;
    if (Math.abs(b.rankScore - a.rankScore) > 1e-6) return b.rankScore - a.rankScore;
    return URGENCY_SEVERITY[b.urgency] - URGENCY_SEVERITY[a.urgency];
  });

  return fused;
}

/**
 * Strict Certainty Filter (Zero Unverified Pollution):
 * - Returns ALL co-existing CONFIRMED diagnoses (up to 4) when >=1 diagnosis meets certainty criteria.
 * - Excludes uncertain pixel estimates ('possible' / unverified) from the confirmed diagnosis list.
 * - If 0 diagnoses meet certainty criteria, emits an honest INSUFFICIENT_DATA banner rather than
 *   promoting uncertain guesses, while preserving suppressed signals separately.
 */
export function filterConfirmedDiagnoses(fused: FusedECGFinding[]): {
  primaryFindings: FusedECGFinding[];
  unverifiedSignals: FusedECGFinding[];
} {
  const demoted = fused.filter((f) => f.isDemoted);
  const confirmed = fused.filter((f) => !f.isDemoted && f.isConfirmed && f.code !== "INSUFFICIENT_DATA");
  const unverified = fused.filter((f) => !f.isDemoted && !f.isConfirmed && f.code !== "INSUFFICIENT_DATA");

  if (confirmed.length > 0) {
    return {
      primaryFindings: [...confirmed.slice(0, 4), ...demoted],
      unverifiedSignals: unverified,
    };
  }

  const insufficient: FusedECGFinding = {
    code: "INSUFFICIENT_DATA",
    urgency: "ABNORMAL",
    priority: 2,
    urgency_label_en: "VERIFY MANUALLY",
    urgency_label_ar: "يتطلب مراجعة يدوية",
    name_en: "Insufficient Certainty — Manual ECG Verification Required",
    name_ar: "عدم كفاية التيقن التشخيصي — يتطلب مراجعة التخطيط يدوياً",
    evidence: [],
    explanation_en:
      "No diagnosis met strict clinical certainty criteria (visual AI was below the certainty threshold and automatic pixel estimates carry wide measurement uncertainty). Review the 12-lead tracing directly or enter confirmed header intervals/vitals to unlock verified diagnoses.",
    explanation_ar:
      "لم يحقق أي تشخيص معايير التيقن السريري الصارمة (الذكاء البصري أقل من عتبة التيقن، والتقديرات البكسلية الآلية وحدها غير كافية للتأكيد). راجع التخطيط الكامل 12-lead يدوياً أو أدخل قياسات الجهاز/العلامات الحيوية المؤكدة.",
    immediate_action_en: "Inspect rhythm, rate, axis, intervals, and ST-T segments across all 12 leads directly.",
    immediate_action_ar: "افحص النظم والسرعة والمحور والفترات وقطعة ST-T في جميع الـ 12 مسرى مباشرة.",
    confidence: "low",
    source: "rules",
    canonicalParent: "INSUFFICIENT_DATA",
    fusionConfidence: 0,
    rankScore: 0,
    tier: 2,
    nStrongSources: 0,
    isDemoted: false,
    isConfirmed: false,
    certaintyTier: "unverified",
    concordanceLevel: "preliminary",
    concordanceLabelEn: "Preliminary Finding",
    concordanceLabelAr: "مؤشر تشخيصي مبدئي",
    sourceBreakdown: [],
  };
  return {
    primaryFindings: [insufficient, ...demoted],
    unverifiedSignals: unverified,
  };
}
