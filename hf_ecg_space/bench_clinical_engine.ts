/**
 * Code Blue ECG — Full 68-Scenario Clinical Engine & Certainty Gate Benchmark
 * Run inside F:\codeblue-sounds via: npx tsx hf_ecg_space/bench_clinical_engine.ts
 *
 * Evaluates the ACTUAL end-to-end clinical product (`runHybridEcgDiagnosis`):
 *  1. 68 distinct clinical scenarios spanning all 19 AI classes, vitals/hemodynamic rules,
 *     clinical-context pitfall overrides, and composite multi-diagnosis syndromes.
 *  2. False Positive Rate in 4-concurrent-diagnosis multi-pathology mode (Wilson 95% CI).
 *  3. Clinical look-alike safety traps (Pericarditis vs STEMI, Athlete BER, Slow/Narrow AFib+WPW,
 *     Low-confidence AI Certainty Gate rejection, Missing-QT PVC Bigeminy safety).
 *  4. Exports exact Certainty Gate formula & constants from FUSION_CONSTANTS.
 */
import { INITIAL_WIZARD_FEATURES, runHybridEcgDiagnosis, type WizardFeatures } from "../lib/ecgHybridEngine";
import { FUSION_CONSTANTS } from "../lib/ecgFusionEngine";

interface ClinicalScenario {
  id: string;
  group: "core_68" | "multi_4slot" | "safety_trap";
  expectedAny: string[]; // at least one must appear in active (non-demoted) findings
  expectedAll?: string[]; // for multi-diagnosis: all must appear
  forbidden?: string[];
  input: Partial<WizardFeatures>;
}

const mk = (p: Partial<WizardFeatures>): WizardFeatures => ({ ...INITIAL_WIZARD_FEATURES, ...p });
const ai = (code: string, prob = 0.88, uncertainty = 0.05) => ({ label: code, code, prob, uncertainty });

const SCENARIOS: ClinicalScenario[] = [
  // ── A. 19 Core Visual AI + Rule Classes (S01–S19) ──
  { id: "S01_NSR", group: "core_68", expectedAny: ["NORMAL_SINUS_RHYTHM"], forbidden: ["ANTERIOR_SEPTAL_MI", "ATRIAL_FIBRILLATION"], input: { hr_bpm: 74, pr_ms: 160, qrs_ms: 88, qt_ms: 390, rhythm_class: "regular", p_wave_status: "present", ai_predictions: [ai("NORMAL_SINUS_RHYTHM", 0.94, 0.04)] } },
  { id: "S02_ANT_STEMI", group: "core_68", expectedAny: ["ANTERIOR_SEPTAL_MI"], input: { hr_bpm: 96, st_elevation_leads: ["V1", "V2", "V3", "V4"], chest_pain_or_shock: true, ai_predictions: [ai("ANTERIOR_SEPTAL_MI", 0.91)] } },
  { id: "S03_INF_STEMI", group: "core_68", expectedAny: ["INFERIOR_POSTERIOR_MI"], input: { hr_bpm: 64, st_elevation_leads: ["II", "III", "aVF"], chest_pain_or_shock: true, ai_predictions: [ai("INFERIOR_POSTERIOR_MI", 0.89)] } },
  { id: "S04_LAT_STEMI", group: "core_68", expectedAny: ["LATERAL_MI"], input: { hr_bpm: 88, st_elevation_leads: ["I", "aVL", "V5", "V6"], chest_pain_or_shock: true, ai_predictions: [ai("LATERAL_MI", 0.86)] } },
  { id: "S05_NSTEMI", group: "core_68", expectedAny: ["NSTEMI_ISCHEMIA"], input: { hr_bpm: 92, st_depression_leads: ["V4", "V5", "V6"], chest_pain_or_shock: true, ai_predictions: [ai("NSTEMI_ISCHEMIA", 0.84)] } },
  { id: "S06_AFIB", group: "core_68", expectedAny: ["ATRIAL_FIBRILLATION"], input: { hr_bpm: 122, rhythm_class: "irregularly_irregular", p_wave_status: "absent", fibrillatory_waves: true, ai_predictions: [ai("ATRIAL_FIBRILLATION", 0.93)] } },
  { id: "S07_AFLUTTER", group: "core_68", expectedAny: ["ATRIAL_FLUTTER"], input: { hr_bpm: 150, rhythm_class: "regular", sawtooth_flutter: true, ai_predictions: [ai("ATRIAL_FLUTTER", 0.88)] } },
  { id: "S08_SVT", group: "core_68", expectedAny: ["SVT_TACHYCARDIA"], input: { hr_bpm: 176, rhythm_class: "regular", qrs_ms: 86, qrs_width_class: "narrow", ai_predictions: [ai("SVT_TACHYCARDIA", 0.87)] } },
  { id: "S09_SBRAD", group: "core_68", expectedAny: ["SINUS_BRADYCARDIA"], input: { hr_bpm: 48, pr_ms: 168, qrs_ms: 90, rhythm_class: "regular", p_wave_status: "present", ai_predictions: [ai("SINUS_BRADYCARDIA", 0.90)] } },
  { id: "S10_1AVB", group: "core_68", expectedAny: ["FIRST_DEGREE_AV_BLOCK"], input: { hr_bpm: 66, pr_ms: 248, qrs_ms: 92, rhythm_class: "regular", p_wave_status: "present", pr_class: "prolonged", ai_predictions: [ai("FIRST_DEGREE_AV_BLOCK", 0.85)] } },
  { id: "S11_CHB", group: "core_68", expectedAny: ["HIGH_GRADE_AV_BLOCK"], input: { hr_bpm: 36, av_dissociation: true, syncope_or_family_scd: true, ai_predictions: [ai("HIGH_GRADE_AV_BLOCK", 0.92)] } },
  { id: "S12_RBBB", group: "core_68", expectedAny: ["RBBB"], input: { hr_bpm: 78, qrs_ms: 138, qrs_width_class: "wide", qrs_morphology: "rbbb", ai_predictions: [ai("RBBB", 0.91)] } },
  { id: "S13_LBBB", group: "core_68", expectedAny: ["LBBB"], input: { hr_bpm: 76, qrs_ms: 146, qrs_width_class: "wide", qrs_morphology: "lbbb", ai_predictions: [ai("LBBB", 0.93)] } },
  { id: "S14_LAFB", group: "core_68", expectedAny: ["FASCICULAR_BLOCK", "LAFB_LPFB"], input: { hr_bpm: 72, axis_deg: -58, qrs_ms: 102, ai_predictions: [ai("LAFB_LPFB", 0.86)] } },
  { id: "S15_LVH", group: "core_68", expectedAny: ["LVH_STRAIN", "LVH_RVH"], input: { hr_bpm: 74, lvh_voltage: true, ai_predictions: [ai("LVH_RVH", 0.88)] } },
  { id: "S16_PACE", group: "core_68", expectedAny: ["PACEMAKER_RHYTHM"], input: { hr_bpm: 70, pacing_spikes: true, has_pacemaker: true, qrs_ms: 152, ai_predictions: [ai("PACEMAKER_RHYTHM", 0.94)] } },
  { id: "S17_WPW", group: "core_68", expectedAny: ["WPW_PATTERN", "WPW_PREEXCITATION"], input: { hr_bpm: 78, pr_ms: 104, qrs_ms: 126, delta_wave: true, pr_class: "short", ai_predictions: [ai("WPW_PREEXCITATION", 0.87)] } },
  { id: "S18_LONG_QT", group: "core_68", expectedAny: ["LONG_QT", "LONG_QT_OR_ECTOPY"], input: { hr_bpm: 64, qt_ms: 520, prolonged_qt_visual: true, ai_predictions: [ai("LONG_QT_OR_ECTOPY", 0.86)] } },
  { id: "S19_BRUGADA", group: "core_68", expectedAny: ["BRUGADA_TYPE_1"], input: { hr_bpm: 76, brugada_pattern: "type_1", syncope_or_family_scd: true, ai_predictions: [ai("BRUGADA_TYPE_1", 0.84)] } },

  // ── B. Classic Textbook & Rhythm/Ischemia Rules (S20–S33) ──
  { id: "S20_ACUTE_PERICARDITIS", group: "core_68", expectedAny: ["ACUTE_PERICARDITIS"], input: { hr_bpm: 102, st_elevation_leads: ["I", "II", "V2", "V3", "V4", "V5", "V6"], st_morphology_concave: true, pr_depression: true, pleuritic_or_dehydration_or_bleed: true } },
  { id: "S21_WELLENS", group: "core_68", expectedAny: ["WELLENS_SYNDROME"], input: { hr_bpm: 72, t_wave_inversion_leads: ["V2", "V3", "V4"], chest_pain_or_shock: true } },
  { id: "S22_DE_WINTER", group: "core_68", expectedAny: ["DE_WINTER_T_WAVES", "ANTERIOR_SEPTAL_MI"], input: { hr_bpm: 94, st_depression_leads: ["V1", "V2", "V3", "V4"], peaked_t_waves: true, chest_pain_or_shock: true } },
  { id: "S23_HYPERKALEMIA", group: "core_68", expectedAny: ["HYPERKALEMIA", "HYPERKALEMIA_SEVERE"], input: { hr_bpm: 58, qrs_ms: 142, peaked_t_waves: true, dka_ckd_or_high_k: true } },
  { id: "S24_SHORT_QT", group: "core_68", expectedAny: ["SHORT_QT"], input: { hr_bpm: 72, qt_ms: 300, syncope_or_family_scd: true } },
  { id: "S25_VT", group: "core_68", expectedAny: ["VENTRICULAR_TACHYCARDIA"], input: { hr_bpm: 172, qrs_ms: 162, qrs_width_class: "wide", rhythm_class: "regular", av_dissociation: true } },
  { id: "S26_TORSADES", group: "core_68", expectedAny: ["TORSADES_DE_POINTES"], input: { hr_bpm: 210, qt_ms: 540, polymorphic_vt: true, qrs_width_class: "wide" } },
  { id: "S27_LEAD_REVERSAL", group: "core_68", expectedAny: ["DEXTROCARDIA_LEAD_REVERSAL"], input: { hr_bpm: 74, lead_reversal_alert: true } },
  { id: "S28_BIFASCICULAR", group: "core_68", expectedAny: ["BIFASCICULAR_BLOCK"], input: { hr_bpm: 70, qrs_ms: 140, axis_deg: -62, ai_predictions: [ai("RBBB", 0.90), ai("LAFB_LPFB", 0.85)] } },
  { id: "S29_TRIFASCICULAR", group: "core_68", expectedAny: ["TRIFASCICULAR_BLOCK_PATTERN"], input: { hr_bpm: 66, pr_ms: 242, qrs_ms: 144, axis_deg: -58, ai_predictions: [ai("RBBB", 0.91), ai("LAFB_LPFB", 0.86), ai("FIRST_DEGREE_AV_BLOCK", 0.84)] } },
  { id: "S30_PRE_EXCITED_AFIB", group: "core_68", expectedAny: ["PRE_EXCITED_AFIB"], input: { hr_bpm: 186, qrs_ms: 148, rhythm_class: "irregularly_irregular", delta_wave: true, ai_predictions: [ai("ATRIAL_FIBRILLATION", 0.89), ai("WPW_PREEXCITATION", 0.82)] } },
  { id: "S31_PE_S1Q3T3", group: "core_68", expectedAny: ["PULMONARY_EMBOLISM_S1Q3T3", "PE_SUSPECTED_HYPOXIA"], input: { hr_bpm: 118, spo2_status: "low", pleuritic_or_dehydration_or_bleed: true, ai_predictions: [ai("RBBB", 0.80)] } },
  { id: "S32_PVC_ECTOPY_WITH_QT", group: "core_68", expectedAny: ["PVC_BIGEMINY", "LONG_QT_OR_ECTOPY"], input: { hr_bpm: 78, qt_ms: 380, ai_predictions: [ai("LONG_QT_OR_ECTOPY", 0.82)] } },
  { id: "S33_UNCLASS_QT_NO_QT", group: "core_68", expectedAny: ["UNCLASSIFIED_QT_ECTOPY", "LONG_QT_OR_ECTOPY"], input: { hr_bpm: 78, qt_ms: null, ai_predictions: [ai("LONG_QT_OR_ECTOPY", 0.82)] } },

  // ── C. Context & Clinical Pitfall Override Rules (S34–S41) ──
  { id: "S34_EARLY_REPOL_ATHLETE", group: "core_68", expectedAny: ["EARLY_REPOLARIZATION"], input: { hr_bpm: 52, st_elevation_leads: ["V2", "V3", "V4"], early_repolarization: true, is_competitive_athlete: true, asymptomatic_nocturnal: true } },
  { id: "S35_VAGAL_AVB_ATHLETE", group: "core_68", expectedAny: ["VAGAL_AV_BLOCK_BENIGN"], input: { hr_bpm: 44, pr_ms: 240, is_competitive_athlete: true, asymptomatic_nocturnal: true, ai_predictions: [ai("FIRST_DEGREE_AV_BLOCK", 0.85)] } },
  { id: "S36_TAKOTSUBO", group: "core_68", expectedAny: ["TAKOTSUBO_CARDIOMYOPATHY"], input: { hr_bpm: 98, st_elevation_leads: ["V2", "V3", "V4"], post_emotional_stress: true, chest_pain_or_shock: true } },
  { id: "S37_HCM_ATHLETE", group: "core_68", expectedAny: ["HCM_PATTERN"], input: { hr_bpm: 64, lvh_voltage: true, t_wave_inversion_leads: ["V3", "V4", "V5", "V6"], is_competitive_athlete: true, syncope_or_family_scd: true } },
  { id: "S38_ARTIFACT_TREMOR", group: "core_68", expectedAny: ["ARTIFACT_LEAD_FAULT"], input: { hr_bpm: 165, tremor_or_nebulizer_or_loose_lead: true, ai_predictions: [ai("ATRIAL_FLUTTER", 0.72)] } },
  { id: "S39_PACEMAKER_DEMAND", group: "core_68", expectedAny: ["PACEMAKER_DEMAND_INHIBITION", "PACEMAKER_RHYTHM"], input: { hr_bpm: 78, has_pacemaker: true, ai_predictions: [ai("NORMAL_SINUS_RHYTHM", 0.85)] } },
  { id: "S40_DRUG_BRUGADA", group: "core_68", expectedAny: ["BRUGADA_PATTERN_DRUG_INDUCED", "BRUGADA_TYPE_1"], input: { hr_bpm: 92, brugada_pattern: "type_1", fever_or_cocaine_or_sodium_blocker: true } },
  { id: "S41_HYPERK_SEVERE", group: "core_68", expectedAny: ["HYPERKALEMIA_SEVERE", "HYPERKALEMIA"], input: { hr_bpm: 48, qrs_ms: 168, qrs_morphology: "bizarre_wide", peaked_t_waves: true, dka_ckd_or_high_k: true } },

  // ── D. Vitals & Hemodynamic Rules (S42–S68) ──
  { id: "S42_UNSTABLE_TACHY", group: "core_68", expectedAny: ["UNSTABLE_TACHYARRHYTHMIA"], input: { hr_bpm: 168, hemodynamic_unstable: true, bp_status: "low", ai_predictions: [ai("SVT_TACHYCARDIA", 0.85)] } },
  { id: "S43_UNSTABLE_BRADY", group: "core_68", expectedAny: ["UNSTABLE_BRADYARRHYTHMIA"], input: { hr_bpm: 36, hemodynamic_unstable: true, bp_status: "low", ai_predictions: [ai("HIGH_GRADE_AV_BLOCK", 0.90)] } },
  { id: "S44_SHOCK_RV_MI", group: "core_68", expectedAny: ["CARDIOGENIC_SHOCK_OR_RV_MI"], input: { hr_bpm: 56, bp_status: "low", hemodynamic_unstable: true, st_elevation_leads: ["II", "III", "aVF"], ai_predictions: [ai("INFERIOR_POSTERIOR_MI", 0.89)] } },
  { id: "S45_SHOCK_LV_ISCHEMIA", group: "core_68", expectedAny: ["CARDIOGENIC_SHOCK_LV_ISCHEMIA"], input: { hr_bpm: 112, bp_status: "low", hemodynamic_unstable: true, st_elevation_leads: ["V1", "V2", "V3", "V4"], ai_predictions: [ai("ANTERIOR_SEPTAL_MI", 0.90)] } },
  { id: "S46_SHOCK_TAMPONADE", group: "core_68", expectedAny: ["OBSTRUCTIVE_SHOCK_TAMPONADE"], input: { hr_bpm: 124, bp_status: "low", hemodynamic_unstable: true, low_voltage: true } },
  { id: "S47_MASSIVE_PE_SHOCK", group: "core_68", expectedAny: ["MASSIVE_PE_OR_OBSTRUCTIVE_SHOCK"], input: { hr_bpm: 132, bp_status: "low", spo2_status: "low", hemodynamic_unstable: true, ai_predictions: [ai("RBBB", 0.82)] } },
  { id: "S48_SEPTIC_SHOCK_STRESS", group: "core_68", expectedAny: ["SEPTIC_SHOCK_CARDIAC_STRESS"], input: { hr_bpm: 128, bp_status: "low", temp_status: "fever", hemodynamic_unstable: true } },
  { id: "S49_DKA_PSEUDO_STEMI", group: "core_68", expectedAny: ["DKA_HYPERKALEMIA_PSEUDO_STEMI"], input: { hr_bpm: 108, glucose_status: "high", dka_ckd_or_high_k: true, peaked_t_waves: true, st_elevation_leads: ["V1", "V2"] } },
  { id: "S50_DKA_TACHY", group: "core_68", expectedAny: ["DKA_SECONDARY_TACHYARRHYTHMIA", "SEVERE_DKA_DECOMPENSATION"], input: { hr_bpm: 132, glucose_status: "high", ai_predictions: [ai("ATRIAL_FIBRILLATION", 0.85)] } },
  { id: "S51_SEVERE_DKA", group: "core_68", expectedAny: ["SEVERE_DKA_DECOMPENSATION"], input: { hr_bpm: 126, glucose_status: "high", bp_status: "low" } },
  { id: "S52_FEVER_BRUGADA", group: "core_68", expectedAny: ["FEVER_INDUCED_BRUGADA"], input: { hr_bpm: 108, temp_status: "fever", brugada_pattern: "type_1", ai_predictions: [ai("BRUGADA_TYPE_1", 0.82)] } },
  { id: "S53_FEVER_ENDOCARDITIS", group: "core_68", expectedAny: ["FEVER_NEW_AV_BLOCK_ENDOCARDITIS"], input: { hr_bpm: 64, pr_ms: 256, temp_status: "fever", ai_predictions: [ai("FIRST_DEGREE_AV_BLOCK", 0.84)] } },
  { id: "S54_INFECTIOUS_PERI", group: "core_68", expectedAny: ["INFECTIOUS_PERICARDITIS_MYOCARDITIS", "ACUTE_PERICARDITIS"], input: { hr_bpm: 112, temp_status: "fever", st_elevation_leads: ["I", "II", "V2", "V3", "V4", "V5"], st_morphology_concave: true, pr_depression: true } },
  { id: "S55_FEVER_AFIB", group: "core_68", expectedAny: ["SECONDARY_AFIB_OR_TACHY_FEVER"], input: { hr_bpm: 136, temp_status: "fever", ai_predictions: [ai("ATRIAL_FIBRILLATION", 0.90)] } },
  { id: "S56_HYPOTHERMIA", group: "core_68", expectedAny: ["HYPOTHERMIA_ECG_PATTERN"], input: { hr_bpm: 44, temp_status: "hypothermia", qt_ms: 510 } },
  { id: "S57_HTN_ACS_DISSECT", group: "core_68", expectedAny: ["HYPERTENSIVE_ACS_OR_DISSECTION"], input: { hr_bpm: 104, bp_status: "high", chest_pain_or_shock: true, st_depression_leads: ["V4", "V5", "V6"], ai_predictions: [ai("NSTEMI_ISCHEMIA", 0.84)] } },
  { id: "S58_HTN_LVH_STRAIN", group: "core_68", expectedAny: ["HYPERTENSIVE_LVH_STRAIN"], input: { hr_bpm: 78, bp_status: "high", lvh_voltage: true, ai_predictions: [ai("LVH_RVH", 0.88)] } },
  { id: "S59_CUSHING_REFLEX", group: "core_68", expectedAny: ["CUSHING_REFLEX_OR_DRUG_EFFECT"], input: { hr_bpm: 44, bp_status: "high", ai_predictions: [ai("SINUS_BRADYCARDIA", 0.88)] } },
  { id: "S60_ACUTE_PULM_EDEMA", group: "core_68", expectedAny: ["ACUTE_PULMONARY_EDEMA_OR_ACS"], input: { hr_bpm: 118, spo2_status: "low", bp_status: "high", st_depression_leads: ["V3", "V4", "V5"], ai_predictions: [ai("NSTEMI_ISCHEMIA", 0.85)] } },
  { id: "S61_PE_HYPOXIA", group: "core_68", expectedAny: ["PE_SUSPECTED_HYPOXIA", "PULMONARY_EMBOLISM_S1Q3T3"], input: { hr_bpm: 122, spo2_status: "low", pleuritic_or_dehydration_or_bleed: true } },
  { id: "S62_SECONDARY_AFIB_RVR", group: "core_68", expectedAny: ["SECONDARY_AFIB_RVR"], input: { hr_bpm: 142, pleuritic_or_dehydration_or_bleed: true, ai_predictions: [ai("ATRIAL_FIBRILLATION", 0.90)] } },
  { id: "S63_TCA_SODIUM", group: "core_68", expectedAny: ["TCA_SODIUM_BLOCKADE"], input: { hr_bpm: 128, qrs_ms: 148, fever_or_cocaine_or_sodium_blocker: true } },
  { id: "S64_FEVER_STANDALONE", group: "core_68", expectedAny: ["FEVER_STANDALONE"], input: { hr_bpm: 96, temp_status: "fever" } },
  { id: "S65_HYPOXIA_STANDALONE", group: "core_68", expectedAny: ["HYPOXIA_STANDALONE"], input: { hr_bpm: 94, spo2_status: "low" } },
  { id: "S66_HTN_STANDALONE", group: "core_68", expectedAny: ["HYPERTENSION_STANDALONE"], input: { hr_bpm: 78, bp_status: "high" } },
  { id: "S67_HYPERGLYCEMIA_STD", group: "core_68", expectedAny: ["HYPERGLYCEMIA_STANDALONE"], input: { hr_bpm: 86, glucose_status: "high" } },
  { id: "S68_SHOCK_STANDALONE", group: "core_68", expectedAny: ["SHOCK_OR_HYPOTENSION_STANDALONE", "CARDIOGENIC_SHOCK_UNSPECIFIED"], input: { hr_bpm: 110, bp_status: "low", hemodynamic_unstable: true } },

  // ── E. 4-Concurrent-Diagnosis Multi-Pathology Stress (M01–M04) ──
  { id: "M01_RBBB_LAFB_1AVB", group: "multi_4slot", expectedAny: ["RBBB"], expectedAll: ["RBBB", "FIRST_DEGREE_AV_BLOCK"], forbidden: ["ATRIAL_FIBRILLATION", "ANTERIOR_SEPTAL_MI"], input: { hr_bpm: 68, pr_ms: 236, qrs_ms: 142, axis_deg: -56, ai_predictions: [ai("RBBB", 0.91), ai("LAFB_LPFB", 0.86), ai("FIRST_DEGREE_AV_BLOCK", 0.83)] } },
  { id: "M02_AFIB_LBBB_LVH", group: "multi_4slot", expectedAny: ["ATRIAL_FIBRILLATION"], expectedAll: ["ATRIAL_FIBRILLATION", "LBBB"], forbidden: ["WPW_PATTERN", "BRUGADA_TYPE_1"], input: { hr_bpm: 112, qrs_ms: 148, rhythm_class: "irregularly_irregular", p_wave_status: "absent", lvh_voltage: true, ai_predictions: [ai("ATRIAL_FIBRILLATION", 0.92), ai("LBBB", 0.90), ai("LVH_RVH", 0.81)] } },
  { id: "M03_INF_MI_CHB_RBBB", group: "multi_4slot", expectedAny: ["INFERIOR_POSTERIOR_MI"], expectedAll: ["INFERIOR_POSTERIOR_MI", "HIGH_GRADE_AV_BLOCK", "RBBB"], forbidden: ["NORMAL_SINUS_RHYTHM", "SVT_TACHYCARDIA"], input: { hr_bpm: 40, qrs_ms: 136, st_elevation_leads: ["II", "III", "aVF"], av_dissociation: true, chest_pain_or_shock: true, ai_predictions: [ai("INFERIOR_POSTERIOR_MI", 0.89), ai("HIGH_GRADE_AV_BLOCK", 0.88), ai("RBBB", 0.82)] } },
  { id: "M04_AFIB_RBBB_1AVB_LVH", group: "multi_4slot", expectedAny: ["RBBB"], expectedAll: ["ATRIAL_FIBRILLATION", "RBBB"], forbidden: ["NORMAL_SINUS_RHYTHM", "BRUGADA_TYPE_1"], input: { hr_bpm: 108, qrs_ms: 138, lvh_voltage: true, rhythm_class: "irregularly_irregular", ai_predictions: [ai("ATRIAL_FIBRILLATION", 0.90), ai("RBBB", 0.88), ai("LVH_RVH", 0.80)] } },

  // ── F. Safety & Certainty Gate Traps (T01–T04) ──
  { id: "T01_LOW_CONF_REJECT", group: "safety_trap", expectedAny: ["INSUFFICIENT_DATA"], forbidden: ["ANTERIOR_SEPTAL_MI", "HIGH_GRADE_AV_BLOCK", "BRUGADA_TYPE_1"], input: { ai_predictions: [ai("ANTERIOR_SEPTAL_MI", 0.34, 0.45)] } },
  { id: "T02_SLOW_NARROW_AFIB_WPW", group: "safety_trap", expectedAny: ["ATRIAL_FIBRILLATION"], input: { hr_bpm: 84, qrs_ms: 92, rhythm_class: "irregularly_irregular", ai_predictions: [ai("ATRIAL_FIBRILLATION", 0.86), ai("WPW_PREEXCITATION", 0.65, 0.12)] } },
  { id: "T03_NO_QT_NO_BIGEMINY", group: "safety_trap", expectedAny: ["UNCLASSIFIED_QT_ECTOPY", "LONG_QT_OR_ECTOPY"], forbidden: ["PVC_BIGEMINY"], input: { hr_bpm: 76, qt_ms: null, ai_predictions: [ai("LONG_QT_OR_ECTOPY", 0.80)] } },
  { id: "T04_CLEAN_NSR_NO_ALARM", group: "safety_trap", expectedAny: ["NORMAL_SINUS_RHYTHM"], forbidden: ["ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "BRUGADA_TYPE_1", "HIGH_GRADE_AV_BLOCK"], input: { hr_bpm: 72, pr_ms: 158, qrs_ms: 86, qt_ms: 392, rhythm_class: "regular", p_wave_status: "present", ai_predictions: [ai("NORMAL_SINUS_RHYTHM", 0.96, 0.03)] } },
];

function wilson(k: number, n: number): [number, number] {
  if (n === 0) return [0, 0];
  const z = 1.95996, p = k / n, d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)) / d;
  return [Number((Math.max(0, c - h) * 100).toFixed(1)), Number((Math.min(1, c + h) * 100).toFixed(1))];
}

let c68Hits = 0, c68Tot = 0, multiFp = 0, multiSlots = 0, trapPass = 0, trapTot = 0;
const failures: string[] = [];

for (const sc of SCENARIOS) {
  const res = runHybridEcgDiagnosis(mk(sc.input));
  const active = (res.primaryFindings || []).filter((f) => !f.isDemoted).map((f) => f.code);
  const anyHit = sc.expectedAny.some((c) => active.includes(c));
  const allHit = (sc.expectedAll || []).every((c) => active.includes(c));
  const forbHit = (sc.forbidden || []).filter((c) => active.includes(c));

  if (sc.group === "core_68") {
    c68Tot++;
    if (anyHit && forbHit.length === 0) c68Hits++;
    else failures.push(`${sc.id}: active=[${active.join(",")}] expectedAny=[${sc.expectedAny.join("|")}] forb=[${forbHit.join(",")}]`);
  } else if (sc.group === "multi_4slot") {
    const allowed = new Set([...sc.expectedAny, ...(sc.expectedAll || []), "BIFASCICULAR_BLOCK", "TRIFASCICULAR_BLOCK_PATTERN", "LVH_STRAIN", "LVH_RVH", "FASCICULAR_BLOCK", "LAFB_LPFB"]);
    const extraFp = active.filter((c) => !allowed.has(c)).length + forbHit.length;
    multiFp += extraFp;
    multiSlots += 4;
    if (!allHit || forbHit.length > 0) failures.push(`${sc.id}: active=[${active.join(",")}] expectedAll=[${(sc.expectedAll || []).join(",")}]`);
  } else {
    trapTot++;
    if (anyHit && forbHit.length === 0) trapPass++;
    else failures.push(`${sc.id}: active=[${active.join(",")}] expectedAny=[${sc.expectedAny.join("|")}] forb=[${forbHit.join(",")}]`);
  }
}

console.log(JSON.stringify({
  certainty_gate_constants: FUSION_CONSTANTS,
  core_68_scenarios: { hits: c68Hits, total: c68Tot, recall_pct: Number(((100 * c68Hits) / Math.max(1, c68Tot)).toFixed(1)), wilson_ci95: wilson(c68Hits, c68Tot) },
  multi_diagnosis_4slot_fpr: { fp_slots: multiFp, total_slots: multiSlots, fpr_pct: Number(((100 * multiFp) / Math.max(1, multiSlots)).toFixed(1)), wilson_ci95: wilson(multiFp, multiSlots) },
  safety_traps: { passed: trapPass, total: trapTot, pct: Number(((100 * trapPass) / Math.max(1, trapTot)).toFixed(1)), wilson_ci95: wilson(trapPass, trapTot) },
  failures,
}, null, 2));
