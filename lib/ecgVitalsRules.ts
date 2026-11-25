import { UrgencyLevel } from "@/lib/ecgRules";
import type { ContextAdjustment, WizardFeatures } from "@/lib/ecgHybridEngine";

/**
 * Exhaustive Vitals x ECG Clinical Matrix (Zero Dropped Combinations).
 * Covers:
 *  - Every individual abnormal vital standalone (1 vital alone)
 *  - Every abnormal vital combined with specific ECG rhythms/conduction/ischemia patterns
 *  - Multi-vital critical emergencies (Septic Shock, Massive PE, Severe DKA, Cushing's Reflex, Cardiogenic Shock)
 *  - EM Cases / CAEP AFIB secondary-trigger rules
 */
export function evaluateVitalsContext(
  f: WizardFeatures,
  ruleCodes: Set<string>,
  aiCodes: Set<string>
): ContextAdjustment[] {
  const out: ContextAdjustment[] = [];
  const hasCode = (...codes: string[]) => codes.some((c) => ruleCodes.has(c) || aiCodes.has(c));

  const hasAfibOrFlutter = hasCode("ATRIAL_FIBRILLATION", "ATRIAL_FIBRILLATION_RVR", "ATRIAL_FLUTTER");
  const hasTachy =
    hasAfibOrFlutter ||
    hasCode("SVT_NARROW", "VENTRICULAR_TACHYCARDIA", "TACHYCARDIA_UNSPECIFIED", "MAT", "PREEXCITED_AF") ||
    (f.hr_bpm != null && f.hr_bpm > 100);
  const hasBrady =
    hasCode("BRADYCARDIA", "MARKED_BRADYCARDIA", "COMPLETE_HEART_BLOCK", "HIGH_GRADE_OR_2_TO_1_AV_BLOCK", "MOBITZ_II_AV_BLOCK", "MOBITZ_I_AV_BLOCK") ||
    (f.hr_bpm != null && f.hr_bpm < 55);
  const hasAvBlock = hasCode(
    "FIRST_DEGREE_AV_BLOCK",
    "MOBITZ_I_AV_BLOCK",
    "MOBITZ_II_AV_BLOCK",
    "HIGH_GRADE_OR_2_TO_1_AV_BLOCK",
    "COMPLETE_HEART_BLOCK"
  );
  const hasIschemia =
    hasCode("STEMI", "POSTERIOR_STEMI", "RV_STEMI", "LMCA_OCCLUSION", "NSTEMI_ISCHEMIA", "WELLENS_SYNDROME", "SGARBOSSA_POSITIVE") ||
    f.st_elevation_leads.length >= 2 ||
    f.st_depression_leads.length >= 2;
  const hasLongQt = hasCode("LONG_QT", "LONG_QT_CRITICAL", "TORSADES_DE_POINTES") || f.prolonged_qt_visual;
  const hasBrugada = hasCode("BRUGADA_TYPE_1", "BRUGADA_TYPE_2") || f.brugada_pattern != null;
  const hasPeakedOrHyperK = hasCode("HYPERKALEMIA", "HYPERKALEMIA_SEVERE") || f.peaked_t_waves || f.qrs_width_class === "wide";
  const hasLvh = hasCode("LVH", "HCM_PATTERN") || f.lvh_voltage;
  const hasRbbbOrStrain = hasCode("RBBB", "PULMONARY_EMBOLISM", "RVH") || f.t_wave_inversion_leads.length >= 2 || f.s1q3t3 || f.rv_strain_twi_v1_v4;
  const hasPericarditisOrLowVolt = hasCode("ACUTE_PERICARDITIS", "CARDIAC_TAMPONADE_SUSPECTED") || f.low_voltage || f.pr_depression;

  const isUnstableOrLowBp = f.hemodynamic_unstable || f.bp_status === "low";
  const isHighBp = f.bp_status === "high";
  const isLowSpo2 = f.spo2_status === "low";
  const isFever = f.temp_status === "fever";
  const isHypothermia = f.temp_status === "hypothermia";
  const isHighGluc = f.glucose_status === "high";
  const isLowGluc = f.glucose_status === "low";

  const add = (adj: ContextAdjustment) => {
    if (!out.some((x) => x.code === adj.code)) out.push(adj);
  };

  // =========================================================================
  // A. MULTI-VITAL CRITICAL EMERGENCIES (2+ Abnormal Vitals Combined)
  // =========================================================================
  if (isFever && isUnstableOrLowBp) {
    add({
      code: "SEPTIC_SHOCK_CARDIAC_STRESS",
      urgency: "CRITICAL",
      title_en: "Suspected Septic Shock (Fever + Hypotension/Instability)",
      title_ar: "اشتباه صدمة إنتانية (حمى + هبوط ضغط / عدم استقرار)",
      reason_en: "Fever with hypotension/instability strongly indicates septic shock driving secondary tachycardia, AFib, or demand ischemia.",
      reason_ar: "اجتماع الحمى مع هبوط الضغط أو عدم الاستقرار يرجّح بقوة صدمة إنتانية تسبب تسارع القلب أو الرجفان الأذيني الثانوي.",
      action_en: "Start immediate sepsis bundle (IV crystalloids, blood cultures, broad-spectrum antibiotics, vasopressors); avoid aggressive rate control.",
      action_ar: "ابدأ بروتوكول الصدمة الإنتانية فوراً (سوائل وريدية، مزارع دم، مضاد حيوي واسع المدى، رافعات ضغط) وتجنب خفض النبض العنيف.",
    });
  }

  if (isLowSpo2 && isUnstableOrLowBp) {
    add({
      code: "MASSIVE_PE_OR_OBSTRUCTIVE_SHOCK",
      urgency: "CRITICAL",
      title_en: "Suspected Massive Pulmonary Embolism / Obstructive Shock (Hypoxia + Hypotension)",
      title_ar: "اشتباه جلطة رئوية جسيمة أو صدمة انسدادية (نقص أكسجين + هبوط ضغط)",
      reason_en: "Simultaneous hypoxemia and hypotension is a red flag for massive PE, tension pneumothorax, or cardiac tamponade.",
      reason_ar: "تزامن نقص الأكسجين مع هبوط الضغط علامة خطر كبرى لجلطة رئوية جسيمة أو استرواح صدري ضاغط أو اندحاس قلبي.",
      action_en: "High-flow O2, immediate bedside POCUS (RV strain/effusion/lung sliding), and urgent systemic thrombolysis or thrombectomy evaluation.",
      action_ar: "أكسجين عالي التدفق + سونار قلب/صدر فوري (POCUS) لتقييم البطين الأيمن والتامور، وتقييم عاجل لمذيب الجلطة.",
    });
  }

  if (isHighGluc && (isUnstableOrLowBp || isLowSpo2)) {
    add({
      code: "SEVERE_DKA_DECOMPENSATION",
      urgency: "CRITICAL",
      title_en: "Severe DKA / Hyperglycemic Crisis with Hemodynamic or Respiratory Compromise",
      title_ar: "غيبوبة سكر كيتونية شديدة (DKA) مع هبوط ضغط أو نقص أكسجين",
      reason_en: "Hyperglycemia/DKA accompanied by hypotension or hypoxia signals severe volume depletion, acidosis, and life-threatening K+ shifts.",
      reason_ar: "ارتفاع السكر/DKA مع هبوط الضغط أو نقص الأكسجين يعني جفافاً حاداً وحماضاً واضطراباً خطيراً بالبوتاسيوم.",
      action_en: "Immediate isotonic IV fluid resuscitation, stat VBG/K+/lactate, and treat hyperkalemia with IV Calcium before insulin if ECG changes exist.",
      action_ar: "إنعاش فوري بالسوائل الوريدية وتحليل غازات وبوتاسيوم فوري، وإعطاء كالسيوم وريدي قبل الإنسولين لو ظهرت تغيرات بوتاسيوم.",
    });
  }

  // =========================================================================
  // B. LOW BP / HEMODYNAMIC INSTABILITY (Every ECG Combo + Standalone)
  // =========================================================================
  if (isUnstableOrLowBp) {
    let matchedLowBpSpecific = false;

    if (hasTachy) {
      matchedLowBpSpecific = true;
      add({
        code: "UNSTABLE_TACHYARRHYTHMIA",
        urgency: "CRITICAL",
        title_en: "Hemodynamically Unstable Tachyarrhythmia (Hypotension / Shock)",
        title_ar: "تسارع قلب غير مستقر دورياً (هبوط ضغط / صدمة)",
        reason_en: "Hypotension/instability with rapid arrhythmia requires immediate synchronized cardioversion if primary (especially pre-excited AF/WPW), while concurrently treating secondary shock causes.",
        reason_ar: "هبوط الضغط أو عدم الاستقرار مع التسارع القلبي يتطلب صدمة كهربائية متزامنة فورية إذا كان السبب أولياً (خاصة AF مع WPW) مع علاج أي سبب ثانوي للصدمة.",
        action_en: "Prepare immediate synchronized cardioversion; administer IV fluids/vasopressors if secondary shock is suspected.",
        action_ar: "جهّز صدمة كهربائية متزامنة فوراً، مع سوائل وريدية ورافعات ضغط إذا كان هناك سبب ثانوي للصدمة.",
      });
    }
    if (hasBrady) {
      matchedLowBpSpecific = true;
      add({
        code: "UNSTABLE_BRADYARRHYTHMIA",
        urgency: "CRITICAL",
        title_en: "Hemodynamically Unstable Bradycardia / AV Block",
        title_ar: "بطء قلب أو إحصار غير مستقر دورياً (مع هبوط ضغط / صدمة)",
        reason_en: "Symptomatic or hypotensive bradycardia/AV block requires immediate chronotropic therapy or pacing.",
        reason_ar: "بطء القلب أو إحصار التوصيل المصحوب بهبوط ضغط أو عدم استقرار يتطلب تدخلاً فورياً لرفع النبض.",
        action_en: "Give IV Atropine 1 mg (if nodal) and prepare immediate transcutaneous pacing (especially for high-grade/CHB) or dopamine/epinephrine infusion.",
        action_ar: "أعطِ أتروبين 1 مجم وريدياً وجهّز منظم ضربات خارجي فوري (Transcutaneous Pacing) خاصة في الإحصار عالي الدرجة/الكامل.",
      });
    }
    if (hasIschemia) {
      matchedLowBpSpecific = true;
      add({
        code: "CARDIOGENIC_SHOCK_OR_RV_MI",
        urgency: "CRITICAL",
        title_en: "Suspected Cardiogenic Shock or Right Ventricular MI",
        title_ar: "اشتباه صدمة قلبية أو جلطة بالبطين الأيمن (RV MI)",
        reason_en: "Hypotension with acute ischemia/STEMI indicates cardiogenic shock or preload-dependent RV infarction.",
        reason_ar: "هبوط الضغط مع الجلطة الحادة قد يعني صدمة قلبية أو جلطة بالبطين الأيمن تعتمد على السوائل.",
        action_en: "Obtain right-sided V4R lead, AVOID nitrates/morphine if RV MI suspected, give cautious fluid bolus if lungs clear, and activate cath lab immediately.",
        action_ar: "اعمل وصلة V4R، تجنب النيترات والمورفين تماماً عند اشتباه RV MI، وفعّل القسطرة الطارئة فوراً.",
      });
    }
    if (hasPericarditisOrLowVolt) {
      matchedLowBpSpecific = true;
      add({
        code: "OBSTRUCTIVE_SHOCK_TAMPONADE",
        urgency: "CRITICAL",
        title_en: "Suspected Cardiac Tamponade (Hypotension + Low Voltage/Pericarditis)",
        title_ar: "اشتباه اندحاس قلبي حاد (هبوط ضغط + انخفاض جهد / التهاب تامور)",
        reason_en: "Low BP with low QRS voltage or pericarditis pattern strongly suggests pericardial effusion with tamponade physiology.",
        reason_ar: "هبوط الضغط مع انخفاض جهد QRS أو التهاب التامور يرجّح بشدة وجود انصباب تاموري ضاغط (Tamponade).",
        action_en: "Perform immediate bedside echo (POCUS) and prepare urgent pericardiocentesis; give IV fluids as a bridge.",
        action_ar: "اعمل إيكو طوارئ فوري (POCUS) وجهّز لبزل التامور العاجل مع سوائل وريدية كجسر مؤقت.",
      });
    }
    if (!matchedLowBpSpecific) {
      add({
        code: "SHOCK_OR_HYPOTENSION_STANDALONE",
        urgency: "CRITICAL",
        title_en: "Hypotension / Hemodynamic Instability — Immediate Shock Evaluation",
        title_ar: "هبوط ضغط / عدم استقرار دوري — تقييم فوري لسبب الصدمة",
        reason_en: "Even without a specific diagnostic ECG pattern yet, hypotension/instability demands immediate shock workup and serial ECGs.",
        reason_ar: "حتى مع عدم اكتمال نمط ECG محدد، هبوط الضغط أو عدم الاستقرار يستوجب تقييم الصدمة فوراً مع إعادة التخطيط دورياً.",
        action_en: "Establish IV access, start resuscitation, perform bedside POCUS (heart/IVC/lungs/FAST), and obtain serial ECGs & troponin/lactate.",
        action_ar: "ركّب خطين وريديين وابدأ الإنعاش مع سونار طوارئ (POCUS) وإعادة تخطيط القلب وتحليل لاكتات وتروبونين.",
      });
    }
  }

  // =========================================================================
  // C. HIGH BLOOD PRESSURE (Every ECG Combo + Standalone)
  // =========================================================================
  if (isHighBp) {
    let matchedHighBp = false;
    if (hasIschemia || f.chest_pain_or_shock) {
      matchedHighBp = true;
      add({
        code: "HYPERTENSIVE_ACS_OR_DISSECTION",
        urgency: "CRITICAL",
        title_en: "Severe Hypertension with Chest Pain/Ischemia — Rule Out Aortic Dissection",
        title_ar: "ارتفاع ضغط مع ألم صدري/إقفار — استبعد تسلخ الشريان الأورطي أولاً",
        reason_en: "Marked hypertension with chest pain or ST changes can reflect ACS or acute aortic dissection extending into a coronary ostium.",
        reason_ar: "ارتفاع الضغط مع ألم الصدر أو تغيرات ST قد يكون جلطة تاجية أو تسلخاً في الشريان الأورطي محاكياً للجلطة.",
        action_en: "Check bilateral arm BPs, pulse deficits, and tearing back pain before anticoagulants/thrombolytics; control BP/HR carefully.",
        action_ar: "قِس الضغط في الذراعين وتأكد من عدم وجود ألم ممزق بالظهر قبل إعطاء مسيلات أو مذيب جلطة.",
      });
    }
    if (hasLvh) {
      matchedHighBp = true;
      add({
        code: "HYPERTENSIVE_LVH_STRAIN",
        urgency: "URGENT",
        title_en: "Hypertensive Heart Disease / LVH Strain Pattern",
        title_ar: "تضخم بطين أيسر مع إجهاد بسبب ارتفاع ضغط الدم (LVH Strain)",
        reason_en: "Chronic or severe hypertension causes LVH voltage and secondary ST-T strain in lateral leads that mimics NSTEMI.",
        reason_ar: "ارتفاع الضغط يسبب تضخم البطين الأيسر وتغيرات ST-T ثانوية في الوصلات الجانبية تحاكي الإقفار الحاد.",
        action_en: "Compare with prior ECGs, check serial troponins if symptomatic, and evaluate for hypertensive end-organ damage.",
        action_ar: "قارن بتخطيط سابق واطلب تروبونين متسلسل عند وجود أعراض، مع تقييم تأثير الضغط على الأعضاء المستهدفة.",
      });
    }
    if (hasBrady) {
      matchedHighBp = true;
      add({
        code: "CUSHING_REFLEX_OR_DRUG_EFFECT",
        urgency: "URGENT",
        title_en: "Hypertension + Bradycardia — Check Cushing's Reflex (High ICP) or Nodal Agents",
        title_ar: "ارتفاع ضغط مع بطء قلب — افحص ارتفاع ضغط المخ (Cushing) أو أدوية الضغط",
        reason_en: "Hypertension paired with bradycardia is a classic warning for raised intracranial pressure (Cushing's triad) or beta-blocker/alpha-agonist effect.",
        reason_ar: "اجتماع ارتفاع الضغط مع بطء القلب علامة تحذير لارتفاع ضغط الجمجمة (منعكس كوشينج) أو تأثير حاصرات بيتا.",
        action_en: "Perform immediate neurological exam (GCS, pupils) and review antihypertensive/nodal-blocking medications.",
        action_ar: "اعمل فحصاً عصبياً فورياً (درجة الوعي وحدقة العين) وراجع أدوية الضغط والقلب.",
      });
    }
    if (!matchedHighBp) {
      add({
        code: "HYPERTENSION_STANDALONE",
        urgency: "ABNORMAL",
        title_en: "Elevated Blood Pressure — Assess End-Organ Risk & Baseline LVH",
        title_ar: "ارتفاع ضغط دم — قيّم خطورة الأعضاء المستهدفة وتضخم البطين",
        reason_en: "Elevated BP increases myocardial oxygen demand and predisposes to LVH, AFib, and demand ischemia.",
        reason_ar: "ارتفاع الضغط يزيد حمل القلب ويهيئ لتضخم البطين والرجفان الأذيني والإقفار.",
        action_en: "Screen for hypertensive emergency symptoms (neurologic, chest pain, dyspnea, renal) and avoid abrupt BP drops if asymptomatic.",
        action_ar: "تأكد من غياب أعراض طوارئ الضغط وتجنب خفض الضغط بشكل مفاجئ وعنيف إذا كان المريض بدون أعراض.",
      });
    }
  }

  // =========================================================================
  // D. LOW SpO2 / HYPOXIA (Every ECG Combo + Standalone)
  // =========================================================================
  if (isLowSpo2) {
    let matchedHypoxia = false;
    if (hasTachy || hasRbbbOrStrain) {
      matchedHypoxia = true;
      add({
        code: "PE_SUSPECTED_HYPOXIA",
        urgency: "URGENT",
        title_en: "Suspected Acute Pulmonary Embolism (Hypoxia + Tachycardia / RV Strain)",
        title_ar: "اشتباه جلطة رئوية حادة (نقص أكسجين + تسارع أو إجهاد بطين أيمن)",
        reason_en: "SpO2 <92% combined with tachycardia, AFib, RBBB, or anterior TWI strongly elevates clinical probability of PE.",
        reason_ar: "انخفاض الأكسجين (<92%) مع التسارع أو الرجفان الأذيني أو RBBB أو انقلاب T يرفع بقوة احتمالية الجلطة الرئوية.",
        action_en: "Administer supplemental O2, calculate Wells score, order D-dimer or CTPA, and evaluate for anticoagulation.",
        action_ar: "أعطِ أكسجين فوراً، احسب Wells Score، واطلب D-dimer أو أشعة مقطعية بالصبغة على شرايين الرئة (CTPA).",
      });
    }
    if (hasIschemia || f.chest_pain_or_shock) {
      matchedHypoxia = true;
      add({
        code: "ACUTE_PULMONARY_EDEMA_OR_ACS",
        urgency: "CRITICAL",
        title_en: "Hypoxia with Acute Ischemia / Dyspnea — Suspect Cardiogenic Pulmonary Edema",
        title_ar: "نقص أكسجين مع إقفار/ضيق نفس — اشتباه وذمة رئوية قلبية حادة",
        reason_en: "Hypoxemia during acute ischemia suggests acute LV failure / pulmonary edema worsening myocardial oxygen supply.",
        reason_ar: "نقص الأكسجين مع الإقفار الحاد يرجّح فشلاً حاداً بالبطين الأيسر ووذمة رئوية تزيد نقص التروية.",
        action_en: "Give O2 / NIV (CPAP/BiPAP), sit patient upright, check lung ultrasound/CXR, and consider diuretics/nitrates if BP permits.",
        action_ar: "أعطِ أكسجين أو تنفساً غير تداخلي (CPAP/BiPAP) واجلس المريض مع تقييم السوائل ومدرات البول حسب الضغط.",
      });
    }
    if (!matchedHypoxia) {
      add({
        code: "HYPOXIA_STANDALONE",
        urgency: "URGENT",
        title_en: "Hypoxemia (SpO2 <92%) — Treat Oxygenation & Search Secondary Cardiopulmonary Cause",
        title_ar: "نقص أكسجين بالدم (<92%) — عالج الأكسجة وابحث عن السبب القلبي الرئوي",
        reason_en: "Hypoxia itself provokes arrhythmias (AFib, MAT, ectopy) and demand ischemia even before classic ECG signs appear.",
        reason_ar: "نقص الأكسجين وحده يثير اضطرابات النظم ونقص التروية الثانوي حتى قبل ظهور علامات ECG كاملة.",
        action_en: "Administer supplemental O2 immediately and evaluate for PE, CHF/pulmonary edema, pneumonia, or COPD exacerbation.",
        action_ar: "أعطِ أكسجين فوراً وقيّم احتمالات الجلطة الرئوية أو هبوط القلب أو الالتهاب الرئوي أو تفاقم COPD.",
      });
    }
  }

  // =========================================================================
  // E. FEVER & HYPOTHERMIA (Every ECG Combo + Standalone)
  // =========================================================================
  if (isFever) {
    let matchedFever = false;
    if (hasBrugada) {
      matchedFever = true;
      add({
        code: "FEVER_INDUCED_BRUGADA",
        urgency: "URGENT",
        title_en: "Fever-Unmasked Brugada Pattern (High Arrhythmic Risk)",
        title_ar: "نمط بروجدا مُستثار بالحمى (خطر اضطراب نظم بطيني)",
        reason_en: "Fever impairs mutant SCN5A sodium channels, unmasking Type 1 Brugada and triggering polymorphic VT/VF.",
        reason_ar: "الحمى تثبّط قنوات الصوديوم وتظهر نمط بروجدا وقد تثير رجفاناً بطينياً خطيراً.",
        action_en: "Give aggressive antipyretics immediately (acetaminophen/cooling), place on telemetry, and avoid sodium-channel blockers.",
        action_ar: "اخفض الحرارة بقوة وفوراً مع مراقبة مستمرة وتجنب أدوية حاصرات قنوات الصوديوم.",
      });
    }
    if (hasAfibOrFlutter || hasTachy) {
      matchedFever = true;
      add({
        code: "SECONDARY_AFIB_OR_TACHY_FEVER",
        urgency: "URGENT",
        title_en: "Fever-Driven Secondary Tachycardia / AFib (Treat Underlying Infection First)",
        title_ar: "تسارع قلب / رجفان أذيني ثانوي بسبب الحمى (عالج العدوى أولاً)",
        reason_en: "Per EM Cases / CAEP AFib guidelines, secondary AFib/tachycardia from fever/sepsis has 3x higher mortality; aggressive rate control can destabilize the patient.",
        reason_ar: "وفق إرشادات طوارئ الرجفان الأذيني (AFIB Checklist)، التسارع الثانوي للحمى/الإنتان يحمل خطورة أعلى 3 أضعاف — عالج السبب أولاً قبل خفض النبض العنيف.",
        action_en: "Give antipyretics, IV fluids, and infectious workup/antibiotics; reassess heart rate after fever and volume are treated.",
        action_ar: "أعطِ خافض حرارة وسوائل وريدية وفحص عدوى/مضاد حيوي وأعد تقييم النبض بعد تحسن الحرارة والسوائل.",
      });
    }
    if (hasPericarditisOrLowVolt) {
      matchedFever = true;
      add({
        code: "INFECTIOUS_PERICARDITIS_MYOCARDITIS",
        urgency: "URGENT",
        title_en: "Suspected Acute Infectious Pericarditis / Myopericarditis",
        title_ar: "اشتباه التهاب تامور حاد أو التهاب عضلة القلب (Pericarditis / Myocarditis)",
        reason_en: "Fever paired with diffuse ST elevation or PR depression strongly points to acute infectious/viral pericarditis or myopericarditis.",
        reason_ar: "الحمى مع ارتفاع ST المنتشر أو انخفاض PR ترجّح التهاب التامور أو عضلة القلب الفيروسي/البكتيري.",
        action_en: "Check troponin, CRP/ESR, and bedside echo for pericardial effusion; consider NSAID + colchicine if pure pericarditis.",
        action_ar: "اطلب تروبونين و CRP وإيكو للتأكد من عدم وجود انصباب تاموري أو إصابة لعضلة القلب.",
      });
    }
    if (hasAvBlock) {
      matchedFever = true;
      add({
        code: "FEVER_NEW_AV_BLOCK_ENDOCARDITIS",
        urgency: "URGENT",
        title_en: "Fever + AV Block — Rule Out Infective Endocarditis (Root Abscess) / Myocarditis / Lyme",
        title_ar: "حمى مع إحصار أذيني بطيني — استبعد التهاب شغاف القلب أو عضلة القلب",
        reason_en: "New AV conduction block in a febrile patient is a high-risk sign for perivalvular abscess in infective endocarditis, myocarditis, or Lyme carditis.",
        reason_ar: "ظهور إحصار AV مع حمى علامة مهمة لاحتمال خراج صمامي في التهاب الشغاف أو التهاب عضلة القلب.",
        action_en: "Obtain blood cultures, urgent echocardiography, telemetry monitoring, and infectious disease/cardiology consult.",
        action_ar: "اسحب مزارع دم واطلب إيكو عاجل مع مراقبة قلب مستمرة واستشارة قلب ومعدية.",
      });
    }
    if (!matchedFever) {
      add({
        code: "FEVER_STANDALONE",
        urgency: "ABNORMAL",
        title_en: "Fever (≥38°C) — Secondary Cardiac Demand & Channelopathy Trigger",
        title_ar: "حمى (≥38°م) — عامل إجهاد قلبي ثانوي قد يكشف اضطرابات نظم",
        reason_en: "Fever elevates heart rate (~10 bpm per °C) and can unmask Brugada or provoke secondary arrhythmias.",
        reason_ar: "الحمى ترفع معدل النبض وتزيد احتياج العضلة للأكسجين وقد تكشف متلازمات كهربائية كامنة.",
        action_en: "Administer antipyretics, evaluate source of infection, and repeat ECG once afebrile if borderline findings exist.",
        action_ar: "أعطِ خافض حرارة وابحث عن مصدر العدوى وأعد التخطيط بعد هبوط الحرارة إذا وجدت تغيرات حدّية.",
      });
    }
  }

  if (isHypothermia) {
    add({
      code: "HYPOTHERMIA_ECG_PATTERN",
      urgency: "URGENT",
      title_en: hasBrady || hasLongQt || f.osborn_j_waves
        ? "Hypothermia-Related ECG Changes (Bradycardia / Osborn J-Waves / QT Prolongation)"
        : "Hypothermia (<35°C) — High Risk of Osborn Waves, Bradycardia & VF",
      title_ar: hasBrady || hasLongQt || f.osborn_j_waves
        ? "تغيرات تخطيط القلب بسبب انخفاض الحرارة الشديد (Hypothermia)"
        : "انخفاض حرارة الجسم (<35°م) — خطر بطء قلب وموجات Osborn ورجفان بطيني",
      reason_en: "Core hypothermia (<35°C) slows conduction, prolongs PR/QRS/QT, produces Osborn J-waves, and makes the myocardium highly irritable to VF.",
      reason_ar: "انخفاض الحرارة (<35°م) يبطئ التوصيل ويطيل الفواصل ويظهر موجات Osborn ويجعل عضلة القلب شديدة التهيج للرجفان البطيني.",
      action_en: "Initiate active core rewarming and gentle handling; avoid unnecessary pacing or repeated shocks until core temp rises >30°C.",
      action_ar: "ابدأ التدفئة التدريجية الفورية مع التعامل اللطيف جداً وتجنب منظم الضربات غير الضروري.",
    });
  }

  // =========================================================================
  // F. BLOOD GLUCOSE ABNORMALITIES (High/DKA & Low — Every Combo + Standalone)
  // =========================================================================
  if (isHighGluc) {
    let matchedHighGluc = false;
    if (hasPeakedOrHyperK || hasIschemia) {
      matchedHighGluc = true;
      add({
        code: "DKA_HYPERKALEMIA_PSEUDO_STEMI",
        urgency: "CRITICAL",
        title_en: "Hyperglycemia / DKA with Repolarization or QRS Changes — Rule Out Hyperkalemia",
        title_ar: "ارتفاع سكر / DKA مع تغيرات موجات — استبعد ارتفاع البوتاسيوم فوراً",
        reason_en: "DKA causes severe transcellular K+ shifts producing peaked T waves, wide QRS, or Brugada/STEMI-like ST elevation.",
        reason_ar: "غيبوبة السكر الكيتونية تسبب ارتفاعاً خطيراً في البوتاسيوم يظهر كـ Peaked T أو توسع QRS أو ارتفاع ST محاكٍ للجلطة.",
        action_en: "Check stat K+ and VBG; give IV Calcium Gluconate FIRST for cardiac membrane protection before starting insulin.",
        action_ar: "اطلب بوتاسيوم وغازات دم فوراً، وأعطِ كالسيوم جلوكونات وريدياً أولاً لحماية القلب قبل بدء الإنسولين.",
      });
    }
    if (hasTachy) {
      matchedHighGluc = true;
      add({
        code: "DKA_SECONDARY_TACHYARRHYTHMIA",
        urgency: "URGENT",
        title_en: "Secondary Tachycardia / AFib Driven by Hyperglycemia / DKA",
        title_ar: "تسارع قلب / رجفان أذيني ثانوي لارتفاع السكر / DKA",
        reason_en: "Osmotic diuresis, dehydration, and acidosis in DKA drive secondary tachycardia or AFib; rate control alone is harmful.",
        reason_ar: "الجفاف الشديد والحماض في DKA يسببان تسارعاً أو رجفاناً أذينياً ثانوياً؛ العلاج الأساسي هو السوائل والإنسولين لا خفض النبض.",
        action_en: "Start IV isotonic fluids, check VBG/ketones/K+, and initiate DKA protocol; reassess rhythm after volume resuscitation.",
        action_ar: "ابدأ السوائل الوريدية وتحاليل DKA والبوتاسيوم، وأعد تقييم الإيقاع بعد تعويض السوائل.",
      });
    }
    if (!matchedHighGluc) {
      add({
        code: "HYPERGLYCEMIA_STANDALONE",
        urgency: "URGENT",
        title_en: "Hyperglycemia / Suspected DKA — Check Electrolytes (K+/Mg2+) & Silent ACS",
        title_ar: "ارتفاع سكر الدم / اشتباه DKA — افحص البوتاسيوم والجلطة الصامتة",
        reason_en: "Hyperglycemia/DKA alters potassium rapidly and diabetic patients frequently present with silent or atypical myocardial ischemia.",
        reason_ar: "ارتفاع السكر يغيّر مستوى البوتاسيوم بسرعة، كما أن مرضى السكر معرضون للجلطات الصامتة بدون ألم صدري كلاسيكي.",
        action_en: "Check serum K+, Mg2+, VBG/ketones, and troponin if any atypical dyspnea, fatigue, or diaphoresis is present.",
        action_ar: "افحص البوتاسيوم والمغنيسيوم وغازات الدم/الكيتون، والتروبونين عند وجود أي نهجان أو عرق أو إعياء غير مفسر.",
      });
    }
  }

  if (isLowGluc) {
    add({
      code: hasLongQt || hasTachy || hasIschemia ? "HYPOGLYCEMIA_ARRHYTHMIA_QT_RISK" : "HYPOGLYCEMIA_IMMEDIATE_TREATMENT",
      urgency: hasLongQt || hasTachy || hasIschemia ? "CRITICAL" : "URGENT",
      title_en: hasLongQt || hasTachy || hasIschemia
        ? "Hypoglycemia-Induced Arrhythmia / QT Prolongation / Demand Ischemia"
        : "Acute Hypoglycemia — Immediate Glucose Correction Required",
      title_ar: hasLongQt || hasTachy || hasIschemia
        ? "هبوط سكر حاد مسبّب لاضطراب النظم / إطالة QT / إقفار ثانوي"
        : "هبوط سكر حاد بالدم — يتطلب تصحيحاً فورياً بالجلوكوز",
      reason_en: "Acute hypoglycemia triggers massive catecholamine release and intracellular K+ shift, causing tachycardia, ST-T shifts, and dangerous QT prolongation.",
      reason_ar: "هبوط السكر الحاد يطلق أدرينالين كثيفاً ويدخل البوتاسيوم للخلايا مما يسبب تسارعاً وتغيرات ST-T وإطالة QT خطيرة.",
      action_en: "Administer IV Dextrose (D50W/D10W) or oral glucose immediately, recheck capillary glucose in 15 min, and repeat ECG after normoglycemia.",
      action_ar: "أعطِ جلوكوز وريدي أو فموي فوراً، أعد قياس السكر بعد 15 دقيقة، وأعد تخطيط القلب بعد عودة السكر للطبيعي.",
    });
  }

  // =========================================================================
  // G. SECONDARY AFIB TRIGGER FROM SYMPTOMS (Dehydration/Bleed/Pleuritic Pain)
  // =========================================================================
  if (hasAfibOrFlutter && f.pleuritic_or_dehydration_or_bleed) {
    add({
      code: "SECONDARY_AFIB_RVR",
      urgency: "URGENT",
      title_en: "Secondary Atrial Fibrillation / Flutter (Dehydration, Bleeding, or Pleuritic/PE Trigger)",
      title_ar: "رجفان أو رفرفة أذينية ثانوية (جفاف / نزيف / ألم جنبي)",
      reason_en: "Per EM Cases / CAEP AFIB Checklist, secondary AFib carries 3x higher mortality; prioritize treating hypovolemia, bleeding, or PE before aggressive rate/rhythm control.",
      reason_ar: "وفق بروتوكول طوارئ الرجفان الأذيني (AFIB Checklist)، الرجفان الثانوي أخطر بـ 3 أضعاف — عالج الجفاف أو النزيف أو الجلطة الرئوية أولاً.",
      action_en: "Treat the underlying trigger (IV fluids, CBC/coags, D-dimer/CTPA if pleuritic) and avoid aggressive AV nodal blockade until volume/cause is stabilized.",
      action_ar: "عالج السبب الأساسي أولاً (سوائل وريدية، صورة دم، فحص جلطة رئوية عند الألم الجنبي) وتجنب خفض النبض العنيف قبل الاستقرار.",
    });
  }

  return out;
}
