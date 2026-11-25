/**
 * Bilingual (English + Medical Arabic) Textbook-Grade ECG Clinical Knowledge Library
 * Covers all 18 DenseNet121 v2 classes + Bifascicular / "Trifascicular" composite syndromes,
 * with dedicated Investigations (Inv) and Treatment (TTT) for floating bedside panels.
 */

export interface EcgClinicalEntry {
  title_en: string;
  title_ar: string;
  explanation_en: string;
  explanation_ar: string;
  inv_en: string;
  inv_ar: string;
  ttt_en: string;
  ttt_ar: string;
  /** Legacy alias kept equal to ttt for backward compatibility */
  action_en: string;
  action_ar: string;
}

type RawEntry = Omit<EcgClinicalEntry, "action_en" | "action_ar">;

function mk(e: RawEntry): EcgClinicalEntry {
  return { ...e, action_en: e.ttt_en, action_ar: e.ttt_ar };
}

export const ECG_CLINICAL_LIBRARY: Record<string, EcgClinicalEntry> = {
  NORMAL_SINUS_RHYTHM: mk({
    title_en: "Normal Sinus Rhythm",
    title_ar: "نظم جيبي طبيعي (Normal Sinus Rhythm)",
    explanation_en:
      "Regular SA-node rhythm (60–100 bpm) with upright P waves in I, II, aVF (inverted in aVR), constant PR (120–200 ms), and narrow QRS (<120 ms) without acute ischemic ST-T shifts.",
    explanation_ar:
      "نظم جيبي منتظم (60–100 نبضة/دقيقة) مع موجات P موجبة في I وII وaVF (ومقلوبة في aVR)، وفترة PR ثابتة (120–200 مللي ثانية)، وQRS ضيق (<120 مللي ثانية) دون تغيرات إقفارية حادة.",
    inv_en:
      "If chest pain persists, do NOT rely on one normal ECG: repeat serial ECGs every 15–30 min, order high-sensitivity Troponin (0h/1h or 0h/3h), CBC, BMP, and Chest X-ray as clinically indicated.",
    inv_ar:
      "إذا استمر ألم الصدر لا تعتمد على تخطيط واحد طبيعي: أعد الـ ECG كل 15–30 دقيقة، واطلب تروبونين عالي الحساسية متسلسل، وصورة دم، وشوارد، وأشعة صدر حسب الحالة.",
    ttt_en:
      "No acute arrhythmia intervention required. Reassure if asymptomatic; if ischemic symptoms exist, monitor on telemetry and evaluate non-ECG causes of chest pain or dyspnea.",
    ttt_ar:
      "لا حاجة لتدخل طارئ لاضطراب نظم. طمئن المريض إذا كان بدون أعراض؛ أما مع وجود أعراض إقفارية فضعه على المراقبة وقيّم الأسباب الأخرى.",
  }),

  ANTERIOR_SEPTAL_MI: mk({
    title_en: "Anterior / Septal STEMI (LAD Territory)",
    title_ar: "احتشاء أمامي / حاجزي حاد (Anterior / Septal STEMI)",
    explanation_en:
      "Convex ST-segment elevation in V1–V4 (or STEMI equivalents such as De Winter upsloping ST depression + tall symmetric T waves in V2–V4) indicating acute LAD coronary occlusion, frequently with reciprocal ST depression in II, III, aVF.",
    explanation_ar:
      "ارتفاع محدّب في قطعة ST بـ V1–V4 (أو معادلات الجلطة مثل نمط De Winter: انخفاض ST صاعد مع موجات T ضخمة متماثلة في V2–V4) يدل على انسداد حاد في الشريان الأمامي النازل (LAD) مع انخفاض عكسي في II وIII وaVF.",
    inv_en:
      "Do NOT delay reperfusion for labs. Draw stat hs-Troponin, CBC, BMP (K+/Mg2+), coagulation profile, lipid panel, and bedside Echo (if mechanical complication or dissection suspected).",
    inv_ar:
      "لا تؤخر القسطرة بانتظار التحاليل. اسحب تروبونين عاجل، صورة دم، شوارد (K+/Mg2+)، سيولة، دهون، وإيكو بجانب السرير عند الاشتباه بمضاعفات ميكانيكية.",
    ttt_en:
      "Activate Cath Lab immediately (Primary PCI <90 min; fibrinolysis if PCI >120 min and no contraindications). Give Aspirin 300 mg chewed, P2Y12 inhibitor, IV Heparin, O2 only if SpO2 <90%, and defibrillator pads.",
    ttt_ar:
      "تفعيل معمل القسطرة فوراً (قسطرة أولية <90 دقيقة؛ أو مذيب جلطة إذا تعذرت خلال 120 دقيقة دون موانع). أعطِ أسبرين 300 مجم مضغاً، مثبط P2Y12، هيبارين وريدي، وأكسجين فقط إذا SpO2 <90%.",
  }),

  INFERIOR_POSTERIOR_MI: mk({
    title_en: "Inferior / Posterior STEMI (RCA / LCx Territory)",
    title_ar: "احتشاء سفلي / خلفي حاد (Inferior / Posterior STEMI)",
    explanation_en:
      "ST elevation in II, III, aVF with reciprocal ST depression in I and aVL (RCA likely if ST III > II). Horizontal ST depression with tall R waves in V1–V3 indicates true posterior wall extension.",
    explanation_ar:
      "ارتفاع ST في II وIII وaVF مع انخفاض عكسي في I وaVL (غالباً RCA إذا كان ارتفاع III > II). انخفاض ST الأفقي مع موجة R طويلة في V1–V3 يدل على امتداد خلفي حقيقي.",
    inv_en:
      "Obtain Right-Sided ECG (V4R) immediately to rule out RV infarction BEFORE nitrates, plus Posterior leads (V7–V9) if V1–V3 ST depression is present. Draw stat Troponin, CBC, BMP, coagulation.",
    inv_ar:
      "اعمل أقطاباً يمنى (V4R) فوراً لاستبعاد احتشاء البطين الأيمن قبل النيترات، وأقطاباً خلفية (V7–V9) إذا وُجد انخفاض ST في V1–V3، مع تروبونين وشوارد وسيولة.",
    ttt_en:
      "Activate Cath Lab immediately (Primary PCI). Give Aspirin + P2Y12 + Heparin. AVOID nitrates/morphine if RV MI or BP is low — give cautious IV crystalloid boluses instead. Keep Atropine/pacing pads ready.",
    ttt_ar:
      "تفعيل القسطرة فوراً مع أسبرين + P2Y12 + هيبارين. تجنّب النيترات والمورفين تماماً مع احتشاء البطين الأيمن أو هبوط الضغط وأعطِ محاليل وريدية بحذر، وجهّز الأتروبين والمنظم الخارجي.",
  }),

  LATERAL_MI: mk({
    title_en: "Lateral / High-Lateral STEMI (LCx / Diagonal)",
    title_ar: "احتشاء جانبي حاد (Lateral / High-Lateral STEMI)",
    explanation_en:
      "ST elevation in high-lateral leads (I, aVL) and/or V5–V6 with prominent reciprocal ST depression in III and aVF, reflecting LCx or first diagonal LAD occlusion.",
    explanation_ar:
      "ارتفاع ST في I وaVL و/أو V5–V6 مع انخفاض عكسي واضح في III وaVF، يشير لانسداد الشريان المنعطف الأيسر (LCx) أو الفرع القطري الأول.",
    inv_en:
      "Confirm subtle ST elevation in I & aVL alongside reciprocal III/aVF depression. Obtain posterior leads (V7–V9) if V1–V3 ST depression exists; draw stat Troponin, CBC, BMP, coagulation.",
    inv_ar:
      "تحقق بدقة من ارتفاع ST في I وaVL مقابل الانخفاض العكسي في III/aVF، واعمل V7–V9 عند اللزوم، مع تروبونين وشوارد وسيولة.",
    ttt_en:
      "Activate Primary PCI immediately, dual antiplatelet therapy (Aspirin + P2Y12), anticoagulation, and continuous telemetry monitoring.",
    ttt_ar:
      "تفعيل القسطرة الأولية فوراً مع مضاد صفائح مزدوج (أسبرين + P2Y12) ومضاد تخثر ومراقبة قلبية مستمرة.",
  }),

  NSTEMI_ISCHEMIA: mk({
    title_en: "Subendocardial Ischemia / NSTEMI Pattern",
    title_ar: "إقفار تحت الشغاف / اشتباه NSTEMI",
    explanation_en:
      "Horizontal/downsloping ST depression (≥0.5–1 mm) and/or deep symmetric T-wave inversions in contiguous leads. Diffuse ST depression (≥6 leads) + ST elevation in aVR warns of left main or severe 3-vessel disease.",
    explanation_ar:
      "انخفاض أفقي/منحدر في ST (≥0.5–1 مم) و/أو انقلاب T عميق متماثل في أقطاب متجاورة. انخفاض ST المنتشر مع ارتفاع ST في aVR يحذر من تضيق الشريان التاجي الرئيسي الأيسر.",
    inv_en:
      "Serial hs-Troponin (0h & 1–3h), repeat ECGs every 15–30 min, CBC (exclude anemia), BMP (K+/Mg2+/Cr), Chest X-ray, and risk scoring (HEART / GRACE).",
    inv_ar:
      "تروبونين متسلسل (عند الساعة 0 ثم 1–3 ساعات)، إعادة ECG كل 15–30 دقيقة، صورة دم (استبعاد الأنيميا)، شوارد ووظائف كلى، أشعة صدر، وتقييم HEART/GRACE.",
    ttt_en:
      "Aspirin 300 mg + anticoagulation (Heparin/Enoxaparin) + sublingual/IV nitroglycerin (if BP allows & no PDE5 inhibitor) + beta-blocker if no heart failure/shock. Urgent cardiology consult for early invasive angiography.",
    ttt_ar:
      "أسبرين 300 مجم + مضاد تخثر (هيبارين/إنوكسابارين) + نيتروجلسرين إذا سمح الضغط + حاصر بيتا في غياب هبوط القلب/الصدمة، مع استشارة قلب لقسطرة مبكرة.",
  }),

  ATRIAL_FIBRILLATION: mk({
    title_en: "Atrial Fibrillation (AFib)",
    title_ar: "رجفان أذيني (Atrial Fibrillation)",
    explanation_en:
      "Irregularly irregular rhythm with absent discrete P waves and fibrillatory baseline waves. Always distinguish primary AFib from secondary AFib (sepsis, PE, bleed, hypovolemia, ACS), where treating the trigger comes first.",
    explanation_ar:
      "نظم غير منتظم بشكل تام مع غياب موجات P وظهور موجات رجفان دقيقة. ميّز دائماً بين الرجفان الأولي والثانوي (إنتان، جلطة رئوية، نزيف، جفاف، إقفار) حيث يُعالج السبب أولاً.",
    inv_en:
      "Screen secondary causes first: CBC, BMP (K+/Mg2+), TSH, hs-Troponin, lactate/CRP if febrile, Chest X-ray, and D-dimer/CTPA if hypoxic. Echocardiogram for structural/valvular assessment.",
    inv_ar:
      "ابحث عن السبب الثانوي أولاً: صورة دم، شوارد (K+/Mg2+)، وظائف غدة درقية (TSH)، تروبونين، لاكتات/CRP عند الحمى، أشعة صدر، وإيكو قلب.",
    ttt_en:
      "Unstable or pre-excited (WPW): immediate synchronized cardioversion. Secondary AFib: treat underlying cause (fluids, antibiotics, blood) before rate control. Primary stable: rate control (Metoprolol/Diltiazem) or rhythm control + DOAC if CHADS-65 positive.",
    ttt_ar:
      "غير مستقر أو WPW: صدمة كهربائية متزامنة فوراً. رجفان ثانوي: عالج السبب أولاً قبل تبطيء النبض. أولي ومستقر: تحكم بالنبض (ميتوبرولول/ديلتيازيم) أو النظم + مضاد تجلط (DOAC) حسب CHADS-65.",
  }),

  ATRIAL_FLUTTER: mk({
    title_en: "Atrial Flutter",
    title_ar: "رفرفة أذينية (Atrial Flutter)",
    explanation_en:
      "Macro-reentrant atrial circuit (~300 bpm) producing regular 'sawtooth' flutter waves (best in II, III, aVF, V1) with fixed (often 2:1 at ~150 bpm) or variable AV conduction.",
    explanation_ar:
      "دائرة إعادة دخول أذينية (~300/دقيقة) تُنتج موجات رفرفة تشبه أسنان المنشار في II وIII وaVF وV1 مع توصيل ثابت (غالباً 2:1 بمعدل ~150/دقيقة) أو متغير.",
    inv_en:
      "Check electrolytes (K+/Mg2+), TSH, CBC, Chest X-ray, and Echo; if 2:1 narrow tachycardia at 150 bpm is uncertain, vagal maneuvers or Adenosine can transiently unmask flutter waves.",
    inv_ar:
      "شوارد (K+/Mg2+)، وظائف غدة درقية، صورة دم، أشعة صدر وإيكو؛ ويمكن لمناورة فالسالفا أو الأدينوزين كشف موجات الرفرفة مؤقتاً.",
    ttt_en:
      "Synchronized cardioversion (often responds to low energy 50–100 J) if unstable; otherwise rate control (beta-blocker/CCB), treat secondary triggers, and apply AFib anticoagulation criteria.",
    ttt_ar:
      "صدمة كهربائية متزامنة (غالباً تستجيب لطاقة منخفضة 50–100 جول) إذا كان غير مستقر؛ وإلا تحكم بالنبض ومضاد تجلط بنفس قواعد الرجفان الأذيني.",
  }),

  SVT_TACHYCARDIA: mk({
    title_en: "Supraventricular Tachycardia (SVT)",
    title_ar: "تسارع فوق بطيني (Supraventricular Tachycardia)",
    explanation_en:
      "Regular narrow-complex tachycardia (150–250 bpm, QRS <120 ms) from AVNRT or orthodromic AVRT, with absent or retrograde P waves.",
    explanation_ar:
      "تسارع منتظم وسريع بمركبات QRS ضيقة (150–250/دقيقة) ناتج عن دائرة إعادة دخول في العقدة الأذينية البطينية (AVNRT) أو مسار إضافي (AVRT).",
    inv_en:
      "Run a continuous rhythm strip during conversion, then obtain a post-conversion 12-lead ECG specifically to check for WPW delta waves. Check K+, Mg2+, TSH, and hemoglobin.",
    inv_ar:
      "سجّل شريط تخطيط مستمراً أثناء الإرجاع، ثم 12-lead ECG بعد الإرجاع لفحص موجات دلتا (WPW)، مع شوارد (K+/Mg2+) وTSH وهيموجلوبين.",
    ttt_en:
      "Unstable: immediate synchronized cardioversion (50–100 J). Stable: Modified Valsalva maneuver first → rapid IV Adenosine 6 mg (then 12 mg) via large proximal vein with saline flush.",
    ttt_ar:
      "غير مستقر: صدمة كهربائية متزامنة فوراً (50–100 جول). مستقر: مناورة فالسالفا المعدّلة أولاً ← أدينوزين وريدي سريع 6 مجم (ثم 12 مجم) مع دفع محلول ملحي فوري.",
  }),

  SINUS_BRADYCARDIA: mk({
    title_en: "Sinus Bradycardia",
    title_ar: "بطء القلب الجيبي (Sinus Bradycardia)",
    explanation_en:
      "Regular sinus rhythm (P before every QRS, constant PR) at <60 bpm. Benign in athletes/sleep, or secondary to beta-blockers/CCBs, inferior MI, hyperkalemia, hypothyroidism, hypothermia, or raised ICP.",
    explanation_ar:
      "نظم جيبي منتظم بمعدل أقل من 60/دقيقة. طبيعي في الرياضيين والنوم، أو ثانوي لحاصرات بيتا/الكالسيوم، الاحتشاء السفلي، ارتفاع البوتاسيوم، قصور الدرقية، انخفاض الحرارة، أو ارتفاع ضغط الجمجمة.",
    inv_en:
      "Medication review (beta-blockers, CCBs, digoxin), BMP (K+/Ca2+), TSH, Troponin (if ischemic symptoms), core temperature, and neuro exam if hypertensive (Cushing's reflex).",
    inv_ar:
      "مراجعة الأدوية (حاصرات بيتا/كالسيوم/ديجوكسين)، شوارد (K+/Ca2+)، TSH، تروبونين عند أعراض إقفار، حرارة الجسم، وفحص عصبي مع ارتفاع الضغط.",
    ttt_en:
      "Asymptomatic: observe and hold offending drugs. Symptomatic (hypotension, syncope, altered mental status, ischemia): IV Atropine 1 mg (repeat q3–5 min up to 3 mg), transcutaneous pacing, or Dopamine/Epinephrine infusion.",
    ttt_ar:
      "بدون أعراض: متابعة وإيقاف الدواء المسبب. بأعراض عدم استقرار: أتروبين 1 مجم وريدياً (يتكرر حتى 3 مجم)، أو منظم جلدي خارجي، أو دوبامين/أدرينالين بالتنقيط.",
  }),

  FIRST_DEGREE_AV_BLOCK: mk({
    title_en: "First-Degree AV Block (PR > 200 ms)",
    title_ar: "إحصار أذيني بطيني من الدرجة الأولى (First-Degree AV Block)",
    explanation_en:
      "Constant PR prolongation >200 ms with 1:1 AV conduction. When combined with RBBB + LAFB, clinicians often call it 'trifascicular block', though per the 2009 AHA/ACCF/HRS statement this usually reflects bifascicular block plus AV-node/His delay rather than true anatomical trifascicular block.",
    explanation_ar:
      "استطالة ثابتة في PR >200 مللي ثانية مع توصيل 1:1. عند اجتماعه مع RBBB وLAFB يُشتبه بـ «إحصار ثلاثي الحزم»، لكن توصيات AHA/ACCF/HRS (2009) توضح أنه غالباً إحصار ثنائي الحزم مع تأخر في العقدة الأذينية البطينية أو حزمة هيس ولا يُثبت الإحصار الثلاثي الحقيقي إلا بتسجيل حزمة هيس.",
    inv_en:
      "Compare with prior ECGs, check QRS width/axis for co-existing RBBB + LAFB, review AV-nodal blockers, and check BMP (K+/Mg2+), Lyme/endocarditis workup if febrile.",
    inv_ar:
      "قارن بتخطيطات سابقة، افحص وجود RBBB + LAFB مصاحب، راجع الأدوية المثبطة للعقدة الأذينية البطينية، وافحص الشوارد (K+/Mg2+).",
    ttt_en:
      "Isolated 1st-degree AVB is benign and needs no pacing; adjust offending AV-nodal blockers if PR >300 ms. If paired with bifascicular block or syncope, monitor on telemetry and refer to Cardiology/EP.",
    ttt_ar:
      "الإحصار المعزول حميد ولا يحتاج منظماً؛ راجع/خفّض الأدوية المثبطة إذا تجاوز PR 300 مللي ثانية. مع إحصار ثنائي الحزم أو إغماء: مراقبة وتحويل لقسم كهرباء القلب.",
  }),

  HIGH_GRADE_AV_BLOCK: mk({
    title_en: "High-Grade / Complete AV Block",
    title_ar: "إحصار أذيني بطيني متقدم / تام (High-Grade / Complete AV Block)",
    explanation_en:
      "Intermittent failure of AV conduction (Mobitz II / 2:1 high-grade block) or complete AV dissociation (3rd-degree CHB with slow escape rhythm), typically infranodal His-Purkinje disease with high asystole risk.",
    explanation_ar:
      "فشل متقطع في التوصيل (موبيتز 2 / إحصار 2:1 متقدم) أو انفصال أذيني بطيني تام (درجة ثالثة مع نظم هروب بطيء)، ويعكس إصابة تحت العقدة مع خطر عالٍ لتوقف القلب.",
    inv_en:
      "Stat BMP (K+/Mg2+/Ca2+), hs-Troponin (exclude inferior/anterior MI), digoxin level if applicable, blood cultures + Echo if febrile (aortic root abscess), and continuous telemetry.",
    inv_ar:
      "شوارد عاجلة (K+/Mg2+/Ca2+)، تروبونين لاستبعاد الجلطة، مستوى ديجوكسين إن وُجد، ومزارع دم + إيكو مع الحمى (خراج جذر الأورطي).",
    ttt_en:
      "Apply transcutaneous pacing pads immediately. Treat reversible causes (Hyperkalemia → IV Calcium; MI → PCI). Note: Atropine is often ineffective in infranodal Mobitz II/CHB — proceed promptly to pacing and urgent Cardiology consult.",
    ttt_ar:
      "ثبّت أقطاب المنظم الجلدي فوراً وعالج الأسباب القابلة للعكس (بوتاسيوم/جلطة). الأتروبين غالباً غير فعال في الإحصار تحت العقدي — جهّز المنظم الخارجي واستدعِ القلب عاجلاً.",
  }),

  RBBB: mk({
    title_en: "Right Bundle Branch Block (RBBB)",
    title_ar: "إحصار الحزمة اليمنى (Right Bundle Branch Block)",
    explanation_en:
      "QRS ≥120 ms with rsR' ('M-shaped') in V1–V2 and wide slurred S wave in I, aVL, V5–V6 (secondary T inversion in V1–V3). Unlike LBBB, RBBB does NOT mask STEMI — new RBBB + LAFB with chest pain strongly suggests proximal LAD occlusion.",
    explanation_ar:
      "QRS ≥120 مللي ثانية مع نمط rsR' في V1–V2 وموجة S عريضة ومبطّأة في I وaVL وV5–V6. بعكس LBBB، الـ RBBB لا يُخفي الجلطة الحادة، وظهور RBBB جديد مع LAFB وألم صدري يحذر من انسداد LAD القريب.",
    inv_en:
      "Compare with prior ECGs (new vs chronic), check frontal axis for LAFB/LPFB (bifascicular block) and PR interval, order Troponin/CXR/D-dimer if chest pain or dyspnea (PE / acute LAD).",
    inv_ar:
      "قارن بتخطيط سابق (جديد أم مزمن)، افحص المحور للبحث عن LAFB/LPFB وطول PR، واطلب تروبونين/أشعة صدر/D-dimer مع ألم الصدر أو ضيق النفس.",
    ttt_en:
      "Chronic isolated RBBB needs no acute treatment. New RBBB with ischemic chest pain or shock → treat as high-risk STEMI equivalent (urgent Cath Lab evaluation); if acute dyspnea/hypoxia → evaluate/treat PE.",
    ttt_ar:
      "RBBB المزمن المعزول لا يحتاج علاجاً طارئاً. أما RBBB الجديد مع ألم صدري/صدمة فيُقيّم فوراً كمعادل لجلطة LAD عالية الخطورة، ومع نقص الأكسجين استبعد الجلطة الرئوية.",
  }),

  LBBB: mk({
    title_en: "Left Bundle Branch Block (LBBB)",
    title_ar: "إحصار الحزمة اليسرى (Left Bundle Branch Block)",
    explanation_en:
      "QRS ≥120 ms, broad monophasic/notched R in I, aVL, V5–V6 (absent septal Q), and deep broad S in V1–V3. Apply Modified Sgarbossa criteria (concordant ST ≥1 mm or discordant ST/S ratio ≤ −0.25) to detect acute MI.",
    explanation_ar:
      "QRS ≥120 مللي ثانية، موجة R عريضة مشقوقة في I وaVL وV5–V6، وموجة S عميقة عريضة في V1–V3. طبّق معايير سجارْبوسا المعدّلة لاكتشاف الجلطة الحادة داخل LBBB.",
    inv_en:
      "Compare with prior ECG, calculate Modified Sgarbossa criteria, draw serial hs-Troponin, BMP, BNP, and arrange Transthoracic Echocardiography to assess LVEF and structural heart disease.",
    inv_ar:
      "قارن بتخطيط سابق، طبّق معايير سجارْبوسا المعدّلة، اسحب تروبونين متسلسل وشوارد وBNP، واطلب إيكو قلب لتقييم كفاءة البطين الأيسر.",
    ttt_en:
      "If Modified Sgarbossa positive or hemodynamic instability with chest pain → immediate Cath Lab activation. Otherwise manage underlying heart failure/hypertension and arrange Cardiology follow-up.",
    ttt_ar:
      "إذا تحققت معايير سجارْبوسا المعدّلة أو عدم استقرار مع ألم صدري ← تفعيل القسطرة فوراً. غير ذلك عالج الضغط/قصور القلب وتابع مع طبيب القلب.",
  }),

  FASCICULAR_BLOCK: mk({
    title_en: "Fascicular Block (LAFB / LPFB)",
    title_ar: "إحصار حُزيمي (LAFB / LPFB)",
    explanation_en:
      "LAFB: marked left axis deviation (−45° to −90°), qR in I/aVL, rS in II/III/aVF. LPFB: right axis deviation (>+90°) after excluding RVH/PE. Combined with RBBB it forms Bifascicular Block.",
    explanation_ar:
      "LAFB: انحراف محور أيسر شديد (−45° إلى −90°) مع qR في I/aVL وrS في II/III/aVF. LPFB: انحراف محور أيمن (>+90°) بعد استبعاد تضخم البطين الأيمن/الجلطة الرئوية. مع RBBB يشكّل إحصاراً ثنائي الحزم.",
    inv_en:
      "Screen specifically for co-existing RBBB and PR prolongation (bifascicular / 'trifascicular' pattern), compare with prior ECG, and check Echo if structural heart disease suspected.",
    inv_ar:
      "افحص بدقة وجود RBBB مصاحب واستطالة PR، وقارن بتخطيط سابق، واطلب إيكو عند الاشتباه بمرض عضوي بالقلب.",
    ttt_en:
      "Isolated chronic LAFB needs no treatment. If combined with RBBB + syncope/presyncope, admit for telemetry and Cardiology/EP pacing evaluation.",
    ttt_ar:
      "LAFB المزمن المعزول لا يحتاج علاجاً. أما مع RBBB وإغماء فيلزم حجز بالمراقبة وتقييم قلب لتركيب منظم ضربات.",
  }),

  LVH_STRAIN: mk({
    title_en: "Left Ventricular Hypertrophy (LVH) with Strain",
    title_ar: "تضخم البطين الأيسر مع إجهاد (LVH with Strain)",
    explanation_en:
      "High QRS voltage (Sokolow-Lyon/Cornell) with asymmetric downsloping ST depression and T-wave inversion in lateral leads (I, aVL, V5–V6). In competitive athletes, raises strong suspicion for HCM.",
    explanation_ar:
      "جهد مرتفع لموجات QRS مع انخفاض غير متماثل في ST وانقلاب T في I وaVL وV5–V6. ينتج عن الضغط المزمن أو ضيق الأورطي، وفي الرياضيين يرفع الاشتباه بـ HCM.",
    inv_en:
      "Compare with prior ECGs, check Troponin if acute chest pain (to distinguish chronic strain from NSTEMI), BP profile, renal function/urinalysis, and Transthoracic Echocardiogram.",
    inv_ar:
      "قارن بتخطيط سابق، اطلب تروبونين مع ألم الصدر الحاد للتمييز عن NSTEMI، وظائف كلى، وإيكو قلب لقياس سمك الجدار والصمام الأورطي.",
    ttt_en:
      "Optimize long-term antihypertensive therapy (ACEi/ARB/CCB) and treat underlying valvular disease. If competitive athlete with suspected HCM, restrict competitive sports pending Echo/Cardiac MRI.",
    ttt_ar:
      "اضبط ضغط الدم المزمن وعالج أي مرض صمامي مصاحب. وفي الرياضي المشتبه بـ HCM أوقف التمرين التنافسي مؤقتاً لحين عمل الإيكو.",
  }),

  PACEMAKER_RHYTHM: mk({
    title_en: "Paced Rhythm (Cardiac Pacemaker)",
    title_ar: "نظم منظم ضربات القلب (Pacemaker Rhythm)",
    explanation_en:
      "Pacing spikes preceding atrial P waves (AAI), wide LBBB-like ventricular complexes (VVI/RV apical), or both (DDD). Use Modified Sgarbossa criteria to screen for acute MI in RV-paced beats.",
    explanation_ar:
      "إشارات كهربائية (Spikes) تسبق P أو QRS عريضة بنمط يشبه LBBB. تُستخدم معايير سجارْبوسا المعدّلة لاكتشاف الجلطة الحادة في النظم المحفّز بطينياً.",
    inv_en:
      "Check capture & sensing on rhythm strip, draw BMP (K+/Mg2+ — hyperkalemia causes failure to capture), Chest X-ray (lead integrity/position), and formal pacemaker interrogation.",
    inv_ar:
      "تأكد من الالتقاط والاستشعار، اسحب شوارد (ارتفاع البوتاسيوم يسبب فشل الالتقاط)، أشعة صدر لفحص الأسلاك، وفحص جهاز المنظم.",
    ttt_en:
      "Normal capture/sensing needs no acute change. Failure to capture/sense or symptomatic bradycardia → apply transcutaneous pacing pads, correct K+, and call EP/Cardiology urgently.",
    ttt_ar:
      "إذا كان الالتقاط والاستشعار سليمين لا حاجة لتدخل طارئ. مع فشل الالتقاط أو بطء نبض بأعراض: منظم خارجي، تصحيح البوتاسيوم، واستدعاء كهرباء القلب.",
  }),

  PVC_BIGEMINY: mk({
    title_en: "Premature Ventricular Contractions (PVCs / Bigeminy)",
    title_ar: "انقباضات بطينية مبكرة (PVCs / Bigeminy)",
    explanation_en:
      "Premature wide (>120 ms) bizarre QRS without preceding P waves and discordant ST-T, followed by a compensatory pause. Alternating normal beats and PVCs defines bigeminy.",
    explanation_ar:
      "مركبات QRS مبكرة وعريضة (>120 مللي ثانية) بدون موجة P سابقة مع وقفة تعويضية. تبادل نبضة طبيعية مع PVC يشكّل Bigeminy.",
    inv_en:
      "Check BMP (K+, Mg2+, Ca2+), hs-Troponin (rule out ischemia), TSH, digoxin level if applicable, and measure QTc to rule out R-on-T Torsades risk.",
    inv_ar:
      "افحص الشوارد (K+, Mg2+, Ca2+)، تروبونين لاستبعاد الإقفار، TSH، مستوى ديجوكسين، وطول QTc لاستبعاد خطر R-on-T.",
    ttt_en:
      "Replete K+ (>4.0 mEq/L) and Mg2+ (>2.0 mg/dL), treat ischemia/hypoxia, and stop sympathomimetics. Beta-blocker if symptomatic frequent PVCs; avoid Class IC antiarrhythmics in structural heart disease.",
    ttt_ar:
      "عوّض البوتاسيوم (>4.0) والمغنيسيوم (>2.0)، عالج الإقفار أو نقص الأكسجين، ويمكن استخدام حاصر بيتا إذا كانت الأعراض مزعجة.",
  }),

  LONG_QT: mk({
    title_en: "Prolonged QT / QTc Interval",
    title_ar: "استطالة فترة QT / QTc",
    explanation_en:
      "Delayed ventricular repolarization (QTc >450 ms men, >460 ms women; critical >500 ms) risking Torsades de Pointes. Use Fridericia formula when HR >100 or <50 bpm.",
    explanation_ar:
      "تأخر إعادة استقطاب البطينين (QTc >450 للرجال أو >460 للنساء، وحرج >500 مللي ثانية) مما يرفع خطر Torsades de Pointes. معادلة Fridericia أدق عند HR >100 أو <50.",
    inv_en:
      "Stat BMP (K+, Mg2+, ionized Ca2+), full medication reconciliation (macrolides, fluoroquinolones, haloperidol, ondansetron, amiodarone, methadone), and continuous telemetry.",
    inv_ar:
      "شوارد عاجلة (K+, Mg2+, Ca2+)، مراجعة شاملة لكل الأدوية المُطيلة لـ QT، ومراقبة قلبية مستمرة.",
    ttt_en:
      "Stop all QT-prolonging drugs immediately. Aggressively replete K+ (>4.0) and Mg2+ (>2.0). If QTc >500 ms or Torsades occurs: IV Magnesium Sulfate 2 g over 2–5 min; defibrillate if pulseless/unstable.",
    ttt_ar:
      "أوقف فوراً كل الأدوية المطيلة لـ QT، وعوّض البوتاسيوم والمغنيسيوم. إذا تجاوز QTc 500 مللي ثانية أو حدث Torsades: سلفات مغنيسيوم 2 جم وريدياً، وصدمة فورية إذا كان غير مستقر.",
  }),

  TRIFASCICULAR_BLOCK_PATTERN: mk({
    title_en: "Bifascicular Block + 1st-Degree AVB ('Trifascicular' Pattern)",
    title_ar: "إحصار ثنائي الحزم + إحصار درجة أولى (نمط ثلاثي الحزم)",
    explanation_en:
      "RBBB + LAFB/LPFB + 1st-Degree AV Block (PR >200 ms). Often called 'trifascicular block', though per the 2009 AHA/ACCF/HRS statement this usually reflects bifascicular block plus AV-node or common His-bundle delay rather than true anatomical trifascicular block (which requires His-bundle recording).",
    explanation_ar:
      "اجتماع RBBB مع LAFB/LPFB وإحصار أذيني بطيني درجة أولى (PR >200 مللي ثانية). يُسمى أحياناً «إحصار ثلاثي الحزم»، لكن وفق بيان AHA/ACCF/HRS (2009) هو غالباً إحصار ثنائي الحزم مع تأخر في العقدة الأذينية البطينية أو حزمة هيس ولا يُثبت الإحصار الثلاثي الحقيقي إلا بدراسة حزمة هيس.",
    inv_en:
      "Compare with prior ECGs, check BMP (K+/Mg2+), hs-Troponin if ischemic symptoms, Echo for structural disease, and Holter/EP study if syncope or presyncope is present.",
    inv_ar:
      "قارن بتخطيطات سابقة، شوارد (K+/Mg2+)، تروبونين مع أعراض إقفار، إيكو قلب، وهولتر/دراسة كهرباء قلب في حال وجود إغماء أو دوخة.",
    ttt_en:
      "Asymptomatic: avoid AV-nodal blockers when possible and arrange outpatient Cardiology follow-up. With syncope, presyncope, or alternating BBB: admit on telemetry with transcutaneous pads ready for urgent pacemaker evaluation.",
    ttt_ar:
      "بدون أعراض: تجنّب مثبطات العقدة الأذينية البطينية قدر الإمكان ومتابعة قلب. مع إغماء أو دوخة شديدة: حجز بالمراقبة وتقييم عاجل لتركيب منظم ضربات.",
  }),

  BIFASCICULAR_BLOCK: mk({
    title_en: "Bifascicular Block (RBBB + Fascicular Block)",
    title_ar: "إحصار ثنائي الحزم (RBBB + LAFB/LPFB)",
    explanation_en:
      "Combined Right Bundle Branch Block (QRS ≥120 ms, rsR' in V1–V2) plus Left Anterior or Left Posterior Fascicular Block, leaving conduction dependent on one remaining fascicle.",
    explanation_ar:
      "اجتماع إحصار الحزمة اليمنى (RBBB) مع إحصار الحُزيمة الأمامية أو الخلفية اليسرى، مما يجعل التوصيل البطيني معتمداً على حُزيمة واحدة متبقية.",
    inv_en:
      "Check PR interval for concomitant 1st-degree AVB, compare with prior ECG, check Troponin if chest pain (new RBBB + LAFB = high-risk proximal LAD), and Echo for structural disease.",
    inv_ar:
      "افحص PR لاحتمال وجود إحصار درجة أولى، قارن بتخطيط سابق، اطلب تروبونين مع ألم الصدر (RBBB + LAFB جديد = خطورة LAD قريب)، وإيكو قلب.",
    ttt_en:
      "Chronic asymptomatic bifascicular block needs routine follow-up only. If syncope or acute ischemic chest pain is present, admit for telemetry and urgent Cardiology evaluation.",
    ttt_ar:
      "الحالة المزمنة بدون أعراض تكتفي بالمتابعة الدورية. مع إغماء أو ألم صدري حاد: حجز بالمراقبة وتقييم قلب عاجل.",
  }),
};

export function lookupClinicalEntry(code: string, labelHint?: string): EcgClinicalEntry | null {
  const key = String(code || "").toUpperCase();
  if (ECG_CLINICAL_LIBRARY[key]) return ECG_CLINICAL_LIBRARY[key];
  const hint = `${key} ${String(labelHint || "").toUpperCase()}`;
  if (hint.includes("INFERIOR") || hint.includes("POSTERIOR")) return ECG_CLINICAL_LIBRARY.INFERIOR_POSTERIOR_MI;
  if (hint.includes("LATERAL")) return ECG_CLINICAL_LIBRARY.LATERAL_MI;
  if (hint.includes("STEMI") || hint.includes("ANTERIOR") || hint.includes("DE_WINTER")) return ECG_CLINICAL_LIBRARY.ANTERIOR_SEPTAL_MI;
  if (hint.includes("NSTEMI") || hint.includes("ISCHEMIA")) return ECG_CLINICAL_LIBRARY.NSTEMI_ISCHEMIA;
  if (hint.includes("ATRIAL_FIBRILLATION") || hint.includes("AFIB")) return ECG_CLINICAL_LIBRARY.ATRIAL_FIBRILLATION;
  if (hint.includes("FLUTTER")) return ECG_CLINICAL_LIBRARY.ATRIAL_FLUTTER;
  if (hint.includes("SVT")) return ECG_CLINICAL_LIBRARY.SVT_TACHYCARDIA;
  if (hint.includes("BRADY")) return ECG_CLINICAL_LIBRARY.SINUS_BRADYCARDIA;
  if (hint.includes("FIRST_DEGREE") || hint.includes("1ST")) return ECG_CLINICAL_LIBRARY.FIRST_DEGREE_AV_BLOCK;
  if (hint.includes("HIGH_GRADE") || hint.includes("COMPLETE_HEART") || hint.includes("MOBITZ")) return ECG_CLINICAL_LIBRARY.HIGH_GRADE_AV_BLOCK;
  if (hint.includes("RBBB")) return ECG_CLINICAL_LIBRARY.RBBB;
  if (hint.includes("LBBB")) return ECG_CLINICAL_LIBRARY.LBBB;
  if (hint.includes("FASCICULAR") || hint.includes("LAFB") || hint.includes("LPFB")) return ECG_CLINICAL_LIBRARY.FASCICULAR_BLOCK;
  if (hint.includes("LVH") || hint.includes("HYPERTROPHY")) return ECG_CLINICAL_LIBRARY.LVH_STRAIN;
  if (hint.includes("PACEMAKER") || hint.includes("PACED")) return ECG_CLINICAL_LIBRARY.PACEMAKER_RHYTHM;
  if (hint.includes("PVC") || hint.includes("BIGEMINY")) return ECG_CLINICAL_LIBRARY.PVC_BIGEMINY;
  if (hint.includes("QT")) return ECG_CLINICAL_LIBRARY.LONG_QT;
  if (hint.includes("NORMAL_SINUS")) return ECG_CLINICAL_LIBRARY.NORMAL_SINUS_RHYTHM;
  return null;
}

/**
 * Returns dedicated Investigations (Inv) and Treatment (TTT) for any finding code
 * (works for AI classes, ecgRules codes, Vitals emergencies, and KB cases).
 */
export function getClinicalInvTtt(
  code: string,
  fallbackActionEn: string,
  fallbackActionAr: string,
  labelHint?: string
): { inv_en: string; inv_ar: string; ttt_en: string; ttt_ar: string } {
  const entry = lookupClinicalEntry(code, labelHint);
  if (entry) {
    return { inv_en: entry.inv_en, inv_ar: entry.inv_ar, ttt_en: entry.ttt_en, ttt_ar: entry.ttt_ar };
  }
  const c = String(code || "").toUpperCase();
  if (c.includes("HYPERKALEMIA") || c.includes("DKA")) {
    return {
      inv_en: "Stat VBG/ABG, serum K+, BMP, glucose, serum ketones, anion gap, and serial ECGs every 10–15 min.",
      inv_ar: "غازات دم وريدية عاجلة (VBG)، بوتاسيوم، وظائف كلى، سكر، كيتونات، فجوة أنيونية، وتخطيط متكرر كل 10–15 دقيقة.",
      ttt_en: fallbackActionEn,
      ttt_ar: fallbackActionAr,
    };
  }
  if (c.includes("PE") || c.includes("HYPOX") || c.includes("PULMONARY")) {
    return {
      inv_en: "ABG, D-dimer / CT Pulmonary Angiography (CTPA), bedside Echo (RV strain), Chest X-ray, Troponin, and BNP.",
      inv_ar: "غازات دم شريانية (ABG)، D-dimer أو أشعة مقطعية بالصبغة لشرايين الرئة (CTPA)، إيكو بجانب السرير، أشعة صدر، وتروبونين.",
      ttt_en: fallbackActionEn,
      ttt_ar: fallbackActionAr,
    };
  }
  if (c.includes("SEPSIS") || c.includes("SEPTIC") || c.includes("FEVER")) {
    return {
      inv_en: "CBC, CRP/Procalcitonin, serum lactate, 2 sets blood cultures, BMP, LFTs, urinalysis, and Chest X-ray.",
      inv_ar: "صورة دم، CRP/Procalcitonin، لاكتات، مزرعتا دم، وظائف كلى وكبد، تحليل بول، وأشعة صدر.",
      ttt_en: fallbackActionEn,
      ttt_ar: fallbackActionAr,
    };
  }
  if (c.includes("BRUGADA")) {
    return {
      inv_en: "High precordial ECG (V1–V2 placed in 2nd–3rd ICS), core temperature, BMP (K+/Ca2+), toxicology/drug review, and EP consult.",
      inv_ar: "تخطيط بأقطاب صدرية عالية (V1–V2 في الفراغ الثاني/الثالث)، قياس الحرارة، شوارد، مراجعة الأدوية، واستشارة كهرباء قلب.",
      ttt_en: fallbackActionEn,
      ttt_ar: fallbackActionAr,
    };
  }
  if (c.includes("TAMPONADE")) {
    return {
      inv_en: "Immediate bedside POCUS / Echocardiography (pericardial effusion + RV diastolic collapse), Chest X-ray, Troponin, CRP.",
      inv_ar: "إيكو فوري بجانب السرير (انصباب تأموري + انخماص البطين الأيمن)، أشعة صدر، تروبونين، وCRP.",
      ttt_en: fallbackActionEn,
      ttt_ar: fallbackActionAr,
    };
  }
  return {
    inv_en: "Repeat 12-lead ECG (compare with prior), BMP (K+/Mg2+/Ca2+), hs-Troponin, CBC, and bedside Echo / Chest X-ray as clinically indicated.",
    inv_ar: "إعادة تخطيط 12-lead ومقارنته بسابق، شوارد (K+/Mg2+/Ca2+)، تروبونين عالي الحساسية، صورة دم، وإيكو/أشعة صدر حسب الحالة.",
    ttt_en: fallbackActionEn,
    ttt_ar: fallbackActionAr,
  };
}

export function detectCompositeConductionSyndrome(opts: {
  hasRbbb: boolean;
  hasFascicular: boolean;
  hasFirstDegAvb: boolean;
  syncopeOrUnstable: boolean;
  chestPainOrShock: boolean;
}): { code: string; urgency: "CRITICAL" | "URGENT" | "ABNORMAL"; entry: EcgClinicalEntry } | null {
  if (opts.hasRbbb && opts.hasFascicular && opts.hasFirstDegAvb) {
    return {
      code: "TRIFASCICULAR_BLOCK_PATTERN",
      urgency: opts.syncopeOrUnstable ? "CRITICAL" : "URGENT",
      entry: ECG_CLINICAL_LIBRARY.TRIFASCICULAR_BLOCK_PATTERN,
    };
  }
  if (opts.hasRbbb && opts.hasFascicular) {
    return {
      code: "BIFASCICULAR_BLOCK",
      urgency: opts.syncopeOrUnstable || opts.chestPainOrShock ? "URGENT" : "ABNORMAL",
      entry: ECG_CLINICAL_LIBRARY.BIFASCICULAR_BLOCK,
    };
  }
  return null;
}
