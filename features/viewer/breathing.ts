'use client';

import type { Material } from "three";

import type { ManifestOrgan } from "@/lib/schemas";

import { CARRY_ATTRIBUTE } from "./breathCarry";

/**
 * Breathing & Respiratory Auscultation Engine (`breathing.ts`).
 *
 * Includes:
 * 1. Biomechanical 3D respiratory shader (diaphragm dome descent, ribcage bucket/pump-handle
 *    expansion, lung base lengthening, visceral displacement).
 * 2. Comprehensive clinical & pathological breath patterns (11 conditions across 4 clinical groups)
 *    with bilingual descriptions, auscultation sites, I:E ratios, and clinical pearls.
 * 3. Real-time Web Audio Stethoscope Auscultation Synthesizer (`primeBreathSound`, `updateBreathSoundFrame`,
 *    `stopBreathSound`) modeling vesicular, bronchial/tubular, polyphonic asthma wheezes, COPD rhonchi,
 *    inspiratory stridor, fine late-inspiratory Velcro crackles (IPF), coarse pulmonary edema crackles,
 *    and pleural friction rub.
 */

export const BREATH_ROLE = {
  none: 0,
  chestWall: 1,
  lung: 2,
  diaphragm: 3,
  viscera: 4,
  belly: 5,
  heart: 6,
  vessel: 7,
} as const;
export type BreathRole = (typeof BREATH_ROLE)[keyof typeof BREATH_ROLE];

const name = (organ: Pick<ManifestOrgan, "name_en">) => organ.name_en.toLowerCase();
const has = (text: string, fragments: readonly string[]) => fragments.some((f) => text.includes(f));

const CHEST_MUSCLES = [
  "intercostal", "pectoralis", "serratus anterior", "subclavius", "transversus thoracis",
  "levatores costarum", "serratus posterior",
];
const BELLY_MUSCLES = [
  "rectus abdominis", "external oblique", "internal oblique", "transversus abdominis",
  "pyramidalis", "rectus sheath", "linea alba",
];
const CHEST_SKIN = [
  "pectoral region", "presternal", "mammary", "infraclavicular", "lateral region of thorax",
];
const BELLY_SKIN = [
  "epigastric", "umbilic", "hypochondriac", "lateral region of abdomen", "hypogastric",
];
const RIBCAGE = ["rib", "costal", "stern", "xiphoid", "manubri", "costo", "chondro"];
const UNDER_DIAPHRAGM = [
  "liver", "hepat", "stomach", "spleen", "splen", "pancrea", "gallbladder", "duoden",
  "kidney", "suprarenal", "porta hepatis",
];

export function breathRole(
  organ: Pick<ManifestOrgan, "system" | "name_en" | "path">,
  heartChamber = false,
): BreathRole {
  const text = name(organ);
  const marking = organ.path?.[organ.path.length - 1] === "Muscular insertions";
  switch (organ.system) {
    case "respiratory":
      return has(text, ["lung", "bronch", "pleura"]) ? BREATH_ROLE.lung : BREATH_ROLE.none;
    case "muscular":
      if (marking) return BREATH_ROLE.none;
      if (text === "diaphragm" || text.startsWith("diaphragmatic fascia")) return BREATH_ROLE.diaphragm;
      if (has(text, CHEST_MUSCLES)) return BREATH_ROLE.chestWall;
      if (has(text, BELLY_MUSCLES)) return BREATH_ROLE.belly;
      return BREATH_ROLE.none;
    case "skeletal":
    case "articular":
      return has(text, RIBCAGE) ? BREATH_ROLE.chestWall : BREATH_ROLE.none;
    case "regional":
      if (has(text, CHEST_SKIN)) return BREATH_ROLE.chestWall;
      if (has(text, BELLY_SKIN)) return BREATH_ROLE.belly;
      return BREATH_ROLE.none;
    case "digestive":
    case "renal":
    case "endocrine":
    case "lymphatic":
      return has(text, UNDER_DIAPHRAGM) && !text.includes("node") ? BREATH_ROLE.viscera : BREATH_ROLE.none;
    case "cardiovascular":
      if (heartChamber) return BREATH_ROLE.heart;
      if (organ.path?.includes("Pulmonary vessels")) return BREATH_ROLE.lung;
      return BREATH_ROLE.vessel;
    default:
      return BREATH_ROLE.none;
  }
}

export type BreathPatternId =
  | "quiet"
  | "deep"
  | "bronchial"
  | "asthma_wheeze"
  | "copd_rhonchi"
  | "stridor"
  | "fine_crackles"
  | "coarse_crackles"
  | "pleural_rub"
  | "cheyne_stokes"
  | "kussmaul";

export const BREATH_GROUPS = [
  "Normal & Bronchial Breath Sounds · الأصوات الطبيعية والقصبية",
  "Airway Obstruction (Wheezes, Rhonchi & Stridor) · انسداد الممرات والصفير",
  "Parenchymal Crackles / Rales · الفرقعات الرئوية وأمراض النسيج",
  "Pleural & Respiratory Rhythm Disorders · الاحتكاك الجنبي واضطرابات النمط",
] as const;

export type BreathGroup = (typeof BREATH_GROUPS)[number];

export interface BreathPattern {
  id: BreathPatternId;
  label: string;
  label_ar: string;
  group: BreathGroup;
  /** Breaths a minute, as said on the panel. */
  rate: string;
  /** Insp:Exp ratio & auscultation site summary */
  auscultation_site: string;
  /** Seconds for one breath. */
  period: number;
  /** The part of it spent breathing in. */
  inhale: number;
  /** How much bigger than a quiet breath. */
  depth: number;
  /** Respiratory sinus arrhythmia swing factor */
  sinusSwing: number;
  /** English clinical description */
  what: string;
  /** Arabic clinical & auscultatory description */
  what_ar: string;
  /** Real clinical YouTube auscultation video ID */
  youtubeVideoId: string;
  /** Start second in the YouTube recording */
  youtubeStart: number;
  /** End second in the YouTube recording */
  youtubeEnd: number;
  /** Direct YouTube watch URL */
  youtubeUrl: string;
}

export const BREATH_PATTERNS: Record<BreathPatternId, BreathPattern> = {
  quiet: {
    id: "quiet",
    youtubeVideoId: "JFWMJGtmG5E",
    youtubeStart: 49,
    youtubeEnd: 58,
    youtubeUrl: "https://www.youtube.com/watch?v=JFWMJGtmG5E&t=46s",
    label: "Normal vesicular breathing (Quiet)",
    label_ar: "تنفس حويصلي طبيعي هادئ (Normal Vesicular)",
    group: "Normal & Bronchial Breath Sounds · الأصوات الطبيعية والقصبية",
    rate: "12 / min · I:E 1:2",
    auscultation_site: "All peripheral lung fields (bases & axillary)",
    period: 5,
    inhale: 0.4,
    depth: 1,
    sinusSwing: 0.05,
    what:
      "Soft, low-pitched rustling (150–450 Hz) heard throughout inspiration and fading in the first third of expiration with no silent gap.",
    what_ar:
      "صوت حفيف ناعم منخفض النبرة (150–450 هرتز) يُسمع طوال فترة الشهيق وفي الثلث الأول فقط من الزفير دون فاصل صامت بينهما.",
  },
  deep: {
    id: "deep",
    youtubeVideoId: "JFWMJGtmG5E",
    youtubeStart: 36,
    youtubeEnd: 45,
    youtubeUrl: "https://www.youtube.com/watch?v=JFWMJGtmG5E&t=33s",
    label: "Deep physiological breathing (Sighing / Exercise)",
    label_ar: "تنفس فسيولوجي عميق (Deep Vesicular)",
    group: "Normal & Bronchial Breath Sounds · الأصوات الطبيعية والقصبية",
    rate: "6 / min · High Tidal Vol",
    auscultation_site: "Bilateral lung zones · Diaphragmatic excursion",
    period: 10,
    inhale: 0.45,
    depth: 2.6,
    sinusSwing: 0.11,
    what:
      "Full diaphragmatic dome descent and bucket-handle rib elevation with amplified laminar vesicular airflow and marked sinus arrhythmia.",
    what_ar:
      "هبوط كامل لقبة الحجاب الحاجز وارتفاع الأضلاع مع صوت هواء حويصلي واضح وتغير فسيولوجي ملحوظ في سرعة ضربات القلب (Sinus Arrhythmia).",
  },
  bronchial: {
    id: "bronchial",
    youtubeVideoId: "JFWMJGtmG5E",
    youtubeStart: 23,
    youtubeEnd: 32,
    youtubeUrl: "https://www.youtube.com/watch?v=JFWMJGtmG5E&t=20s",
    label: "Bronchial (Tubular) breath sounds — Lobar Pneumonia",
    label_ar: "تنفس قصبي أنبوبي — تكثف رئوي فصي (Lobar Consolidation)",
    group: "Normal & Bronchial Breath Sounds · الأصوات الطبيعية والقصبية",
    rate: "18 / min · I:E 1:1.5",
    auscultation_site: "Over consolidated lobe (pathological) or manubrium",
    period: 3.3,
    inhale: 0.38,
    depth: 0.9,
    sinusSwing: 0.04,
    what:
      "Loud, harsh, high-pitched hollow tubular sound (500–1100 Hz) with a distinct silent pause between inspiration and a louder, longer expiration. Classic in lobar pneumonia consolidation.",
    what_ar:
      "صوت أنبوبي مجوف عالي النبرة وشديد الوضوح مع فاصل صامت مميز بين الشهيق والزفير، ويكون الزفير أعلى وأطول من الشهيق بسبب انتقال الصوت عبر الفص الرئوي المتكثف في الالتهاب الرئوي الفصي.",
  },
  asthma_wheeze: {
    id: "asthma_wheeze",
    youtubeVideoId: "OXe5-zk3dls",
    youtubeStart: 0,
    youtubeEnd: 45,
    youtubeUrl: "https://youtube.com/shorts/OXe5-zk3dls?si=aJL1AfE0RdeI-vJy",
    label: "Expiratory polyphonic wheezes — Bronchial Asthma",
    label_ar: "صفير زفيري متعدد النغمات — الربو الشعبي الحاد (Asthma)",
    group: "Airway Obstruction (Wheezes, Rhonchi & Stridor) · انسداد الممرات والصفير",
    rate: "22 / min · Prolonged Exp (I:E 1:3)",
    auscultation_site: "Diffuse bilateral lung fields (expiratory phase)",
    period: 2.8,
    inhale: 0.28,
    depth: 1.15,
    sinusSwing: 0.03,
    what:
      "Continuous musical high-pitched whistling chords (380–720 Hz) heard predominantly during a prolonged expiration due to diffuse bronchospasm and airway narrowing.",
    what_ar:
      "أصوات صفير موسيقية متصلة عالية النبرة (380–720 هرتز) متعددة الترددات تُسمع خلال مرحلة الزفير المتطاولة نتيجة تشنج العضلات الملساء وضيق الشعب الهوائية في نوبات الربو.",
  },
  copd_rhonchi: {
    id: "copd_rhonchi",
    youtubeVideoId: "U8byn2NT_lo",
    youtubeStart: 44,
    youtubeEnd: 53,
    youtubeUrl: "https://www.youtube.com/watch?v=U8byn2NT_lo&t=41s",
    label: "Low-pitched sonorous rhonchi — COPD / Chronic Bronchitis",
    label_ar: "خراخر وغطيط منخفض النبرة — السدة الرئوية المزمنة (COPD)",
    group: "Airway Obstruction (Wheezes, Rhonchi & Stridor) · انسداد الممرات والصفير",
    rate: "20 / min · Prolonged Exp",
    auscultation_site: "Central & mainstem bronchi (often clears with cough)",
    period: 3.0,
    inhale: 0.30,
    depth: 0.85,
    sinusSwing: 0.03,
    what:
      "Low-pitched snoring/rattling continuous vibrations (140–220 Hz) during expiration and inspiration caused by thick mucus secretions fluttering in larger airways.",
    what_ar:
      "أصوات غطيط أو شخير منخفضة النبرة (140–220 هرتز) ناتجة عن اهتزاز الإفرازات المخاطية الكثيفة داخل القصبات الهوائية الكبيرة في التهاب الشعب المزمن والسدة الرئوية، وقد تتغير بالكحة.",
  },
  stridor: {
    id: "stridor",
    youtubeVideoId: "JSdEK79J4dw",
    youtubeStart: 45,
    youtubeEnd: 60,
    youtubeUrl: "https://www.youtube.com/watch?v=JSdEK79J4dw&t=45s",
    label: "Inspiratory stridor — Upper Airway / Laryngeal Obstruction",
    label_ar: "صرير شهيقي حاد — انسداد الحنجرة أو الرغامي (Stridor)",
    group: "Airway Obstruction (Wheezes, Rhonchi & Stridor) · انسداد الممرات والصفير",
    rate: "24 / min · Inspiratory Emergency",
    auscultation_site: "Anterior neck / Trachea & suprasternal notch",
    period: 2.5,
    inhale: 0.52,
    depth: 1.2,
    sinusSwing: 0.02,
    what:
      "Loud, harsh, high-pitched monophonic crowing sound (600–850 Hz) during inspiration caused by turbulent flow through a critically narrowed larynx or extrathoracic trachea (croup, foreign body, post-thyroidectomy hematoma).",
    what_ar:
      "صوت صرير حاد أحادي النغمة عالي التردد (600–850 هرتز) يُسمع أثناء الشهيق فوق الرقبة والقصبة الهوائية، ويُعد علامة طوارئ خطيرة على انسداد الحنجرة أو الرغامي (مثل التجمع الدموي بعد استئصال الدرقية أو شلل الأحبال الصوتية).",
  },
  fine_crackles: {
    id: "fine_crackles",
    youtubeVideoId: "U8byn2NT_lo",
    youtubeStart: 84,
    youtubeEnd: 97,
    youtubeUrl: "https://www.youtube.com/watch?v=U8byn2NT_lo&t=81s",
    label: "Fine late-inspiratory crackles (Velcro) — Pulmonary Fibrosis",
    label_ar: "فرقعات ناعمة بنهاية الشهيق (Velcro) — تليف الرئة البيني (IPF)",
    group: "Parenchymal Crackles / Rales · الفرقعات الرئوية وأمراض النسيج",
    rate: "22 / min · Restrictive (Shallow)",
    auscultation_site: "Bilateral posterobasal lung zones (unaffected by cough)",
    period: 2.7,
    inhale: 0.42,
    depth: 0.65,
    sinusSwing: 0.03,
    what:
      "Brief, dry, high-pitched non-musical popping sounds (750–1400 Hz, like pulling Velcro apart) in mid-to-late inspiration as stiff, fibrotic peripheral alveoli suddenly snap open.",
    what_ar:
      "فرقعات جافة ناعمة وعالية النبرة تشبه صوت فتح شريط الفيلكرو (Velcro Rales) وتظهر في النصف الثاني ونهاية الشهيق في قاعدتي الرئتين نتيجة الانفتاح المفاجئ للحويصلات الهوائية المتليفة، ولا تتغير بالكحة.",
  },
  coarse_crackles: {
    id: "coarse_crackles",
    youtubeVideoId: "U8byn2NT_lo",
    youtubeStart: 66,
    youtubeEnd: 80,
    youtubeUrl: "https://www.youtube.com/watch?v=U8byn2NT_lo&t=63s",
    label: "Coarse bubbling crackles — Acute Pulmonary Edema / Pneumonia",
    label_ar: "فرقعات خشنة رطبة وفقاعية — وذمة رئوية حادة / توسع الشعب",
    group: "Parenchymal Crackles / Rales · الفرقعات الرئوية وأمراض النسيج",
    rate: "24 / min · Early Insp & Exp",
    auscultation_site: "Bilateral dependent lung zones & mid-zones",
    period: 2.5,
    inhale: 0.40,
    depth: 0.95,
    sinusSwing: 0.02,
    what:
      "Loud, low-pitched, wet bubbling discontinuous pops (200–450 Hz) beginning in early inspiration and recurring in expiration as air passes through pulmonary edema fluid or copious secretions.",
    what_ar:
      "فرقعات خشنة رطبة ومنخفضة النبرة تبدأ منذ بداية الشهيق وتمتد للزفير نتيجة مرور الهواء عبر السوائل المتراكمة داخل الحويصلات والشعب في حالات الوذمة الرئوية الحادة (فشل القلب الأيسر) أو توسع الشعب الهوائية.",
  },
  pleural_rub: {
    id: "pleural_rub",
    youtubeVideoId: "U8byn2NT_lo",
    youtubeStart: 105,
    youtubeEnd: 117,
    youtubeUrl: "https://www.youtube.com/watch?v=U8byn2NT_lo&t=98s",
    label: "Pleural friction rub — Fibrinous Pleurisy / Pulmonary Embolism",
    label_ar: "احتكاك جنبي خشن — التهاب الجنب الفبريني / الانصمام الرئوي",
    group: "Pleural & Respiratory Rhythm Disorders · الاحتكاك الجنبي واضطرابات النمط",
    rate: "18 / min · Biphasic (Late Insp + Early Exp)",
    auscultation_site: "Anterolateral & basal chest wall (disappears on breath-hold)",
    period: 3.3,
    inhale: 0.42,
    depth: 0.75,
    sinusSwing: 0.03,
    what:
      "Dry, low-pitched grating or leathery creaking sound (140–360 Hz, like walking on fresh snow) heard in both late inspiration and early expiration as inflamed visceral and parietal pleura slide against each other.",
    what_ar:
      "صوت احتكاك جلدي خشن أو صرير منخفض النبرة يُسمع على مرحلتين (نهاية الشهيق وبداية الزفير) نتيجة احتكاك طبقتي غشاء الجنب الملتهبتين والمغطاتين بالفيبرين، ويتوقف تماماً عند حبس النفس (بخلاف الاحتكاك التأموري).",
  },
  cheyne_stokes: {
    id: "cheyne_stokes",
    youtubeVideoId: "ViGjOiPE2mY",
    youtubeStart: 145,
    youtubeEnd: 155,
    youtubeUrl: "https://www.youtube.com/watch?v=ViGjOiPE2mY&t=145s",
    label: "Cheyne–Stokes respiration — Severe Heart Failure / Brainstem Stroke",
    label_ar: "تنفس شاين-ستوكس الدوري — فشل القلب المتقدم / إصابات جذع المخ",
    group: "Pleural & Respiratory Rhythm Disorders · الاحتكاك الجنبي واضطرابات النمط",
    rate: "Cyclical Crescendo–Decrescendo + Apnea",
    auscultation_site: "All lung fields · Central respiratory drive oscillation",
    period: 3.6,
    inhale: 0.42,
    depth: 1.5,
    sinusSwing: 0.06,
    what:
      "Periodic breathing where tidal volume progressively waxes (crescendo) then wanes (decrescendo) followed by a brief central apneic pause, driven by delayed medullary chemoreceptor feedback.",
    what_ar:
      "تنفس دوري متناوب يزداد فيه عمق التنفس تدريجياً (Crescendo) ثم يتناقص تدريجياً (Decrescendo) يعقبه توقف مؤقت للتنفس (Apnea)، ويحدث في فشل القلب الاحتقاني الشديد أو إصابات المراكز التنفسية الدماغية.",
  },
  kussmaul: {
    id: "kussmaul",
    youtubeVideoId: "ViGjOiPE2mY",
    youtubeStart: 184,
    youtubeEnd: 193,
    youtubeUrl: "https://www.youtube.com/watch?v=ViGjOiPE2mY&t=184s",
    label: "Kussmaul breathing (Air Hunger) — Diabetic Ketoacidosis (DKA)",
    label_ar: "تنفس كوسماول العميق السريع — الحماض الكيتوني السكري (DKA)",
    group: "Pleural & Respiratory Rhythm Disorders · الاحتكاك الجنبي واضطرابات النمط",
    rate: "26 / min · Deep & Rapid Hyperpnea",
    auscultation_site: "Diffuse bilateral fields · Loud sighing airflow",
    period: 2.3,
    inhale: 0.48,
    depth: 2.4,
    sinusSwing: 0.04,
    what:
      "Deep, rapid, labored sighing hyperventilation ('air hunger') blowing off CO2 to compensate for severe metabolic acidosis (DKA, uremia, or salicylate toxicity).",
    what_ar:
      "تنفس عميق وسريع جداً ('جوع الهواء') يعمل كاستجابة تعويضية لطرد غاز ثاني أكسيد الكربون في حالات الحماض الاستقلابي الشديد مثل الحماض الكيتوني السكري (DKA) أو الفشل الكلوي الحاد.",
  },
};

export const BREATH_PATTERN_LIST: readonly BreathPattern[] = Object.values(BREATH_PATTERNS);

/**
 * Envelope modulation for Cheyne-Stokes waxing-waning periodic breathing.
 */
export function cheyneStokesEnvelope(seconds: number): number {
  const macroCycle = 18;
  const phase = (((seconds % macroCycle) + macroCycle) % macroCycle) / macroCycle;
  if (phase > 0.82) return 0.04; // Apneic phase
  const activePhase = phase / 0.82;
  return Math.sin(Math.PI * activePhase);
}

/**
 * How full the lungs are `seconds` in, 0 empty to 1 full.
 */
export function breathFill(seconds: number, pattern: BreathPattern): number {
  const period = pattern.period;
  const phase = (((seconds % period) + period) % period) / period;
  const baseFill =
    phase < pattern.inhale
      ? 0.5 - 0.5 * Math.cos((Math.PI * phase) / pattern.inhale)
      : 0.5 + 0.5 * Math.cos((Math.PI * (phase - pattern.inhale)) / (1 - pattern.inhale));

  if (pattern.id === "cheyne_stokes") {
    return baseFill * cheyneStokesEnvelope(seconds);
  }
  return baseFill;
}

/** Whether `seconds` in is breathing in. */
export function breathingIn(seconds: number, pattern: BreathPattern): boolean {
  const phase = (((seconds % pattern.period) + pattern.period) % pattern.period) / pattern.period;
  return phase < pattern.inhale;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Real-Time Web Audio Stethoscope Auscultation Synthesizer for Breath Sounds
 * ═══════════════════════════════════════════════════════════════════════════ */

type AudioCtxCtor = typeof AudioContext;

let breathAudioCtx: AudioContext | null = null;
let breathMasterGain: GainNode | null = null;
let lastBreathIndex = -1;
let lastPhaseInhaling: boolean | null = null;

export function primeBreathSound(): void {
  try {
    const Ctx: AudioCtxCtor | undefined =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: AudioCtxCtor }).webkitAudioContext;
    if (!Ctx) return;
    if (!breathAudioCtx) {
      breathAudioCtx = new Ctx();
      const comp = breathAudioCtx.createDynamicsCompressor();
      comp.threshold.value = -16;
      comp.knee.value = 10;
      comp.ratio.value = 5;
      comp.attack.value = 0.005;
      comp.release.value = 0.12;

      breathMasterGain = breathAudioCtx.createGain();
      breathMasterGain.gain.value = 1.45;
      breathMasterGain.connect(comp).connect(breathAudioCtx.destination);
    }
    if (breathAudioCtx.state === "suspended") {
      void breathAudioCtx.resume();
    }
  } catch {
    // Ignore audio context errors on restricted devices
  }
}

export function triggerImmediateBreathSound(_pattern?: BreathPattern): void {
  primeBreathSound();
}

export function stopBreathSound(): void {
  lastBreathIndex = -1;
  lastPhaseInhaling = null;
}

function createNoiseBuffer(ctx: AudioContext, durationSec: number): AudioBuffer {
  const samples = Math.max(128, Math.ceil(ctx.sampleRate * durationSec));
  const buffer = ctx.createBuffer(1, samples, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  // Pink-weighted airflow noise for realistic chest-wall stethoscope timbre
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  for (let i = 0; i < samples; i++) {
    const white = Math.random() * 2 - 1;
    b0 = 0.997 * b0 + white * 0.029;
    b1 = 0.985 * b1 + white * 0.032;
    b2 = 0.950 * b2 + white * 0.048;
    data[i] = (b0 + b1 + b2 + white * 0.18) * 0.85;
  }
  return buffer;
}

function scheduleAirflowRush(
  ctx: AudioContext,
  out: AudioNode,
  duration: number,
  centerFreq: number,
  q: number,
  peakGain: number,
  attackFrac = 0.25,
  releaseFrac = 0.35,
) {
  if (duration <= 0.04 || peakGain <= 0.001) return;
  const now = ctx.currentTime;
  const src = ctx.createBufferSource();
  src.buffer = createNoiseBuffer(ctx, duration);

  const bandpass = ctx.createBiquadFilter();
  bandpass.type = "bandpass";
  bandpass.frequency.setValueAtTime(centerFreq, now);
  bandpass.Q.setValueAtTime(q, now);

  const lowpass = ctx.createBiquadFilter();
  lowpass.type = "lowpass";
  lowpass.frequency.setValueAtTime(Math.min(1800, centerFreq * 2.3), now);

  const env = ctx.createGain();
  env.gain.setValueAtTime(0.0001, now);
  env.gain.linearRampToValueAtTime(peakGain, now + duration * attackFrac);
  env.gain.setValueAtTime(peakGain * 0.88, now + duration * (1 - releaseFrac));
  env.gain.exponentialRampToValueAtTime(0.0001, now + duration);

  src.connect(bandpass).connect(lowpass).connect(env).connect(out);
  src.start(now);
  src.stop(now + duration + 0.02);
}

function scheduleWheezeOrStridor(
  ctx: AudioContext,
  out: AudioNode,
  duration: number,
  freqs: readonly number[],
  peakGain: number,
  delaySec = 0,
) {
  if (duration <= 0.05 || peakGain <= 0.001) return;
  const start = ctx.currentTime + delaySec;
  const end = start + duration;

  const env = ctx.createGain();
  env.gain.setValueAtTime(0.0001, start);
  env.gain.linearRampToValueAtTime(peakGain, start + duration * 0.18);
  env.gain.setValueAtTime(peakGain * 0.92, start + duration * 0.78);
  env.gain.exponentialRampToValueAtTime(0.0001, end);
  env.connect(out);

  freqs.forEach((f, idx) => {
    const osc = ctx.createOscillator();
    osc.type = idx % 2 === 0 ? "sine" : "triangle";
    osc.frequency.setValueAtTime(f * 1.02, start);
    osc.frequency.linearRampToValueAtTime(f * 0.96, end);

    const toneGain = ctx.createGain();
    toneGain.gain.value = idx === 0 ? 0.55 : 0.3 / idx;
    osc.connect(toneGain).connect(env);
    osc.start(start);
    osc.stop(end + 0.02);
  });
}

function scheduleCrackleTrain(
  ctx: AudioContext,
  out: AudioNode,
  startDelay: number,
  windowDuration: number,
  count: number,
  centerFreq: number,
  popDuration: number,
  gain: number,
) {
  const baseNow = ctx.currentTime + startDelay;
  for (let i = 0; i < count; i++) {
    const offset = (i / Math.max(1, count)) * windowDuration + (Math.random() - 0.5) * 0.025;
    const t0 = Math.max(ctx.currentTime, baseNow + offset);
    const src = ctx.createBufferSource();
    src.buffer = createNoiseBuffer(ctx, popDuration);

    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.setValueAtTime(centerFreq * (0.88 + Math.random() * 0.24), t0);
    bp.Q.setValueAtTime(2.4, t0);

    const popEnv = ctx.createGain();
    const popPeak = gain * (0.7 + Math.random() * 0.45);
    popEnv.gain.setValueAtTime(0.0001, t0);
    popEnv.gain.exponentialRampToValueAtTime(popPeak, t0 + popDuration * 0.2);
    popEnv.gain.exponentialRampToValueAtTime(0.0001, t0 + popDuration);

    src.connect(bp).connect(popEnv).connect(out);
    src.start(t0);
    src.stop(t0 + popDuration + 0.005);
  }
}

/**
 * Triggered per frame from `BreathDriver` when `sound` is enabled.
 * Detects the exact onset of Inspiration and Expiration for the active `BreathPatternId`
 * and schedules the corresponding physiological or pathological stethoscope sounds.
 */
export function updateBreathSoundFrame(
  clockSeconds: number,
  pattern: BreathPattern,
  soundEnabled: boolean,
): void {
  // Real clinical YouTube auscultation recordings are played directly via BreathControls.tsx
  if (pattern.youtubeVideoId) return;
  if (!soundEnabled) {
    lastBreathIndex = -1;
    lastPhaseInhaling = null;
    return;
  }
  const ctx = breathAudioCtx;
  const out = breathMasterGain;
  if (!ctx || !out || ctx.state !== "running") return;

  const period = pattern.period;
  const breathIndex = Math.floor(clockSeconds / period);
  const phase = (((clockSeconds % period) + period) % period) / period;
  const isInhaling = phase < pattern.inhale;

  if (breathIndex === lastBreathIndex && isInhaling === lastPhaseInhaling) {
    return;
  }
  lastBreathIndex = breathIndex;
  lastPhaseInhaling = isInhaling;

  const inhDur = period * pattern.inhale;
  const expDur = period * (1 - pattern.inhale);
  const csScale = pattern.id === "cheyne_stokes" ? cheyneStokesEnvelope(clockSeconds) : 1;
  if (csScale < 0.08) return; // Silent during Cheyne-Stokes apnea

  try {
    if (isInhaling) {
      // ─── INSPIRATORY PHASE AUSCULTATION ───
      switch (pattern.id) {
        case "quiet":
          scheduleAirflowRush(ctx, out, inhDur * 0.92, 240, 0.9, 0.28);
          break;
        case "deep":
          scheduleAirflowRush(ctx, out, inhDur * 0.94, 270, 0.85, 0.42);
          break;
        case "bronchial":
          // Loud hollow tubular inspiration with pause before expiration
          scheduleAirflowRush(ctx, out, inhDur * 0.78, 620, 1.8, 0.46, 0.18, 0.25);
          break;
        case "asthma_wheeze":
          scheduleAirflowRush(ctx, out, inhDur * 0.9, 310, 1.1, 0.25);
          break;
        case "copd_rhonchi":
          scheduleAirflowRush(ctx, out, inhDur * 0.9, 210, 1.0, 0.22);
          scheduleWheezeOrStridor(ctx, out, inhDur * 0.65, [155, 210], 0.12, inhDur * 0.15);
          break;
        case "stridor":
          // Loud high-pitched monophonic inspiratory stridor over trachea
          scheduleAirflowRush(ctx, out, inhDur * 0.92, 420, 1.4, 0.28);
          scheduleWheezeOrStridor(ctx, out, inhDur * 0.86, [640, 1280], 0.36, 0.04);
          break;
        case "fine_crackles":
          // Shallow inspiration + dense burst of fine Velcro crackles in late inspiration
          scheduleAirflowRush(ctx, out, inhDur * 0.9, 260, 1.0, 0.22);
          scheduleCrackleTrain(ctx, out, inhDur * 0.48, inhDur * 0.44, 18, 980, 0.006, 0.48);
          break;
        case "coarse_crackles":
          // Early-to-mid inspiratory coarse bubbling crackles
          scheduleAirflowRush(ctx, out, inhDur * 0.9, 240, 0.9, 0.28);
          scheduleCrackleTrain(ctx, out, inhDur * 0.08, inhDur * 0.65, 11, 310, 0.016, 0.56);
          break;
        case "pleural_rub":
          // Late inspiratory leathery grating rub
          scheduleAirflowRush(ctx, out, inhDur * 0.88, 230, 1.0, 0.2);
          scheduleCrackleTrain(ctx, out, inhDur * 0.52, inhDur * 0.36, 9, 220, 0.022, 0.52);
          break;
        case "cheyne_stokes":
          scheduleAirflowRush(ctx, out, inhDur * 0.9, 250, 0.9, 0.38 * csScale);
          break;
        case "kussmaul":
          scheduleAirflowRush(ctx, out, inhDur * 0.94, 320, 0.85, 0.48);
          break;
      }
    } else {
      // ─── EXPIRATORY PHASE AUSCULTATION ───
      switch (pattern.id) {
        case "quiet":
          // Normal vesicular expiration is heard only in the first third, softer & lower
          scheduleAirflowRush(ctx, out, expDur * 0.38, 190, 1.0, 0.14, 0.15, 0.65);
          break;
        case "deep":
          scheduleAirflowRush(ctx, out, expDur * 0.55, 210, 0.9, 0.24, 0.15, 0.6);
          break;
        case "bronchial":
          // Loud, high-pitched tubular expiration lasting throughout the expiratory phase
          scheduleAirflowRush(ctx, out, expDur * 0.88, 680, 1.9, 0.52, 0.15, 0.3);
          break;
        case "asthma_wheeze":
          // Prolonged expiration with musical polyphonic wheezes
          scheduleAirflowRush(ctx, out, expDur * 0.92, 340, 1.2, 0.22);
          scheduleWheezeOrStridor(ctx, out, expDur * 0.86, [410, 545, 710], 0.32, 0.05);
          break;
        case "copd_rhonchi":
          // Prolonged expiration with low-pitched snoring rhonchi
          scheduleAirflowRush(ctx, out, expDur * 0.85, 180, 1.1, 0.2);
          scheduleWheezeOrStridor(ctx, out, expDur * 0.82, [145, 195, 290], 0.34, 0.04);
          break;
        case "stridor":
          scheduleAirflowRush(ctx, out, expDur * 0.45, 240, 1.0, 0.16);
          break;
        case "fine_crackles":
          scheduleAirflowRush(ctx, out, expDur * 0.36, 195, 1.0, 0.12);
          break;
        case "coarse_crackles":
          // Expiratory bubbling crackles
          scheduleAirflowRush(ctx, out, expDur * 0.65, 210, 0.95, 0.2);
          scheduleCrackleTrain(ctx, out, expDur * 0.1, expDur * 0.55, 7, 290, 0.015, 0.42);
          break;
        case "pleural_rub":
          // Early expiratory grating rub (mirroring late inspiration)
          scheduleAirflowRush(ctx, out, expDur * 0.45, 195, 1.0, 0.15);
          scheduleCrackleTrain(ctx, out, expDur * 0.06, expDur * 0.35, 9, 210, 0.022, 0.5);
          break;
        case "cheyne_stokes":
          scheduleAirflowRush(ctx, out, expDur * 0.45, 195, 1.0, 0.2 * csScale);
          break;
        case "kussmaul":
          scheduleAirflowRush(ctx, out, expDur * 0.78, 270, 0.9, 0.36);
          break;
      }
    }
  } catch {
    // Ignore transient scheduling errors
  }
}

export const BREATH_TRAVEL = {
  diaphragm: 0.015,
  lungDown: 0.011,
  viscera: 0.012,
  chestForward: 0.005,
  chestOut: 0.0035,
  chestUp: 0.003,
  bellyForward: 0.007,
  heart: 0.005,
} as const;

export const BREATH_FILL = { value: 0 };
export const BREATH_FRAME: { value: number[] } = { value: [1.16, 1.46, 1.36] };
export const BREATH_AXIS: { value: number[] } = { value: [0, 1.32, 0.0] };

type Shader = Parameters<Material["onBeforeCompile"]>[0];

const ANCHOR = "#include <project_vertex>";
const m = (value: number) => value.toFixed(4);

export function injectBreath(shader: Shader, owner: unknown): void {
  if (!shader.vertexShader.includes(ANCHOR)) {
    throw new Error("Breathing: the expected three.js vertex chunk is missing.");
  }
  const data = (owner as { userData?: { breathRole?: number; breathSpan?: readonly [number, number] } } | undefined)
    ?.userData;
  const span = data?.breathSpan ?? [0, 1];
  shader.uniforms.uBreathFill = BREATH_FILL;
  shader.uniforms.uBreathAxis = BREATH_AXIS;
  shader.uniforms.uBreathRole = { value: data?.breathRole ?? BREATH_ROLE.none };
  shader.uniforms.uBreathSpan = { value: [span[0], span[1]] };
  shader.uniforms.uBreathFrame = BREATH_FRAME;
  if (owner && typeof owner === "object") {
    const material = owner as { defaultAttributeValues?: Record<string, number[]> };
    material.defaultAttributeValues = { ...material.defaultAttributeValues, [CARRY_ATTRIBUTE]: [0] };
  }

  shader.vertexShader =
    "uniform float uBreathFill;\nuniform vec3 uBreathAxis;\nuniform float uBreathRole;\nuniform vec2 uBreathSpan;\nuniform vec3 uBreathFrame;\n" +
    `attribute float ${CARRY_ATTRIBUTE};\n` +
    shader.vertexShader.replace(
      ANCHOR,
      `${ANCHOR}
if (uBreathFill > 0.0 && uBreathRole > 0.5) {
  vec4 breathWorld = modelMatrix * vec4(transformed, 1.0);
  vec2 breathFromAxis = vec2(breathWorld.x - uBreathAxis.x, breathWorld.z - uBreathAxis.z);
  vec2 breathOut = breathFromAxis / max(length(breathFromAxis), 1e-4);
  float breathUp = clamp((breathWorld.y - uBreathSpan.x) / max(uBreathSpan.y - uBreathSpan.x, 1e-4), 0.0, 1.0);
  float breathFront = 0.15 + 0.85 * clamp(0.5 + 0.5 * breathOut.y, 0.0, 1.0);
  float breathTall = (breathWorld.y - uBreathAxis.y) / 0.17;
  float breathTrunk = 1.0 - smoothstep(0.17, 0.24, abs(breathFromAxis.x));
  vec3 breathChest = vec3(breathOut.x * ${m(BREATH_TRAVEL.chestOut)}, ${m(BREATH_TRAVEL.chestUp)}, breathOut.y * ${m(BREATH_TRAVEL.chestForward)})
    * exp(-breathTall * breathTall) * breathFront * breathTrunk;
  float breathBellyTall = (breathWorld.y - (uBreathAxis.y - 0.27)) / 0.14;
  vec3 breathBelly = vec3(breathOut.x * ${m(BREATH_TRAVEL.bellyForward * 0.3)}, 0.0, breathOut.y * ${m(BREATH_TRAVEL.bellyForward)})
    * exp(-breathBellyTall * breathBellyTall) * breathFront;
  vec3 breathViscera = vec3(0.0, -${m(BREATH_TRAVEL.viscera)}, ${m(BREATH_TRAVEL.viscera * 0.15)});
  vec3 breathMove = vec3(0.0);
  if (abs(uBreathRole - ${BREATH_ROLE.chestWall}.0) < 0.5) {
    breathMove = breathChest;
  } else if (abs(uBreathRole - ${BREATH_ROLE.lung}.0) < 0.5) {
    float base = 1.0 - clamp((breathWorld.y - uBreathFrame.x) / max(uBreathFrame.y - uBreathFrame.x, 1e-4), 0.0, 1.0);
    float onDome = 1.0 - smoothstep(0.07, 0.12, abs(breathFromAxis.x));
    breathMove = breathChest + vec3(0.0, -${m(BREATH_TRAVEL.lungDown)} * base * base * onDome, 0.0);
  } else if (abs(uBreathRole - ${BREATH_ROLE.diaphragm}.0) < 0.5) {
    breathMove = vec3(0.0, -${m(BREATH_TRAVEL.diaphragm)} * breathUp * breathUp, 0.0);
  } else if (abs(uBreathRole - ${BREATH_ROLE.viscera}.0) < 0.5) {
    breathMove = breathViscera;
  } else if (abs(uBreathRole - ${BREATH_ROLE.belly}.0) < 0.5) {
    breathMove = breathBelly;
  } else if (abs(uBreathRole - ${BREATH_ROLE.heart}.0) < 0.5) {
    breathMove = vec3(0.0, -${m(BREATH_TRAVEL.heart)}, 0.0);
  } else if (abs(uBreathRole - ${BREATH_ROLE.vessel}.0) < 0.5) {
    float wall = max(smoothstep(0.03, 0.05, length(breathFromAxis)),
      smoothstep(uBreathFrame.z, uBreathFrame.z + 0.04, breathWorld.y));
    float middle = (1.0 - smoothstep(0.06, 0.11, abs(breathFromAxis.x)))
      * (1.0 - smoothstep(uBreathFrame.z, uBreathFrame.z + 0.08, breathWorld.y))
      * smoothstep(uBreathFrame.x - 0.06, uBreathFrame.x, breathWorld.y);
    vec3 own = (breathChest + breathBelly) * wall + vec3(0.0, -${m(BREATH_TRAVEL.heart)}, 0.0) * middle * (1.0 - wall);
    breathMove = mix(own, breathViscera, clamp(${CARRY_ATTRIBUTE}, 0.0, 1.0));
  }
  breathWorld.xyz += breathMove * uBreathFill;
  mvPosition = viewMatrix * breathWorld;
  gl_Position = projectionMatrix * mvPosition;
}`,
    );
}

export function breathOnBeforeCompile(this: unknown, shader: Shader): void {
  injectBreath(shader, this);
}

export interface BreathData {
  breathRole: BreathRole;
  breathSpan?: readonly [number, number];
}

const STILL: BreathData = Object.freeze({ breathRole: BREATH_ROLE.none });

export function breathData(
  organ: Pick<ManifestOrgan, "system" | "name_en" | "path">,
  worldBox?: { min: { y: number }; max: { y: number } },
  heartChamber = false,
): BreathData {
  const role = breathRole(organ, heartChamber);
  if (role === BREATH_ROLE.none) return STILL;
  const span: readonly [number, number] | undefined =
    worldBox && role === BREATH_ROLE.diaphragm
      ? [worldBox.min.y, worldBox.max.y]
      : undefined;
  return Object.freeze(span ? { breathRole: role, breathSpan: span } : { breathRole: role });
}
