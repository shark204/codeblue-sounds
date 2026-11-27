/**
 * Composite & High-Risk ECG Rule Activations (Clinically Hardened)
 * - Enforces strict clinical specificity (Pre-Excited AFib requires rapid wide QRS; PE framed as suspicion;
 *   Trifascicular framed as Bifascicular + 1st-deg AVB; no subtractive PVC Bigeminy without OCR QT).
 * - Tags every machine-header-derived rule explicitly as "Machine Header (OCR) — Verify Manually".
 * - Frames all therapeutic guidance strictly for Education & Clinical Decision Support (requires physician sign-off).
 */

import { getClinicalInvTtt } from "@/lib/ecgClinicalLibrary";

export interface CompositeFinding {
  code: string;
  name_en: string;
  name_ar: string;
  urgency: "CRITICAL" | "URGENT" | "MODERATE";
  tier: "primary_override" | "composite" | "co_existing";
  replaces?: string[];
  explanation_en: string;
  explanation_ar: string;
  immediate_action_en: string;
  immediate_action_ar: string;
}

export interface CompositeRuleInput {
  confirmedCodes: string[];
  hrBpm: number | null;
  prMs: number | null;
  qrsMs: number | null;
  qtMs: number | null;
  axisDeg: number | null;
  spo2Low: boolean;
  pleuriticOrDyspnea: boolean;
  sodiumBlockerOrTca: boolean;
  syncopeOrScd: boolean;
  leadReversalDetected: boolean;
  wideQrsSuspected?: boolean;
  measurementSource?: "device_ocr" | "auto_waveform" | "manual" | null;
}

const CDS_TAG_EN = " [For education & clinical decision support — requires treating physician review & sign-off before action.]";
const CDS_TAG_AR = " [للتعليم ودعم القرار السريري فقط — يتطلب مراجعة وتوقيع الطبيب المعالج قبل أي إجراء علاجي.]";

function hdrTag(src: CompositeRuleInput["measurementSource"], ar: boolean): string {
  if (src === "manual") return ar ? " (إدخال يدوي)" : " (Manual Entry)";
  return ar
    ? " (قراءة الجهاز OCR — يلزم التحقق اليدوي؛ الأجهزة تخطئ في AF والإحصارات)"
    : " (Machine Header OCR — verify manually; devices often err in AF & blocks)";
}

function fromLib(
  code: string,
  urgency: "CRITICAL" | "URGENT" | "MODERATE",
  tier: "primary_override" | "composite" | "co_existing",
  titleEn: string,
  titleAr: string,
  opts?: {
    replaces?: string[];
    extraExpEn?: string;
    extraExpAr?: string;
    appendCds?: boolean;
  }
): CompositeFinding {
  const entry = getClinicalInvTtt(code, "", "", titleEn);
  const cds = opts?.appendCds !== false;
  return {
    code,
    name_en: titleEn,
    name_ar: titleAr,
    urgency,
    tier,
    replaces: opts?.replaces,
    explanation_en: `${entry.explanation_en}${opts?.extraExpEn ? ` ${opts.extraExpEn}` : ""}`,
    explanation_ar: `${entry.explanation_ar}${opts?.extraExpAr ? ` ${opts.extraExpAr}` : ""}`,
    immediate_action_en: `${entry.ttt_en}${cds ? CDS_TAG_EN : ""}`,
    immediate_action_ar: `${entry.ttt_ar}${cds ? CDS_TAG_AR : ""}`,
  };
}

export function evaluateCompositeEcgRules(inp: CompositeRuleInput): CompositeFinding[] {
  const codes = new Set(inp.confirmedCodes.map((c) => c.toUpperCase()));
  const has = (...keys: string[]) => keys.some((k) => codes.has(k));
  const out: CompositeFinding[] = [];
  const src = inp.measurementSource ?? "device_ocr";

  // 1. PRE_EXCITED_AFIB — Requires AFib + WPW + rapid rate (>=110) + wide QRS (>=120ms or wideQrsSuspected)
  if (has("ATRIAL_FIBRILLATION", "ATRIAL_FIBRILLATION_RVR") && has("WPW_PATTERN", "WPW_PREEXCITATION")) {
    const isRapid = inp.hrBpm !== null && inp.hrBpm >= 110;
    const isWide = (inp.qrsMs !== null && inp.qrsMs >= 120) || Boolean(inp.wideQrsSuspected);
    if (isRapid && isWide) {
      const qrsSrc = inp.qrsMs !== null ? hdrTag(src, false) : "";
      const qrsSrcAr = inp.qrsMs !== null ? hdrTag(src, true) : "";
      out.push(
        fromLib(
          "PRE_EXCITED_AFIB",
          "CRITICAL",
          "primary_override",
          "Pre-Excited Atrial Fibrillation (Irregular Rapid Wide-QRS + WPW)",
          "رجفان أذيني مع تنبيه مبكر (تسارع غير منتظم عريض QRS مع WPW)",
          {
            replaces: ["ATRIAL_FIBRILLATION", "ATRIAL_FIBRILLATION_RVR", "WPW_PATTERN", "WPW_PREEXCITATION"],
            extraExpEn: `Triggered by AFib + WPW + rapid rate (${inp.hrBpm} bpm) + wide QRS${qrsSrc}.`,
            extraExpAr: `تحقق بوجود رجفان أذيني + WPW + تسارع (${inp.hrBpm} نبضة/د) + QRS عريض${qrsSrcAr}.`,
          }
        )
      );
    } else {
      out.push(
        fromLib(
          "PRE_EXCITED_AFIB",
          "URGENT",
          "co_existing",
          "AFib + WPW History/Signal — Verify Irregular Wide-Complex Tachycardia",
          "رجفان أذيني مع إشارة WPW — تحقق من وجود تسارع غير منتظم بمركب QRS عريض",
          {
            extraExpEn:
              "Co-occurrence of AFib and WPW alone does not prove pre-excited AFib (patients with prior WPW may have ordinary narrow-QRS AFib). Confirm rapid irregular wide-complex QRS with delta waves manually before avoiding AV-nodal agents.",
            extraExpAr:
              "اجتماع AFib مع WPW وحده لا يكفي لإثبات Pre-Excited AFib (قد يكون لدى مريض WPW سابق رجفان أذيني عادي بمركب QRS ضيق). تحقق يدوياً من وجود تسارع غير منتظم عريض QRS مع موجة دلتا.",
          }
        )
      );
    }
  }

  // 2. TCA_SODIUM_BLOCKADE (Wide QRS >= 100ms + Na+ channel blocker / TCA context)
  const qrsWideFromHdr = inp.qrsMs !== null && inp.qrsMs >= 100;
  if (inp.sodiumBlockerOrTca && (qrsWideFromHdr || has("RBBB", "LBBB", "WIDE_COMPLEX_TACHYCARDIA"))) {
    out.push(
      fromLib(
        "TCA_SODIUM_BLOCKADE",
        "CRITICAL",
        "primary_override",
        `Suspected TCA / Sodium-Channel Blockade${qrsWideFromHdr ? " (Machine Header QRS)" : ""}`,
        `اشتباه تسمم حاصرات قنوات الصوديوم / TCA${qrsWideFromHdr ? " (قراءة الجهاز للـ QRS)" : ""}`,
        {
          extraExpEn: qrsWideFromHdr ? `QRS ${inp.qrsMs} ms${hdrTag(src, false)}.` : undefined,
          extraExpAr: qrsWideFromHdr ? `عرض QRS = ${inp.qrsMs} ms${hdrTag(src, true)}.` : undefined,
        }
      )
    );
  }

  // 3. PULMONARY_EMBOLISM_S1Q3T3 — Downgraded to URGENT clinical suspicion (must evaluate Wells/D-dimer/CTPA & differentials)
  const tachyOrStrain =
    has("SVT_NARROW", "SVT_TACHYCARDIA", "SINUS_TACHYCARDIA", "RBBB") ||
    (inp.hrBpm !== null && inp.hrBpm >= 105);
  if (tachyOrStrain && inp.spo2Low && inp.pleuriticOrDyspnea) {
    out.push(
      fromLib(
        "PULMONARY_EMBOLISM_S1Q3T3",
        "URGENT",
        "co_existing",
        "Suspected PE — Evaluate Wells / D-dimer / CTPA",
        "اشتباه PE: قيّم Wells / D-dimer / CTPA",
        {
          extraExpEn:
            "Non-specific hypoxia + dyspnea + tachycardia/RBBB trigger: overlaps heavily with COPD exacerbation, pneumonia, and acute heart failure. Do NOT treat as confirmed PE without clinical probability scoring (Wells) and objective testing (D-dimer / CTPA / bedside Echo).",
          extraExpAr:
            "قاعدة غير نوعية (تسارع/RBBB + نقص أكسجين <92% + ضيق نفس) تنطبق أيضاً على نوبات COPD والالتهاب الرئوي وقصور القلب الحاد. لا يُعتمد كتشخيص مؤكد دون تقييم Wells وفحص D-dimer أو CTPA.",
        }
      )
    );
  }

  // 4. SHORT_QT (Strictly QTc < 340 ms AND verified from Machine Header OCR or Manual input — never raw QT at high HR!)
  const qtcEst = inp.qtMs !== null && inp.hrBpm !== null && inp.hrBpm > 25 ? Math.round(inp.qtMs / Math.cbrt(60 / inp.hrBpm)) : inp.qtMs;
  if ((src === "device_ocr" || src === "manual") && qtcEst !== null && qtcEst >= 240 && qtcEst < 340) {
    out.push(
      fromLib(
        "SHORT_QT",
        inp.syncopeOrScd ? "CRITICAL" : "URGENT",
        "co_existing",
        `Short QTc (<340 ms) — ${src === "manual" ? "Manual" : "Machine Header"}`,
        `قِصر فترة QTc (<340 ms) — ${src === "manual" ? "إدخال يدوي" : "قراءة الجهاز"}`,
        {
          extraExpEn: `Derived from QTc ${qtcEst} ms (raw QT ${inp.qtMs} ms)${hdrTag(src, false)}.`,
          extraExpAr: `مبني على QTc = ${qtcEst} ms (QT الخام ${inp.qtMs} ms)${hdrTag(src, true)}.`,
        }
      )
    );
  }

  // 4b. Fast Broad / Short-PR Tachyarrhythmia Guard (FBI / Pre-Excited AFib / VT — discordant ST mimics STEMI!)
  const isFastBroadOrShortPr =
    inp.hrBpm !== null &&
    inp.hrBpm >= 150 &&
    ((inp.qrsMs !== null && inp.qrsMs >= 115) || (inp.prMs !== null && inp.prMs > 50 && inp.prMs < 110));
  if (isFastBroadOrShortPr && !has("PRE_EXCITED_AFIB", "VENTRICULAR_TACHYCARDIA", "TORSADES_DE_POINTES")) {
    out.push(
      fromLib(
        "PRE_EXCITED_AFIB",
        "CRITICAL",
        "primary",
        "Suspected Pre-Excited AFib / Wide-Complex Tachycardia (Rule Out VT / WPW)",
        "اشتباه رجفان أذيني مع WPW أو تسارع بطيني عريض (استبعد VT / FBI)",
        {
          extraExpEn: `Extreme tachycardia (HR ~${Math.round(inp.hrBpm!)} bpm) with ${inp.qrsMs && inp.qrsMs >= 115 ? `wide/borderline QRS (${Math.round(inp.qrsMs)} ms)` : ""}${inp.prMs && inp.prMs < 110 ? ` short PR (${Math.round(inp.prMs)} ms)` : ""}. Discordant ST shifts during rapid wide tachycardia frequently mimic STEMI — evaluate rhythm first and AVOID AV-nodal blockers if irregular.`,
          extraExpAr: `تسارع شديد (~${Math.round(inp.hrBpm!)} نبضة/دقيقة) مع ${inp.qrsMs && inp.qrsMs >= 115 ? `QRS عريض/حدي (${Math.round(inp.qrsMs)} ms)` : ""}${inp.prMs && inp.prMs < 110 ? ` وPR قصير (${Math.round(inp.prMs)} ms)` : ""}. تغيرات ST الثانوية أثناء التسارع العريض تحاكي STEMI — قيّم النظم أولاً وتجنّب حاصرات عقدة AV إذا كان غير منتظم.`,
        }
      )
    );
  }

  // 5 & 6. TRIFASCICULAR_BLOCK_PATTERN & BIFASCICULAR_BLOCK
  const hasRbbb = has("RBBB");
  const hasFasc = has("LAFB", "LPFB", "LAFB_LPFB");
  const prFromHeader = inp.prMs !== null && inp.prMs > 200;
  const hasFirstAvb = has("FIRST_DEGREE_AV_BLOCK") || prFromHeader;
  if (hasRbbb && hasFasc && hasFirstAvb) {
    out.push(
      fromLib(
        "TRIFASCICULAR_BLOCK_PATTERN",
        "URGENT",
        "composite",
        `Bifascicular Block + 1st-Degree AVB (Suspected Trifascicular${prFromHeader ? " — Machine Header PR" : ""})`,
        `إحصار ثنائي الحزم + إحصار درجة أولى (يُشتبه trifascicular${prFromHeader ? " — قراءة الجهاز للـ PR" : ""})`,
        {
          replaces: ["RBBB", "LAFB_LPFB", "FIRST_DEGREE_AV_BLOCK"],
          extraExpEn: `Prolonged PR may reflect AV nodal delay rather than true infranodal third-fascicle disease.${prFromHeader ? ` PR ${inp.prMs} ms${hdrTag(src, false)}.` : ""}`,
          extraExpAr: `تطاول PR قد يكون من العقدة الأذينية البطينية (AV node) وليس إحصاراً حقيقياً في الحزمة الثالثة.${prFromHeader ? ` قراءة PR = ${inp.prMs} ms${hdrTag(src, true)}.` : ""}`,
        }
      )
    );
  } else if (hasRbbb && hasFasc) {
    out.push(
      fromLib(
        "BIFASCICULAR_BLOCK",
        "MODERATE",
        "composite",
        "Bifascicular Block (RBBB + LAFB/LPFB)",
        "إحصار ثنائي الحزم (RBBB + LAFB/LPFB)",
        { replaces: ["RBBB", "LAFB_LPFB"] }
      )
    );
  }

  // 7. DEXTROCARDIA_LEAD_REVERSAL
  const extremeAxis = inp.axisDeg !== null && (inp.axisDeg <= -90 || inp.axisDeg >= 150);
  if (inp.leadReversalDetected || extremeAxis) {
    out.push(
      fromLib(
        "DEXTROCARDIA_LEAD_REVERSAL",
        "URGENT",
        "co_existing",
        "Suspected Limb Lead Reversal (LA–RA Swap) vs Dextrocardia",
        "⚠️ اشتباه خطأ في تركيب أقطاب الـ ECG (عكس اليدين) أو انعكاس موضع القلب — راجع الأقطاب وأعد التخطيط",
        {
          extraExpEn: extremeAxis ? `Extreme axis (${inp.axisDeg}°)${hdrTag(src, false)}.` : undefined,
          extraExpAr: extremeAxis ? `محور قلبي متطرف (${inp.axisDeg}°)${hdrTag(src, true)}.` : undefined,
        }
      )
    );
  }

  // 8. Ectopy vs Unclassified LONG_QT_OR_ECTOPY — NEVER subtractively claim Bigeminy; require actual QT for ectopy separation
  if (has("LONG_QT", "LONG_QT_OR_ECTOPY")) {
    if (inp.qtMs !== null && inp.qtMs < 450) {
      out.push(
        fromLib(
          "PVC_BIGEMINY",
          "MODERATE",
          "co_existing",
          `Ventricular Ectopy (Non-Specific — Verify Rhythm Strip${src !== "manual" ? ", Machine Header QT" : ""})`,
          `نبضات هاجرة بطينية محتملة (Ectopy غير محدد النمط — راجع الشريط${src !== "manual" ? "، قراءة الجهاز للـ QT" : ""})`,
          {
            extraExpEn: `Separated from composite Long-QT/Ectopy signal using QT ${inp.qtMs} ms${hdrTag(src, false)}. Does NOT establish bigeminy pattern — inspect rhythm strip manually.`,
            extraExpAr: `فُصلت عن الفئة المدمجة بناءً على QT = ${inp.qtMs} ms${hdrTag(src, true)}. لا يُثبت نمط Bigeminy تلقائياً — يلزم فحص الشريط يدوياً.`,
          }
        )
      );
    } else if (inp.qtMs === null && has("LONG_QT_OR_ECTOPY")) {
      out.push({
        code: "UNCLASSIFIED_QT_ECTOPY",
        name_en: "Unclassified Long-QT / Ectopy Signal (Missing QT — Verify Manually)",
        name_ar: "غير محدد: إشارة تطاول QT أو نبضات هاجرة (بدون قراءة QT — يلزم التحقق اليدوي)",
        urgency: "MODERATE",
        tier: "co_existing",
        explanation_en:
          "Visual model flagged the combined Long-QT/Ectopy class, but no printed machine-header QT/QTc was available to separate them. Inspect rhythm strip for ectopy and measure QTc manually.",
        explanation_ar:
          "رصد النموذج البصري إشارة ضمن الفئة المدمجة (تطاول QT أو نبضات هاجرة)، وبغياب قراءة QT/QTc من الهيدر لا يمكن الفصل بينهما تلقائياً. راجع الشريط وقِس QTc يدوياً.",
        immediate_action_en: `Measure QTc manually (leads II/V5) and inspect rhythm strip for premature beats.${CDS_TAG_EN}`,
        immediate_action_ar: `قِس فترة QTc يدوياً في II وV5 وافحص شريط النظم بحثاً عن نبضات هاجرة.${CDS_TAG_AR}`,
      });
    }
  }

  return out;
}

export function mergeCompositeFindings<T extends { code: string; isDemoted?: boolean }>(
  gatedFindings: T[],
  ctx: CompositeRuleInput
): T[] {
  const hits = evaluateCompositeEcgRules(ctx);
  if (hits.length === 0) return gatedFindings;
  const out = [...gatedFindings];
  for (const ch of hits) {
    if (!out.some((g) => g.code === ch.code)) {
      const item = {
        code: ch.code,
        name_en: ch.name_en,
        name_ar: ch.name_ar,
        urgency: ch.urgency,
        source: "rules",
        confidence: 0.85,
        explanation_en: ch.explanation_en,
        explanation_ar: ch.explanation_ar,
        immediate_action_en: ch.immediate_action_en,
        immediate_action_ar: ch.immediate_action_ar,
        fusionConfidence: 0.85,
        concordanceLevel: ch.urgency === "CRITICAL" ? "strong" : "moderate",
        concordanceLabelEn: ch.urgency === "CRITICAL" ? "Clinical Rule Alert" : "Clinical Suspicion",
        concordanceLabelAr: ch.urgency === "CRITICAL" ? "تنبيه قاعدة سريرية" : "اشتباه سريري",
        isConfirmed: true,
      } as unknown as T;
      if (ch.tier === "primary_override") out.unshift(item);
      else out.push(item);
    }
  }
  return out;
}
