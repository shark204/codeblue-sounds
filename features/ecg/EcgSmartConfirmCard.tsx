"use client";

import React from "react";
import { Check, ShieldCheck, Sparkles, SkipForward } from "lucide-react";
import { WizardFeatures } from "@/lib/ecgHybridEngine";

export interface AiDxItem {
  label: string;
  code: string | null;
  prob: number;
}

interface ConfirmOption {
  id: string;
  labelAr: string;
  labelEn: string;
  subAr: string;
  subEn: string;
  isActive: (f: WizardFeatures) => boolean;
  toggle: (f: WizardFeatures) => WizardFeatures;
}

interface ConfirmGroup {
  titleAr: string;
  titleEn: string;
  subtitleAr: string;
  subtitleEn: string;
  options: ConfirmOption[];
}

export function getSmartConfirmGroup(aiDiagnoses?: AiDxItem[]): ConfirmGroup | null {
  if (!aiDiagnoses || aiDiagnoses.length === 0) return null;
  const codes = new Set(
    aiDiagnoses
      .filter((d) => d.prob >= 0.12)
      .map((d) => (d.code || d.label || "").toUpperCase())
  );
  const has = (...keys: string[]) => keys.some((k) => Array.from(codes).some((c) => c.includes(k)));

  // 1. STEMI / MI territory -> differentiate Pericarditis & Cardiogenic/RV Shock
  if (has("STEMI", "ANTERIOR_SEPTAL_MI", "INFERIOR_POSTERIOR_MI", "LATERAL_MI")) {
    return {
      titleAr: "تأكيد سريري سريع قبل التشخيص (اشتباه ارتفاع ST / احتشاء)",
      titleEn: "Quick Pre-Diagnosis Clinical Check (ST Elevation / MI Suspected)",
      subtitleAr: "اختر ما ينطبق على المريض لحسم التداخل بين STEMI والتهاب التامور والصدمة:",
      subtitleEn: "Select any bedside finding to separate STEMI from Pericarditis and Shock:",
      options: [
        {
          id: "ischemic_pain",
          labelAr: "ألم صدري ضاغط كلاسيكي (أو ضيق نفس / تعرق)",
          labelEn: "Classic crushing ischemic chest pain / diaphoresis",
          subAr: "يرجّح احتشاء عضلة القلب الحاد (STEMI)",
          subEn: "Supports Acute MI / STEMI pathway",
          isActive: (f) => Boolean(f.chest_pain_or_shock),
          toggle: (f) => ({ ...f, chest_pain_or_shock: !f.chest_pain_or_shock }),
        },
        {
          id: "pericarditis_pain",
          labelAr: "ألم حاد بيزيد مع النفس ويخف بالميل للأمام + سخونية/دور برد",
          labelEn: "Sharp pleuritic pain better leaning forward + recent fever/URI",
          subAr: "يرجّح Acute Pericarditis بدل STEMI",
          subEn: "Raises Acute Pericarditis differential vs STEMI",
          isActive: (f) => Boolean(f.pleuritic_or_dehydration_or_bleed && f.temp_status === "fever"),
          toggle: (f) => {
            const on = !(f.pleuritic_or_dehydration_or_bleed && f.temp_status === "fever");
            return { ...f, pleuritic_or_dehydration_or_bleed: on, temp_status: on ? "fever" : "normal" };
          },
        },
        {
          id: "shock_low_bp",
          labelAr: "هبوط ضغط (BP < 90) أو عدم استقرار hemodynamic",
          labelEn: "Hypotension (BP < 90) or hemodynamic instability",
          subAr: "يفعّل بروتوكول الصدمة حسب المنطقة (RV MI vs LV Shock)",
          subEn: "Triggers territory-accurate Shock protocol",
          isActive: (f) => f.bp_status === "low" || Boolean(f.hemodynamic_unstable),
          toggle: (f) => {
            const on = !(f.bp_status === "low" || f.hemodynamic_unstable);
            return { ...f, bp_status: on ? "low" : "normal", hemodynamic_unstable: on };
          },
        },
      ],
    };
  }

  // 2. NSTEMI / Ischemia -> Wellens / De Winter / Hyperkalemia check
  if (has("NSTEMI", "ISCHEMIA")) {
    return {
      titleAr: "تأكيد سريري سريع (اشتباه نقص تروية / تغيرات موجة T)",
      titleEn: "Quick Pre-Diagnosis Check (Ischemia / T-Wave Changes)",
      subtitleAr: "يساعد في كشف Wellens وDe Winter وارتفاع البوتاسيوم:",
      subtitleEn: "Helps separate Wellens, De Winter, and Hyperkalemia:",
      options: [
        {
          id: "wellens_or_dewinter",
          labelAr: "ألم صدري إقفاري (مستمر الآن أو كان موجود وهدي مؤخراً)",
          labelEn: "Ischemic chest pain (ongoing or recently resolved)",
          subAr: "ينبّه لـ Wellens أو De Winter (انسداد LAD قريب)",
          subEn: "Flags Wellens / De Winter proximal LAD risk",
          isActive: (f) => Boolean(f.chest_pain_or_shock),
          toggle: (f) => ({ ...f, chest_pain_or_shock: !f.chest_pain_or_shock }),
        },
        {
          id: "hyperkalemia_risk",
          labelAr: "فشل كلوي / جلسة غسيل متأخرة / DKA / اشتباه ارتفاع بوتاسيوم",
          labelEn: "CKD / Missed dialysis / DKA / Suspected Hyperkalemia",
          subAr: "يحذّر فوراً من Hyperkalemia (كالسيوم وريدي + قياس K+)",
          subEn: "Triggers Hyperkalemia alert & IV Calcium protection",
          isActive: (f) => Boolean(f.dka_ckd_or_high_k),
          toggle: (f) => ({ ...f, dka_ckd_or_high_k: !f.dka_ckd_or_high_k }),
        },
      ],
    };
  }

  // 3. AFib / Flutter / SVT -> PE / Sepsis / Unstable / WPW check
  if (has("ATRIAL_FIBRILLATION", "ATRIAL_FLUTTER", "SVT", "WPW")) {
    return {
      titleAr: "تأكيد سريري سريع (اضطراب نظم / تسارع ضربات القلب)",
      titleEn: "Quick Pre-Diagnosis Check (Tachyarrhythmia / WPW)",
      subtitleAr: "يحدد هل الحالة طارئة (صدمة كهربائية) أم ثانوية (PE / إنتان):",
      subtitleEn: "Clarifies instability vs secondary causes (PE / Sepsis):",
      options: [
        {
          id: "unstable_tachy",
          labelAr: "غير مستقر hemodynamic (هبوط ضغط / ألم صدر / تغيّر وعي)",
          labelEn: "Hemodynamically unstable (hypotension / chest pain / altered mental status)",
          subAr: "يفعّل مسار Synchronized Cardioversion الفوري",
          subEn: "Prioritizes immediate Synchronized Cardioversion",
          isActive: (f) => Boolean(f.hemodynamic_unstable),
          toggle: (f) => ({ ...f, hemodynamic_unstable: !f.hemodynamic_unstable }),
        },
        {
          id: "pe_risk",
          labelAr: "نهجان مفاجئ / نقص أكسجين (SpO2 < 92%) / تورم ساق أو عملية حديثة",
          labelEn: "Sudden dyspnea / Hypoxia (SpO2 < 92%) / Leg swelling or recent surgery",
          subAr: "يرفع اشتباه جلطة الرئة (PE / RV Strain)",
          subEn: "Flags Pulmonary Embolism / RV Strain workup",
          isActive: (f) => f.spo2_status === "low",
          toggle: (f) => ({
            ...f,
            spo2_status: f.spo2_status === "low" ? "normal" : "low",
            pleuritic_or_dehydration_or_bleed: f.spo2_status !== "low",
          }),
        },
        {
          id: "sepsis_fever",
          labelAr: "سخونية / إنتان / جفاف شديد",
          labelEn: "Fever / Sepsis / Severe dehydration",
          subAr: "يرجّح تسارع/رفرفة ثانوية علاجها علاج السبب أولاً",
          subEn: "Points to Secondary AFib/Tachycardia (treat root cause)",
          isActive: (f) => f.temp_status === "fever",
          toggle: (f) => ({ ...f, temp_status: f.temp_status === "fever" ? "normal" : "fever" }),
        },
      ],
    };
  }

  // 4. LBBB / Brugada / Bradycardia / High-Grade AVB / Long QT / Pacemaker
  if (has("LBBB", "BRUGADA", "BRADY", "HIGH_GRADE", "LONG_QT", "PACEMAKER")) {
    return {
      titleAr: "تأكيد سريري سريع (اضطراب توصيل / Brugada / Long QT / Bradycardia)",
      titleEn: "Quick Pre-Diagnosis Check (Conduction / Brugada / Long QT / Brady)",
      subtitleAr: "اختر أي علامة سريرية لتحديد خطورة الحالة بدقة:",
      subtitleEn: "Select any bedside risk factor to refine urgency:",
      options: [
        {
          id: "syncope_scd",
          labelAr: "إغماء / دوخة شديدة أو تاريخ عائلي لموت مفاجئ",
          labelEn: "Syncope / Presyncope or family history of sudden cardiac death",
          subAr: "يرفع الخطورة الفورية لـ Brugada وLong QT وHigh-Grade AVB",
          subEn: "Escalates Brugada, Torsades/Long QT, and AV Block risk",
          isActive: (f) => Boolean(f.syncope_or_family_scd),
          toggle: (f) => ({ ...f, syncope_or_family_scd: !f.syncope_or_family_scd }),
        },
        {
          id: "acute_cp_lbbb",
          labelAr: "ألم صدري حاد (للتحقق من Sgarbossa في LBBB)",
          labelEn: "Acute chest pain (Sgarbossa STEMI-equivalent check in LBBB)",
          subAr: "يفرّق بين LBBB قديم مستقر وبين احتشاء حاد",
          subEn: "Separates stable old LBBB from acute MI",
          isActive: (f) => Boolean(f.chest_pain_or_shock),
          toggle: (f) => ({ ...f, chest_pain_or_shock: !f.chest_pain_or_shock }),
        },
        {
          id: "fever_na_or_pace",
          labelAr: "حمى شديدة / أدوية قلب (Beta-blocker / Na+ blocker) أو مركّب Pacemaker",
          labelEn: "High fever / Na+ or AV-nodal blocker drugs / Implanted Pacemaker",
          subAr: "يكشف Brugada المحفّز بالحمى أو سمّية الأدوية",
          subEn: "Flags fever-unmasked Brugada or drug-induced block",
          isActive: (f) => Boolean(f.fever_or_cocaine_or_sodium_blocker || f.has_pacemaker),
          toggle: (f) => ({
            ...f,
            fever_or_cocaine_or_sodium_blocker: !f.fever_or_cocaine_or_sodium_blocker,
          }),
        },
      ],
    };
  }

  return null;
}

interface Props {
  group: ConfirmGroup;
  features: WizardFeatures;
  isAr: boolean;
  isLight: boolean;
  onUpdateFeatures: (updater: (prev: WizardFeatures) => WizardFeatures) => void;
  onConfirm: () => void;
  onSkip: () => void;
}

export function EcgSmartConfirmCard({
  group,
  features,
  isAr,
  isLight,
  onUpdateFeatures,
  onConfirm,
  onSkip,
}: Props) {
  return (
    <div
      className={`cb-glass-card p-5 space-y-4 border ${
        isLight
          ? "border-rose-300/80 bg-white/85 text-slate-900"
          : "border-rose-500/40 bg-slate-950/80 text-slate-100"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-xl bg-rose-600/20 border border-rose-400/40 text-rose-300 shrink-0">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <div>
          <div className="text-sm font-black">
            {isAr ? group.titleAr : group.titleEn}
          </div>

        </div>
      </div>

      <div className="space-y-2">
        {group.options.map((opt) => {
          const active = opt.isActive(features);
          return (
            <button
              key={opt.id}
              type="button"
              onClick={() => onUpdateFeatures(opt.toggle)}
              className={`w-full text-start rounded-xl border px-3.5 py-3 transition-all cursor-pointer flex items-center gap-3 ${
                active
                  ? "border-rose-400 bg-rose-600/20 text-white"
                  : isLight
                  ? "border-slate-200 bg-white/70 hover:border-rose-300 text-slate-900"
                  : "border-white/15 bg-white/[0.04] hover:border-rose-400/50 text-slate-100"
              }`}
            >
              <span
                className={`w-5 h-5 rounded-md border flex items-center justify-center shrink-0 ${
                  active
                    ? "bg-rose-600 border-rose-400 text-white"
                    : "border-slate-400/60"
                }`}
              >
                {active && <Check className="w-3.5 h-3.5" />}
              </span>
              <div className="min-w-0">
                <div className="text-xs font-extrabold">{isAr ? opt.labelAr : opt.labelEn}</div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2 pt-1">
        <button
          type="button"
          onClick={onConfirm}
          className="cb-btn-primary flex-1 min-h-[44px] px-4 py-2.5 rounded-xl text-xs font-black flex items-center justify-center gap-2 cursor-pointer"
        >
          <Sparkles className="w-4 h-4" />
          <span>{isAr ? "تأكيد وعرض التشخيص الدقيق" : "Confirm & Show Diagnosis"}</span>
        </button>
        <button
          type="button"
          onClick={onSkip}
          className={`min-h-[44px] px-4 py-2.5 rounded-xl text-xs font-bold border inline-flex items-center gap-1.5 cursor-pointer ${
            isLight
              ? "border-slate-300 bg-white text-slate-700 hover:bg-slate-50"
              : "border-white/15 bg-white/5 text-slate-300 hover:bg-white/10"
          }`}
        >
          <SkipForward className="w-3.5 h-3.5" />
          <span>{isAr ? "تخطي (غير متاح)" : "Skip"}</span>
        </button>
      </div>
    </div>
  );
}

export function getRelevantWizardSteps(aiDiagnoses?: AiDxItem[], hrBpm?: number | null): number[] {
  const steps: number[] = [1, 2];
  if (!aiDiagnoses || aiDiagnoses.length === 0) return steps;
  const codes = new Set(aiDiagnoses.filter((d) => d.prob >= 0.12).map((d) => (d.code || d.label || "").toUpperCase()));
  const has = (...keys: string[]) => keys.some((k) => Array.from(codes).some((c) => c.includes(k)));
  const needsAthlete = has("LVH", "HYPERTROPHY", "PERICARDITIS", "BRADY", "FIRST_DEGREE", "EARLY_REPOL", "AV_BLOCK") || (hrBpm != null && hrBpm < 60);
  const needsBackground = has("STEMI", "MI", "ISCHEMIA", "NSTEMI", "LBBB", "RBBB", "BRUGADA", "LONG_QT", "PACEMAKER", "ATRIAL_FIBRILLATION", "FLUTTER", "SVT", "WPW", "HIGH_GRADE", "ECTOPY");
  if (needsAthlete) steps.push(3);
  if (needsBackground) steps.push(4);
  return steps;
}
