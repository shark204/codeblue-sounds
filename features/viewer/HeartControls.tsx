'use client';

import { useAtlasLocale } from "@/lib/atlasI18n";
import { HeartPulse, Minimize2, Pause, Play, X } from "lucide-react";
import { useEffect, useState } from "react";

import { useBreathStore } from "@/stores/breathStore";
import { useHeartStore } from "@/stores/heartStore";
import { useSceneStore } from "@/stores/sceneStore";

import { EcgStrip } from "./EcgStrip";
import { primeHeartSound } from "./heartSound";
import { OVERLAY_CHIP, OVERLAY_CHIP_ACTION } from "./overlayChrome";
import { RHYTHM_GROUPS, RHYTHMS, rhythm } from "./rhythms";
import { breathNote } from "./sinusArrhythmia";

/**
 * Full Heartbeat & Auscultation (Heart Sounds) Panel (`HeartControls.tsx`).
 * Displays live ECG telemetry, S1/S2 & murmur audio toggle, clinical description,
 * and the complete interactive list of all 8 cardiac rhythms and murmurs.
 */

const HEART_CLINICAL_INFO: Record<string, { arTitle: string; site: string; ecg: string; causes: string; mgmt: string }> = {
  normal: {
    arTitle: "النظم الجيبي الطبيعي (Normal Sinus Rhythm)",
    site: "All 4 valve areas (Aortic, Pulmonary, Tricuspid, Mitral)",
    ecg: "Regular P before every QRS · PR 120–200 ms · Narrow QRS",
    causes: "Normal physiological cardiac conduction",
    mgmt: "No intervention required; routine observation",
  },
  sinus_tachycardia: {
    arTitle: "تسرع القلب الجيبي (Sinus Tachycardia)",
    site: "Apex / Left sternal border (rapid crisp S1 & S2)",
    ecg: "HR > 100 bpm · Normal upright P waves · Regular R-R",
    causes: "Fever, hypovolemia, pain, anxiety, anemia, thyrotoxicosis, PE",
    mgmt: "Treat underlying cause (fluids, antipyretics, analgesia, O2)",
  },
  sinus_bradycardia: {
    arTitle: "بطء القلب الجيبي (Sinus Bradycardia)",
    site: "Apex (slow, well-spaced S1/S2)",
    ecg: "HR < 60 bpm · Normal P-QRS relationship",
    causes: "Athletes, beta-blockers, hypothyroidism, inferior MI, raised ICP",
    mgmt: "Observe if asymptomatic; Atropine / pacing if unstable",
  },
  av_block_1: {
    arTitle: "إحصار أذيني بطيني من الدرجة الأولى (1st Degree AV Block)",
    site: "Apex (slightly softened S1)",
    ecg: "Fixed prolonged PR interval > 200 ms · Every P conducted",
    causes: "High vagal tone, AV nodal blockers, ischemic heart disease",
    mgmt: "Usually benign; monitor and review rate-controlling meds",
  },
  mobitz_1: {
    arTitle: "إحصار من الدرجة الثانية — وينكباخ (Mobitz Type I)",
    site: "Apex (progressive S1 softening before dropped beat)",
    ecg: "Progressive PR prolongation until a dropped QRS",
    causes: "Inferior MI, digoxin/beta-blockers, athletes",
    mgmt: "Usually benign; hold offending drugs, Atropine if symptomatic",
  },
  mobitz_2: {
    arTitle: "إحصار من الدرجة الثانية — موبيتز ٢ (Mobitz Type II)",
    site: "Apex (sudden unexpected missed beats)",
    ecg: "Constant PR interval with intermittent dropped QRS (2:1 / 3:1)",
    causes: "Anterior MI, fibrosis of His-Purkinje system",
    mgmt: "High risk of complete heart block — prepare transcutaneous/permanent pacemaker",
  },
  av_block_3: {
    arTitle: "إحصار قلب كامل — الدرجة الثالثة (Complete Heart Block)",
    site: "Apex (variable S1 intensity + cannon A waves)",
    ecg: "Complete AV dissociation · P waves & QRS independent (30–40 bpm)",
    causes: "Acute MI, Lev/Lenègre degeneration, Lyme, drug toxicity",
    mgmt: "Emergent pacing (transcutaneous -> permanent pacemaker), Atropine rarely effective",
  },
  atrial_fibrillation: {
    arTitle: "الرجفان الأذيني (Atrial Fibrillation)",
    site: "Apex (irregularly irregular rhythm, variable S1)",
    ecg: "Absent P waves · Irregularly irregular R-R · Fibrillatory baseline",
    causes: "HTN, CAD, mitral valve disease, thyrotoxicosis, alcohol",
    mgmt: "Rate/rhythm control (BB/CCB/Amiodarone) + anticoagulation (CHA2DS2-VASc)",
  },
  atrial_flutter: {
    arTitle: "الرفرفة الأذينية (Atrial Flutter)",
    site: "Apex (regular or regularly irregular rapid pulse)",
    ecg: "Sawtooth flutter waves (~300 bpm atrial) with 2:1 or 4:1 block",
    causes: "COPD, heart failure, post-cardiac surgery, HTN",
    mgmt: "Rate control, anticoagulation, cardioversion / cavotricuspid ablation",
  },
  premature_ventricular: {
    arTitle: "انقباض بطيني مبكر (PVCs)",
    site: "Apex (early beat followed by compensatory pause)",
    ecg: "Premature wide bizarre QRS > 120 ms without preceding P wave",
    causes: "Electrolyte imbalance (K/Mg), ischemia, caffeine, stress",
    mgmt: "Correct K+/Mg2+; beta-blockers if frequent/symptomatic",
  },
  ventricular_tachycardia: {
    arTitle: "تسرع القلب البطيني (Ventricular Tachycardia)",
    site: "Apex (rapid weak beats, AV dissociation)",
    ecg: "Wide-complex regular tachycardia > 150 bpm · AV dissociation",
    causes: "Acute MI/scar, cardiomyopathy, electrolyte derangement",
    mgmt: "Pulseless: Defibrillate + CPR. Stable: Amiodarone / Procainamide",
  },
  ventricular_fibrillation: {
    arTitle: "الرجفان البطيني (Ventricular Fibrillation — توقف قلب)",
    site: "No organized cardiac output or audible S1/S2",
    ecg: "Chaotic irregular undulating baseline · Zero QRS complexes",
    causes: "Acute myocardial ischemia/MI, severe hypoxia, electrocution",
    mgmt: "CARDIAC ARREST — Immediate high-quality CPR + unsynchronized defibrillation + Epinephrine",
  },
  asystole: {
    arTitle: "توقف القلب التام (Asystole — خط مستقيم)",
    site: "Complete silence on auscultation (flatline arrest)",
    ecg: "Flat isoelectric line · No P, QRS, or T waves",
    causes: "Hs & Ts (Hypoxia, Hypovolemia, H+ acidosis, Hyper/Hypo-K, Tamponade, Tension PTX, Toxins, Thrombosis)",
    mgmt: "Non-shockable arrest — Immediate CPR + Epinephrine IV q3–5 min + treat reversible Hs & Ts",
  },
  aortic_stenosis: {
    arTitle: "تضيق الصمام الأورطي (Aortic Stenosis)",
    site: "Right 2nd ICS (radiates to carotids)",
    ecg: "Crescendo-decrescendo systolic ejection murmur + soft/absent S2",
    causes: "Senile calcification, bicuspid aortic valve, rheumatic fever",
    mgmt: "Avoid vasodilators; TAVR / surgical valve replacement if symptomatic",
  },
  mitral_regurgitation: {
    arTitle: "ارتجاع الصمام الميترالي (Mitral Regurgitation)",
    site: "Apex / Left 5th ICS midclavicular (radiates to axilla)",
    ecg: "Holosystolic (pansystolic) blowing murmur · Soft S1, +/- S3",
    causes: "Mitral valve prolapse, papillary muscle rupture, LV dilation",
    mgmt: "Afterload reduction (ACEi), diuretics; surgical repair/replacement",
  },
  aortic_regurgitation: {
    arTitle: "ارتجاع الصمام الأورطي (Aortic Regurgitation)",
    site: "Left 3rd ICS (Erb point — leaning forward, full expiration)",
    ecg: "Early diastolic decrescendo high-pitched blowing murmur + wide pulse pressure",
    causes: "Aortic root dilation/dissection, endocarditis, bicuspid valve",
    mgmt: "Vasodilators (ACEi/CCB); urgent surgery if acute/endocarditis",
  },
  mitral_stenosis: {
    arTitle: "تضيق الصمام الميترالي (Mitral Stenosis)",
    site: "Apex (left lateral decubitus, bell of stethoscope)",
    ecg: "Loud S1 + Opening Snap + low-pitched diastolic rumble",
    causes: "Rheumatic heart disease (most common)",
    mgmt: "Diuretics, beta-blockers, anticoagulation if AF; balloon valvotomy",
  },
  third_heart_sound: {
    arTitle: "صوت القلب الثالث (S3 Gallop — Kentucky)",
    site: "Apex (bell, left lateral decubitus)",
    ecg: "Low-pitched early diastolic thud right after S2",
    causes: "Physiologic <35y/pregnancy; Pathologic: HFrEF, volume overload",
    mgmt: "Treat volume overload / heart failure (diuretics, GDMT)",
  },
  fourth_heart_sound: {
    arTitle: "صوت القلب الرابع (S4 Gallop — Tennessee)",
    site: "Apex (bell, left lateral decubitus)",
    ecg: "Low-pitched late diastolic sound just before S1 (atrial kick)",
    causes: "Stiff LV (HTN, aortic stenosis, HCM, ischemic heart disease)",
    mgmt: "Control hypertension / ischemia; absent in atrial fibrillation",
  },
};

export function HeartControls() {
  const { isAr, at } = useAtlasLocale();
  const enabled = useHeartStore((s) => s.enabled);
  const sound = useHeartStore((s) => s.sound);
  const rhythmId = useHeartStore((s) => s.rhythm);
  const toggle = useHeartStore((s) => s.toggle);
  const setSound = useHeartStore((s) => s.setSound);
  const setRhythm = useHeartStore((s) => s.setRhythm);
  const breathing = useBreathStore((s) => s.enabled);
  const clearSelection = useSceneStore((s) => s.clearSelection);
  const clearIsolation = useSceneStore((s) => s.clearIsolation);
  const hideAllSystems = useSceneStore((s) => s.hideAllSystems);

  const [panelMinimized, setPanelMinimized] = useState(false);

  useEffect(() => {
    if (enabled) {
      setPanelMinimized(false);
      clearSelection();
      primeHeartSound();
      setSound(false);
    }
  }, [enabled, clearSelection, setSound]);

  const current = rhythm(rhythmId);
  const info = HEART_CLINICAL_INFO[current.id] ?? HEART_CLINICAL_INFO.normal!;

  const stopAndClose = () => {
    setSound(false);
    setPanelMinimized(false);
    clearIsolation();
    clearSelection();
    hideAllSystems();
    if (enabled) toggle();
  };

  if (!enabled) return null;

  return (
    <div dir={isAr ? "rtl" : "ltr"} className="pointer-events-auto flex flex-col items-start gap-1.5">
      {panelMinimized && (
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setPanelMinimized(false)}
            className={`pointer-events-auto flex items-center gap-1 ${OVERLAY_CHIP} ${OVERLAY_CHIP_ACTION}`}
            title={at("إظهار نافذة تخطيط وأصوات القلب", "Show Heart Sounds panel")}
          >
            <span className="inline-flex items-center gap-1">
              <HeartPulse size={12} className="text-[#d98a96]" />
              <span>{at("إظهار نافذة أصوات القلب", "Heart Sounds")}</span>
            </span>
          </button>
          <button
            type="button"
            onClick={stopAndClose}
            title={at("إغلاق النافذة وإيقاف صوت ونبض القلب", "Close and stop heartbeat")}
            className="flex h-6 w-6 items-center justify-center rounded-lg border border-[#26324d] bg-[#131c31] text-xs font-bold text-[#f1f5f9] shadow transition hover:border-[#d98a96]"
          >
            <X size={12} />
          </button>
        </div>
      )}

      {!panelMinimized && (
        <div
          onWheel={(e) => e.stopPropagation()}
          className="cb-ausc-panel pointer-events-auto atlas-scrollbar w-[86vw] max-w-[340px] sm:w-[360px] md:w-[390px] sm:max-w-none max-h-[calc(100dvh-80px)] overflow-y-auto overscroll-contain rounded-2xl border border-[#26324d] bg-[#131c31]/98 p-3.5 text-[11px] leading-snug text-[#f1f5f9] shadow-2xl backdrop-blur-md"
        >
          <div className="mb-2 flex items-center justify-between border-b border-[#26324d] pb-2">
            <span className="font-bold text-[#d98a96]">
              <span className="inline-flex items-center gap-1.5">
                <HeartPulse size={13} className="text-[#d98a96]" />
                <span>{current.rate} · {at("تخطيط وأصوات القلب", "ECG & Heart Sounds")}</span>
              </span>
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setPanelMinimized(true)}
                title={at("تصغير النافذة مع استمرار النبض والصوت", "Minimize panel")}
                className="inline-flex items-center gap-1 rounded-lg border border-[#26324d] bg-[#0f172a] px-2 py-0.5 text-[9.5px] font-bold text-[#94a3b8] transition hover:border-[#3b4d71] hover:text-[#f1f5f9]"
              >
                <Minimize2 size={10} />
                <span>{at("تصغير", "Hide")}</span>
              </button>
              <button
                type="button"
                onClick={stopAndClose}
                title={at("إغلاق النافذة وإيقاف الصوت والنبض", "Close and stop")}
                className="inline-flex items-center gap-1 rounded-lg border border-[#26324d] bg-[#0f172a] px-2 py-0.5 text-[9.5px] font-bold text-[#f1f5f9] transition hover:border-[#d98a96]"
              >
                <X size={10} />
                <span>{at("إغلاق", "Close")}</span>
              </button>
            </div>
          </div>

          <p className="text-xs font-bold text-[#d98a96]">{current.label}</p>
          <p className="mt-0.5 text-[10px] leading-relaxed text-[#94a3b8]">{current.what}</p>

          <div className="my-2 rounded-xl border border-[#26324d] bg-[#0b1120] p-1.5">
            <EcgStrip />
          </div>

          {breathing && (
            <p className="mb-1.5 text-[9.5px] text-[#94a3b8]">
              {breathNote(current.followsBreath ?? 0)}
            </p>
          )}

          <button
            type="button"
            onClick={() => {
              const next = !sound;
              if (next) {
                primeHeartSound();
              }
              setSound(next);
            }}
            className={`mt-1 flex w-full items-center justify-center gap-1.5 rounded-xl border px-3 py-1.5 text-[10.5px] font-black shadow transition ${
              sound
                ? "border-[#d98a96] bg-[#d98a96]/20 text-[#f1f5f9] hover:bg-[#d98a96]/30"
                : "border-emerald-500/60 bg-emerald-500/20 text-emerald-200 hover:bg-emerald-500/30"
            }`}
          >
            {sound ? (
              <>
                <Pause size={13} className="shrink-0 text-[#d98a96]" />
                <span>{at("إيقاف صوت القلب مؤقتاً — Pause Heart Audio", "Pause Heart Sounds")}</span>
              </>
            ) : (
              <>
                <Play size={13} className="shrink-0 text-emerald-300" />
                <span>{at("تشغيل أصوات القلب والنفخات (S1/S2 & Murmurs)", "Play Heart Sounds (S1/S2 & Murmurs)")}</span>
              </>
            )}
          </button>

          {/* Direct Complete List of all 8 Heart Rhythms & Valvular Murmurs */}
          <div className="mt-2.5">
            <p className="mb-1 font-mono text-[9.5px] font-bold text-[#d98a90]">
              {at("النظم واللغط القلبي (8 أنماط سريرية):", "Cardiac Rhythms & Murmurs (8 patterns):")}
            </p>
            <div
              role="radiogroup"
              aria-label="Heart rhythm"
              className="space-y-1.5 rounded-xl border border-[#26324d] bg-[#0b1120] p-2"
            >
              {RHYTHM_GROUPS.map((group, gIdx) => {
                const accentBar =
                  gIdx % 3 === 0
                    ? "border-s-[3.5px] border-s-[#c83e4d]"
                    : gIdx % 3 === 1
                      ? "border-s-[3.5px] border-s-[#2563eb]"
                      : "border-s-[3.5px] border-s-[#ca8a04]";
                return (
                  <div key={group} className="space-y-1">
                    <p className="px-1 font-mono text-[8.5px] font-medium uppercase tracking-[0.15em] text-[#d98a96]">
                      {group}
                    </p>
                    <div className="relative space-y-1 ps-3.5">
                      {RHYTHMS.filter((candidate) => candidate.group === group && candidate.id !== "ventricular_fibrillation" && candidate.id !== "asystole").map((candidate) => {
                        const active = candidate.id === rhythmId;
                        return (
                          <div key={candidate.id} className="relative flex items-center">
                            <svg
                              aria-hidden
                              viewBox="0 0 14 24"
                              className="pointer-events-none absolute -start-3.5 top-1/2 h-5 w-3.5 -translate-y-1/2 overflow-visible"
                            >
                              <path
                                d="M 1 2 C 1 12, 6 12, 13 12"
                                fill="none"
                                stroke={active ? "#d98a96" : "rgba(163, 155, 144, 0.42)"}
                                strokeWidth={active ? "1.6" : "1.15"}
                                strokeLinecap="round"
                              />
                            </svg>
                            <button
                              type="button"
                              role="radio"
                              aria-checked={active}
                              onClick={() => {
                                primeHeartSound();
                                setRhythm(candidate.id);
                                if (!sound) {
                                  setSound(true);
                                }
                              }}
                              className={`flex w-full items-center justify-between gap-1.5 rounded-xl border ${accentBar} px-2.5 py-1.5 text-start text-[10px] font-medium transition ${
                                active
                                  ? "border-[#3b4d71] bg-[#1e293b] font-semibold text-[#f1f5f9]"
                                  : "border-[#26324d] bg-[#131c31]/85 text-[#f1f5f9] hover:border-[#3b4d71] hover:bg-[#1e293b]"
                              }`}
                            >
                              <span className="truncate">{candidate.label}</span>
                              <span className="shrink-0 font-mono text-[8.5px] text-[#d98a96]">
                                {candidate.rate}
                              </span>
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <button
            type="button"
            onClick={stopAndClose}
            className="mt-2.5 flex w-full items-center justify-center gap-1.5 rounded-xl border border-[#26324d] bg-[#0f172a] py-1.5 text-[10px] font-bold text-[#f1f5f9] transition hover:border-[#d98a96]"
          >
            <X size={11} className="text-[#d98a96]" />
            <span>{at("إيقاف الصوت والنبض وإغلاق النافذة", "Stop Heartbeat & Close")}</span>
          </button>
        </div>
      )}

      {!panelMinimized && (
        <div
          onWheel={(e) => e.stopPropagation()}
          className="cb-ausc-panel pointer-events-auto atlas-scrollbar mt-2 w-full md:mt-0 md:fixed md:right-4 lg:right-12 xl:right-20 md:top-16 md:z-30 md:w-[300px] lg:w-[325px] max-h-[calc(100dvh-90px)] overflow-y-auto overscroll-contain rounded-2xl border border-[#26324d] bg-[#131c31]/96 p-3.5 text-[11px] leading-relaxed text-[#f1f5f9] shadow-2xl backdrop-blur-md"
        >
          <div className="mb-2 flex items-center justify-between border-b border-[#26324d] pb-2">
            <span className="font-bold text-[#d98a96]">{at("بطاقة المعلومات السريرية", "Clinical Disease Card")}</span>
            <span className="rounded-md border border-[#d98a96]/40 bg-[#d98a96]/15 px-2 py-0.5 font-mono text-[9px] font-bold text-[#d98a96]">
              {current.rate}
            </span>
          </div>
          <p className="text-xs font-black text-[#f1f5f9]">{isAr ? info.arTitle : current.label}</p>
          <p className="mt-1 text-[10px] text-[#cbd5e1]">{current.what}</p>
          <div className="mt-2.5 space-y-2 rounded-xl border border-[#26324d] bg-[#0b1120] p-2.5 text-[10px]">
            <div>
              <span className="font-bold text-[#d98a96]">{at("موضع السماعة:", "Auscultation Site:")} </span>
              <span className="text-[#e2e8f0]">{info.site}</span>
            </div>
            <div>
              <span className="font-bold text-sky-300">{at("خصائص التخطيط والصوت:", "ECG & Acoustic Key:")} </span>
              <span className="text-[#e2e8f0]">{info.ecg}</span>
            </div>
            <div>
              <span className="font-bold text-amber-300">{at("الأسباب الشائعة:", "Common Causes:")} </span>
              <span className="text-[#e2e8f0]">{info.causes}</span>
            </div>
            <div>
              <span className="font-bold text-emerald-300">{at("التدبير العلاجي:", "Management:")} </span>
              <span className="text-[#e2e8f0]">{info.mgmt}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
