"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  Flame,
  FlaskConical,
  Pill,
  ShieldAlert,
  X,
} from "lucide-react";
import { HybridDiagnosisResult, WizardFeatures } from "@/lib/ecgHybridEngine";
import { getClinicalInvTtt } from "@/lib/ecgClinicalLibrary";

interface Props {
  result: HybridDiagnosisResult;
  isAr: boolean;
  isLight: boolean;
  cardCls: string;
  onEditInputs: () => void;
  onApplyCheck: (checkId: string, flagKey: keyof WizardFeatures) => void;
}

export function EcgResultsPanel({
  result,
  isAr,
  isLight,
  cardCls,
  onEditInputs,
  onApplyCheck,
}: Props) {
  // Compact anchored floating popover for Secondary Diagnosis Inv / TTT (NOT full-screen zoom)
  const [secondaryPopover, setSecondaryPopover] = useState<"inv" | "ttt" | null>(null);

  const urgencyBannerStyle = (u: string) => {
    if (u === "CRITICAL") return "bg-red-500/20 border-red-500/60 text-red-100";
    if (u === "URGENT") return "bg-amber-500/20 border-amber-400/60 text-amber-100";
    if (u === "NORMAL") return "bg-emerald-500/20 border-emerald-400/60 text-emerald-100";
    return "bg-sky-500/20 border-sky-400/60 text-sky-100";
  };

  const m = result.rulesResult.measurements;
  const subCardCls = isLight
    ? "border-slate-200/90 bg-white/75 text-slate-900"
    : "border-white/10 bg-[#0a1224]/65 text-slate-100";

  // Strictly Top-2 diagnoses ONLY (Primary + Secondary Differential — never 3rd/4th)
  const activeOnly = result.primaryFindings.filter((f) => !("isDemoted" in f && f.isDemoted));
  const primary = activeOnly[0] ?? result.primaryFindings[0];
  const secondary = activeOnly[1] ?? null;

  const isAfibCase = [primary, secondary].some(
    (f) =>
      f &&
      ["ATRIAL_FIBRILLATION", "ATRIAL_FIBRILLATION_RVR", "ATRIAL_FLUTTER", "SECONDARY_AFIB_RVR", "SECONDARY_AFIB_SEPSIS_FEVER"].includes(
        f.code
      )
  );

  const primaryInvTtt = primary
    ? getClinicalInvTtt(primary.code, primary.immediate_action_en, primary.immediate_action_ar, primary.name_en)
    : null;
  const secondaryInvTtt = secondary
    ? getClinicalInvTtt(secondary.code, secondary.immediate_action_en, secondary.immediate_action_ar, secondary.name_en)
    : null;

  const sourceLabel = (src: string) => {
    if (src === "context_override") return isAr ? "سياق المريض والفايتلز" : "Vitals & Context";
    if (src === "ai_model") return isAr ? "الذكاء الاصطناعي البصري" : "Visual AI";
    if (src === "case_kb") return isAr ? "قاعدة الحالات" : "Case KB";
    return isAr ? "محرك القواعد" : "Rules Engine";
  };

  return (
    <div className="space-y-3">
      {/* TOP PRIMARY DIAGNOSIS BANNER — Clean inline Inv & TTT panels (no redundant top buttons) */}
      {primary && primaryInvTtt && (
        <div
          className={`rounded-2xl border px-4 py-3.5 space-y-2.5 backdrop-blur-xl ${urgencyBannerStyle(
            result.finalUrgency
          )}`}
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex items-start gap-3 min-w-0 flex-1">
              {result.finalUrgency === "CRITICAL" ? (
                <AlertOctagon className="w-6 h-6 shrink-0 mt-0.5 text-red-300" />
              ) : result.finalUrgency === "URGENT" ? (
                <AlertTriangle className="w-6 h-6 shrink-0 mt-0.5 text-amber-300" />
              ) : (
                <CheckCircle2 className="w-6 h-6 shrink-0 mt-0.5 text-emerald-300" />
              )}
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[10px] font-mono font-black uppercase tracking-wider px-2 py-0.5 rounded bg-black/30">
                    {result.finalUrgency}
                  </span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-white/15">
                    {sourceLabel(primary.source)}
                  </span>
                </div>
                <div className="font-black text-base sm:text-lg leading-snug mt-1">
                  {isAr ? primary.name_ar : primary.name_en}
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={onEditInputs}
              className="px-3 py-1.5 rounded-xl bg-black/30 hover:bg-black/45 border border-white/20 text-xs font-bold cursor-pointer shrink-0"
            >
              {isAr ? "تعديل الاختيارات" : "Edit Inputs"}
            </button>
          </div>

          {/* Primary Textbook Clinical Explanation */}
          <p className="text-xs sm:text-[13px] leading-relaxed opacity-95">
            {isAr ? primary.explanation_ar : primary.explanation_en}
          </p>

          {/* Full Inline Inv & TTT Panels directly beneath Primary Diagnosis */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1">
            <div className="rounded-xl bg-black/25 border border-cyan-300/35 px-3.5 py-2.5 space-y-1">
              <div className="text-[11px] font-black text-cyan-200 flex items-center gap-1.5">
                <FlaskConical className="w-3.5 h-3.5 shrink-0" />
                <span>{isAr ? "الفحوصات المطلوبة (Inv)" : "Investigations (Inv)"}</span>
              </div>
              <p className="text-[12px] leading-relaxed opacity-95">
                {isAr ? primaryInvTtt.inv_ar : primaryInvTtt.inv_en}
              </p>
            </div>

            <div className="rounded-xl bg-black/25 border border-emerald-300/35 px-3.5 py-2.5 space-y-1">
              <div className="text-[11px] font-black text-emerald-200 flex items-center gap-1.5">
                <Pill className="w-3.5 h-3.5 shrink-0" />
                <span>{isAr ? "خطة العلاج والإجراء الفوري (TTT)" : "Treatment & Immediate Action (TTT)"}</span>
              </div>
              <p className="text-[12px] leading-relaxed opacity-95">
                {isAr ? primaryInvTtt.ttt_ar : primaryInvTtt.ttt_en}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Conditional 1-Tap Pitfall Verification Cards */}
      {result.conditionalChecks.length > 0 && (
        <div className="space-y-2">
          {result.conditionalChecks.map((chk) => (
            <div
              key={chk.id}
              className="rounded-xl border border-amber-400/45 bg-amber-500/12 backdrop-blur-xl p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5"
            >
              <div className="space-y-0.5">
                <div className="text-xs font-extrabold text-amber-200 flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>{isAr ? chk.question_ar : chk.question_en}</span>
                </div>
                <div className="text-[11px] text-amber-100/80">
                  {isAr ? chk.why_ar : chk.why_en}
                </div>
              </div>
              <button
                type="button"
                onClick={() => onApplyCheck(chk.id, chk.flagKey)}
                className="shrink-0 px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-black text-xs cursor-pointer"
              >
                {isAr ? "نعم — عدّل التشخيص" : "Yes — Update"}
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Computed Measurements Strip */}
      <div className={`rounded-2xl border p-2.5 ${cardCls}`}>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-center">
          {[
            { label: "HR", val: m.hr_bpm ? `${m.hr_bpm} bpm` : "—" },
            { label: "PR", val: m.pr_ms ? `${m.pr_ms} ms` : "—" },
            { label: "QRS", val: m.qrs_ms ? `${m.qrs_ms} ms` : "—" },
            { label: "QT", val: m.qt_ms ? `${m.qt_ms} ms` : "—" },
            {
              label: m.hr_bpm && (m.hr_bpm > 100 || m.hr_bpm < 50) ? "QTc (Frid)" : "QTc (Baz)",
              val:
                m.hr_bpm && (m.hr_bpm > 100 || m.hr_bpm < 50)
                  ? m.qtc_fridericia_ms ? `${m.qtc_fridericia_ms} ms` : "—"
                  : m.qtc_bazett_ms ? `${m.qtc_bazett_ms} ms` : "—",
            },
            { label: "Axis", val: m.axis_deg != null ? `${m.axis_deg}°` : "—" },
          ].map((item) => (
            <div key={item.label} className={`rounded-xl border px-2 py-1.5 ${subCardCls}`}>
              <div className="text-[10px] font-mono text-slate-400">{item.label}</div>
              <div className="text-xs font-extrabold font-mono text-cyan-400 mt-0.5">{item.val}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 2/3 Secondary Differential ONLY (with compact anchored floating Inv/TTT popover) + 1/3 Clinical Pitfalls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-stretch">
        {(secondary || isAfibCase || result.lookAlikeWarnings.length > 0) && (
          <div
            className={`rounded-2xl border p-4 space-y-3 flex flex-col relative ${
              result.aggregatedPitfalls.length > 0 ? "lg:col-span-8" : "lg:col-span-12"
            } ${cardCls}`}
          >
            {secondary && secondaryInvTtt && (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2 relative">
                  <h3 className="text-sm font-black flex items-center gap-2 text-cyan-300">
                    <Activity className="w-4 h-4 shrink-0" />
                    <span>{isAr ? "التشخيص الثاني المحتمل (التشخيص التفريقي)" : "Secondary / Differential Diagnosis"}</span>
                  </h3>

                  {/* Compact Floating Popover Trigger Buttons ONLY on Secondary */}
                  <div className="relative flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => setSecondaryPopover((cur) => (cur === "inv" ? null : "inv"))}
                      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-[11px] font-black cursor-pointer transition ${
                        secondaryPopover === "inv"
                          ? "bg-cyan-500/35 border-cyan-300 text-cyan-100"
                          : "bg-cyan-500/20 hover:bg-cyan-500/30 border-cyan-400/40 text-cyan-200"
                      }`}
                    >
                      <FlaskConical className="w-3 h-3" />
                      <span>{isAr ? "Inv" : "Inv"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setSecondaryPopover((cur) => (cur === "ttt" ? null : "ttt"))}
                      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-[11px] font-black cursor-pointer transition ${
                        secondaryPopover === "ttt"
                          ? "bg-emerald-500/35 border-emerald-300 text-emerald-100"
                          : "bg-emerald-500/20 hover:bg-emerald-500/30 border-emerald-400/40 text-emerald-200"
                      }`}
                    >
                      <Pill className="w-3 h-3" />
                      <span>{isAr ? "TTT" : "TTT"}</span>
                    </button>

                    {/* Compact Anchored Floating Panel (NOT full-screen modal!) */}
                    {secondaryPopover && (
                      <div
                        className={`absolute top-full mt-2 z-40 w-72 sm:w-96 rounded-xl border p-3.5 shadow-2xl backdrop-blur-xl ${
                          isAr ? "left-0" : "right-0"
                        } ${
                          secondaryPopover === "inv"
                            ? "bg-[#08172e]/95 border-cyan-400/60 text-slate-100"
                            : "bg-[#071f1a]/95 border-emerald-400/60 text-slate-100"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 pb-2 mb-2 border-b border-white/10">
                          <div className="flex items-center gap-1.5 text-xs font-black">
                            {secondaryPopover === "inv" ? (
                              <>
                                <FlaskConical className="w-3.5 h-3.5 text-cyan-300" />
                                <span className="text-cyan-200">
                                  {isAr ? "الفحوصات (Inv)" : "Investigations (Inv)"}
                                </span>
                              </>
                            ) : (
                              <>
                                <Pill className="w-3.5 h-3.5 text-emerald-300" />
                                <span className="text-emerald-200">
                                  {isAr ? "العلاج (TTT)" : "Treatment (TTT)"}
                                </span>
                              </>
                            )}
                          </div>
                          <button
                            type="button"
                            onClick={() => setSecondaryPopover(null)}
                            className="p-1 rounded-md bg-white/10 hover:bg-white/20 text-slate-200 cursor-pointer"
                            aria-label="Close"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </div>
                        <p className="text-xs leading-relaxed text-slate-100">
                          {secondaryPopover === "inv"
                            ? isAr
                              ? secondaryInvTtt.inv_ar
                              : secondaryInvTtt.inv_en
                            : isAr
                              ? secondaryInvTtt.ttt_ar
                              : secondaryInvTtt.ttt_en}
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                <div className={`rounded-xl border border-cyan-400/30 p-3.5 space-y-2 ${subCardCls}`}>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-extrabold text-sm">{isAr ? secondary.name_ar : secondary.name_en}</span>
                    <div className="flex items-center gap-1.5">
                      <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-400/30">
                        {sourceLabel(secondary.source)}
                      </span>
                      <span className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-slate-800/80 text-slate-200">
                        {secondary.urgency}
                      </span>
                    </div>
                  </div>
                  <p className={`text-xs leading-relaxed ${isLight ? "text-slate-600" : "text-slate-300"}`}>
                    {isAr ? secondary.explanation_ar : secondary.explanation_en}
                  </p>
                </div>
              </>
            )}

            {isAfibCase && (
              <div className="rounded-xl border border-sky-400/35 bg-sky-950/25 p-3 space-y-1.5 text-[11.5px]">
                <div className="font-black text-sky-300">
                  {isAr ? "بروتوكول طوارئ الرجفان الأذيني (AFIB Checklist):" : "ED Atrial Fibrillation Approach (AFIB Checklist):"}
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-slate-200">
                  <div><strong className="text-sky-300">A</strong> — {isAr ? "تأكد إنه AFib حقيقي وليس Flutter أو Artifact" : "Confirm AF vs Flutter / Artifact"}</div>
                  <div><strong className="text-sky-300">F</strong> — {isAr ? "ابحث عن سبب ثانوي (إنتان، جفاف، PE، نزيف) وعالجه أولاً" : "Fast from secondary cause (sepsis, PE, bleed, hypovolemia)?"}</div>
                  <div><strong className="text-sky-300">I</strong> — {isAr ? "صدمة متزامنة لغير المستقر، أو Rhythm/Rate للمستقر" : "Cardiovert if unstable; rhythm vs rate if stable"}</div>
                  <div><strong className="text-sky-300">B</strong> — {isAr ? "مضاد تجلط (DOAC) لو في عوامل CHADS-65 بدون موانع" : "Blood thinner (DOAC) if CHADS-65 risk factors"}</div>
                </div>
              </div>
            )}

            {result.lookAlikeWarnings.length > 0 && (
              <div className="rounded-xl border border-cyan-500/30 bg-cyan-950/25 p-3 space-y-1">
                <div className="text-xs font-extrabold text-cyan-300">
                  {isAr ? "تشخيصات مشابهة تم استبعادها بالسياق (Look-Alikes):" : "Ruled-Down Look-Alikes:"}
                </div>
                {result.lookAlikeWarnings.map((w) => (
                  <div key={w.code} className="text-xs text-slate-300">
                    • {isAr ? w.label_ar : w.label_en}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* 1/3 Width: Clinical Pitfalls & Traps */}
        {result.aggregatedPitfalls.length > 0 && (
          <div
            className={`${
              secondary || isAfibCase || result.lookAlikeWarnings.length > 0 ? "lg:col-span-4" : "lg:col-span-12"
            } rounded-2xl border border-amber-500/35 p-3.5 space-y-2.5 flex flex-col ${cardCls}`}
          >
            <h3 className="text-xs sm:text-sm font-black text-amber-400 flex items-center gap-1.5">
              <ShieldAlert className="w-4 h-4 shrink-0" />
              <span>{isAr ? "الفخاخ السريرية" : "Clinical Pitfalls & Traps"}</span>
            </h3>
            <div className="space-y-2 flex-1 overflow-y-auto pr-0.5">
              {result.aggregatedPitfalls.map((p, idx) => (
                <div key={idx} className="rounded-xl bg-amber-950/20 border border-amber-500/25 p-2.5 space-y-1 text-[11.5px]">
                  <div className="font-extrabold text-amber-300 leading-snug">⚠ {p.trap}</div>
                  <div className={`leading-snug ${isLight ? "text-slate-600" : "text-slate-300"}`}>
                    {p.why_it_misleads}
                  </div>
                  <div className="text-emerald-400 font-semibold leading-snug">✓ {p.how_to_avoid}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Compact Grad-CAM Visual Attention Panel directly beneath the ECG Image inside the scrollable column.
 */
export function EcgGradCamPanel({
  isAr,
  isLight,
  cardCls,
  gradcamUrl,
  gradcamFocusRegion,
}: {
  isAr: boolean;
  isLight: boolean;
  cardCls: string;
  gradcamUrl: string;
  gradcamFocusRegion?: string | null;
}) {
  return (
    <div className={`rounded-2xl border p-3 space-y-2 ${cardCls}`}>
      <div className="flex flex-wrap items-center justify-between gap-1.5">
        <div className="text-xs font-black flex items-center gap-1.5">
          <Flame className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{isAr ? "منطقة التركيز البصري (Grad-CAM)" : "Visual Focus Zone (Grad-CAM)"}</span>
        </div>
        {gradcamFocusRegion && (
          <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-rose-500/20 text-rose-200 border border-rose-400/35">
            {gradcamFocusRegion}
          </span>
        )}
      </div>

      <div className="overflow-hidden rounded-xl border border-white/10 bg-black/45 flex items-center justify-center h-48">
        <img
          src={gradcamUrl}
          alt="ECG Grad-CAM Heatmap"
          className="w-full h-full object-contain"
        />
      </div>

      {gradcamFocusRegion && (
        <p className={`text-[11px] leading-snug ${isLight ? "text-slate-600" : "text-slate-300"}`}>
          <strong className="text-rose-300">{isAr ? "الجزء المُركّز عليه: " : "Focused Segment: "}</strong>
          {gradcamFocusRegion}
        </p>
      )}
    </div>
  );
}
