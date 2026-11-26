"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  Download,
  Loader2,
  Flame,
  FlaskConical,
  Minus,
  Pill,
  Plus,
  ShieldAlert,
  X,
} from "lucide-react";
import { HybridDiagnosisResult, WizardFeatures } from "@/lib/ecgHybridEngine";
import { getClinicalInvTtt } from "@/lib/ecgClinicalLibrary";
import { downloadEcgReportSlide } from "@/lib/ecgReportSlide";

interface Props {
  result: HybridDiagnosisResult;
  isAr: boolean;
  isLight: boolean;
  cardCls: string;
  onEditInputs: () => void;
  onApplyCheck: (checkId: string, flagKey: keyof WizardFeatures) => void;
  measurementSource?: "device_ocr" | "auto_waveform" | "manual" | null;
  showCalibratedPercent?: boolean;
  ecgImageDataUrl?: string | null;
}

export function EcgResultsPanel({
  result,
  isAr,
  isLight,
  cardCls,
  onEditInputs,
  onApplyCheck,
  measurementSource,
  showCalibratedPercent = false,
  ecgImageDataUrl = null,
}: Props) {
  const [exportingSlide, setExportingSlide] = useState(false);
  // Expandable inline Inv/TTT drawer for each Co-Existing Confirmed Diagnosis inside the main panel
  const [expandedCoDx, setExpandedCoDx] = useState<string | null>(null);

  const urgencyBannerStyle = (u: string, insufficient: boolean) => {
    if (insufficient) {
      return isLight
        ? "bg-amber-100/90 border-amber-500/80 text-amber-950"
        : "bg-amber-500/18 border-amber-400/70 text-amber-50";
    }
    if (u === "CRITICAL") {
      return isLight
        ? "bg-red-100/90 border-red-500/80 text-red-950"
        : "bg-red-500/20 border-red-500/70 text-red-50";
    }
    if (u === "URGENT") {
      return isLight
        ? "bg-amber-100/85 border-amber-500/75 text-amber-950"
        : "bg-amber-500/18 border-amber-400/65 text-amber-50";
    }
    if (u === "NORMAL") {
      return isLight
        ? "bg-teal-100/85 border-teal-500/70 text-teal-950"
        : "bg-teal-500/16 border-teal-400/55 text-teal-50";
    }
    return isLight
      ? "bg-sky-100/85 border-sky-500/70 text-slate-900"
      : "bg-sky-500/16 border-sky-400/55 text-sky-50";
  };

  // 3-Level Evidence Meter (Low / Moderate / High — never color-only, respects showCalibratedPercent)
  const renderEvidenceMeter = (f: typeof primary) => {
    if (!f || !("concordanceLevel" in f)) return null;
    const level = f.concordanceLevel === "strong" ? 3 : f.concordanceLevel === "moderate" ? 2 : 1;
    const label =
      level === 3
        ? isAr ? "دليل قوي" : "High Evidence"
        : level === 2
          ? isAr ? "دليل متوسط" : "Moderate Evidence"
          : isAr ? "دليل مبدئي" : "Low Evidence";
    const pct = "fusionConfidence" in f && f.fusionConfidence > 0 ? Math.round(f.fusionConfidence * 100) : null;
    const barColor =
      level === 3 ? "bg-teal-400" : level === 2 ? "bg-amber-400" : "bg-slate-400";
    return (
      <span
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10.5px] font-bold border ${
          isLight ? "bg-white/75 border-slate-300 text-slate-800" : "bg-black/30 border-white/15 text-slate-100"
        }`}
      >
        <span className="inline-flex items-end gap-0.5 h-3" aria-hidden="true">
          {[1, 2, 3].map((idx) => (
            <span
              key={idx}
              className={`w-1 rounded-full ${idx === 1 ? "h-1.5" : idx === 2 ? "h-2.5" : "h-3"} ${
                idx <= level ? barColor : isLight ? "bg-slate-300" : "bg-white/20"
              }`}
            />
          ))}
        </span>
        <span>{label}</span>
        {showCalibratedPercent && pct !== null && <span className="font-mono opacity-85">({pct}%)</span>}
      </span>
    );
  };

  const m = result.rulesResult.measurements;
  const subCardCls = isLight
    ? "border-slate-200/90 bg-white/75 text-slate-900"
    : "border-white/10 bg-[#0a1224]/65 text-slate-100";

  // Certainty-Gated Multi-Diagnosis: Primary + all additional CONFIRMED co-existing diagnoses (up to 4 total; zero unverified)
  const activeOnly = result.primaryFindings.filter((f) => !("isDemoted" in f && f.isDemoted));
  const primary = activeOnly[0] ?? result.primaryFindings[0];
  const isInsufficient = primary?.code === "INSUFFICIENT_DATA";
  const coExisting = isInsufficient
    ? []
    : activeOnly.slice(1, 4).filter((f) => !("isConfirmed" in f) || f.isConfirmed);

  const isAfibCase = [primary, ...coExisting].some(
    (f) =>
      f &&
      ["ATRIAL_FIBRILLATION", "ATRIAL_FIBRILLATION_RVR", "ATRIAL_FLUTTER", "SECONDARY_AFIB_RVR", "SECONDARY_AFIB_SEPSIS_FEVER"].includes(
        f.code
      )
  );

  const primaryInvTtt = primary
    ? getClinicalInvTtt(primary.code, primary.immediate_action_en, primary.immediate_action_ar, primary.name_en)
    : null;

  const sourceLabel = (src: string) => {
    if (src === "context_override") return isAr ? "سياق المريض والفايتلز" : "Vitals & Context";
    if (src === "ai_model") return isAr ? "الذكاء الاصطناعي البصري" : "Visual AI";
    if (src === "case_kb") return isAr ? "حالة مرجعية" : "Reference Case";
    return isAr ? "محرك القواعد" : "Rules Engine";
  };

  const concordancePill = (f: typeof primary) => {
    if (!f || !("concordanceLevel" in f)) return null;
    const cls =
      f.concordanceLevel === "strong"
        ? "bg-emerald-500/20 text-emerald-200 border-emerald-400/40"
        : f.concordanceLevel === "moderate"
          ? "bg-amber-500/20 text-amber-200 border-amber-400/40"
          : "bg-slate-700/60 text-slate-200 border-slate-500/40";
    return (
      <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${cls}`}>
        {isAr ? f.concordanceLabelAr : f.concordanceLabelEn}
      </span>
    );
  };

  const measurementSourceBadge =
    measurementSource === "device_ocr"
      ? isAr
        ? "قراءة الجهاز — تُراجع سريرياً"
        : "Device Reading — Verify Clinically"
      : measurementSource === "auto_waveform"
        ? isAr
          ? "تقدير آلي من الموجة"
          : "Auto Waveform Estimate"
        : measurementSource === "manual"
          ? isAr
            ? "إدخال يدوي"
            : "Manual Entry"
          : null;

  return (
    <div className="space-y-3">
      {/* TOP PRIMARY DIAGNOSIS BANNER — Clean inline Inv & TTT panels (no redundant top buttons) */}
      {primary && primaryInvTtt && (
        <div
          className={`cb-glass-card px-5 py-4 space-y-3 ${urgencyBannerStyle(
            result.finalUrgency,
            isInsufficient
          )}`}
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex items-start gap-3 min-w-0 flex-1">
              {isInsufficient ? (
                <AlertTriangle className={`w-6 h-6 shrink-0 mt-0.5 ${isLight ? "text-amber-700" : "text-amber-300"}`} />
              ) : result.finalUrgency === "CRITICAL" ? (
                <AlertOctagon className={`w-6 h-6 shrink-0 mt-0.5 ${isLight ? "text-red-700" : "text-red-300"}`} />
              ) : result.finalUrgency === "URGENT" ? (
                <AlertTriangle className={`w-6 h-6 shrink-0 mt-0.5 ${isLight ? "text-amber-700" : "text-amber-300"}`} />
              ) : (
                <CheckCircle2 className={`w-6 h-6 shrink-0 mt-0.5 ${isLight ? "text-teal-700" : "text-teal-300"}`} />
              )}
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[10px] font-mono font-black uppercase tracking-wider px-2 py-0.5 rounded bg-black/30">
                    {result.finalUrgency}
                  </span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-white/15">
                    {sourceLabel(primary.source)}
                  </span>
                  {!isInsufficient && renderEvidenceMeter(primary)}
                </div>
                <div className="font-black text-base sm:text-lg leading-snug mt-1">
                  {isAr ? primary.name_ar : primary.name_en}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 shrink-0">
              <button
                type="button"
                disabled={exportingSlide}
                onClick={async () => {
                  setExportingSlide(true);
                  try {
                    await downloadEcgReportSlide({
                      result,
                      isAr,
                      ecgImageDataUrl,
                      measurementSource,
                    });
                  } finally {
                    setExportingSlide(false);
                  }
                }}
                className="cb-btn-primary inline-flex items-center gap-1.5 min-h-[44px] px-3.5 py-2 rounded-full text-xs font-extrabold cursor-pointer shrink-0 disabled:opacity-60"
              >
                {exportingSlide ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                <span>{isAr ? "تحميل تقرير سلايد" : "Download Slide Report"}</span>
              </button>
              <button
                type="button"
                onClick={onEditInputs}
                className={`min-h-[44px] px-3.5 py-2 rounded-full border text-xs font-bold cursor-pointer shrink-0 cb-hover-lift ${isLight ? "bg-white/75 border-slate-300 text-slate-900 hover:bg-white" : "bg-black/30 hover:bg-black/45 border-white/20 text-white"}`}
              >
                {isAr ? "تعديل الاختيارات" : "Edit Inputs"}
              </button>
            </div>
          </div>

          {/* Mandatory Safety Warning on Insufficient Certainty (never looks normal/green) */}
          {isInsufficient && (
            <div className={`rounded-xl border px-3.5 py-2.5 text-xs font-extrabold flex items-center gap-2 ${isLight ? "bg-amber-200/70 border-amber-500/70 text-amber-950" : "bg-amber-950/55 border-amber-400/60 text-amber-100"}`}>
              <ShieldAlert className="w-4 h-4 shrink-0" />
              <span>
                {isAr
                  ? "هذا لا يستبعد وجود حالة طارئة — راجع التخطيط يدوياً أو صعّد الحالة فوراً."
                  : "This does NOT rule out an emergency — verify manually / escalate."}
              </span>
            </div>
          )}

          {/* Primary Textbook Clinical Explanation */}
          <p className="text-xs sm:text-[13px] leading-relaxed opacity-95">
            {isAr ? primary.explanation_ar : primary.explanation_en}
          </p>

          {/* Full Inline Inv & TTT Panels directly beneath Primary Diagnosis (hidden on INSUFFICIENT_DATA) */}
          {!isInsufficient && (
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
          )}

          {/* Co-Existing Confirmed Diagnoses INSIDE the same Primary Panel (Name Only + '+' + Certainty %) */}
          {!isInsufficient && coExisting.length > 0 && (
            <div className="pt-2 mt-1 border-t border-white/15 space-y-1.5">
              <div className="text-[11px] font-black uppercase tracking-wide opacity-90 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 shrink-0" />
                <span>
                  {isAr ? "تشخيصات أخرى مؤكدة في نفس التخطيط" : "Other Confirmed Diagnoses"}
                </span>
              </div>
              <div className="space-y-1.5">
                {coExisting.map((dx) => {
                  const isOpen = expandedCoDx === dx.code;
                  const dxInvTtt = getClinicalInvTtt(dx.code, dx.immediate_action_en, dx.immediate_action_ar, dx.name_en);
                  const certPct = "fusionConfidence" in dx ? Math.round(dx.fusionConfidence * 100) : null;
                  return (
                    <div key={dx.code} className="rounded-xl bg-black/25 border border-white/15 overflow-hidden">
                      <button
                        type="button"
                        onClick={() => setExpandedCoDx((cur) => (cur === dx.code ? null : dx.code))}
                        className="w-full min-h-[44px] px-3.5 py-2 flex items-center justify-between gap-2 text-start hover:bg-white/5 transition cursor-pointer"
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="w-5 h-5 rounded-md bg-white/15 border border-white/25 flex items-center justify-center shrink-0">
                            {isOpen ? <Minus className="w-3 h-3" /> : <Plus className="w-3 h-3" />}
                          </span>
                          <span className="font-extrabold text-xs sm:text-[13px] truncate">
                            {isAr ? dx.name_ar : dx.name_en}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          {renderEvidenceMeter(dx)}
                        </div>
                      </button>

                      {isOpen && (
                        <div className="px-3 pb-3 pt-1 border-t border-white/10 space-y-2">
                          <p className="text-[11.5px] leading-relaxed opacity-90">
                            {isAr ? dx.explanation_ar : dx.explanation_en}
                          </p>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                            <div className="rounded-lg bg-black/30 border border-cyan-300/30 px-2.5 py-2 space-y-0.5">
                              <div className="text-[10.5px] font-black text-cyan-200 flex items-center gap-1">
                                <FlaskConical className="w-3 h-3 shrink-0" />
                                <span>{isAr ? "الفحوصات (Inv)" : "Investigations (Inv)"}</span>
                              </div>
                              <p className="text-[11.5px] leading-relaxed opacity-95">
                                {isAr ? dxInvTtt.inv_ar : dxInvTtt.inv_en}
                              </p>
                            </div>
                            <div className="rounded-lg bg-black/30 border border-emerald-300/30 px-2.5 py-2 space-y-0.5">
                              <div className="text-[10.5px] font-black text-emerald-200 flex items-center gap-1">
                                <Pill className="w-3 h-3 shrink-0" />
                                <span>{isAr ? "العلاج (TTT)" : "Treatment (TTT)"}</span>
                              </div>
                              <p className="text-[11.5px] leading-relaxed opacity-95">
                                {isAr ? dxInvTtt.ttt_ar : dxInvTtt.ttt_en}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Explicit Source Conflict Banner (Requirement 5: never silently outvote) */}
      {result.sourceConflict && (
        <div className="rounded-xl border border-amber-400/60 bg-amber-500/15 backdrop-blur-xl px-4 py-2.5 flex items-start gap-2.5 text-amber-100">
          <AlertTriangle className="w-5 h-5 text-amber-300 shrink-0 mt-0.5" />
          <div className="text-xs leading-relaxed">
            <div className="font-black text-amber-200">
              {isAr ? "تعارض بين التقدير الآلي والذكاء البصري — تحقق من القياس" : "Conflict — Verify Measurement"}
            </div>
            <div className="opacity-95 mt-0.5">
              {isAr ? result.sourceConflict.messageAr : result.sourceConflict.messageEn}
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

      {/* Computed Measurements Strip with explicit Device/Auto source disclaimer */}
      <div className={`rounded-2xl border p-2.5 space-y-2 ${cardCls}`}>
        {measurementSourceBadge && (
          <div className="flex items-center justify-between text-[10.5px] px-1">
            <span className="font-bold text-slate-400">{isAr ? "مصدر القياسات:" : "Measurement Source:"}</span>
            <span className="px-2 py-0.5 rounded-md font-mono font-bold bg-amber-500/15 text-amber-200 border border-amber-400/30">
              {measurementSourceBadge}
            </span>
          </div>
        )}
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-center">
          {[
            {
              label: "HR",
              val: m.hr_bpm
                ? measurementSource === "auto_waveform"
                  ? `~${Math.max(25, Math.round(m.hr_bpm) - 28)}–${Math.round(m.hr_bpm) + 28} bpm`
                  : `${m.hr_bpm} bpm`
                : "—",
            },
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
              <div className="text-xs font-extrabold font-mono text-teal-400 mt-0.5">{item.val}</div>
            </div>
          ))}
        </div>
        {measurementSource !== "device_ocr" && measurementSource !== "manual" && m.pr_ms == null && m.qrs_ms == null && m.qt_ms == null && (
          <p className={`text-[11px] px-1 ${isLight ? "text-slate-600" : "text-slate-400"}`}>
            {isAr
              ? "فترات PR و QRS و QT غير قابلة للقياس الموثوق من الصورة — أدخل أرقام الهيدر المطبوعة أو قِسها يدوياً."
              : "PR, QRS, and QT are not reliably measurable from raw image pixels — enter printed header values or measure manually."}
          </p>
        )}
      </div>

      {/* Context Checklists (AFIB / Look-Alikes) + Clinical Pitfalls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-stretch">
        {(isAfibCase || result.lookAlikeWarnings.length > 0) && (
          <div
            className={`rounded-2xl border p-4 space-y-3 flex flex-col relative ${
              result.aggregatedPitfalls.length > 0 ? "lg:col-span-8" : "lg:col-span-12"
            } ${cardCls}`}
          >
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
              isAfibCase || result.lookAlikeWarnings.length > 0 ? "lg:col-span-4" : "lg:col-span-12"
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

      {/* Explicit Educational & Clinical Decision-Support Notice */}
      <p className={`text-[11px] px-2 text-center ${isLight ? "text-slate-600" : "text-slate-400"}`}>
        {isAr
          ? "للتعليم الطبي ودعم القرار فقط — لا يُعدّ بديلاً عن قراءة الطبيب اليدوية أو التقييم السريري المباشر."
          : "For medical education and decision support only — not a substitute for clinician ECG interpretation or direct clinical assessment."}
      </p>
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

      <div className="overflow-hidden rounded-xl border border-white/10 bg-black/45">
        <img
          src={gradcamUrl}
          alt="ECG Grad-CAM Heatmap"
          className="w-full h-auto object-contain"
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
