"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FileText,
  ImagePlus,
  Loader2,
  RefreshCw,
  ScanLine,
  ShieldAlert,
  Sparkles,
  Stethoscope,
  Trash2,
} from "lucide-react";
import { ECG_UNITS } from "@/lib/ecgRules";
import { analyzeEcgImageLocally, LocalECGAnalysisResult } from "@/lib/ecgLocalAnalyzer";
import { useAtlasLocale } from "@/lib/atlasI18n";
import { useUIStore } from "@/stores/uiStore";

export function EcgDiagnosticStudio() {
  const { locale } = useAtlasLocale();
  const theme = useUIStore((s) => s.theme);
  const isAr = locale === "ar";
  const isLight = theme === "light";

  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string>("");
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<LocalECGAnalysisResult | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const runLocalDiagnosis = async (dataUrl: string) => {
    setAnalyzing(true);
    setErrorMsg(null);
    setAnalysis(null);
    try {
      await new Promise((r) => setTimeout(r, 70));
      const res = await analyzeEcgImageLocally(dataUrl);
      setAnalysis(res);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Image decode error";
      setErrorMsg(isAr ? `تعذر قراءة صورة رسم القلب: ${msg}` : `Could not analyze ECG image: ${msg}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setErrorMsg(isAr ? "يرجى رفع صورة رسم قلب بصيغة PNG أو JPG." : "Please upload an ECG image (PNG / JPG).");
      return;
    }
    setImageName(file.name);
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      setImagePreview(dataUrl);
      void runLocalDiagnosis(dataUrl);
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  };

  const clearAll = () => {
    setImagePreview(null);
    setImageName("");
    setAnalysis(null);
    setErrorMsg(null);
  };

  const cardClass = isLight
    ? "bg-white border-slate-200/90 text-slate-900 shadow-sm"
    : "bg-[#0c1426]/92 border-white/10 text-slate-100 shadow-lg shadow-black/30";

  const urgencyBadge = (u: string) => {
    if (u === "CRITICAL") return "bg-red-600 text-white border-red-400 shadow-red-600/30 shadow-md";
    if (u === "URGENT") return "bg-amber-500 text-slate-950 border-amber-300";
    if (u === "ABNORMAL") return "bg-sky-500/20 text-sky-300 border-sky-400/40";
    return "bg-emerald-500/20 text-emerald-300 border-emerald-400/40";
  };

  const result = analysis?.interpretation || null;

  return (
    <div
      dir={isAr ? "rtl" : "ltr"}
      className={`h-full w-full overflow-y-auto px-4 md:px-8 py-5 ${
        isLight ? "bg-slate-50 text-slate-900" : "bg-[#050912] text-slate-100"
      }`}
    >
      <div className="max-w-5xl mx-auto space-y-5">
        {/* Sleek Top Header */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-rose-600/25 to-rose-400/10 border border-rose-500/35 flex items-center justify-center text-rose-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-black text-base sm:text-lg tracking-tight">
                {isAr ? "محرك تشخيص رسم القلب السريري" : "Clinical ECG Diagnostic Engine"}
              </h2>
              <p className={`text-xs ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                {isAr
                  ? `ارفع صورة رسم القلب فقط • معايرة ${ECG_UNITS.paper_speed_mm_per_sec} mm/s • تحليل محلي فوري`
                  : `Upload any ECG tracing • ${ECG_UNITS.paper_speed_mm_per_sec} mm/s standard calibration • Instant local reading`}
              </p>
            </div>
          </div>

          {imagePreview && (
            <div className="flex items-center gap-2">
              <label className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-bold transition cursor-pointer">
                <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                <ImagePlus className="w-3.5 h-3.5 text-rose-400" />
                {isAr ? "صورة جديدة" : "New ECG"}
              </label>
              <button
                type="button"
                disabled={analyzing}
                onClick={() => void runLocalDiagnosis(imagePreview)}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 disabled:opacity-60 text-white text-xs font-bold transition cursor-pointer"
              >
                {analyzing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                {isAr ? "إعادة الفحص" : "Re-analyze"}
              </button>
              <button
                type="button"
                onClick={clearAll}
                className="p-2 rounded-xl border border-white/15 hover:bg-rose-500/15 hover:text-rose-400 text-xs transition cursor-pointer"
                title={isAr ? "مسح" : "Clear"}
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        {/* Empty Upload State (Centered Hero Dropzone) */}
        {!imagePreview && (
          <div className={`rounded-3xl border p-6 sm:p-10 ${cardClass}`}>
            <label
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`block rounded-2xl border-2 border-dashed py-14 px-6 text-center cursor-pointer transition ${
                isDragging
                  ? "border-rose-400 bg-rose-500/15"
                  : "border-rose-400/35 hover:border-rose-400/75 hover:bg-rose-500/5"
              }`}
            >
              <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-rose-500/20 to-rose-700/10 border border-rose-400/35 flex items-center justify-center text-rose-400">
                <ScanLine className="w-8 h-8" />
              </div>
              <div className="text-base sm:text-lg font-black">
                {isAr ? "اسحب صورة رسم القلب هنا أو اضغط لاختيار ملف" : "Drop ECG Image Here or Click to Upload"}
              </div>
              <p className="text-xs sm:text-sm opacity-70 max-w-md mx-auto mt-2 leading-relaxed">
                {isAr
                  ? "يدعم رسم القلب الكامل (12-Lead ECG) وشرائط النظم الأحادية (Rhythm Strip) بصيغة PNG أو JPG — يقرأ الموجة مباشرة ويُخرج الحالات الطارئة أولاً مع الشرح والتصرف الطبي."
                  : "Supports both full 12-Lead ECGs and single-lead rhythm strips (PNG / JPG) — direct waveform analysis with urgent-first clinical reporting."}
              </p>
            </label>

            {errorMsg && (
              <div className="mt-4 rounded-xl border border-amber-500/45 bg-amber-500/10 p-3 text-xs text-amber-300 text-center">
                {errorMsg}
              </div>
            )}
          </div>
        )}

        {/* Active Diagnosis Workspace (Panoramic ECG Top + Organized Report Below) */}
        {imagePreview && (
          <div className="space-y-4">
            {/* 1. Urgent / Normal Status Banner */}
            {result && (
              result.has_urgent_findings ? (
                <div className="rounded-2xl border-2 border-red-500/75 bg-gradient-to-r from-red-950/80 via-red-900/50 to-red-950/80 px-4 py-3.5 text-red-100 shadow-lg shadow-red-950/40 flex items-center gap-3">
                  <ShieldAlert className="w-5 h-5 text-red-400 animate-pulse shrink-0" />
                  <span className="font-black text-sm sm:text-base">
                    {isAr
                      ? `تنبيه طوارئ: تم اكتشاف ${result.urgent_count} حالة حرجة/عاجلة تستدعي الانتباه الفوري`
                      : `URGENT ALERT: ${result.urgent_count} critical/urgent finding(s) detected — immediate attention required`}
                  </span>
                </div>
              ) : (
                <div className="rounded-2xl border border-emerald-500/40 bg-emerald-950/30 px-4 py-3 text-emerald-200 flex items-center gap-2.5 text-xs sm:text-sm font-bold">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  {isAr ? "لا توجد علامات طوارئ حرجة مكتشفة في الرسم" : "No immediate life-threatening emergencies detected"}
                </div>
              )
            )}

            {/* 2. Panoramic ECG Tracing Viewer */}
            <div className={`rounded-2xl border p-4 space-y-3 ${cardClass}`}>
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="font-extrabold flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-rose-400" />
                  {isAr ? "صورة رسم القلب المرفوعة" : "Uploaded ECG Tracing"}
                </span>
                <div className="flex items-center gap-2">
                  {analysis && (
                    <span className="px-2.5 py-0.5 rounded-full text-[10.5px] font-bold bg-rose-500/15 text-rose-300 border border-rose-400/30">
                      {analysis.layout === "single_lead"
                        ? isAr ? "شريط نظم أحادي (Rhythm Strip)" : "Single-Lead Rhythm Strip"
                        : isAr ? "رسم قلب 12 ليد (12-Lead ECG)" : "12-Lead ECG"}
                    </span>
                  )}
                  {imageName && <span className="text-[11px] opacity-60 truncate max-w-[180px]">{imageName}</span>}
                </div>
              </div>

              <div className="rounded-xl overflow-hidden bg-white p-2 border border-white/10">
                <img
                  src={imagePreview}
                  alt="Uploaded ECG"
                  className="w-full max-h-[270px] object-contain mx-auto"
                />
              </div>

              {analyzing && (
                <div className="rounded-xl border border-rose-400/35 bg-rose-500/10 p-3 flex items-center justify-center gap-2.5 text-xs font-bold text-rose-300">
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                  <span>{isAr ? "جاري قراءة الموجة وتحليل النظم والفترات..." : "Digitizing ECG waveform & evaluating criteria..."}</span>
                </div>
              )}

              {errorMsg && (
                <div className="rounded-xl border border-amber-500/45 bg-amber-500/10 p-3 text-xs text-amber-300">
                  {errorMsg}
                </div>
              )}
            </div>

            {/* 3. Clinical Telemetry Strip */}
            {result && analysis && (
              <>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
                  {[
                    { label: isAr ? "معدل النبض (HR)" : "Heart Rate", value: result.measurements.hr_bpm ? `${result.measurements.hr_bpm} bpm` : "—" },
                    { label: isAr ? "مسافة PR" : "PR Interval", value: result.measurements.pr_ms ? `${result.measurements.pr_ms} ms` : "—" },
                    { label: isAr ? "عرض QRS" : "QRS Duration", value: result.measurements.qrs_ms ? `${result.measurements.qrs_ms} ms` : "—" },
                    { label: "QTc (Bazett)", value: result.measurements.qtc_bazett_ms ? `${result.measurements.qtc_bazett_ms} ms` : "—" },
                    { label: isAr ? "محور القلب" : "QRS Axis", value: result.measurements.axis_deg != null ? `${result.measurements.axis_deg}°` : "—" },
                  ].map((m) => (
                    <div key={m.label} className={`rounded-2xl border px-3 py-3 text-center ${cardClass}`}>
                      <div className="text-[10.5px] opacity-65 font-semibold">{m.label}</div>
                      <div className="text-sm sm:text-base font-black mt-1 tracking-tight">{m.value}</div>
                    </div>
                  ))}
                </div>

                {/* Concise Clinical Summary */}
                <div className={`rounded-2xl border px-4 py-3 text-xs sm:text-sm leading-relaxed flex items-start gap-2.5 ${cardClass}`}>
                  <FileText className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                  <span>{isAr ? analysis.summary_ar : analysis.summary_en}</span>
                </div>

                {/* 4. Structured Diagnostic Cards (Urgent First) */}
                <div className="space-y-3.5 pb-4">
                  {result.all_findings.map((f, idx) => (
                    <div key={f.code} className={`rounded-2xl border p-5 space-y-3.5 ${cardClass}`}>
                      <div className="flex flex-wrap items-start justify-between gap-2 pb-2.5 border-b border-white/10">
                        <div className="space-y-0.5">
                          <div className="flex items-center gap-2">
                            <Stethoscope className="w-4 h-4 text-rose-400 shrink-0" />
                            <h3 className="font-black text-base sm:text-lg">
                              {idx + 1}. {isAr ? f.name_ar : f.name_en}
                            </h3>
                          </div>
                          <div className="text-xs font-mono opacity-65">{isAr ? f.name_en : f.name_ar}</div>
                        </div>
                        <span className={`px-3 py-1 rounded-full text-[10.5px] font-black border ${urgencyBadge(f.urgency)}`}>
                          {isAr ? f.urgency_label_ar : f.urgency_label_en}
                        </span>
                      </div>

                      {f.evidence.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {f.evidence.map((ev, i) => (
                            <span
                              key={i}
                              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border ${
                                isLight
                                  ? "bg-slate-100 border-slate-200 text-slate-700"
                                  : "bg-white/[0.04] border-white/10 text-slate-200"
                              }`}
                            >
                              {ev}
                            </span>
                          ))}
                        </div>
                      )}

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div className={`rounded-xl p-3.5 text-xs sm:text-[13px] leading-relaxed ${isLight ? "bg-slate-50 border border-slate-200" : "bg-white/[0.03] border border-white/10"}`}>
                          <div className="font-extrabold text-cyan-400 mb-1">
                            {isAr ? "الشرح الإكلينيكي المفصل" : "Clinical Explanation"}
                          </div>
                          {isAr ? f.explanation_ar : f.explanation_en}
                        </div>

                        <div
                          className={`rounded-xl border p-3.5 text-xs sm:text-[13px] leading-relaxed ${
                            f.urgency === "CRITICAL" || f.urgency === "URGENT"
                              ? "border-rose-500/40 bg-rose-500/10"
                              : isLight
                              ? "border-slate-200 bg-slate-50"
                              : "border-white/10 bg-white/[0.03]"
                          }`}
                        >
                          <div className="font-extrabold inline-flex items-center gap-1.5 text-amber-400 mb-1">
                            <AlertTriangle className="w-3.5 h-3.5" />
                            {isAr ? "التصرف الطبي العاجل" : "Immediate Management"}
                          </div>
                          <div>{isAr ? f.immediate_action_ar : f.immediate_action_en}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
