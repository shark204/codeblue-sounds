"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  ImagePlus,
  Loader2,
  RefreshCw,
  ScanLine,
  ShieldAlert,
  Sparkles,
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
      await new Promise((r) => setTimeout(r, 80));
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
    ? "bg-white/95 border-slate-200 text-slate-900 shadow-sm"
    : "bg-[#0d1528]/90 border-white/10 text-slate-100";

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
      className={`h-full w-full overflow-y-auto px-4 md:px-6 py-4 ${
        isLight ? "bg-slate-50/90 text-slate-900" : "bg-[#060a14] text-slate-100"
      }`}
    >
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Clean Header Bar */}
        <div className={`rounded-2xl border p-4 flex flex-wrap items-center justify-between gap-3 ${cardClass}`}>
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-rose-500/15 border border-rose-400/35 flex items-center justify-center text-rose-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="font-black text-sm sm:text-base">
                  {isAr ? "محرك تشخيص رسم القلب السريري" : "Clinical ECG Diagnostic Engine"}
                </h2>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-500/15 text-emerald-400 border border-emerald-400/30">
                  <Cpu className="w-3 h-3" />
                  {isAr ? "تحليل محلي مباشر • بدون توكنز" : "Local Waveform Engine • 0 Tokens"}
                </span>
              </div>
              <p className={`text-[11px] mt-0.5 ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                {isAr
                  ? `ارفع صورة رسم القلب فقط — قراءة رقمية مباشرة للموجة وتطبيق قواعد ecg_criteria.json (${ECG_UNITS.paper_speed_mm_per_sec} mm/s)`
                  : `Upload your ECG image only — direct 1D waveform digitization & ecg_criteria.json diagnostic rules (${ECG_UNITS.paper_speed_mm_per_sec} mm/s)`}
              </p>
            </div>
          </div>

          {imagePreview && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={analyzing}
                onClick={() => void runLocalDiagnosis(imagePreview)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 disabled:opacity-60 text-white text-xs font-bold transition cursor-pointer"
              >
                {analyzing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                {isAr ? "إعادة التحليل" : "Re-analyze"}
              </button>
              <button
                type="button"
                onClick={clearAll}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-white/15 hover:bg-white/10 text-xs font-bold transition cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                {isAr ? "مسح" : "Clear"}
              </button>
            </div>
          )}
        </div>

        {/* Clean 2-Column Workspace (Image Left | Organized Clinical Report Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          {/* LEFT: Clean ECG Image Card Only */}
          <div className="lg:col-span-5">
            <div className={`rounded-2xl border p-4 space-y-3 ${cardClass}`}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold flex items-center gap-1.5">
                  <ImagePlus className="w-4 h-4 text-rose-400" />
                  {isAr ? "صورة رسم القلب (12-Lead ECG)" : "ECG Tracing Image"}
                </span>
                {imageName && <span className="text-[10.5px] opacity-70 truncate max-w-[160px]">{imageName}</span>}
              </div>

              <label
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                className={`block rounded-2xl border-2 border-dashed p-4 text-center cursor-pointer transition ${
                  isDragging
                    ? "border-rose-400 bg-rose-500/15"
                    : "border-rose-400/40 hover:border-rose-400/75 hover:bg-rose-500/5"
                }`}
              >
                <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                {imagePreview ? (
                  <div className="space-y-2.5">
                    <img
                      src={imagePreview}
                      alt="Uploaded ECG"
                      className="w-full max-h-[420px] object-contain rounded-xl mx-auto bg-white"
                    />
                    <div className="text-[11px] opacity-70">
                      {isAr ? "اضغط أو اسحب صورة جديدة لاستبدال رسم القلب" : "Click or drop another image to replace ECG"}
                    </div>
                  </div>
                ) : (
                  <div className="py-14 px-4 space-y-2.5">
                    <div className="w-12 h-12 mx-auto rounded-2xl bg-rose-500/15 border border-rose-400/35 flex items-center justify-center text-rose-400">
                      <ScanLine className="w-6 h-6" />
                    </div>
                    <div className="text-sm font-black">
                      {isAr ? "اسحب صورة رسم القلب هنا أو اضغط للرفع" : "Drop ECG Image Here or Click to Upload"}
                    </div>
                    <p className="text-xs opacity-70 max-w-xs mx-auto leading-relaxed">
                      {isAr
                        ? "ارفع صورة الـ ECG ليقوم المحرك بقراءة الموجة مباشرة وتطبيق معايير التشخيص وترتيب الحالات الطارئة أولاً."
                        : "Upload an ECG tracing for direct waveform analysis and urgent-first clinical reporting."}
                    </p>
                  </div>
                )}
              </label>

              {analyzing && (
                <div className="rounded-xl border border-rose-400/35 bg-rose-500/10 p-3 flex items-center gap-2.5 text-xs font-bold text-rose-300">
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                  <span>{isAr ? "جاري قراءة الموجة وتحليل النظم والفترات..." : "Digitizing ECG waveform & evaluating criteria..."}</span>
                </div>
              )}

              {errorMsg && (
                <div className="rounded-xl border border-amber-500/45 bg-amber-500/10 p-3 text-xs text-amber-300 leading-relaxed">
                  {errorMsg}
                </div>
              )}
            </div>
          </div>

          {/* RIGHT: Clean Urgent-First Clinical Report */}
          <div className="lg:col-span-7 space-y-3.5">
            {!result && !analyzing && (
              <div className={`rounded-2xl border p-12 text-center space-y-3 ${cardClass}`}>
                <Activity className="w-10 h-10 mx-auto text-rose-400/80" />
                <h3 className="font-black text-base">
                  {isAr ? "بانتظار صورة رسم القلب للبدء" : "Ready for ECG Analysis"}
                </h3>
                <p className="text-xs opacity-75 max-w-md mx-auto leading-relaxed">
                  {isAr
                    ? "ارفع صورة الـ ECG من اللوحة الجانبية للحصول على التشخيص الفوري مع ترتيب الحالات الحرجة أولاً والشرح الإكلينيكي وخطة التصرف الطبي."
                    : "Upload an ECG image on the left to receive an instant urgent-first diagnosis with clinical explanations and management."}
                </p>
              </div>
            )}

            {result && analysis && (
              <>
                {result.has_urgent_findings ? (
                  <div className="rounded-2xl border-2 border-red-500/75 bg-red-950/55 p-3.5 text-red-100 shadow-lg shadow-red-900/30">
                    <div className="flex items-center gap-2 font-black text-sm sm:text-base text-red-300">
                      <ShieldAlert className="w-5 h-5 animate-pulse shrink-0" />
                      {isAr
                        ? `تنبيه طوارئ: تم اكتشاف ${result.urgent_count} حالة حرجة/عاجلة تستدعي تدخلاً فورياً!`
                        : `URGENT ALERT: ${result.urgent_count} critical/urgent finding(s) detected — immediate action required!`}
                    </div>
                  </div>
                ) : (
                  <div className="rounded-2xl border border-emerald-500/40 bg-emerald-950/30 p-3 text-emerald-200 flex items-center gap-2 text-xs font-bold">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    {isAr ? "لا توجد علامات طوارئ حرجة مكتشفة في الرسم" : "No immediate life-threatening emergencies detected"}
                  </div>
                )}

                {/* Concise Measured Telemetry Strip */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                  {[
                    { label: "HR", value: result.measurements.hr_bpm ? `${result.measurements.hr_bpm} bpm` : "—" },
                    { label: "PR", value: result.measurements.pr_ms ? `${result.measurements.pr_ms} ms` : "—" },
                    { label: "QRS", value: result.measurements.qrs_ms ? `${result.measurements.qrs_ms} ms` : "—" },
                    { label: "QTc (Bazett)", value: result.measurements.qtc_bazett_ms ? `${result.measurements.qtc_bazett_ms} ms` : "—" },
                    { label: "Axis", value: result.measurements.axis_deg != null ? `${result.measurements.axis_deg}°` : "—" },
                  ].map((m) => (
                    <div key={m.label} className={`rounded-xl border p-2.5 text-center ${cardClass}`}>
                      <div className="text-[10px] opacity-65 font-bold">{m.label}</div>
                      <div className="text-xs sm:text-sm font-black mt-0.5">{m.value}</div>
                    </div>
                  ))}
                </div>

                {/* Diagnostic Summary */}
                <div className={`rounded-xl border px-3.5 py-2.5 text-xs leading-relaxed ${cardClass}`}>
                  {isAr ? analysis.summary_ar : analysis.summary_en}
                </div>

                {/* Findings Cards (Urgent First) */}
                <div className="space-y-3">
                  {result.all_findings.map((f, idx) => (
                    <div key={f.code} className={`rounded-2xl border p-4 space-y-2.5 ${cardClass}`}>
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <Activity className="w-4 h-4 text-rose-400 shrink-0" />
                          <h3 className="font-black text-sm sm:text-base">
                            {idx + 1}. {isAr ? f.name_ar : f.name_en}
                          </h3>
                        </div>
                        <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black border ${urgencyBadge(f.urgency)}`}>
                          {isAr ? f.urgency_label_ar : f.urgency_label_en}
                        </span>
                      </div>

                      <div className="text-[11px] font-mono opacity-75">{isAr ? f.name_en : f.name_ar}</div>

                      {f.evidence.length > 0 && (
                        <div className={`rounded-xl p-2.5 text-xs space-y-1 ${isLight ? "bg-slate-100" : "bg-white/5"}`}>
                          <div className="font-bold text-[11px] text-rose-400">
                            {isAr ? "الأدلة والمعايير المتحققة:" : "Matched ECG Criteria & Evidence:"}
                          </div>
                          <ul className="list-disc list-inside space-y-0.5">
                            {f.evidence.map((ev, i) => (
                              <li key={i}>{ev}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      <div className="text-xs leading-relaxed">
                        <span className="font-extrabold text-cyan-400">{isAr ? "الشرح الإكلينيكي المفصل: " : "Clinical Explanation: "}</span>
                        {isAr ? f.explanation_ar : f.explanation_en}
                      </div>

                      <div
                        className={`rounded-xl border p-2.5 text-xs leading-relaxed ${
                          f.urgency === "CRITICAL" || f.urgency === "URGENT"
                            ? "border-rose-500/40 bg-rose-500/10"
                            : isLight
                            ? "border-slate-200 bg-slate-50"
                            : "border-white/10 bg-white/5"
                        }`}
                      >
                        <span className="font-extrabold inline-flex items-center gap-1 text-amber-400">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          {isAr ? "التصرف الطبي العاجل: " : "Immediate Management: "}
                        </span>
                        {isAr ? f.immediate_action_ar : f.immediate_action_en}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
