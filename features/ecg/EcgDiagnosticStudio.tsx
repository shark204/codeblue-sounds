"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
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
      await new Promise((r) => setTimeout(r, 60));
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
    : "bg-[#0c1426]/92 border-white/10 text-slate-100 shadow-md shadow-black/30";

  const urgencyBadge = (u: string) => {
    if (u === "CRITICAL") return "bg-red-600 text-white border-red-400 shadow-red-600/30 shadow-sm";
    if (u === "URGENT") return "bg-amber-500 text-slate-950 border-amber-300";
    if (u === "ABNORMAL") return "bg-sky-500/20 text-sky-300 border-sky-400/40";
    return "bg-emerald-500/20 text-emerald-300 border-emerald-400/40";
  };

  const result = analysis?.interpretation || null;
  // Red Emergency / Interventional Warning Banner triggers ONLY for CRITICAL conditions
  const criticalFindings = result ? result.all_findings.filter((f) => f.urgency === "CRITICAL") : [];

  return (
    <div
      dir={isAr ? "rtl" : "ltr"}
      className={`h-full w-full overflow-y-auto px-3 md:px-6 py-3 ${
        isLight ? "bg-slate-50 text-slate-900" : "bg-[#050912] text-slate-100"
      }`}
    >
      <div className="max-w-7xl mx-auto space-y-3">
        {/* Compact Single-Row Top Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-rose-500/15 border border-rose-500/35 flex items-center justify-center text-rose-400">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h2 className="font-black text-sm sm:text-base leading-tight">
                {isAr ? "محرك تشخيص رسم القلب السريري" : "Clinical ECG Diagnostic Engine"}
              </h2>
              <p className={`text-[11px] ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                {isAr
                  ? `قاعدة 215 رسم قلب مرجعي + ecg_criteria.json • معايرة ${ECG_UNITS.paper_speed_mm_per_sec} mm/s`
                  : `215 Textbook Reference ECGs + ecg_criteria.json • ${ECG_UNITS.paper_speed_mm_per_sec} mm/s`}
              </p>
            </div>
          </div>

          {imagePreview && (
            <div className="flex items-center gap-1.5">
              <label className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-white/15 bg-white/5 hover:bg-white/10 text-xs font-bold transition cursor-pointer">
                <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                <ImagePlus className="w-3.5 h-3.5 text-rose-400" />
                {isAr ? "صورة جديدة" : "New ECG"}
              </label>
              <button
                type="button"
                disabled={analyzing}
                onClick={() => void runLocalDiagnosis(imagePreview)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 disabled:opacity-60 text-white text-xs font-bold transition cursor-pointer"
              >
                {analyzing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                {isAr ? "إعادة الفحص" : "Re-analyze"}
              </button>
              <button
                type="button"
                onClick={clearAll}
                className="p-1.5 rounded-xl border border-white/15 hover:bg-rose-500/15 hover:text-rose-400 transition cursor-pointer"
                title={isAr ? "مسح" : "Clear"}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>

        {/* CRITICAL Emergency Intervention Banner ONLY (e.g. STEMI, LMCA, Tamponade, VT/VF, Complete Heart Block) */}
        {criticalFindings.length > 0 && (
          <div className="rounded-xl border-2 border-red-500/80 bg-gradient-to-r from-red-950/90 via-red-900/60 to-red-950/90 px-4 py-2.5 text-red-100 shadow-lg shadow-red-950/40 flex items-center gap-2.5">
            <ShieldAlert className="w-5 h-5 text-red-400 animate-pulse shrink-0" />
            <span className="font-black text-xs sm:text-sm">
              {isAr
                ? `تحذير تدخل طارئ فوري: تم رصد ${criticalFindings.map((c) => c.name_ar).join(" • ")} — يتطلب تدخلاً عاجلاً (قسطرة / بزل / إنعاش / جراحة)!`
                : `CRITICAL INTERVENTION ALERT: ${criticalFindings.map((c) => c.name_en).join(" • ")} — Immediate emergency/interventional action required!`}
            </span>
          </div>
        )}

        {/* Single-Screen Balanced 2-Column Layout (Scrollable when needed) */}
        {!imagePreview ? (
          <div className={`rounded-2xl border p-6 sm:p-10 max-w-2xl mx-auto mt-6 ${cardClass}`}>
            <label
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`block rounded-2xl border-2 border-dashed py-12 px-6 text-center cursor-pointer transition ${
                isDragging ? "border-rose-400 bg-rose-500/15" : "border-rose-400/35 hover:border-rose-400/75 hover:bg-rose-500/5"
              }`}
            >
              <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              <div className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-rose-500/15 border border-rose-400/35 flex items-center justify-center text-rose-400">
                <ScanLine className="w-7 h-7" />
              </div>
              <div className="text-base font-black">
                {isAr ? "اسحب صورة رسم القلب هنا أو اضغط للرفع" : "Drop ECG Image Here or Click to Upload"}
              </div>
              <p className="text-xs opacity-70 max-w-md mx-auto mt-1.5 leading-relaxed">
                {isAr
                  ? "يدعم رسم القلب الكامل (12-Lead) وشرائط النظم الأحادية — تحليل محلي فوري بدون توكنز عبر قاعدة 215 رسم قلب مرجعي."
                  : "Supports 12-Lead ECGs and single-lead rhythm strips — instant offline analysis backed by 215 textbook reference ECGs."}
              </p>
            </label>
            {errorMsg && (
              <div className="mt-3 rounded-xl border border-amber-500/45 bg-amber-500/10 p-2.5 text-xs text-amber-300 text-center">
                {errorMsg}
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 items-start">
            {/* LEFT COLUMN: Sticky ECG Image Card */}
            <div className="lg:col-span-5 lg:sticky lg:top-2">
              <div className={`rounded-2xl border p-3.5 space-y-2.5 ${cardClass}`}>
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="font-extrabold flex items-center gap-1.5">
                    <Activity className="w-4 h-4 text-rose-400" />
                    {isAr ? "صورة رسم القلب" : "ECG Tracing"}
                  </span>
                  {analysis && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-500/15 text-rose-300 border border-rose-400/30">
                      {analysis.layout === "single_lead"
                        ? isAr ? "شريط أحادي" : "Single-Lead"
                        : "12-Lead ECG"}
                    </span>
                  )}
                </div>

                <label
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  className="block rounded-xl overflow-hidden bg-white p-1.5 border border-white/10 cursor-pointer"
                >
                  <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                  <img
                    src={imagePreview}
                    alt="Uploaded ECG"
                    className="w-full max-h-[300px] object-contain mx-auto"
                  />
                </label>

                <div className="flex items-center justify-between text-[10.5px] opacity-65">
                  <span className="truncate max-w-[200px]">{imageName}</span>
                  <span>{isAr ? "اضغط على الصورة لتغييرها" : "Click image to replace"}</span>
                </div>

                {analyzing && (
                  <div className="rounded-xl border border-rose-400/35 bg-rose-500/10 p-2.5 flex items-center justify-center gap-2 text-xs font-bold text-rose-300">
                    <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                    <span>{isAr ? "جاري فحص الموجة..." : "Analyzing waveform..."}</span>
                  </div>
                )}

                {errorMsg && (
                  <div className="rounded-xl border border-amber-500/45 bg-amber-500/10 p-2.5 text-xs text-amber-300">
                    {errorMsg}
                  </div>
                )}
              </div>
            </div>

            {/* RIGHT COLUMN: Compact Single-Screen Report (Telemetry + Diagnosis Cards) */}
            <div className="lg:col-span-7 space-y-3 pb-3">
              {result && analysis && (
                <>
                  {/* Compact 5-Tile Telemetry Bar */}
                  <div className="grid grid-cols-5 gap-2">
                    {[
                      { label: "HR", value: result.measurements.hr_bpm ? `${result.measurements.hr_bpm}` : "—", unit: "bpm" },
                      { label: "PR", value: result.measurements.pr_ms ? `${result.measurements.pr_ms}` : "—", unit: "ms" },
                      { label: "QRS", value: result.measurements.qrs_ms ? `${result.measurements.qrs_ms}` : "—", unit: "ms" },
                      { label: "QTc", value: result.measurements.qtc_bazett_ms ? `${result.measurements.qtc_bazett_ms}` : "—", unit: "ms" },
                      { label: "Axis", value: result.measurements.axis_deg != null ? `${result.measurements.axis_deg}°` : "—", unit: "" },
                    ].map((m) => (
                      <div key={m.label} className={`rounded-xl border px-2 py-2 text-center ${cardClass}`}>
                        <div className="text-[10px] opacity-65 font-bold">{m.label}</div>
                        <div className="text-xs sm:text-sm font-black mt-0.5">
                          {m.value} <span className="text-[9.5px] font-semibold opacity-65">{m.value !== "—" ? m.unit : ""}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Summary Strip */}
                  <div className={`rounded-xl border px-3.5 py-2 text-xs leading-relaxed ${cardClass}`}>
                    {isAr ? analysis.summary_ar : analysis.summary_en}
                  </div>

                  {/* Diagnosis Cards */}
                  <div className="space-y-2.5">
                    {result.all_findings.map((f, idx) => (
                      <div key={f.code} className={`rounded-2xl border p-4 space-y-2.5 ${cardClass}`}>
                        <div className="flex flex-wrap items-start justify-between gap-2 pb-2 border-b border-white/10">
                          <div>
                            <div className="flex items-center gap-1.5">
                              <Stethoscope className="w-4 h-4 text-rose-400 shrink-0" />
                              <h3 className="font-black text-sm sm:text-base">
                                {idx + 1}. {isAr ? f.name_ar : f.name_en}
                              </h3>
                            </div>
                            <div className="text-[11px] font-mono opacity-65 mt-0.5">{isAr ? f.name_en : f.name_ar}</div>
                          </div>
                          <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black border ${urgencyBadge(f.urgency)}`}>
                            {isAr ? f.urgency_label_ar : f.urgency_label_en}
                          </span>
                        </div>

                        {f.evidence.length > 0 && (
                          <div className="flex flex-wrap gap-1.5">
                            {f.evidence.map((ev, i) => (
                              <span
                                key={i}
                                className={`px-2 py-0.5 rounded-md text-[10.5px] font-semibold border ${
                                  isLight ? "bg-slate-100 border-slate-200 text-slate-700" : "bg-white/[0.04] border-white/10 text-slate-200"
                                }`}
                              >
                                {ev}
                              </span>
                            ))}
                          </div>
                        )}

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                          <div className={`rounded-xl p-3 text-xs leading-relaxed ${isLight ? "bg-slate-50 border border-slate-200" : "bg-white/[0.03] border border-white/10"}`}>
                            <div className="font-extrabold text-cyan-400 mb-1">
                              {isAr ? "الشرح الإكلينيكي" : "Clinical Explanation"}
                            </div>
                            {isAr ? f.explanation_ar : f.explanation_en}
                          </div>

                          <div
                            className={`rounded-xl border p-3 text-xs leading-relaxed ${
                              f.urgency === "CRITICAL"
                                ? "border-red-500/45 bg-red-500/10"
                                : isLight
                                ? "border-slate-200 bg-slate-50"
                                : "border-white/10 bg-white/[0.03]"
                            }`}
                          >
                            <div className="font-extrabold inline-flex items-center gap-1 text-amber-400 mb-1">
                              <AlertTriangle className="w-3.5 h-3.5" />
                              {isAr ? "التصرف الطبي" : "Management"}
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
          </div>
        )}
      </div>
    </div>
  );
}
