"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ImagePlus,
  Loader2,
  RefreshCw,
  ScanLine,
  ShieldAlert,
  Sparkles,
  Trash2,
} from "lucide-react";
import { ECGInterpretationResult, ECG_UNITS } from "@/lib/ecgRules";
import { useAtlasLocale } from "@/lib/atlasI18n";
import { useUIStore } from "@/stores/uiStore";

export function EcgDiagnosticStudio() {
  const { locale } = useAtlasLocale();
  const theme = useUIStore((s) => s.theme);
  const isAr = locale === "ar";
  const isLight = theme === "light";

  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageMime, setImageMime] = useState<string>("image/png");
  const [imageName, setImageName] = useState<string>("");
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [visionError, setVisionError] = useState<string | null>(null);
  const [result, setResult] = useState<ECGInterpretationResult | null>(null);
  const [visionNotes, setVisionNotes] = useState<{ ar: string; en: string; provider: string } | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const runVisionDiagnosis = async (b64: string, mime: string) => {
    setAnalyzing(true);
    setVisionError(null);
    setResult(null);
    setVisionNotes(null);
    try {
      const res = await fetch("/api/ecg-analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageBase64: b64, mimeType: mime }),
      });
      const data = await res.json();
      if (!res.ok) {
        setVisionError(isAr ? data.error_ar || data.error : data.error || "ECG analysis failed");
        return;
      }
      setResult(data.interpretation || null);
      setVisionNotes({
        ar: data.vision_notes_ar || "",
        en: data.vision_notes_en || "",
        provider: data.provider || "Vision AI",
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Network error";
      setVisionError(isAr ? `تعذر الاتصال بمحرك تشخيص رسم القلب: ${msg}` : `Connection error: ${msg}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setVisionError(isAr ? "يرجى رفع صورة رسم قلب بصيغة PNG أو JPG." : "Please upload an ECG image (PNG / JPG).");
      return;
    }
    setImageName(file.name);
    setImageMime(file.type || "image/png");
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      setImagePreview(dataUrl);
      void runVisionDiagnosis(dataUrl, file.type || "image/png");
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
    setResult(null);
    setVisionNotes(null);
    setVisionError(null);
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

  return (
    <div
      dir={isAr ? "rtl" : "ltr"}
      className={`h-full w-full overflow-y-auto px-4 md:px-6 py-4 ${
        isLight ? "bg-slate-50/90 text-slate-900" : "bg-[#060a14] text-slate-100"
      }`}
    >
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Header Bar */}
        <div className={`rounded-2xl border p-4 flex flex-wrap items-center justify-between gap-3 ${cardClass}`}>
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-rose-500/15 border border-rose-400/35 flex items-center justify-center text-rose-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-black text-sm sm:text-base">
                {isAr ? "أداة التشخيص الذكي لرسم القلب (ECG AI Diagnosis)" : "Automated AI ECG Diagnostic Engine"}
              </h2>
              <p className={`text-[11px] ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                {isAr
                  ? `ارفع صورة رسم القلب فقط — التحليل مبني آلياً على معايير ecg_criteria.json (${ECG_UNITS.paper_speed_mm_per_sec} mm/s • ${ECG_UNITS.small_box_ms} ms/مربع)`
                  : `Upload your ECG image only — automated reading governed by ecg_criteria.json (${ECG_UNITS.paper_speed_mm_per_sec} mm/s • ${ECG_UNITS.small_box_ms} ms/box)`}
              </p>
            </div>
          </div>

          {imagePreview && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={analyzing}
                onClick={() => void runVisionDiagnosis(imagePreview, imageMime)}
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

        {/* 2-Column Pure Image-to-Diagnosis Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          {/* LEFT: ECG Image Input & Viewer Only */}
          <div className="lg:col-span-5 space-y-3">
            <div className={`rounded-2xl border p-4 space-y-3 ${cardClass}`}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold flex items-center gap-1.5">
                  <ImagePlus className="w-4 h-4 text-rose-400" />
                  {isAr ? "صورة رسم القلب (12-Lead ECG / Rhythm Strip)" : "ECG Tracing Image"}
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
                  <div className="space-y-2">
                    <img
                      src={imagePreview}
                      alt="Uploaded ECG"
                      className="w-full max-h-[380px] object-contain rounded-xl mx-auto bg-white"
                    />
                    <div className="text-[11px] opacity-70">
                      {isAr ? "اضغط هنا أو اسحب صورة أخرى لاستبدال رسم القلب" : "Click or drop another image to replace ECG"}
                    </div>
                  </div>
                ) : (
                  <div className="py-12 px-4 space-y-2.5">
                    <div className="w-12 h-12 mx-auto rounded-2xl bg-rose-500/15 border border-rose-400/35 flex items-center justify-center text-rose-400">
                      <ScanLine className="w-6 h-6" />
                    </div>
                    <div className="text-sm font-black">
                      {isAr ? "اسحب صورة رسم القلب هنا أو اضغط للرفع" : "Drop ECG Image Here or Click to Upload"}
                    </div>
                    <p className="text-xs opacity-70 max-w-xs mx-auto leading-relaxed">
                      {isAr
                        ? "يكفي رفع الصورة فقط — سيقوم النظام بقراءة النظم، وحساب HR وQTc والمحور، واكتشاف STEMI وAF وVT وإحصار القلب تلقائياً."
                        : "Just upload the image — the engine automatically extracts intervals, ST shifts, and rhythm, then outputs urgent findings first."}
                    </p>
                  </div>
                )}
              </label>

              {analyzing && (
                <div className="rounded-xl border border-rose-400/35 bg-rose-500/10 p-3 flex items-center gap-2.5 text-xs font-bold text-rose-300">
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                  <span>
                    {isAr
                      ? "جاري فحص صورة الـ ECG واستخراج المعايير وتطبيق قواعد التشخيص..."
                      : "Scanning ECG tracing, extracting measurements, and running diagnostic rules..."}
                  </span>
                </div>
              )}

              {visionError && (
                <div className="rounded-xl border border-amber-500/45 bg-amber-500/10 p-3 text-xs text-amber-300 leading-relaxed">
                  {visionError}
                </div>
              )}

              {visionNotes && (visionNotes.ar || visionNotes.en) && (
                <div className="rounded-xl border border-cyan-500/35 bg-cyan-500/10 p-3 text-xs space-y-1">
                  <div className="font-extrabold text-cyan-300 flex items-center justify-between">
                    <span>{isAr ? "القراءة البصرية للرسم (AI Vision):" : "Visual Tracing Summary:"}</span>
                    <span className="text-[10px] font-mono opacity-75">{visionNotes.provider}</span>
                  </div>
                  <p className="leading-relaxed opacity-90">{isAr ? visionNotes.ar || visionNotes.en : visionNotes.en || visionNotes.ar}</p>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT: Read-Only Automated Diagnostic Report */}
          <div className="lg:col-span-7 space-y-4">
            {!result && !analyzing && (
              <div className={`rounded-2xl border p-10 text-center space-y-3 ${cardClass}`}>
                <Activity className="w-10 h-10 mx-auto text-rose-400/80" />
                <h3 className="font-black text-base">
                  {isAr ? "بانتظار صورة رسم القلب للبدء" : "Ready for ECG Image Analysis"}
                </h3>
                <p className="text-xs opacity-75 max-w-md mx-auto leading-relaxed">
                  {isAr
                    ? "ارفع صورة الـ ECG من اللوحة الجانبية للحصول على تقرير تشخيصي تلقائي كامل يبرز الحالات الطارئة والحرجة أولاً مع الشرح الإكلينيكي وخطة التصرف الفوري."
                    : "Upload an ECG image on the left to receive a full automated clinical report with critical/urgent alerts prioritized first, detailed explanations, and immediate management."}
                </p>
              </div>
            )}

            {result && (
              <>
                {result.has_urgent_findings ? (
                  <div className="rounded-2xl border-2 border-red-500/75 bg-red-950/55 p-4 text-red-100 shadow-lg shadow-red-900/30">
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

                {/* Read-Only Telemetry Metrics */}
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

                {result.measurements.qtc_bazett_ms && (
                  <div className={`rounded-xl border px-3 py-2 text-[11px] flex flex-wrap gap-4 ${cardClass}`}>
                    <span className="font-bold text-rose-400">QTc:</span>
                    <span>Bazett: <b>{result.measurements.qtc_bazett_ms} ms</b></span>
                    <span>Fridericia: <b>{result.measurements.qtc_fridericia_ms} ms</b></span>
                    <span>Hodges: <b>{result.measurements.qtc_hodges_ms} ms</b></span>
                    {result.measurements.axis_classification && <span>Axis: <b>{result.measurements.axis_classification}</b></span>}
                  </div>
                )}

                {/* Diagnostic Findings (Urgent First) */}
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
                            {isAr ? "الأدلة المستخلصة من رسم القلب:" : "Extracted ECG Evidence:"}
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
