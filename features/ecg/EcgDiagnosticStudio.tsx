"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  BookOpen,
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
      // Yield 1 frame so UI renders scanning indicator smoothly
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
        {/* Header Bar */}
        <div className={`rounded-2xl border p-4 flex flex-wrap items-center justify-between gap-3 ${cardClass}`}>
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-rose-500/15 border border-rose-400/35 flex items-center justify-center text-rose-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="font-black text-sm sm:text-base">
                  {isAr ? "محرك تشخيص رسم القلب المحلي (بدون API أو توكنز)" : "Local Clinical ECG Diagnostic Engine (Offline • 0 Tokens)"}
                </h2>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-500/15 text-emerald-400 border border-emerald-400/30">
                  <Cpu className="w-3 h-3" />
                  {isAr ? "100 صورة مرجعية + ecg_criteria.json" : "100 Reference ECGs + ecg_criteria.json"}
                </span>
              </div>
              <p className={`text-[11px] mt-0.5 ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                {isAr
                  ? `ارفع صورة رسم القلب فقط — يحللها المحرك محلياً ويقارنها بقاعدة الصور المرجعية ومعايير ${ECG_UNITS.paper_speed_mm_per_sec} mm/s (${ECG_UNITS.small_box_ms} ms/مربع)`
                  : `Upload your ECG image only — analyzed 100% locally against 100 textbook ECG references & ecg_criteria.json (${ECG_UNITS.paper_speed_mm_per_sec} mm/s)`}
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

        {/* 2-Column Pure Image-to-Diagnosis Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          {/* LEFT: ECG Image Input & Local DSP + Reference Atlas Matches */}
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
                      className="w-full max-h-[360px] object-contain rounded-xl mx-auto bg-white"
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
                        ? "يعمل 100% محلياً بدون أي API خارجي أو توكنز — يقارن الصورة مع 100 صورة مرجعية من الكتب ويطبق قواعد ecg_criteria.json فوراً."
                        : "Runs 100% locally with zero API keys or tokens — matches against 100 reference ECGs and evaluates ecg_criteria.json instantly."}
                    </p>
                  </div>
                )}
              </label>

              {analyzing && (
                <div className="rounded-xl border border-rose-400/35 bg-rose-500/10 p-3 flex items-center gap-2.5 text-xs font-bold text-rose-300">
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                  <span>
                    {isAr
                      ? "جاري تحليل الموجات ومقارنتها مع قاعدة الـ 100 صورة مرجعية..."
                      : "Analyzing waveform locally & matching against 100 reference ECGs..."}
                  </span>
                </div>
              )}

              {errorMsg && (
                <div className="rounded-xl border border-amber-500/45 bg-amber-500/10 p-3 text-xs text-amber-300 leading-relaxed">
                  {errorMsg}
                </div>
              )}

              {/* Direct Signal DSP Telemetry */}
              {analysis && (
                <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-3 space-y-2 text-[11px]">
                  <div className="font-extrabold text-cyan-400">
                    {isAr ? "تحليل الإشارة المباشر من الصورة (Local Signal DSP):" : "Direct Image Waveform Analysis:"}
                  </div>
                  <div className="grid grid-cols-2 gap-1.5">
                    <div>{isAr ? "قمم R المكتشفة:" : "Detected R-peaks:"} <b>{analysis.signal.detected_r_peaks}</b></div>
                    <div>{isAr ? "انتظام الإيقاع:" : "RR Regularity:"} <b>{analysis.signal.rr_regularity_pct}%</b></div>
                    <div>{isAr ? "شبكة ورق ECG:" : "ECG Grid:"} <b>{analysis.signal.pink_grid_detected ? (isAr ? "مرصودة" : "Detected") : (isAr ? "أبيض/رمادي" : "Monochrome")}</b></div>
                    <div>{isAr ? "الأبعاد:" : "Resolution:"} <b>{analysis.signal.width}×{analysis.signal.height}</b></div>
                  </div>
                </div>
              )}

              {/* Top Matched Textbook Reference Cases */}
              {analysis && analysis.top_matches.length > 0 && (
                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 space-y-2">
                  <div className="text-[11px] font-extrabold text-rose-400 flex items-center gap-1.5">
                    <BookOpen className="w-3.5 h-3.5" />
                    {isAr ? "أقرب الحالات المرجعية من كتب القلب المرفوعة:" : "Top Matched Textbook Reference ECGs:"}
                  </div>
                  <div className="space-y-1.5">
                    {analysis.top_matches.slice(0, 3).map((m, i) => (
                      <div
                        key={m.id}
                        className={`rounded-lg px-2.5 py-1.5 text-[11px] flex items-center justify-between gap-2 ${
                          isLight ? "bg-slate-100" : "bg-slate-900/80"
                        }`}
                      >
                        <div className="min-w-0">
                          <div className="font-bold truncate">
                            {i + 1}. {m.caption}
                          </div>
                          <div className="text-[10px] opacity-70 truncate">
                            {m.source_book} — p.{m.source_page}
                          </div>
                        </div>
                        <span className="shrink-0 px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 font-black text-[10px]">
                          {m.similarity_pct}%
                        </span>
                      </div>
                    ))}
                  </div>
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
                  {isAr ? "بانتظار صورة رسم القلب للتحليل المحلي الفوري" : "Ready for Instant Local ECG Diagnosis"}
                </h3>
                <p className="text-xs opacity-75 max-w-md mx-auto leading-relaxed">
                  {isAr
                    ? "ارفع صورة الـ ECG من اللوحة الجانبية — سيقارنها المحرك محلياً مع قاعدة الـ 100 صورة مرجعية وقواعد ecg_criteria.json بدون أي API خارجي أو توكنز، ويبرز الحالات الطارئة أولاً مع الشرح الكامل."
                    : "Upload an ECG image on the left — the local engine matches it against 100 reference textbook ECGs and ecg_criteria.json with zero external APIs or tokens."}
                </p>
              </div>
            )}

            {result && analysis && (
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

                {/* Local Engine Match Summary */}
                <div className={`rounded-xl border px-3.5 py-2.5 text-xs leading-relaxed ${cardClass}`}>
                  {isAr ? analysis.summary_ar : analysis.summary_en}
                </div>

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
