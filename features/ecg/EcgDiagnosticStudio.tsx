"use client";

import React, { useMemo, useState } from "react";
import {
  Activity,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Eye,
  HeartPulse,
  ImagePlus,
  Layers,
  Loader2,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  UserCheck,
  Zap,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  INITIAL_WIZARD_FEATURES,
  WizardFeatures,
  runHybridEcgDiagnosis,
} from "@/lib/ecgHybridEngine";
import {
  LocalECGAnalysisResult,
  analyzeEcgImageLocally,
} from "@/lib/ecgImageEngine";
import { EcgResultsPanel, EcgGradCamPanel } from "@/features/ecg/EcgDiagnosisView";
import { useAtlasLocale } from "@/lib/atlasI18n";
import { useUIStore } from "@/stores/uiStore";

interface UploadedImage {
  id: string;
  name: string;
  dataUrl: string;
}

const MEDICAL_CONTEXT_CHIPS: { key: keyof WizardFeatures; labelAr: string; labelEn: string }[] = [
  { key: "dka_ckd_or_high_k", labelAr: "فشل كلوي / غيبوبة سكر (DKA) / اشتباه ارتفاع بوتاسيوم", labelEn: "CKD / DKA / Suspected High K+" },
  { key: "fever_or_cocaine_or_sodium_blocker", labelAr: "حمى شديدة / كوكايين / دواء حاصر لقنوات الصوديوم", labelEn: "High Fever / Cocaine / Na+ Blocker" },
  { key: "has_pacemaker", labelAr: "مركّب منظم ضربات قلب (Pacemaker)", labelEn: "Has Pacemaker" },
  { key: "tremor_or_nebulizer_or_loose_lead", labelAr: "رعشة / يستخدم نيبولايزر / حركة أثناء التخطيط", labelEn: "Tremor / Nebulizer / Movement" },
  { key: "syncope_or_family_scd", labelAr: "تاريخ عائلي لموت مفاجئ أو إغماء متكرر", labelEn: "Recurrent Syncope / Family SCD" },
  { key: "post_emotional_stress", labelAr: "صدمة عاطفية أو ضغط نفسي/جسدي شديد", labelEn: "Severe Emotional / Physical Stress" },
];

interface RemoteAnalysisSummary {
  hr_bpm: number | null;
  pr_ms: number | null;
  qrs_ms: number | null;
  qt_ms: number | null;
  axis_deg: number | null;
  rhythm_class: string | null;
  st_elevation_leads: string[];
  st_depression_leads: string[];
  t_wave_inversion_leads: string[];
  peaked_t_waves: boolean;
  detected_speed_mm_s: 25 | 50;
  detected_gain_mm_mv: 10 | 5;
  ai_diagnoses: { label: string; code: string | null; prob: number }[];
  elapsed_sec?: number;
}

const ECG_ANALYZE_ENDPOINT = "/api/ecg-analyze";

async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const res = await fetch(dataUrl);
  return res.blob();
}

export function EcgDiagnosticStudio() {
  const { locale } = useAtlasLocale();
  const theme = useUIStore((s) => s.theme);
  const isAr = locale === "ar";
  const isLight = theme === "light";

  // Multi-image & auto-analysis state
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [activeImgIdx, setActiveImgIdx] = useState<number>(0);
  const [zoom, setZoom] = useState<number>(1);
  const [highContrast, setHighContrast] = useState<boolean>(false);
  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);
  const [gradcamFocusRegion, setGradcamFocusRegion] = useState<string | null>(null);
  const [measurementSource, setMeasurementSource] = useState<"device_ocr" | "auto_waveform" | "manual" | null>(null);

  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [analyzingImg, setAnalyzingImg] = useState<boolean>(false);
  const [imgAnalysis, setImgAnalysis] = useState<LocalECGAnalysisResult | null>(null);
  const [backendSource, setBackendSource] = useState<"modal-ai" | "local" | null>(null);
  const [remoteAnalysis, setRemoteAnalysis] = useState<RemoteAnalysisSummary | null>(null);

  // ONLY 3 Patient-Context Cards (Zero ECG-reading cards!)
  const [step, setStep] = useState<number>(1);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [features, setFeatures] = useState<WizardFeatures>(INITIAL_WIZARD_FEATURES);
  const [symptomChoice, setSymptomChoice] = useState<"chest_pain" | "syncope" | "secondary_trigger" | "asymptomatic" | null>(null);

  const runImageExtraction = async (dataUrl: string, rawFile?: File) => {
    setAnalyzingImg(true);
    setRemoteAnalysis(null);
    setGradcamUrl(null);
    setGradcamFocusRegion(null);
    try {
      try {
        const blob = rawFile ?? (await dataUrlToBlob(dataUrl));
        const fd = new FormData();
        fd.append("file", blob, rawFile?.name || "ecg.png");
        const resp = await fetch(ECG_ANALYZE_ENDPOINT, {
          method: "POST",
          body: fd,
          signal: AbortSignal.timeout(42000),
        });
        if (resp.ok) {
          const j = await resp.json();
          if (j && j.ok) {
            const bm = j.measurements || {};
            const detSpeed: 25 | 50 = j.detected_speed_mm_s === 50 ? 50 : 25;
            const detGain: 10 | 5 = j.detected_gain_mm_mv === 5 ? 5 : 10;
            const aiList: { label: string; code: string | null; prob: number; uncertainty?: number }[] = Array.isArray(j.ai_diagnoses)
              ? j.ai_diagnoses
              : [];
            if (typeof j.gradcam_data_url === "string" && j.gradcam_data_url.startsWith("data:image/")) {
              setGradcamUrl(j.gradcam_data_url);
            }
            if (typeof j.gradcam_focus_region === "string") {
              setGradcamFocusRegion(j.gradcam_focus_region);
            }
            setBackendSource("modal-ai");
            setRemoteAnalysis({
              hr_bpm: bm.hr_bpm ?? null,
              pr_ms: bm.pr_ms ?? null,
              qrs_ms: bm.qrs_ms ?? null,
              qt_ms: bm.qt_ms ?? null,
              axis_deg: bm.axis_deg ?? null,
              rhythm_class: null,
              st_elevation_leads: [],
              st_depression_leads: [],
              t_wave_inversion_leads: [],
              peaked_t_waves: false,
              detected_speed_mm_s: detSpeed,
              detected_gain_mm_mv: detGain,
              ai_diagnoses: aiList,
              elapsed_sec: typeof j.elapsed_sec === "number" ? j.elapsed_sec : undefined,
            });
            // If the ECG image has no printed computer header text, estimate HR/intervals from the waveform pixels locally
            let fallbackM: { hr_bpm?: number | null; pr_ms?: number | null; qrs_ms?: number | null; qt_ms?: number | null } = {};
            if ((bm.hr_bpm == null || bm.qrs_ms == null || bm.qt_ms == null) && dataUrl.startsWith("data:image/")) {
              try {
                const localRes = await analyzeEcgImageLocally(dataUrl, { paperSpeedMmS: detSpeed, gainMmMv: detGain });
                const lm = localRes.interpretation.measurements;
                fallbackM = {
                  hr_bpm: lm.hr_bpm != null ? Math.round(lm.hr_bpm) : null,
                  pr_ms: lm.pr_ms != null ? Math.round(lm.pr_ms) : null,
                  qrs_ms: lm.qrs_ms != null ? Math.round(lm.qrs_ms) : null,
                  qt_ms: lm.qt_ms != null ? Math.round(lm.qt_ms) : null,
                };
              } catch {
                // ignore local fallback error
              }
            }
            setFeatures((prev) => ({
              ...prev,
              paper_speed_mm_s: detSpeed,
              gain_mm_mv: detGain,
              hr_bpm: bm.hr_bpm ?? fallbackM.hr_bpm ?? prev.hr_bpm,
              pr_ms: bm.pr_ms ?? fallbackM.pr_ms ?? prev.pr_ms,
              qrs_ms: bm.qrs_ms ?? fallbackM.qrs_ms ?? prev.qrs_ms,
              qt_ms: bm.qt_ms ?? fallbackM.qt_ms ?? prev.qt_ms,
              axis_deg: bm.axis_deg ?? prev.axis_deg,
              ai_predictions: j.low_confidence ? [] : aiList,
            }));
            const hasOcrNum = bm.hr_bpm != null || bm.pr_ms != null || bm.qrs_ms != null || bm.qt_ms != null || bm.axis_deg != null;
            const hasFallbackNum = fallbackM.hr_bpm != null || fallbackM.qrs_ms != null || fallbackM.qt_ms != null;
            setMeasurementSource(hasOcrNum ? "device_ocr" : hasFallbackNum ? "auto_waveform" : null);
            setImgAnalysis(null);
            return;
          }
        }
      } catch {
        // Fallback to local browser quality/grid analyzer if backend is offline
      }

      if (!dataUrl.startsWith("data:image/")) return;
      setBackendSource("local");
      const res = await analyzeEcgImageLocally(dataUrl, { paperSpeedMmS: 25, gainMmMv: 10 });
      setImgAnalysis(res);
      setMeasurementSource("auto_waveform");
      const m = res.interpretation.measurements;
      const morph = res.extracted_morphology;
      const rrPat = res.extracted_input.rr_pattern;
      setFeatures((prev) => ({
        ...prev,
        paper_speed_mm_s: 25,
        gain_mm_mv: 10,
        hr_bpm: m.hr_bpm != null ? Math.round(m.hr_bpm) : prev.hr_bpm,
        pr_ms: m.pr_ms != null ? Math.round(m.pr_ms) : prev.pr_ms,
        qrs_ms: m.qrs_ms != null ? Math.round(m.qrs_ms) : prev.qrs_ms,
        qt_ms: m.qt_ms != null ? Math.round(m.qt_ms) : prev.qt_ms,
        rhythm_class:
          rrPat === "regular" || rrPat === "irregularly_irregular" || rrPat === "pauses"
            ? rrPat
            : prev.rhythm_class,
        st_elevation_leads: morph.st_elevation_leads,
        st_depression_leads: morph.st_depression_leads,
        t_wave_inversion_leads: morph.t_wave_inversion_leads,
        peaked_t_waves: morph.peaked_t_waves,
        pacing_spikes: morph.pacing_spikes,
        brugada_pattern: morph.brugada_pattern,
        ai_predictions: [],
      }));
    } catch {
      setImgAnalysis(null);
      setRemoteAnalysis(null);
    } finally {
      setAnalyzingImg(false);
    }
  };

  const addFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    Array.from(fileList).forEach((file) => {
      if (!file.type.startsWith("image/") && file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) return;
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = String(reader.result || "");
        setImages((prev) => {
          const next = [...prev, { id: `${Date.now()}-${Math.random()}`, name: file.name, dataUrl }];
          setActiveImgIdx(next.length - 1);
          return next;
        });
        void runImageExtraction(dataUrl, file);
      };
      reader.readAsDataURL(file);
    });
  };

  const removeImage = (idx: number) => {
    setImages((prev) => {
      const next = prev.filter((_, i) => i !== idx);
      const nextIdx = Math.max(0, Math.min(activeImgIdx, next.length - 1));
      setActiveImgIdx(nextIdx);
      if (next.length === 0) { setAnalyzingImg(false); setImgAnalysis(null); setRemoteAnalysis(null); setBackendSource(null); setGradcamUrl(null); setGradcamFocusRegion(null); }
      return next;
    });
  };

  const toggleFlag = (key: keyof WizardFeatures) => {
    setFeatures((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleApplyCheck = (checkId: string, flagKey: keyof WizardFeatures) => {
    if (checkId === "chk_athlete_vagal") {
      setFeatures((p) => ({ ...p, is_competitive_athlete: true, asymptomatic_nocturnal: true }));
    } else {
      toggleFlag(flagKey);
    }
  };

  const resetAll = () => {
    setFeatures(INITIAL_WIZARD_FEATURES);
    setSymptomChoice(null);
    setStep(1);
    setShowResults(false);
    setImages([]);
    setImgAnalysis(null);
    setRemoteAnalysis(null);
    setBackendSource(null);
    setGradcamUrl(null);
    setGradcamFocusRegion(null);
    setMeasurementSource(null);
  };

  const parseOptNum = (val: string): number | null => {
    const trimmed = val.trim();
    if (!trimmed) return null;
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  };

  const hybridResult = useMemo(() => {
    if (!showResults) return null;
    return runHybridEcgDiagnosis(features);
  }, [showResults, features]);

  const cardCls = isLight
    ? "bg-white/72 backdrop-blur-xl border-slate-200/80 text-slate-900 shadow-[0_12px_28px_rgba(15,23,42,0.10)]"
    : "bg-[#0b1326]/68 backdrop-blur-xl border-white/14 text-slate-100 shadow-[0_14px_34px_rgba(0,0,0,0.45)]";

  const btnOptionCls = (active?: boolean) =>
    `w-full text-start px-4 py-3.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-3 ${
      active
        ? "bg-rose-500/20 border-rose-400 text-rose-100 shadow-[0_0_18px_rgba(244,63,94,0.2)]"
        : isLight
          ? "bg-white/70 hover:bg-rose-50/90 border-slate-200/90 hover:border-rose-300 text-slate-800 backdrop-blur-md"
          : "bg-[#0f1930]/65 hover:bg-[#162442]/80 border-white/12 hover:border-rose-400/50 text-slate-100 backdrop-blur-md"
    }`;

  const activeImage = images[activeImgIdx] || null;

  return (
    <div
      dir={isAr ? "rtl" : "ltr"}
      className={`h-full w-full overflow-y-auto px-3 sm:px-5 py-2.5 ${
        isLight ? "bg-transparent text-slate-900" : "bg-transparent text-slate-100"
      }`}
    >
      <div className="max-w-[1440px] mx-auto space-y-2.5 pb-6">
        {/* Top Engine Strip */}
        <div className={`rounded-2xl border px-4 py-2 flex flex-wrap items-center justify-between gap-3 ${cardCls}`}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-rose-500/15 border border-rose-400/40 flex items-center justify-center text-rose-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-black text-base sm:text-lg leading-tight">
                {isAr ? "محرك تشخيص الـ ECG" : "ECG Diagnostic Engine"}
              </h1>

            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={resetAll}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold border border-slate-700 bg-slate-800/70 hover:bg-slate-700 text-slate-200 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>{isAr ? "حالة جديدة" : "New Case"}</span>
            </button>
            {!showResults && (
              <button
                type="button"
                onClick={() => setShowResults(true)}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-extrabold bg-rose-600 hover:bg-rose-500 text-white cursor-pointer"
              >
                <Zap className="w-3.5 h-3.5" />
                <span>{isAr ? "اعرض التشخيص الآن" : "Diagnose Now"}</span>
              </button>
            )}
          </div>
        </div>

        {/* Main 2-Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-start">
          {/* LEFT: Auto-Analyzing ECG Image Uploader + Auto-Filled Printed Header Numbers */}
          <div className="lg:col-span-4 space-y-2.5 lg:max-h-[calc(100vh-110px)] lg:overflow-y-auto lg:pr-1.5 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:rounded-full [&::-webkit-scrollbar-track]:bg-slate-900/40 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-rose-400/50 hover:[&::-webkit-scrollbar-thumb]:bg-rose-400/80">
            <div className={`rounded-2xl border p-3 space-y-2 ${cardCls}`}>
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-xs font-extrabold">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <span>{isAr ? "صورة تخطيط القلب" : "ECG Image"}</span>
                </div>
                {activeImage && (
                  <div className="flex items-center gap-1">
                    <button type="button" onClick={() => setZoom((z) => Math.max(1, Number((z - 0.5).toFixed(1))))} className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-200 cursor-pointer">
                      <ZoomOut className="w-3.5 h-3.5" />
                    </button>
                    <span className="text-[11px] font-mono px-1.5">{Math.round(zoom * 100)}%</span>
                    <button type="button" onClick={() => setZoom((z) => Math.min(3, Number((z + 0.5).toFixed(1))))} className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-200 cursor-pointer">
                      <ZoomIn className="w-3.5 h-3.5" />
                    </button>
                    <button type="button" onClick={() => setHighContrast((v) => !v)} className={`p-1.5 rounded-lg cursor-pointer ${highContrast ? "bg-cyan-500/30 text-cyan-200" : "bg-slate-800/80 text-slate-300"}`}>
                      <Eye className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )}
              </div>

              {activeImage ? (
                <div className="space-y-2">
                  <div className="relative w-full h-44 sm:h-48 overflow-auto rounded-xl border border-slate-700/70 bg-black/60">
                    <img
                      src={activeImage.dataUrl}
                      alt={activeImage.name}
                      style={{ transform: `scale(${zoom})`, transformOrigin: "top left", filter: highContrast ? "contrast(1.45) saturate(1.2)" : undefined }}
                      className="w-full h-auto transition-transform duration-150"
                    />
                  </div>
                  <div className="flex items-center gap-2 overflow-x-auto pb-1">
                    {images.map((img, idx) => (
                      <div
                        key={img.id}
                        className={`relative shrink-0 rounded-lg border overflow-hidden cursor-pointer ${idx === activeImgIdx ? "border-rose-400 ring-1 ring-rose-400" : "border-slate-700"}`}
                        onClick={() => {
                          setActiveImgIdx(idx);
                          void runImageExtraction(img.dataUrl);
                        }}
                      >
                        <img src={img.dataUrl} alt={img.name} className="w-12 h-9 object-cover" />
                        <button type="button" onClick={(e) => { e.stopPropagation(); removeImage(idx); }} className="absolute top-0.5 right-0.5 p-0.5 rounded bg-black/70 text-rose-300 hover:text-rose-100">
                          <Trash2 className="w-2.5 h-2.5" />
                        </button>
                      </div>
                    ))}
                    <label className="shrink-0 w-12 h-9 rounded-lg border border-dashed border-slate-600 hover:border-cyan-400 flex items-center justify-center text-slate-400 hover:text-cyan-300 cursor-pointer">
                      <ImagePlus className="w-4 h-4" />
                      <input type="file" accept="image/*,application/pdf" multiple className="hidden" onChange={(e) => addFiles(e.target.files)} />
                    </label>
                  </div>
                </div>
              ) : (
                <label
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => { e.preventDefault(); setIsDragging(false); addFiles(e.dataTransfer.files); }}
                  className={`flex flex-col items-center justify-center gap-1.5 py-4 px-3 rounded-xl border-2 border-dashed cursor-pointer transition ${isDragging ? "border-cyan-400 bg-cyan-500/10" : "border-slate-700/80 hover:border-rose-400/60 bg-slate-900/40"}`}
                >
                  <ImagePlus className="w-7 h-7 text-rose-400" />
                  <div className="text-xs font-bold text-center">{isAr ? "اسحب صورة الـ ECG هنا أو اضغط للرفع" : "Drop ECG image(s) here or click to upload"}</div>
                  
                  <input type="file" accept="image/*,application/pdf" multiple className="hidden" onChange={(e) => addFiles(e.target.files)} />
                </label>
              )}

              {/* Live Auto-Extraction Status Banner (100% Automatic from Image) */}
              {activeImage && analyzingImg && (
                <div className="rounded-xl bg-cyan-500/10 border border-cyan-400/30 px-3 py-2 text-xs text-cyan-200 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                  <span>{isAr ? "جاري قراءة الصورة..." : "Reading image..."}</span>
                </div>
              )}

              {activeImage && !analyzingImg && (remoteAnalysis || imgAnalysis) && (
                <div className="rounded-xl bg-emerald-500/12 border border-emerald-400/35 px-3 py-2 text-xs font-bold text-emerald-300 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  <span>
                    {images.length > 1
                      ? isAr
                        ? "تم تحليل التخطيطات بنجاح"
                        : "ECGs Analyzed & Ready"
                      : isAr
                        ? "تم تحليل التخطيط بنجاح"
                        : "ECG Analyzed & Ready"}
                  </span>
                </div>
              )}
            </div>

            {/* Grad-CAM Visual Focus Panel directly below ECG Image inside the scrollable column */}
            {showResults && gradcamUrl && (
                <EcgGradCamPanel
                  isAr={isAr}
                  isLight={isLight}
                  cardCls={cardCls}
                  gradcamUrl={gradcamUrl}
                  gradcamFocusRegion={gradcamFocusRegion}
                />
            )}

            {/* Auto-Filled Printed Header Numbers (hidden on Diagnosis Results screen to avoid repetition) */}
            {!showResults && (
            <div className={`rounded-2xl border p-3 space-y-2 ${cardCls}`}>
              <div className="flex items-center justify-between text-xs font-extrabold">
                <span className="flex items-center gap-2">
                  <SlidersHorizontal className="w-4 h-4 text-amber-400" />
                  {isAr ? "أرقام التخطيط (Header)" : "ECG Header Numbers"}
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                {[
                  { key: "hr_bpm", label: "HR (bpm)", ph: "—" },
                  { key: "pr_ms", label: "PR (ms)", ph: "—" },
                  { key: "qrs_ms", label: "QRS (ms)", ph: "—" },
                  { key: "qt_ms", label: "QT (ms)", ph: "—" },
                  { key: "axis_deg", label: "Axis (°)", ph: "—" },
                ].map((fld) => (
                  <label key={fld.key} className="space-y-1">
                    <span className="text-[11px] font-mono text-slate-400">{fld.label}</span>
                    <input
                      type="number"
                      placeholder={fld.ph}
                      value={(features[fld.key as keyof WizardFeatures] as number | null) ?? ""}
                      onChange={(e) => { setMeasurementSource("manual"); setFeatures((p) => ({ ...p, [fld.key]: parseOptNum(e.target.value) })); }}
                      className={`w-full px-2.5 py-1.5 rounded-lg border text-xs font-mono focus:outline-none focus:border-cyan-400 ${isLight ? "bg-white/85 border-slate-300 text-slate-900" : "bg-[#0a1224]/75 border-white/15 text-white"}`}
                    />
                  </label>
                ))}
              </div>
            </div>
            )}
          </div>

          {/* RIGHT: ONLY 3 Patient Clinical Context Cards OR Hybrid Results */}
          <div className="lg:col-span-8 space-y-3">
            {!showResults ? (
              <div className={`rounded-2xl border p-4 sm:p-5 space-y-4 ${cardCls}`}>
                {/* 4-Step Progress Bar */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs font-bold">
                    <span className="text-rose-400 flex items-center gap-1.5">
                      <UserCheck className="w-4 h-4" />
                      {isAr ? `حالة المريض — بطاقة ${step} من 4` : `Patient Context — Card ${step} of 4`}
                    </span>
                    <div className="flex items-center gap-1.5">
                      {[1, 2, 3, 4].map((n) => (
                        <button
                          key={n}
                          type="button"
                          onClick={() => setStep(n)}
                          className={`w-6 h-6 rounded-full text-[11px] font-bold cursor-pointer transition ${
                            n === step
                              ? "bg-rose-500 text-white"
                              : n < step
                                ? "bg-emerald-500/25 text-emerald-300 border border-emerald-500/40"
                                : "bg-slate-800 text-slate-400"
                          }`}
                        >
                          {n}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-rose-500 to-emerald-400 transition-all duration-300" style={{ width: `${(step / 4) * 100}%` }} />
                  </div>
                </div>

                {/* CARD 1: Compact Vitals & Hemodynamic Stability Mini-Panels */}
                {step === 1 && (
                  <div className="space-y-2.5">
                    <h2 className="text-base font-black flex items-center gap-2">
                      <HeartPulse className="w-5 h-5 text-cyan-400" />
                      <span>{isAr ? "1. العلامات الحيوية والاستقرار (Vitals)" : "1. Vitals & Hemodynamic Stability"}</span>
                    </h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <div className="rounded-xl border border-white/10 bg-slate-900/50 p-2.5 space-y-1.5">
                        <div className="text-[11px] font-bold text-slate-300">{isAr ? "ضغط الدم (BP)" : "Blood Pressure"}</div>
                        <div className="grid grid-cols-3 gap-1">
                          {([
                            { v: "normal", ar: "طبيعي", en: "Normal" },
                            { v: "high", ar: "مرتفع", en: "High" },
                            { v: "low", ar: "منخفض", en: "Low" },
                          ] as const).map((o) => (
                            <button
                              key={o.v}
                              type="button"
                              onClick={() => setFeatures((p) => ({ ...p, bp_status: o.v }))}
                              className={`py-1.5 rounded-lg text-[11px] font-bold border cursor-pointer transition ${
                                features.bp_status === o.v
                                  ? o.v === "normal"
                                    ? "bg-emerald-500/25 border-emerald-400 text-emerald-200"
                                    : "bg-rose-500/25 border-rose-400 text-rose-100"
                                  : "bg-slate-800/70 border-slate-700 text-slate-400 hover:border-slate-500"
                              }`}
                            >
                              {isAr ? o.ar : o.en}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-slate-900/50 p-2.5 space-y-1.5">
                        <div className="text-[11px] font-bold text-slate-300">{isAr ? "سكر الدم (Glucose)" : "Blood Glucose"}</div>
                        <div className="grid grid-cols-3 gap-1">
                          {([
                            { v: "normal", ar: "طبيعي", en: "Normal" },
                            { v: "high", ar: "مرتفع/DKA", en: "High/DKA" },
                            { v: "low", ar: "منخفض", en: "Low" },
                          ] as const).map((o) => (
                            <button
                              key={o.v}
                              type="button"
                              onClick={() => setFeatures((p) => ({ ...p, glucose_status: o.v }))}
                              className={`py-1.5 rounded-lg text-[11px] font-bold border cursor-pointer transition ${
                                features.glucose_status === o.v
                                  ? o.v === "normal"
                                    ? "bg-emerald-500/25 border-emerald-400 text-emerald-200"
                                    : "bg-amber-500/25 border-amber-400 text-amber-100"
                                  : "bg-slate-800/70 border-slate-700 text-slate-400 hover:border-slate-500"
                              }`}
                            >
                              {isAr ? o.ar : o.en}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-slate-900/50 p-2.5 space-y-1.5">
                        <div className="text-[11px] font-bold text-slate-300">{isAr ? "الأكسجين (SpO₂)" : "Oxygen (SpO₂)"}</div>
                        <div className="grid grid-cols-2 gap-1">
                          {([
                            { v: "normal", ar: "طبيعي (≥95%)", en: "Normal (≥95%)" },
                            { v: "low", ar: "ناقص (<92%)", en: "Low (<92%)" },
                          ] as const).map((o) => (
                            <button
                              key={o.v}
                              type="button"
                              onClick={() => setFeatures((p) => ({ ...p, spo2_status: o.v }))}
                              className={`py-1.5 rounded-lg text-[11px] font-bold border cursor-pointer transition ${
                                features.spo2_status === o.v
                                  ? o.v === "normal"
                                    ? "bg-emerald-500/25 border-emerald-400 text-emerald-200"
                                    : "bg-rose-500/25 border-rose-400 text-rose-100"
                                  : "bg-slate-800/70 border-slate-700 text-slate-400 hover:border-slate-500"
                              }`}
                            >
                              {isAr ? o.ar : o.en}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-xl border border-white/10 bg-slate-900/50 p-2.5 space-y-1.5">
                        <div className="text-[11px] font-bold text-slate-300">{isAr ? "الحرارة (Temp)" : "Temperature"}</div>
                        <div className="grid grid-cols-3 gap-1">
                          {([
                            { v: "normal", ar: "طبيعية", en: "Normal" },
                            { v: "fever", ar: "حمى ≥38°", en: "Fever" },
                            { v: "hypothermia", ar: "برد <35°", en: "Hypoth." },
                          ] as const).map((o) => (
                            <button
                              key={o.v}
                              type="button"
                              onClick={() => setFeatures((p) => ({ ...p, temp_status: o.v }))}
                              className={`py-1.5 rounded-lg text-[11px] font-bold border cursor-pointer transition ${
                                features.temp_status === o.v
                                  ? o.v === "normal"
                                    ? "bg-emerald-500/25 border-emerald-400 text-emerald-200"
                                    : "bg-amber-500/25 border-amber-400 text-amber-100"
                                  : "bg-slate-800/70 border-slate-700 text-slate-400 hover:border-slate-500"
                              }`}
                            >
                              {isAr ? o.ar : o.en}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="rounded-xl border border-white/10 bg-slate-900/50 p-2.5 flex flex-wrap items-center justify-between gap-2">
                      <span className="text-[11px] font-bold text-slate-300">{isAr ? "الحالة العامة والوعي:" : "Hemodynamic Status:"}</span>
                      <div className="flex gap-1.5">
                        <button
                          type="button"
                          onClick={() => setFeatures((p) => ({ ...p, hemodynamic_unstable: false }))}
                          className={`px-3 py-1 rounded-lg text-[11px] font-bold border cursor-pointer ${
                            !features.hemodynamic_unstable
                              ? "bg-emerald-500/25 border-emerald-400 text-emerald-200"
                              : "bg-slate-800/70 border-slate-700 text-slate-400"
                          }`}
                        >
                          {isAr ? "مستقر وواعٍ" : "Stable / Alert"}
                        </button>
                        <button
                          type="button"
                          onClick={() => setFeatures((p) => ({ ...p, hemodynamic_unstable: true }))}
                          className={`px-3 py-1 rounded-lg text-[11px] font-bold border cursor-pointer ${
                            features.hemodynamic_unstable
                              ? "bg-rose-500/30 border-rose-400 text-rose-100"
                              : "bg-slate-800/70 border-slate-700 text-slate-400"
                          }`}
                        >
                          {isAr ? "غير مستقر (صدمة / اضطراب وعي)" : "Unstable (Shock / Altered)"}
                        </button>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => setStep(2)}
                      className="w-full py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-black text-xs cursor-pointer"
                    >
                      {isAr ? "التالي: الأعراض الحالية ←" : "Next: Current Symptoms →"}
                    </button>
                  </div>
                )}

                {/* CARD 2: Symptoms */}
                {step === 2 && (
                  <div className="space-y-2.5">
                    <h2 className="text-base font-black flex items-center gap-2">
                      <HeartPulse className="w-5 h-5 text-rose-400" />
                      <span>{isAr ? "2. هل المريض يعاني من أعراض حالياً؟" : "2. Does the patient have current symptoms?"}</span>
                    </h2>
                    <button
                      type="button"
                      onClick={() => {
                        setSymptomChoice("chest_pain");
                        setFeatures((p) => ({ ...p, chest_pain_or_shock: true, asymptomatic_nocturnal: false, syncope_or_family_scd: false, pleuritic_or_dehydration_or_bleed: false }));
                        setStep(3);
                      }}
                      className={btnOptionCls(symptomChoice === "chest_pain")}
                    >
                      <span className="font-bold text-sm">{isAr ? "ألم صدري حاد / ضيق نفس شديد / صدمة دورانية" : "Acute Chest Pain / Severe Dyspnea / Shock"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setSymptomChoice("syncope");
                        setFeatures((p) => ({ ...p, chest_pain_or_shock: false, asymptomatic_nocturnal: false, syncope_or_family_scd: true, pleuritic_or_dehydration_or_bleed: false }));
                        setStep(3);
                      }}
                      className={btnOptionCls(symptomChoice === "syncope")}
                    >
                      <span className="font-bold text-sm">{isAr ? "دوخة شديدة / إغماء / خفقان" : "Dizziness / Syncope / Palpitations"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setSymptomChoice("secondary_trigger");
                        setFeatures((p) => ({ ...p, chest_pain_or_shock: false, asymptomatic_nocturnal: false, syncope_or_family_scd: false, pleuritic_or_dehydration_or_bleed: true }));
                        setStep(3);
                      }}
                      className={btnOptionCls(symptomChoice === "secondary_trigger")}
                    >
                      <span className="font-bold text-sm">{isAr ? "ألم جنبي مع النفس / جفاف شديد / نزيف / عدوى" : "Pleuritic Pain / Dehydration / Bleeding / Infection"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setSymptomChoice("asymptomatic");
                        setFeatures((p) => ({ ...p, chest_pain_or_shock: false, asymptomatic_nocturnal: true, syncope_or_family_scd: false, pleuritic_or_dehydration_or_bleed: false }));
                        setStep(3);
                      }}
                      className={btnOptionCls(symptomChoice === "asymptomatic")}
                    >
                      <span className="font-bold text-sm">{isAr ? "بدون أعراض (فحص روتيني أو حدث أثناء النوم)" : "Asymptomatic (Routine Check or Nocturnal)"}</span>
                    </button>
                  </div>
                )}

                {/* CARD 3: Competitive Athlete */}
                {step === 3 && (
                  <div className="space-y-2.5">
                    <h2 className="text-base font-black">
                      {isAr ? "3. هل المريض رياضي محترف (تمرين شاق يومياً)؟" : "3. Is the patient a Competitive Athlete (intense daily training)?"}
                    </h2>
                    <button
                      type="button"
                      onClick={() => {
                        setFeatures((p) => ({ ...p, is_competitive_athlete: true }));
                        setStep(4);
                      }}
                      className={btnOptionCls(features.is_competitive_athlete)}
                    >
                      <span className="font-bold text-sm">{isAr ? "نعم — رياضي محترف / تمرين شاق يوميًا" : "Yes — Competitive Athlete (intense daily training)"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setFeatures((p) => ({ ...p, is_competitive_athlete: false }));
                        setStep(4);
                      }}
                      className={btnOptionCls(!features.is_competitive_athlete && step > 3)}
                    >
                      <span className="font-bold text-sm">{isAr ? "لا — مريض عادي" : "No — Non-athlete"}</span>
                    </button>
                  </div>
                )}

                {/* CARD 4: Quick Medical Background */}
                {step === 4 && (
                  <div className="space-y-3">
                    <h2 className="text-base font-black">
                      {isAr ? "4. هل ينطبق على المريض أي من الحالات التالية؟" : "4. Do any of these clinical backgrounds apply?"}
                    </h2>
                    <div className="flex flex-wrap gap-2">
                      {MEDICAL_CONTEXT_CHIPS.map((chip) => {
                        const active = Boolean(features[chip.key]);
                        return (
                          <button
                            key={chip.key}
                            type="button"
                            onClick={() => toggleFlag(chip.key)}
                            className={`px-3 py-2.5 rounded-xl text-xs font-bold border transition cursor-pointer ${
                              active
                                ? "bg-emerald-500/20 border-emerald-400 text-emerald-200 shadow-sm"
                                : "bg-slate-900/75 border-slate-700 text-slate-300 hover:border-slate-500"
                            }`}
                          >
                            {active ? "✓ " : "+ "}
                            {isAr ? chip.labelAr : chip.labelEn}
                          </button>
                        );
                      })}
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowResults(true)}
                      className="w-full py-3 rounded-xl bg-gradient-to-r from-rose-600 to-pink-600 hover:from-rose-500 hover:to-pink-500 text-white font-black text-sm flex items-center justify-center gap-2 shadow-lg cursor-pointer"
                    >
                      <Sparkles className="w-4 h-4" />
                      <span>{isAr ? "اعرض التشخيص الهجين والفخاخ السريرية" : "Show Hybrid Diagnosis & Clinical Pitfalls"}</span>
                    </button>
                  </div>
                )}

                <div className="flex items-center justify-between pt-2 border-t border-slate-800/70">
                  <button
                    type="button"
                    disabled={step === 1}
                    onClick={() => setStep((s) => Math.max(1, s - 1))}
                    className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800/70 disabled:opacity-35 text-slate-200 cursor-pointer"
                  >
                    {isAr ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
                    <span>{isAr ? "السابق" : "Back"}</span>
                  </button>
                  {step < 4 && (
                    <button
                      type="button"
                      onClick={() => setStep((s) => Math.min(4, s + 1))}
                      className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800/70 hover:bg-slate-700 text-slate-200 cursor-pointer"
                    >
                      <span>{isAr ? "تخطي" : "Skip"}</span>
                      {isAr ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    </button>
                  )}
                </div>
              </div>
            ) : (
                            hybridResult && (
                <EcgResultsPanel
                  result={hybridResult}
                  isAr={isAr}
                  isLight={isLight}
                  cardCls={cardCls}
                  onEditInputs={() => setShowResults(false)}
                  onApplyCheck={handleApplyCheck}
                  measurementSource={measurementSource}
                />
              )
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
