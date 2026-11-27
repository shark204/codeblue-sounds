"use client";
import React, { useEffect, useMemo, useState } from "react";
import { Activity, CheckCircle2, ChevronLeft, ChevronRight, Eye, HeartPulse, ImagePlus, Camera, ClipboardPaste, Layers, Loader2, RotateCcw, SlidersHorizontal, Sparkles, Trash2, UserCheck, Zap, ZoomIn, ZoomOut } from "lucide-react";
import { INITIAL_WIZARD_FEATURES, WizardFeatures, runHybridEcgDiagnosis } from "@/lib/ecgHybridEngine";
import { LocalECGAnalysisResult, analyzeEcgImageLocally } from "@/lib/ecgImageEngine";
import { EcgResultsPanel, EcgGradCamPanel } from "@/features/ecg/EcgDiagnosisView";
import { EcgSmartCameraModal } from "@/features/ecg/EcgSmartCameraModal";
import { EcgSmartConfirmCard, getRelevantWizardSteps, getSmartConfirmGroup } from "@/features/ecg/EcgSmartConfirmCard";
import { makeProvenanced } from "@/lib/ecgMeasurementUncertainty";
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
  const [pasteMenu, setPasteMenu] = useState<{ x: number; y: number } | null>(null);
  const [pasteHint, setPasteHint] = useState<string | null>(null);
  const [cameraOpen, setCameraOpen] = useState<boolean>(false);
  const handlePasteFromClipboard = async () => {
    setPasteMenu(null);
    setPasteHint(null);
    try {
      if (navigator.clipboard && typeof navigator.clipboard.read === "function") {
        const items = await navigator.clipboard.read();
        const found: File[] = [];
        for (const item of items) {
          const imgType = item.types.find((t) => t.startsWith("image/"));
          if (imgType) {
            const blob = await item.getType(imgType);
            const ext = imgType.split("/")[1] || "png";
            found.push(new File([blob], `pasted-ecg-${Date.now()}.${ext}`, { type: imgType }));
          }
        }
        if (found.length > 0) {
          const dt = new DataTransfer();
          found.forEach((f) => dt.items.add(f));
          addFiles(dt.files);
          return;
        }
        setPasteHint(isAr ? "لا توجد صورة منسوخة في الحافظة — انسخ صورة ECG أولاً." : "No image found in clipboard — copy an ECG image first.");
      } else {
        setPasteHint(isAr ? "استخدم Ctrl+V للصق الصورة مباشرة." : "Press Ctrl+V to paste an image directly.");
      }
    } catch {
      setPasteHint(isAr ? "اسمح بالوصول للحافظة أو اضغط Ctrl+V للصق الصورة." : "Allow clipboard access or press Ctrl+V to paste.");
    }
  };
  useEffect(() => {
    if (!pasteMenu) return;
    const close = () => setPasteMenu(null);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setPasteMenu(null); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [pasteMenu]);
  useEffect(() => {
    if (!pasteHint) return;
    const t = setTimeout(() => setPasteHint(null), 4000);
    return () => clearTimeout(t);
  }, [pasteHint]);
  const [analyzingImg, setAnalyzingImg] = useState<boolean>(false);
  const [imgAnalysis, setImgAnalysis] = useState<LocalECGAnalysisResult | null>(null);
  const [backendSource, setBackendSource] = useState<"modal-ai" | "local" | null>(null);
  const [remoteAnalysis, setRemoteAnalysis] = useState<RemoteAnalysisSummary | null>(null);
  // ONLY 3 Patient-Context Cards (Zero ECG-reading cards!)
  const [step, setStep] = useState<number>(1);
  const [showResults, setShowResults] = useState<boolean>(false);
  const [awaitingImage, setAwaitingImage] = useState(false);
  const [smartConfirmOpen, setSmartConfirmOpen] = useState<boolean>(false);
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
            if (j.invalid_ecg_image) {
              setImgAnalysis(isAr ? "⚠️ الصورة لا تبدو تخطيط ECG واضحاً (ورقة فارغة أو ضوضاء أو طمس شديد) — أعد التصوير بوضوح." : "⚠️ Image does not appear to be a readable ECG (blank, noisy, or too blurry) — please retake clearly.");
              setFeatures((prev) => ({ ...prev, ai_predictions: [], lead_reversal_alert: false }));
              return;
            }
            const aiList: { label: string; code: string | null; prob: number; uncertainty?: number }[] = Array.isArray(j.ai_diagnoses)
              ? j.ai_diagnoses.map((d: { label: string; code: string | null; prob: number; uncertainty?: number; uncertainty_std?: number }) => ({
                  ...d,
                  uncertainty: typeof d.uncertainty === "number" ? d.uncertainty : typeof d.uncertainty_std === "number" ? Math.min(0.9, d.uncertainty_std * 2.5) : undefined,
                }))
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
            // Full automatic waveform estimation fallback with per-field provenance & P95 uncertainty
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
            // Clinical safety: pixel P95 errors for PR (±80ms), QRS (±135ms), QT (±173ms) make pixel interval
            // estimation clinically unusable. Populate PR/QRS/QT ONLY from printed device OCR or manual entry.
            const aiIv = j.ai_intervals ?? {};
            const nextHr = bm.hr_bpm ?? aiIv.hr_bpm ?? fallbackM.hr_bpm ?? null;
            const nextPr = bm.pr_ms ?? aiIv.pr_ms ?? null;
            const nextQrs = bm.qrs_ms ?? aiIv.qrs_ms ?? null;
            const nextQt = bm.qt_ms ?? aiIv.qt_ms ?? null;
            const nextAx = bm.axis_deg ?? aiIv.axis_deg ?? null;
            setFeatures((prev) => ({
              ...prev,
              paper_speed_mm_s: detSpeed,
              gain_mm_mv: detGain,
              hr_bpm: nextHr ?? prev.hr_bpm,
              pr_ms: nextPr ?? prev.pr_ms,
              qrs_ms: nextQrs ?? prev.qrs_ms,
              qt_ms: nextQt ?? prev.qt_ms,
              axis_deg: nextAx ?? prev.axis_deg,
              measurement_provenance: {
                hr_bpm: makeProvenanced(nextHr, bm.hr_bpm != null ? "device_ocr" : "pixel_estimate", "hr_bpm"),
                pr_ms: makeProvenanced(nextPr, bm.pr_ms != null ? "device_ocr" : "pixel_estimate", "pr_ms"),
                qrs_ms: makeProvenanced(nextQrs, bm.qrs_ms != null ? "device_ocr" : "pixel_estimate", "qrs_ms"),
                qt_ms: makeProvenanced(nextQt, bm.qt_ms != null ? "device_ocr" : "pixel_estimate", "qt_ms"),
                axis_deg: makeProvenanced(nextAx, bm.axis_deg != null ? "device_ocr" : "pixel_estimate", "axis_deg"),
              },
              ai_predictions: aiList, lead_reversal_alert: Boolean(j.lead_reversal_alert),
            }));
            const hasOcrNum = bm.hr_bpm != null || bm.pr_ms != null || bm.qrs_ms != null || bm.qt_ms != null || bm.axis_deg != null;
            const hasFallbackNum = aiIv.hr_bpm != null || aiIv.qrs_ms != null || aiIv.qt_ms != null || fallbackM.hr_bpm != null;
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
      // Only HR may use approximate visual estimate; PR/QRS/QT require printed header or manual entry
      const offHr = m.hr_bpm != null ? Math.round(m.hr_bpm) : null;
      const offPr = null;
      const offQrs = null;
      const offQt = null;
      setFeatures((prev) => ({
        ...prev,
        paper_speed_mm_s: 25,
        gain_mm_mv: 10,
        hr_bpm: offHr ?? prev.hr_bpm,
        pr_ms: offPr ?? prev.pr_ms,
        qrs_ms: offQrs ?? prev.qrs_ms,
        qt_ms: offQt ?? prev.qt_ms,
        measurement_provenance: {
          hr_bpm: makeProvenanced(offHr, "pixel_estimate", "hr_bpm"),
          pr_ms: makeProvenanced(offPr, "pixel_estimate", "pr_ms"),
          qrs_ms: makeProvenanced(offQrs, "pixel_estimate", "qrs_ms"),
          qt_ms: makeProvenanced(offQt, "pixel_estimate", "qt_ms"),
        },
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
        ai_predictions: [], lead_reversal_alert: false,
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
  const smartConfirmGroup = useMemo(
    () => getSmartConfirmGroup(remoteAnalysis?.ai_diagnoses),
    [remoteAnalysis]
  );
  const hasAnalyzedImage = images.length > 0 && !imgAnalysis && ((features.ai_predictions?.length ?? 0) > 0 || remoteAnalysis !== null);
  const visibleSteps = useMemo(() => (hasAnalyzedImage ? getRelevantWizardSteps(features.ai_predictions, features.hr_bpm) : [1, 2]), [hasAnalyzedImage, features.ai_predictions, features.hr_bpm]);
  const stepIdx = Math.max(0, visibleSteps.indexOf(step));
  const requestShowResults = () => {
    if (!hasAnalyzedImage) {
      setAwaitingImage(true);
      return;
    }
    if (smartConfirmGroup) {
      setSmartConfirmOpen(true);
    } else {
      setShowResults(true);
    }
  };
  const advanceFrom = (cur: number) => {
    if (cur === 2) {
      if (!hasAnalyzedImage) { setAwaitingImage(true); return; }
      const nextAfter2 = visibleSteps.find((n) => n > 2);
      if (nextAfter2) setStep(nextAfter2); else requestShowResults();
      return;
    }
    const i = visibleSteps.indexOf(cur);
    if (i >= 0 && i < visibleSteps.length - 1) setStep(visibleSteps[i + 1]); else requestShowResults();
  };
  useEffect(() => { if (!visibleSteps.includes(step)) setStep(visibleSteps[0] ?? 1); }, [visibleSteps, step]);
  useEffect(() => {
    if (!awaitingImage || !hasAnalyzedImage) return;
    setAwaitingImage(false);
    const nextAfter2 = visibleSteps.find((n) => n > 2);
    if (nextAfter2) setStep(nextAfter2); else requestShowResults();
  }, [awaitingImage, hasAnalyzedImage, visibleSteps]);
  const resetAll = () => {
    setFeatures(INITIAL_WIZARD_FEATURES);
    setSymptomChoice(null);
    setStep(1);
    setShowResults(false);
    setSmartConfirmOpen(false); setAwaitingImage(false);
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
    try { return runHybridEcgDiagnosis(features); } catch (err) { console.error("Hybrid ECG diagnosis error:", err); return null; }
  }, [showResults, features]);
  const cardCls = isLight
    ? "cb-glass-card text-slate-900"
    : "cb-glass-card text-slate-100";
  const btnOptionCls = (active?: boolean) =>
    `w-full min-h-[44px] text-start px-4 py-3.5 rounded-2xl border cb-hover-lift cursor-pointer flex items-center justify-between gap-3 ${
      active
        ? isLight
          ? "bg-rose-500/15 border-[#9f1239] text-[#881337] shadow-[0_0_18px_rgba(159,18,57,0.16)]"
          : "bg-[#9f1239]/28 border-rose-400/75 text-rose-50 shadow-[0_0_20px_rgba(159,18,57,0.28)]"
        : isLight
          ? "cb-glass-nested hover:bg-teal-50/80 hover:border-teal-400/70 text-slate-800"
          : "cb-glass-nested hover:bg-white/10 hover:border-teal-400/50 text-slate-100"
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
        {/* Top Engine Floating Glass Pill */}
        <div className="cb-glass-pill px-4 sm:px-5 py-2 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-[#9f1239]/25 border border-rose-400/45 flex items-center justify-center text-rose-300">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-black text-base sm:text-lg leading-tight">
                {isAr ? "محرك تشخيص الـ ECG" : "ECG Diagnostic Engine"}
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`inline-flex items-center gap-1.5 min-h-[40px] px-3.5 py-1.5 rounded-full text-[11px] font-extrabold border ${showResults ? "bg-emerald-500/15 border-emerald-400/40 text-emerald-300" : imgAnalysis ? "bg-amber-500/15 border-amber-400/40 text-amber-200" : hasAnalyzedImage ? "bg-teal-500/15 border-teal-400/40 text-teal-200" : "bg-white/5 border-white/10 text-slate-300"}`}>
              {imgAnalysis ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5" />}
              <span>{showResults ? (isAr ? "تم التشخيص" : "Diagnosis Ready") : imgAnalysis ? (isAr ? "جاري تحليل الصورة..." : "Analyzing ECG...") : hasAnalyzedImage ? (isAr ? "تمت القراءة — أكمل البطاقات" : "ECG Read — Complete Cards") : (isAr ? "بانتظار صورة الـ ECG" : "Awaiting ECG Image")}</span>
            </div>
            <button
              type="button"
              disabled={images.length === 0 && step === 1 && !showResults && !awaitingImage && !symptomChoice}
              onClick={resetAll}
              className={`inline-flex items-center gap-1.5 min-h-[44px] px-3.5 py-2 rounded-full text-xs font-bold border cb-hover-lift disabled:opacity-35 disabled:cursor-not-allowed cursor-pointer ${isLight ? "bg-white/70 border-slate-300/80 text-slate-800 hover:bg-white" : "cb-glass-nested text-slate-100 hover:border-teal-400/40"}`}
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>{isAr ? "حالة جديدة" : "New Case"}</span>
            </button>
          </div>
        </div>
        {/* Main 2-Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 items-start">
          {/* LEFT: Auto-Analyzing ECG Image Uploader + Auto-Filled Printed Header Numbers */}
          <div className="lg:col-span-4 space-y-2.5">
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
                  <div className="relative w-full overflow-auto rounded-xl border border-slate-700/70 bg-black/60">
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
                        <button type="button" onClick={(e) => { e.stopPropagation(); removeImage(idx); }} className="absolute top-0.5 end-0.5 p-0.5 rounded bg-black/70 text-rose-300 hover:text-rose-100">
                          <Trash2 className="w-2.5 h-2.5" />
                        </button>
                      </div>
                    ))}
                    <label onContextMenu={(e) => { e.preventDefault(); setPasteMenu({ x: e.clientX, y: e.clientY }); }} className="shrink-0 w-12 h-9 rounded-lg border border-dashed border-slate-600 hover:border-cyan-400 flex items-center justify-center text-slate-400 hover:text-cyan-300 cursor-pointer">
                      <ImagePlus className="w-4 h-4" />
                      <input type="file" accept="image/*,application/pdf" multiple className="hidden" onChange={(e) => addFiles(e.target.files)} />
                    </label>
                  </div>
                </div>
              ) : (
                <label
                  onContextMenu={(e) => { e.preventDefault(); setPasteMenu({ x: e.clientX, y: e.clientY }); }}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => { e.preventDefault(); setIsDragging(false); addFiles(e.dataTransfer.files); }}
                  className={`flex flex-col items-center justify-center gap-2 min-h-[112px] py-5 px-4 rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 ${isDragging ? "cb-dropzone-active" : isLight ? "border-slate-300 hover:border-teal-500/70 bg-white/45" : "border-white/20 hover:border-teal-400/60 bg-white/[0.03]"}`}
                >
                  <ImagePlus className="cb-drop-icon w-7 h-7 text-rose-400" />
                  <div className="text-xs font-bold text-center">{isAr ? "اسحب صورة الـ ECG هنا أو اضغط للرفع — أو كليك يمين للصق" : "Drop ECG image(s) here, click to upload, or right-click to Paste"}</div>
                  
                  <input type="file" accept="image/*,application/pdf" multiple className="hidden" onChange={(e) => addFiles(e.target.files)} />
                </label>
              )}

              <button
                type="button"
                onClick={() => setCameraOpen(true)}
                className="md:hidden cb-btn-primary w-full min-h-[44px] py-2.5 px-4 rounded-xl text-xs font-extrabold flex items-center justify-center gap-2 cursor-pointer"
              >
                <Camera className="w-4 h-4" />
                <span>{isAr ? "مسح الـ ECG بالكاميرا الذكية (التقاط تلقائي)" : "Smart Camera ECG Scan (Auto-Capture)"}</span>
              </button>

              {/* Live Auto-Extraction Status Banner (100% Automatic from Image) */}
              {activeImage && !diagnosisOpen && (
                <button
                  type="button"
                  onClick={() => cardsSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                  className="lg:hidden w-full min-h-[42px] px-3.5 py-2 rounded-xl border border-cyan-400/50 bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-100 text-xs font-extrabold flex items-center justify-center gap-2 cursor-pointer transition"
                >
                  <ArrowDown className="w-4 h-4 animate-bounce" />
                  <span>{isAr ? `انتقل للبطاقة ${step} (${step === 1 ? "العلامات الحيوية" : step === 2 ? "الأعراض" : "المتابعة السريرية"})` : `Go to Card ${step} (${step === 1 ? "Vitals" : step === 2 ? "Symptoms" : "Clinical Follow-up"})`}</span>
                </button>
              )}
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
                  { key: "hr_bpm", label: "HR", unit: "bpm", ph: "—" },
                  { key: "pr_ms", label: "PR", unit: "ms", ph: "—" },
                  { key: "qrs_ms", label: "QRS", unit: "ms", ph: "—" },
                  { key: "qt_ms", label: "QT", unit: "ms", ph: "—" },
                  { key: "axis_deg", label: "Axis", unit: "°", ph: "—" },
                ].map((fld) => (
                  <label key={fld.key} className="space-y-1 block">
                    <span className={`text-[11px] font-mono font-semibold ${isLight ? "text-slate-600" : "text-slate-300"}`}>{fld.label}</span>
                    <div className="relative flex items-center">
                      <input
                        type="number"
                        placeholder={fld.ph}
                        value={(features[fld.key as keyof WizardFeatures] as number | null) ?? ""}
                        onChange={(e) => {
                            const v = parseOptNum(e.target.value);
                            setMeasurementSource("manual");
                            setFeatures((p) => ({
                              ...p,
                              [fld.key]: v,
                              measurement_provenance: {
                                ...(p.measurement_provenance || {}),
                                [fld.key]: makeProvenanced(v, "manual", fld.key as "hr_bpm" | "pr_ms" | "qrs_ms" | "qt_ms" | "axis_deg"),
                              },
                            }));
                          }}
                        className={`cb-glass-input w-full min-h-[44px] ps-3 pe-10 py-2 rounded-xl text-xs font-mono ${isLight ? "text-slate-900 placeholder:text-slate-400" : "text-slate-50 placeholder:text-slate-400"}`}
                      />
                      <span
                        aria-hidden="true"
                        className={`pointer-events-none absolute end-2.5 text-[10.5px] font-mono font-semibold ${isLight ? "text-slate-500" : "text-slate-400"}`}
                      >
                        {fld.unit}
                      </span>
                    </div>
                  </label>
                ))}
              </div>
            </div>
            )}
          </div>

          {/* RIGHT: ONLY 3 Patient Clinical Context Cards OR Hybrid Results */}
          <div ref={cardsSectionRef} className="lg:col-span-8 space-y-3">
            {!showResults && !smartConfirmOpen && awaitingImage ? (
              <div className={`rounded-2xl border p-8 flex flex-col items-center justify-center text-center space-y-4 min-h-[280px] ${cardCls}`}>
                <Loader2 className="w-10 h-10 text-rose-400 animate-spin" />
                <div className="space-y-1.5 max-w-md">
                  <h3 className="text-base font-black">{images.length > 0 ? (isAr ? "جاري قراءة وتحليل صورة الـ ECG..." : "Reading & Analyzing ECG Image...") : (isAr ? "بانتظار رفع صورة الـ ECG..." : "Waiting for ECG Image Upload...")}</h3>
                  <p className={`text-xs leading-relaxed ${isLight ? "text-slate-600" : "text-slate-300"}`}>
                    {images.length > 0
                      ? (isAr ? "تم تسجيل الفايتلز والأعراض. بمجرد انتهاء القراءة ستظهر البطاقات الخاصة بهذه الحالة أو صفحة التشخيص تلقائياً." : "Vitals & symptoms saved. Once analysis completes, relevant follow-up cards or the diagnosis page will open automatically.")
                      : (isAr ? "تم حفظ الفايتلز وحالة المريض. ارفع أو الصق صورة الـ ECG أو التقطها بالكاميرا الذكية، وبمجرد قراءتها ستظهر البطاقات المناسبة أو التشخيص تلقائياً." : "Vitals & symptoms saved. Upload, paste, or scan the ECG image — relevant cards or the diagnosis will appear automatically once read.")}
                  </p>
                </div>
                <button type="button" onClick={() => { setAwaitingImage(false); setStep(2); }} className="px-4 py-2 rounded-xl text-xs font-bold border border-white/15 bg-white/5 hover:bg-white/10 text-slate-200 cursor-pointer">
                  {isAr ? "الرجوع لتعديل الأعراض" : "Back to Edit Symptoms"}
                </button>
              </div>
            ) : !showResults && !smartConfirmOpen ? (
              <div className={`rounded-2xl border p-4 sm:p-5 space-y-4 ${cardCls}`}>
                {/* Adaptive Progress Bar — ONLY Vitals (Card 1) before ECG image analysis; relevant ordered cards after */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2 text-xs font-bold flex-wrap">
                    <span className="text-rose-300 flex items-center gap-1.5">
                      <UserCheck className="w-4 h-4 shrink-0" />
                      {hasAnalyzedImage
                        ? (isAr ? `حالة المريض — بطاقة ${stepIdx + 1} من ${visibleSteps.length}` : `Patient Context — Card ${stepIdx + 1} of ${visibleSteps.length}`)
                        : (isAr ? `الفايتلز وحالة المريض (${stepIdx + 1} من 2) — باقي البطاقات تظهر بعد قراءة الصورة عند الحاجة` : `Vitals & Patient Status (${stepIdx + 1} of 2) — Extra cards unlock after ECG analysis if needed`)}
                    </span>
                    <div className="flex items-center gap-1.5">
                      {visibleSteps.map((n, idx) => (
                          <button
                            key={n}
                            type="button"
                            onClick={() => setStep(n)}
                            className={`w-8 h-8 rounded-full text-xs font-bold cursor-pointer transition-all duration-200 ${
                              n === step
                                ? "bg-gradient-to-br from-[#881337] to-[#be123c] text-white shadow-[0_0_16px_rgba(159,18,57,0.5)]"
                                : idx < stepIdx
                                  ? isLight ? "bg-teal-500/20 text-teal-800 border border-teal-500/50" : "bg-teal-500/20 text-teal-200 border border-teal-400/45"
                                  : isLight ? "bg-slate-200/80 text-slate-600" : "bg-white/10 text-slate-300 border border-white/10"
                            }`}
                          >
                            {idx + 1}
                          </button>
                        ))}
                    </div>
                  </div>
                  <div className={`h-2 w-full rounded-full overflow-hidden border ${isLight ? "bg-slate-200/70 border-slate-300/60" : "bg-white/10 border-white/10"}`}>
                    <div className="h-full rounded-full bg-gradient-to-r from-[#9f1239] via-[#e11d48] to-teal-400 transition-all duration-250 ease-out" style={{ width: `${((stepIdx + 1) / visibleSteps.length) * 100}%` }} />
                  </div>
                </div>

                {/* CARD 1: Compact Vitals & Hemodynamic Stability Mini-Panels */}
                {step === 1 && (
                  <div className="space-y-2.5">
                    <h2 className="text-base font-black flex items-center gap-2">
                      <HeartPulse className="w-5 h-5 text-cyan-400" />
                      <span>{isAr ? "1. العلامات الحيوية والاستقرار (Vitals)" : "1. Vitals & Hemodynamic Stability"}</span>
                    </h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                      {([
                        {
                          label: isAr ? "ضغط الدم (BP)" : "Blood Pressure",
                          current: features.bp_status,
                          opts: [
                            { v: "normal", ar: "طبيعي", en: "Normal" },
                            { v: "high", ar: "مرتفع", en: "High" },
                            { v: "low", ar: "منخفض", en: "Low" },
                          ],
                          onSelect: (v: string) => setFeatures((p) => ({ ...p, bp_status: v as "normal" | "high" | "low" })),
                        },
                        {
                          label: isAr ? "سكر الدم (Glucose)" : "Blood Glucose",
                          current: features.glucose_status,
                          opts: [
                            { v: "normal", ar: "طبيعي", en: "Normal" },
                            { v: "high", ar: "مرتفع/DKA", en: "High/DKA" },
                            { v: "low", ar: "منخفض", en: "Low" },
                          ],
                          onSelect: (v: string) => setFeatures((p) => ({ ...p, glucose_status: v as "normal" | "high" | "low" })),
                        },
                        {
                          label: isAr ? "الأكسجين (SpO₂)" : "Oxygen (SpO₂)",
                          current: features.spo2_status,
                          opts: [
                            { v: "normal", ar: "طبيعي (≥95%)", en: "Normal (≥95%)" },
                            { v: "low", ar: "ناقص (<92%)", en: "Low (<92%)" },
                          ],
                          onSelect: (v: string) => setFeatures((p) => ({ ...p, spo2_status: v as "normal" | "low" })),
                        },
                        {
                          label: isAr ? "الحرارة (Temp)" : "Temperature",
                          current: features.temp_status,
                          opts: [
                            { v: "normal", ar: "طبيعية", en: "Normal" },
                            { v: "fever", ar: "حمى ≥38°", en: "Fever" },
                            { v: "hypothermia", ar: "برد <35°", en: "Hypoth." },
                          ],
                          onSelect: (v: string) => setFeatures((p) => ({ ...p, temp_status: v as "normal" | "fever" | "hypothermia" })),
                        },
                      ]).map((group) => {
                        const activeIdx = Math.max(0, group.opts.findIndex((o) => o.v === group.current));
                        const pctW = 100 / group.opts.length;
                        return (
                          <div key={group.label} className="cb-glass-nested rounded-2xl p-3 space-y-2">
                            <div className={`text-xs font-bold ${isLight ? "text-slate-700" : "text-slate-200"}`}>{group.label}</div>
                            <div
                              className="cb-segmented"
                              style={{ gridTemplateColumns: `repeat(${group.opts.length}, minmax(0, 1fr))` }}
                            >
                              <div
                                aria-hidden="true"
                                className="cb-segmented-thumb"
                                style={{
                                  width: `calc(${pctW}% - 6px)`,
                                  insetInlineStart: `calc(${activeIdx * pctW}% + 3px)`,
                                }}
                              />
                              {group.opts.map((o) => {
                                const isSel = group.current === o.v;
                                return (
                                  <button
                                    key={o.v}
                                    type="button"
                                    aria-pressed={isSel}
                                    onClick={() => group.onSelect(o.v)}
                                    className={`relative z-10 min-h-[40px] py-2 px-1 rounded-lg text-[11.5px] font-bold cursor-pointer transition-colors duration-150 ${
                                      isSel
                                        ? isLight
                                          ? "text-[#881337]"
                                          : "text-rose-50"
                                        : isLight
                                          ? "text-slate-600 hover:text-slate-900"
                                          : "text-slate-300 hover:text-white"
                                    }`}
                                  >
                                    {isAr ? o.ar : o.en}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="cb-glass-nested rounded-2xl p-3 flex flex-wrap items-center justify-between gap-2">
                      <span className={`text-xs font-bold ${isLight ? "text-slate-700" : "text-slate-200"}`}>{isAr ? "مدة الأعراض / الحالة:" : "Clinical Course:"}</span>
                      <div className="flex flex-wrap gap-1.5">
                        {([{ v: "acute", ar: "حاد (<48س)", en: "Acute (<48h)" }, { v: "subacute", ar: "تحت حاد", en: "Subacute (2d–2w)" }, { v: "chronic", ar: "مزمن (>2أسبوع)", en: "Chronic (>2w)" }] as const).map((o) => (
                          <button key={o.v} type="button" onClick={() => setFeatures((p) => ({ ...p, symptom_onset: o.v }))} className={`min-h-[38px] px-3 py-1.5 rounded-xl text-xs font-bold border cursor-pointer transition ${(features.symptom_onset ?? "acute") === o.v ? "bg-rose-600/30 border-rose-400 text-rose-100" : isLight ? "bg-white/75 border-slate-300 text-slate-700" : "bg-slate-800/70 border-slate-700 text-slate-300"}`}>{isAr ? o.ar : o.en}</button>
                        ))}
                      </div>
                    </div>
                    <div className="cb-glass-nested rounded-2xl p-3 flex flex-wrap items-center justify-between gap-2">
                      <span className={`text-xs font-bold ${isLight ? "text-slate-700" : "text-slate-200"}`}>{isAr ? "الحالة العامة والوعي:" : "Hemodynamic Status:"}</span>
                      <div className="flex gap-1.5">
                        <button
                          type="button"
                          onClick={() => setFeatures((p) => ({ ...p, hemodynamic_unstable: false }))}
                          className={`min-h-[44px] px-3.5 py-2 rounded-xl text-xs font-bold border cursor-pointer transition ${
                            !features.hemodynamic_unstable
                              ? "bg-teal-500/25 border-teal-400 text-teal-100"
                              : "bg-slate-800/70 border-slate-700 text-slate-400"
                          }`}
                        >
                          {isAr ? "مستقر وواعٍ" : "Stable / Alert"}
                        </button>
                        <button
                          type="button"
                          onClick={() => setFeatures((p) => ({ ...p, hemodynamic_unstable: true }))}
                          className={`min-h-[44px] px-3.5 py-2 rounded-xl text-xs font-bold border cursor-pointer transition ${
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
                      className="cb-btn-primary w-full min-h-[44px] py-2.5 rounded-2xl font-black text-xs cursor-pointer"
                    >
                      {isAr ? "التالي: الأعراض الحالية ←" : "Next: Current Symptoms →"}
                    </button>
                  </div>
                )}

                {/* CARD 2: Symptoms */}
                {step === 2 && (
                  <div className="space-y-2.5">
                    <h2 className="text-base font-black flex items-center gap-2">
                      <HeartPulse className="w-5 h-5 text-teal-400" />
                      <span>{isAr ? "2. هل المريض يعاني من أعراض حالياً؟" : "2. Does the patient have current symptoms?"}</span>
                    </h2>
                    <button
                      type="button"
                      onClick={() => {
                        setSymptomChoice("chest_pain");
                        setFeatures((p) => ({ ...p, chest_pain_or_shock: true, asymptomatic_nocturnal: false, syncope_or_family_scd: false, pleuritic_or_dehydration_or_bleed: false }));
                        advanceFrom(2);
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
                        advanceFrom(2);
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
                        advanceFrom(2);
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
                        advanceFrom(2);
                      }}
                      className={btnOptionCls(symptomChoice === "asymptomatic")}
                    >
                      <span className="font-bold text-sm">{isAr ? "بدون أعراض (فحص روتيني أو حدث أثناء النوم)" : "Asymptomatic (Routine Check or Nocturnal)"}</span>
                    </button>
                  </div>
                )}

                {/* CARD 3: Competitive Athlete */}
                {step === 3 && visibleSteps.includes(3) && (
                  <div className="space-y-2.5">
                    <h2 className="text-base font-black">
                      {isAr ? "3. هل المريض رياضي محترف (تمرين شاق يومياً)؟" : "3. Is the patient a Competitive Athlete (intense daily training)?"}
                    </h2>
                    <button
                      type="button"
                      onClick={() => {
                        setFeatures((p) => ({ ...p, is_competitive_athlete: true }));
                        advanceFrom(3);
                      }}
                      className={btnOptionCls(features.is_competitive_athlete)}
                    >
                      <span className="font-bold text-sm">{isAr ? "نعم — رياضي محترف / تمرين شاق يوميًا" : "Yes — Competitive Athlete (intense daily training)"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setFeatures((p) => ({ ...p, is_competitive_athlete: false }));
                        advanceFrom(3);
                      }}
                      className={btnOptionCls(!features.is_competitive_athlete && step > 3)}
                    >
                      <span className="font-bold text-sm">{isAr ? "لا — مريض عادي" : "No — Non-athlete"}</span>
                    </button>
                  </div>
                )}

                {/* CARD 4: Quick Medical Background */}
                {step === 4 && visibleSteps.includes(4) && (
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
                            className={`min-h-[44px] px-3.5 py-2.5 rounded-xl text-xs font-bold border cb-hover-lift cursor-pointer ${
                              active
                                ? isLight
                                  ? "bg-teal-500/20 border-teal-600 text-[#881337] shadow-sm"
                                  : "bg-teal-500/22 border-teal-400 text-rose-50 shadow-sm"
                                : isLight
                                  ? "cb-glass-nested text-slate-700 hover:border-teal-500/50"
                                  : "cb-glass-nested text-slate-200 hover:border-teal-400/50"
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
                      disabled={!hasAnalyzedImage}
                      onClick={requestShowResults}
                      className="cb-btn-primary w-full min-h-[48px] py-3 rounded-2xl font-black text-sm flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                    >
                      <Sparkles className="w-4 h-4" />
                      <span>{isAr ? "اعرض التشخيص الهجين والفخاخ السريرية" : "Show Hybrid Diagnosis & Clinical Pitfalls"}</span>
                    </button>
                  </div>
                )}

                {visibleSteps.length > 1 && (
                  <div className="flex items-center justify-between pt-2 border-t border-slate-800/70">
                    <button
                      type="button"
                      disabled={stepIdx <= 0}
                      onClick={() => setStep(visibleSteps[Math.max(0, stepIdx - 1)] ?? 1)}
                      className="inline-flex items-center gap-1 min-h-[44px] px-3.5 py-2 rounded-xl text-xs font-bold border border-white/10 bg-white/5 hover:bg-white/10 disabled:opacity-35 text-slate-200 cursor-pointer"
                    >
                      {isAr ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
                      <span>{isAr ? "السابق" : "Back"}</span>
                    </button>
                    {stepIdx < visibleSteps.length - 1 && (
                      <button
                        type="button"
                        onClick={() => setStep(visibleSteps[Math.min(visibleSteps.length - 1, stepIdx + 1)] ?? 1)}
                        className="inline-flex items-center gap-1 min-h-[44px] px-3.5 py-2 rounded-xl text-xs font-bold border border-white/10 bg-white/5 hover:bg-white/10 text-slate-200 cursor-pointer"
                      >
                        <span>{isAr ? "التالي" : "Next"}</span>
                        {isAr ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                      </button>
                    )}
                  </div>
                )}
              </div>
            ) : smartConfirmOpen && smartConfirmGroup ? (
              <EcgSmartConfirmCard
                group={smartConfirmGroup}
                features={features}
                isAr={isAr}
                isLight={isLight}
                onUpdateFeatures={setFeatures}
                onConfirm={() => { setSmartConfirmOpen(false); setShowResults(true); }}
                onSkip={() => { setSmartConfirmOpen(false); setShowResults(true); }}
              />
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
                  ecgImageDataUrl={activeImage?.dataUrl ?? null}
                />
              )
            )}
          </div>
        </div>

      </div>
      <EcgSmartCameraModal
        isOpen={cameraOpen}
        isAr={isAr}
        onClose={() => setCameraOpen(false)}
        onCapture={(file) => {
          const dt = new DataTransfer();
          dt.items.add(file);
          addFiles(dt.files);
        }}
      />
      {pasteHint && (
        <div className="fixed bottom-5 end-5 z-50 rounded-xl border border-amber-400/50 bg-slate-950/95 px-4 py-2.5 text-xs font-bold text-amber-200 shadow-2xl backdrop-blur-md">
          {pasteHint}
        </div>
      )}
      {pasteMenu && (
        <div
          style={{ top: pasteMenu.y, left: pasteMenu.x }}
          onClick={(e) => e.stopPropagation()}
          className="fixed z-50 min-w-[180px] rounded-xl border border-white/20 bg-slate-950/95 p-1.5 shadow-2xl backdrop-blur-xl"
        >
          <button
            type="button"
            onClick={() => void handlePasteFromClipboard()}
            className="w-full flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-bold text-slate-100 hover:bg-rose-700/70 transition-colors cursor-pointer"
          >
            <ClipboardPaste className="w-4 h-4 text-rose-300" />
            <span>{isAr ? "لصق صورة (Paste)" : "Paste Image"}</span>
          </button>
        </div>
      )}
    </div>
  );
}
