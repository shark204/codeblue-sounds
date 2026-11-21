"use client";

import React, { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  RotateCcw,
  ScanLine,
  ShieldAlert,
  Sparkles,
  Upload,
} from "lucide-react";
import { ECGInput, ECG_UNITS, interpretECG } from "@/lib/ecgRules";
import { useAtlasLocale } from "@/lib/atlasI18n";
import { useUIStore } from "@/stores/uiStore";

const LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "V4R", "V7", "V8", "V9"];

const PRESETS: { id: string; labelEn: string; labelAr: string; input: ECGInput }[] = [
  {
    id: "anterior_stemi",
    labelEn: "Anterior STEMI",
    labelAr: "جلطة أمامية (Anterior STEMI)",
    input: {
      age: 58, sex: "male", hr_bpm: 94, pr_ms: 160, qrs_ms: 94, qt_ms: 380, axis_deg: 30,
      rhythm_regular: true, p_waves_present: true,
      st_elevation_mm: { V2: 3.5, V3: 4.0, V4: 3.0, II: -1.5, III: -2.0, aVF: -1.5 },
    },
  },
  {
    id: "inferior_rv_stemi",
    labelEn: "Inferior + RV STEMI",
    labelAr: "جلطة سفلية + بطين أيمن",
    input: {
      age: 62, sex: "male", hr_bpm: 54, pr_ms: 190, qrs_ms: 92, qt_ms: 410, axis_deg: 45,
      rhythm_regular: true, p_waves_present: true,
      st_elevation_mm: { II: 2.0, III: 3.5, aVF: 2.5, V4R: 1.5, I: -1.5, aVL: -2.0 },
    },
  },
  {
    id: "lmca",
    labelEn: "LMCA (aVR STE)",
    labelAr: "انسداد LMCA (aVR)",
    input: {
      age: 68, sex: "male", hr_bpm: 108, pr_ms: 164, qrs_ms: 98, qt_ms: 360, axis_deg: 20,
      rhythm_regular: true, p_waves_present: true,
      st_elevation_mm: { aVR: 2.0, V1: 1.0, I: -1.5, II: -2.0, aVF: -1.5, V3: -2.0, V4: -2.5, V5: -2.5, V6: -2.0 },
    },
  },
  {
    id: "vt",
    labelEn: "Monomorphic VT",
    labelAr: "تسارع بطيني (VT)",
    input: {
      age: 66, sex: "male", hr_bpm: 176, qrs_ms: 168, axis_deg: -125,
      rhythm_regular: true, p_waves_present: false, av_dissociation: true, capture_or_fusion_beats: true,
    },
  },
  {
    id: "af_rvr",
    labelEn: "AF with RVR",
    labelAr: "رجفان أذيني سريع (AF RVR)",
    input: {
      age: 65, sex: "female", hr_bpm: 142, qrs_ms: 88, qt_ms: 320, axis_deg: 45,
      rhythm_regular: false, p_waves_present: false, fibrillatory_waves: true,
    },
  },
  {
    id: "preexcited_af",
    labelEn: "Pre-excited AF (WPW)",
    labelAr: "رجفان أذيني مع WPW (حرج)",
    input: {
      age: 26, sex: "male", hr_bpm: 215, qrs_ms: 150, shortest_rr_ms: 210,
      rhythm_regular: false, p_waves_present: false, delta_wave: true, varying_qrs_morphology: true,
    },
  },
  {
    id: "chb",
    labelEn: "Complete Heart Block",
    labelAr: "إحصار قلب كامل (3rd AVB)",
    input: {
      age: 78, sex: "male", hr_bpm: 34, atrial_rate_bpm: 88, qrs_ms: 146,
      av_block_type: "3rd", av_dissociation: true,
    },
  },
  {
    id: "hyperk",
    labelEn: "Severe Hyperkalemia",
    labelAr: "ارتفاع بوتاسيوم حرج",
    input: {
      age: 69, sex: "male", hr_bpm: 52, pr_ms: 240, qrs_ms: 148, qt_ms: 400, axis_deg: 20,
      peaked_t_waves: true, max_t_wave_amplitude_mm: 15, flattened_or_absent_p_hyperk: true,
    },
  },
  {
    id: "brugada1",
    labelEn: "Brugada Type 1",
    labelAr: "بروجادا النوع الأول",
    input: {
      age: 38, sex: "male", hr_bpm: 74, pr_ms: 176, qrs_ms: 106, qt_ms: 380, axis_deg: 45,
      brugada_pattern: "type_1", coved_ste_v1_v2: true, st_elevation_mm: { V1: 3.0, V2: 2.5 },
    },
  },
  {
    id: "pericarditis",
    labelEn: "Acute Pericarditis",
    labelAr: "التهاب التامور الحاد",
    input: {
      age: 29, sex: "male", hr_bpm: 96, pr_ms: 152, qrs_ms: 86, qt_ms: 360, axis_deg: 50,
      st_morphology_concave: true, pr_depression: true, v6_st_t_ratio: 0.4,
      st_elevation_mm: { I: 1.5, II: 2.0, aVF: 1.5, V4: 2.0, V5: 2.0, V6: 1.8, aVR: -1.5 },
    },
  },
  {
    id: "pe",
    labelEn: "Pulmonary Embolism",
    labelAr: "جلطة رئوية (S1Q3T3)",
    input: {
      age: 54, sex: "female", hr_bpm: 122, pr_ms: 150, qrs_ms: 92, qt_ms: 335, axis_deg: 95,
      s1q3t3: true, rv_strain_twi_v1_v4: true,
    },
  },
  {
    id: "nsr",
    labelEn: "Normal Sinus Rhythm",
    labelAr: "إيقاع جيبي طبيعي (NSR)",
    input: {
      age: 30, sex: "male", hr_bpm: 75, pr_ms: 160, qrs_ms: 88, qt_ms: 380, axis_deg: 45,
      rhythm_regular: true, p_waves_present: true,
    },
  },
];

const DEFAULT_ECG: ECGInput = PRESETS[0].input;

export function EcgDiagnosticStudio() {
  const { locale } = useAtlasLocale();
  const theme = useUIStore((s) => s.theme);
  const isAr = locale === "ar";
  const isLight = theme === "light";

  const [ecg, setEcg] = useState<ECGInput>(DEFAULT_ECG);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageMime, setImageMime] = useState<string>("image/png");
  const [imageName, setImageName] = useState<string>("");
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [visionError, setVisionError] = useState<string | null>(null);
  const [visionNotes, setVisionNotes] = useState<{ ar: string; en: string; provider: string } | null>(null);

  const result = useMemo(() => interpretECG(ecg), [ecg]);

  const updateNum = (key: keyof ECGInput, val: string) => {
    const parsed = val.trim() === "" ? null : Number(val);
    setEcg((prev) => ({ ...prev, [key]: Number.isNaN(parsed) ? null : parsed }));
  };

  const updateStLead = (lead: string, val: string) => {
    setEcg((prev) => {
      const nextMap = { ...(prev.st_elevation_mm || {}) };
      if (val.trim() === "") {
        delete nextMap[lead];
      } else {
        const n = Number(val);
        if (!Number.isNaN(n)) nextMap[lead] = n;
      }
      return { ...prev, st_elevation_mm: nextMap };
    });
  };

  const runVisionAnalysis = async (b64: string, mime: string) => {
    setAnalyzing(true);
    setVisionError(null);
    try {
      const res = await fetch("/api/ecg-analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageBase64: b64,
          mimeType: mime,
          age: ecg.age ?? null,
          sex: ecg.sex || null,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setVisionError(isAr ? data.error_ar || data.error : data.error || "Analysis failed");
        return;
      }
      if (data.extracted) {
        setEcg(data.extracted);
      }
      setVisionNotes({
        ar: data.vision_notes_ar || "",
        en: data.vision_notes_en || "",
        provider: data.provider || "Vision AI",
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Network error";
      setVisionError(isAr ? `خطأ في الاتصال بمحرك الذكاء الاصطناعي: ${msg}` : `Vision API connection error: ${msg}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageName(file.name);
    setImageMime(file.type || "image/png");
    setVisionError(null);
    setVisionNotes(null);

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      setImagePreview(dataUrl);
      setImageBase64(dataUrl);
      void runVisionAnalysis(dataUrl, file.type || "image/png");
    };
    reader.readAsDataURL(file);
  };

  const cardClass = isLight
    ? "bg-white/92 border-slate-200 text-slate-900 shadow-sm"
    : "bg-[#0d1528]/90 border-white/10 text-slate-100";

  const inputClass = isLight
    ? "w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-slate-900 focus:border-rose-500 focus:outline-none"
    : "w-full rounded-lg border border-white/15 bg-slate-900/90 px-2.5 py-1.5 text-xs text-white focus:border-rose-400 focus:outline-none";

  const urgencyBadge = (u: string) => {
    if (u === "CRITICAL") return "bg-red-600 text-white border-red-400 shadow-red-600/30 shadow-md";
    if (u === "URGENT") return "bg-amber-500 text-slate-950 border-amber-300";
    if (u === "ABNORMAL") return "bg-sky-500/20 text-sky-300 border-sky-400/40";
    return "bg-emerald-500/20 text-emerald-300 border-emerald-400/40";
  };

  return (
    <div
      dir={isAr ? "rtl" : "ltr"}
      className={`h-full w-full overflow-y-auto px-4 md:px-6 py-4 ${isLight ? "bg-slate-50/90 text-slate-900" : "bg-[#060a14] text-slate-100"}`}
    >
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Top Calibration & Presets Bar */}
        <div className={`rounded-2xl border p-4 ${cardClass}`}>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-rose-500" />
              <div>
                <h2 className="font-black text-sm sm:text-base">
                  {isAr ? "محرك تشخيص رسم القلب الذكي (Vision AI + Rules Engine)" : "AI ECG Vision & Clinical Rules Engine"}
                </h2>
                <p className={`text-[11px] ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                  {isAr
                    ? `المعايرة القياسية: ${ECG_UNITS.paper_speed_mm_per_sec} mm/s • المربع الصغير = ${ECG_UNITS.small_box_ms} ms • المربع الكبير = ${ECG_UNITS.large_box_ms} ms • الفولتية ${ECG_UNITS.voltage_calibration_mm_per_mv} mm/mV`
                    : `Calibration: ${ECG_UNITS.paper_speed_mm_per_sec} mm/s • 1 small box = ${ECG_UNITS.small_box_ms} ms • 1 large box = ${ECG_UNITS.large_box_ms} ms • ${ECG_UNITS.voltage_calibration_mm_per_mv} mm/mV`}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => {
                setEcg({ age: 40, sex: "male", hr_bpm: 75, pr_ms: 160, qrs_ms: 88, qt_ms: 380, axis_deg: 45, rhythm_regular: true, p_waves_present: true });
                setVisionNotes(null);
                setVisionError(null);
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 text-xs font-bold transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              {isAr ? "إعادة ضبط" : "Reset"}
            </button>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => {
                  setEcg(p.input);
                  setVisionNotes(null);
                }}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-bold border transition cursor-pointer ${
                  isLight
                    ? "bg-slate-100 hover:bg-rose-50 text-slate-700 hover:text-rose-700 border-slate-200"
                    : "bg-white/5 hover:bg-rose-500/20 text-slate-200 hover:text-rose-200 border-white/10"
                }`}
              >
                {isAr ? p.labelAr : p.labelEn}
              </button>
            ))}
          </div>
        </div>

        {/* Main 2-Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* LEFT: Vision AI Uploader + Editable Measurements */}
          <div className="lg:col-span-5 space-y-4">
            <div className={`rounded-2xl border p-4 space-y-2.5 ${cardClass}`}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold flex items-center gap-1.5">
                  <Upload className="w-4 h-4 text-rose-400" />
                  {isAr ? "ارفع صورة الـ ECG للتشخيص التلقائي بالـ AI" : "Upload ECG Image for Auto AI Diagnosis"}
                </span>
                {imageName && <span className="text-[10px] opacity-70 truncate max-w-[140px]">{imageName}</span>}
              </div>

              <label className="block border border-dashed border-rose-400/45 rounded-xl p-3 text-center cursor-pointer hover:bg-rose-500/5 transition">
                <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                {imagePreview ? (
                  <img src={imagePreview} alt="Uploaded ECG" className="w-full max-h-52 object-contain rounded-lg mx-auto" />
                ) : (
                  <div className="py-4 text-xs opacity-80 space-y-1">
                    <div className="font-bold text-rose-400">
                      {isAr ? "اضغط لاختيار صورة رسم قلب (PNG / JPG)" : "Click to upload a 12-lead ECG image (PNG / JPG)"}
                    </div>
                    <div className="text-[10.5px] opacity-70">
                      {isAr
                        ? "سيقوم الذكاء الاصطناعي باستخراج الأرقام وارتفاعات ST وتمريرها تلقائياً لمحرك القواعد"
                        : "Vision AI extracts intervals & ST shifts automatically and runs ecg_criteria.json rules"}
                    </div>
                  </div>
                )}
              </label>

              {imageBase64 && (
                <button
                  type="button"
                  disabled={analyzing}
                  onClick={() => void runVisionAnalysis(imageBase64, imageMime)}
                  className="w-full py-2 rounded-xl bg-gradient-to-r from-[#9f1239] to-[#e11d48] hover:from-[#be123c] hover:to-[#f43f5e] disabled:opacity-60 text-white font-extrabold text-xs inline-flex items-center justify-center gap-2 transition cursor-pointer"
                >
                  {analyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <ScanLine className="w-4 h-4" />}
                  {analyzing
                    ? isAr
                      ? "جاري قراءة الصورة واستخراج القياسات..."
                      : "Reading ECG tracing & extracting criteria..."
                    : isAr
                    ? "إعادة تحليل الصورة بالذكاء الاصطناعي"
                    : "Re-analyze ECG with Vision AI"}
                </button>
              )}

              {visionError && (
                <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-2.5 text-[11px] text-amber-300">
                  {visionError}
                </div>
              )}

              {visionNotes && (visionNotes.ar || visionNotes.en) && (
                <div className="rounded-xl border border-cyan-500/35 bg-cyan-500/10 p-2.5 text-[11px] space-y-1">
                  <div className="font-extrabold text-cyan-300 flex items-center justify-between">
                    <span>{isAr ? "ملاحظات القراءة البصرية (Vision AI):" : "Vision AI Observations:"}</span>
                    <span className="text-[9.5px] font-mono opacity-75">{visionNotes.provider}</span>
                  </div>
                  <p className="leading-relaxed opacity-90">{isAr ? visionNotes.ar || visionNotes.en : visionNotes.en || visionNotes.ar}</p>
                </div>
              )}
            </div>

            {/* Core Measurements */}
            <div className={`rounded-2xl border p-4 space-y-3 ${cardClass}`}>
              <div className="text-xs font-extrabold">{isAr ? "البيانات والقياسات الأساسية (قابلة للتعديل اليدوي)" : "Demographics & Core Intervals (Editable)"}</div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="text-[10px] opacity-75">{isAr ? "السن" : "Age"}</label>
                  <input type="number" value={ecg.age ?? ""} onChange={(e) => updateNum("age", e.target.value)} className={inputClass} />
                </div>
                <div>
                  <label className="text-[10px] opacity-75">{isAr ? "الجنس" : "Sex"}</label>
                  <select
                    value={ecg.sex ?? ""}
                    onChange={(e) => setEcg((p) => ({ ...p, sex: e.target.value as "male" | "female" | "" }))}
                    className={inputClass}
                  >
                    <option value="male">{isAr ? "ذكر (Male)" : "Male"}</option>
                    <option value="female">{isAr ? "أنثى (Female)" : "Female"}</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] opacity-75">HR (bpm)</label>
                  <input type="number" value={ecg.hr_bpm ?? ""} onChange={(e) => updateNum("hr_bpm", e.target.value)} className={inputClass} />
                </div>
                <div>
                  <label className="text-[10px] opacity-75">PR (ms)</label>
                  <input type="number" value={ecg.pr_ms ?? ""} onChange={(e) => updateNum("pr_ms", e.target.value)} className={inputClass} />
                </div>
                <div>
                  <label className="text-[10px] opacity-75">QRS (ms)</label>
                  <input type="number" value={ecg.qrs_ms ?? ""} onChange={(e) => updateNum("qrs_ms", e.target.value)} className={inputClass} />
                </div>
                <div>
                  <label className="text-[10px] opacity-75">QT (ms)</label>
                  <input type="number" value={ecg.qt_ms ?? ""} onChange={(e) => updateNum("qt_ms", e.target.value)} className={inputClass} />
                </div>
                <div>
                  <label className="text-[10px] opacity-75">{isAr ? "المحور Axis°" : "Axis (°)"}</label>
                  <input type="number" value={ecg.axis_deg ?? ""} onChange={(e) => updateNum("axis_deg", e.target.value)} className={inputClass} />
                </div>
                <div>
                  <label className="text-[10px] opacity-75">AV Block</label>
                  <select
                    value={ecg.av_block_type ?? ""}
                    onChange={(e) => setEcg((p) => ({ ...p, av_block_type: e.target.value }))}
                    className={inputClass}
                  >
                    <option value="">None</option>
                    <option value="mobitz_1">Mobitz I</option>
                    <option value="mobitz_2">Mobitz II</option>
                    <option value="2:1">2:1 Block</option>
                    <option value="3rd">3rd (Complete)</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] opacity-75">BBB</label>
                  <select
                    value={ecg.qrs_morphology ?? ""}
                    onChange={(e) => setEcg((p) => ({ ...p, qrs_morphology: e.target.value }))}
                    className={inputClass}
                  >
                    <option value="">None</option>
                    <option value="rbbb">RBBB</option>
                    <option value="lbbb">LBBB</option>
                  </select>
                </div>
              </div>

              <div className="pt-2 border-t border-white/10">
                <div className="text-[11px] font-bold mb-2">{isAr ? "علامات النظم والشكل (Clinical Flags)" : "Rhythm & Morphology Flags"}</div>
                <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                  {[
                    ["rhythm_regular", isAr ? "إيقاع منتظم (Regular)" : "Regular Rhythm"],
                    ["p_waves_present", isAr ? "وجود موجات P" : "P Waves Present"],
                    ["sawtooth_flutter_waves", "Flutter (Sawtooth)"],
                    ["delta_wave", "WPW Delta Wave"],
                    ["av_dissociation", "AV Dissociation"],
                    ["peaked_t_waves", "Peaked T (HyperK)"],
                    ["prominent_u_waves", "Prominent U (HypoK)"],
                    ["pr_depression", "PR Depression"],
                    ["s1q3t3", "PE (S1Q3T3)"],
                    ["coved_ste_v1_v2", "Brugada Coved V1-V2"],
                    ["osborn_j_waves", "Osborn (J) Waves"],
                    ["electrical_alternans", "Electrical Alternans"],
                  ].map(([k, label]) => (
                    <label key={k} className="flex items-center gap-1.5 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={Boolean(ecg[k as keyof ECGInput])}
                        onChange={(e) => setEcg((prev) => ({ ...prev, [k]: e.target.checked }))}
                        className="accent-rose-500"
                      />
                      <span>{label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="pt-2 border-t border-white/10">
                <div className="text-[11px] font-bold mb-2">
                  {isAr ? "ارتفاع / انخفاض ST بالمليمتر (+ ارتفاع / - انخفاض)" : "ST Segment Shift per Lead (mm, +STE / -STD)"}
                </div>
                <div className="grid grid-cols-4 gap-1.5">
                  {LEADS.map((lead) => (
                    <div key={lead}>
                      <label className="text-[10px] font-mono opacity-75">{lead}</label>
                      <input
                        type="number"
                        step="0.5"
                        placeholder="0"
                        value={ecg.st_elevation_mm?.[lead] ?? ""}
                        onChange={(e) => updateStLead(lead, e.target.value)}
                        className={inputClass}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: Urgent-First Diagnostic Report */}
          <div className="lg:col-span-7 space-y-4">
            {result.has_urgent_findings ? (
              <div className="rounded-2xl border-2 border-red-500/70 bg-red-950/50 p-4 text-red-100 shadow-lg shadow-red-900/30">
                <div className="flex items-center gap-2 font-black text-sm sm:text-base text-red-300">
                  <ShieldAlert className="w-5 h-5 animate-pulse" />
                  {isAr
                    ? `تنبيه طوارئ: تم اكتشاف ${result.urgent_count} حالة حرجة/عاجلة تستدعي تدخلاً فورياً!`
                    : `URGENT ALERT: ${result.urgent_count} critical/urgent finding(s) detected — immediate action required!`}
                </div>
              </div>
            ) : (
              <div className="rounded-2xl border border-emerald-500/40 bg-emerald-950/30 p-3 text-emerald-200 flex items-center gap-2 text-xs font-bold">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                {isAr ? "لا توجد علامات طوارئ حرجة مكتشفة حالياً" : "No immediate life-threatening emergencies triggered"}
              </div>
            )}

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
                <span className="font-bold text-rose-400">QTc Formulas:</span>
                <span>Bazett: <b>{result.measurements.qtc_bazett_ms} ms</b></span>
                <span>Fridericia: <b>{result.measurements.qtc_fridericia_ms} ms</b></span>
                <span>Hodges: <b>{result.measurements.qtc_hodges_ms} ms</b></span>
                {result.measurements.axis_classification && <span>Axis: <b>{result.measurements.axis_classification}</b></span>}
              </div>
            )}

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
                        {isAr ? "الأدلة المستخلصة من القياسات:" : "Diagnostic Evidence:"}
                      </div>
                      <ul className="list-disc list-inside space-y-0.5">
                        {f.evidence.map((ev, i) => (
                          <li key={i}>{ev}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="text-xs leading-relaxed">
                    <span className="font-extrabold text-cyan-400">{isAr ? "الشرح الإكلينيكي: " : "Clinical Explanation: "}</span>
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
          </div>
        </div>
      </div>
    </div>
  );
}
