"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  ArrowLeft,
  Heart,
  HeartPulse,
  Home,
  Moon,
  Sparkles,
  Stethoscope,
  Sun,
  Wind,
} from "lucide-react";
import { SoundsSpecimenCanvas } from "@/features/viewer/SoundsSpecimenCanvas";
import { HeartControls } from "@/features/viewer/HeartControls";
import { BreathControls } from "@/features/viewer/BreathControls";
import { EcgDiagnosticStudio } from "@/features/ecg/EcgDiagnosticStudio";
import { useHeartStore } from "@/stores/heartStore";
import { useBreathStore } from "@/stores/breathStore";
import { useUIStore } from "@/stores/uiStore";
import { primeHeartSound } from "@/features/viewer/heartSound";
import { primeBreathSound } from "@/features/viewer/breathing";
import { useAtlasLocale } from "@/lib/atlasI18n";

const MAIN_URL = process.env.NEXT_PUBLIC_MAIN_URL || "https://mrcs-bank.vercel.app";

export default function SoundsLabPage() {
  const [mounted, setMounted] = useState(false);
  const [view, setView] = useState<"landing" | "workspace">("landing");
  const [mode, setMode] = useState<"heart" | "breath" | "ecg-ai">("heart");

  const heartEnabled = useHeartStore((s) => s.enabled);
  const toggleHeart = useHeartStore((s) => s.toggle);
  const breathEnabled = useBreathStore((s) => s.enabled);
  const toggleBreath = useBreathStore((s) => s.toggle);
  const theme = useUIStore((s) => s.theme);
  const toggleTheme = useUIStore((s) => s.toggleTheme);
  const { locale, setLocale } = useAtlasLocale();
  const isAr = locale === "ar";
  const isLight = theme === "light";

  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("codeblue_theme", theme);
      localStorage.removeItem("sky:mode-override");
      document.documentElement.dataset.theme = theme;
      document.documentElement.classList.toggle("dark", theme === "dark");
      document.documentElement.classList.toggle("light", theme === "light");
    } catch {}
  }, [theme]);

  const handleToggleTheme = () => {
    toggleTheme();
    const next = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("codeblue_theme", next);
      document.documentElement.dataset.theme = next;
      document.documentElement.classList.toggle("dark", next === "dark");
    } catch {}
  };

  const launchMode = (target: "heart" | "breath" | "ecg-ai") => {
    if (target === "heart") {
      primeHeartSound();
      if (breathEnabled) toggleBreath();
      if (!heartEnabled) toggleHeart();
    } else if (target === "breath") {
      primeBreathSound();
      if (heartEnabled) toggleHeart();
      if (!breathEnabled) toggleBreath();
    } else {
      if (heartEnabled) toggleHeart();
      if (breathEnabled) toggleBreath();
    }
    setMode(target);
    setView("workspace");
  };

  if (!mounted) {
    return (
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 9999,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "14px",
          background: "radial-gradient(circle at 50% 35%, #0f1b38 0%, #060913 75%)",
          color: "#e2e8f0",
          fontFamily: "system-ui, -apple-system, sans-serif",
        }}
      >
        <img
          src="/icon.png"
          alt="Code Blue ECG & Sounds"
          style={{ width: 68, height: 68, borderRadius: 16, boxShadow: "0 0 30px rgba(16,185,129,0.35)" }}
        />
        <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.04em" }}>CODE BLUE ECG &amp; SOUNDS</div>
      </div>
    );
  }

  if (view === "landing") {
    const petalBase = isLight
      ? "bg-white/90 border-slate-200/90 text-slate-800 shadow-[0_14px_30px_rgba(15,23,42,0.14),inset_0_1px_0_rgba(255,255,255,0.9)] hover:bg-white"
      : "bg-[#0e172c]/82 border-white/14 text-slate-100 shadow-[0_16px_34px_rgba(0,0,0,0.55),inset_0_1px_0_rgba(255,255,255,0.12)] hover:bg-[#13203d]/92";

    return (
      <div
        dir={isAr ? "rtl" : "ltr"}
        className={`min-h-screen lg:h-screen lg:overflow-hidden w-full bg-transparent flex flex-col ${
          isLight ? "text-slate-900" : "text-slate-100"
        }`}
      >
        <header
          className={`shrink-0 h-14 px-5 md:px-10 border-b backdrop-blur-xl flex items-center justify-between gap-4 z-40 ${
            isLight ? "border-slate-200/80 bg-white/75" : "border-white/10 bg-[#070b16]/70"
          }`}
        >
          <div className="flex items-center gap-3">
            <img
              src="/icon.png"
              alt="Code Blue"
              className="w-8 h-8 rounded-xl border border-rose-500/40 shadow-lg shadow-rose-900/30"
            />
            <div className="leading-tight">
              <div className={`font-extrabold text-sm md:text-base tracking-tight ${isLight ? "text-slate-900" : "text-white"}`}>
                Code Blue
              </div>
              <div className={`text-[10.5px] font-mono ${isLight ? "text-[#9f1239]" : "text-[#f47292]"}`}>
                codeblue-ecg&amp;sounds
              </div>
            </div>
          </div>

          <div
            aria-hidden="true"
            className={`hidden md:flex items-center gap-5 text-xs font-semibold select-none cursor-default ${
              isLight ? "text-slate-500" : "text-slate-300/80"
            }`}
          >
            <span>{isAr ? "أصوات القلب" : "Heart Sounds"}</span>
            <span className="opacity-40">•</span>
            <span>{isAr ? "تشخيص ECG الذكي" : "AI ECG Engine"}</span>
            <span className="opacity-40">•</span>
            <span>{isAr ? "أصوات التنفس" : "Breath Sounds"}</span>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <a
              href={MAIN_URL}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border transition ${
                isLight
                  ? "bg-slate-100/80 hover:bg-slate-200/80 text-slate-800 border-slate-300/70"
                  : "bg-white/5 hover:bg-white/10 text-slate-200 border-white/10"
              }`}
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>{isAr ? "بنك الأسئلة" : "Main Bank"}</span>
            </a>
            <button
              type="button"
              onClick={() => setLocale(isAr ? "en" : "ar")}
              className={`px-3 py-1.5 rounded-xl border font-bold transition ${
                isLight
                  ? "bg-rose-50 hover:bg-rose-100 text-[#9f1239] border-rose-200"
                  : "bg-rose-950/40 hover:bg-rose-900/50 text-rose-200 border-rose-400/25"
              }`}
            >
              {isAr ? "EN" : "AR"}
            </button>
            <button
              type="button"
              onClick={handleToggleTheme}
              className={`p-2 rounded-xl border transition ${
                isLight
                  ? "bg-slate-100/80 hover:bg-slate-200/80 text-slate-800 border-slate-300/70"
                  : "bg-white/5 hover:bg-white/10 text-slate-200 border-white/10"
              }`}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </header>

        <main className="flex-1 min-h-0 w-full max-w-7xl mx-auto px-5 md:px-10 py-4 lg:py-0 flex items-center">
          <div className="w-full grid grid-cols-1 lg:grid-cols-12 items-center gap-6 lg:gap-8">
            <div className="lg:col-span-6 flex flex-col gap-4">
              <div className="space-y-2.5">
                <h1 className={`text-2xl sm:text-3xl xl:text-4xl font-black leading-[1.18] ${isLight ? "text-slate-900" : "text-white"}`}>
                  {isAr ? "منصة أصوات القلب والتنفس والتشخيص الذكي" : "Interactive Heart, Breath Sounds & Live ECG"}
                </h1>
                <p className={`text-sm sm:text-base leading-relaxed max-w-xl ${isLight ? "text-slate-600" : "text-slate-300"}`}>
                  {isAr ? (
                    <>
                      محاكي تفاعلي ثلاثي الأبعاد لأصوات القلب واللغط القلبي ومحرك تشخيص الـ ECG{" "}
                      <span className={isLight ? "text-[#9f1239] font-extrabold" : "text-[#fb7185] font-extrabold"}>
                        وأصوات الصدر والتنفس السريرية
                      </span>
                    </>
                  ) : (
                    <>
                      Interactive 3D simulator for heart murmurs, clinical ECG diagnostic rules engine,{" "}
                      <span className={isLight ? "text-[#9f1239] font-extrabold" : "text-[#fb7185] font-extrabold"}>
                        live ECG strips &amp; clinical breath auscultation
                      </span>
                    </>
                  )}
                </p>
                <div className="pt-1 flex flex-wrap gap-2.5">
                  <button
                    type="button"
                    onClick={() => launchMode("heart")}
                    className="px-6 py-3 rounded-2xl bg-gradient-to-r from-[#881337] via-[#be123c] to-[#e11d48] hover:from-[#9f1239] hover:via-[#e11d48] hover:to-[#f43f5e] text-white font-black text-sm sm:text-base shadow-[0_0_30px_rgba(190,18,60,0.45)] transition transform hover:scale-[1.03] active:scale-[0.99] cursor-pointer"
                  >
                    {isAr ? "ابدأ المحاكاة الآن" : "Start Simulation Now"}
                  </button>
                  <button
                    type="button"
                    onClick={() => launchMode("ecg-ai")}
                    className={`px-5 py-3 rounded-2xl border font-extrabold text-sm sm:text-base inline-flex items-center gap-2 transition transform hover:scale-[1.03] cursor-pointer ${
                      isLight
                        ? "bg-white border-rose-300 text-[#9f1239] hover:bg-rose-50"
                        : "bg-rose-950/50 border-rose-400/40 text-rose-200 hover:bg-rose-900/60"
                    }`}
                  >
                    <Sparkles className="w-4 h-4" />
                    {isAr ? "محرك تشخيص الـ ECG" : "ECG Diagnostic Engine"}
                  </button>
                </div>
              </div>

              <div className="pt-1 space-y-2.5">
                <div className={`text-xs font-extrabold uppercase tracking-wider ${isLight ? "text-[#9f1239]" : "text-[#fb7185]"}`}>
                  {isAr ? "ماذا تقدم المنصة؟" : "What Does the Platform Offer?"}
                </div>

                <div dir="ltr" className="grid grid-cols-2 gap-2.5 max-w-[440px]">
                  <div
                    dir={isAr ? "rtl" : "ltr"}
                    className={`group select-none cursor-default rounded-tl-[64px] rounded-tr-xl rounded-bl-xl rounded-br-md border backdrop-blur-xl px-4 py-3.5 pl-6 flex flex-col items-center justify-center text-center transition-all duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:border-rose-400/60 hover:shadow-[0_18px_36px_rgba(244,63,94,0.22)] ${petalBase}`}
                  >
                    <div className="w-9 h-9 mb-1.5 rounded-2xl bg-rose-500/15 border border-rose-400/35 flex items-center justify-center text-rose-400 group-hover:scale-110 transition-transform duration-300">
                      <HeartPulse className="w-5 h-5" />
                    </div>
                    <div className="font-extrabold text-xs sm:text-[13px] leading-snug">{isAr ? "أصوات ولغط القلب" : "Heart Murmurs"}</div>
                    <div className={`text-[10.5px] mt-0.5 ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                      {isAr ? "S1/S2 وS3/S4 واضطرابات الصمامات" : "S1–S4 & valve murmurs"}
                    </div>
                  </div>

                  <div
                    dir={isAr ? "rtl" : "ltr"}
                    className={`group select-none cursor-default rounded-tr-[64px] rounded-tl-xl rounded-br-xl rounded-bl-md border backdrop-blur-xl px-4 py-3.5 pr-6 flex flex-col items-center justify-center text-center transition-all duration-300 hover:-translate-y-1 hover:scale-[1.03] hover:border-cyan-400/60 hover:shadow-[0_18px_36px_rgba(34,211,238,0.24)] ${petalBase}`}
                  >
                    <div className="w-9 h-9 mb-1.5 rounded-2xl bg-cyan-500/15 border border-cyan-400/35 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform duration-300">
                      <Activity className="w-5 h-5" />
                    </div>
                    <div className="font-extrabold text-xs sm:text-[13px] leading-snug">{isAr ? "محرك تشخيص ECG" : "ECG Rules Engine"}</div>
                    <div className={`text-[10.5px] mt-0.5 ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                      {isAr ? "STEMI وQTc وAF وVT وAV Blocks" : "STEMI, QTc, AF, VT & AV Blocks"}
                    </div>
                  </div>

                  <div
                    dir={isAr ? "rtl" : "ltr"}
                    className={`group select-none cursor-default rounded-bl-[64px] rounded-tl-xl rounded-br-xl rounded-tr-md border backdrop-blur-xl px-4 py-3.5 pl-6 flex flex-col items-center justify-center text-center transition-all duration-300 hover:translate-y-1 hover:scale-[1.03] hover:border-emerald-400/60 hover:shadow-[0_18px_36px_rgba(16,185,129,0.22)] ${petalBase}`}
                  >
                    <div className="w-9 h-9 mb-1.5 rounded-2xl bg-emerald-500/15 border border-emerald-400/35 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform duration-300">
                      <Wind className="w-5 h-5" />
                    </div>
                    <div className="font-extrabold text-xs sm:text-[13px] leading-snug">{isAr ? "أصوات الصدر والتنفس" : "Breath Sounds"}</div>
                    <div className={`text-[10.5px] mt-0.5 ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                      {isAr ? "Wheezes وCrackles وStridor" : "Wheezes, Crackles & Stridor"}
                    </div>
                  </div>

                  <div
                    dir={isAr ? "rtl" : "ltr"}
                    className={`group select-none cursor-default rounded-br-[64px] rounded-tr-xl rounded-bl-xl rounded-tl-md border backdrop-blur-xl px-4 py-3.5 pr-6 flex flex-col items-center justify-center text-center transition-all duration-300 hover:translate-y-1 hover:scale-[1.03] hover:border-violet-400/60 hover:shadow-[0_18px_36px_rgba(139,92,246,0.22)] ${petalBase}`}
                  >
                    <div className="w-9 h-9 mb-1.5 rounded-2xl bg-violet-500/15 border border-violet-400/35 flex items-center justify-center text-violet-400 group-hover:scale-110 transition-transform duration-300">
                      <Stethoscope className="w-5 h-5" />
                    </div>
                    <div className="font-extrabold text-xs sm:text-[13px] leading-snug">{isAr ? "حركة القلب 4 حجرات" : "4-Chamber 3D Sync"}</div>
                    <div className={`text-[10.5px] mt-0.5 ${isLight ? "text-slate-500" : "text-slate-400"}`}>
                      {isAr ? "تزامن إيقاعي مع بطاقة سريرية" : "Rhythmic chambers & cards"}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="lg:col-span-6 flex items-center justify-center">
              <img
                src="/hero-sounds.png?v=3"
                alt="Heart & ECG Simulator Preview"
                draggable={false}
                className={`w-full max-w-[540px] max-h-[68vh] h-auto object-contain select-none pointer-events-none ${
                  isLight ? "drop-shadow-[0_0_28px_rgba(255,255,255,0.95)]" : "drop-shadow-[0_18px_40px_rgba(190,18,60,0.22)]"
                }`}
              />
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-transparent text-slate-100 flex flex-col overflow-hidden">
      <header className="h-13 shrink-0 px-4 border-b border-slate-800/80 bg-[#070b16]/85 backdrop-blur-md flex items-center justify-between gap-3 z-30">
        <div className="flex items-center gap-3 overflow-x-auto">
          <button
            type="button"
            onClick={() => setView("landing")}
            className="w-8 h-8 shrink-0 rounded-lg bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-400 hover:bg-emerald-500/25 transition"
            title={isAr ? "العودة للرئيسية" : "Back to Home"}
          >
            <Home className="w-4 h-4" />
          </button>
          <span className="font-bold text-sm tracking-tight text-white shrink-0">
            Code Blue <span className="text-emerald-400">Sounds &amp; ECG</span>
          </span>

          <div className="ml-2 flex items-center p-1 rounded-xl bg-slate-900/90 border border-slate-800 shrink-0">
            <button
              type="button"
              onClick={() => launchMode("heart")}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                mode === "heart" ? "bg-rose-500/20 text-rose-300 border border-rose-500/40" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Heart className="w-3.5 h-3.5" />
              <span>{isAr ? "أصوات القلب والـ ECG" : "Heart & ECG"}</span>
            </button>
            <button
              type="button"
              onClick={() => launchMode("breath")}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                mode === "breath" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Wind className="w-3.5 h-3.5" />
              <span>{isAr ? "أصوات الصدر والتنفس" : "Chest & Breath"}</span>
            </button>
            <button
              type="button"
              onClick={() => launchMode("ecg-ai")}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                mode === "ecg-ai" ? "bg-amber-500/20 text-amber-300 border border-amber-500/40" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>{isAr ? "تشخيص الـ ECG الذكي" : "AI ECG Diagnosis"}</span>
            </button>
          </div>
        </div>

        <nav className="flex items-center gap-1.5 text-xs shrink-0">
          <a
            href={MAIN_URL}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/70 hover:bg-slate-700/80 text-slate-200 border border-slate-700/60 transition"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>{isAr ? "بنك الأسئلة" : "Main Bank"}</span>
          </a>
          <button
            type="button"
            onClick={() => setLocale(isAr ? "en" : "ar")}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800/70 hover:bg-slate-700/80 text-slate-200 border border-slate-700/60 font-semibold"
          >
            {isAr ? "EN" : "AR"}
          </button>
        </nav>
      </header>

      <main className="relative flex-1 w-full overflow-hidden">
        {mode === "ecg-ai" ? (
          <EcgDiagnosticStudio />
        ) : (
          <>
            <SoundsSpecimenCanvas mode={mode} />
            <div className="pointer-events-none fixed inset-x-0 top-16 bottom-3 z-20 flex items-start justify-start px-3">
              <div className="pointer-events-none w-full max-w-[340px]">
                {mode === "heart" ? <HeartControls /> : <BreathControls />}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
