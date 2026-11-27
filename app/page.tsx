"use client";

import React, { useEffect, useState } from "react";
import {
  ArrowLeft,
  Heart,
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
import { NeuralDiagnosticTree } from "@/components/NeuralDiagnosticTree";
import { useHeartStore } from "@/stores/heartStore";
import { useBreathStore } from "@/stores/breathStore";
import { useUIStore } from "@/stores/uiStore";
import { primeHeartSound } from "@/features/viewer/heartSound";
import { primeBreathSound } from "@/features/viewer/breathing";
import { useAtlasLocale } from "@/lib/atlasI18n";

const MAIN_URL = process.env.NEXT_PUBLIC_MAIN_URL || "https://mrcs-bank.vercel.app";

export default function SoundsLabPage() {
  const [mounted, setMounted] = useState(false);
  const [isDesktop, setIsDesktop] = useState(true);
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
    let alive = true;
    const finish = () => { if (alive) setMounted(true); };
    const preloadImg = (src: string) =>
      new Promise<void>((resolve) => {
        const img = new window.Image();
        img.src = src;
        if (typeof img.decode === "function") img.decode().then(resolve).catch(resolve);
        else { img.onload = () => resolve(); img.onerror = () => resolve(); }
      });
    const timer = window.setTimeout(finish, 1100);
    Promise.all([preloadImg("/hero-ecg-monitor.png"), preloadImg("/neural-tree-burgundy.png")]).then(() => {
      window.clearTimeout(timer);
      finish();
    });
    // Background warmup for 3D GLB models & YouTube audio host
    try {
      ["https://www.youtube.com", "https://i.ytimg.com"].forEach((href) => {
        if (!document.querySelector(`link[rel="preconnect"][href="${href}"]`)) {
          const lk = document.createElement("link");
          lk.rel = "preconnect"; lk.href = href; lk.crossOrigin = "anonymous";
          document.head.appendChild(lk);
        }
      });
      const mq = window.matchMedia("(min-width: 768px)");
      setIsDesktop(mq.matches);
      if (mq.matches) {
        fetch("/anatomy/heart.glb", { cache: "force-cache" }).catch(() => {});
        fetch("/anatomy/lungs.glb", { cache: "force-cache" }).catch(() => {});
      }
    } catch {}
    return () => { alive = false; window.clearTimeout(timer); };
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
          style={{ width: 68, height: 68, borderRadius: 16, boxShadow: "0 0 30px rgba(159,18,57,0.4)" }}
        />
        <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.04em" }}>CODE BLUE ECG &amp; SOUNDS</div>
      </div>
    );
  }

  if (view === "landing") {
    return (
      <div
        dir={isAr ? "rtl" : "ltr"}
        className={`h-screen w-full overflow-y-auto overflow-x-hidden bg-transparent flex flex-col ${
          isLight ? "text-slate-900" : "text-slate-100"
        }`}
      >
        <header className="shrink-0 mx-3 sm:mx-6 mt-2.5 cb-glass-pill min-h-[54px] px-4 sm:px-6 py-1.5 flex items-center justify-between gap-4 z-40">
          <div className="flex items-center gap-2.5">
            <img
              src="/icon.png"
              alt="Code Blue"
              className="w-7 h-7 rounded-xl border border-rose-500/45 shadow-lg shadow-rose-950/40"
            />
            <div className="leading-tight">
              <div className={`font-extrabold text-sm tracking-tight ${isLight ? "text-slate-900" : "text-white"}`}>
                Code Blue
              </div>
              <div className={`text-[10px] font-mono ${isLight ? "text-[#9f1239]" : "text-[#fb7185]"}`}>
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
            <span>{isAr ? "أصوات التنفس" : "Breath Sounds"}</span>
            <span className="opacity-40">•</span>
            <span>{isAr ? "محرك تشخيص الـ ECG" : "ECG Diagnostic Engine"}</span>
          </div>

          <div className="flex items-center gap-1.5 text-xs">
            <a
              href={MAIN_URL}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-xl border transition ${
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
              className={`px-2.5 py-1 rounded-xl border font-bold transition cursor-pointer ${
                isLight
                  ? "bg-rose-50/90 hover:bg-rose-100 text-[#9f1239] border-rose-300/80"
                  : "bg-rose-950/45 hover:bg-rose-900/55 text-rose-100 border-rose-400/35"
              }`}
            >
              {isAr ? "EN" : "AR"}
            </button>
            <button
              type="button"
              onClick={handleToggleTheme}
              className={`p-1.5 rounded-xl border transition cursor-pointer ${
                isLight
                  ? "bg-slate-100/80 hover:bg-slate-200/80 text-slate-800 border-slate-300/70"
                  : "bg-white/5 hover:bg-white/10 text-slate-200 border-white/10"
              }`}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
            </button>
          </div>
        </header>

        <main className="flex-1 w-full max-w-[1440px] mx-auto px-4 md:px-8 lg:px-10 py-3 flex items-center">
          <div className="w-full grid grid-cols-1 lg:grid-cols-12 items-center gap-4 lg:gap-6 xl:gap-8">
            {/* Start Column: Entry / Launch Controls + Unmodified ECG Monitor Directly Below */}
            <div className="lg:col-span-4 flex flex-col gap-3">
              <div className="space-y-1.5">
                <h1 className={`text-xl sm:text-2xl xl:text-[28px] font-black leading-[1.18] ${isLight ? "text-slate-900" : "text-white"}`}>
                  {isAr ? "منصة أصوات القلب والتنفس ومحرك تشخيص الـ ECG" : "Interactive Heart, Breath Sounds & ECG Engine"}
                </h1>
                <p className={`text-xs sm:text-[13px] leading-relaxed ${isLight ? "text-slate-600" : "text-slate-300"}`}>
                  {isAr ? (
                    <>
                      محاكي تفاعلي ثلاثي الأبعاد لأصوات القلب واللغط القلبي وأصوات الصدر والتنفس، مع{" "}
                      <span className={isLight ? "text-[#9f1239] font-extrabold" : "text-[#fb7185] font-extrabold"}>
                        محرك تشخيص الـ ECG الهجين وكشف الفخاخ السريرية
                      </span>
                    </>
                  ) : (
                    <>
                      Interactive 3D simulator for heart murmurs &amp; clinical breath auscultation, paired with a{" "}
                      <span className={isLight ? "text-[#9f1239] font-extrabold" : "text-[#fb7185] font-extrabold"}>
                        hybrid ECG diagnostic engine &amp; clinical pitfalls detector
                      </span>
                    </>
                  )}
                </p>
                <div className="pt-1 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => launchMode("ecg-ai")}
                    className="cb-btn-primary min-h-[42px] px-4 py-2 rounded-2xl font-black text-xs sm:text-sm inline-flex items-center gap-2 cursor-pointer"
                  >
                    <Sparkles className="w-4 h-4" />
                    {isAr ? "محرك تشخيص الـ ECG" : "ECG Diagnostic Engine"}
                  </button>
                  <button
                    type="button"
                    onClick={() => launchMode("heart")}
                    className={`min-h-[42px] px-4 py-2 rounded-2xl border font-extrabold text-xs sm:text-sm inline-flex items-center gap-2 transition transform hover:scale-[1.02] cursor-pointer ${
                      isLight
                        ? "cb-glass-nested border-rose-400/60 text-[#9f1239] hover:bg-rose-50/90"
                        : "cb-glass-nested border-rose-400/45 text-rose-100 hover:bg-rose-950/45"
                    }`}
                  >
                    <Stethoscope className="w-4 h-4" />
                    {isAr ? "أصوات القلب والتنفس" : "Heart & Breath Sounds"}
                  </button>
                </div>
              </div>

              {/* Exact Unmodified ECG Monitor Directly Under Entry Section */}
              <div className="hidden lg:flex justify-start pt-1">
                <img
                  src="/hero-ecg-monitor.png"
                  alt="Code Blue ECG Diagnostic Monitor"
                  draggable={false}
                  className="w-full max-w-[310px] sm:max-w-[345px] h-auto object-contain select-none pointer-events-none"
                />
              </div>
            </div>

            {/* End Column: Custom Flowing Burgundy Neural Tree with Compact Boxes on Both Sides */}
            <div className="lg:col-span-8 flex items-center justify-center">
              <NeuralDiagnosticTree
                isAr={isAr}
                isLight={isLight}
                onLaunchEcg={() => launchMode("ecg-ai")}
              />
            </div>
          </div>
        </main>
      </div>
    );
  }

  const isEcgMode = mode === "ecg-ai";

  return (
    <div className={`h-screen w-screen bg-transparent flex flex-col overflow-hidden ${isLight ? "text-slate-900" : "text-slate-100"}`}>
      <header className="shrink-0 mx-2.5 sm:mx-5 mt-1.5 sm:mt-2 cb-glass-pill min-h-[44px] sm:min-h-[52px] px-2.5 sm:px-5 py-1 sm:py-1.5 flex items-center justify-between gap-2 z-30">
        <div className="flex items-center gap-3 overflow-x-auto">
          <button
            type="button"
            onClick={() => setView("landing")}
            className="w-7 h-7 sm:w-8 sm:h-8 shrink-0 rounded-lg bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-400 hover:bg-emerald-500/25 transition cursor-pointer"
            title={isAr ? "العودة للرئيسية" : "Back to Home"}
          >
            <Home className="w-4 h-4" />
          </button>
          <span className={`font-bold text-xs sm:text-sm tracking-tight shrink-0 ${isLight ? "text-slate-900" : "text-white"}`}>
            Code Blue <span className="text-emerald-400">{isEcgMode ? "ECG Engine" : "Sounds Lab"}</span>
          </span>

          {isEcgMode ? (
            <div className="hidden md:inline-flex ms-2 items-center gap-1.5 px-3 py-1 rounded-full bg-rose-950/50 border border-rose-400/40 text-rose-200 text-xs font-bold shrink-0">
              <Sparkles className="w-3.5 h-3.5 text-rose-400" />
              <span>{isAr ? "محرك تشخيص الـ ECG الهجين" : "ECG Diagnostic Engine"}</span>
            </div>
          ) : (
            <div className="hidden md:flex ms-2 items-center p-1 rounded-full cb-glass-nested shrink-0">
              <button
                type="button"
                onClick={() => launchMode("heart")}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition cursor-pointer ${
                  mode === "heart" ? "bg-[#9f1239]/40 text-rose-100 border border-rose-400/50" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Heart className="w-3.5 h-3.5" />
                <span>{isAr ? "أصوات القلب" : "Heart Sounds"}</span>
              </button>
              <button
                type="button"
                onClick={() => launchMode("breath")}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition cursor-pointer ${
                  mode === "breath" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Wind className="w-3.5 h-3.5" />
                <span>{isAr ? "أصوات الصدر والتنفس" : "Chest & Breath"}</span>
              </button>
            </div>
          )}
        </div>

        <nav className="flex items-center gap-1.5 text-xs shrink-0">
          {!isEcgMode && (
            <button
              type="button"
              onClick={() => launchMode(mode === "heart" ? "breath" : "heart")}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border font-bold cursor-pointer transition ${
                mode === "heart"
                  ? isLight
                    ? "bg-cyan-100 hover:bg-cyan-200 text-cyan-900 border-cyan-400"
                    : "bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-200 border-cyan-400/50"
                  : isLight
                    ? "bg-rose-100 hover:bg-rose-200 text-[#881337] border-rose-400"
                    : "bg-[#9f1239]/40 hover:bg-[#9f1239]/55 text-rose-100 border-rose-400/50"
              }`}
            >
              {mode === "heart" ? <Wind className="w-3.5 h-3.5" /> : <Heart className="w-3.5 h-3.5" />}
              <span>{mode === "heart" ? (isAr ? "أصوات التنفس" : "Breath Sounds") : (isAr ? "أصوات القلب" : "Heart Sounds")}</span>
            </button>
          )}
          <button
            type="button"
            onClick={() => setLocale(isAr ? "en" : "ar")}
            className={`px-2.5 py-1.5 rounded-lg border font-semibold cursor-pointer transition ${isLight ? "bg-white/80 hover:bg-white text-slate-800 border-slate-300/80" : "bg-slate-800/70 hover:bg-slate-700/80 text-slate-200 border-slate-700/60"}`}
          >
            {isAr ? "EN" : "AR"}
          </button>
          <button
            type="button"
            onClick={handleToggleTheme}
            className={`p-1.5 rounded-lg border transition cursor-pointer ${isLight ? "bg-white/80 hover:bg-white text-slate-800 border-slate-300/80" : "bg-slate-800/70 hover:bg-slate-700/80 text-slate-200 border-slate-700/60"}`}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </nav>
      </header>

      <main className="relative flex-1 w-full overflow-hidden">
        {isEcgMode ? (
          <EcgDiagnosticStudio />
        ) : (
          <>
            {isDesktop && <SoundsSpecimenCanvas mode={mode} />}
            <div className="pointer-events-none fixed inset-x-0 top-16 bottom-3 z-20 flex items-start justify-center md:justify-start px-3 lg:px-12 xl:px-20">
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
