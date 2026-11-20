"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  ArrowLeft,
  Heart,
  HeartPulse,
  Home,
  Moon,
  Stethoscope,
  Sun,
  Wind,
} from "lucide-react";
import { SoundsSpecimenCanvas } from "@/features/viewer/SoundsSpecimenCanvas";
import { HeartControls } from "@/features/viewer/HeartControls";
import { BreathControls } from "@/features/viewer/BreathControls";
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
  const [mode, setMode] = useState<"heart" | "breath">("heart");

  const heartEnabled = useHeartStore((s) => s.enabled);
  const toggleHeart = useHeartStore((s) => s.toggle);
  const breathEnabled = useBreathStore((s) => s.enabled);
  const toggleBreath = useBreathStore((s) => s.toggle);
  const theme = useUIStore((s) => s.theme);
  const toggleTheme = useUIStore((s) => s.toggleTheme);
  const { locale, setLocale } = useAtlasLocale();
  const isAr = locale === "ar";

  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const handleToggleTheme = () => {
    toggleTheme();
    const next = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("codeblue_theme", next);
      document.documentElement.dataset.theme = next;
      document.documentElement.classList.toggle("dark", next === "dark");
    } catch {}
  };

  const launchMode = (target: "heart" | "breath") => {
    if (target === "heart") {
      primeHeartSound();
      if (breathEnabled) toggleBreath();
      if (!heartEnabled) toggleHeart();
    } else {
      primeBreathSound();
      if (heartEnabled) toggleHeart();
      if (!breathEnabled) toggleBreath();
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
    return (
      <div
        dir={isAr ? "rtl" : "ltr"}
        className="min-h-screen w-full bg-transparent text-slate-100 flex flex-col overflow-y-auto"
      >
        {/* Top Bar */}
        <header className="sticky top-0 z-40 h-16 px-5 md:px-10 border-b border-white/10 bg-[#070b16]/70 backdrop-blur-xl flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img src="/icon.png" alt="Code Blue" className="w-9 h-9 rounded-xl border border-cyan-400/30 shadow-lg shadow-cyan-500/20" />
            <div className="leading-tight">
              <div className="font-extrabold text-base text-white tracking-tight">Code Blue</div>
              <div className="text-[11px] font-mono text-cyan-300/80">codeblue-ecg&amp;sounds</div>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-6 text-sm font-semibold text-slate-200">
            <button type="button" onClick={() => launchMode("heart")} className="hover:text-cyan-300 transition">
              {isAr ? "أصوات القلب" : "Heart Sounds"}
            </button>
            <button type="button" onClick={() => launchMode("heart")} className="hover:text-cyan-300 transition">
              {isAr ? "تخطيط ECG" : "Live ECG"}
            </button>
            <button type="button" onClick={() => launchMode("breath")} className="hover:text-cyan-300 transition">
              {isAr ? "أصوات التنفس" : "Breath Sounds"}
            </button>
          </nav>

          <div className="flex items-center gap-2 text-xs">
            <a
              href={MAIN_URL}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-slate-200 border border-white/10 transition"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>{isAr ? "بنك الأسئلة" : "Main Bank"}</span>
            </a>
            <button
              type="button"
              onClick={() => setLocale(isAr ? "en" : "ar")}
              className="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-cyan-200 border border-white/10 font-bold transition"
            >
              {isAr ? "EN" : "AR"}
            </button>
            <button
              type="button"
              onClick={handleToggleTheme}
              className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-200 border border-white/10 transition"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </header>

        {/* Hero Section */}
        <main className="flex-1 w-full max-w-7xl mx-auto px-5 md:px-10 py-8 md:py-12 flex flex-col justify-between gap-10">
          <div className="grid grid-cols-1 lg:grid-cols-12 items-center gap-8">
            <div className="lg:col-span-6 space-y-5">
              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black leading-tight text-white">
                {isAr
                  ? "منصة أصوات القلب والتنفس والتشخيص الذكي"
                  : "Interactive Heart, Breath Sounds & Live ECG Platform"}
              </h1>
              <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-xl">
                {isAr
                  ? "محاكي تفاعلي ثلاثي الأبعاد لأصوات القلب واللغط (Murmurs) وتخطيط الـ ECG المباشر وأصوات الصدر والتنفس السريرية."
                  : "Interactive 3D physiological simulator for heart murmurs, 4-chamber rhythm synchronization, live ECG strips, and clinical breath auscultation."}
              </p>
              <div className="pt-2 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => launchMode("heart")}
                  className="px-7 py-3.5 rounded-2xl bg-gradient-to-r from-cyan-400 to-teal-300 hover:from-cyan-300 hover:to-teal-200 text-slate-950 font-black text-base shadow-[0_0_32px_rgba(34,211,238,0.45)] transition transform hover:scale-[1.02]"
                >
                  {isAr ? "ابدأ المحاكاة الآن" : "Start Simulation Now"}
                </button>
                <button
                  type="button"
                  onClick={() => launchMode("breath")}
                  className="px-5 py-3.5 rounded-2xl bg-white/5 hover:bg-white/10 text-cyan-200 border border-cyan-400/30 font-bold text-sm transition"
                >
                  {isAr ? "أصوات الصدر والتنفس" : "Chest & Breath Lab"}
                </button>
              </div>
            </div>

            <div className="lg:col-span-6 flex items-center justify-center">
              <img
                src="/hero-sounds.png"
                alt="Heart & ECG Simulator Preview"
                className="w-full max-w-[600px] h-auto object-contain drop-shadow-[0_0_40px_rgba(34,211,238,0.28)] transition duration-500 hover:scale-[1.02]"
              />
            </div>
          </div>

          {/* Bottom 3 Feature Cards */}
          <section className="space-y-4">
            <h2 className="text-lg font-extrabold text-white">
              {isAr ? "ماذا تقدم المنصة؟" : "What Does the Platform Offer?"}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button
                type="button"
                onClick={() => launchMode("heart")}
                className="group text-start rounded-2xl border border-white/12 bg-[#0d1528]/70 hover:bg-[#121e38]/85 hover:border-rose-400/50 backdrop-blur-xl p-5 flex items-center justify-between gap-4 transition shadow-xl"
              >
                <div>
                  <div className="font-bold text-white text-base group-hover:text-rose-200 transition">
                    {isAr ? "محاكاة أصوات اضطرابات القلب" : "Heart Murmurs & Sounds"}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {isAr ? "S1/S2 وS3/S4 واللغط القلبي مع حركة 4 حجرات" : "S1/S2, S3/S4 & murmurs with 4-chamber sync"}
                  </div>
                </div>
                <div className="w-12 h-12 shrink-0 rounded-2xl bg-rose-500/15 border border-rose-400/35 flex items-center justify-center text-rose-300">
                  <HeartPulse className="w-6 h-6" />
                </div>
              </button>

              <button
                type="button"
                onClick={() => launchMode("heart")}
                className="group text-start rounded-2xl border border-white/12 bg-[#0d1528]/70 hover:bg-[#121e38]/85 hover:border-cyan-400/50 backdrop-blur-xl p-5 flex items-center justify-between gap-4 transition shadow-xl"
              >
                <div>
                  <div className="font-bold text-white text-base group-hover:text-cyan-200 transition">
                    {isAr ? "تشخيص اضطرابات النبض الفورية" : "Live ECG Arrhythmia Lab"}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {isAr ? "AFib وVFib والـ Heart Blocks وبطاقة المرض" : "AFib, VFib, Heart Blocks & clinical cards"}
                  </div>
                </div>
                <div className="w-12 h-12 shrink-0 rounded-2xl bg-cyan-500/15 border border-cyan-400/35 flex items-center justify-center text-cyan-300">
                  <Activity className="w-6 h-6" />
                </div>
              </button>

              <button
                type="button"
                onClick={() => launchMode("breath")}
                className="group text-start rounded-2xl border border-white/12 bg-[#0d1528]/70 hover:bg-[#121e38]/85 hover:border-emerald-400/50 backdrop-blur-xl p-5 flex items-center justify-between gap-4 transition shadow-xl"
              >
                <div>
                  <div className="font-bold text-white text-base group-hover:text-emerald-200 transition">
                    {isAr ? "محاكاة أصوات الصدر والتنفس" : "Chest & Breath Auscultation"}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {isAr ? "Wheezes وCrackles وStridor مع رئة 3D" : "Wheezes, Crackles, Stridor & 3D lungs"}
                  </div>
                </div>
                <div className="w-12 h-12 shrink-0 rounded-2xl bg-emerald-500/15 border border-emerald-400/35 flex items-center justify-center text-emerald-300">
                  <Wind className="w-6 h-6" />
                </div>
              </button>
            </div>
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-transparent text-slate-100 flex flex-col overflow-hidden">
      <header className="h-13 shrink-0 px-4 border-b border-slate-800/80 bg-[#070b16]/85 backdrop-blur-md flex items-center justify-between gap-3 z-30">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setView("landing")}
            className="w-8 h-8 rounded-lg bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-400 hover:bg-emerald-500/25 transition"
            title={isAr ? "العودة للرئيسية" : "Back to Home"}
          >
            <Home className="w-4 h-4" />
          </button>
          <span className="font-bold text-sm tracking-tight text-white">
            Code Blue <span className="text-emerald-400">Sounds &amp; ECG</span>
          </span>

          <div className="ml-3 flex items-center p-1 rounded-xl bg-slate-900/90 border border-slate-800">
            <button
              type="button"
              onClick={() => launchMode("heart")}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                mode === "heart"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Heart className="w-3.5 h-3.5" />
              <span>{isAr ? "أصوات القلب والـ ECG" : "Heart & ECG"}</span>
            </button>
            <button
              type="button"
              onClick={() => launchMode("breath")}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                mode === "breath"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Wind className="w-3.5 h-3.5" />
              <span>{isAr ? "أصوات الصدر والتنفس" : "Chest & Breath"}</span>
            </button>
          </div>
        </div>

        <nav className="flex items-center gap-1.5 text-xs">
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
        <SoundsSpecimenCanvas mode={mode} />
        <div className="pointer-events-none fixed inset-x-0 top-16 bottom-3 z-20 flex items-start justify-start px-3">
          <div className="pointer-events-none w-full max-w-[340px]">
            {mode === "heart" ? <HeartControls /> : <BreathControls />}
          </div>
        </div>
      </main>
    </div>
  );
}
