'use client';

import React, { useState } from "react";
import { ArrowLeft, Box, Heart, ScanLine, Stethoscope, Wind } from "lucide-react";
import { SoundsSpecimenCanvas } from "@/features/viewer/SoundsSpecimenCanvas";
import { HeartControls } from "@/features/viewer/HeartControls";
import { BreathControls } from "@/features/viewer/BreathControls";
import { useHeartStore } from "@/stores/heartStore";
import { useBreathStore } from "@/stores/breathStore";
import { primeHeartSound } from "@/features/viewer/heartSound";
import { primeBreathSound } from "@/features/viewer/breathing";
import { useAtlasLocale } from "@/lib/atlasI18n";

const MAIN_URL = process.env.NEXT_PUBLIC_MAIN_URL || "http://localhost:3000";
const RADIOLOGY_URL = process.env.NEXT_PUBLIC_RADIOLOGY_URL || "http://localhost:3001";
const ATLAS_URL = process.env.NEXT_PUBLIC_ATLAS_URL || "http://localhost:3003";

export default function SoundsLabPage() {
  const [mode, setMode] = useState<"heart" | "breath">("heart");
  const heartEnabled = useHeartStore((s) => s.enabled);
  const toggleHeart = useHeartStore((s) => s.toggle);
  const breathEnabled = useBreathStore((s) => s.enabled);
  const toggleBreath = useBreathStore((s) => s.toggle);
  const { locale, setLocale } = useAtlasLocale();

  const switchToHeart = () => {
    primeHeartSound();
    if (breathEnabled) toggleBreath();
    if (!heartEnabled) toggleHeart();
    setMode("heart");
  };

  const switchToBreath = () => {
    primeBreathSound();
    if (heartEnabled) toggleHeart();
    if (!breathEnabled) toggleBreath();
    setMode("breath");
  };

  return (
    <div className="h-screen w-screen bg-[#060913] text-slate-100 flex flex-col overflow-hidden">
      <header className="h-13 shrink-0 px-4 border-b border-slate-800/80 bg-[#070b16]/95 backdrop-blur-md flex items-center justify-between gap-3 z-30">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
            <Stethoscope className="w-4 h-4" />
          </div>
          <span className="font-bold text-sm tracking-tight text-white">
            Code Blue <span className="text-emerald-400">Sounds &amp; ECG</span>
          </span>

          {/* Mode Switcher */}
          <div className="ml-3 flex items-center p-1 rounded-xl bg-slate-900/90 border border-slate-800">
            <button
              type="button"
              onClick={switchToHeart}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                mode === "heart"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Heart className="w-3.5 h-3.5" />
              <span>{locale === "ar" ? "أصوات القلب والـ ECG" : "Heart & ECG"}</span>
            </button>
            <button
              type="button"
              onClick={switchToBreath}
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition ${
                mode === "breath"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Wind className="w-3.5 h-3.5" />
              <span>{locale === "ar" ? "أصوات الصدر والتنفس" : "Chest & Breath"}</span>
            </button>
          </div>
        </div>

        <nav className="flex items-center gap-1.5 text-xs">
          <a
            href={MAIN_URL}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/70 hover:bg-slate-700/80 text-slate-200 border border-slate-700/60 transition"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>{locale === "ar" ? "بنك الأسئلة" : "Main Bank"}</span>
          </a>
          <a
            href={RADIOLOGY_URL}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 transition"
          >
            <ScanLine className="w-3.5 h-3.5" />
            <span>{locale === "ar" ? "الأشعة والـ AI" : "Radiology AI"}</span>
          </a>
          <a
            href={ATLAS_URL}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-violet-500/10 hover:bg-violet-500/20 text-violet-300 border border-violet-500/30 transition"
          >
            <Box className="w-3.5 h-3.5" />
            <span>{locale === "ar" ? "الأطلس 3D" : "3D Atlas"}</span>
          </a>
          <button
            type="button"
            onClick={() => setLocale(locale === "ar" ? "en" : "ar")}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800/70 hover:bg-slate-700/80 text-slate-200 border border-slate-700/60 font-semibold"
          >
            {locale === "ar" ? "EN" : "AR"}
          </button>
        </nav>
      </header>

      <main className="relative flex-1 w-full overflow-hidden">
        <SoundsSpecimenCanvas mode={mode} />
        <div className="pointer-events-none absolute inset-0 flex items-end justify-center p-4 z-20">
          <div className="pointer-events-auto w-full max-w-4xl">
            {mode === "heart" ? <HeartControls /> : <BreathControls />}
          </div>
        </div>
      </main>
    </div>
  );
}
