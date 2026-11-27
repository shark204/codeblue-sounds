"use client";

import React, { useState } from "react";

interface NeuralStep {
  num: string;
  titleAr: string;
  bodyAr: string;
  titleEn: string;
  bodyEn: string;
}

const LEFT_STEPS: NeuralStep[] = [
  {
    num: "01",
    titleAr: "التقاط ذكي",
    bodyAr: "فحص الوضوح، وتنظيف الورق والشبكة، وتنبيه لو الأقطاب متبدلة (LA–RA / LA–LL / RA–LL).",
    titleEn: "Smart capture",
    bodyEn: "Clarity check, paper/grid cleaning, and alert if limb leads are reversed (LA–RA, LA–LL, RA–LL).",
  },
  {
    num: "03",
    titleAr: "التحليل العصبي الثنائي",
    bodyAr: "شبكتان عصبيتان (EfficientNetV2-S + DenseNet121) مُدرّبتان على +72,000 تخطيط من 4 قواعد عالمية.",
    titleEn: "Dual deep ensemble",
    bodyEn: "Two neural networks (EfficientNetV2-S + DenseNet121) trained on 72,000+ real ECGs from 4 global databases.",
  },
  {
    num: "05",
    titleAr: "بطاقات سريرية تكيفية",
    bodyAr: "علامات حيوية وأعراض تحسم المتشابهات (Pericarditis مقابل STEMI، Wellens، De Winter...).",
    titleEn: "Adaptive clinical cards",
    bodyEn: "Vitals and symptoms resolve look-alikes (pericarditis vs STEMI, Wellens, De Winter, and others).",
  },
];

const RIGHT_STEPS: NeuralStep[] = [
  {
    num: "02",
    titleAr: "قراءة هيدر الجهاز (OCR)",
    bodyAr: "HR وPR وQRS وQT وAxis، أو تحديدها بشكل ذكي وتلقائي.",
    titleEn: "Device header OCR",
    bodyEn: "HR, PR, QRS, QT, and axis read from machine header, or determined intelligently and automatically.",
  },
  {
    num: "04",
    titleAr: "معايرة وخريطة انتباه",
    bodyAr: "معايرة الاحتمالات لكل فئة، وخريطة حرارية (Grad-CAM) توضّح مناطق التركيز.",
    titleEn: "Calibration & Grad-CAM",
    bodyEn: "Per-class calibrated probabilities and a Grad-CAM heatmap revealing model focus regions.",
  },
  {
    num: "06",
    titleAr: "دمج وقواعد طوارئ",
    bodyAr: "جمع الصورة وأرقام الجهاز والعلامات الحيوية في 68 حالة بضوابط سريرية مشددة.",
    titleEn: "Fusion & emergency rules",
    bodyEn: "Image, device numbers, and vitals fused across 68 cases under strict clinical controls.",
  },
];

const STEP_7: NeuralStep = {
  num: "07",
  titleAr: "بوابة يقين وتقرير",
  bodyAr: "رصد حتى 4 تشخيصات متواجدة في نفس الرسم.",
  titleEn: "Certainty gate & report",
  bodyEn: "Up to four concurrent diagnoses are detected within the same tracing.",
};

export function NeuralDiagnosticTree({
  isAr,
  isLight,
}: {
  isAr: boolean;
  isLight: boolean;
  onLaunchEcg?: () => void;
}) {
  const [activeNum, setActiveNum] = useState<string | null>(null);

  const renderBox = (st: NeuralStep, dotPos: "right" | "left" | "top") => {
    const isActive = activeNum === st.num;
    return (
      <div
        key={st.num}
        dir={isAr ? "rtl" : "ltr"}
        onMouseEnter={() => setActiveNum(st.num)}
        onMouseLeave={() => setActiveNum(null)}
        className={`relative rounded-xl md:rounded-2xl border p-1.5 sm:p-2 md:p-2.5 backdrop-blur-xl transition-all duration-300 select-none cursor-default ${
          isLight
            ? isActive
              ? "bg-white/95 border-[#be123c]/65 shadow-[0_10px_24px_rgba(159,18,57,0.18)] -translate-y-0.5 scale-[1.02]"
              : "bg-white/82 border-rose-200/80 shadow-sm hover:border-[#be123c]/45"
            : isActive
              ? "bg-[#220a15]/88 border-[#fb7185]/65 shadow-[0_0_24px_rgba(190,18,60,0.38)] -translate-y-0.5 scale-[1.02]"
              : "bg-[#120710]/68 border-rose-400/22 hover:border-rose-400/45"
        }`}
      >
        <span
          aria-hidden="true"
          className={`block absolute w-1.5 h-1.5 md:w-2.5 md:h-2.5 rounded-full transition-all duration-300 ${
            dotPos === "right"
              ? "top-1/2 -translate-y-1/2 -right-[4px] md:-right-[6px]"
              : dotPos === "left"
                ? "top-1/2 -translate-y-1/2 -left-[4px] md:-left-[6px]"
                : "left-1/2 -translate-x-1/2 -top-[4px] md:-top-[6px]"
          } ${
            isActive
              ? "bg-[#ffe4e6] scale-125 shadow-[0_0_12px_rgba(251,113,133,1)] ring-2 ring-[#be123c]/70"
              : isLight
                ? "bg-[#be123c] shadow-[0_0_7px_rgba(190,18,60,0.55)]"
                : "bg-[#fb7185] shadow-[0_0_9px_rgba(251,113,133,0.85)]"
          }`}
        />

        <div className="flex items-center gap-1 md:gap-1.5 mb-0.5 md:mb-1">
          <span
            className={`inline-flex items-center justify-center w-3.5 h-3.5 md:w-5 md:h-5 rounded-full text-[7px] md:text-[9.5px] font-black font-mono shrink-0 border ${
              isLight
                ? "bg-rose-100 text-[#881337] border-rose-300"
                : "bg-[#881337]/55 text-[#fda4af] border-[#fb7185]/45"
            }`}
          >
            {st.num}
          </span>
          <h3 className={`text-[8.5px] sm:text-[10px] md:text-[11.5px] font-black leading-tight ${isLight ? "text-slate-900" : "text-white"}`}>
            {isAr ? st.titleAr : st.titleEn}
          </h3>
        </div>
        <p className={`text-[7.2px] sm:text-[8.5px] md:text-[10.5px] leading-[1.26] md:leading-[1.38] ${isLight ? "text-slate-600" : "text-slate-300/90"}`}>
          {isAr ? st.bodyAr : st.bodyEn}
        </p>
      </div>
    );
  };

  const maskStyle: React.CSSProperties = {
    WebkitMaskImage: "url('/neural-tree-burgundy.png')",
    maskImage: "url('/neural-tree-burgundy.png')",
    WebkitMaskSize: "contain",
    maskSize: "contain",
    WebkitMaskRepeat: "no-repeat",
    maskRepeat: "no-repeat",
    WebkitMaskPosition: "center",
    maskPosition: "center",
  };

  return (
    <div className="w-full space-y-2 select-none">
      <style>{`
        @keyframes cbTreeRootImpulse {
          0%   { background-position: 50% 135%; opacity: 0; }
          6%   { opacity: 0.95; }
          56%  { background-position: 50% -25%; opacity: 0.92; }
          64%  { background-position: 50% -45%; opacity: 0; }
          100% { background-position: 50% -45%; opacity: 0; }
        }
        .cb-tree-impulse {
          background-image: linear-gradient(
            to top,
            transparent 0%,
            rgba(190, 18, 60, 0.35) 36%,
            rgba(253, 164, 175, 0.92) 48%,
            rgba(255, 241, 242, 1) 51%,
            rgba(251, 113, 133, 0.85) 55%,
            transparent 70%
          );
          background-size: 100% 340%;
          background-repeat: no-repeat;
          animation: cbTreeRootImpulse 10s cubic-bezier(0.32, 0.04, 0.25, 1) infinite;
        }
      `}</style>

      {/* Eyebrow Headline */}
      <div
        dir={isAr ? "rtl" : "ltr"}
        className={`text-center text-[11px] sm:text-xs md:text-sm font-black leading-snug ${
          isLight ? "text-[#881337]" : "text-[#fda4af]"
        }`}
      >
        {isAr
          ? "من صورة رسم القلب للنتيجة النهائية في 7 خطوات سريعة. نظام بيرفض التخمين."
          : "From an ECG image to a final result in seven quick steps — a system that refuses to guess."}
      </div>

      {/* Unified Staggered Tree Layout — Scaled Compactly on Mobile & Full Size on Desktop */}
      <div dir="ltr" className="relative w-full h-[395px] sm:h-[460px] md:h-[550px]">
        {/* Central Flowing Burgundy Neural Tree */}
        <div className="pointer-events-none absolute inset-x-0 top-0 bottom-[62px] md:bottom-[78px] flex items-center justify-center">
          <div
            aria-hidden="true"
            className={`absolute w-[170px] h-[170px] sm:w-[240px] sm:h-[240px] md:w-[350px] md:h-[350px] rounded-full blur-3xl ${
              isLight ? "bg-[#be123c]/12" : "bg-[#9f1239]/26"
            }`}
          />
          <div className="relative h-[275px] w-[160px] sm:h-[350px] sm:w-[250px] md:h-[460px] md:w-[380px] max-w-[445px] flex items-center justify-center">
            <img
              src="/neural-tree-burgundy.png"
              alt="Code Blue Flowing Burgundy Neural Tree"
              draggable={false}
              className={`relative h-full w-full object-contain select-none ${
                isLight
                  ? "brightness-95 contrast-110 drop-shadow-[0_10px_24px_rgba(136,19,55,0.22)]"
                  : "opacity-95 drop-shadow-[0_0_28px_rgba(190,18,60,0.45)]"
              }`}
            />
            <div
              aria-hidden="true"
              style={maskStyle}
              className="cb-tree-impulse pointer-events-none absolute inset-0 mix-blend-screen drop-shadow-[0_0_14px_rgba(251,113,133,0.95)]"
            />
          </div>
        </div>

        {/* Left 3 Panels (01, 03, 05) */}
        <div className="absolute left-0 top-[2px] md:top-[6px] bottom-[98px] md:bottom-[136px] w-[34%] sm:w-[31%] md:w-[29%] flex flex-col justify-between z-10">
          {LEFT_STEPS.map((st) => renderBox(st, "right"))}
        </div>

        {/* Right 3 Panels (Staggered downward: 02, 04, 06) */}
        <div className="absolute right-0 top-[44px] md:top-[70px] bottom-[54px] md:bottom-[72px] w-[34%] sm:w-[31%] md:w-[29%] flex flex-col justify-between z-10">
          {RIGHT_STEPS.map((st) => renderBox(st, "left"))}
        </div>

        {/* Bottom Center Finale Box (07) */}
        <div className="absolute left-1/2 -translate-x-1/2 bottom-0 md:bottom-1 w-[46%] sm:w-[40%] md:w-[34%] z-10">
          {renderBox(STEP_7, "top")}
        </div>
      </div>
    </div>
  );
}
