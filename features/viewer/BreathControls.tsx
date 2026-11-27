'use client';

import { useAtlasLocale } from "@/lib/atlasI18n";
import {
  Minimize2,
  Pause,
  Play,
  Stethoscope,
  Volume2,
  VolumeX,
  Wind,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { useBreathStore } from "@/stores/breathStore";
import { useSceneStore } from "@/stores/sceneStore";

import { BREATH_NOW } from "./BreathDriver";
import {
  BREATH_GROUPS,
  BREATH_PATTERN_LIST,
  BREATH_PATTERNS,
  type BreathPattern,
} from "./breathing";
import { OVERLAY_CHIP, OVERLAY_CHIP_ACTION } from "./overlayChrome";

function useBreathTelemetry(active: boolean): {
  phase: "in" | "out";
  fillPct: number;
} {
  const [state, setState] = useState<{
    phase: "in" | "out";
    fillPct: number;
  }>({
    phase: "in",
    fillPct: 0,
  });

  useEffect(() => {
    if (!active) return;
    let frame = 0;
    const read = () => {
      const nowPhase = BREATH_NOW.inhaling ? "in" : "out";
      const pct = Math.round(Math.max(0, Math.min(1, BREATH_NOW.fill)) * 100);
      setState((prev) =>
        prev.phase === nowPhase && Math.abs(prev.fillPct - pct) < 2
          ? prev
          : { phase: nowPhase, fillPct: pct },
      );
      frame = requestAnimationFrame(read);
    };
    frame = requestAnimationFrame(read);
    return () => cancelAnimationFrame(frame);
  }, [active]);

  return state;
}

function formatTwoLineDiseaseLabel(
  candidate: BreathPattern,
  isAr: boolean,
): {
  line1Title: string;
  line1DashSub: string;
  line2Clinical: string;
} {
  const raw = isAr ? candidate.label_ar : candidate.label;
  const parenMatch = raw.match(/^(.*?)\s*\((.*?)\)\s*$/);
  const line1Title = parenMatch ? parenMatch[1]!.trim() : raw.trim();
  const line1DashSub = parenMatch
    ? parenMatch[2]!.trim()
    : candidate.id.replace(/_/g, " ").toUpperCase();
  const siteShort =
    candidate.auscultation_site.split("·")[0]?.trim() ||
    candidate.auscultation_site;
  return {
    line1Title,
    line1DashSub,
    line2Clinical: `${siteShort} — ${candidate.rate}`,
  };
}

/**
 * Full Chest & Respiratory Auscultation Panel (`BreathControls.tsx`).
 * - Plays the real clinical YouTube auscultation audio in a completely hidden 0x0 offscreen iframe
 *   (`اصوات التنفس رجعها لاصوات اليوتيوب بدون عرض شاشة اليوتيوب وبالمواصفات اللي ذكرتها سابقا`).
 * - Skips the "Let's listen" intro by starting at `pattern.youtubeStart` and looping cleanly within
 *   `[pattern.youtubeStart, pattern.youtubeEnd]`.
 * - Includes an interactive Audio Timeline Scrollbar (`شريط تمرير الصوت`) + vertical panel scrollbar.
 * - Keeps the 3D scene completely empty when closed.
 */

const BREATH_CLINICAL_INFO: Record<string, { pitch: string; causes: string; mgmt: string }> = {
  quiet: {
    pitch: "Soft, low-pitched rustling · Inspiratory > Expiratory (no pause)",
    causes: "Healthy normal lung parenchyma and unobstructed airways",
    mgmt: "Normal finding — no intervention needed",
  },
  deep: {
    pitch: "Augmented vesicular breath sounds · Increased tidal volume",
    causes: "Post-exercise hyperpnea, physiological sighing, anxiety",
    mgmt: "Reassurance if physiological; check SpO2 if persistent",
  },
  bronchial: {
    pitch: "Loud, hollow, high-pitched (500–1100 Hz) · Distinct pause between I & E",
    causes: "Lobar pneumonia consolidation, dense pulmonary fibrosis, above pleural effusion",
    mgmt: "CXR/CT confirmation, antibiotics for pneumonia, O2 support",
  },
  asthma_wheeze: {
    pitch: "High-pitched musical polyphonic whistles (380–720 Hz) · Prolonged expiration",
    causes: "Bronchial asthma exacerbation, acute bronchospasm, anaphylaxis",
    mgmt: "Inhaled SABA (Albuterol) + Ipratropium + systemic corticosteroids + O2",
  },
  copd_rhonchi: {
    pitch: "Low-pitched snoring/rattling (140–220 Hz) · Often clears with coughing",
    causes: "Chronic bronchitis (COPD), retained thick bronchial secretions",
    mgmt: "Bronchodilators, airway clearance/mucolytics, smoking cessation, controlled O2",
  },
  stridor: {
    pitch: "Loud monophonic high-pitched crowing (600–850 Hz) · Predominantly inspiratory",
    causes: "Croup, epiglottitis, foreign body aspiration, post-thyroidectomy hematoma, anaphylaxis",
    mgmt: "AIRWAY EMERGENCY — Nebulized epinephrine, IV dexamethasone, heliox, prepare intubation/cricothyrotomy",
  },
  fine_crackles: {
    pitch: "Dry, high-pitched late-inspiratory Velcro pops (750–1400 Hz)",
    causes: "Idiopathic pulmonary fibrosis (IPF), interstitial lung disease, early CHF",
    mgmt: "HRCT chest, pulmonary function tests, antifibrotics / treat underlying ILD",
  },
  coarse_crackles: {
    pitch: "Low-pitched wet bubbling pops (200–450 Hz) · Early inspiration & expiration",
    causes: "Acute pulmonary edema (CHF), bronchiectasis, severe pneumonia",
    mgmt: "Diuretics (Furosemide) + nitrates + NIV if pulmonary edema; antibiotics if infection",
  },
  pleural_rub: {
    pitch: "Dry leathery grating/creaking (140–360 Hz) · Biphasic · Stops on breath-hold",
    causes: "Fibrinous pleurisy, pulmonary embolism (PE), viral pleuritis, malignancy",
    mgmt: "Rule out PE (CTPA/D-dimer), analgesia (NSAIDs), treat underlying cause",
  },
  cheyne_stokes: {
    pitch: "Cyclical waxing-waning depth (crescendo–decrescendo) followed by central apnea",
    causes: "Advanced congestive heart failure, brainstem stroke, raised ICP, high altitude",
    mgmt: "Optimize heart failure GDMT / neuro status; ASV or nocturnal O2 if indicated",
  },
  kussmaul: {
    pitch: "Deep, rapid, labored sighing hyperpnea ('Air Hunger')",
    causes: "Diabetic ketoacidosis (DKA), uremia, lactic acidosis, salicylate/toxic alcohol poisoning",
    mgmt: "Check ABG, BMP, ketones/lactate — Treat DKA (IV fluids + insulin + K+) or underlying acidosis",
  },
};

export function BreathControls() {
  const { isAr, at } = useAtlasLocale();
  const enabled = useBreathStore((s) => s.enabled);
  const sound = useBreathStore((s) => s.sound);
  const patternId = useBreathStore((s) => s.pattern);
  const toggle = useBreathStore((s) => s.toggle);
  const setSound = useBreathStore((s) => s.setSound);
  const setPattern = useBreathStore((s) => s.setPattern);
  const clearSelection = useSceneStore((s) => s.clearSelection);
  const clearIsolation = useSceneStore((s) => s.clearIsolation);
  const hideAllSystems = useSceneStore((s) => s.hideAllSystems);

  const { phase, fillPct } = useBreathTelemetry(enabled);
  const pattern = BREATH_PATTERNS[patternId] ?? BREATH_PATTERNS.quiet;
  const clipDuration = Math.max(4, pattern.youtubeEnd - pattern.youtubeStart);

  const [panelMinimized, setPanelMinimized] = useState(false);
  const [volumePct, setVolumePct] = useState(100);
  const [muted, setMuted] = useState(false);
  const [audioElapsed, setAudioElapsed] = useState(0);

  const ytIframeRef = useRef<HTMLIFrameElement | null>(null);

  const sendYouTubeCommand = useCallback((func: string, args: unknown[] = []) => {
    const win = ytIframeRef.current?.contentWindow;
    if (!win) return;
    try {
      win.postMessage(
        JSON.stringify({ event: "command", func, args }),
        "*",
      );
    } catch {}
  }, []);

  useEffect(() => {
    if (enabled) {
      setPanelMinimized(false);
      clearSelection();
      setAudioElapsed(0);
      setSound(false);
    }
  }, [enabled, clearSelection, setSound]);

  const ytReadyRef = useRef(false);
  const initialVideoRef = useRef<{ id: string; start: number; end: number } | null>(null);
  if (!initialVideoRef.current && pattern.youtubeVideoId) {
    initialVideoRef.current = { id: pattern.youtubeVideoId, start: pattern.youtubeStart, end: pattern.youtubeEnd };
  }

  // Fast instant switching via persistent YouTube player (loadVideoById avoids reloading 1.5MB YouTube JS!)
  useEffect(() => {
    setAudioElapsed(0);
    if (!enabled) return;
    if (!sound) {
      sendYouTubeCommand("pauseVideo", []);
      return;
    }
    const triggerPlay = () => {
      if (pattern.youtubeVideoId) {
        sendYouTubeCommand("loadVideoById", [{
          videoId: pattern.youtubeVideoId,
          startSeconds: pattern.youtubeStart,
          endSeconds: pattern.youtubeEnd,
        }]);
      }
      sendYouTubeCommand("setVolume", [muted ? 0 : volumePct]);
      if (muted) sendYouTubeCommand("mute", []);
      else sendYouTubeCommand("unMute", []);
      sendYouTubeCommand("playVideo", []);
    };
    if (ytReadyRef.current) {
      triggerPlay();
      return;
    }
    const initTimer = window.setTimeout(triggerPlay, 180);
    return () => window.clearTimeout(initTimer);
  }, [enabled, sound, pattern.id, pattern.youtubeVideoId, pattern.youtubeStart, pattern.youtubeEnd, sendYouTubeCommand]);

  // Loop cleanly inside [youtubeStart, youtubeEnd] so "Let's listen" is never heard and track the scrollbar position
  useEffect(() => {
    if (!enabled || !sound) return;
    const interval = window.setInterval(() => {
      setAudioElapsed((prev) => {
        const next = prev + 0.25;
        if (next >= clipDuration) {
          sendYouTubeCommand("seekTo", [pattern.youtubeStart, true]);
          sendYouTubeCommand("playVideo", []);
          return 0;
        }
        return Number(next.toFixed(2));
      });
    }, 250);
    return () => window.clearInterval(interval);
  }, [enabled, sound, clipDuration, pattern.youtubeStart, sendYouTubeCommand]);

  const handleSeekAudio = (offsetSec: number) => {
    const clamped = Math.max(0, Math.min(clipDuration, offsetSec));
    setAudioElapsed(clamped);
    if (!sound) {
      setSound(true);
    }
    sendYouTubeCommand("seekTo", [pattern.youtubeStart + clamped, true]);
    sendYouTubeCommand("playVideo", []);
  };

  const stopAndClose = () => {
    setSound(false);
    setPanelMinimized(false);
    clearIsolation();
    clearSelection();
    hideAllSystems();
    if (enabled) toggle();
  };

  if (!enabled) return null;

  const activeFormatted = formatTwoLineDiseaseLabel(pattern, isAr);
  const breathInfo = BREATH_CLINICAL_INFO[pattern.id] ?? BREATH_CLINICAL_INFO.quiet!;
  const initVid = initialVideoRef.current ?? { id: pattern.youtubeVideoId, start: pattern.youtubeStart, end: pattern.youtubeEnd };
  const ytEmbedUrl = `https://www.youtube.com/embed/${initVid.id}?enablejsapi=1&autoplay=0&mute=0&controls=0&modestbranding=1&rel=0&playsinline=1&start=${initVid.start}&end=${initVid.end}`;

  return (
    <div dir={isAr ? "rtl" : "ltr"} className="pointer-events-auto flex flex-col items-start gap-1.5">
      {/* Persistent hidden 1x1 YouTube audio player: stays warm so switching breath sounds is ~10x faster */}
      {initVid.id && (
        <iframe
          ref={ytIframeRef}
          src={ytEmbedUrl}
          title={pattern.label}
          allow="autoplay; encrypted-media"
          onLoad={() => {
            ytReadyRef.current = true;
            if (!sound) {
              sendYouTubeCommand("pauseVideo", []);
              return;
            }
            if (pattern.youtubeVideoId && pattern.youtubeVideoId !== initVid.id) {
              sendYouTubeCommand("loadVideoById", [{
                videoId: pattern.youtubeVideoId,
                startSeconds: pattern.youtubeStart,
                endSeconds: pattern.youtubeEnd,
              }]);
            } else {
              sendYouTubeCommand("seekTo", [pattern.youtubeStart, true]);
            }
            sendYouTubeCommand("setVolume", [muted ? 0 : volumePct]);
            if (!muted) sendYouTubeCommand("unMute", []);
            sendYouTubeCommand("playVideo", []);
          }}
          aria-hidden="true"
          tabIndex={-1}
          className="pointer-events-none fixed -left-[9999px] -top-[9999px] h-[1px] w-[1px] border-0 opacity-0"
        />
      )}

      {panelMinimized && (
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setPanelMinimized(false)}
            className={`pointer-events-auto flex items-center gap-1 ${OVERLAY_CHIP} ${OVERLAY_CHIP_ACTION}`}
          >
            <span className="inline-flex items-center gap-1">
              <Wind size={12} className="text-[#d98a96]" />
              <span>{at("إظهار نافذة أصوات الصدر والتنفس", "Breath Sounds")}</span>
            </span>
          </button>
          <button
            type="button"
            onClick={stopAndClose}
            className="flex h-6 w-6 items-center justify-center rounded-lg border border-[#26324d] bg-[#131c31] text-xs font-bold text-[#f1f5f9] shadow transition hover:border-[#d98a96]"
          >
            <X size={12} />
          </button>
        </div>
      )}

      {!panelMinimized && (
        <div
          onWheel={(e) => e.stopPropagation()}
          className="cb-ausc-panel pointer-events-auto atlas-scrollbar w-[86vw] max-w-[340px] sm:w-[360px] md:w-[390px] sm:max-w-none max-h-[calc(100dvh-80px)] overflow-y-auto overscroll-contain rounded-2xl border border-[#26324d] bg-[#131c31]/98 p-3.5 text-[11px] leading-snug text-[#f1f5f9] shadow-2xl backdrop-blur-md"
        >
          <div className="mb-2 flex items-center justify-between border-b border-[#26324d] pb-2">
            <span className="font-bold text-[#d98a96]">
              <span className="inline-flex items-center gap-1.5">
                <Wind size={13} className="text-[#d98a96]" />
                <span>
                  {at("أصوات الصدر والتنفس", "Chest & Breath Sounds")} — {pattern.rate}
                </span>
              </span>
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setPanelMinimized(true)}
                className="inline-flex items-center gap-1 rounded-lg border border-[#26324d] bg-[#0f172a] px-2 py-0.5 text-[9.5px] font-bold text-[#94a3b8] transition hover:border-[#3b4d71] hover:text-[#f1f5f9]"
              >
                <Minimize2 size={10} />
                <span>{at("تصغير", "Hide")}</span>
              </button>
              <button
                type="button"
                onClick={stopAndClose}
                className="inline-flex items-center gap-1 rounded-lg border border-[#26324d] bg-[#0f172a] px-2 py-0.5 text-[9.5px] font-bold text-[#f1f5f9] transition hover:border-[#d98a96]"
              >
                <X size={10} />
                <span>{at("إغلاق", "Close")}</span>
              </button>
            </div>
          </div>

          {/* Active Pattern Card */}
          <div className="rounded-xl border border-[#26324d] bg-[#0b1120] p-2.5">
            <p className="text-[11px] font-bold text-[#d98a96]">
              {activeFormatted.line1Title} —{" "}
              <span className="text-[#f1f5f9]">{activeFormatted.line1DashSub}</span>
            </p>
            <p className="mt-0.5 font-mono text-[9.5px] text-[#94a3b8]">
              <span className="inline-flex items-center gap-1">
                <Stethoscope size={11} className="text-[#d98a96]" />
                <span>{activeFormatted.line2Clinical}</span>
              </span>
            </p>
            <p className="mt-1.5 text-[10px] leading-relaxed text-[#f1f5f9]">
              {isAr ? pattern.what_ar : pattern.what}
            </p>
          </div>

          {/* Live Inspiration / Expiration Progress Bar */}
          <div className="my-2 rounded-xl border border-[#26324d] bg-[#0b1120] p-2">
            <div className="mb-1 flex items-center justify-between text-[9.5px] font-bold">
              <span className={phase === "in" ? "text-[#d98a96]" : "text-[#94a3b8]"}>
                {phase === "in"
                  ? at("● شهيق (Inspiration — انقباض الحجاب الحاجز)", "● Inspiration (Diaphragm descent)")
                  : at("○ زفير (Expiration — ارتخاء القفص الصدري)", "○ Expiration (Elastic recoil)")}
              </span>
              <span className="font-mono text-[#f1f5f9]">{fillPct}%</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#131c31]">
              <div
                className="h-full rounded-full bg-[#d98a96] transition-all duration-75"
                style={{ width: `${fillPct}%` }}
              />
            </div>
          </div>

          {/* Dedicated Audio Control Box + Interactive Audio Scrollbar (`شريط الالسكرول`) */}
          <div className="my-2 rounded-xl border border-[#26324d] bg-[#0b1120] p-2.5">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  const next = !sound;
                  setSound(next);
                  if (!next) {
                    sendYouTubeCommand("pauseVideo", []);
                  }
                }}
                className={`flex flex-1 items-center justify-center gap-1.5 rounded-xl border px-3 py-1.5 text-[10.5px] font-bold transition ${
                  sound
                    ? "border-[#3b4d71] bg-[#1e293b] text-[#f1f5f9]"
                    : "border-[#26324d] bg-[#0f172a] text-[#94a3b8] hover:text-[#f1f5f9]"
                }`}
              >
                {sound ? (
                  <>
                    <Pause size={13} className="shrink-0 text-[#d98a96]" />
                    <span>{at("إيقاف الصوت مؤقتاً", "Pause Breath Sound")}</span>
                  </>
                ) : (
                  <>
                    <Play size={13} className="shrink-0 text-[#d98a96]" />
                    <span>{at("تشغيل صوت السماعة الطبية", "Play Stethoscope Sound")}</span>
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={() => {
                  const nextMuted = !muted;
                  setMuted(nextMuted);
                  if (nextMuted) sendYouTubeCommand("mute", []);
                  else {
                    sendYouTubeCommand("unMute", []);
                    sendYouTubeCommand("setVolume", [volumePct]);
                  }
                }}
                title={muted ? at("إلغاء كتم الصوت", "Unmute") : at("كتم الصوت", "Mute")}
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border transition ${
                  muted
                    ? "border-rose-500/50 bg-rose-500/20 text-rose-300"
                    : "border-[#26324d] bg-[#0f172a] text-[#f1f5f9] hover:border-[#3b4d71]"
                }`}
              >
                {muted ? <VolumeX size={14} /> : <Volume2 size={14} className="text-[#d98a96]" />}
              </button>
            </div>

            {/* Interactive Audio Timeline Scrollbar (`شريط السكرول للصوت المطلوب`) */}
            <div className="mt-2.5 space-y-1">
              <div className="flex items-center justify-between font-mono text-[9px] text-[#94a3b8]">
                <span>{at("شريط تمرير الصوت (Audio Scrollbar)", "Audio Timeline Scrollbar")}</span>
                <span className="text-[#f1f5f9]">
                  {audioElapsed.toFixed(1)}s / {clipDuration.toFixed(1)}s
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={clipDuration}
                step={0.1}
                value={Math.min(clipDuration, audioElapsed)}
                onChange={(e) => handleSeekAudio(Number(e.target.value))}
                className="h-2 w-full cursor-pointer accent-[#d98a96]"
              />
            </div>

            {/* Volume Slider */}
            <div className="mt-2 flex items-center gap-2">
              <span className="font-mono text-[9px] text-[#94a3b8]">
                {at("مستوى الصوت:", "Vol:")}
              </span>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={muted ? 0 : volumePct}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setVolumePct(v);
                  if (v > 0 && muted) {
                    setMuted(false);
                    sendYouTubeCommand("unMute", []);
                  }
                  sendYouTubeCommand("setVolume", [v]);
                }}
                className="h-1.5 flex-1 cursor-pointer accent-[#d98a96]"
              />
              <span className="w-8 text-end font-mono text-[9px] text-[#f1f5f9]">
                {muted ? "0%" : `${volumePct}%`}
              </span>
            </div>
          </div>

          {/* 11 Respiratory Auscultation Patterns with Visible Vertical Scrollbar */}
          <div className="mt-2.5">
            <p className="mb-1.5 font-mono text-[9.5px] font-bold text-[#d98a96]">
              {at(
                "أنماط التنفس وأصوات الصدر المرضية (11 حالة):",
                "Respiratory Patterns & Lung Sounds (11 cases):",
              )}
            </p>
            <div
              role="radiogroup"
              aria-label="Respiratory pattern"
              className="space-y-2 rounded-xl border border-[#26324d] bg-[#0b1120] p-2"
            >
              {BREATH_GROUPS.map((group, gIdx) => {
                const items = BREATH_PATTERN_LIST.filter((c) => c.group === group);
                if (items.length === 0) return null;
                const accentBar =
                  gIdx % 3 === 0
                    ? "border-s-[3.5px] border-s-[#c83e4d]"
                    : gIdx % 3 === 1
                      ? "border-s-[3.5px] border-s-[#2563eb]"
                      : "border-s-[3.5px] border-s-[#ca8a04]";
                return (
                  <div key={group} className="space-y-1">
                    <p className="px-1 font-mono text-[8.5px] font-medium uppercase tracking-wider text-[#d98a96]">
                      {group}
                    </p>
                    <div className="relative space-y-1.5 ps-3.5 before:absolute before:inset-y-2 before:start-1 before:w-[1.5px] before:bg-gradient-to-b before:from-[#d98a96]/65 before:via-[#3b4d71]/45 before:to-transparent">
                      {items.map((candidate) => {
                        const active = candidate.id === pattern.id;
                        const formatted = formatTwoLineDiseaseLabel(candidate, isAr);
                        return (
                          <div key={candidate.id} className="relative flex items-center">
                            <svg
                              aria-hidden
                              viewBox="0 0 14 24"
                              className="pointer-events-none absolute -start-3.5 top-1/2 h-5 w-3.5 -translate-y-1/2 overflow-visible"
                            >
                              <path
                                d="M 1 2 C 1 12, 6 12, 13 12"
                                fill="none"
                                stroke={active ? "#d98a96" : "rgba(163, 155, 144, 0.42)"}
                                strokeWidth={active ? "1.6" : "1.15"}
                                strokeLinecap="round"
                              />
                            </svg>
                            <button
                              type="button"
                              role="radio"
                              aria-checked={active}
                              onClick={() => {
                                setPattern(candidate.id);
                                setAudioElapsed(0);
                                if (!sound) {
                                  setSound(true);
                                }
                              }}
                              className={`flex w-full flex-col gap-0.5 rounded-xl border ${accentBar} px-2.5 py-1.5 text-start transition ${
                                active
                                  ? "border-[#3b4d71] bg-[#1e293b] text-[#f1f5f9]"
                                  : "border-[#26324d] bg-[#131c31]/85 text-[#f1f5f9] hover:border-[#3b4d71] hover:bg-[#1e293b]"
                              }`}
                            >
                              <div className="w-full text-[10px] font-semibold leading-snug text-[#f1f5f9]">
                                <span>{formatted.line1Title}</span>
                                <span className="mx-1 text-[#d98a96]">—</span>
                                <span className="font-mono text-[9px] text-[#d98a96]">
                                  {formatted.line1DashSub}
                                </span>
                              </div>
                              <div className="w-full font-mono text-[8.5px] leading-tight text-[#94a3b8]">
                                {formatted.line2Clinical}
                              </div>
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <button
            type="button"
            onClick={stopAndClose}
            className="mt-2.5 flex w-full items-center justify-center gap-1.5 rounded-xl border border-[#26324d] bg-[#0f172a] py-1.5 text-[10px] font-bold text-[#f1f5f9] transition hover:border-[#d98a96]"
          >
            <X size={11} className="text-[#d98a96]" />
            <span>{at("إيقاف الصوت والتنفس وإغلاق النافذة", "Stop Breathing & Close")}</span>
          </button>
        </div>
      )}

      {!panelMinimized && (
        <div
          onWheel={(e) => e.stopPropagation()}
          className="cb-ausc-panel pointer-events-auto atlas-scrollbar mt-2 w-full md:mt-0 md:fixed md:right-4 lg:right-12 xl:right-20 md:top-16 md:z-30 md:w-[300px] lg:w-[325px] max-h-[calc(100dvh-90px)] overflow-y-auto overscroll-contain rounded-2xl border border-[#26324d] bg-[#131c31]/96 p-3.5 text-[11px] leading-relaxed text-[#f1f5f9] shadow-2xl backdrop-blur-md"
        >
          <div className="mb-2 flex items-center justify-between border-b border-[#26324d] pb-2">
            <span className="font-bold text-[#d98a96]">{at("بطاقة المعلومات السريرية", "Clinical Disease Card")}</span>
            <span className="rounded-md border border-[#d98a96]/40 bg-[#d98a96]/15 px-2 py-0.5 font-mono text-[9px] font-bold text-[#d98a96]">
              {pattern.rate}
            </span>
          </div>
          <p className="text-xs font-black text-[#f1f5f9]">{isAr ? pattern.label_ar : pattern.label}</p>
          <p className="mt-1 text-[10px] text-[#cbd5e1]">{isAr ? pattern.what_ar : pattern.what}</p>
          <div className="mt-2.5 space-y-2 rounded-xl border border-[#26324d] bg-[#0b1120] p-2.5 text-[10px]">
            <div>
              <span className="font-bold text-[#d98a96]">{at("موضع السماعة:", "Auscultation Site:")} </span>
              <span className="text-[#e2e8f0]">{pattern.auscultation_site}</span>
            </div>
            <div>
              <span className="font-bold text-sky-300">{at("الخصائص الصوتية:", "Acoustic Profile:")} </span>
              <span className="text-[#e2e8f0]">{breathInfo.pitch}</span>
            </div>
            <div>
              <span className="font-bold text-amber-300">{at("الأسباب الشائعة:", "Common Causes:")} </span>
              <span className="text-[#e2e8f0]">{breathInfo.causes}</span>
            </div>
            <div>
              <span className="font-bold text-emerald-300">{at("التدبير العلاجي:", "Management:")} </span>
              <span className="text-[#e2e8f0]">{breathInfo.mgmt}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
