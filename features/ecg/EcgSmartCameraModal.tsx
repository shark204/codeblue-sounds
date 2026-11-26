"use client";

import React, { useEffect, useRef, useState } from "react";
import { Camera, CheckCircle2, Flashlight, Focus, X } from "lucide-react";

interface Props {
  isOpen: boolean;
  isAr: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
}

type ScanState = "searching" | "too_dark" | "blurry" | "hold_steady" | "locked";

export function EcgSmartCameraModal({ isOpen, isAr, onClose, onCapture }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const prevFrameRef = useRef<Uint8ClampedArray | null>(null);
  const stableHitsRef = useRef<number>(0);
  const capturedRef = useRef<boolean>(false);

  const [scanState, setScanState] = useState<ScanState>("searching");
  const [lockProgress, setLockProgress] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [torchOn, setTorchOn] = useState<boolean>(false);
  const [torchSupported, setTorchSupported] = useState<boolean>(false);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    prevFrameRef.current = null;
    stableHitsRef.current = 0;
  };

  const captureCurrentFrame = () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || capturedRef.current) return;
    capturedRef.current = true;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    // Crop cleanly to the central 16:10 ECG guide box (86% width, matching visual overlay)
    const cropW = Math.round(vw * 0.90);
    const cropH = Math.min(vh, Math.round(cropW * 0.64));
    const sx = Math.max(0, Math.round((vw - cropW) / 2));
    const sy = Math.max(0, Math.round((vh - cropH) / 2));

    const out = document.createElement("canvas");
    out.width = cropW;
    out.height = cropH;
    const ctx = out.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, sx, sy, cropW, cropH, 0, 0, cropW, cropH);

    out.toBlob(
      (blob) => {
        stopCamera();
        if (blob) {
          const file = new File([blob], `ecg-smart-scan-${Date.now()}.jpg`, {
            type: "image/jpeg",
          });
          onCapture(file);
        }
        onClose();
      },
      "image/jpeg",
      0.94
    );
  };

  useEffect(() => {
    if (!isOpen) {
      stopCamera();
      return;
    }
    capturedRef.current = false;
    setErrorMsg(null);
    setScanState("searching");
    setLockProgress(0);

    let cancelled = false;
    let timerId: number | null = null;
    const probe = document.createElement("canvas");
    probe.width = 240;
    probe.height = 150;
    const pctx = probe.getContext("2d", { willReadFrequently: true });

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const track = stream.getVideoTracks()[0];
        const caps = (track.getCapabilities?.() ?? {}) as Record<string, unknown>;
        setTorchSupported(Boolean(caps.torch));

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        // Real-time QR-style ECG paper + sharpness + stability lock analyzer (every 220ms)
        timerId = window.setInterval(() => {
          const video = videoRef.current;
          if (!video || video.readyState < 2 || !pctx || capturedRef.current) return;
          const vw = video.videoWidth;
          const vh = video.videoHeight;
          const cw = Math.round(vw * 0.88);
          const ch = Math.min(vh, Math.round(cw * 0.62));
          const sx = Math.round((vw - cw) / 2);
          const sy = Math.round((vh - ch) / 2);
          pctx.drawImage(video, sx, sy, cw, ch, 0, 0, probe.width, probe.height);

          const data = pctx.getImageData(0, 0, probe.width, probe.height).data;
          const n = probe.width * probe.height;
          const gray = new Uint8ClampedArray(n);
          let sumLuma = 0;
          let darkInk = 0;
          let paperPx = 0;
          for (let i = 0; i < n; i++) {
            const r = data[i * 4];
            const g = data[i * 4 + 1];
            const b = data[i * 4 + 2];
            const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
            gray[i] = lum;
            sumLuma += lum;
            if (lum > 145) paperPx++;
            if (lum < 95) darkInk++;
          }
          const meanLuma = sumLuma / n;
          const paperRatio = paperPx / n;
          const inkRatio = darkInk / n;

          // Laplacian sharpness variance inside guide frame
          let lapSum = 0;
          let lapSq = 0;
          let lapCount = 0;
          const W = probe.width;
          const H = probe.height;
          for (let y = 1; y < H - 1; y += 2) {
            for (let x = 1; x < W - 1; x += 2) {
              const idx = y * W + x;
              const lap =
                4 * gray[idx] -
                gray[idx - 1] -
                gray[idx + 1] -
                gray[idx - W] -
                gray[idx + W];
              lapSum += lap;
              lapSq += lap * lap;
              lapCount++;
            }
          }
          const lapMean = lapSum / Math.max(1, lapCount);
          const sharpness = lapSq / Math.max(1, lapCount) - lapMean * lapMean;

          // Frame-to-frame stability (hand steadiness)
          let motion = 999;
          if (prevFrameRef.current) {
            let diff = 0;
            for (let i = 0; i < n; i += 4) {
              diff += Math.abs(gray[i] - prevFrameRef.current[i]);
            }
            motion = diff / (n / 4);
          }
          prevFrameRef.current = gray;

          const looksLikeEcgPaper = paperRatio >= 0.42 && inkRatio >= 0.015 && inkRatio <= 0.45;

          if (meanLuma < 80) {
            stableHitsRef.current = 0;
            setLockProgress(0);
            setScanState("too_dark");
          } else if (!looksLikeEcgPaper) {
            stableHitsRef.current = 0;
            setLockProgress(0);
            setScanState("searching");
          } else if (sharpness < 140) {
            stableHitsRef.current = 0;
            setLockProgress(15);
            setScanState("blurry");
          } else if (motion > 7.5) {
            stableHitsRef.current = Math.max(0, stableHitsRef.current - 1);
            setLockProgress(Math.min(60, stableHitsRef.current * 25));
            setScanState("hold_steady");
          } else {
            stableHitsRef.current += 1;
            const pct = Math.min(100, stableHitsRef.current * 34);
            setLockProgress(pct);
            if (stableHitsRef.current >= 3) {
              setScanState("locked");
              captureCurrentFrame();
            } else {
              setScanState("hold_steady");
            }
          }
        }, 220);
      } catch {
        setErrorMsg(
          isAr
            ? "تعذّر فتح الكاميرا — تأكد من السماح بصلاحية الكاميرا في المتصفح."
            : "Camera access denied — please allow camera permissions in your browser."
        );
      }
    };

    void start();
    return () => {
      cancelled = true;
      if (timerId !== null) window.clearInterval(timerId);
      stopCamera();
    };
  }, [isOpen]);

  const toggleTorch = async () => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) return;
    try {
      const next = !torchOn;
      await track.applyConstraints({
        advanced: [{ torch: next } as MediaTrackConstraintSet],
      });
      setTorchOn(next);
    } catch {
      // Ignore if device rejects torch toggle
    }
  };

  if (!isOpen) return null;

  const statusText: Record<ScanState, string> = {
    searching: isAr ? "وجّه ورقة الـ ECG داخل الإطار" : "Align ECG paper inside the frame",
    too_dark: isAr ? "الإضاءة ضعيفة — شغّل الفلاش أو زوّد النور" : "Too dark — enable flash or add light",
    blurry: isAr ? "الصورة مهزوزة — قرّب أو انتظر التركيز" : "Out of focus — adjust distance",
    hold_steady: isAr ? "ممتاز! ثبّت إيدك... جاري التثبيت" : "Great! Hold steady — locking...",
    locked: isAr ? "تم التثبيت! جاري الالتقاط..." : "Locked! Capturing...",
  };

  const frameColor =
    scanState === "locked" || scanState === "hold_steady"
      ? "border-emerald-400 shadow-[0_0_28px_rgba(52,211,153,0.45)]"
      : scanState === "blurry" || scanState === "too_dark"
      ? "border-amber-400"
      : "border-rose-400/90";

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black/95 text-white">
      <div className="flex items-center justify-between px-4 py-3 bg-slate-950/90 border-b border-white/10">
        <div className="flex items-center gap-2 text-sm font-extrabold">
          <Focus className="w-4 h-4 text-rose-400" />
          <span>{isAr ? "ماسح الـ ECG الذكي (التقاط تلقائي)" : "Smart ECG Auto-Capture Scanner"}</span>
        </div>
        <div className="flex items-center gap-2">
          {torchSupported && (
            <button
              type="button"
              onClick={() => void toggleTorch()}
              className={`p-2 rounded-full border cursor-pointer ${
                torchOn ? "bg-amber-400/25 border-amber-400 text-amber-200" : "bg-white/10 border-white/20 text-white"
              }`}
              aria-label="Toggle Flash"
            >
              <Flashlight className="w-4 h-4" />
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              stopCamera();
              onClose();
            }}
            className="p-2 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 cursor-pointer"
            aria-label="Close Scanner"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="relative flex-1 flex items-center justify-center overflow-hidden">
        {errorMsg ? (
          <div className="max-w-sm text-center p-6 rounded-2xl bg-slate-900 border border-rose-500/40 text-sm text-rose-200">
            {errorMsg}
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            {/* QR-Style Guide Frame Overlay */}
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center p-4">
              <div
                className={`relative w-[90%] max-w-[640px] aspect-[16/10] rounded-2xl border-2 transition-all duration-200 ${frameColor}`}
              >
                <span className="absolute -top-1 -start-1 w-7 h-7 border-t-4 border-s-4 border-white rounded-tl-xl" />
                <span className="absolute -top-1 -end-1 w-7 h-7 border-t-4 border-e-4 border-white rounded-tr-xl" />
                <span className="absolute -bottom-1 -start-1 w-7 h-7 border-b-4 border-s-4 border-white rounded-bl-xl" />
                <span className="absolute -bottom-1 -end-1 w-7 h-7 border-b-4 border-e-4 border-white rounded-br-xl" />
              </div>

              <div className="mt-4 flex flex-col items-center gap-2">
                <div className="inline-flex items-center gap-2 rounded-full bg-black/75 border border-white/20 px-4 py-1.5 text-xs font-bold backdrop-blur-md">
                  {scanState === "locked" ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <Focus className="w-4 h-4 text-rose-400" />
                  )}
                  <span>{statusText[scanState]}</span>
                </div>
                <div className="w-48 h-1.5 rounded-full bg-white/15 overflow-hidden">
                  <div
                    className="h-full bg-emerald-400 transition-all duration-200"
                    style={{ width: `${lockProgress}%` }}
                  />
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="p-4 bg-slate-950/95 border-t border-white/10 flex items-center justify-center">
        <button
          type="button"
          onClick={captureCurrentFrame}
          disabled={Boolean(errorMsg)}
          className="cb-btn-primary inline-flex items-center gap-2 px-6 py-3 rounded-full text-sm font-extrabold cursor-pointer disabled:opacity-40"
        >
          <Camera className="w-4 h-4" />
          <span>{isAr ? "التقاط يدوي الآن" : "Capture Manually Now"}</span>
        </button>
      </div>
    </div>
  );
}
