"use client";

import { HybridDiagnosisResult } from "@/lib/ecgHybridEngine";
import { getClinicalInvTtt } from "@/lib/ecgClinicalLibrary";

export interface ReportSlideOptions {
  result: HybridDiagnosisResult;
  isAr: boolean;
  ecgImageDataUrl?: string | null;
  measurementSource?: "device_ocr" | "auto_waveform" | "manual" | null;
}

// Verified ISO/IEC 18004 Version 3-M QR matrix for https://codeblue-sounds.vercel.app
const SITE_URL = "https://codeblue-sounds.vercel.app";
const SITE_QR_MATRIX: readonly string[] = [
  "#######....#.#..#..#..#######",
  "#.....#.##....###..#..#.....#",
  "#.###.#....####.#.....#.###.#",
  "#.###.#...##.###...#..#.###.#",
  "#.###.#.##..#.#..###..#.###.#",
  "#.....#....###....##..#.....#",
  "#######.#.#.#.#.#.#.#.#######",
  ".............##.##..#........",
  "#.#.#.#.......####.##...#..#.",
  "#..###..####..##.##.###..#..#",
  "#....###..#..#..##....###.###",
  "....#..##..#...#.#.####....#.",
  "###...#.#.###....#...##..#.##",
  ".#####..#.#..#..##..###..#..#",
  "..#...###...#.#.#....#..##.##",
  ".###....#.#..###.##.##...#.#.",
  "###.#.##..#...#..#.####..#.##",
  "...###.###.#..#.###.#.#..##.#",
  "#....###.#.###...##..###...##",
  ".#..#...###....#.#......##.#.",
  "#.#####.....#...##..#####....",
  "........###..#..##..#...#.###",
  "#######...##..#.#####.#.##.##",
  "#.....#...#####.###.#...##.##",
  "#.###.#.#.....####..#####...#",
  "#.###.#...#..##..#.....##.###",
  "#.###.#.##.##...#..#...###..#",
  "#.....#..##...#######...#..#.",
  "#######.####.#.###..###.#..##"
];

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  maxLines: number
): string[] {
  const words = text.replace(/\s+/g, " ").trim().split(" ");
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const test = current ? `${current} ${word}` : word;
    if (ctx.measureText(test).width <= maxWidth) {
      current = test;
    } else {
      if (current) lines.push(current);
      current = word;
      if (lines.length >= maxLines - 1) break;
    }
  }
  if (current && lines.length < maxLines) lines.push(current);
  if (lines.length === maxLines && words.join(" ").length > lines.join(" ").length) {
    lines[maxLines - 1] = lines[maxLines - 1].replace(/.{0,3}$/, "...");
  }
  return lines;
}

function drawCodeBlueLogo(ctx: CanvasRenderingContext2D, cx: number, cy: number) {
  // Burgundy rounded shield badge
  const s = 54;
  const x = cx - s / 2;
  const y = cy - s / 2;
  const grad = ctx.createLinearGradient(x, y, x + s, y + s);
  grad.addColorStop(0, "#881337");
  grad.addColorStop(0.5, "#9F1239");
  grad.addColorStop(1, "#BE123C");
  drawRoundedRect(ctx, x, y, s, s, 14);
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Crisp ECG QRS pulse inside badge
  ctx.save();
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(x + 9, cy + 2);
  ctx.lineTo(x + 17, cy + 2);
  ctx.lineTo(x + 21, cy - 13);
  ctx.lineTo(x + 26, cy + 14);
  ctx.lineTo(x + 31, cy - 6);
  ctx.lineTo(x + 36, cy + 2);
  ctx.lineTo(x + 45, cy + 2);
  ctx.stroke();
  ctx.restore();
}

function drawQrCode(ctx: CanvasRenderingContext2D, x: number, y: number, size: number) {
  const n = SITE_QR_MATRIX.length;
  const quiet = 2;
  const cell = size / (n + quiet * 2);
  drawRoundedRect(ctx, x, y, size, size, 10);
  ctx.fillStyle = "#FFFFFF";
  ctx.fill();
  ctx.fillStyle = "#0B1120";
  for (let r = 0; r < n; r++) {
    const row = SITE_QR_MATRIX[r];
    for (let c = 0; c < n; c++) {
      if (row[c] === "#") {
        ctx.fillRect(
          x + (c + quiet) * cell,
          y + (r + quiet) * cell,
          Math.ceil(cell),
          Math.ceil(cell)
        );
      }
    }
  }
}

function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

export async function downloadEcgReportSlide(opts: ReportSlideOptions): Promise<void> {
  const { result, isAr, ecgImageDataUrl, measurementSource } = opts;
  const W = 1600;
  const H = 900;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // 1. Aurora Night Sky Background
  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, "#070B14");
  bg.addColorStop(0.55, "#0F172A");
  bg.addColorStop(1, "#180B16");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Subtle Burgundy + Teal aurora glow
  const glow1 = ctx.createRadialGradient(220, 120, 10, 220, 120, 420);
  glow1.addColorStop(0, "rgba(190, 18, 60, 0.22)");
  glow1.addColorStop(1, "rgba(190, 18, 60, 0)");
  ctx.fillStyle = glow1;
  ctx.fillRect(0, 0, W, H);

  const glow2 = ctx.createRadialGradient(1380, 760, 10, 1380, 760, 460);
  glow2.addColorStop(0, "rgba(20, 184, 166, 0.14)");
  glow2.addColorStop(1, "rgba(20, 184, 166, 0)");
  ctx.fillStyle = glow2;
  ctx.fillRect(0, 0, W, H);

  // Outer Glass Frame
  drawRoundedRect(ctx, 28, 28, W - 56, H - 56, 28);
  ctx.fillStyle = "rgba(15, 23, 42, 0.72)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Top Burgundy Accent Strip
  const accent = ctx.createLinearGradient(28, 28, W - 28, 28);
  accent.addColorStop(0, "#881337");
  accent.addColorStop(0.5, "#BE123C");
  accent.addColorStop(1, "#14B8A6");
  drawRoundedRect(ctx, 28, 28, W - 56, 8, 4);
  ctx.fillStyle = accent;
  ctx.fill();

  // 2. Header: Logo with Site Name Underneath + Report Title + Date
  drawCodeBlueLogo(ctx, 98, 86);
  ctx.fillStyle = "#F8FAFC";
  ctx.font = "800 18px system-ui, -apple-system, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Code Blue ECG", 98, 134);

  ctx.textAlign = isAr ? "right" : "left";
  ctx.fillStyle = "#F1F5F9";
  ctx.font = "800 28px system-ui, -apple-system, Segoe UI, sans-serif";
  const titleX = isAr ? W - 64 : 184;
  ctx.fillText(
    isAr ? "تقرير تحليل تخطيط القلب الهجين" : "Hybrid 12-Lead ECG Clinical Summary",
    titleX,
    84
  );

  const nowStr = new Date().toLocaleString(isAr ? "ar-EG" : "en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  ctx.fillStyle = "#94A3B8";
  ctx.font = "600 15px system-ui, -apple-system, Segoe UI, sans-serif";
  ctx.fillText(
    isAr ? `تاريخ التقرير: ${nowStr}` : `Generated: ${nowStr}  •  ${SITE_URL}`,
    titleX,
    114
  );

  // Header divider
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(56, 152);
  ctx.lineTo(W - 56, 152);
  ctx.stroke();

  // 3. Left Column: ECG Image Preview + Measurements Card
  const leftX = 56;
  const leftW = 600;
  const imgBoxY = 176;
  const imgBoxH = 390;

  drawRoundedRect(ctx, leftX, imgBoxY, leftW, imgBoxH, 20);
  ctx.fillStyle = "rgba(2, 6, 23, 0.72)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  if (ecgImageDataUrl) {
    const loaded = await loadImage(ecgImageDataUrl);
    if (loaded) {
      ctx.save();
      drawRoundedRect(ctx, leftX + 10, imgBoxY + 10, leftW - 20, imgBoxH - 20, 14);
      ctx.clip();
      const scale = Math.min((leftW - 20) / loaded.width, (imgBoxH - 20) / loaded.height);
      const dw = loaded.width * scale;
      const dh = loaded.height * scale;
      ctx.drawImage(
        loaded,
        leftX + (leftW - dw) / 2,
        imgBoxY + (imgBoxH - dh) / 2,
        dw,
        dh
      );
      ctx.restore();
    }
  } else {
    ctx.fillStyle = "#64748B";
    ctx.textAlign = "center";
    ctx.font = "600 18px system-ui, sans-serif";
    ctx.fillText(
      isAr ? "لا توجد صورة ECG مرفقة" : "No ECG Image Attached",
      leftX + leftW / 2,
      imgBoxY + imgBoxH / 2
    );
  }

  // Measurements / Verification Box below ECG image
  const mBoxY = 584;
  const mBoxH = 128;
  drawRoundedRect(ctx, leftX, mBoxY, leftW, mBoxH, 18);
  ctx.fillStyle = "rgba(30, 41, 59, 0.52)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.stroke();

  ctx.textAlign = isAr ? "right" : "left";
  const mTextX = isAr ? leftX + leftW - 20 : leftX + 20;
  ctx.fillStyle = "#E2E8F0";
  ctx.font = "700 15px system-ui, sans-serif";
  ctx.fillText(
    isAr ? "حالة القياسات والمصدر" : "Measurement Verification Status",
    mTextX,
    mBoxY + 34
  );

  const srcNote =
    measurementSource === "device_ocr"
      ? isAr
        ? "القياسات مستخرجة من ترويسة جهاز الـ ECG المطبوعة (OCR)."
        : "Intervals extracted from printed ECG machine header (OCR)."
      : measurementSource === "manual"
      ? isAr
        ? "القياسات مُدخلة يدوياً بواسطة الطبيب."
        : "Intervals verified and entered manually by clinician."
      : isAr
      ? "لا يتم عرض قياسات PR/QRS/QT التقديرية بالبكسل — راجع المربعات يدوياً."
      : "Pixel-estimated PR/QRS/QT intervals are suppressed — verify calipers manually.";
  ctx.fillStyle = "#94A3B8";
  ctx.font = "500 14px system-ui, sans-serif";
  wrapText(ctx, srcNote, leftW - 40, 2).forEach((ln, idx) => {
    ctx.fillText(ln, mTextX, mBoxY + 66 + idx * 24);
  });

  // 4. Right Column: Confirmed Diagnosis + Co-existing + Inv / TTT
  const rightX = 684;
  const rightW = W - rightX - 56;
  const activeOnly = result.primaryFindings.filter((f) => !("isDemoted" in f && f.isDemoted));
  const primary = activeOnly[0] ?? result.primaryFindings[0];
  const isInsufficient = !primary || primary.code === "INSUFFICIENT_DATA";
  const coExisting = isInsufficient
    ? []
    : activeOnly.slice(1, 4).filter((f) => !("isConfirmed" in f) || f.isConfirmed);
  const urgency = result.finalUrgency || "MODERATE";

  // Diagnosis Hero Card
  const heroY = 176;
  const heroH = 188;
  drawRoundedRect(ctx, rightX, heroY, rightW, heroH, 20);
  ctx.fillStyle = isInsufficient
    ? "rgba(217, 119, 6, 0.22)"
    : urgency === "CRITICAL"
    ? "rgba(190, 18, 60, 0.26)"
    : urgency === "URGENT"
    ? "rgba(217, 119, 6, 0.20)"
    : "rgba(13, 148, 136, 0.20)";
  ctx.fill();
  ctx.strokeStyle = isInsufficient
    ? "rgba(251, 191, 36, 0.70)"
    : urgency === "CRITICAL"
    ? "rgba(251, 113, 133, 0.75)"
    : urgency === "URGENT"
    ? "rgba(251, 191, 36, 0.65)"
    : "rgba(45, 212, 191, 0.65)";
  ctx.lineWidth = 1.8;
  ctx.stroke();

  const rTextX = isAr ? rightX + rightW - 24 : rightX + 24;
  ctx.textAlign = isAr ? "right" : "left";

  // Urgency + Evidence Level pill text (NO raw uncalibrated %)
  const concLevel = primary && "concordanceLevel" in primary ? primary.concordanceLevel : "moderate";
  const evLabel = isInsufficient
    ? isAr
      ? "يقين غير كافٍ — راجع يدوياً"
      : "INSUFFICIENT CERTAINTY"
    : concLevel === "strong"
    ? isAr
      ? "مستوى الدليل: قوي"
      : "Evidence Level: HIGH"
    : concLevel === "moderate"
    ? isAr
      ? "مستوى الدليل: متوسط"
      : "Evidence Level: MODERATE"
    : isAr
    ? "مستوى الدليل: مبدئي"
    : "Evidence Level: LOW";

  ctx.fillStyle = isInsufficient
    ? "#FCD34D"
    : urgency === "CRITICAL"
    ? "#FDA4AF"
    : "#5EEAD4";
  ctx.font = "800 14px system-ui, sans-serif";
  ctx.fillText(`${urgency}  •  ${evLabel}`, rTextX, heroY + 34);

  // Primary Diagnosis Name
  ctx.fillStyle = "#FFFFFF";
  ctx.font = "800 28px system-ui, -apple-system, Segoe UI, sans-serif";
  const primaryTitle = primary
    ? isAr
      ? primary.name_ar
      : primary.name_en
    : isAr
    ? "لا يوجد تشخيص مؤكد"
    : "No Confirmed Diagnosis";
  wrapText(ctx, primaryTitle, rightW - 48, 2).forEach((ln, idx) => {
    ctx.fillText(ln, rTextX, heroY + 76 + idx * 36);
  });

  // Co-existing confirmed diagnoses OR mandatory insufficient safety warning
  if (isInsufficient) {
    ctx.fillStyle = "#FEF3C7";
    ctx.font = "700 15px system-ui, sans-serif";
    const warn = isAr
      ? "هذا لا يستبعد وجود حالة طارئة — راجع التخطيط يدوياً أو صعّد الحالة فوراً."
      : "This does NOT rule out an emergency — verify manually / escalate immediately.";
    ctx.fillText(warn, rTextX, heroY + 162);
  } else if (coExisting.length > 0) {
    const coNames = coExisting
      .slice(0, 3)
      .map((d) => (isAr ? d.name_ar : d.name_en))
      .join("  •  ");
    ctx.fillStyle = "#E2E8F0";
    ctx.font = "600 14px system-ui, sans-serif";
    const prefix = isAr ? "تشخيصات مصاحبة مؤكدة: " : "Co-existing Confirmed: ";
    wrapText(ctx, prefix + coNames, rightW - 48, 1).forEach((ln) => {
      ctx.fillText(ln, rTextX, heroY + 162);
    });
  }

  // Inv & TTT Split Cards
  const invTtt = primary
    ? getClinicalInvTtt(primary.code, primary.immediate_action_en, primary.immediate_action_ar, primary.name_en)
    : null;
  const splitY = 384;
  const splitH = 328;
  const halfW = (rightW - 18) / 2;

  const drawClinicalListCard = (
    x: number,
    title: string,
    accentColor: string,
    items: string[]
  ) => {
    drawRoundedRect(ctx, x, splitY, halfW, splitH, 18);
    ctx.fillStyle = "rgba(30, 41, 59, 0.55)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    const tx = isAr ? x + halfW - 20 : x + 20;
    ctx.textAlign = isAr ? "right" : "left";
    ctx.fillStyle = accentColor;
    ctx.font = "800 16px system-ui, sans-serif";
    ctx.fillText(title, tx, splitY + 34);

    ctx.fillStyle = "#E2E8F0";
    ctx.font = "500 14px system-ui, sans-serif";
    let curY = splitY + 66;
    for (const item of items.slice(0, 5)) {
      const wrapped = wrapText(ctx, `• ${item}`, halfW - 40, 2);
      for (const ln of wrapped) {
        if (curY > splitY + splitH - 18) break;
        ctx.fillText(ln, tx, curY);
        curY += 22;
      }
      curY += 6;
    }
  };

  const splitPoints = (raw: string): string[] =>
    raw
      .split(/[\n•;]+/)
      .map((s) => s.trim())
      .filter(Boolean);

  const invItems = invTtt
    ? splitPoints(isAr ? invTtt.inv_ar : invTtt.inv_en)
    : [isAr ? "تأكيد التخطيط بـ 12-lead ومقارنته بتخطيط سابق" : "Verify 12-lead ECG and compare with prior tracing"];
  const tttItems = invTtt
    ? splitPoints(isAr ? invTtt.ttt_ar : invTtt.ttt_en)
    : [isAr ? "تقييم سريري مباشر وتصعيد الحالة عند وجود أعراض خطورة" : "Direct clinical correlation and escalate if red-flag symptoms present"];

  drawClinicalListCard(
    rightX,
    isAr ? "الفحوصات المطلوبة (Inv)" : "Investigations (Inv)",
    "#38BDF8",
    invItems
  );
  drawClinicalListCard(
    rightX + halfW + 18,
    isAr ? "خطة التعامل والعلاج (TTT)" : "Management (TTT)",
    "#FDA4AF",
    tttItems
  );

  // 5. Footer Bar + Verified Scannable QR Code to https://codeblue-sounds.vercel.app
  const footY = 736;
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(56, footY);
  ctx.lineTo(W - 56, footY);
  ctx.stroke();

  const qrSize = 112;
  const qrX = W - 56 - qrSize;
  const qrY = footY + 14;
  drawQrCode(ctx, qrX, qrY, qrSize);

  ctx.textAlign = "right";
  ctx.fillStyle = "#F8FAFC";
  ctx.font = "700 14px system-ui, sans-serif";
  ctx.fillText(
    isAr ? "امسح الباركود لفتح الموقع" : "Scan QR to Open Code Blue ECG",
    qrX - 18,
    qrY + 46
  );
  ctx.fillStyle = "#38BDF8";
  ctx.font = "600 13px ui-monospace, SFMono-Regular, monospace";
  ctx.fillText(SITE_URL, qrX - 18, qrY + 70);

  ctx.textAlign = "left";
  ctx.fillStyle = "#94A3B8";
  ctx.font = "600 13px system-ui, sans-serif";
  ctx.fillText(
    "Educational & Clinical Decision Support Only — Not a standalone medical diagnosis.",
    56,
    qrY + 46
  );
  ctx.fillText(
    "للدعم السريري والتعليم الطبي فقط — لا يغني عن التقييم الطبي المباشر ومراجعة الطبيب المختص.",
    56,
    qrY + 72
  );

  // Trigger instant PNG download
  const ts = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 12);
  const link = document.createElement("a");
  link.download = `CodeBlue-ECG-Slide-${ts}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}
