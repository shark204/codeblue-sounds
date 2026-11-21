import { NextRequest, NextResponse } from "next/server";
import { ECGInput, interpretECG } from "@/lib/ecgRules";

export const runtime = "nodejs";
export const maxDuration = 30;

const EXTRACTION_PROMPT = `You are a board-certified cardiologist and clinical electrocardiographer extracting structured measurements from a 12-lead ECG or rhythm strip image.
Standard calibration unless stated otherwise: paper speed = 25 mm/s (1 small box = 40 ms, 1 large box = 200 ms), voltage = 10 mm/mV (1 mm = 0.1 mV).

Analyze the uploaded ECG image carefully and return ONLY a single valid JSON object (no markdown fences, no commentary) with this exact schema:
{
  "extracted": {
    "hr_bpm": number | null,
    "pr_ms": number | null,
    "qrs_ms": number | null,
    "qt_ms": number | null,
    "axis_deg": number | null,
    "rhythm_regular": boolean,
    "p_waves_present": boolean,
    "rhythm_type": "" | "af" | "atrial_flutter" | "svt" | "vt" | "torsades_de_pointes" | "vf" | "asystole" | "mat" | "preexcited_af",
    "sawtooth_flutter_waves": boolean,
    "flutter_wave_rate_bpm": number | null,
    "fibrillatory_waves": boolean,
    "av_block_type": "" | "mobitz_1" | "mobitz_2" | "2:1" | "3rd",
    "av_dissociation": boolean,
    "qrs_morphology": "" | "rbbb" | "lbbb",
    "delta_wave": boolean,
    "shortest_rr_ms": number | null,
    "st_elevation_mm": {
      "I": number, "II": number, "III": number, "aVR": number, "aVL": number, "aVF": number,
      "V1": number, "V2": number, "V3": number, "V4": number, "V5": number, "V6": number
    },
    "st_morphology_concave": boolean,
    "pr_depression": boolean,
    "wellens_pattern": "" | "type_a" | "type_b",
    "t_wave_inversion_leads": string[],
    "tall_r_v1_v2": boolean,
    "brugada_pattern": "" | "type_1" | "type_2",
    "coved_ste_v1_v2": boolean,
    "saddleback_ste_v1_v2": boolean,
    "peaked_t_waves": boolean,
    "flattened_or_absent_p_hyperk": boolean,
    "sine_wave_pattern": boolean,
    "prominent_u_waves": boolean,
    "osborn_j_waves": boolean,
    "low_voltage": boolean,
    "electrical_alternans": boolean,
    "s1q3t3": boolean,
    "rv_strain_twi_v1_v4": boolean,
    "avr_terminal_r_mm": number | null,
    "s_v1_mm": number | null,
    "r_v5_mm": number | null,
    "r_v6_mm": number | null,
    "r_avl_mm": number | null,
    "s_v3_mm": number | null,
    "r_v1_mm": number | null,
    "pacing_spikes": boolean,
    "negative_p_qrs_lead_i": boolean
  },
  "vision_notes_ar": "ملخص بصري موجز بالعربية لما تم رصده في صورة رسم القلب (النظم، المحور، التغيرات الرئيسية)",
  "vision_notes_en": "Concise visual summary in English of what was observed on the ECG tracing"
}

Rules:
- In st_elevation_mm, include ONLY leads with noticeable ST shift (positive mm for elevation, negative mm for depression; omit leads at 0).
- Set boolean flags to true ONLY when clearly visible on the tracing.
- Be conservative and accurate; never fabricate findings.`;

function extractJsonObject(raw: string): Record<string, unknown> {
  const cleaned = raw.replace(/```json/gi, "").replace(/```/g, "").trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1) {
    throw new Error("Model did not return valid JSON");
  }
  return JSON.parse(cleaned.slice(start, end + 1));
}

async function callGeminiVision(apiKey: string, base64Data: string, mimeType: string) {
  const models = ["gemini-2.5-flash", "gemini-2.0-flash"];
  let lastError = "Gemini request failed";

  for (const model of models) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { text: EXTRACTION_PROMPT },
              { inline_data: { mime_type: mimeType, data: base64Data } },
            ],
          },
        ],
        generationConfig: {
          temperature: 0.1,
          responseMimeType: "application/json",
        },
      }),
    });

    if (!res.ok) {
      lastError = `Gemini (${model}) HTTP ${res.status}`;
      continue;
    }

    const data = await res.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (text) {
      return { parsed: extractJsonObject(text), provider: model };
    }
  }
  throw new Error(lastError);
}

async function callOpenAIVision(apiKey: string, base64Data: string, mimeType: string) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      temperature: 0.1,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: EXTRACTION_PROMPT },
            { type: "image_url", image_url: { url: `data:${mimeType};base64,${base64Data}` } },
          ],
        },
      ],
    }),
  });

  if (!res.ok) {
    throw new Error(`OpenAI Vision HTTP ${res.status}`);
  }
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error("Empty OpenAI Vision response");
  return { parsed: extractJsonObject(text), provider: "gpt-4o-mini" };
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const imageBase64: string = body?.imageBase64 || "";
    const mimeType: string = body?.mimeType || "image/png";
    const age = body?.age ?? null;
    const sex = body?.sex ?? null;

    if (!imageBase64) {
      return NextResponse.json(
        { error: "Missing ECG image payload.", error_ar: "يرجى رفع صورة رسم قلب أولاً." },
        { status: 400 }
      );
    }

    const cleanBase64 = imageBase64.includes(",") ? imageBase64.split(",")[1] : imageBase64;
    const geminiKey =
      process.env.GEMINI_API_KEY ||
      process.env.GOOGLE_API_KEY ||
      process.env.GOOGLE_GENERATIVE_AI_API_KEY ||
      "";
    const openaiKey = process.env.OPENAI_API_KEY || "";

    if (!geminiKey && !openaiKey) {
      return NextResponse.json(
        {
          error: "No Vision API key configured. Add GEMINI_API_KEY (or OPENAI_API_KEY) to Vercel Environment Variables.",
          error_ar: "لم يتم إعداد مفتاح الذكاء الاصطناعي بعد. أضف GEMINI_API_KEY في إعدادات Vercel (Environment Variables) ثم أعد النشر.",
        },
        { status: 503 }
      );
    }

    const visionResult = geminiKey
      ? await callGeminiVision(geminiKey, cleanBase64, mimeType)
      : await callOpenAIVision(openaiKey, cleanBase64, mimeType);

    const extractedRaw = (visionResult.parsed.extracted || visionResult.parsed) as ECGInput;
    const mergedInput: ECGInput = {
      ...extractedRaw,
      age: age ?? extractedRaw.age ?? 50,
      sex: sex || extractedRaw.sex || "male",
    };

    const interpretation = interpretECG(mergedInput);

    return NextResponse.json({
      ok: true,
      provider: visionResult.provider,
      extracted: mergedInput,
      interpretation,
      vision_notes_ar: String(visionResult.parsed.vision_notes_ar || ""),
      vision_notes_en: String(visionResult.parsed.vision_notes_en || ""),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown ECG Vision error";
    return NextResponse.json(
      {
        error: message,
        error_ar: `تعذر تحليل صورة رسم القلب تلقائياً: ${message}`,
      },
      { status: 500 }
    );
  }
}
