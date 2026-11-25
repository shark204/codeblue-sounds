import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 45;

const DENSENET_BACKEND_URL = (
  process.env.ECG_BACKEND_URL ||
  process.env.NEXT_PUBLIC_ECG_BACKEND_URL ||
  "https://mohama2468d--codeblue-ecg-ai-fastapi-app.modal.run"
).replace(/\/+$/, "");

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const file = form.get("file");
    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ ok: false, error: "Missing ECG image or PDF file." }, { status: 400 });
    }

    const outForm = new FormData();
    const fileName = (file as File).name || "ecg.png";
    outForm.append("file", file, fileName);

    const resp = await fetch(`${DENSENET_BACKEND_URL}/analyze-ecg`, {
      method: "POST",
      body: outForm,
      signal: AbortSignal.timeout(40000),
    });

    if (!resp.ok) {
      const errTxt = (await resp.text()).slice(0, 300);
      return NextResponse.json(
        { ok: false, error: `DenseNet backend error (${resp.status}): ${errTxt}` },
        { status: 502 }
      );
    }

    const data = await resp.json();
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json(
      { ok: false, error: err instanceof Error ? err.message : "Failed to reach ECG AI backend." },
      { status: 502 }
    );
  }
}
