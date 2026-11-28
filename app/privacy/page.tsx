import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy Policy — Code Blue: ECG & Sounds",
  description: "Privacy Policy and Medical Disclaimer for Code Blue: ECG & Sounds.",
};

export default function PrivacyPolicyPage() {
  return (
    <main className="min-h-screen w-full overflow-y-auto px-4 py-10 text-slate-100">
      <div className="mx-auto max-w-3xl rounded-3xl border border-white/15 bg-slate-950/80 p-6 sm:p-10 shadow-2xl backdrop-blur-xl space-y-6">
        <div className="flex items-center justify-between gap-4 border-b border-white/10 pb-4">
          <div>
            <h1 className="text-2xl font-black text-white">Privacy Policy</h1>
            <p className="text-xs text-rose-300 mt-1">Code Blue: ECG &amp; Sounds · Effective November 28, 2026</p>
          </div>
          <Link href="/" className="rounded-xl border border-white/15 bg-white/5 px-3.5 py-2 text-xs font-bold text-slate-200 hover:bg-white/10">
            Back to App
          </Link>
        </div>

        <section className="space-y-2 text-sm leading-relaxed text-slate-200">
          <h2 className="text-base font-extrabold text-white">1. No Storage of Personal Health Information (PHI)</h2>
          <p>
            Any ECG image or PDF you upload or scan is transmitted securely over HTTPS/TLS solely to perform real-time
            automated visual analysis and is processed ephemerally in memory and immediately discarded. We do not store,
            archive, sell, or share uploaded ECG images, patient identifiers, or clinical inputs on any server.
          </p>
        </section>

        <section className="space-y-2 text-sm leading-relaxed text-slate-200">
          <h2 className="text-base font-extrabold text-white">2. Permissions &amp; Local Preferences</h2>
          <p>
            Camera access is used strictly on-device when you open the Smart Camera ECG Scanner. Only non-personal UI
            settings (Arabic/English language and Dark/Light theme) are saved locally on your device. No account or login
            is required.
          </p>
        </section>

        <section className="space-y-2 text-sm leading-relaxed text-slate-200">
          <h2 className="text-base font-extrabold text-white">3. Medical Disclaimer</h2>
          <p>
            Code Blue: ECG &amp; Sounds is intended strictly for medical education, training, and clinical reference for
            healthcare students and professionals. It is not a certified medical device and must not be used as a
            substitute for independent clinical judgment or emergency care.
          </p>
        </section>

        <section className="space-y-2 text-sm leading-relaxed text-slate-200">
          <h2 className="text-base font-extrabold text-white">4. Contact</h2>
          <p>
            Developer: Dr. Mohamed Elsharkay (Code Blue Bank) · Email:{" "}
            <a href="mailto:mohama2468d@gmail.com" className="text-rose-300 underline">
              mohama2468d@gmail.com
            </a>
          </p>
        </section>
      </div>
    </main>
  );
}
