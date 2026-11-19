import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Code Blue Heart & Breath Sounds + ECG Lab",
  description: "Standalone interactive Heart Auscultation, Live ECG Rhythms, Murmurs, and Chest/Breath Sounds Simulator.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#060913] text-slate-100 antialiased overflow-hidden">
        {children}
      </body>
    </html>
  );
}
