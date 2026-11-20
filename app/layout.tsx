import type { Metadata } from "next";
import "./globals.css";
import "./glass-theme.css";

export const metadata: Metadata = {
  title: "codeblue-ecg&sounds",
  description: "Standalone interactive Heart Auscultation, Live ECG Rhythms, Murmurs, and Chest/Breath Sounds Simulator.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      data-theme="dark"
      className="dark"
      suppressHydrationWarning
      style={{
        "--font-inter": "'Inter', system-ui, -apple-system, sans-serif",
        "--font-cairo": "'Cairo', system-ui, -apple-system, sans-serif",
        "--font-orbitron": "'Orbitron', sans-serif",
      } as React.CSSProperties}
    >
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Cairo:wght@400..900&family=DM+Sans:wght@400..700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Inter:wght@400..900&family=Plus+Jakarta+Sans:wght@400..700&display=swap"
        />
      </head>
      <body className="min-h-screen bg-[#060913] text-slate-100 antialiased overflow-hidden" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
