import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";
import "./glass-theme.css";
import SkyBackground from "@/components/SkyBackground";

export const metadata: Metadata = {
  metadataBase: new URL("https://codeblue-sounds.vercel.app"),
  title: "Code Blue — ECG & Sounds",
  description: "Hybrid ECG Diagnostic Engine, Interactive 3D Heart Auscultation, Live ECG Rhythms, Murmurs, and Breath Sounds Lab.",
  openGraph: {
    title: "Code Blue — ECG & Sounds",
    description: "Hybrid ECG Diagnostic Engine, Interactive 3D Heart Auscultation, Live ECG Rhythms, Murmurs, and Breath Sounds Lab.",
    url: "https://codeblue-sounds.vercel.app",
    siteName: "Code Blue — ECG & Sounds",
    type: "website",
    images: [
      {
        url: "/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "Code Blue — Hybrid ECG Diagnostic Engine & 3D Auscultation Lab",
        type: "image/jpeg",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Code Blue — ECG & Sounds",
    description: "Hybrid ECG Diagnostic Engine, Interactive 3D Heart Auscultation, Live ECG Rhythms, Murmurs, and Breath Sounds Lab.",
    images: ["/og-image.jpg"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const Sky = SkyBackground as any;
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
      <body className="min-h-screen bg-transparent text-slate-100 antialiased overflow-hidden" suppressHydrationWarning>
        <Script id="theme-sync" strategy="beforeInteractive">{`
          (function () {
            try {
              var saved = localStorage.getItem('codeblue_theme');
              var theme = (saved === 'light' || saved === 'dark') ? saved : 'dark';
              var root = document.documentElement;
              root.dataset.theme = theme;
              if (theme === 'dark') root.classList.add('dark');
              else root.classList.remove('dark');
            } catch (e) {}
          })();
        `}</Script>
        <Sky governorate="القاهرة" showSwitcher={false} />
        <div className="cb-glass" style={{ position: "relative", zIndex: 1 }}>
          {children}
        </div>
      </body>
    </html>
  );
}
