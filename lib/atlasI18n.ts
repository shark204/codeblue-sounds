'use client';

import { useLanguageStore } from "@/stores/languageStore";
import type { AnatomicalSystem } from "./schemas";

/**
 * Smart Medical Arabic Localization Helper for Code Blue 3D Anatomy Atlas.
 *
 * Design Rule:
 * - Translates UI controls, clinical explanations, dissection instructions,
 *   and study tools into natural, idiomatic medical Arabic (non-literal).
 * - NEVER translates fixed anatomical structure names (`ta2_latin`, `name_en`,
 *   muscle names, bone names, nerve names, vessel names, or organ hierarchy nodes),
 *   preserving standard international medical nomenclature for exams (MRCS / USMLE).
 */

export function useAtlasLocale() {
  const locale = useLanguageStore((s) => s.locale);
  const toggleLocale = useLanguageStore((s) => s.toggleLocale);
  const setLocale = useLanguageStore((s) => s.setLocale);
  const isAr = locale === "ar";

  const at = <T,>(ar: T, en: T): T => (isAr ? ar : en);

  return {
    locale,
    isAr,
    at,
    toggleLocale,
    setLocale,
  };
}

export function isAtlasArabic(): boolean {
  try {
    return useLanguageStore.getState().locale === "ar";
  } catch {
    return true;
  }
}

export function atStatic<T>(ar: T, en: T): T {
  return isAtlasArabic() ? ar : en;
}

/**
 * Bilingual System Names: Preserves the English anatomical system term
 * alongside its natural medical Arabic classification.
 */
export const SYSTEM_TITLE_AR: Record<AnatomicalSystem, string> = {
  skeletal: "Skeletal · العظام والهيكل العظمي",
  articular: "Articular · المفاصل والأربطة",
  muscular: "Muscular · الجهاز العضلي (Muscles)",
  cardiovascular: "Cardiovascular · القلب والأوعية الدموية",
  nervous: "Nervous · الجهاز العصبي والأعصاب",
  lymphatic: "Lymphatic · الجهاز والعقد اللمفاوية",
  digestive: "Digestive · الجهاز الهضمي والقناة الصفراوية",
  respiratory: "Respiratory · الجهاز التنفسي والرئتان",
  renal: "Renal / Urinary · الكليتان والمسالك البولية",
  endocrine: "Endocrine · الغدد الصماء",
  reproductive: "Reproductive · الجهاز التناسلي والحوض",
  integumentary: "Integumentary · الجلد والثدي (Breast)",
  regional: "Regional · التشريح السطحي والمناطق",
};
