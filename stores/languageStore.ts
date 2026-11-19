'use client';
import { create } from 'zustand';

export type Locale = 'en' | 'ar';

interface LanguageState {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
}

export const useLanguageStore = create<LanguageState>((set) => ({
  locale: 'en',
  setLocale: (locale) => set({ locale }),
  toggleLocale: () => set((s) => ({ locale: s.locale === 'ar' ? 'en' : 'ar' })),
}));
