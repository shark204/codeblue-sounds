'use client';
import { create } from 'zustand';
import type { ManifestOrgan } from '@/lib/schemas';

interface SoundsSceneState {
  organs: Record<string, ManifestOrgan>;
  background: 'dark' | 'light' | 'studio';
  selectedId: string | null;
  hidden: Record<string, boolean>;
  select: (id: string | null) => void;
  setOrgans: (organs: Record<string, ManifestOrgan>) => void;
}

export const useSceneStore = create<SoundsSceneState>((set) => ({
  organs: {},
  background: 'dark',
  selectedId: null,
  hidden: {},
  select: (selectedId) => set({ selectedId }),
  setOrgans: (organs) => set({ organs }),
}));
