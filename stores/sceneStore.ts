'use client';
import { create } from 'zustand';
import type { ManifestOrgan } from '@/lib/schemas';

interface SoundsSceneState {
  organs: Record<string, ManifestOrgan>;
  background: 'dark' | 'light' | 'studio';
  selectedId: string | null;
  selectedOrganIds: string[];
  isolatedOrganIds: string[] | null;
  hidden: Record<string, boolean>;
  hiddenSystems: string[];
  systemOpacity: Record<string, number>;
  select: (id: string | null) => void;
  setOrgans: (organs: Record<string, ManifestOrgan>) => void;
  clearSelection: () => void;
  clearIsolation: () => void;
  hideAllSystems: () => void;
}

export const useSceneStore = create<SoundsSceneState>((set) => ({
  organs: {},
  background: 'dark',
  selectedId: null,
  selectedOrganIds: [],
  isolatedOrganIds: null,
  hidden: {},
  hiddenSystems: [],
  systemOpacity: { cardiovascular: 1, respiratory: 1 },
  select: (selectedId) => set({ selectedId }),
  setOrgans: (organs) => set({ organs }),
  clearSelection: () => set({ selectedId: null, selectedOrganIds: [] }),
  clearIsolation: () => set({ isolatedOrganIds: null }),
  hideAllSystems: () => set({ hiddenSystems: [] }),
}));
