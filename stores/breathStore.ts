'use client';

import { create } from "zustand";

import { readLocal, writeLocal } from "@/lib/localStore";
import {
  primeBreathSound,
  stopBreathSound,
  type BreathPatternId,
} from "@/features/viewer/breathing";

const BREATH_SOUND_KEY = "anatria3d.breath.sound.v1";

function storedBreathSound(): boolean {
  return readLocal(BREATH_SOUND_KEY) === "on";
}

interface BreathStore {
  enabled: boolean;
  sound: boolean;
  pattern: BreathPatternId;
  toggle: () => void;
  setSound: (on: boolean) => void;
  setPattern: (pattern: BreathPatternId) => void;
}

export const useBreathStore = create<BreathStore>()((set, get) => ({
  enabled: false,
  sound: storedBreathSound(),
  pattern: "quiet",
  toggle: () =>
    set((state) => {
      const next = !state.enabled;
      if (!next) stopBreathSound();
      return { enabled: next };
    }),
  setSound: (on) => {
    if (on === get().sound) return;
    writeLocal(BREATH_SOUND_KEY, on ? "on" : "off");
    if (on) primeBreathSound();
    else stopBreathSound();
    set({ sound: on });
  },
  setPattern: (pattern) => {
    stopBreathSound();
    set({ pattern });
  },
}));
