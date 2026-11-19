'use client';

import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import type { ManifestOrgan } from "@/lib/schemas";
import { useBreathStore } from "@/stores/breathStore";
import { useSceneStore } from "@/stores/sceneStore";

import {
  BREATH_AXIS,
  BREATH_FILL,
  BREATH_FRAME,
  BREATH_PATTERNS,
  BREATH_ROLE,
  breathFill,
  breathingIn,
  breathRole,
  stopBreathSound,
  updateBreathSoundFrame,
} from "./breathing";
import { CARRY_VERSION, carryPlacement, measureCarry } from "./breathCarry";
import { beatChamber, isChamberWall } from "./heartbeat";
import { sinusPace } from "./sinusArrhythmia";

/** Seconds to come up to full depth, or to settle. */
const EASE = 1;

const CARRY_SETTLE_FRAMES = 20;
const CARRY_STILL = 1e-5;
const CARRY_MOVED = 0.002;

export const BREATH_NOW = { inhaling: false, running: false, swing: 0, fill: 0 };

type Measured = Pick<ManifestOrgan, "organ_id" | "system" | "name_en" | "ta2_latin" | "path">;

export function breathFrame(
  boxes: ReadonlyMap<string, THREE.Box3>,
  organs: Iterable<Measured>,
  sternumY: number,
): [number, number, number] {
  const lungs = new THREE.Box3();
  const heart = new THREE.Box3();
  for (const organ of organs) {
    const box = boxes.get(organ.organ_id);
    if (!box || box.isEmpty()) continue;
    if (organ.system === "respiratory" && breathRole(organ) === BREATH_ROLE.lung) lungs.union(box);
    else if (beatChamber(organ) !== null && isChamberWall(organ)) heart.union(box);
  }
  return [
    lungs.isEmpty() ? sternumY - 0.16 : lungs.min.y,
    lungs.isEmpty() ? sternumY + 0.14 : lungs.max.y,
    heart.isEmpty() ? sternumY + 0.04 : heart.max.y,
  ];
}

export function BreathDriver({
  boxes,
  bounds,
  revision,
}: {
  boxes: Map<string, THREE.Box3>;
  bounds: THREE.Box3 | null;
  revision: number;
}) {
  const axis = useMemo(() => {
    const sternum = boxes.get("body_of_sternum");
    if (sternum) {
      const centre = sternum.getCenter(new THREE.Vector3());
      return [centre.x, centre.y, sternum.max.z - 0.1];
    }
    if (bounds && !bounds.isEmpty()) {
      const size = bounds.getSize(new THREE.Vector3());
      const middle = bounds.getCenter(new THREE.Vector3());
      return [middle.x, bounds.min.y + size.y * 0.73, middle.z];
    }
    return [0, 1.32, 0];
  }, [boxes, bounds, revision]);
  useEffect(() => {
    BREATH_AXIS.value = axis;
  }, [axis]);

  useEffect(() => {
    const updateFrame = () => {
      const organs = useSceneStore.getState().organs;
      BREATH_FRAME.value = breathFrame(boxes, Object.values(organs), axis[1]!);
    };
    updateFrame();
    return useSceneStore.subscribe((st, prev) => {
      if (st.organs !== prev.organs) updateFrame();
    });
  }, [boxes, axis, revision]);

  const clock = useRef(0);
  const amplitude = useRef(0);
  const carried = useRef({ version: -1, revision: -1, placement: Number.NaN });
  const carryStill = useRef({ version: -1, placement: Number.NaN, frames: 0 });
  const depth = useRef(1);
  const sinusSwing = useRef(BREATH_PATTERNS.quiet.sinusSwing);

  useEffect(
    () => () => {
      BREATH_FILL.value = 0;
      BREATH_NOW.running = false;
      BREATH_NOW.swing = 0;
      BREATH_NOW.fill = 0;
      stopBreathSound();
    },
    [],
  );

  useFrame((_, delta) => {
    const { enabled, sound, pattern: id } = useBreathStore.getState();
    const pattern = BREATH_PATTERNS[id] ?? BREATH_PATTERNS.quiet;
    if (!enabled && amplitude.current === 0) {
      if (BREATH_NOW.running) {
        BREATH_FILL.value = 0;
        BREATH_NOW.running = false;
        BREATH_NOW.swing = 0;
        BREATH_NOW.fill = 0;
        clock.current = 0;
        stopBreathSound();
      }
      return;
    }

    const version = CARRY_VERSION.value;
    const placement = carryPlacement();
    const still = carryStill.current;
    const holding = version === still.version && Math.abs(placement - still.placement) < CARRY_STILL;
    still.frames = holding ? still.frames + 1 : 0;
    still.version = version;
    still.placement = placement;
    const last = carried.current;
    const changed =
      version !== last.version ||
      revision !== last.revision ||
      !(Math.abs(placement - last.placement) < CARRY_MOVED);
    if (changed && still.frames >= CARRY_SETTLE_FRAMES) {
      carried.current = { version, revision, placement };
      measureCarry();
    }

    const step = Math.min(delta, 0.05);
    clock.current += step;
    const fill = breathFill(clock.current, pattern);
    const inhaling = breathingIn(clock.current, pattern);

    if (enabled) {
      amplitude.current = Math.min(1, amplitude.current + step / EASE);
      updateBreathSoundFrame(clock.current, pattern, sound);
    } else if (!inhaling) {
      amplitude.current = Math.max(0, amplitude.current - step / EASE);
      if (fill < 0.02) amplitude.current = 0;
    }
    depth.current += (pattern.depth - depth.current) * Math.min(1, step * 1.5);
    sinusSwing.current += (pattern.sinusSwing - sinusSwing.current) * Math.min(1, step * 1.5);

    BREATH_FILL.value = fill * amplitude.current * depth.current;
    BREATH_NOW.inhaling = inhaling;
    BREATH_NOW.running = amplitude.current > 0;
    BREATH_NOW.fill = fill;
    BREATH_NOW.swing = amplitude.current * sinusSwing.current * sinusPace(clock.current, pattern);
  });

  return null;
}
