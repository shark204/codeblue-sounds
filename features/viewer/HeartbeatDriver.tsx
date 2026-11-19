'use client';

import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import type { ManifestOrgan } from "@/lib/schemas";
import { useHeartStore } from "@/stores/heartStore";
import { useSceneStore } from "@/stores/sceneStore";

import {
  BEAT_ATRIA,
  BEAT_BASE_ATRIA,
  BEAT_BASE_VENTRICLES,
  BEAT_GLOW,
  BEAT_LIGHT_ATRIA,
  BEAT_LIGHT_VENTRICLES,
  BEAT_REACH_ATRIA,
  BEAT_REACH_VENTRICLES,
  BEAT_VENTRICLES,
  BEATING,
  BEATING_VERSION,
  chamberCentres,
  chamberHolds,
  HEART_PROBE,
  SEAM_ATTRIBUTE,
  SQUEEZE,
  seamFreedom,
  seamGrid,
  type BeatingMesh,
  type ChamberPoints,
} from "./heartbeat";
import {
  FLOW_CHAMBER,
  FLOW_PATH_ATTRIBUTE,
  FLOW_PROBE,
  FLOW_SOURCES,
  FLOW_UNIFORMS,
  FLOWING,
  FLOWING_VERSION,
  PulseTrain,
  type FlowKind,
} from "./bloodFlow";
import { BREATH_NOW } from "./BreathDriver";
import { HEART_TRACE } from "./EcgStrip";
import { flowPaths, type PathMesh } from "./flowPaths";
import { LightGuard, prefersReducedMotion } from "./lightGuard";
import { playDub, playExtra, playLub, playMurmur, playVentricularFibrillationTick, startAsystoleTone, stopAsystoleTone } from "./heartSound";
import { rhythm, RhythmPlayer } from "./rhythms";
import { BreathPacer } from "./sinusArrhythmia";

const LONGEST_STEP_S = 0.1;

function pointsOf(entry: BeatingMesh): ChamberPoints {
  return {
    chamber: entry.chamber,
    vertices: entry.mesh.geometry.getAttribute("position"),
    matrixWorld: entry.mesh.matrixWorld,
  };
}

function measureSeams(): void {
  const started = performance.now();
  const walls: ChamberPoints[] = [];
  for (const entry of BEATING.values()) if (entry.wall) walls.push(pointsOf(entry));
  const grid = seamGrid(walls);
  for (const entry of BEATING.values()) {
    const free = seamFreedom(grid, pointsOf(entry));
    entry.mesh.geometry.setAttribute(SEAM_ATTRIBUTE, new THREE.BufferAttribute(free, 1));
  }
  HEART_PROBE.seamMs = performance.now() - started;
}

function measurePaths(centres: ReadonlyMap<string, THREE.Vector3>): void {
  const started = performance.now();
  let vertices = 0;
  for (const kind of Object.keys(FLOW_CHAMBER) as FlowKind[]) {
    const origin = centres.get(FLOW_CHAMBER[kind]);
    const entries = [...FLOWING.values()].filter((entry) => entry.kind === kind);
    if (!origin || entries.length === 0) continue;
    const meshes: PathMesh[] = entries.map((entry) => {
      entry.mesh.updateWorldMatrix(true, false);
      const position = entry.mesh.geometry.getAttribute("position");
      vertices += position.count;
      return { vertices: position, matrixWorld: entry.mesh.matrixWorld, seed: entry.seed };
    });
    const paths = flowPaths(meshes, origin);
    entries.forEach((entry, i) => {
      entry.mesh.geometry.setAttribute(FLOW_PATH_ATTRIBUTE, new THREE.BufferAttribute(paths[i]!, 1));
    });
  }
  FLOW_PROBE.pathMs = performance.now() - started;
  FLOW_PROBE.vertices = vertices;
}

const PATH_SETTLE_FRAMES = 20;

export function HeartbeatDriver({
  boxes,
  organs,
  revision,
}: {
  boxes: ReadonlyMap<string, THREE.Box3>;
  organs: readonly ManifestOrgan[];
  revision: number;
}) {
  // Subscribe via standard useEffect instead of useSyncExternalStore (`getSnapshot`) inside the R3F Canvas fiber reconciler
  const [heartSnap, setHeartSnap] = useState(() => {
    const st = useHeartStore.getState();
    return { enabled: st.enabled, sound: st.sound, rhythmId: st.rhythm };
  });

  useEffect(() => {
    setHeartSnap({
      enabled: useHeartStore.getState().enabled,
      sound: useHeartStore.getState().sound,
      rhythmId: useHeartStore.getState().rhythm,
    });
    return useHeartStore.subscribe((st) => {
      setHeartSnap((prev) =>
        prev.enabled === st.enabled && prev.sound === st.sound && prev.rhythmId === st.rhythm
          ? prev
          : { enabled: st.enabled, sound: st.sound, rhythmId: st.rhythm },
      );
    });
  }, []);

  const { enabled, sound, rhythmId } = heartSnap;

  const centres = useMemo(
    () => (enabled && boxes && organs ? chamberCentres(boxes, organs) : null),
    [enabled, boxes, organs, revision],
  );

  useEffect(() => {
    if (!enabled || !boxes || !organs) return;
    const holds = chamberHolds(boxes, organs);
    BEAT_BASE_ATRIA.value = holds.atria?.base ?? 0;
    BEAT_REACH_ATRIA.value = holds.atria?.reach ?? 0;
    BEAT_BASE_VENTRICLES.value = holds.ventricles?.base ?? 0;
    BEAT_REACH_VENTRICLES.value = holds.ventricles?.reach ?? 0;
  }, [enabled, boxes, organs, revision]);

  const clock = useRef(0);
  const real = useRef(0);
  const resting = useRef(true);
  const pacer = useMemo(() => new BreathPacer(), []);
  const measured = useRef(-1);
  const pathsMeasured = useRef(-1);
  const pathsSeen = useRef({ version: -1, frames: 0 });
  const player = useRef<RhythmPlayer | null>(null);
  const lastVfTick = useRef<number>(0);
  const pulses = useMemo(() => new PulseTrain(), []);
  const guards = useMemo(() => ({ atria: new LightGuard(), ventricles: new LightGuard() }), []);

  // A new rhythm or freshly enabled audio starts cleanly from the top of the cardiac cycle (zero erroneous first beat).
  useEffect(() => {
    player.current = null;
    clock.current = 0;
    real.current = 0;
    resting.current = true;
    pacer.reset();
    HEART_TRACE.clock.reset();
    pulses.reset();
    lastVfTick.current = 0;
    if (!sound || rhythmId !== 'asystole') stopAsystoleTone();
  }, [rhythmId, sound, pulses, pacer]);

  const inverse = useMemo(() => new THREE.Matrix4(), []);

  useEffect(() => {
    if (enabled) return;
    clock.current = 0;
    real.current = 0;
    resting.current = true;
    pacer.reset();
    HEART_TRACE.clock.reset();
    measured.current = -1;
    pathsMeasured.current = -1;
    player.current = null;
    BEAT_ATRIA.value = 0;
    BEAT_VENTRICLES.value = 0;
    BEAT_GLOW.value = 0;
    HEART_TRACE.player = null;
    BEAT_LIGHT_ATRIA.value = 0;
    BEAT_LIGHT_VENTRICLES.value = 0;
    guards.atria.reset();
    guards.ventricles.reset();
    pulses.reset();
    FLOW_UNIFORMS.uPulseStrength.value.fill(0);
    FLOW_UNIFORMS.uVenous.value = 0;
    stopAsystoleTone();
  }, [enabled, pulses, guards, pacer]);

  useFrame((_, delta) => {
    const hState = useHeartStore.getState();
    if (!hState.enabled || !centres) return;

    if (measured.current !== BEATING_VERSION.value) {
      measured.current = BEATING_VERSION.value;
      measureSeams();
    }

    const version = FLOWING_VERSION.value;
    if (version !== pathsMeasured.current && FLOWING.size > 0) {
      const seen = pathsSeen.current;
      if (seen.version !== version) {
        seen.version = version;
        seen.frames = 0;
      } else if (++seen.frames >= PATH_SETTLE_FRAMES) {
        pathsMeasured.current = version;
        measurePaths(centres);
      }
    }

    const step = Math.min(delta, LONGEST_STEP_S);
    const definition = rhythm(hState.rhythm);
    const swing = BREATH_NOW.swing * (definition.followsBreath ?? 0);
    const before = clock.current;
    const after = before + pacer.advance(step, swing, resting.current);
    clock.current = after;
    real.current += step;
    const now = real.current;

    player.current ??= new RhythmPlayer(definition);
    const beat = player.current.advance(before, after);
    resting.current = beat.atria === 0 && beat.ventricles === 0;
    HEART_TRACE.player = player.current;
    HEART_TRACE.now = now;
    HEART_TRACE.clock.push(now, after);
    BEAT_ATRIA.value = beat.atria;
    BEAT_VENTRICLES.value = beat.ventricles;

    for (const entry of BEATING.values()) {
      const centre = centres.get(entry.chamber);
      if (!centre) continue;
      entry.mesh.updateWorldMatrix(true, false);
      if (Math.abs(entry.mesh.matrixWorld.determinant()) > 1e-6) {
        const nextLocal = centre.clone().applyMatrix4(inverse.copy(entry.mesh.matrixWorld).invert());
        if (Number.isFinite(nextLocal.x) && Number.isFinite(nextLocal.y) && Number.isFinite(nextLocal.z)) {
          entry.centre.value.copy(nextLocal);
        }
      }
    }

    BEAT_GLOW.value = (useSceneStore.getState().systemOpacity.cardiovascular ?? 1) < 1 ? 1 : 0;
    const calm = prefersReducedMotion();
    BEAT_LIGHT_ATRIA.value = guards.atria.next(now, step, beat.atria / SQUEEZE.atria, calm);
    BEAT_LIGHT_VENTRICLES.value = guards.ventricles.next(now, step, beat.ventricles / SQUEEZE.ventricles, calm);
    FLOW_UNIFORMS.uFlowSteady.value = guards.ventricles.steady ? 1 : 0;

    for (const level of beat.lub) pulses.launch(now, level);
    pulses.advance(now, step, FLOW_UNIFORMS.uPulseAge.value, FLOW_UNIFORMS.uPulseStrength.value);
    FLOW_UNIFORMS.uVenous.value = pulses.venous;
    FLOW_UNIFORMS.uFlowTime.value = now;
    for (const kind of Object.keys(FLOW_CHAMBER) as FlowKind[]) {
      const centre = centres.get(FLOW_CHAMBER[kind]);
      if (centre) FLOW_SOURCES[kind].value.copy(centre);
    }

    if (hState.sound && before > 0.02) {
      if (hState.rhythm === "ventricular_fibrillation") {
        stopAsystoleTone();
        if (now - lastVfTick.current >= 0.13) {
          lastVfTick.current = now;
          playVentricularFibrillationTick(0.85);
        }
      } else if (hState.rhythm === "asystole") {
        startAsystoleTone();
      } else {
        stopAsystoleTone();
        for (const level of beat.lub) playLub(level);
        for (const level of beat.dub) playDub(level);
        for (const murmur of beat.murmurs) playMurmur(murmur);
        for (const kind of beat.extras) playExtra(kind);
      }
    } else {
      stopAsystoleTone();
    }
  });

  return null;
}
