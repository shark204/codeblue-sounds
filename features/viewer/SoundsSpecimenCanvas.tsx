'use client';

import React, { Component, Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, ContactShadows, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { useHeartStore } from "@/stores/heartStore";
import { useBreathStore } from "@/stores/breathStore";
import { rhythm, RhythmPlayer } from "./rhythms";
import { SQUEEZE } from "./heartbeat";
import {
  playDub,
  playExtra,
  playLub,
  playMurmur,
  playVentricularFibrillationTick,
  startAsystoleTone,
  stopAsystoleTone,
} from "./heartSound";
import { HEART_TRACE } from "./EcgStrip";
import { BREATH_NOW } from "./BreathDriver";
import {
  BREATH_PATTERNS,
  breathFill,
  breathingIn,
  stopBreathSound,
  updateBreathSoundFrame,
} from "./breathing";

const HEART_GLB = "/anatomy/heart.glb";
const LUNGS_GLB = "/anatomy/lungs.glb";

// Preload both Atlas GLB models so tab switching is instant with zero placeholder flash
useGLTF.preload(HEART_GLB);
useGLTF.preload(LUNGS_GLB);

/** Live physiological contraction state shared directly between the rhythm driver and 3D mesh (zero React re-renders). */
const HEART_BEAT_NOW = { atria: 0, ventricles: 0, rhythmId: "normal" };

function StandaloneAudioAndMotionDriver({ mode }: { mode: "heart" | "breath" }) {
  const heartClockRef = useRef(0);
  const heartRealRef = useRef(0);
  const playerRef = useRef<RhythmPlayer | null>(null);
  const lastRhythmRef = useRef<string>("");
  const lastVfTickRef = useRef(0);

  const breathClockRef = useRef(0);
  const lastBreathPatternRef = useRef<string>("");

  useEffect(() => {
    return () => {
      stopAsystoleTone();
      stopBreathSound();
      HEART_TRACE.player = null;
      HEART_BEAT_NOW.atria = 0;
      HEART_BEAT_NOW.ventricles = 0;
      BREATH_NOW.fill = 0;
      BREATH_NOW.running = false;
    };
  }, []);

  useEffect(() => {
    if (mode !== "heart") {
      stopAsystoleTone();
      HEART_TRACE.player = null;
      HEART_BEAT_NOW.atria = 0;
      HEART_BEAT_NOW.ventricles = 0;
    }
    if (mode !== "breath") {
      stopBreathSound();
      BREATH_NOW.fill = 0;
      BREATH_NOW.running = false;
    }
  }, [mode]);

  useFrame((_, rawDelta) => {
    const dt = Math.min(rawDelta, 0.05);

    if (mode === "heart") {
      const hState = useHeartStore.getState();
      if (!hState.enabled) {
        stopAsystoleTone();
        HEART_BEAT_NOW.atria = 0;
        HEART_BEAT_NOW.ventricles = 0;
        return;
      }

      if (!playerRef.current || lastRhythmRef.current !== hState.rhythm) {
        const def = rhythm(hState.rhythm);
        playerRef.current = new RhythmPlayer(def);
        lastRhythmRef.current = hState.rhythm;
        heartClockRef.current = 0;
        heartRealRef.current = 0;
        lastVfTickRef.current = 0;
        HEART_TRACE.clock.reset();
        stopAsystoleTone();
      }

      const before = heartClockRef.current;
      const after = before + dt;
      heartClockRef.current = after;
      heartRealRef.current += dt;
      const now = heartRealRef.current;

      const beat = playerRef.current.advance(before, after);
      HEART_TRACE.player = playerRef.current;
      HEART_TRACE.now = now;
      HEART_TRACE.clock.push(now, after);

      HEART_BEAT_NOW.atria = beat.atria;
      HEART_BEAT_NOW.ventricles = beat.ventricles;
      HEART_BEAT_NOW.rhythmId = hState.rhythm;

      if (hState.sound && before > 0.02) {
        if (hState.rhythm === "ventricular_fibrillation") {
          stopAsystoleTone();
          if (now - lastVfTickRef.current >= 0.13) {
            lastVfTickRef.current = now;
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
      return;
    }

    // Breath mode
    const bState = useBreathStore.getState();
    const pattern = BREATH_PATTERNS[bState.pattern] ?? BREATH_PATTERNS.quiet;
    if (!bState.enabled) {
      if (BREATH_NOW.running) {
        BREATH_NOW.fill = 0;
        BREATH_NOW.running = false;
        breathClockRef.current = 0;
        stopBreathSound();
      }
      return;
    }

    if (lastBreathPatternRef.current !== bState.pattern) {
      lastBreathPatternRef.current = bState.pattern;
      breathClockRef.current = 0;
    }

    breathClockRef.current += dt;
    const fill = breathFill(breathClockRef.current, pattern);
    const inhaling = breathingIn(breathClockRef.current, pattern);

    BREATH_NOW.fill = fill * (pattern.depth ?? 1);
    BREATH_NOW.inhaling = inhaling;
    BREATH_NOW.running = true;

    try {
      updateBreathSoundFrame(breathClockRef.current, pattern, bState.sound);
    } catch {
      // ignore audio context init before first user gesture
    }
  });

  return null;
}

function SpecimenMesh({ url, mode }: { url: string; mode: "heart" | "breath" }) {
  const { scene } = useGLTF(url);
  const groupRef = useRef<THREE.Group>(null);

  // Clone once and auto-center/normalize scale so both heart.glb and lungs.glb frame cleanly at origin
  const normalized = useMemo(() => {
    const clone = scene.clone(true);
    const box = new THREE.Box3().setFromObject(clone);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.001);
    const targetSize = mode === "heart" ? 2.25 : 2.55;
    const scale = targetSize / maxDim;

    clone.scale.setScalar(scale);
    clone.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
    return clone;
  }, [scene, mode]);

  useFrame((state) => {
    if (!groupRef.current) return;

    if (mode === "heart") {
      const hs = useHeartStore.getState();
      if (!hs.enabled || hs.rhythm === "asystole") {
        groupRef.current.scale.set(1, 1, 1);
        return;
      }

      if (hs.rhythm === "ventricular_fibrillation") {
        const t = state.clock.elapsedTime;
        groupRef.current.scale.set(
          1 + Math.sin(t * 39) * 0.015 + Math.cos(t * 23) * 0.008,
          1 + Math.cos(t * 43) * 0.012,
          1 + Math.sin(t * 31) * 0.015,
        );
        return;
      }

      // Normalize real RhythmPlayer contractions (0..1) directly from HEART_BEAT_NOW
      const vNorm = Math.max(0, Math.min(1.4, HEART_BEAT_NOW.ventricles / Math.max(SQUEEZE.ventricles, 0.01)));
      const aNorm = Math.max(0, Math.min(1.4, HEART_BEAT_NOW.atria / Math.max(SQUEEZE.atria, 0.01)));
      const rId = String(hs.rhythm || "").toLowerCase();
      const isMurmur = rId.includes("stenosis") || rId.includes("regurg") || rId.includes("murmur");
      const thrill = isMurmur && vNorm > 0.2 ? Math.sin(state.clock.elapsedTime * 55) * 0.009 * vNorm : 0;

      // Realistic ventricular systolic inward squeeze + slight apex elongation + atrial kick
      const sx = 1 - vNorm * 0.065 + aNorm * 0.018 + thrill;
      const sy = 1 + vNorm * 0.028 - aNorm * 0.014;
      const sz = 1 - vNorm * 0.055 + aNorm * 0.016 + thrill;
      groupRef.current.scale.set(sx, sy, sz);
    } else {
      const fill = Math.max(0, Math.min(1.5, BREATH_NOW.fill || 0));
      const expX = 1 + fill * 0.072;
      const expY = 1 + fill * 0.038;
      const expZ = 1 + fill * 0.062;
      groupRef.current.scale.set(expX, expY, expZ);
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={normalized} />
    </group>
  );
}

class SpecimenErrorBoundary extends Component<
  { mode: "heart" | "breath"; children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { mode: "heart" | "breath"; children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidUpdate(prevProps: { mode: "heart" | "breath" }) {
    if (prevProps.mode !== this.props.mode && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (this.state.hasError) {
      return null;
    }
    return this.props.children;
  }
}

export function SoundsSpecimenCanvas({ mode }: { mode: "heart" | "breath" }) {
  const url = mode === "heart" ? HEART_GLB : LUNGS_GLB;

  return (
    <div className="relative w-full h-full min-h-[420px] bg-transparent">
      <Canvas
        camera={{ position: [0, 0.15, 4.3], fov: 40 }}
        dpr={[1, 1.5]}
        gl={{ alpha: true, antialias: true, powerPreference: "high-performance" }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.85} />
        <hemisphereLight args={["#ffffff", "#334155", 0.7]} />
        <directionalLight position={[4, 6, 5]} intensity={1.45} color="#fff7ed" />
        <directionalLight position={[-4, 2, -4]} intensity={0.75} color="#bae6fd" />
        <directionalLight position={[0, -3, 3]} intensity={0.4} color="#fda4af" />

        <SpecimenErrorBoundary mode={mode}>
          <Suspense fallback={null}>
            <SpecimenMesh key={mode} url={url} mode={mode} />
          </Suspense>
        </SpecimenErrorBoundary>

        <ContactShadows frames={1} position={[0, -1.65, 0]} opacity={0.28} scale={7} blur={2.2} />
        <OrbitControls enablePan={false} enableDamping dampingFactor={0.08} minDistance={2.2} maxDistance={7.5} />
        <StandaloneAudioAndMotionDriver mode={mode} />
      </Canvas>
    </div>
  );
}
