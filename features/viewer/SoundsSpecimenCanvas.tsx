'use client';

import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, ContactShadows, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { useHeartStore } from "@/stores/heartStore";
import { useBreathStore } from "@/stores/breathStore";
import { rhythm, RhythmPlayer } from "./rhythms";
import { BreathPacer } from "./sinusArrhythmia";
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

function StandaloneAudioAndMotionDriver({ mode }: { mode: "heart" | "breath" }) {
  const heartEnabled = useHeartStore((s) => s.enabled);
  const heartSound = useHeartStore((s) => s.sound);
  const heartRhythmId = useHeartStore((s) => s.rhythm);
  const breathEnabled = useBreathStore((s) => s.enabled);
  const breathSound = useBreathStore((s) => s.sound);
  const breathPattern = useBreathStore((s) => s.pattern);

  const playerRef = useRef<RhythmPlayer | null>(null);
  const pacerRef = useRef<BreathPacer>(new BreathPacer());
  const clockRef = useRef<number>(0);
  const realRef = useRef<number>(0);
  const restingRef = useRef<boolean>(true);
  const lastVfTickRef = useRef<number>(0);
  const breathClockRef = useRef<number>(0);

  // Reset cardiac cycle cleanly whenever rhythm or audio toggle changes
  useEffect(() => {
    playerRef.current = null;
    clockRef.current = 0;
    realRef.current = 0;
    restingRef.current = true;
    pacerRef.current.reset();
    HEART_TRACE.clock.reset();
    HEART_TRACE.player = null;
    lastVfTickRef.current = 0;
    if (!heartSound || heartRhythmId !== "asystole") {
      stopAsystoleTone();
    }
  }, [heartRhythmId, heartEnabled, heartSound, mode]);

  useEffect(() => {
    return () => {
      stopAsystoleTone();
      stopBreathSound();
      HEART_TRACE.player = null;
      HEART_TRACE.clock.reset();
    };
  }, []);

  useFrame((_, rawDelta) => {
    const dt = Math.min(rawDelta, 0.05);

    if (mode === "heart" && heartEnabled) {
      const definition = rhythm(heartRhythmId);
      const swing = BREATH_NOW.swing * (definition.followsBreath ?? 0);
      const before = clockRef.current;
      const after = before + pacerRef.current.advance(dt, swing, restingRef.current);
      clockRef.current = after;
      realRef.current += dt;
      const now = realRef.current;

      playerRef.current ??= new RhythmPlayer(definition);
      const beat = playerRef.current.advance(before, after);
      restingRef.current = beat.atria === 0 && beat.ventricles === 0;

      HEART_TRACE.player = playerRef.current;
      HEART_TRACE.now = now;
      HEART_TRACE.clock.push(now, after);

      if (heartSound && before > 0.02) {
        if (heartRhythmId === "ventricular_fibrillation") {
          stopAsystoleTone();
          if (now - lastVfTickRef.current >= 0.13) {
            lastVfTickRef.current = now;
            playVentricularFibrillationTick(0.85);
          }
        } else if (heartRhythmId === "asystole") {
          startAsystoleTone();
        } else {
          stopAsystoleTone();
          for (const level of beat.lub) playLub(level);
          for (const level of beat.dub) playDub(level);
          for (const murmur of beat.murmurs) playMurmur(murmur);
          for (const kind of beat.extras) playExtra(kind);
        }
      } else if (!heartSound) {
        stopAsystoleTone();
      }
    } else {
      stopAsystoleTone();
    }

    if (mode === "breath" && breathEnabled) {
      breathClockRef.current += dt;
      const pat = BREATH_PATTERNS[breathPattern] ?? BREATH_PATTERNS.quiet;
      const fill = breathFill(breathClockRef.current, pat);
      BREATH_NOW.running = true;
      BREATH_NOW.fill = fill;
      BREATH_NOW.inhaling = breathingIn(breathClockRef.current, pat);
      BREATH_NOW.swing = fill;
      updateBreathSoundFrame(breathClockRef.current, pat, breathSound);
    } else {
      BREATH_NOW.running = false;
    }
  });

  return null;
}

function SpecimenMesh({ url, mode }: { url: string; mode: "heart" | "breath" }) {
  const { scene } = useGLTF(url);
  const groupRef = useRef<THREE.Group>(null);
  const { normalized, baseScale } = useMemo(() => {
    const c = scene.clone(true);
    const box = new THREE.Box3().setFromObject(c);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    c.position.sub(center);
    const maxDim = Math.max(size.x, size.y, size.z, 0.001);
    const target = mode === "heart" ? 2.55 : 2.85;
    return { normalized: c, baseScale: target / maxDim };
  }, [scene, mode]);

  useFrame((state) => {
    if (!groupRef.current) return;
    if (mode === "heart") {
      const pulse = baseScale * (1 + Math.sin(state.clock.elapsedTime * 7.5) * 0.028);
      groupRef.current.scale.setScalar(pulse);
    } else {
      const f = BREATH_NOW.fill || 0;
      groupRef.current.scale.set(
        baseScale * (1 + f * 0.065),
        baseScale * (1 + f * 0.03),
        baseScale * (1 + f * 0.065),
      );
    }
  });

  return (
    <group ref={groupRef} position={[0, 0.05, 0]}>
      <primitive object={normalized} />
    </group>
  );
}

function ProceduralOrganFallback({ mode }: { mode: "heart" | "breath" }) {
  const ref = useRef<THREE.Group>(null);
  useFrame((state) => {
    if (!ref.current) return;
    ref.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.35) * 0.25;
    if (mode === "heart") {
      const s = 1 + Math.pow(Math.max(0, Math.sin(state.clock.elapsedTime * 4.2)), 6) * 0.08;
      ref.current.scale.setScalar(s);
    } else {
      const b = 1 + (BREATH_NOW.fill || (Math.sin(state.clock.elapsedTime * 1.4) * 0.5 + 0.5)) * 0.09;
      ref.current.scale.set(b, 1 + (b - 1) * 0.45, b);
    }
  });

  if (mode === "heart") {
    return (
      <group ref={ref}>
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[1.05, 48, 48]} />
          <meshStandardMaterial color="#b91c1c" roughness={0.35} metalness={0.08} />
        </mesh>
        <mesh position={[-0.35, 0.95, 0.05]} rotation={[0, 0, 0.25]}>
          <cylinderGeometry args={[0.24, 0.28, 1.1, 32]} />
          <meshStandardMaterial color="#dc2626" roughness={0.35} />
        </mesh>
        <mesh position={[0.45, 0.85, -0.1]} rotation={[0, 0, -0.35]}>
          <cylinderGeometry args={[0.22, 0.25, 1.0, 32]} />
          <meshStandardMaterial color="#2563eb" roughness={0.4} />
        </mesh>
      </group>
    );
  }

  return (
    <group ref={ref}>
      <mesh position={[-0.78, 0, 0]} scale={[0.72, 1.25, 0.65]}>
        <sphereGeometry args={[0.95, 40, 40]} />
        <meshStandardMaterial color="#e89cb0" roughness={0.48} />
      </mesh>
      <mesh position={[0.78, 0, 0]} scale={[0.72, 1.25, 0.65]}>
        <sphereGeometry args={[0.95, 40, 40]} />
        <meshStandardMaterial color="#e89cb0" roughness={0.48} />
      </mesh>
      <mesh position={[0, 1.15, 0]}>
        <cylinderGeometry args={[0.16, 0.18, 1.1, 24]} />
        <meshStandardMaterial color="#38bdf8" roughness={0.4} />
      </mesh>
    </group>
  );
}

export function SoundsSpecimenCanvas({ mode }: { mode: "heart" | "breath" }) {
  const [glbAvailable, setGlbAvailable] = useState(false);
  const url = mode === "heart" ? "/anatomy/heart.glb" : "/anatomy/lungs.glb";

  useEffect(() => {
    let active = true;
    fetch(url, { method: "HEAD" })
      .then((res) => {
        if (active) setGlbAvailable(res.ok);
      })
      .catch(() => {
        if (active) setGlbAvailable(false);
      });
    return () => {
      active = false;
    };
  }, [url]);

  return (
    <div className="relative w-full h-full min-h-[420px]">
      {!glbAvailable && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 px-3 py-1.5 rounded-full bg-slate-900/85 border border-cyan-500/30 text-[11px] text-cyan-200 backdrop-blur">
          3D Preview Mode — Copy <code className="font-mono text-cyan-400">{mode === "heart" ? "heart.glb" : "lungs.glb"}</code> into <code className="font-mono text-cyan-400">public/anatomy/</code>
        </div>
      )}
      <Canvas camera={{ position: [0, 0.08, 3.55], fov: 40 }} dpr={[1, 2]} gl={{ alpha: true, antialias: true }} style={{ background: "transparent" }}>
        <ambientLight intensity={0.65} />
        <directionalLight position={[4, 6, 5]} intensity={1.35} />
        <directionalLight position={[-4, -2, -4]} intensity={0.45} color="#38bdf8" />
        <Suspense fallback={<ProceduralOrganFallback mode={mode} />}>
          {glbAvailable ? <SpecimenMesh url={url} mode={mode} /> : <ProceduralOrganFallback mode={mode} />}
          <Environment preset="studio" />
        </Suspense>
        <ContactShadows position={[0, -1.7, 0]} opacity={0.35} scale={8} blur={2.4} />
        <OrbitControls enablePan={false} minDistance={2.2} maxDistance={7.5} />
        <StandaloneAudioAndMotionDriver mode={mode} />
      </Canvas>
    </div>
  );
}
