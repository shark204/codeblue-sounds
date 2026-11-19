'use client';

import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Environment, ContactShadows, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { useHeartStore } from "@/stores/heartStore";
import { useBreathStore } from "@/stores/breathStore";
import { rhythm, RhythmPlayer } from "./rhythms";
import { playDub, playExtra, playLub, playMurmur, playVentricularFibrillationTick, startAsystoleTone, stopAsystoleTone } from "./heartSound";
import { HEART_TRACE } from "./EcgStrip";
import { BREATH_NOW } from "./BreathDriver";
import { BREATH_PATTERNS, breathFill, breathingIn, stopBreathSound, updateBreathSoundFrame } from "./breathing";

function StandaloneAudioAndMotionDriver({ mode }: { mode: "heart" | "breath" }) {
  const heartEnabled = useHeartStore((s) => s.enabled);
  const heartSound = useHeartStore((s) => s.sound);
  const heartRhythmId = useHeartStore((s) => s.rhythm);
  const breathEnabled = useBreathStore((s) => s.enabled);
  const breathSound = useBreathStore((s) => s.sound);
  const breathPattern = useBreathStore((s) => s.pattern);

  const playerRef = useRef<RhythmPlayer | null>(null);
  const lastRhythmRef = useRef<string>("");
  const breathPhaseRef = useRef<number>(0);

  useEffect(() => {
    return () => {
      stopAsystoleTone();
      stopBreathSound();
    };
  }, []);

  useFrame((_, rawDelta) => {
    const dt = Math.min(rawDelta, 0.05);

    if (mode === "heart" && heartEnabled) {
      const spec = rhythm(heartRhythmId);
      if (!playerRef.current || lastRhythmRef.current !== heartRhythmId) {
        playerRef.current = new RhythmPlayer(spec);
        lastRhythmRef.current = heartRhythmId;
        stopAsystoleTone();
      }
      const step = playerRef.current.step(dt, 1);
      HEART_TRACE.push?.(step.ecg, dt);
      if (heartSound) {
        if (spec.id === "asystole") {
          startAsystoleTone();
        } else {
          stopAsystoleTone();
          if (step.lub) playLub(spec.s1Gain ?? 1);
          if (step.dub) playDub(spec.s2Gain ?? 1);
          if (step.extra) playExtra(step.extra);
          if (step.murmur) playMurmur(step.murmur);
          if (spec.id === "vfib" && Math.random() < dt * 8) playVentricularFibrillationTick();
        }
      } else {
        stopAsystoleTone();
      }
    } else {
      stopAsystoleTone();
    }

    if (mode === "breath" && breathEnabled) {
      breathPhaseRef.current = (breathPhaseRef.current + dt * 0.28) % 1;
      const p = breathPhaseRef.current;
      const expansion = p < 0.42 ? Math.sin((p / 0.42) * Math.PI * 0.5) : Math.cos(((p - 0.42) / 0.58) * Math.PI * 0.5);
      BREATH_NOW.fill = Math.max(0, expansion);
      BREATH_NOW.inhaling = p < 0.42;
      if (breathSound && true) {
        try {
          const pat = BREATH_PATTERNS[breathPattern] ?? BREATH_PATTERNS.quiet; updateBreathSoundFrame(p * 4, pat, breathSound);
        } catch {}
      }
    }
  });

  return null;
}

function SpecimenMesh({ url, mode }: { url: string; mode: "heart" | "breath" }) {
  const { scene } = useGLTF(url);
  const groupRef = useRef<THREE.Group>(null);
  const cloned = useMemo(() => scene.clone(true), [scene]);

  useFrame((state) => {
    if (!groupRef.current) return;
    if (mode === "heart") {
      const pulse = 1 + Math.sin(state.clock.elapsedTime * 7.5) * 0.025;
      groupRef.current.scale.setScalar(pulse);
    } else {
      const exp = 1 + (BREATH_NOW.fill || 0) * 0.055;
      groupRef.current.scale.set(exp, 1 + (BREATH_NOW.fill || 0) * 0.025, exp);
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={cloned} />
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
          3D Preview Mode â€” Run <code className="font-mono text-cyan-400">.\COPY_GLB_ASSETS.ps1</code> to load real {mode === "heart" ? "heart.glb" : "lungs.glb"}
        </div>
      )}
      <Canvas camera={{ position: [0, 0.3, 4.2], fov: 42 }} dpr={[1, 2]}>
        <color attach="background" args={["#060913"]} />
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

