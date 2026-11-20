"use client";

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

const LIVE_HEART_MOTION = {
  atria: 0,
  ventricles: 0,
  rhythmId: "sinus_normal",
  enabled: true,
};

const CHAMBER_UNIFORMS = {
  uAtria: { value: 0 },
  uVentricles: { value: 0 },
  uVfib: { value: 0 },
  uAfib: { value: 0 },
  uTime: { value: 0 },
};

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
      LIVE_HEART_MOTION.atria = beat.atria;
      LIVE_HEART_MOTION.ventricles = beat.ventricles;
      LIVE_HEART_MOTION.rhythmId = heartRhythmId;
      LIVE_HEART_MOTION.enabled = true;

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
      LIVE_HEART_MOTION.atria = 0;
      LIVE_HEART_MOTION.ventricles = 0;
      LIVE_HEART_MOTION.enabled = false;
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
    const target = mode === "heart" ? 2.0 : 2.2;

    if (mode === "heart") {
      const halfY = Math.max(size.y * 0.5, 0.001);
      const halfX = Math.max(size.x * 0.5, 0.001);
      const cx = center.x.toFixed(4);
      const cy = center.y.toFixed(4);
      const cz = center.z.toFixed(4);
      const hy = halfY.toFixed(4);
      const hx = halfX.toFixed(4);
      const radAmp = (maxDim * 0.42).toFixed(4);
      const lonAmp = (maxDim * 0.14).toFixed(4);

      c.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh || !mesh.material) return;
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        const patched = mats.map((m) => {
          const mat = m.clone();
          mat.onBeforeCompile = (shader) => {
            shader.uniforms.uAtria = CHAMBER_UNIFORMS.uAtria;
            shader.uniforms.uVentricles = CHAMBER_UNIFORMS.uVentricles;
            shader.uniforms.uVfib = CHAMBER_UNIFORMS.uVfib;
            shader.uniforms.uAfib = CHAMBER_UNIFORMS.uAfib;
            shader.uniforms.uTime = CHAMBER_UNIFORMS.uTime;
            shader.vertexShader =
              "uniform float uAtria;\nuniform float uVentricles;\nuniform float uVfib;\nuniform float uAfib;\nuniform float uTime;\n" +
              shader.vertexShader;
            shader.vertexShader = shader.vertexShader.replace(
              "#include <begin_vertex>",
              `#include <begin_vertex>
              {
                float ny = (position.y - ${cy}) / ${hy};
                float nx = (position.x - ${cx}) / ${hx};
                // 4 distinct anatomical cardiac chambers (RA, LA, RV, LV)
                float atrialMask = smoothstep(0.02, 0.42, ny) * (1.0 - smoothstep(0.72, 0.98, ny));
                float ventMask   = smoothstep(0.18, -0.35, ny);
                float rightSide  = smoothstep(0.12, -0.25, nx);
                float leftSide   = smoothstep(-0.12, 0.25, nx);

                float raSqueeze = uAtria * 0.105 * atrialMask * rightSide;
                float laSqueeze = uAtria * 0.095 * atrialMask * leftSide;
                float rvSqueeze = uVentricles * 0.105 * ventMask * rightSide;
                float lvSqueeze = uVentricles * 0.145 * ventMask * leftSide;

                float afibQuiver = uAfib * atrialMask * 0.018 * sin(uTime * 36.0 + position.x * 18.0 + position.z * 14.0);
                float vfibQuiver = uVfib * ventMask * 0.024 * sin(uTime * 44.0 + position.y * 22.0 + position.x * 19.0);

                float totalInward = (raSqueeze + laSqueeze + rvSqueeze + lvSqueeze) + afibQuiver + vfibQuiver;
                vec3 radialDir = normalize(vec3(position.x - ${cx}, (position.y - ${cy}) * 0.35, position.z - ${cz}) + vec3(0.0001));
                transformed -= radialDir * totalInward * ${radAmp};
                transformed.y += (rvSqueeze + lvSqueeze) * ${lonAmp};
              }`
            );
          };
          mat.needsUpdate = true;
          return mat;
        });
        mesh.material = Array.isArray(mesh.material) ? patched : patched[0]!;
      });
    }

    return { normalized: c, baseScale: target / maxDim };
  }, [scene, mode]);

  useFrame((state) => {
    if (!groupRef.current) return;
    if (mode === "heart") {
      CHAMBER_UNIFORMS.uTime.value = state.clock.elapsedTime;
      const rid = LIVE_HEART_MOTION.rhythmId;
      if (!LIVE_HEART_MOTION.enabled || rid === "asystole") {
        CHAMBER_UNIFORMS.uAtria.value = 0;
        CHAMBER_UNIFORMS.uVentricles.value = 0;
        CHAMBER_UNIFORMS.uVfib.value = 0;
        CHAMBER_UNIFORMS.uAfib.value = 0;
      } else {
        CHAMBER_UNIFORMS.uAtria.value = Math.min(1.4, Math.max(0, LIVE_HEART_MOTION.atria * 2.2));
        CHAMBER_UNIFORMS.uVentricles.value = Math.min(1.4, Math.max(0, LIVE_HEART_MOTION.ventricles * 2.2));
        CHAMBER_UNIFORMS.uVfib.value = rid === "ventricular_fibrillation" ? 1 : 0;
        CHAMBER_UNIFORMS.uAfib.value = rid === "atrial_fibrillation" ? 1 : 0;
      }
      groupRef.current.scale.setScalar(baseScale);
    } else {
      const f = BREATH_NOW.running ? (BREATH_NOW.fill || 0) : 0;
      groupRef.current.scale.set(
        baseScale * (1 + f * 0.085),
        baseScale * (1 + f * 0.045),
        baseScale * (1 + f * 0.085),
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
  });
  return (
    <group ref={ref}>
      <mesh>
        <sphereGeometry args={[0.9, 32, 32]} />
        <meshStandardMaterial color={mode === "heart" ? "#b91c1c" : "#e89cb0"} roughness={0.4} />
      </mesh>
    </group>
  );
}

export function SoundsSpecimenCanvas({ mode }: { mode: "heart" | "breath" }) {
  const [glbState, setGlbState] = useState<"checking" | "ready" | "missing">("checking");
  const url = mode === "heart" ? "/anatomy/heart.glb" : "/anatomy/lungs.glb";

  useEffect(() => {
    let active = true;
    setGlbState("checking");
    fetch(url, { method: "HEAD" })
      .then((res) => {
        if (!active) return;
        if (res.ok) {
          try { useGLTF.preload(url); } catch {}
          setGlbState("ready");
        } else {
          setGlbState("missing");
        }
      })
      .catch(() => {
        if (active) setGlbState("missing");
      });
    return () => {
      active = false;
    };
  }, [url]);

  return (
    <div className="relative w-full h-full min-h-[420px]">
      {glbState === "checking" && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="rounded-2xl border border-cyan-500/30 bg-[#0b1120]/80 px-4 py-2 text-xs font-semibold text-cyan-200 backdrop-blur-md">
            Loading 3D {mode === "heart" ? "Heart" : "Lungs"}...
          </div>
        </div>
      )}
      <Canvas camera={{ position: [0, 0.08, 3.55], fov: 40 }} dpr={[1, 2]} gl={{ alpha: true, antialias: true }} style={{ background: "transparent" }}>
        <ambientLight intensity={0.65} />
        <directionalLight position={[4, 6, 5]} intensity={1.35} />
        <directionalLight position={[-4, -2, -4]} intensity={0.45} color="#38bdf8" />
        <Suspense fallback={null}>
          {glbState === "ready" && <SpecimenMesh url={url} mode={mode} />}
          {glbState === "missing" && <ProceduralOrganFallback mode={mode} />}
          <Environment preset="studio" />
        </Suspense>
        <ContactShadows position={[0, -1.7, 0]} opacity={0.35} scale={8} blur={2.4} />
        <OrbitControls enablePan={false} minDistance={2.2} maxDistance={7.5} />
        <StandaloneAudioAndMotionDriver mode={mode} />
      </Canvas>
    </div>
  );
}
