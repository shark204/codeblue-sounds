'use client';
// components/SkyBackground.jsx
import { useUIStore } from '../stores/uiStore';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';

const useIsoLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect;

const CSS = `
.sky{position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden;background:#050a1c}
.sky-day,.sky-night{position:absolute;inset:0;transition:opacity 1.4s ease}
.sky-day{background:linear-gradient(180deg,#2a7ad0 0%,#6eb2ec 50%,#c3e1f7 100%);opacity:0}
.sky-night{background:radial-gradient(ellipse at 50% 125%,#16244f 0%,#0a1230 48%,#040818 100%);opacity:1}
.sky[data-mode="day"] .sky-day{opacity:1}
.sky[data-mode="day"] .sky-night{opacity:0}
.day-canvas,.stars,.rain{position:absolute;inset:0;width:100%;height:100%;display:block}
.lightning{position:absolute;inset:0;opacity:0;pointer-events:none}
.stars,.moon-wrap{transition:opacity 2.5s ease}
.sky[data-rain="1"] .stars{opacity:.18}
.sky[data-rain="1"] .moon-wrap{opacity:.3}
.moon-wrap{position:absolute;top:7%;left:9%;width:clamp(80px,10vw,140px);aspect-ratio:1}
.moon-wrap::before{content:"";position:absolute;inset:-160%;background:radial-gradient(circle closest-side,rgba(205,220,255,.15) 0,rgba(160,185,255,.05) 40%,transparent 100%)}
.moon{position:relative;width:100%;height:100%;display:block;transform:rotate(-22deg);filter:drop-shadow(0 0 10px rgba(255,248,225,.28))}
.sky-sound{position:fixed;left:14px;bottom:70px;width:40px;height:40px;padding:0;border-radius:50%;border:1px solid rgba(255,255,255,.3);background:rgba(10,20,45,.55);color:#fff;display:none;place-items:center;cursor:pointer;z-index:50;backdrop-filter:blur(6px)}
.sky-sound[data-show="1"]{display:none}
.sky-sound:focus-visible{outline:2px solid #7fb4ff;outline-offset:2px}
.sky-sound[data-on="1"] .off,.sky-sound[data-on="0"] .on{display:none}
.sky-sound[data-locked="1"]{animation:sky-pulse 1.6s ease-in-out infinite}
@keyframes sky-pulse{0%,100%{box-shadow:0 0 0 0 rgba(127,180,255,.55)}50%{box-shadow:0 0 0 10px rgba(127,180,255,0)}}
@media (prefers-reduced-motion:reduce){.sky-sound[data-locked="1"]{animation:none}}
.sky-switch{position:fixed;top:14px;left:14px;z-index:50;display:flex;gap:2px;padding:3px;border-radius:12px;background:rgba(10,20,45,.55);border:1px solid rgba(255,255,255,.25);backdrop-filter:blur(8px)}
.sky-switch button{display:flex;align-items:center;gap:6px;padding:7px 10px;border:0;border-radius:9px;background:transparent;color:#fff;font:inherit;font-size:13px;line-height:1;cursor:pointer}
.sky-switch button[aria-pressed="true"]{background:rgba(127,180,255,.38)}
.sky-switch button:focus-visible{outline:2px solid #7fb4ff;outline-offset:1px}
@media (max-width:520px){.sky-switch span{display:none}}
`;

const SKY_FRAG = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform vec2 R;
uniform float T;
uniform float W;
uniform float B;
uniform float STEPS;
uniform sampler2D N;
const float BASE=0.9;
const float TOP=1.9;
const vec3 SUN=vec3(0.580,0.600,0.550);

float noise3(vec3 x){
  vec3 p=floor(x);
  vec3 f=fract(x);
  f=f*f*(3.0-2.0*f);
  vec2 uv=p.xy+vec2(37.0,17.0)*p.z+f.xy;
  vec2 rg=texture2D(N,(uv+0.5)/256.0).rg;
  return mix(rg.x,rg.y,f.z);
}
float fbmH(vec3 p){
  float f=0.5*noise3(p);p*=2.02;
  f+=0.25*noise3(p);p*=2.03;
  f+=0.125*noise3(p);p*=2.01;
  f+=0.0625*noise3(p);
  return f/0.9375;
}
float fbmL(vec3 p){
  float f=0.5*noise3(p);p*=2.02;
  f+=0.25*noise3(p);
  return f/0.75;
}
float hash12(vec2 p){
  vec3 p3=fract(vec3(p.xyx)*0.1031);
  p3+=dot(p3,p3.yzx+33.33);
  return fract((p3.x+p3.y)*p3.z);
}
float cover(vec2 xz,vec2 wind){
  return noise3(vec3(xz*0.20+wind,5.1))*0.65+noise3(vec3(xz*0.43-wind*0.7,9.7))*0.35;
}
float shapeFn(vec3 p,float wet,vec3 wind,float n){
  float h=(p.y-BASE)/(TOP-BASE);
  float prof=smoothstep(0.0,0.10,h)*smoothstep(1.0,0.40,h);
  float c=cover(p.xz,wind.xz);
  float d=n*prof+(c-0.5)*0.85-(0.60-0.22*wet)+0.10;
  return clamp(d*5.0,0.0,1.0);
}
void main(){
  vec2 uv=gl_FragCoord.xy/R;
  float asp=R.x/R.y;
  vec2 s=vec2((uv.x-0.5)*asp,uv.y);
  vec3 rd=normalize(vec3(s.x*1.15,0.03+s.y*0.92,1.0));
  float wet=W;
  float cosA=max(dot(rd,SUN),0.0);

  vec3 zen=mix(vec3(0.12,0.38,0.78),vec3(0.36,0.42,0.50),wet*0.85);
  vec3 hor=mix(vec3(0.70,0.84,0.96),vec3(0.64,0.68,0.73),wet*0.85);
  vec3 sky=mix(hor,zen,pow(clamp(rd.y,0.0,1.0),0.5));
  float sunV=1.0-0.85*wet;
  sky+=vec3(1.0,0.92,0.75)*(pow(cosA,6.0)*0.25+pow(cosA,40.0)*0.45+pow(cosA,900.0)*3.0)*sunV;

  vec3 wind=vec3(T*0.050,T*0.004,T*0.020);
  float t0=BASE/rd.y;
  float t1=min(TOP/rd.y,46.0);
  vec3 acc=vec3(0.0);
  float Tr=1.0;
  if(t0<46.0){
    float dt=(t1-t0)/STEPS;
    float jit=hash12(gl_FragCoord.xy);
    float phase=0.55+1.9*pow(cosA,5.0)+3.0*pow(cosA,24.0);
    vec3 sunCol=vec3(1.0,0.93,0.80)*1.9*(1.0-0.72*wet);
    for(int i=0;i<64;i++){
      if(float(i)>=STEPS) break;
      float t=t0+(float(i)+jit)*dt;
      vec3 p=rd*t;
      float h=(p.y-BASE)/(TOP-BASE);
      vec3 q=p*2.6+wind;
      float lod=clamp(1.0-t*0.02,0.0,1.0);
      float n=mix(fbmL(q),fbmH(q),lod);
      float d=shapeFn(p,wet,wind,n);
      if(d>0.004){
        float ld=0.0;
        for(int j=0;j<4;j++){
          vec3 lp=p+SUN*(0.2*(float(j)+0.5));
          float lh=(lp.y-BASE)/(TOP-BASE);
          float ln=fbmL(lp*2.6+wind);
          ld+=shapeFn(lp,wet,wind,ln);
        }
        float Tl=exp(-ld*0.2*2.0*(1.0+0.9*wet));
        float dark=smoothstep(0.50,0.78,noise3(vec3(p.xz*0.17+wind.xz*0.4,13.3)));
        vec3 amb=mix(vec3(0.34,0.43,0.60),vec3(0.66,0.77,0.96),clamp(h,0.0,1.0));
        amb=mix(amb,vec3(dot(amb,vec3(0.33))),wet*0.6);
        amb*=mix(1.0,0.38,dark)*mix(0.50,1.0,smoothstep(0.0,0.6,h));
        vec3 light=sunCol*Tl*phase*mix(1.0,0.35,dark)+amb;
        float fog=exp(-t*0.055);
        light=mix(hor*0.95,light,fog);
        float sig=d*5.0*(1.0+0.6*wet);
        float tr=exp(-sig*abs(dt));
        acc+=Tr*light*(1.0-tr);
        Tr*=tr;
        if(Tr<0.02) break;
      }
    }
  }
  vec3 col=sky*Tr+acc;
  float haze=pow(1.0-clamp(rd.y*1.6,0.0,1.0),3.0)*0.55;
  col=mix(col,hor*mix(1.0,0.85,wet),haze*(0.7+0.3*wet));
  col=1.0-exp(-col*1.25);
  col*=1.0-0.06*wet;
  gl_FragColor=vec4(col,1.0);
}
`;

function createDaySky(canvas) {
  const gl = canvas.getContext('webgl', { antialias: false, alpha: false, powerPreference: 'default' });
  if (!gl) return null;
  const mk = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { console.warn(gl.getShaderInfoLog(s)); return null; }
    return s;
  };
  const vs = mk(gl.VERTEX_SHADER, 'attribute vec2 a;void main(){gl_Position=vec4(a,0.0,1.0);}');
  const fs = mk(gl.FRAGMENT_SHADER, SKY_FRAG);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'a');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const SZ = 256, data = new Uint8Array(SZ * SZ * 4);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const r = new Uint8Array(SZ * SZ);
  for (let i = 0; i < r.length; i++) r[i] = (rnd() * 256) | 0;
  for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) {
    const i = (y * SZ + x) * 4;
    data[i] = r[y * SZ + x];
    data[i + 1] = r[((y + 17) & 255) * SZ + ((x + 37) & 255)];
    data[i + 3] = 255;
  }
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, SZ, SZ, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  const U = (n) => gl.getUniformLocation(prog, n);
  const uR = U('R'), uT = U('T'), uW = U('W'), uB = U('B'), uS = U('STEPS');
  gl.uniform1i(U('N'), 0);
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const t0 = performance.now();
  let W = 0, B = 0, Wt = 0, Bt = 0, raf = 0, last = 0;
  let scale = 0.4, steps = 26, slow = 0, frames = 0;

  function resize() {
    canvas.width = Math.max(2, Math.round(canvas.clientWidth * scale));
    canvas.height = Math.max(2, Math.round(canvas.clientHeight * scale));
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  function draw(now) {
    W += (Wt - W) * 0.025; B += (Bt - B) * 0.025;
    gl.uniform2f(uR, canvas.width, canvas.height);
    gl.uniform1f(uT, 40 + (reduce ? 0 : (now - t0) / 1000));
    gl.uniform1f(uW, W); gl.uniform1f(uB, B); gl.uniform1f(uS, steps);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  function frame(now) {
    raf = requestAnimationFrame(frame);
    const dt = now - last;
    if (dt < 33) return;
    if (last && ++frames > 15) {
      slow = dt > 55 ? slow + 1 : Math.max(0, slow - 1);
      if (slow > 12 && scale > 0.2) { scale = Math.max(0.2, scale - 0.07); steps = Math.max(14, steps - 4); slow = 0; resize(); }
    }
    last = now; draw(now);
  }
  resize();
  window.addEventListener('resize', resize);
  if (reduce) { draw(t0); } else { raf = requestAnimationFrame(frame); }
  return {
    setRain(i) {
      Wt = i > 0 ? Math.min(1, 0.4 + i * 0.6) : 0;
      Bt = i > 0 ? Wt * (1 - 0.55 * i) : 0;
      if (reduce) { W = Wt; B = Bt; draw(t0); }
    },
    destroy() { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); },
  };
}

function drawMoon(canvas) {
  const S = 360, R = S / 2;
  canvas.width = canvas.height = S;
  const a = document.createElement('canvas'); a.width = a.height = S;
  const g = a.getContext('2d');
  let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  g.fillStyle = '#cbc8bf'; g.fillRect(0, 0, S, S);
  for (let i = 0; i < 900; i++) {
    const x = rnd() * S, y = rnd() * S, r = 3 + rnd() * 20;
    g.fillStyle = rnd() < 0.5 ? `rgba(255,255,248,${0.015 + rnd() * 0.03})` : `rgba(80,78,72,${0.015 + rnd() * 0.035})`;
    g.beginPath(); g.arc(x, y, r, 0, 6.283); g.fill();
  }
  const maria = [
    [.70, -.30, .14, .10], [.33, -.38, .17, .16], [.50, -.06, .22, .15], [.66, .20, .14, .15],
    [.42, .22, .08, .08], [-.28, -.42, .30, .24], [-.66, -.05, .26, .40], [-.22, .36, .20, .14],
    [-.55, .42, .10, .10], [0, -.78, .35, .07], [.08, -.18, .10, .09],
  ];
  for (const [mx, my, rx, ry] of maria) {
    g.save(); g.translate(R + mx * R, R + my * R); g.scale(rx * R, ry * R);
    const gr = g.createRadialGradient(0, 0, 0, 0, 0, 1);
    gr.addColorStop(0, 'rgba(70,72,78,.6)'); gr.addColorStop(0.55, 'rgba(76,78,84,.46)'); gr.addColorStop(1, 'rgba(70,72,78,0)');
    g.fillStyle = gr; g.beginPath(); g.arc(0, 0, 1, 0, 6.283); g.fill(); g.restore();
  }
  for (let i = 0; i < 110; i++) {
    const ang = rnd() * 6.283, rr = Math.sqrt(rnd()) * 0.93 * R;
    const x = R + Math.cos(ang) * rr, y = R + Math.sin(ang) * rr, r = 1.5 + Math.pow(rnd(), 2.2) * 15;
    g.fillStyle = 'rgba(40,40,44,.12)'; g.beginPath(); g.arc(x + r * 0.12, y + r * 0.12, r, 0, 6.283); g.fill();
    g.strokeStyle = 'rgba(255,255,250,.18)'; g.lineWidth = Math.max(0.8, r * 0.14);
    g.beginPath(); g.arc(x, y, r, 0, 6.283); g.stroke();
  }
  const tx = R - 0.1 * R, ty = R + 0.72 * R;
  const tg = g.createRadialGradient(tx, ty, 0, tx, ty, 0.2 * R);
  tg.addColorStop(0, 'rgba(255,255,250,.9)'); tg.addColorStop(1, 'rgba(255,255,250,0)');
  g.fillStyle = tg; g.beginPath(); g.arc(tx, ty, 0.2 * R, 0, 6.283); g.fill();
  g.strokeStyle = 'rgba(255,255,250,.12)'; g.lineWidth = 1.5;
  for (let i = 0; i < 14; i++) { const an = rnd() * 6.283, l = (0.25 + rnd() * 0.5) * R; g.beginPath(); g.moveTo(tx, ty); g.lineTo(tx + Math.cos(an) * l, ty + Math.sin(an) * l); g.stroke(); }

  const src = g.getImageData(0, 0, S, S).data;
  const ctx = canvas.getContext('2d');
  const out = ctx.createImageData(S, S), o = out.data;
  const sm = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
  for (let py = 0; py < S; py++) for (let px = 0; px < S; px++) {
    const x = (px + 0.5 - R) / R, y = (py + 0.5 - R) / R, rr = x * x + y * y, i = (py * S + px) * 4;
    if (rr >= 1.02) continue;
    const edge = Math.min(1, Math.max(0, (1 - Math.sqrt(rr)) * R));
    const nz = Math.sqrt(Math.max(0, 1 - rr));
    const lit = sm(-0.015, 0.34, x), limb = 0.78 + 0.22 * nz;
    o[i] = Math.min(255, src[i] * lit * limb * 1.12);
    o[i + 1] = Math.min(255, src[i + 1] * lit * 0.975 * limb * 1.12);
    o[i + 2] = Math.min(255, src[i + 2] * lit * 0.9 * limb * 1.12);
    o[i + 3] = edge * Math.pow(lit, 0.8) * 255;
  }
  ctx.putImageData(out, 0, 0);
}

function createNightSky(canvas, { cometEveryMs = 15 * 1000 } = {}) {
  const ctx = canvas.getContext('2d');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const TINTS = ['255,255,255', '200,220,255', '255,236,210'];
  let w = 0, h = 0, stars = [], raf = 0, nextGlow = 0, comet = null;

  const BURST_INTERVAL = 60 * 1000;
  const BURST_GAP = 15 * 1000;
  let nextBurstAt = 0;
  let burstRemaining = 0;
  let nextBurstCometAt = 0;

  /* ✅ قائمة منفصلة للمذنبات المجدولة (لما تدخل بانل)
     كل عنصر: { at: time, remaining: 1 أو 2, nextCometAt: time } */
  const scheduledBursts = [];
  let panelEntryCounter = 0;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round((w * h) / 3200);
    stars = Array.from({ length: n }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      r: Math.random() < 0.1 ? 1.5 : Math.random() * 0.9 + 0.35,
      a: Math.random() * 0.5 + 0.35, sp: Math.random() * 1.6 + 0.5,
      ph: Math.random() * 6.28, tint: TINTS[(Math.random() * 3) | 0],
      gs: 0, gd: 0,
    }));
  }

  function spawnComet(t) {
    const fromLeft = Math.random() < 0.4;
    const angleDeg = 22 + Math.random() * 40;
    const angleRad = angleDeg * Math.PI / 180;
    const travel = Math.max(w, h) * (1.1 + Math.random() * 0.6);
    const bright = Math.random() < 0.12;

    let x0, y0;
    if (fromLeft) {
      x0 = -80 + Math.random() * w * 0.55;
      y0 = -60 - Math.random() * 120;
    } else {
      x0 = w * 0.35 + Math.random() * (w * 0.75);
      y0 = -60 - Math.random() * 120;
    }

    const dirX = fromLeft ? Math.cos(angleRad) : -Math.cos(angleRad);
    const dirY = Math.sin(angleRad);
    const x1 = x0 + dirX * travel;
    const y1 = y0 + dirY * travel;

    const speed = bright ? 0.75 + Math.random() * 0.15 : 0.95 + Math.random() * 0.25;
    const dur = (2200 + Math.random() * 1800) / speed;

    comet = {
      t0: t, dur, x0, y0, x1, y1,
      ang: Math.atan2(dirY, dirX),
      bright,
      trailLen: bright ? 380 + Math.random() * 260 : 180 + Math.random() * 140,
      headSize: bright ? 3.4 : 2.4,
    };
  }

  function drawComet(t) {
    const p = (t - comet.t0) / comet.dur;
    if (p >= 1) { comet = null; return; }

    const e = 1 - Math.pow(1 - p, 1.6);
    const x = comet.x0 + (comet.x1 - comet.x0) * e;
    const y = comet.y0 + (comet.y1 - comet.y0) * e;

    const fadeIn = Math.min(1, p * 6);
    const fadeOut = Math.min(1, (1 - p) * 4);
    const fade = Math.min(fadeIn, fadeOut);

    const tailLen = comet.trailLen * (0.7 + 0.3 * Math.sin(Math.PI * p));
    const tx = x - Math.cos(comet.ang) * tailLen;
    const ty = y - Math.sin(comet.ang) * tailLen;

    ctx.save();
    ctx.globalAlpha = fade;
    ctx.lineCap = 'round';

    const layers = comet.bright
      ? [[12, 0.08], [6, 0.18], [2.6, 0.55]]
      : [[8, 0.08], [4, 0.20], [1.8, 0.60]];

    layers.forEach(([lw, al]) => {
      const g = ctx.createLinearGradient(x, y, tx, ty);
      g.addColorStop(0, `rgba(215,230,255,${al})`);
      g.addColorStop(0.4, `rgba(180,210,255,${al * 0.6})`);
      g.addColorStop(1, 'rgba(120,170,255,0)');
      ctx.strokeStyle = g;
      ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tx, ty); ctx.stroke();
    });

    const hs = comet.headSize;
    const hr = hs * (comet.bright ? 12 : 8);
    const hg = ctx.createRadialGradient(x, y, 0, x, y, hr);
    hg.addColorStop(0, 'rgba(255,255,255,1)');
    hg.addColorStop(0.2, `rgba(220,235,255,${comet.bright ? 0.85 : 0.7})`);
    hg.addColorStop(0.6, 'rgba(160,200,255,0.25)');
    hg.addColorStop(1, 'rgba(120,170,255,0)');
    ctx.fillStyle = hg;
    ctx.beginPath(); ctx.arc(x, y, hr, 0, 6.283); ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(x, y, hs, 0, 6.283); ctx.fill();

    if (comet.bright) {
      ctx.globalAlpha = fade * 0.6;
      for (let i = 0; i < 5; i++) {
        const fr = Math.random() * 0.6;
        const sx = x + (tx - x) * fr + (Math.random() - 0.5) * 8;
        const sy = y + (ty - y) * fr + (Math.random() - 0.5) * 8;
        ctx.fillStyle = `rgba(200,220,255,${0.5 + Math.random() * 0.4})`;
        ctx.beginPath(); ctx.arc(sx, sy, 0.7 + Math.random() * 1.2, 0, 6.283); ctx.fill();
      }
    }
    ctx.restore();
  }

  function frame(t) {
    ctx.clearRect(0, 0, w, h);

    // توهج النجوم
    if (!nextGlow) nextGlow = t + 2000;
    if (t >= nextGlow && !reduce) {
      const pool = stars.filter((s) => s.r >= 0.7 && s.x > 20 && s.x < w - 20 && s.y > 20 && s.y < h - 20);
      for (let k = 0; k < 2 && pool.length; k++) {
        const s = pool.splice((Math.random() * pool.length) | 0, 1)[0];
        s.gs = t; s.gd = 2400 + Math.random() * 1400;
      }
      nextGlow = t + 15000;
    }

    // النجوم
    for (const s of stars) {
      const tw = reduce ? 1 : 0.72 + 0.28 * Math.sin(t * 0.001 * s.sp + s.ph);
      let alpha = s.a * tw, rad = s.r;
      const k = s.gd && t - s.gs < s.gd ? Math.sin((Math.PI * (t - s.gs)) / s.gd) : 0;
      if (k > 0) {
        const hr = 7 + 16 * k;
        const g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, hr);
        g.addColorStop(0, `rgba(${s.tint},${0.75 * k})`);
        g.addColorStop(1, `rgba(${s.tint},0)`);
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(s.x, s.y, hr, 0, 6.283); ctx.fill();
        alpha = Math.min(1, alpha + k); rad = s.r + 0.9 * k;
      }
      ctx.fillStyle = `rgba(${s.tint},${alpha})`;
      ctx.beginPath(); ctx.arc(s.x, s.y, rad, 0, 6.283); ctx.fill();
    }

    if (!reduce) {
      // ═══ النمط الأساسي: كل 60 ثانية → مذنبين بينهم 15 ثانية ═══
      if (!nextBurstAt) nextBurstAt = t + 1500;
      if (burstRemaining === 0 && t >= nextBurstAt) {
        burstRemaining = 2;
        nextBurstCometAt = t;
        nextBurstAt = t + BURST_INTERVAL;
      }
      if (burstRemaining > 0 && t >= nextBurstCometAt) {
        spawnComet(t);
        burstRemaining--;
        if (burstRemaining > 0) nextBurstCometAt = t + BURST_GAP;
      }

      // ═══ المذنبات المجدولة (دخول بانل) — 1 أو 2 عشوائياً ═══
      for (let i = scheduledBursts.length - 1; i >= 0; i--) {
        const b = scheduledBursts[i];
        if (b.remaining > 0 && t >= b.nextCometAt) {
          spawnComet(t);
          b.remaining--;
          if (b.remaining > 0) b.nextCometAt = t + BURST_GAP;
        }
        if (b.remaining === 0) scheduledBursts.splice(i, 1);
      }
    }

    if (comet) drawComet(t);
    raf = requestAnimationFrame(frame);
  }

  resize();
  window.addEventListener('resize', resize);
  raf = requestAnimationFrame(frame);

  return {
    // ✅ دخول بانل: كل تالت مرة مفيش حاجة، والباقي 1 أو 2 عشوائياً
    triggerAfter(ms) {
      panelEntryCounter++;
      if (panelEntryCounter % 3 === 0) return;
      const count = Math.random() < 0.5 ? 1 : 2;
      scheduledBursts.push({
        remaining: count,
        nextCometAt: performance.now() + ms,
      });
    },
    destroy() { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); },
  };
}

function skyIsWinter(lat, months) {
  const m = new Date().getMonth() + 1;
  const mm = lat < 0 ? ((m + 5) % 12) + 1 : m;
  return (months || [11, 12, 1, 2, 3]).includes(mm);
}
function skyRainFromCode(code, precip) {
  let i = 0, storm = false;
  if (code >= 51 && code <= 57) i = 0.25;
  else if (code === 61) i = 0.35;
  else if (code === 63 || code === 66) i = 0.6;
  else if (code === 65 || code === 67) i = 0.9;
  else if (code === 80) i = 0.4;
  else if (code === 81) i = 0.65;
  else if (code === 82) i = 0.95;
  else if (code >= 95 && code <= 99) { i = 0.8; storm = true; }
  i = Math.max(i, Math.min(1, (precip || 0) / 6));
  return { intensity: i < 0.12 ? 0 : i, storm };
}
function skyLocation({ lat, lon, askLocation }) {
  try {
    const c = JSON.parse(localStorage.getItem('sky:loc') || 'null');
    if (c && Date.now() - c.ts < 7 * 864e5) return Promise.resolve(c);
  } catch (e) {}
  if (!askLocation || !navigator.geolocation) return Promise.resolve({ lat, lon });
  return new Promise((res) => navigator.geolocation.getCurrentPosition(
    (p) => {
      const v = { lat: p.coords.latitude, lon: p.coords.longitude, ts: Date.now() };
      try { localStorage.setItem('sky:loc', JSON.stringify(v)); } catch (e) {}
      res(v);
    },
    () => res({ lat, lon }),
    { timeout: 8000, maximumAge: 864e5 }
  ));
}
function createWeather({ lat = 30.0444, lon = 31.2357, askLocation = true, winterOnly = true, winterMonths, onChange }) {
  let stopped = false, timer = 0;
  async function tick() {
    try {
      const loc = await skyLocation({ lat, lon, askLocation });
      if (winterOnly && !skyIsWinter(loc.lat, winterMonths)) {
        if (!stopped) onChange({ intensity: 0, storm: false });
      } else {
        let cur = null;
        try {
          const c = JSON.parse(localStorage.getItem('sky:wx') || 'null');
          if (c && Date.now() - c.ts < 20 * 60 * 1000 && Math.abs(c.lat - loc.lat) < 0.2 && Math.abs(c.lon - loc.lon) < 0.2) cur = c.cur;
        } catch (e) {}
        if (!cur) {
          const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.lat.toFixed(2)}&longitude=${loc.lon.toFixed(2)}&current=weather_code,precipitation&timezone=auto`);
          cur = (await r.json()).current;
          try { localStorage.setItem('sky:wx', JSON.stringify({ ts: Date.now(), lat: loc.lat, lon: loc.lon, cur })); } catch (e) {}
        }
        if (!stopped && cur) onChange(skyRainFromCode(cur.weather_code, cur.precipitation));
      }
    } catch (e) {}
    if (!stopped) timer = setTimeout(tick, 20 * 60 * 1000);
  }
  tick();
  return { destroy() { stopped = true; clearTimeout(timer); } };
}

function createRain(canvas) {
  const ctx = canvas.getContext('2d');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let w = 0, h = 0, drops = [], raf = 0, cur = 0, target = 0, night = false;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = canvas.clientWidth; h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(900, Math.round((w * h) / 3600));
    drops = Array.from({ length: n }, () => ({ x: Math.random() * w, y: Math.random() * h, z: 0.45 + Math.random() * 0.55, th: Math.random() }));
  }
  const WIND = 0.2, BUCKETS = [[0, 0.65], [0.65, 0.82], [0.82, 1.01]];
  function frame() {
    cur += (target - cur) * 0.03;
    if (target === 0 && cur < 0.01) { cur = 0; ctx.clearRect(0, 0, w, h); raf = 0; return; }
    ctx.clearRect(0, 0, w, h);
    ctx.lineCap = 'round';
    for (const [lo, hi] of BUCKETS) {
      const zc = (lo + Math.min(hi, 1)) / 2;
      const a = (0.1 + 0.28 * zc) * (0.4 + 0.6 * cur) * 0.55;
      ctx.strokeStyle = night ? `rgba(170,195,240,${a})` : `rgba(160,178,205,${a * 1.25})`;
      ctx.lineWidth = 0.7 + zc * 0.7;
      ctx.beginPath();
      for (const d of drops) {
        if (d.z < lo || d.z >= hi || d.th > cur) continue;
        const len = 14 + d.z * 26;
        ctx.moveTo(d.x, d.y); ctx.lineTo(d.x - WIND * len, d.y - len);
      }
      ctx.stroke();
    }
    for (const d of drops) {
      const sp = (11 + d.z * 16) * (0.8 + cur * 0.5);
      d.y += sp; d.x += sp * WIND;
      if (d.y > h + 30) { d.y = -30 - Math.random() * 120; d.x = Math.random() * (w + 200) - 100; }
      if (d.x > w + 60) d.x -= w + 120;
    }
    raf = requestAnimationFrame(frame);
  }
  resize();
  window.addEventListener('resize', resize);
  return {
    set(i, isNight) {
      target = i; night = isNight;
      if (!reduce && !raf && (target > 0 || cur > 0)) raf = requestAnimationFrame(frame);
    },
    destroy() { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); },
  };
}

function createRainSound({ volume = 1 } = {}) {
  let ctx = null, master = null, comp = null, buf = null, intensity = 0, enabled = true, tickTimer = 0;
  let vol = volume;
  const MAXG = 0.007;
  const EVENTS = ['pointerdown', 'pointerup', 'click', 'keydown', 'touchend'];
  function tickLoop() {
    if (!ctx) return;
    if (enabled && intensity > 0 && ctx.state === 'running') {
      const n = ctx.createBufferSource(); n.buffer = buf;
      const f = ctx.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 2500 + Math.random() * 3500; f.Q.value = 2;
      const g = ctx.createGain(), t = ctx.currentTime;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.05 + 0.2 * Math.random(), t + 0.002);
      g.gain.exponentialRampToValueAtTime(0.0005, t + 0.05);
      n.connect(f); f.connect(g); g.connect(master); n.start(t, Math.random() * 2, 0.06);
    }
    tickTimer = setTimeout(tickLoop, 40 + Math.random() * (260 - 180 * intensity));
  }
  function build() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ctx = new AC();
    master = ctx.createGain(); master.gain.value = 0;
    comp = ctx.createDynamicsCompressor();
    master.connect(comp); comp.connect(ctx.destination);
    buf = ctx.createBuffer(1, ctx.sampleRate * 3, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    const mk = (offset) => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(0, offset); return s; };
    const body = mk(0), hp = ctx.createBiquadFilter(), lp = ctx.createBiquadFilter(), gA = ctx.createGain();
    hp.type = 'highpass'; hp.frequency.value = 700; lp.type = 'lowpass'; lp.frequency.value = 5500; gA.gain.value = 0.55;
    body.connect(hp); hp.connect(lp); lp.connect(gA); gA.connect(master);
    const hiss = mk(1.3), bp = ctx.createBiquadFilter(), gB = ctx.createGain();
    bp.type = 'bandpass'; bp.frequency.value = 4200; bp.Q.value = 0.4; gB.gain.value = 0.25;
    hiss.connect(bp); bp.connect(gB); gB.connect(master);
    const lfo = ctx.createOscillator(), lg = ctx.createGain();
    lfo.frequency.value = 0.13; lg.gain.value = 0.08; lfo.connect(lg); lg.connect(gA.gain); lfo.start();
    tickLoop();
  }
  function apply() {
    if (!ctx) return;
    master.gain.setTargetAtTime(enabled && intensity > 0 ? MAXG * vol * (0.35 + 0.65 * intensity) : 0, ctx.currentTime, 1);
  }
  function unlock() {
    if (!ctx) build();
    if (!ctx) return;
    const done = () => { if (ctx && ctx.state === 'running') disarm(); apply(); };
    if (ctx.state === 'suspended') ctx.resume().then(done, () => {}); else done();
  }
  function disarm() { EVENTS.forEach((ev) => window.removeEventListener(ev, unlock, true)); }
  EVENTS.forEach((ev) => window.addEventListener(ev, unlock, true));
  return {
    unlock,
    isRunning: () => !!ctx && ctx.state === 'running',
    setIntensity(i) { intensity = i; apply(); },
    setEnabled(e) { enabled = e; apply(); },
    setVolume(v) { vol = v; apply(); },
    thunder(delay = 3, power = 0.3) {
      setTimeout(() => {
        if (!ctx || !enabled || intensity <= 0 || ctx.state !== 'running') return;
        const s = ctx.createBufferSource(); s.buffer = buf;
        const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 140;
        const g = ctx.createGain(), t = ctx.currentTime;
        g.gain.setValueAtTime(0, t);
        g.gain.linearRampToValueAtTime(0.7 * power, t + 0.7);
        g.gain.exponentialRampToValueAtTime(0.001, t + 4.5);
        s.connect(lp); lp.connect(g); g.connect(comp); s.start(t, Math.random() * 1.5, 5);
      }, delay * 1000);
    },
    destroy() { clearTimeout(tickTimer); disarm(); if (ctx) ctx.close(); ctx = null; },
  };
}

function createLightning(el, { storm = false, onFlash } = {}) {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let timer = 0, stopped = false;
  function flash() {
    if (reduce || !el) return;
    const x = 10 + Math.random() * 80, y = 55 + Math.random() * 30;
    el.style.background =
      `radial-gradient(ellipse 38% 28% at ${x}% ${y}%, rgba(205,220,255,.55), rgba(160,185,255,.18) 45%, transparent 75%),` +
      `linear-gradient(180deg, transparent 40%, rgba(150,175,235,.10))`;
    const peak = 0.5 + Math.random() * 0.3;
    el.animate(
      [{ opacity: 0 }, { opacity: peak, offset: 0.07 }, { opacity: 0.12, offset: 0.18 }, { opacity: peak * 0.7, offset: 0.28 }, { opacity: 0 }],
      { duration: 650 + Math.random() * 350, easing: 'ease-out' }
    );
    if (onFlash) onFlash();
  }
  function next() {
    if (stopped) return;
    timer = setTimeout(() => { flash(); next(); }, (storm ? 6000 : 14000) + Math.random() * (storm ? 14000 : 30000));
  }
  next();
  return { flash, destroy() { stopped = true; clearTimeout(timer); } };
}

const readMode = () => {
  const el = document.documentElement;
  return el.classList.contains('dark') || el.dataset.theme === 'dark' ? 'night' : 'day';
};

const SKY_GOV = {
  'القاهرة': [30.04, 31.24], 'الجيزة': [30.01, 31.21], 'الإسكندرية': [31.20, 29.92], 'القليوبية': [30.18, 31.18],
  'الدقهلية': [31.04, 31.38], 'الشرقية': [30.59, 31.50], 'الغربية': [30.79, 31.00], 'المنوفية': [30.55, 31.01],
  'البحيرة': [31.04, 30.47], 'كفر الشيخ': [31.11, 30.94], 'دمياط': [31.42, 31.81], 'بورسعيد': [31.27, 32.30],
  'الإسماعيلية': [30.59, 32.27], 'السويس': [29.97, 32.55], 'شمال سيناء': [31.13, 33.80], 'جنوب سيناء': [28.24, 33.62],
  'مطروح': [31.35, 27.24], 'الفيوم': [29.31, 30.84], 'بني سويف': [29.07, 31.10], 'المنيا': [28.11, 30.75],
  'أسيوط': [27.18, 31.18], 'سوهاج': [26.56, 31.69], 'قنا': [26.16, 32.72], 'الأقصر': [25.69, 32.64],
  'أسوان': [24.09, 32.90], 'البحر الأحمر': [27.26, 33.81], 'الوادي الجديد': [25.45, 30.55],
};

export default function SkyBackground({
  governorate, lat = 30.0444, lon = 31.2357, askLocation = false, winterOnly = true, winterMonths,
  cometEveryMs = 2 * 60 * 1000, rainVolume = 1, showSwitcher = false, onModeChange, forceRain, theme,
}) {
  const [siteMode, setSiteMode] = useState(theme || 'night');
  const [manualBase, setManualBase] = useState(null);
  const [rainOverride, setRainOverride] = useState(null);
  const [wx, setWx] = useState({ intensity: 0, storm: false });
  const [soundOn, setSoundOn] = useState(true);
  const [locked, setLocked] = useState(false);
  const [switcherOpen, setSwitcherOpen] = useState(false);
  const starsRef = useRef(null), dayRef = useRef(null), moonRef = useRef(null), rainRef = useRef(null), lightRef = useRef(null);
  const dayEng = useRef(null), rainEng = useRef(null), sound = useRef(null), rainNow = useRef(0);
  const nightSkyRef = useRef(null);
  const currentScreen = useUIStore(s => s.currentScreen);
  const prevScreen = useRef(currentScreen);

  const mode = manualBase || siteMode;
  const rain = typeof forceRain === 'number' ? { intensity: forceRain, storm: false }
    : rainOverride === 'on' ? { intensity: 0.6, storm: false }
    : rainOverride === 'off' ? { intensity: 0, storm: false }
    : wx;
  const gov = governorate && SKY_GOV[governorate];
  const wlat = gov ? gov[0] : lat, wlon = gov ? gov[1] : lon;

  useEffect(() => {
    if (moonRef.current) drawMoon(moonRef.current);
    rainEng.current = createRain(rainRef.current);
    sound.current = createRainSound({ volume: rainVolume });
    try { setSoundOn(localStorage.getItem('sky:rain-sound') !== '0'); } catch (e) {}
    return () => { rainEng.current.destroy(); sound.current.destroy(); };
  }, []);

  // ✅ تحميل الاختيارات المحفوظة بعد mount (عشان مفيش hydration mismatch)
  useEffect(() => {
    try {
      const savedMode = localStorage.getItem('sky:manual-mode');
      if (savedMode === 'day' || savedMode === 'night') setManualBase(savedMode);
      const savedRain = localStorage.getItem('sky:rain-override');
      if (savedRain === 'on' || savedRain === 'off') setRainOverride(savedRain);
    } catch {}
  }, []);

  useIsoLayoutEffect(() => {
    if (theme) { setSiteMode(theme); return; }
    setSiteMode(readMode());
    const mo = new MutationObserver(() => { setSiteMode(readMode()); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
    return () => mo.disconnect();
  }, [theme]);

  useEffect(() => {
    if (typeof forceRain === 'number') return;
    const w = createWeather({ lat: wlat, lon: wlon, askLocation, winterOnly, winterMonths, onChange: setWx });
    return () => w.destroy();
  }, [forceRain, wlat, wlon, askLocation, winterOnly]);

  // استقبال أوامر من بره (زراير السايد بار واللاندينج)
  useEffect(() => {
  const onMode = (e) => {
    const m = e.detail;
    setManualBase(m);
    try { localStorage.setItem('sky:manual-mode', m); } catch {}
    if (onModeChange) onModeChange(m);
  };
  const onRain = (e) => {
    const v = e.detail;
    if (v === 'toggle') {
      setRainOverride(prev => {
        const next = prev === 'on' ? 'off' : 'on';
        try { localStorage.setItem('sky:rain-override', next); } catch {}
        return next;
      });
    } else {
      setRainOverride(v);
      try { localStorage.setItem('sky:rain-override', v); } catch {}
    }
  };
  const onSound = (e) => {
    const v = e.detail;
    const next = v === 'on';
    setSoundOn(next);
    try { localStorage.setItem('sky:rain-sound', next ? '1' : '0'); } catch {}
    if (sound.current) {
      sound.current.setEnabled(next);
      sound.current.unlock();
    }
  };
  window.addEventListener('sky:mode', onMode);
  window.addEventListener('sky:rain', onRain);
  window.addEventListener('sky:sound', onSound);
  return () => {
    window.removeEventListener('sky:mode', onMode);
    window.removeEventListener('sky:rain', onRain);
    window.removeEventListener('sky:sound', onSound);
  };
}, [onModeChange]);

  // الليل: نجوم + مذنب
  useEffect(() => {
    if (mode !== 'night' || !starsRef.current) return;
    const s = createNightSky(starsRef.current, { cometEveryMs });
    nightSkyRef.current = s;
    return () => { s.destroy(); nightSkyRef.current = null; };
  }, [mode, cometEveryMs]);

  // ✅ لما المستخدم يدخل أي بانل من السايد بار → مذنب بعد 3 ثواني
  useEffect(() => {
    if (prevScreen.current === currentScreen) return;
    prevScreen.current = currentScreen;
    if (!nightSkyRef.current) return;
    nightSkyRef.current.triggerAfter(3000);
  }, [currentScreen]);

  useEffect(() => {
    if (mode !== 'day' || !dayRef.current) return;
    const s = createDaySky(dayRef.current);
    dayEng.current = s;
    if (s) s.setRain(rainNow.current);
    return () => { dayEng.current = null; if (s) s.destroy(); };
  }, [mode]);

  useEffect(() => {
    rainNow.current = rain.intensity;
    if (dayEng.current) dayEng.current.setRain(rain.intensity);
    if (rainEng.current) rainEng.current.set(rain.intensity, mode === 'night');
    if (sound.current) { sound.current.setVolume(rainVolume); sound.current.setEnabled(soundOn); sound.current.setIntensity(rain.intensity); }
  }, [rain.intensity, mode, soundOn, rainVolume]);

  useEffect(() => {
    if (rain.intensity <= 0 || !soundOn) { setLocked(false); return; }
    const id = setInterval(() => setLocked(!!sound.current && !sound.current.isRunning()), 700);
    return () => clearInterval(id);
  }, [rain.intensity > 0, soundOn]);

  useEffect(() => {
    if (mode !== 'night' || rain.intensity <= 0) return;
    const l = createLightning(lightRef.current, {
      storm: rain.storm,
      onFlash: () => sound.current && sound.current.thunder(2 + Math.random() * 3, 0.25 + Math.random() * 0.2),
    });
    return () => l.destroy();
  }, [mode, rain.intensity > 0, rain.storm]);

  const toggleSound = () => {
    const next = !soundOn;
    setSoundOn(next);
    try { localStorage.setItem('sky:rain-sound', next ? '1' : '0'); } catch (e) {}
    if (sound.current) sound.current.unlock();
  };
  const pick = (m) => {
    setManualBase(m);
    try { localStorage.setItem('sky:manual-mode', m); } catch {}
    if (onModeChange) onModeChange(m);
  };
  const toggleRain = () => {
    const next = rain.intensity > 0 ? 'off' : 'on';
    setRainOverride(next);
    try { localStorage.setItem('sky:rain-override', next); } catch {}
  };

  return (
    <>
      <div className="sky" data-mode={mode} data-rain={rain.intensity > 0 ? '1' : '0'} aria-hidden="true">
        <style>{CSS}</style>
        <div className="sky-night">
          <canvas ref={starsRef} className="stars" />
          <div className="moon-wrap"><canvas ref={moonRef} className="moon" /></div>
          <div ref={lightRef} className="lightning" />
        </div>
        <div className="sky-day"><canvas ref={dayRef} className="day-canvas" /></div>
        <canvas ref={rainRef} className="rain" />
      </div>

    
      <button type="button" className="sky-sound" data-show={rain.intensity > 0 ? '1' : '0'}
        data-on={soundOn ? '1' : '0'} data-locked={locked ? '1' : '0'} onClick={toggleSound}
        aria-label={!soundOn ? 'تشغيل صوت المطر' : locked ? 'اضغط لتشغيل صوت المطر' : 'إيقاف صوت المطر'} aria-pressed={soundOn}>
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M11 5 6 9H3v6h3l5 4z"/><path className="on" d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13"/><path className="off" d="M16 9l5 6M21 9l-5 6"/></svg>
      </button>
    </>
  );
}