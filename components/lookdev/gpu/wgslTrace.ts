// The path tracer as a WebGPU compute kernel: a port of traceFrag in shaders.ts, variant for variant. Each
// invocation traces one pixel's new samples and blends them into the running means held in storage buffers
// (accumulated in place: no ping-pong). Rows go bottom up, as gl_FragCoord does, so a pixel draws the same samples
// on both renderers. The comments on the models and their sources live with the GLSL original.
import { COMMON, SAMPLING } from './wgslCommon'
import { MICROFACET, TABLE_CONSTS, EON, FUZZ, THIN_FILM, OPENPBR } from './wgslBsdf'
import { SCENE, PASS, MAT_UNPACK } from './sceneLayout'
import { preprocess } from './preprocess'

const BINDINGS = /* wgsl */ `
${SCENE.wgsl()}
${PASS.wgsl()}
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> pu: Pass; // this dispatch's ('pass' is reserved in WGSL)
// Running means, plane after plane of width x height: key (a = coverage), fill (a = mean Y^2), rim, environment;
// and the denoiser's: albedo (a = mean Y), normal.
@group(0) @binding(2) var<storage, read_write> accum: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> aux: array<vec4f>;
@group(0) @binding(4) var eTable: texture_3d<f32>;
@group(0) @binding(5) var eTableT: texture_3d<f32>;
@group(0) @binding(6) var eSampler: sampler;
@group(0) @binding(7) var envMap: texture_2d<f32>;
@group(0) @binding(8) var envSampler: sampler;
@group(0) @binding(9) var<storage, read> envAlias: array<vec4f>;
#ifdef MESH
@group(0) @binding(10) var<storage, read> bvh: array<vec4u>;
@group(0) @binding(11) var<storage, read> triPos: array<vec4f>;
@group(0) @binding(12) var<storage, read> triNrm: array<vec4u>;
#endif
`

const LIGHTS = /* wgsl */ `
fn lightCorner(k: i32) -> vec3f { return scene.lightCorner[k].xyz; }
fn lightU(k: i32) -> vec3f { return scene.lightU[k].xyz; }
fn lightV(k: i32) -> vec3f { return scene.lightV[k].xyz; }
fn lightRadiance(k: i32) -> vec3f { return scene.lightRad[k].xyz; }
fn lightUV(k: i32) -> f32 { return scene.lightRad[k].w; }
// Light 3 is the environment, when on.
fn lightOn(k: i32) -> bool { return k >= 3 || maxc(scene.lightRad[min(k, 2)].xyz) > 0.0; }
fn facesLight(k: i32, o: vec3f) -> bool { return dot(o - lightCorner(k), cross(lightU(k), lightV(k))) > 0.0; }

struct SphQuad {
  o: vec3f, x: vec3f, y: vec3f, z: vec3f,
  z0: f32, z0sq: f32, x0: f32, y0: f32, y0sq: f32, x1: f32, y1: f32, y1sq: f32,
  b0: f32, b1: f32, b0sq: f32, k: f32, S: f32, Spdf: f32, ok: bool,
}
fn safeAsin(x: f32) -> f32 { return asin(clamp(x, -1.0, 1.0)); }
fn angleBetween(v1: vec3f, v2: vec3f) -> f32 {
  if (dot(v1, v2) < 0.0) { return PI - 2.0 * safeAsin(length(v1 + v2) * 0.5); }
  return 2.0 * safeAsin(length(v2 - v1) * 0.5);
}
fn sphQuadInit(s: vec3f, ex: vec3f, ey: vec3f, o: vec3f) -> SphQuad {
  var q: SphQuad;
  q.o = o;
  let exl = length(ex);
  let eyl = length(ey);
  q.x = ex / exl;
  q.y = ey / eyl;
  q.z = cross(q.x, q.y);
  let d = s - o;
  q.z0 = dot(d, q.z);
  if (q.z0 > 0.0) { q.z = -q.z; q.z0 = -q.z0; }
  q.z0sq = q.z0 * q.z0;
  q.x0 = dot(d, q.x);
  q.y0 = dot(d, q.y);
  q.x1 = q.x0 + exl;
  q.y1 = q.y0 + eyl;
  q.y0sq = q.y0 * q.y0;
  q.y1sq = q.y1 * q.y1;
  q.ok = false;
  if (-q.z0 <= 1e-6 * max(exl, eyl)) { return q; }
  let v00 = vec3f(q.x0, q.y0, q.z0);
  let v01 = vec3f(q.x0, q.y1, q.z0);
  let v10 = vec3f(q.x1, q.y0, q.z0);
  let v11 = vec3f(q.x1, q.y1, q.z0);
  let n0 = normalize(vec3f(0.0, q.z0, -q.y0));
  let n1 = normalize(vec3f(-q.z0, 0.0, q.x1));
  let n2 = normalize(vec3f(0.0, -q.z0, q.y1));
  let n3 = normalize(vec3f(q.z0, 0.0, -q.x0));
  let g0 = angleBetween(-n0, n1);
  let g1 = angleBetween(-n1, n2);
  let g2 = angleBetween(-n2, n3);
  let g3 = angleBetween(-n3, n0);
  q.b0 = n0.z;
  q.b1 = n2.z;
  q.b0sq = q.b0 * q.b0;
  q.k = TWO_PI - g2 - g3;
  q.S = g0 + g1 - q.k;
  let N = exl * eyl * (-q.z0);
  let l00 = length(v00);
  let l10 = length(v10);
  let l11 = length(v11);
  let l01 = length(v01);
  let D1 = l00 * l10 * l11 + dot(v00, v10) * l11 + dot(v00, v11) * l10 + dot(v10, v11) * l00;
  let D2 = l00 * l11 * l01 + dot(v00, v11) * l01 + dot(v00, v01) * l11 + dot(v11, v01) * l00;
  q.Spdf = 2.0 * atan2(N, D1) + 2.0 * atan2(N, D2);
  q.ok = q.S > 0.0 && q.Spdf > 0.0;
  return q;
}
fn sphQuadSample(q: SphQuad, u: f32, v: f32) -> vec3f {
  let au = u * q.S + q.k;
  var sa = sin(au);
  if (abs(sa) < 1e-7) { sa = select(1e-7, -1e-7, sa < 0.0); }
  let fu = (cos(au) * q.b0 - q.b1) / sa;
  var cu = select(-1.0, 1.0, fu > 0.0) * inverseSqrt(fu * fu + q.b0sq);
  cu = clamp(cu, -ONE_MINUS_EPS, ONE_MINUS_EPS);
  var xu = -(cu * q.z0) / sqrt(max(0.0, 1.0 - cu * cu));
  xu = clamp(xu, q.x0, q.x1);
  let dd = sqrt(xu * xu + q.z0sq);
  let h0 = q.y0 / sqrt(dd * dd + q.y0sq);
  let h1 = q.y1 / sqrt(dd * dd + q.y1sq);
  let hv = h0 + v * (h1 - h0);
  let hv2 = hv * hv;
  let yv = select(q.y1, (hv * dd) / sqrt(max(1.0 - hv2, 1e-30)), hv2 < 1.0 - 1e-6);
  return q.o + xu * q.x + yv * q.y + q.z0 * q.z;
}
const MIN_SPHERICAL_SAMPLE_AREA: f32 = 1e-4;
struct LightSample { ok: bool, wi: vec3f, dist: f32, pdf: f32 }
fn sampleLight(k: i32, o: vec3f, u: vec2f) -> LightSample {
  var r = LightSample(false, vec3f(0.0, 0.0, 1.0), 0.0, 0.0);
  if (!facesLight(k, o)) { return r; }
  let q = sphQuadInit(lightCorner(k), lightU(k), lightV(k), o);
  var p: vec3f;
  var pdf: f32;
  if (q.ok && q.Spdf > MIN_SPHERICAL_SAMPLE_AREA) {
    p = sphQuadSample(q, u.x, u.y);
    pdf = 1.0 / q.Spdf;
  } else {
    p = lightCorner(k) + u.x * lightU(k) + u.y * lightV(k);
    let nL = cross(lightU(k), lightV(k));
    let area = length(nL);
    let w = p - o;
    let d2 = dot(w, w);
    let cosL = abs(dot(nL / area, w)) * inverseSqrt(max(d2, 1e-30));
    pdf = select(0.0, d2 / (area * cosL), cosL > 0.0);
  }
  let w = p - o;
  let dist = length(w);
  if (dist <= 0.0 || pdf <= 0.0) { return r; }
  return LightSample(true, w / dist, dist, pdf);
}
fn rectSolidAngle(k: i32, o: vec3f) -> f32 {
  let ex = lightU(k);
  let ey = lightV(k);
  let a = lightCorner(k) - o;
  let b = a + ex;
  let c = b + ey;
  let d = a + ey;
  let N = abs(dot(a, cross(ex, ey)));
  let la = length(a);
  let lb = length(b);
  let lc = length(c);
  let ld = length(d);
  let D1 = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
  let D2 = la * lc * ld + dot(a, c) * ld + dot(a, d) * lc + dot(c, d) * la;
  return 2.0 * atan2(N, D1) + 2.0 * atan2(N, D2);
}
fn lightPdf(k: i32, o: vec3f, wi: vec3f, dist: f32) -> f32 {
  if (!facesLight(k, o)) { return 0.0; }
  let S = rectSolidAngle(k, o);
  if (S > MIN_SPHERICAL_SAMPLE_AREA) { return 1.0 / S; }
  let nL = cross(lightU(k), lightV(k));
  let area = length(nL);
  let cosL = abs(dot(nL / area, wi));
  return select(0.0, dist * dist / (area * cosL), cosL > 0.0);
}
fn intersectRect(ro: vec3f, rd: vec3f, k: i32, tMax: f32) -> f32 {
  let s = lightCorner(k);
  let ex = lightU(k);
  let ey = lightV(k);
  let n = cross(ex, ey);
  let denom = dot(n, rd);
  if (denom >= 0.0) { return -1.0; }
  let t = dot(s - ro, n) / denom;
  if (!(t > 1e-4 && t < tMax)) { return -1.0; }
  let d = (ro - s) + t * rd;
  let uv = vec2f(dot(d, ex) / dot(ex, ex), dot(d, ey) / dot(ey, ey));
  if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) { return t; }
  return -1.0;
}

// Image-based lighting: light 3 (see environments.ts and the GLSL tracer).
const ENV_LIGHT: i32 = 3;
fn envSize() -> vec2i { return scene.iparams3.xy; }
fn envUV(d: vec3f) -> vec2f {
  return vec2f(fract(atan2(d.x, -d.z) * (0.5 / PI) + 0.5 + scene.fparams.w), acos(clamp(d.y, -1.0, 1.0)) / PI);
}
fn envTexel(uv: vec2f) -> vec2i { return min(vec2i(uv * vec2f(envSize())), envSize() - 1); }
fn envIndex(t: vec2i) -> i32 { return t.y * envSize().x + t.x; }
fn envLe(d: vec3f) -> vec3f { return textureLoad(envMap, envTexel(envUV(d)), 0).rgb; }
fn envBackdrop(d: vec3f) -> vec3f { return textureSampleLevel(envMap, envSampler, envUV(d), scene.ambient.w).rgb; }
fn envPdf(d: vec3f) -> f32 {
  let s = sqrt(max(0.0, 1.0 - d.y * d.y));
  return select(0.0, envAlias[envIndex(envTexel(envUV(d)))].z / s, s > 1e-6);
}
struct EnvSample { ok: bool, wi: vec3f, pdf: f32 }
fn sampleEnv(u: vec4f) -> EnvSample {
  let sz = envSize();
  let n = sz.x * sz.y;
  var i = min(i32(u.x * f32(n)), n - 1);
  let a = envAlias[i];
  if (u.y >= a.x) { i = i32(a.y); }
  let px = vec2i(i % sz.x, i / sz.x);
  let theta = (f32(px.y) + u.w) / f32(sz.y) * PI;
  let phi = ((f32(px.x) + u.z) / f32(sz.x) - 0.5 - scene.fparams.w) * TWO_PI;
  let s = sin(theta);
  let wi = vec3f(s * sin(phi), cos(theta), -s * cos(phi));
  let pdf = select(0.0, envAlias[i].z / s, s > 1e-6);
  return EnvSample(pdf > 0.0, wi, pdf);
}

fn powerHeuristic(f: f32, g: f32) -> f32 {
  let f2 = f * f;
  let g2 = g * g;
  if (f2 > 1e30) { return 1.0; }
  return select(0.0, f2 / (f2 + g2), f2 + g2 > 0.0);
}
`

const MESH = /* wgsl */ `
fn modelRot() -> mat3x3f { return mat3x3f(scene.modelRot[0].xyz, scene.modelRot[1].xyz, scene.modelRot[2].xyz); }
fn thinGlassT(c: f32) -> vec3f {
  let g = loadMat(4u);
  let T = 1.0 - fresnelDielectric(c, max(g.specular_ior, 1.0));
  let tint = select(max(g.transmission_color, vec3f(0.0)), vec3f(1.0), g.transmission_depth > 0.0);
  return T * tint * sat(g.transmission_weight) * (1.0 - sat(g.base_metalness));
}
fn octDecode(u: u32) -> vec3f {
  let f = unpack2x16snorm(u);
  var n = vec3f(f, 1.0 - abs(f.x) - abs(f.y));
  let t = max(-n.z, 0.0);
  n = vec3f(n.x + select(t, -t, n.x >= 0.0), n.y + select(t, -t, n.y >= 0.0), n.z);
  return normalize(n);
}
fn boxEnter(bmin: vec3f, bmax: vec3f, o: vec3f, invD: vec3f, tBest: f32) -> f32 {
  let t0 = (bmin - o) * invD;
  let t1 = (bmax - o) * invD;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  let tEnter = max(max(tn.x, tn.y), max(tn.z, 0.0));
  let tExit = min(min(tf.x, tf.y), tf.z);
  return select(1e30, tEnter, tEnter <= tExit && tEnter < tBest);
}
const BVH_STACK: i32 = 48;
// The traversal stack lives in workgroup memory, a column per invocation (8 x 8 per workgroup, see main), and holds
// node references only: as a per-invocation array (with each entry's distance, as the GLSL tracer keeps them) it
// went to the GPU's scratch memory, and model scenes ran at a third of the speed. Popped nodes are not culled by
// their distance; their children's boxes are still tested against the nearest hit, so every hit is the same.
var<private> gLane: u32;
var<workgroup> stackRef: array<i32, 3072>; // BVH_STACK x 64
struct MeshHit { t: f32, tri: i32, bary: vec2f }
fn traceMesh(roW: vec3f, rdW: vec3f, tMax: f32, anyHit: bool, Tthin: ptr<function, vec3f>) -> MeshHit {
  var res = MeshHit(-1.0, -1, vec2f(0.0));
  let toObj = transpose(modelRot());
  let sc = scene.modelPos.w;
  let o = toObj * (roW - scene.modelPos.xyz) / sc;
  let d = toObj * rdW / sc;
  let dSafe = vec3f(select(d.x, 1e-20, abs(d.x) < 1e-20), select(d.y, 1e-20, abs(d.y) < 1e-20), select(d.z, 1e-20, abs(d.z) < 1e-20));
  let invD = 1.0 / dSafe;
  var tBest = tMax;
  var sp = 0;
  var cur = 0;
  var triHit = -1;
  var bary = vec2f(0.0);
  let maxVisits = scene.iparams2.w;
  for (var visit = 0; visit < maxVisits; visit++) {
    let count = cur & 31;
    let index = cur >> 5u;
    if (count == 0) {
      let h = bvh[4 * index];
      let q0 = bvh[4 * index + 1];
      let q1 = bvh[4 * index + 2];
      let refs = bvh[4 * index + 3];
      let corner = bitcast<vec3f>(h.xyz);
      let stepSize = bitcast<vec3f>(vec3u(h.w & 255u, (h.w >> 8u) & 255u, (h.w >> 16u) & 255u) << vec3u(23u));
      var rc = vec4i(refs);
      var tc: vec4f;
      for (var i = 0; i < 4; i++) {
        let sh = u32(8 * i);
        let lo = corner + vec3f((q0.xyz >> vec3u(sh)) & vec3u(255u)) * stepSize;
        let hi = corner + vec3f(vec3u(q0.w >> sh, q1.x >> sh, q1.y >> sh) & vec3u(255u)) * stepSize;
        tc[i] = select(boxEnter(lo, hi, o, invD, tBest), 1e30, refs[i] == 0xffffffffu);
      }
      // Nearest first: a sorting network on (distance, ref) pairs, misses (1e30) last.
      if (tc.y < tc.x) { let t = tc.x; tc.x = tc.y; tc.y = t; let r = rc.x; rc.x = rc.y; rc.y = r; }
      if (tc.w < tc.z) { let t = tc.z; tc.z = tc.w; tc.w = t; let r = rc.z; rc.z = rc.w; rc.w = r; }
      if (tc.z < tc.x) { let t = tc.x; tc.x = tc.z; tc.z = t; let r = rc.x; rc.x = rc.z; rc.z = r; }
      if (tc.w < tc.y) { let t = tc.y; tc.y = tc.w; tc.w = t; let r = rc.y; rc.y = rc.w; rc.w = r; }
      if (tc.z < tc.y) { let t = tc.y; tc.y = tc.z; tc.z = t; let r = rc.y; rc.y = rc.z; rc.z = r; }
      if (tc.x < 1e30) {
        if (tc.w < 1e30 && sp < BVH_STACK) { stackRef[u32(sp) * 64u + gLane] = rc.w; sp++; }
        if (tc.z < 1e30 && sp < BVH_STACK) { stackRef[u32(sp) * 64u + gLane] = rc.z; sp++; }
        if (tc.y < 1e30 && sp < BVH_STACK) { stackRef[u32(sp) * 64u + gLane] = rc.y; sp++; }
        cur = rc.x;
        continue;
      }
    } else {
      for (var i = 0; i < count; i++) {
        let tri = index + i;
        let v0 = triPos[3 * tri].xyz;
        let e1 = triPos[3 * tri + 1].xyz;
        let e2 = triPos[3 * tri + 2].xyz;
        let pv = cross(d, e2);
        let det = dot(e1, pv);
        if (det == 0.0) { continue; }
        let inv = 1.0 / det;
        let tv = o - v0;
        let u = dot(tv, pv) * inv;
        if (u < 0.0 || u > 1.0) { continue; }
        let qv = cross(tv, e1);
        let v = dot(d, qv) * inv;
        if (v < 0.0 || u + v > 1.0) { continue; }
        let t = dot(e2, qv) * inv;
        if (t > 0.0 && t < tBest) {
#if defined(FLUOR) && defined(GLASS)
          if (anyHit && gMediumShadow && i32(triNrm[tri].w) == 2) {
            mediumPass(abs(dot(normalize(cross(e1, e2)), normalize(d))), t, Tthin);
            if (maxc(*Tthin) <= 0.0) { return MeshHit(t, -1, vec2f(0.0)); }
            continue;
          }
#endif
#ifdef THIN
          if (anyHit && scene.iparams2.z >= 0 && i32(triNrm[tri].w) == scene.iparams2.z) {
            *Tthin *= thinGlassT(abs(dot(normalize(cross(e1, e2)), normalize(d))));
            if (maxc(*Tthin) <= 0.0) { return MeshHit(t, -1, vec2f(0.0)); }
            continue;
          }
#endif
          tBest = t;
          triHit = tri;
          bary = vec2f(u, v);
        }
      }
      if (anyHit && triHit >= 0) { return MeshHit(tBest, triHit, bary); }
    }
    if (sp <= 0) { break; }
    sp--;
    cur = stackRef[u32(sp) * 64u + gLane];
  }
  if (triHit >= 0) { return MeshHit(tBest, triHit, bary); }
  return res;
}
fn intersectBall(ro: vec3f, rd: vec3f, s: vec4f, tMax: f32) -> f32 {
  if (s.w <= 0.0) { return -1.0; }
  let t = intersectSphere((ro - s.xyz) / s.w, rd, vec3f(0.0), tMax / s.w);
  return select(-1.0, t * s.w, t > 0.0);
}
`

const SCENE_QUERY = /* wgsl */ `
fn intersectSphere(ro: vec3f, rd: vec3f, center: vec3f, tMax: f32) -> f32 {
  let f = ro - center;
  let bp = -dot(f, rd);
  let l = f + bp * rd;
  let disc = 1.0 - dot(l, l);
  if (disc < 0.0) { return -1.0; }
  let c = dot(f, f) - 1.0;
  let q = bp + select(-1.0, 1.0, bp > 0.0) * sqrt(disc);
  if (q == 0.0) { return -1.0; }
  let t0 = c / q;
  let t1 = q;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  if (tn > 0.0 && tn < tMax) { return tn; }
  if (tf > 0.0 && tf < tMax) { return tf; }
  return -1.0;
}
// Self-intersection offset: Waechter and Binder, Ray Tracing Gems ch. 6, Listing 6-1.
fn offsetRay(p: vec3f, n: vec3f) -> vec3f {
  let ORIGIN = 1.0 / 32.0;
  let FLOAT_SCALE = 1.0 / 65536.0;
  let INT_SCALE = 256.0;
  let of_i = vec3i(INT_SCALE * n);
  let p_i = bitcast<vec3f>(bitcast<vec3i>(p) + vec3i(
    select(of_i.x, -of_i.x, p.x < 0.0),
    select(of_i.y, -of_i.y, p.y < 0.0),
    select(of_i.z, -of_i.z, p.z < 0.0)));
  return vec3f(
    select(p_i.x, p.x + FLOAT_SCALE * n.x, abs(p.x) < ORIGIN),
    select(p_i.y, p.y + FLOAT_SCALE * n.y, abs(p.y) < ORIGIN),
    select(p_i.z, p.z + FLOAT_SCALE * n.z, abs(p.z) < ORIGIN));
}

struct Hit { t: f32, n: vec3f, ng: vec3f, mat: i32, light: i32, lightT: f32, back: bool }

fn cycZ() -> f32 { return scene.fparams.x; }
fn cycR() -> f32 { return scene.fparams.y; }
struct CycHit { t: f32, n: vec3f }
fn cycHit(ro: vec3f, rd: vec3f, tMaxIn: f32) -> CycHit {
  var tMax = tMaxIn;
  var r = CycHit(-1.0, vec3f(0.0, 0.0, 1.0));
  if (rd.z < 0.0) {
    let t = (cycZ() - cycR() - ro.z) / rd.z;
    if (t > 1e-4 && t < tMax && ro.y + rd.y * t >= cycR()) { r.t = t; tMax = t; }
  }
  let o2 = vec2f(ro.y - cycR(), ro.z - cycZ());
  let d2 = rd.yz;
  let a = dot(d2, d2);
  let b = dot(o2, d2);
  let c = dot(o2, o2) - cycR() * cycR();
  let disc = b * b - a * c;
  if (a > 0.0 && disc > 0.0) {
    let sqd = sqrt(disc);
    for (var i = 0; i < 2; i++) {
      let t = (-b + select(sqd, -sqd, i == 0)) / a;
      let q = o2 + d2 * t;
      if (t > 1e-4 && t < tMax && q.x <= 0.0 && q.y <= 0.0) {
        r.t = t;
        r.n = vec3f(0.0, -q.x, -q.y) / cycR();
        break;
      }
    }
  }
  return r;
}

const CHART_MAT: i32 = 16;
const CHART_GAP: f32 = 0.18;
struct ChartHit { t: f32, n: vec3f, cell: i32 }
fn chartHit(ro: vec3f, rd: vec3f, tMax: f32) -> ChartHit {
  let U = scene.chartU.xyz;
  let V = scene.chartV.xyz;
  let N = normalize(cross(U, V));
  let dn = dot(rd, N);
  var r = ChartHit(-1.0, select(-N, N, dn < 0.0), 24);
  if (abs(dn) < 1e-8) { return r; }
  let t = dot(scene.chartO.xyz - ro, N) / dn;
  if (t <= 1e-4 || t >= tMax) { return r; }
  let q = ro + rd * t - scene.chartO.xyz;
  let ab = vec2f(dot(q, U) / dot(U, U), dot(q, V) / dot(V, V));
  if (any(ab < vec2f(0.0)) || any(ab > vec2f(1.0))) { return r; }
  if (dn < 0.0) {
    let g = ab * vec2f(6.0 + 7.0 * CHART_GAP, 4.0 + 5.0 * CHART_GAP) - CHART_GAP;
    let cell = floor(g / (1.0 + CHART_GAP));
    let f = g - cell * (1.0 + CHART_GAP);
    if (cell.x >= 0.0 && cell.y >= 0.0 && cell.x < 6.0 && cell.y < 4.0 && f.x < 1.0 && f.y < 1.0) {
      r.cell = i32(cell.x) + 6 * (3 - i32(cell.y));
    }
  }
  r.t = t;
  return r;
}

#if defined(FLUOR) && defined(GLASS)
var<private> gMediumShadow: bool = false;
var<private> gTuv: f32 = 1.0;
var<private> gExitT: f32 = 0.0;
fn mediumPass(cosT: f32, t: f32, T: ptr<function, vec3f>) {
  let through = 1.0 - fresnelDielectric(cosT, 1.0 / max(loadMat(2u).specular_ior, 1.0));
  *T *= through;
  gTuv *= through;
  gExitT = min(gExitT, t);
}
#endif

${'${MESH_PLACEHOLDER}'}

fn smoothThin(m: Mat) -> bool {
  return m.geometry_thin_walled > 0.5 && m.transmission_weight > 0.0 && m.specular_roughness <= 0.01 && m.coat_weight <= 0.0;
}
#if !defined(MESH) && defined(THIN)
fn thinPassT(m: Mat, c: f32) -> vec3f {
  let nd = max(m.specular_ior, 1.0);
  var T = vec3f(1.0 - fresnelDielectric(c, nd));
  if (m.thin_film_weight > 0.0 && m.thin_film_thickness > 0.0) {
    T = mix(T, vec3f(1.0) - thinFilmF(c, m.thin_film_ior, m.thin_film_thickness, vec3f(f0FromEta(nd))), sat(m.thin_film_weight));
  }
  let tint = select(max(m.transmission_color, vec3f(0.0)), vec3f(1.0), m.transmission_depth > 0.0);
  return T * tint * sat(m.transmission_weight) * (1.0 - sat(m.base_metalness));
}
#endif
fn ballX(i: i32) -> f32 { return scene.ballX[i]; }

// The one scene query, for both kinds of ray (called from a single place: see main).
#ifdef MESH
// A model hit's material, side and normals (shading normal interpolated, kept facing the ray's side).
fn meshHit(mh: MeshHit, rd: vec3f, h: ptr<function, Hit>) {
  let nn = triNrm[mh.tri];
  (*h).mat = i32(nn.w);
  let e1 = triPos[3 * mh.tri + 1].xyz;
  let e2 = triPos[3 * mh.tri + 2].xyz;
  let R = modelRot();
  var ng = normalize(R * cross(e1, e2));
  let bc = mh.bary;
  var ns = normalize(R * ((1.0 - bc.x - bc.y) * octDecode(nn.x) + bc.x * octDecode(nn.y) + bc.y * octDecode(nn.z)));
  (*h).back = dot(rd, select(ng, -ng, dot(ng, ns) < 0.0)) > 0.0;
  if (dot(ng, rd) > 0.0) { ng = -ng; }
  if (dot(ns, ng) < 0.0) { ns = -ns; }
  if (dot(ns, -rd) <= 1e-4) { ns = ng; }
  (*h).n = ns;
  (*h).ng = ng;
}
#endif

fn trace(ro: vec3f, rd: vec3f, tMax: f32, shadow: bool, withLights: bool, T: ptr<function, vec3f>) -> Hit {
  var h: Hit;
  h.t = tMax; h.mat = -1; h.light = -1; h.lightT = tMax; h.n = vec3f(0.0, 1.0, 0.0); h.ng = h.n; h.back = false;
  let furnace = scene.iparams0.x == 1;
#ifdef MESH
  for (var i = 0; i < 3; i++) {
    let s = scene.balls[i];
    let t = intersectBall(ro, rd, s, h.t);
    if (t > 0.0) {
      let nOut = normalize(ro + rd * t - s.xyz);
      h.t = t; h.back = dot(rd, nOut) > 0.0; h.n = select(nOut, -nOut, h.back); h.ng = h.n; h.mat = i;
      if (shadow) { return h; }
    }
  }
  let mh = traceMesh(ro, rd, h.t, shadow, T);
  if (mh.t > 0.0) {
    h.t = mh.t;
    if (shadow) { h.mat = 4; return h; }
    meshHit(mh, rd, &h);
  }
#else
  for (var i = 0; i < 3; i++) {
    let c = vec3f(ballX(i), 1.0, 0.0);
#ifdef THIN
#if defined(FLUOR) && defined(GLASS)
    let passHero = scene.iparams1.z == 1 || gMediumShadow;
#else
    let passHero = scene.iparams1.z == 1;
#endif
    if (shadow && i == 2 && passHero) {
      let f = ro - c;
      let b = dot(f, rd);
      let disc = b * b - (dot(f, f) - 1.0);
      if (disc > 0.0) {
        let sqd = sqrt(disc);
        for (var j = 0; j < 2; j++) {
          let t = -b + select(sqd, -sqd, j == 0);
          if (t > 1e-4 && t < tMax) {
            let cs = abs(dot(rd, normalize(ro + rd * t - c)));
#if defined(FLUOR) && defined(GLASS)
            if (gMediumShadow) { mediumPass(cs, t, T); } else { *T *= thinPassT(loadMat(2u), cs); }
#else
            *T *= thinPassT(loadMat(2u), cs);
#endif
          }
        }
      }
      continue;
    }
#endif
    let t = intersectSphere(ro, rd, c, h.t);
    if (t > 0.0) {
      let nOut = normalize(ro + rd * t - c);
      h.t = t; h.back = dot(rd, nOut) > 0.0; h.n = select(nOut, -nOut, h.back); h.ng = h.n; h.mat = i;
      if (shadow) { return h; }
    }
  }
#endif
  if (scene.iparams1.y == 1 && !furnace) {
    let ch = chartHit(ro, rd, h.t);
    if (ch.t > 0.0) {
      h.t = ch.t; h.n = ch.n; h.ng = ch.n; h.mat = CHART_MAT + ch.cell; h.back = false;
      if (shadow) { return h; }
    }
  }
  let cyc = scene.iparams1.x == 1;
  if (!furnace && rd.y < 0.0) {
    let t = -ro.y / rd.y;
    if (t > 0.0 && t < h.t && (!cyc || ro.z + rd.z * t >= cycZ())) {
      h.t = t; h.n = vec3f(0.0, 1.0, 0.0); h.ng = h.n; h.mat = 3; h.back = false;
      if (shadow) { return h; }
    }
  }
  if (!furnace && cyc) {
    let cy = cycHit(ro, rd, h.t);
    if (cy.t > 0.0) { h.t = cy.t; h.n = cy.n; h.ng = cy.n; h.mat = 3; h.back = false; }
  }
  if (withLights) {
    for (var k = 0; k < scene.iparams1.w; k++) {
      if (!lightOn(k)) { continue; }
      let t = intersectRect(ro, rd, k, min(h.lightT, h.t));
      if (t > 0.0) { h.lightT = t; h.light = k; }
    }
  }
  return h;
}

#ifdef SSS
// A subsurface walk's flight can only end on the object walked in: the hero ball, or the model (the reference
// balls, floor, chart and lights are all outside it). Same hits as trace(), for a fraction of the work.
fn traceWalk(ro: vec3f, rd: vec3f, tMax: f32) -> Hit {
  var h: Hit;
  h.t = tMax; h.mat = -1; h.light = -1; h.lightT = tMax; h.n = vec3f(0.0, 1.0, 0.0); h.ng = h.n; h.back = false;
#ifdef MESH
  var T = vec3f(1.0);
  let mh = traceMesh(ro, rd, h.t, false, &T);
  if (mh.t > 0.0) {
    h.t = mh.t;
    meshHit(mh, rd, &h);
  }
#else
  let c = vec3f(ballX(2), 1.0, 0.0);
  let t = intersectSphere(ro, rd, c, h.t);
  if (t > 0.0) {
    let nOut = normalize(ro + rd * t - c);
    h.t = t; h.back = dot(rd, nOut) > 0.0; h.n = select(nOut, -nOut, h.back); h.ng = h.n; h.mat = 2;
  }
#endif
  return h;
}
#endif

fn getMat(i: i32) -> Mat {
  var m: Mat;
  if (i >= CHART_MAT) {
    m = loadMat(0u);
    m.base_color = scene.chartColor[i - CHART_MAT].xyz;
  } else {
    m = loadMat(u32(i));
  }
  if (scene.iparams0.x == 1) {
    m.base_weight = 1.0;
    m.base_color = vec3f(1.0);
    m.specular_weight = 1.0;
    m.specular_color = vec3f(1.0);
    m.coat_color = vec3f(1.0);
    m.fuzz_color = vec3f(1.0);
    m.transmission_color = vec3f(1.0);
    m.subsurface_color = vec3f(1.0);
    m.lab_fluor_weight = 0.0;
  }
  return m;
}

fn flakeNormal(m: Mat, P: vec3f) -> vec3f {
  if (m.lab_flake_coverage <= 0.0) { return vec3f(0.0, 0.0, 1.0); }
  let h = pcg3d(vec3u(vec3i(floor(P / m.lab_flake_size)) + 1048576));
  if (u01(h.x) >= m.lab_flake_coverage) { return vec3f(0.0, 0.0, 1.0); }
  let r = m.lab_flake_tilt * sqrt(u01(h.y));
  let phi = TWO_PI * u01(h.z);
  return normalize(vec3f(r * cos(phi), r * sin(phi), 1.0));
}
fn clampIndirect(c: vec3f) -> vec3f {
  let m = maxc(c);
  let cl = scene.fparams.z;
  return select(c, c * (cl / m), cl > 0.0 && m > cl);
}
fn nonFinite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u; }
fn sampleCosine(u: vec2f) -> vec3f {
  let r = sqrt(u.x);
  let phi = TWO_PI * u.y;
  return vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}
#ifdef FLUOR
const FLUOR_LOBE_P: f32 = 0.5;
fn fluorBase(s: Surf) -> vec3f { return s.tBase * (1.0 - s.metal) * (1.0 - s.transW) * (1.0 - s.EspecR3); }
fn uvAlbedo(fD: vec3f, fS: vec3f, m: Mat) -> f32 { return dot(fS, vec3f(1.0 / 3.0)) + dot(fD, vec3f(1.0 / 3.0)) * m.lab_uv_ratio; }
#ifdef GLASS
const FLUOR_EVENT_P: f32 = 0.5;
fn truncExpSample(c: f32, T: f32, u: f32) -> f32 {
  let a = 1.0 - exp(-c * T);
  return select(-log(max(1.0 - u * a, 1e-30)) / c, u * T, c * T < 1e-4);
}
fn truncExpPdf(c: f32, T: f32, t: f32) -> f32 {
  return select(c * exp(-c * t) / max(1.0 - exp(-c * T), 1e-30), 1.0 / T, c * T < 1e-4);
}
fn sampleSphere(u: vec2f) -> vec3f {
  let z = 1.0 - 2.0 * u.x;
  let r = sqrt(max(0.0, 1.0 - z * z));
  let phi = TWO_PI * u.y;
  return vec3f(r * cos(phi), r * sin(phi), z);
}
#endif
#endif
#ifdef SSS
fn walkRand(seed: ptr<function, u32>) -> f32 {
  *seed = pcg(*seed);
  return u01(*seed);
}
fn sampleHG(d: vec3f, g: f32, u1: f32, u2: f32) -> vec3f {
  var c = 1.0 - 2.0 * u1;
  if (abs(g) > 1e-3) {
    let q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
    c = clamp((1.0 + g * g - q * q) / (2.0 * g), -1.0, 1.0);
  }
  let sn = sqrt(max(0.0, 1.0 - c * c));
  let phi = TWO_PI * u2;
  let B = onb(d);
  return normalize(B.b1 * (sn * cos(phi)) + B.b2 * (sn * sin(phi)) + d * c);
}
#endif
`

const KERNEL = /* wgsl */ `
struct Lights4 { L0: vec3f, L1: vec3f, L2: vec3f, L3: vec3f }
fn addLight(k: i32, c: vec3f, Ls: ptr<function, Lights4>) {
  if (k == 0) { (*Ls).L0 += c; }
  else if (k == 1) { (*Ls).L1 += c; }
  else if (k == 2) { (*Ls).L2 += c; }
  else { (*Ls).L3 += c; }
}

// A vertex's surface, rebuilt (from the material uniforms) wherever it is needed rather than kept: kept across a
// shadow ray it overflowed the registers of some GPUs into memory, and the tracer ran at a fraction of its speed.
fn surfAt(mat: i32, wo: vec3f, back: bool, lambda: f32, colored: bool, nf: vec3f) -> Surf {
  var s = setupSurf(getMat(mat), wo, back, lambda, colored);
  s.nf = nf;
  return s;
}

// One path per sample, a bounce per iteration of the outer loop: the closest hit, the vertex (a surface, or a
// fluorescence event in a medium), next-event estimation towards each light with its own shadow ray, then the
// BSDF sample that continues the path. The GLSL tracer instead runs every light sample and shadow ray through
// its one scene query (a single call site keeps ANGLE's compile in check); compiled by the D3D12 and Vulkan
// drivers, that one giant loop ran tens of times slower than this conventional shape, which also carries far
// less state from one iteration to the next. The random numbers are drawn exactly as the GLSL tracer draws
// them, and 'steps' counts that loop's iterations so a path ends where it would have.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lane: u32) {
#ifdef MESH
  gLane = lane;
#endif
  let W = pu.dims.x;
  let H = pu.dims.y;
  // This dispatch's slice: every slice-count-th block of rows, from block 'slice'.
  let blockRows = pu.slice.z;
  let y = (pu.slice.x + (gid.y / blockRows) * pu.slice.y) * blockRows + gid.y % blockRows;
  let x = gid.x;
  if (x >= W || y >= H) { return; }
  let fragCoord = vec2f(f32(x) + 0.5, f32(y) + 0.5);
  let resolution = pu.res.xy;
  let pixSeed = pcg3d(vec3u(x, y, 0x9e37u)).x;
  gMultiScatter = select(scene.iparams0.z == 1, fragCoord.x >= scene.fparams2.x * resolution.x, scene.iparams0.w == 1);
  let uPass = scene.iparams0.y;
  let uMaxBounces = pu.iparams.x;
  let uNumLights = scene.iparams1.w;
  let uEnvOn = scene.iparams2.x;
  let uMaxScatter = scene.iparams2.y;
  let uThinGlass = scene.iparams1.z;
  var acc0 = vec3f(0.0);
  var acc1 = vec3f(0.0);
  var acc2 = vec3f(0.0);
  var acc3 = vec3f(0.0);
  let nL = uNumLights + uEnvOn;
  var accMask = 0.0;
  var accAlbedo = vec3f(0.0);
  var accN = vec3f(0.0);
  var accY = 0.0;
  var accY2 = 0.0;
  let sppDone = pu.dims.z;
  let sppNew = pu.dims.w;

  for (var sIdx = 0u; sIdx < sppNew; sIdx++) {
    let sampleIndex = sppDone + sIdx;
    let jitter = sample4(sampleIndex, pixSeed, 0, 7).xy;
    let ndc = ((fragCoord - 0.5 + jitter) / resolution) * 2.0 - 1.0;
    var ro = scene.camPos.xyz;
    var rd = normalize(scene.camFwd.xyz + ndc.x * scene.tanHalf.x * scene.camRight.xyz + ndc.y * scene.tanHalf.y * scene.camUp.xyz);

    var Ls = Lights4(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    var firstAlbedo = vec3f(0.0);
    var firstN = vec3f(0.0);
    var beta = vec3f(1.0);
    var mask = 0.0;
    var prevPdf = 0.0;
    var prevP = ro;
    var prevTrans = false;
    let lambda = 380.0 + 400.0 * sample4(sampleIndex, pixSeed, 0, 6).x;
    var colored = false;
    var depth = 0;
    var maxSteps = uMaxBounces * (nL + 1) + 1;
    var steps = 1; // the camera ray's
#if defined(FLUOR) && defined(GLASS)
    var fromEvent = false;
    maxSteps += 2 * nL + 1;
#endif
#ifdef FLUOR
    var excited = false;
    var flOut = vec3f(0.0);
    var flIn = vec4f(0.0);
    var betaUV = 1.0;
#endif
    var sssCross = 0;
#ifdef SSS
    // A walk's exit keeps the surface it entered by (its material, direction, side and color).
    var sMat = 0;
    var sWo = vec3f(0.0, 0.0, 1.0);
    var sBack = false;
    var sColored = false;
    var inSSS = false;
    var sigT = vec3f(1.0);
    var ssAlb = vec3f(0.0);
    var walkR = vec3f(1.0);
    var heroMask = vec3f(1.0, 0.0, 0.0);
    var hgG = 0.0;
    var betaIn = 1.0;
    var scatters = 0;
    var walkSeed = pcg3d(vec3u(x, y, sampleIndex)).x;
    maxSteps += uMaxScatter;
#endif

    loop {
      // The next hit: through a subsurface medium by a random walk, else the closest along the ray.
      var walkExit = false;
      var h: Hit;
#ifdef SSS
      if (inSSS) {
        // The walk in a loop of its own, querying only the object walked in (see traceWalk): through the
        // path's main loop, each flight carried the whole path's state and ran several times slower.
        var ended = false;
        loop {
          let tFlight = -log(max(1.0 - walkRand(&walkSeed), 1e-30)) / dot(sigT, heroMask);
          let hw = traceWalk(ro, rd, tFlight);
          let scattered = hw.mat < 0;
          let tt = select(hw.t, tFlight, scattered);
          let Tr = exp(-sigT * tt);
          let pc = select(Tr, sigT * Tr, scattered);
          let ph = max(dot(pc, heroMask), 1e-30);
          beta *= select(Tr, ssAlb * sigT * Tr, scattered) / ph;
          walkR *= pc / ph;
          let wn = dot(walkR, vec3f(1.0 / 3.0));
          beta /= wn;
          walkR /= wn;
          if (!scattered) {
            h = hw;
            break;
          }
          scatters++;
          if (scatters > uMaxScatter || maxc(beta) <= 0.0) { ended = true; break; }
          let q = min(maxc(beta) / betaIn, 1.0);
          if (q < 1.0) {
            if (walkRand(&walkSeed) >= q) { ended = true; break; }
            beta /= q;
          }
          ro += rd * tt;
          let r1 = walkRand(&walkSeed);
          let r2 = walkRand(&walkSeed);
          rd = sampleHG(rd, hgG, r1, r2);
          if (steps >= maxSteps) { ended = true; break; }
          steps++;
        }
        if (ended) { break; }
        inSSS = false;
        walkExit = true;
      } else {
#endif
        var T = vec3f(1.0);
#if defined(FLUOR) && defined(GLASS)
        gMediumShadow = false;
        gTuv = 1.0;
        gExitT = 1e30;
#endif
        h = trace(ro, rd, 1e30, false, depth > 0, &T);
#ifdef SSS
      }
#endif

      // Light reached by the path itself.
      if (h.light >= 0) {
        let pl = lightPdf(h.light, prevP, rd, distance(prevP, ro + rd * h.lightT));
        let wL = select(powerHeuristic(prevPdf, pl), 1.0, prevTrans);
        var c = beta * lightRadiance(h.light) * wL;
#ifdef FLUOR
        if (excited) { c = flOut * (dot(flIn.rgb, c) + flIn.a * betaUV * lightUV(h.light) * wL); }
#endif
#if defined(FLUOR) && defined(GLASS)
        if (excited && fromEvent) { c = vec3f(0.0); }
#endif
        addLight(h.light, select(c, clampIndirect(c), depth - sssCross >= 2), &Ls);
      }
      if (h.mat < 0) {
        if (!(depth == 0 && uPass >= 3)) {
          if (uEnvOn == 1) {
            let Lenv = select(envLe(rd), envBackdrop(rd), depth == 0);
            var c = beta * Lenv * select(powerHeuristic(prevPdf, envPdf(rd)), 1.0, depth == 0 || prevTrans);
#ifdef FLUOR
            if (excited) { c = flOut * dot(flIn.rgb, c); }
#endif
#if defined(FLUOR) && defined(GLASS)
            if (excited && fromEvent) { c = vec3f(0.0); }
#endif
            Ls.L3 += select(c, clampIndirect(c), depth - sssCross >= 2);
          } else {
            var c = beta * scene.ambient.xyz;
#ifdef FLUOR
            if (excited) { c = flOut * dot(flIn.rgb, c); }
#endif
            Ls.L0 += select(c, clampIndirect(c), depth - sssCross >= 2);
          }
        }
        break;
      }

      // The vertex.
      let p = ro + rd * h.t;
      let sssExit = walkExit;
      let flip = walkExit && h.back;
      let n = select(h.n, -h.n, flip);
      let ng = select(h.ng, -h.ng, flip);
      let m = getMat(h.mat);
      let vBack = h.back && !walkExit;
      var solidInside = false;
      var evVertex = false; // a fluorescence event in the medium: light it, then go on in a random direction
      var po = offsetRay(p, ng);
#if defined(FLUOR) && defined(GLASS)
      if (!(m.transmission_weight > 0.0 && m.lab_fluor_weight > 0.0)) { fromEvent = false; }
#endif
#ifdef GLASS
      solidInside = vBack && (m.transmission_weight > 0.0 || m.subsurface_weight > 0.0) && m.geometry_thin_walled < 0.5;
#ifdef FLUOR
      if (solidInside && excited) { betaUV *= exp(-m.lab_fluor_uv * h.t); }
      if (solidInside && !excited && m.lab_fluor_weight > 0.0) {
        let sig = vec4f(m.lab_fluor_absorb, m.lab_fluor_uv);
        let cU = sig.a;
        let cV = dot(sig.rgb, vec3f(1.0 / 3.0));
        let ue = sample4(sampleIndex, pixSeed, depth, 5);
        if (ue.x < FLUOR_EVENT_P) {
          let te = truncExpSample(select(cV, cU, ue.x < 0.5 * FLUOR_EVENT_P), h.t, ue.y);
          let pdfT = 0.5 * (truncExpPdf(cU, h.t, te) + truncExpPdf(cV, h.t, te));
          let Tr = select(vec3f(1.0), exp(log(clamp(m.transmission_color, vec3f(1e-4), vec3f(1.0))) / m.transmission_depth * te), m.transmission_depth > 0.0);
          excited = true;
          flOut = beta * Tr * m.lab_fluor_weight * m.lab_fluor_color / (FLUOR_EVENT_P * pdfT);
          flIn = sig;
          beta = vec3f(1.0);
          betaUV = 1.0;
          ro = ro + rd * te;
          rd = sampleSphere(ue.zw);
          po = ro;
          evVertex = true;
          fromEvent = true;
          prevTrans = true;
          prevP = ro;
          depth++;
        } else {
          beta /= 1.0 - FLUOR_EVENT_P;
        }
      }
#endif
      if (!evVertex && solidInside && m.transmission_depth > 0.0) {
        beta *= exp(log(clamp(m.transmission_color, vec3f(1e-4), vec3f(1.0))) / m.transmission_depth * h.t);
      }
#endif

      var t1 = vec3f(1.0, 0.0, 0.0);
      var t2 = vec3f(0.0, 0.0, 1.0);
      var wo = vec3f(0.0, 0.0, 1.0);
      var filt = 0;
      var last = false;
      // The surface's inputs (see surfAt).
      var sfMat = h.mat;
      var sfWo = vec3f(0.0, 0.0, 1.0);
      var sfBack = h.back;
      var sfColored = colored;
      var sfNf = vec3f(0.0, 0.0, 1.0);
#ifdef FLUOR_SURFACE
      var pF = 0.0;
      var flEmit = vec3f(0.0);
      var flAbs = vec4f(0.0);
#endif
      if (!evVertex) {
        if (depth == 0) {
          mask = select(0.0, 1.0, h.mat != 3);
          firstAlbedo = albedoAOV(m);
          firstN = n;
          if (uPass == 3) { Ls.L0 = firstAlbedo; break; }
          if (uPass == 4) { Ls.L0 = n * 0.5 + 0.5; break; }
        }
        filt = select(0, select(select(0, 2, uPass == 2), 1, uPass == 1), depth == 0);
        let B = onb(n);
        t1 = B.b1;
        t2 = B.b2;
        if (m.specular_roughness_anisotropy > 0.0 && !walkExit) {
          let Tg = vec3f(0.0, 1.0, 0.0) - n * n.y;
          let lT = length(Tg);
          if (lT > 1e-4) {
            t1 = Tg / lT;
            t2 = cross(n, t1);
          }
        }
        wo = vec3f(dot(-rd, t1), dot(-rd, t2), dot(-rd, n));
        if (wo.z <= 1e-6 && !walkExit) { break; }
#ifdef SSS
        if (!walkExit) {
          sMat = h.mat;
          sWo = wo;
          sBack = h.back;
          sColored = colored;
        }
        sfMat = sMat;
        sfWo = sWo;
        sfBack = sBack;
        sfColored = sColored;
#else
        sfWo = wo;
#endif
#ifdef MESH
        sfNf = flakeNormal(m, transpose(modelRot()) * (p - scene.modelPos.xyz));
#else
        sfNf = flakeNormal(m, p - vec3f(ballX(2), 1.0, 0.0));
#endif
#ifdef FLUOR_SURFACE
        flEmit = select(fluorBase(surfAt(sfMat, sfWo, sfBack, lambda, sfColored, sfNf)) * m.lab_fluor_weight * m.lab_fluor_color, vec3f(0.0), filt == 2);
        flAbs = vec4f(m.lab_fluor_absorb, m.lab_fluor_uv);
        pF = select(0.0, FLUOR_LOBE_P, !excited && maxc(flEmit) > 0.0 && dot(flAbs, vec4f(1.0)) > 0.0);
#endif
        last = depth == uMaxBounces - 1;
      }

      // Next-event estimation: each light, with its own shadow ray.
      var capped = false;
      for (var k = 0; k < nL; k++) {
        if (steps >= maxSteps) { capped = true; break; }
        steps++;
#if defined(FLUOR) && defined(GLASS)
        if (!lightOn(k) || (solidInside && !evVertex)) { continue; }
        let u = sample4(sampleIndex, pixSeed, select(depth, depth + 16, evVertex), k);
#else
        if (!lightOn(k) || solidInside) { continue; }
        let u = sample4(sampleIndex, pixSeed, depth, k);
#endif
        var dirW: vec3f;
        var ldist: f32;
        var lpdf: f32;
        if (k == ENV_LIGHT) {
          let es = sampleEnv(u);
          if (!es.ok) { continue; }
          dirW = es.wi;
          lpdf = es.pdf;
          ldist = 1e30;
        } else {
          let ls = sampleLight(k, po, u.xy);
          if (!ls.ok) { continue; }
          dirW = ls.wi;
          ldist = ls.dist;
          lpdf = ls.pdf;
        }
        let wi = vec3f(dot(dirW, t1), dot(dirW, t2), dot(dirW, n));
        if (wi.z <= 0.0 && !evVertex) { continue; }
        var fD: vec3f;
        var fS = vec3f(0.0);
        var bpdf: f32;
        if (evVertex) {
          fD = vec3f(1.0 / (4.0 * PI));
          bpdf = 1.0 / (4.0 * PI);
        } else if (sssExit) {
          fD = vec3f(wi.z / PI);
          bpdf = wi.z / PI;
        } else {
          let ev = evalSurf(surfAt(sfMat, sfWo, sfBack, lambda, sfColored, sfNf), wo, wi, filt);
          fD = ev.fD;
          fS = ev.fS;
          bpdf = ev.pdf;
#ifdef FLUOR_SURFACE
          bpdf *= 1.0 - pF;
#endif
        }
        let f = fD + fS;
#ifdef FLUOR_SURFACE
        if (maxc(f) <= 0.0 && pF <= 0.0) { continue; }
#else
        if (maxc(f) <= 0.0) { continue; }
#endif
        let wl = select(powerHeuristic(lpdf, bpdf), 1.0, last);
        var Le: vec3f;
        if (k == ENV_LIGHT) { Le = envLe(dirW); } else { Le = lightRadiance(k); }
        var pending = beta * f * Le * (wl / lpdf);
#ifdef FLUOR
        let Luv = select(lightUV(min(k, 2)), 0.0, k == ENV_LIGHT);
        if (!evVertex) {
          if (excited) {
            pending = flOut * (dot(flIn.rgb, pending) + flIn.a * betaUV * uvAlbedo(fD, fS, m) * Luv * (wl / lpdf));
          }
#ifdef FLUOR_SURFACE
          else if (pF > 0.0) {
            let wf = select(powerHeuristic(lpdf, pF * wi.z / PI), 1.0, last);
            pending += beta * flEmit * (wi.z / PI) * (dot(flAbs.rgb, Le) + flAbs.a * Luv) * (wf / lpdf);
          }
#endif
        }
#endif
        var Ts = vec3f(1.0);
#if defined(FLUOR) && defined(GLASS)
        gMediumShadow = evVertex;
        gTuv = 1.0;
        gExitT = 1e30;
#endif
        let hs = trace(po, dirW, ldist * (1.0 - 1e-4), true, false, &Ts);
        if (hs.mat < 0 && maxc(Ts) > 0.0) {
          var c = pending * Ts;
#if defined(FLUOR) && defined(GLASS)
          if (evVertex) {
            let mm = loadMat(2u);
            let dIn = select(0.0, gExitT, gExitT < 1e29);
            let Trgb = select(vec3f(1.0), exp(log(clamp(mm.transmission_color, vec3f(1e-4), vec3f(1.0))) / mm.transmission_depth * dIn), mm.transmission_depth > 0.0);
            c = flOut * (dot(flIn.rgb, Le * Ts * Trgb) + flIn.a * Luv * gTuv * exp(-mm.lab_fluor_uv * dIn)) / (4.0 * PI * lpdf);
          }
#endif
          addLight(k, select(c, clampIndirect(c), depth - sssCross >= 1), &Ls);
        }
      }
      // The surface and material for the BSDF sample, fetched again rather than kept across the shadow rays, and
      // here rather than after the exits below: either way round, the D3D12 and Vulkan drivers' code for this
      // kernel ran several times slower (measured on an RDNA 2 GPU against the GLSL tracer's time).
      let s = surfAt(sfMat, sfWo, sfBack, lambda, sfColored, sfNf);
      let mv = getMat(h.mat);
      if (capped || steps >= maxSteps) { break; }
      steps++;
      if (evVertex) { continue; } // on from the event, in the direction drawn with it
      if (last) { break; }

      // The BSDF sample that continues the path.
      let u = sample4(sampleIndex, pixSeed, depth, nL);
      var wi: vec3f;
#ifdef FLUOR_SURFACE
      var flLobe = false;
#endif
      var sampled = true;
#ifdef SSS
      if (sssExit) { wi = sampleCosine(u.xy); sampled = false; }
#endif
#ifdef FLUOR_SURFACE
      flLobe = pF > 0.0 && sample4(sampleIndex, pixSeed, depth, 5).x < pF;
      if (flLobe) { wi = sampleCosine(u.xy); sampled = false; }
#endif
      if (sampled) {
        let ss = sampleSurf(s, wo, filt, u.xyz);
        if (!ss.ok) { break; }
        wi = ss.wi;
      }
      let dirW = normalize(t1 * wi.x + t2 * wi.y + n * wi.z);
      var fD: vec3f;
      var fS = vec3f(0.0);
      var bpdf: f32;
      var evaluated = false;
#ifdef SSS
      if (sssExit) {
        fD = vec3f(wi.z / PI);
        bpdf = wi.z / PI;
        evaluated = true;
      }
#endif
#ifdef FLUOR_SURFACE
      if (flLobe) {
        fD = flEmit * (wi.z / PI);
        bpdf = pF * wi.z / PI;
        evaluated = true;
      }
#endif
      if (!evaluated) {
        let ev = evalSurf(s, wo, wi, filt);
        fD = ev.fD;
        fS = ev.fS;
        bpdf = ev.pdf;
#ifdef FLUOR_SURFACE
        bpdf *= 1.0 - pF;
#endif
      }
      let f = fD + fS;
      if (bpdf <= 0.0) { break; }
#ifdef FLUOR
      if (excited) { betaUV *= uvAlbedo(fD, fS, mv) / bpdf; }
#endif
      beta *= f / bpdf;
#ifdef FLUOR_SURFACE
      if (flLobe) {
        excited = true;
        flOut = beta;
        flIn = flAbs;
        beta = vec3f(1.0);
        betaUV = 1.0;
      }
#endif
#ifdef FLUOR
      if (maxc(beta) <= 0.0 && !(excited && betaUV > 0.0)) { break; }
#else
      if (maxc(beta) <= 0.0) { break; }
#endif
      if (depth >= 3) {
        var q = min(maxc(beta), 0.95);
#ifdef FLUOR
        if (excited) { q = min(maxc(flOut) * max(maxc(beta * flIn.rgb), betaUV * flIn.a), 0.95); }
#endif
        if (u.w >= q) { break; }
        beta /= q;
#ifdef FLUOR
        betaUV /= q;
#endif
      }
      let through = wi.z < 0.0;
#if defined(GLASS) && !defined(NO_DISPERSION)
      if (through && !colored && disperses(mv)) {
        colored = true;
        beta *= spectralWeight(lambda);
      }
#endif
      ro = select(po, offsetRay(p, -ng), through);
#ifdef THIN
      let passed = through && s.thin && uThinGlass == 1 && smoothThin(mv);
#else
      let passed = false;
#endif
#ifdef SSS
      if (through && !sssExit && !s.thin && !vBack && mv.subsurface_weight > 0.0) {
        inSSS = true;
        sssCross++;
        beta *= sq(s.etaT);
        betaIn = max(maxc(beta), 1e-20);
        let uh = walkRand(&walkSeed) * 3.0;
        heroMask = vec3f(f32(uh < 1.0), f32(uh >= 1.0 && uh < 2.0), f32(uh >= 2.0));
        walkR = vec3f(1.0);
        let r = max(mv.subsurface_radius * mv.subsurface_radius_scale, vec3f(1e-6));
        sigT = 1.0 / r;
        hgG = clamp(mv.subsurface_scatter_anisotropy, -0.95, 0.95);
        let C = sat3(mv.subsurface_color);
        let sv = 4.09712 + 4.20863 * C - sqrt(9.59217 + 41.6808 * C + 17.7126 * C * C);
        ssAlb = sat3((1.0 - sv * sv) / (1.0 - hgG * sv * sv));
      }
#endif
      if (!passed) {
        prevPdf = bpdf;
        prevTrans = through;
        prevP = ro;
      }
      rd = dirW;
      depth++;
    }
    let all = Ls.L0 + Ls.L1 + Ls.L2 + Ls.L3;
    if (!(nonFinite(all.r) || nonFinite(all.g) || nonFinite(all.b))) {
      acc0 += Ls.L0;
      acc1 += Ls.L1;
      acc2 += Ls.L2;
      acc3 += Ls.L3;
      let Y = lum(all);
      accY += Y;
      accY2 += Y * Y;
    }
    accMask += mask;
    accAlbedo += firstAlbedo;
    accN += firstN;
  }

  // The running means: the new samples blend in with weight new / (done + new), in place.
  let nNew = f32(sppNew);
  let w = select(nNew / (f32(sppDone) + nNew), 1.0, sppDone == 0u);
  let N = W * H;
  let idx = y * W + x;
  let cur0 = vec4f(acc0 / nNew, accMask / nNew);
  let cur1 = vec4f(acc1 / nNew, accY2 / nNew);
  let cur2 = vec4f(acc2 / nNew, 1.0);
  let cur3 = vec4f(acc3 / nNew, 1.0);
  let cur4 = vec4f(accAlbedo / nNew, accY / nNew);
  let cur5 = vec4f(accN / nNew, 1.0);
  if (sppDone == 0u) {
    accum[idx] = cur0;
    accum[N + idx] = cur1;
    accum[2u * N + idx] = cur2;
    accum[3u * N + idx] = cur3;
    aux[idx] = cur4;
    aux[N + idx] = cur5;
  } else {
    accum[idx] = mix(accum[idx], cur0, w);
    accum[N + idx] = mix(accum[N + idx], cur1, w);
    accum[2u * N + idx] = mix(accum[2u * N + idx], cur2, w);
    accum[3u * N + idx] = mix(accum[3u * N + idx], cur3, w);
    aux[idx] = mix(aux[idx], cur4, w);
    aux[N + idx] = mix(aux[N + idx], cur5, w);
  }
}
`

// The tracer for a variant: the same switches as traceFrag's (see shaders.ts).
export function traceWGSL(mesh: boolean, glass: 'none' | 'thin' | 'full' | 'sss' | 'fluor' | 'fluorglass') {
  const defs: string[] = []
  if (mesh) defs.push('MESH')
  if (glass === 'full' || glass === 'sss' || glass === 'fluorglass') defs.push('GLASS')
  if (glass === 'sss' || glass === 'fluorglass') defs.push('NO_DISPERSION')
  if (glass === 'sss') defs.push('SSS')
  if (glass !== 'none' && glass !== 'fluor') defs.push('THIN')
  if (glass === 'fluor' || glass === 'fluorglass') defs.push('FLUOR')
  if (glass === 'fluor') defs.push('FLUOR_SURFACE')
  const sceneQuery = SCENE_QUERY.replace('${MESH_PLACEHOLDER}', '#ifdef MESH\n' + MESH + '\n#endif')
  const src = [COMMON, BINDINGS, SAMPLING, MICROFACET, TABLE_CONSTS, EON, FUZZ, THIN_FILM, OPENPBR, MAT_UNPACK, LIGHTS, sceneQuery, KERNEL].join('\n')
  return preprocess(src, defs)
}
