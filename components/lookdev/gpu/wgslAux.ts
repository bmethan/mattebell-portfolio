// WGSL ports of the lab's other GPU programs (shaders.ts): the albedo tables, the environment's mipmaps (WebGPU has
// no generateMipmap), the display transform and the denoiser. Comments on their sources live with the GLSL.
import { COMMON } from './wgslCommon'
import { MICROFACET, TABLE_CONSTS } from './wgslBsdf'
import { UniformLayout } from './layout'

// ------------------------------------------------------------------------------------------------------------
// Albedo tables (E_TABLE_FRAG): both at once, texel (mu, roughness) of IOR layer z. R/G/B as in the GLSL.
// ------------------------------------------------------------------------------------------------------------
export const E_TABLE_WGSL = /* wgsl */ `
${COMMON}
${MICROFACET}
${TABLE_CONSTS}
@group(0) @binding(0) var tableR: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var tableT: texture_storage_3d<rgba16float, write>;
const N_SAMPLES: u32 = 2048u;
const N_FRESNEL: u32 = 256u;

fn dielectricAlbedo(wo: vec3f, m: vec3f, alpha: vec2f, eta: f32) -> f32 {
  let F = fresnelDielectric(dot(wo, m), eta);
  var e = 0.0;
  let wr = reflect(-wo, m);
  if (wr.z > 0.0) { e += F * G2_GGX(wo, wr, alpha) / G1_GGX(wo, alpha); }
  let wt = refract(-wo, m, 1.0 / eta);
  if (dot(wt, wt) > 0.0 && wt.z < 0.0) { e += (1.0 - F) * G2_GGX(wo, wt, alpha) / G1_GGX(wo, alpha); }
  return e;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = vec3u(E_DIMS);
  if (any(gid >= dims)) { return; }
  let uv = vec2f(gid.xy) / (E_DIMS.xy - 1.0); // texel i holds mu = i / (N - 1)
  let mu = max(uv.x, 1e-4);
  let r = uv.y;
  let alpha = vec2f(max(r * r, 1e-4));
  let x = f32(gid.z) / (E_DIMS.z - 1.0) * E_X_MAX;
  let eta = (1.0 + x) / (1.0 - x);
  let wo = vec3f(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
  var eIn = 0.0;
  var eOut = 0.0;
  var e1 = 0.0;
  var ed = 0.0;
  for (var i = 0u; i < N_SAMPLES; i++) {
    let u = vec2f((f32(i) + 0.5) / f32(N_SAMPLES), u01(reverseBits(i)));
    let m = sampleVNDF_SphericalCap(u, wo, alpha);
    eIn += dielectricAlbedo(wo, m, alpha, eta);
    eOut += dielectricAlbedo(wo, m, alpha, 1.0 / eta);
    let wi = reflect(-wo, m);
    if (wi.z > 0.0) {
      let w = G2_GGX(wo, wi, alpha) / G1_GGX(wo, alpha);
      e1 += w;
      ed += w * fresnelDielectric(dot(wo, m), eta);
    }
  }
  var favg = 0.0;
  for (var j = 0u; j < N_FRESNEL; j++) {
    let c = (f32(j) + 0.5) / f32(N_FRESNEL);
    favg += fresnelDielectric(c, eta) * c;
  }
  favg *= 2.0 / f32(N_FRESNEL);
  let n = f32(N_SAMPLES);
  textureStore(tableR, gid, vec4f(e1 / n, ed / n, favg, 1.0));
  textureStore(tableT, gid, vec4f(eIn / n, eOut / n, 0.0, 1.0));
}
`

// ------------------------------------------------------------------------------------------------------------
// Environment mipmaps: each level the 2 x 2 box average of the one above (as generateMipmap).
// ------------------------------------------------------------------------------------------------------------
export const MIP_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = textureDimensions(dst);
  if (gid.x >= d.x || gid.y >= d.y) { return; }
  let s = vec2i(textureDimensions(src)) - 1;
  let p = vec2i(gid.xy) * 2;
  let c = textureLoad(src, min(p, s), 0) + textureLoad(src, min(p + vec2i(1, 0), s), 0) +
    textureLoad(src, min(p + vec2i(0, 1), s), 0) + textureLoad(src, min(p + vec2i(1, 1), s), 0);
  textureStore(dst, gid.xy, c * 0.25);
}
`

// ------------------------------------------------------------------------------------------------------------
// Display (DISPLAY_FRAG): the mixed light images, the denoiser's share, exposure and a view transform, onto the
// canvas. The images' rows go bottom up; the canvas's top down.
// ------------------------------------------------------------------------------------------------------------
export const DISPLAY = new UniformLayout('DisplayU', [
  { name: 'mixW', kind: 'vec4f', count: 4 }, // the light mixer's weights (key, fill, rim, environment)
  { name: 'params', kind: 'vec4f' }, // exposure (EV), denoiser share, split (canvas px)
  { name: 'iparams', kind: 'vec4i' }, // view, pass, compare, ACES LUT ready
  { name: 'dims', kind: 'vec4u' }, // source width, height, its proxy divisor, canvas height
])

export const DISPLAY_WGSL = /* wgsl */ `
${COMMON}
${DISPLAY.wgsl()}
@group(0) @binding(0) var<uniform> du: DisplayU;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var<storage, read> denoised: array<vec4f>;
@group(0) @binding(3) var lut: texture_2d<f32>; // the ACES 2.0 LUT: 65 slices of 65 x 65 stacked (blue by slice)
@group(0) @binding(4) var lutSampler: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn srgbOETF(Lin: vec3f) -> vec3f {
  let L = clamp(Lin, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(L, vec3f(1.0 / 2.4)) - 0.055, 12.92 * L, L <= vec3f(0.0031308));
}
fn PBRNeutralToneMapping(colorIn: vec3f) -> vec3f {
  let startCompression = 0.8 - 0.04;
  let desaturation = 0.15;
  let x = min(colorIn.r, min(colorIn.g, colorIn.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  var color = colorIn - offset;
  let peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) { return color; }
  let d = 1.0 - startCompression;
  let newPeak = 1.0 - d * d / (peak + d - startCompression);
  color *= newPeak / peak;
  let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(color, newPeak * vec3f(1.0), g);
}
const LINEAR_REC2020_TO_LINEAR_SRGB = mat3x3f(
  vec3f( 1.6605, -0.1246, -0.0182),
  vec3f(-0.5876,  1.1329, -0.1006),
  vec3f(-0.0728, -0.0083,  1.1187));
const LINEAR_SRGB_TO_LINEAR_REC2020 = mat3x3f(
  vec3f(0.6274, 0.0691, 0.0164),
  vec3f(0.3293, 0.9195, 0.0880),
  vec3f(0.0433, 0.0114, 0.8956));
fn agxDefaultContrastApprox(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn AgXToneMapping(colorIn: vec3f) -> vec3f {
  let AgXInsetMatrix = mat3x3f(
    vec3f(0.856627153315983, 0.137318972929847, 0.11189821299995),
    vec3f(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
    vec3f(0.0482516061458583, 0.101439036467562, 0.811302368396859));
  let AgXOutsetMatrix = mat3x3f(
    vec3f(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
    vec3f(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
    vec3f(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
  let AgxMinEv = -12.47393;
  let AgxMaxEv = 4.026069;
  var color = LINEAR_SRGB_TO_LINEAR_REC2020 * colorIn;
  color = AgXInsetMatrix * color;
  color = max(color, vec3f(1e-10));
  color = log2(color);
  color = (color - AgxMinEv) / (AgxMaxEv - AgxMinEv);
  color = clamp(color, vec3f(0.0), vec3f(1.0));
  color = agxDefaultContrastApprox(color);
  color = AgXOutsetMatrix * color;
  color = pow(max(vec3f(0.0), color), vec3f(2.2));
  color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
  return clamp(color, vec3f(0.0), vec3f(1.0));
}
// The 3D LUT stored as a strip of slices: bilinear within the two slices around blue, then between them.
fn aces2(ap1: vec3f) -> vec3f {
  let MIN_EV = -12.0;
  let MAX_EV = 10.0;
  let N = 65.0;
  let s = clamp((log2(max(ap1, vec3f(1e-10)) / 0.18) - MIN_EV) / (MAX_EV - MIN_EV), vec3f(0.0), vec3f(1.0));
  let zf = s.b * (N - 1.0);
  let z0 = floor(zf);
  let z1 = min(z0 + 1.0, N - 1.0);
  let u = (s.r * (N - 1.0) + 0.5) / N;
  let v0 = (z0 * N + s.g * (N - 1.0) + 0.5) / (N * N);
  let v1 = (z1 * N + s.g * (N - 1.0) + 0.5) / (N * N);
  let c0 = textureSampleLevel(lut, lutSampler, vec2f(u, v0), 0.0).rgb;
  let c1 = textureSampleLevel(lut, lutSampler, vec2f(u, v1), 0.0).rgb;
  return mix(c0, c1, zf - z0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let canvasH = du.dims.w;
  let k = du.dims.z;
  let px = vec2u(u32(pos.x), canvasH - 1u - u32(pos.y)) / k;
  let W = du.dims.x;
  let N = W * du.dims.y;
  let idx = min(px.y, du.dims.y - 1u) * W + min(px.x, W - 1u);
  var c = accum[idx].rgb;
  let passId = du.iparams.y;
  if (passId == 4) { return vec4f(c, 1.0); }
  if (passId == 3) { return vec4f(srgbOETF(max(AP1_TO_REC709 * c, vec3f(0.0))), 1.0); }
  c = c * du.mixW[0].rgb + accum[N + idx].rgb * du.mixW[1].rgb + accum[2u * N + idx].rgb * du.mixW[2].rgb +
    accum[3u * N + idx].rgb * du.mixW[3].rgb;
  let right = pos.x >= du.params.z;
  let compare = du.iparams.z;
  let dm = select(du.params.y, select(0.0, 1.0, right), compare == 1);
  if (dm > 0.0) { c = mix(c, denoised[idx].rgb, dm); }
  let view = select(du.iparams.x, select(0, 1, right), compare == 2);
  c = max(c, vec3f(0.0)) * exp2(du.params.x);
  if (view == 0 && du.iparams.w == 1) { return vec4f(aces2(c), 1.0); }
  var r = max(AP1_TO_REC709 * c, vec3f(0.0));
  if (view == 2) { r = PBRNeutralToneMapping(r); }
  else if (view != 3) { r = AgXToneMapping(r); }
  return vec4f(srgbOETF(r), 1.0);
}
`

// ------------------------------------------------------------------------------------------------------------
// Denoiser (DENOISE_PREP_FRAG, ATROUS_FRAG): a-trous wavelet with an SVGF variance guide, on storage buffers.
// ------------------------------------------------------------------------------------------------------------
export const DENOISE = new UniformLayout('DenoiseU', [
  { name: 'mixW', kind: 'vec4f', count: 4 },
  { name: 'params', kind: 'vec4f' }, // samples per pixel in the image
  { name: 'iparams', kind: 'vec4i' }, // width, height, step, last level
])
export const DENOISE_STRIDE = 256

const DENOISE_COMMON = /* wgsl */ `
${COMMON}
${DENOISE.wgsl()}
@group(0) @binding(0) var<uniform> dn: DenoiseU;
fn sizeI() -> vec2i { return dn.iparams.xy; }
fn clampPx(q: vec2i) -> vec2i { return clamp(q, vec2i(0), sizeI() - 1); }
fn at(q: vec2i) -> i32 { return q.y * sizeI().x + q.x; }
fn demodBase(a: vec3f) -> vec3f { return select(max(a, vec3f(0.02)), vec3f(1.0), lum(a) < 1e-3); }
`

export const DENOISE_PREP_WGSL = /* wgsl */ `
${DENOISE_COMMON}
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var<storage, read> aux: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outColor: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outGuide: array<vec4f>;
fn plane(p: i32) -> i32 { return p * sizeI().x * sizeI().y; }
fn demod(q: vec2i) -> vec3f {
  let i = at(q);
  let c = accum[i].rgb * dn.mixW[0].rgb + accum[plane(1) + i].rgb * dn.mixW[1].rgb + accum[plane(2) + i].rgb * dn.mixW[2].rgb +
    accum[plane(3) + i].rgb * dn.mixW[3].rgb;
  return max(c, vec3f(0.0)) / demodBase(aux[i].rgb);
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (any(p >= sizeI())) { return; }
  let d = demod(p);
  let i = at(p);
  let aux0 = aux[i];
  let n = dn.params.x;
  var vr: f32;
  if (n >= 4.0) {
    let m1 = aux0.a;
    let m2 = accum[plane(1) + i].a;
    let s = lum(d) / max(m1, 1e-6);
    vr = max(m2 - m1 * m1, 0.0) / (n - 1.0) * s * s;
  } else {
    var sum = 0.0;
    var sum2 = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let l = lum(demod(clampPx(p + vec2i(dx, dy))));
        sum += l;
        sum2 += l * l;
      }
    }
    vr = max(sum2 / 9.0 - sq(sum / 9.0), 0.0) / max(n, 1.0);
  }
  let nrm = aux[plane(1) + i].rgb;
  let len = length(nrm);
  outColor[i] = vec4f(d, vr);
  outGuide[i] = vec4f(select(nrm / len, vec3f(0.0), len < 0.1), lum(aux0.rgb));
}
`

export const ATROUS_WGSL = /* wgsl */ `
${DENOISE_COMMON}
@group(0) @binding(1) var<storage, read> inColor: array<vec4f>;
@group(0) @binding(2) var<storage, read> guide: array<vec4f>;
@group(0) @binding(3) var<storage, read> aux: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outColor: array<vec4f>;
const SIGMA_L: f32 = 4.0;
const SIGMA_N: f32 = 128.0;
const SIGMA_A: f32 = 0.05;
fn hk(i: i32) -> f32 { return select(0.25, 0.5, i == 0); }
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (any(p >= sizeI())) { return; }
  let stepPx = dn.iparams.z;
  let cP = inColor[at(p)];
  let gP = guide[at(p)];
  let bgP = dot(gP.xyz, gP.xyz) < 0.01;
  let lP = lum(cP.rgb);
  var v = cP.a;
  if (stepPx == 1) {
    v = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) { v += hk(dx) * hk(dy) * inColor[at(clampPx(p + vec2i(dx, dy)))].a; }
    }
  }
  let sigL = SIGMA_L * sqrt(max(v, 0.0)) + 1e-5;
  var sum = vec3f(0.0);
  var wSum = 0.0;
  var vSum = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = p + vec2i(dx, dy) * stepPx;
      if (any(q < vec2i(0)) || any(q >= sizeI())) { continue; }
      let cQ = inColor[at(q)];
      let gQ = guide[at(q)];
      let bgQ = dot(gQ.xyz, gQ.xyz) < 0.01;
      let wN = select(pow(max(dot(gP.xyz, gQ.xyz), 0.0), SIGMA_N), f32(bgP && bgQ), bgP || bgQ);
      let wA = exp(-abs(gP.w - gQ.w) / SIGMA_A);
      let wL = exp(-abs(lP - lum(cQ.rgb)) / sigL);
      let w = hk(dx) * hk(dy) * wN * wA * wL;
      sum += w * cQ.rgb;
      wSum += w;
      vSum += w * w * cQ.a;
    }
  }
  var c = sum / wSum;
  let vr = vSum / (wSum * wSum);
  if (dn.iparams.w == 1) { c *= demodBase(aux[at(p)].rgb); }
  outColor[at(p)] = vec4f(c, vr);
}
`
