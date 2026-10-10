// The labs' color pipeline, shared by the Lookdev lab and the Pyro lab so the two read as one camera: the ACEScg
// (AP1) working space, its matrices to and from linear Rec.709 (OpenColorIO 2.5's built-in
// cg-config-v4.0.0_aces-v2.0_ocio-v2.5, Bradford D65 <-> D60), the luminance of an ACEScg color, and the view
// transforms that put it on screen: ACES 2.0 (its SDR Output Transform baked from OpenColorIO 2.5 into a 65^3 LUT, a
// PNG strip), AgX, Khronos PBR Neutral, or a plain sRGB curve ("Standard").

import { LUT_SIZE, LUT_URL } from '../lookdev/scene'

// Rows (row-major).
export const REC709_TO_AP1 = [
  [0.6130974, 0.3395231, 0.0473795],
  [0.0701937, 0.9163539, 0.0134524],
  [0.0206156, 0.1095698, 0.8698146],
]
export const AP1_TO_REC709 = [
  [1.705051, -0.6217921, -0.0832589],
  [-0.1302564, 1.1408048, -0.0105483],
  [-0.0240034, -0.128969, 1.1529723],
]
// The Rec.709 luminance row pushed through AP1_TO_REC709, so a color's luminance is the same in either space.
export const AP1_LUMINANCE = [0.2676014, 0.674399, 0.0579996]
export const ap1Luminance = (c: number[]) => AP1_LUMINANCE[0] * c[0] + AP1_LUMINANCE[1] * c[1] + AP1_LUMINANCE[2] * c[2]
export const mat3Apply = (m: number[][], v: number[]) => m.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2])
export const mat3Mul = (a: number[][], b: number[][]) => a.map(r => b[0].map((_, j) => r[0] * b[0][j] + r[1] * b[1][j] + r[2] * b[2][j]))

// WGSL's mat3x3f takes columns.
const wgslMat = (m: number[][]) => `mat3x3f(${[0, 1, 2].map(j => m.map(r => r[j].toFixed(7)).join(', ')).join(',\n  ')})`
// For a shader that does not already have them (the Lookdev lab's declares its own, the same numbers).
export const AP1_WGSL = /* wgsl */ `
const AP1_TO_REC709 = ${wgslMat(AP1_TO_REC709)};
const REC709_TO_AP1 = ${wgslMat(REC709_TO_AP1)};
fn ap1Lum(c: vec3f) -> f32 { return dot(c, vec3f(${AP1_LUMINANCE.map(v => v.toFixed(7)).join(', ')})); }
`

export type ViewTransform = 'aces' | 'agx' | 'neutral' | 'standard'
export const VIEW_TRANSFORMS: { id: ViewTransform; label: string }[] = [
  { id: 'aces', label: 'ACES 2.0' },
  { id: 'agx', label: 'AgX' },
  { id: 'neutral', label: 'PBR Neutral' },
  { id: 'standard', label: 'Standard' },
]
// The index a shader takes for each (0 to 3).
export const viewIndex = (v: ViewTransform) => VIEW_TRANSFORMS.findIndex(x => x.id === v)

// The view transforms, in WGSL: srgbOETF, PBRNeutralToneMapping and AgXToneMapping take linear Rec.709; aces2 takes
// ACEScg (exposed, 0.18 for middle grey) and returns display values, sRGB encoded. aces2 reads the LUT through
// bindings the including shader declares, named lut (texture_2d<f32>) and lutSampler.
export const VIEW_TRANSFORMS_WGSL = /* wgsl */ `fn srgbOETF(Lin: vec3f) -> vec3f {
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

`

export { LUT_SIZE, LUT_URL }

// The ACES 2.0 LUT as a WebGPU texture (the PNG strip, N wide and N x N tall), or null if it cannot be had.
export async function loadAcesLutGPU(device: GPUDevice): Promise<GPUTexture | null> {
  try {
    const blob = await (await fetch(LUT_URL)).blob()
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
    if (bitmap.width !== LUT_SIZE || bitmap.height !== LUT_SIZE * LUT_SIZE) throw new Error('Unexpected LUT size')
    const lut = device.createTexture({
      size: [bitmap.width, bitmap.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: lut, premultipliedAlpha: false }, [bitmap.width, bitmap.height])
    bitmap.close()
    return lut
  } catch (err) {
    console.warn('ACES 2.0 LUT unavailable; using AgX', err)
    return null
  }
}
