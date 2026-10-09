// Blackbody light for the pyro lab, in absolute photometric units: Planck's spectral radiance integrated against the
// CIE 1931 colour matching functions (the multi-lobe Gaussian fit of Wyman, Sloan and Shirley 2013, as the lookdev
// lab uses) from 380 to 780 nm, times 683 lm/W, as linear Rec.709 in cd/m2 (so green carries most of the luminance).
// Hot soot emits its absorption coefficient times this (Kirchhoff's law), so a flame's brightness follows from its
// temperature and soot alone. Also the camera's white balance: a Bradford chromatic adaptation from a blackbody white
// to the display's D65.

export const BB_MIN_K = 500
export const BB_MAX_K = 3000
export const BB_SIZE = 256

const H = 6.62607015e-34 // J s
const C = 299792458 // m/s
const C2 = 1.438777e-2 // m K, hc / k
const LM_PER_W = 683

const lobe = (w: number, mu: number, a: number, b: number) => {
  const t = (w - mu) * (w < mu ? a : b)
  return Math.exp(-0.5 * t * t)
}
const cieXYZ = (w: number): [number, number, number] => [
  0.362 * lobe(w, 442.0, 0.0624, 0.0374) + 1.056 * lobe(w, 599.8, 0.0264, 0.0323) - 0.065 * lobe(w, 501.1, 0.049, 0.0382),
  0.821 * lobe(w, 568.8, 0.0213, 0.0247) + 0.286 * lobe(w, 530.9, 0.0613, 0.0322),
  1.217 * lobe(w, 437.0, 0.0845, 0.0278) + 0.681 * lobe(w, 459.0, 0.0385, 0.0725),
]
// IEC 61966-2-1 XYZ to linear sRGB (Rec.709 primaries, D65 white), and back.
const XYZ_TO_REC709 = [
  [3.2406255, -1.537208, -0.4986286],
  [-0.9689307, 1.8757561, 0.0415175],
  [0.0557101, -0.2040211, 1.0569959],
]
const REC709_TO_XYZ = [
  [0.4124, 0.3576, 0.1805],
  [0.2126, 0.7152, 0.0722],
  [0.0193, 0.1192, 0.9505],
]
const D65: [number, number, number] = [0.95047, 1, 1.08883]

type M3 = number[][]
const mul = (a: M3, b: M3): M3 => a.map(r => b[0].map((_, j) => r[0] * b[0][j] + r[1] * b[1][j] + r[2] * b[2][j]))
const apply = (m: M3, v: number[]) => m.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2])

// Spectral radiance, W / (m2 sr m), at a wavelength in nm.
const planck = (nm: number, kelvin: number) => {
  const l = nm * 1e-9
  return (2 * H * C * C) / (l ** 5 * Math.expm1(C2 / (l * kelvin)))
}

// Luminance-weighted XYZ of a blackbody, cd/m2 (Y is its luminance).
export function blackbodyXYZ(kelvin: number): [number, number, number] {
  const s: [number, number, number] = [0, 0, 0]
  const step = 2 // nm
  for (let nm = 380; nm <= 780; nm += step) {
    const b = planck(nm, kelvin) * step * 1e-9 * LM_PER_W
    const c = cieXYZ(nm)
    s[0] += b * c[0]
    s[1] += b * c[1]
    s[2] += b * c[2]
  }
  return s
}

// BB_SIZE entries from BB_MIN_K to BB_MAX_K, as rgba: linear Rec.709 radiance in cd/m2 (a = luminance). Below about
// 1900 K a blackbody's colour lies outside Rec.709; negative components clip to 0.
export function planckTable(): Float32Array<ArrayBuffer> {
  const out = new Float32Array(BB_SIZE * 4)
  for (let i = 0; i < BB_SIZE; i++) {
    const k = BB_MIN_K + ((BB_MAX_K - BB_MIN_K) * i) / (BB_SIZE - 1)
    const xyz = blackbodyXYZ(k)
    const rgb = apply(XYZ_TO_REC709, xyz).map(v => Math.max(0, v))
    out.set([rgb[0], rgb[1], rgb[2], xyz[1]], i * 4)
  }
  return out
}

// Bradford cone-response matrix (Lam 1985) and its inverse.
const BRADFORD = [
  [0.8951, 0.2664, -0.1614],
  [-0.7502, 1.7135, 0.0367],
  [0.0389, -0.0685, 1.0296],
]
const BRADFORD_INV = [
  [0.9869929, -0.1470543, 0.1599627],
  [0.4323053, 0.5183603, 0.0492912],
  [-0.0085287, 0.0400428, 0.9684867],
]

// The camera's white balance as a 3x3 on linear Rec.709 (rows): light from a blackbody at `kelvin` comes out
// neutral, as a camera set to that white balance renders it.
export function whiteBalance(kelvin: number): number[][] {
  const w = blackbodyXYZ(kelvin)
  const src = apply(BRADFORD, w.map(v => v / w[1]))
  const dst = apply(BRADFORD, D65)
  const scale = [
    [dst[0] / src[0], 0, 0],
    [0, dst[1] / src[1], 0],
    [0, 0, dst[2] / src[2]],
  ]
  const cat = mul(BRADFORD_INV, mul(scale, BRADFORD))
  return mul(XYZ_TO_REC709, mul(cat, REC709_TO_XYZ))
}
