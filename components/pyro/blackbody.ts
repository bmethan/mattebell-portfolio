// Blackbody emission for the pyro lab's flames: Planck's law integrated against the CIE 1931 colour matching
// functions (the multi-lobe Gaussian fit of Wyman, Sloan and Shirley 2013, as the lookdev lab uses) from 380 to 780 nm,
// to linear Rec.709. Not normalised per entry: the table keeps the true relative brightness across temperatures,
// scaled so that 1500 K has unit luminance (a flame at 2500 K is that much brighter, as it is in life).

const C2 = 1.4388e-2 // second radiation constant, m K
export const BB_MIN_K = 800
export const BB_MAX_K = 4000
export const BB_SIZE = 128

const lobe = (w: number, mu: number, a: number, b: number) => {
  const t = (w - mu) * (w < mu ? a : b)
  return Math.exp(-0.5 * t * t)
}
const cieXYZ = (w: number): [number, number, number] => [
  0.362 * lobe(w, 442.0, 0.0624, 0.0374) + 1.056 * lobe(w, 599.8, 0.0264, 0.0323) - 0.065 * lobe(w, 501.1, 0.049, 0.0382),
  0.821 * lobe(w, 568.8, 0.0213, 0.0247) + 0.286 * lobe(w, 530.9, 0.0613, 0.0322),
  1.217 * lobe(w, 437.0, 0.0845, 0.0278) + 0.681 * lobe(w, 459.0, 0.0385, 0.0725),
]
// IEC 61966-2-1 XYZ to linear sRGB (Rec.709 primaries).
const XYZ_TO_REC709 = [
  [3.2406255, -1.537208, -0.4986286],
  [-0.9689307, 1.8757561, 0.0415175],
  [0.0557101, -0.2040211, 1.0569959],
]

// Spectral radiance up to a constant factor: lambda^-5 / (exp(c2 / (lambda T)) - 1), lambda in metres.
const planck = (nm: number, kelvin: number) => {
  const l = nm * 1e-9
  return 1 / (l ** 5 * Math.expm1(C2 / (l * kelvin)))
}

function blackbodyXYZ(kelvin: number): [number, number, number] {
  const s: [number, number, number] = [0, 0, 0]
  for (let nm = 380; nm <= 780; nm += 2) {
    const b = planck(nm, kelvin)
    const c = cieXYZ(nm)
    s[0] += b * c[0]
    s[1] += b * c[1]
    s[2] += b * c[2]
  }
  return s
}

// BB_SIZE entries from BB_MIN_K to BB_MAX_K, as rgba (a = luminance), linear Rec.709, unit luminance at 1500 K.
// Below about 1900 K the colour lies outside Rec.709; negative components clip to 0, as in the lookdev lab.
export function blackbodyTable(): Float32Array<ArrayBuffer> {
  const ref = blackbodyXYZ(1500)[1]
  const out = new Float32Array(BB_SIZE * 4)
  for (let i = 0; i < BB_SIZE; i++) {
    const k = BB_MIN_K + ((BB_MAX_K - BB_MIN_K) * i) / (BB_SIZE - 1)
    const xyz = blackbodyXYZ(k).map(v => v / ref)
    const rgb = XYZ_TO_REC709.map(r => Math.max(0, r[0] * xyz[0] + r[1] * xyz[1] + r[2] * xyz[2]))
    out.set([rgb[0], rgb[1], rgb[2], xyz[1]], i * 4)
  }
  return out
}
