// The wooden ball's layout as the shaders lay it out (wgsl.ts SIM_COMMON and BALL_HEAT), for the engine's readouts: its
// skin on a cube's six faces at equal angles, and the cells beneath each texel.
import { SKIN_N, WOOD_LAYERS } from './wgsl'

type Vec3 = [number, number, number]

// Each face's two directions across it and its outward axis (cubeAt, cubeDir).
const FACES: [Vec3, Vec3, Vec3][] = [
  [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
  [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
  [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
  [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
]
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

// The skin texel the direction d (from the ball's centre) passes through (texelOf).
export function skinTexel(d: Vec3): number {
  const m = d.map(Math.abs)
  let f = d[2] > 0 ? 4 : 5
  if (m[0] >= m[1] && m[0] >= m[2]) f = d[0] > 0 ? 0 : 1
  else if (m[1] >= m[2]) f = d[1] > 0 ? 2 : 3
  const [u, v, n] = FACES[f]
  const h = dot(d, n)
  const at = (a: Vec3) => Math.min(Math.floor(((Math.atan(dot(d, a) / h) * 4) / Math.PI + 1) * 0.5 * SKIN_N), SKIN_N - 1)
  return (f * SKIN_N + at(v)) * SKIN_N + at(u)
}

// The solid angle of each texel of a face (the same on every face), in steradians: (pi / 2N)^2 (1 + u^2) (1 + v^2) /
// (1 + u^2 + v^2)^1.5 at its centre, u and v where its direction crosses the face's plane.
let solid: Float32Array | null = null
export function skinSolidAngles(): Float32Array {
  if (solid) return solid
  solid = new Float32Array(SKIN_N * SKIN_N)
  const a = Math.PI / (2 * SKIN_N)
  for (let j = 0; j < SKIN_N; j++) {
    const v = Math.tan((((j + 0.5) * 2) / SKIN_N - 1) * (Math.PI / 4))
    for (let i = 0; i < SKIN_N; i++) {
      const u = Math.tan((((i + 0.5) * 2) / SKIN_N - 1) * (Math.PI / 4))
      const q = 1 + u * u + v * v
      solid[j * SKIN_N + i] = (a * a * (1 + u * u) * (1 + v * v)) / (q * Math.sqrt(q))
    }
  }
  return solid
}

// The cells beneath a texel, by depth below the surface as made (m): nine from 0.5 mm, each 1.3 times the last, then
// thirty of equal depth to three quarters of the radius in, then the core (layerTop, layerThick).
export function woodLayers(r: number): { top: number; thick: number }[] {
  const graded = (0.0005 * (1.3 ** 9 - 1)) / 0.3
  const deep = (0.75 * r - graded) / 30
  const out: { top: number; thick: number }[] = []
  for (let k = 0; k < WOOD_LAYERS; k++) {
    if (k < 9) out.push({ top: (0.0005 * (1.3 ** k - 1)) / 0.3, thick: 0.0005 * 1.3 ** k })
    else if (k < 39) out.push({ top: graded + (k - 9) * deep, thick: deep })
    else out.push({ top: 0.75 * r, thick: 0.25 * r })
  }
  return out
}

// The direction (from the ball's centre) through the centre of each texel, three numbers to a texel (cubeDir).
let dirs: Float32Array | null = null
export function skinDirs(): Float32Array {
  if (dirs) return dirs
  dirs = new Float32Array(6 * SKIN_N * SKIN_N * 3)
  const at = (i: number) => Math.tan((((i + 0.5) * 2) / SKIN_N - 1) * (Math.PI / 4))
  FACES.forEach(([u, v, n], f) => {
    for (let j = 0; j < SKIN_N; j++) {
      for (let i = 0; i < SKIN_N; i++) {
        const d = [0, 1, 2].map(k => n[k] + at(i) * u[k] + at(j) * v[k])
        const l = Math.hypot(d[0], d[1], d[2])
        dirs!.set([d[0] / l, d[1] / l, d[2] / l], ((f * SKIN_N + j) * SKIN_N + i) * 3)
      }
    }
  })
  return dirs
}

// The wear over the ball's surface as the gas sees it (wornRadius in wgsl.ts): the smooth shape that best fits the wear
// of every texel (m), weighted by its solid angle, of the form a x + b y + c z + d x^2 + e y^2 + f z^2 + g xy + h yz +
// i zx in the direction (x, y, z). These are the spherical harmonics to the second order (the mean wear is carried by
// the squares, which sum to one): the wear's offset to one side and its stretch into an egg, as much of its shape as
// the gas's cells (3 cm across at medium) can hold. Returned as wornRadius takes it, three vec4s: (a, b, c), (d, e, f)
// and (g, h, i).
const basis = (x: number, y: number, z: number) => [x, y, z, x * x, y * y, z * z, x * y, y * z, z * x]
let wearInverse: Float64Array | null = null
export function fitWear(wear: Float32Array): Float32Array {
  const d = skinDirs()
  const solidAngle = skinSolidAngles()
  const face = SKIN_N * SKIN_N
  if (!wearInverse) {
    // (The normal equations' matrix depends only on the texels: summed, and inverted by Gauss-Jordan, once.)
    const m = new Float64Array(81)
    for (let t = 0; t < 6 * face; t++) {
      const b = basis(d[3 * t], d[3 * t + 1], d[3 * t + 2])
      const w = solidAngle[t % face]
      for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) m[9 * i + j] += w * b[i] * b[j]
    }
    const inv = new Float64Array(81)
    for (let i = 0; i < 9; i++) inv[10 * i] = 1
    for (let c = 0; c < 9; c++) {
      let p = c
      for (let r = c + 1; r < 9; r++) if (Math.abs(m[9 * r + c]) > Math.abs(m[9 * p + c])) p = r
      for (let k = 0; k < 9; k++) {
        ;[m[9 * c + k], m[9 * p + k]] = [m[9 * p + k], m[9 * c + k]]
        ;[inv[9 * c + k], inv[9 * p + k]] = [inv[9 * p + k], inv[9 * c + k]]
      }
      const s = 1 / m[10 * c]
      for (let k = 0; k < 9; k++) {
        m[9 * c + k] *= s
        inv[9 * c + k] *= s
      }
      for (let r = 0; r < 9; r++) {
        if (r === c) continue
        const q = m[9 * r + c]
        for (let k = 0; k < 9; k++) {
          m[9 * r + k] -= q * m[9 * c + k]
          inv[9 * r + k] -= q * inv[9 * c + k]
        }
      }
    }
    wearInverse = inv
  }
  const rhs = new Float64Array(9)
  for (let t = 0; t < 6 * face; t++) {
    if (!(wear[t] > 0)) continue
    const b = basis(d[3 * t], d[3 * t + 1], d[3 * t + 2])
    const w = solidAngle[t % face] * wear[t]
    for (let i = 0; i < 9; i++) rhs[i] += w * b[i]
  }
  const out = new Float32Array(12)
  for (let i = 0; i < 9; i++) {
    let c = 0
    for (let j = 0; j < 9; j++) c += wearInverse[9 * i + j] * rhs[j]
    out[4 * Math.floor(i / 3) + (i % 3)] = c
  }
  return out
}
