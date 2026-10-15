// The wooden ball's layout as the shaders lay it out (wgsl.ts SIM_COMMON and BALL_HEAT), for the engine's readouts: its
// skin on a cube's six faces at equal angles, and the cells beneath each texel.
import { SKIN_N, WOOD_LAYERS } from './wgsl'

type Vec3 = [number, number, number]

// Each face's two directions across it and its outward axis (faceAxes).
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
