// The Lookdev lab's HDR environments, prepared to light the Pyro lab's fire and smoke. A volume renderer cannot sample
// a map per scattering event at this lab's frame rate, so each map becomes:
//   - a key, where the map has one: its brightest light pulled out as a directional light, so it casts the smoke's
//     shadows through the renderer's light volume (the midday sun, 68% of that map's light within a degree; the photo
//     studio's main softbox, 43% within 15 degrees). The overcast sky and the sunset's glow have none (about 1%).
//   - the rest as spherical harmonics (three bands, as Ramamoorthi and Hanrahan light diffuse surfaces with), which
//     the renderer convolves with the smoke's phase function for its in-scattering and with the cosine for surfaces.
//     Below the horizon the map gives way to the floor, lit by the whole environment.
//   - a small copy of the map for the backdrop, softened as the Lookdev lab shows it.
// All in the map's own units (ACEScg); the renderer scales them so the environment's illuminance on open ground,
// key included, meets the lab's EV, as an incident meter facing up reads it.
//
// Directions follow the Lookdev lab's mapping (envUV in its tracer): u = 0.5 + atan2(x, -z) / 2pi, v = acos(y) / pi.

import { ENVIRONMENTS, loadEnvironment, type Env } from '../lookdev/environments'
import { AP1_LUMINANCE } from '../lab/viewTransforms'

export type PyroEnv = Exclude<Env, 'none'>
type Vec3 = [number, number, number]

export interface EnvLight {
  // The key, in the map's frame: toward it, and its illuminance at normal incidence by colour; or null.
  key: { dir: Vec3; irradiance: Vec3 } | null
  // Where the map's light centres (the key, or the brightest region of a map without one): the lab's light azimuth
  // turns the map so this faces it.
  anchor: Vec3
  sh: Float32Array // 9 coefficients x rgb (the rest of the map, the floor below the horizon)
  illuminance: number // on open ground (lux in the map's units), key included
  backdrop: { w: number; h: number; rgba: Float32Array }
}

export const BACKDROP_W = 256
export const BACKDROP_H = 128
const FLOOR_ALBEDO = 0.14 // the lab's floor (the renderer's floorCol)
// A key is pulled out where the light above the sky around it within this cone gives at least this share of the
// map's illuminance on open ground: a sun first, then a softbox.
const KEY_CONES: [number, number][] = [
  [3, 0.5],
  [15, 0.3],
]

// The real spherical harmonics to the second band, y up (the renderer's shBasis is the same).
export function shBasis(x: number, y: number, z: number, out: Float64Array) {
  out[0] = 0.282095
  out[1] = 0.488603 * y
  out[2] = 0.488603 * z
  out[3] = 0.488603 * x
  out[4] = 1.092548 * x * y
  out[5] = 1.092548 * y * z
  out[6] = 0.315392 * (3 * y * y - 1)
  out[7] = 1.092548 * x * z
  out[8] = 0.546274 * (x * x - z * z)
}

const lum = (r: number, g: number, b: number) => AP1_LUMINANCE[0] * r + AP1_LUMINANCE[1] * g + AP1_LUMINANCE[2] * b

function median(v: number[]) {
  if (!v.length) return 0
  const s = v.slice().sort((a, b) => a - b)
  return s[s.length >> 1]
}

const cache = new Map<PyroEnv, Promise<EnvLight>>()
export function environmentLight(env: PyroEnv): Promise<EnvLight> {
  let p = cache.get(env)
  if (!p) {
    p = prepare(env)
    p.catch(() => cache.delete(env))
    cache.set(env, p)
  }
  return p
}

async function prepare(env: PyroEnv): Promise<EnvLight> {
  const { w, h, rgba } = await loadEnvironment(env)
  const n = w * h
  const dirs = new Float32Array(n * 3)
  const omega = new Float32Array(n)
  const L = new Float32Array(n)
  for (let j = 0; j < h; j++) {
    const th = ((j + 0.5) / h) * Math.PI
    const st = Math.sin(th)
    for (let i = 0; i < w; i++) {
      const ph = 2 * Math.PI * ((i + 0.5) / w - 0.5)
      const k = j * w + i
      dirs[3 * k] = st * Math.sin(ph)
      dirs[3 * k + 1] = Math.cos(th)
      dirs[3 * k + 2] = -st * Math.cos(ph)
      omega[k] = ((2 * Math.PI) / w) * (Math.PI / h) * st
      L[k] = lum(rgba[4 * k], rgba[4 * k + 1], rgba[4 * k + 2])
    }
  }
  // Illuminance on open ground: the upper hemisphere, cosine weighted.
  let eh = 0
  let peak = -1
  for (let k = 0; k < n; k++) {
    const y = dirs[3 * k + 1]
    if (y <= 0) continue
    eh += L[k] * y * omega[k]
    if (peak < 0 || L[k] > L[peak]) peak = k
  }
  const pd: Vec3 = [dirs[3 * peak], dirs[3 * peak + 1], dirs[3 * peak + 2]]

  // The key: the light above the median of the ring around the cone (5 degrees wide), within the cone.
  const residual = rgba.slice()
  let key: EnvLight['key'] = null
  for (const [cone, share] of KEY_CONES) {
    const cIn = Math.cos((cone * Math.PI) / 180)
    const cOut = Math.cos(((cone + 5) * Math.PI) / 180)
    const ring: number[][] = [[], [], []]
    const inside: number[] = []
    for (let k = 0; k < n; k++) {
      const c = dirs[3 * k] * pd[0] + dirs[3 * k + 1] * pd[1] + dirs[3 * k + 2] * pd[2]
      if (c > cIn) inside.push(k)
      else if (c > cOut) for (let ch = 0; ch < 3; ch++) ring[ch].push(rgba[4 * k + ch])
    }
    const base = ring.map(median)
    const e: Vec3 = [0, 0, 0]
    const d: Vec3 = [0, 0, 0]
    for (const k of inside) {
      for (let ch = 0; ch < 3; ch++) e[ch] += Math.max(rgba[4 * k + ch] - base[ch], 0) * omega[k]
      const x = Math.max(L[k] - lum(base[0], base[1], base[2]), 0) * omega[k]
      d[0] += x * dirs[3 * k]
      d[1] += x * dirs[3 * k + 1]
      d[2] += x * dirs[3 * k + 2]
    }
    const len = Math.hypot(d[0], d[1], d[2])
    if (len <= 0) continue
    const dir: Vec3 = [d[0] / len, d[1] / len, d[2] / len]
    if (lum(...e) * Math.max(dir[1], 0) < share * eh) continue
    key = { dir, irradiance: e }
    for (const k of inside) for (let ch = 0; ch < 3; ch++) residual[4 * k + ch] = Math.min(rgba[4 * k + ch], base[ch])
    break
  }

  // Spherical harmonics of the rest, the floor below the horizon (lit by all of it, key included, unshadowed).
  const floorL = (FLOOR_ALBEDO * eh) / Math.PI
  // The floor's colour: the environment's light on open ground, at that luminance.
  const eColour: Vec3 = [0, 0, 0]
  for (let k = 0; k < n; k++) {
    const y = dirs[3 * k + 1]
    if (y <= 0) continue
    for (let ch = 0; ch < 3; ch++) eColour[ch] += rgba[4 * k + ch] * y * omega[k]
  }
  const ec = lum(...eColour) || 1
  const floor = eColour.map(v => (v / ec) * floorL)
  const sh = new Float64Array(27)
  const b = new Float64Array(9)
  for (let k = 0; k < n; k++) {
    const x = dirs[3 * k], y = dirs[3 * k + 1], z = dirs[3 * k + 2]
    shBasis(x, y, z, b)
    for (let ch = 0; ch < 3; ch++) {
      const v = (y > 0 ? residual[4 * k + ch] : floor[ch]) * omega[k]
      for (let m = 0; m < 9; m++) sh[3 * m + ch] += v * b[m]
    }
  }

  // The backdrop: the map in blocks, as a softened copy (the sun's disc spread over its block).
  const fx = w / BACKDROP_W, fy = h / BACKDROP_H
  const bd = new Float32Array(BACKDROP_W * BACKDROP_H * 4)
  for (let j = 0; j < BACKDROP_H; j++) {
    for (let i = 0; i < BACKDROP_W; i++) {
      const s = [0, 0, 0]
      for (let y = 0; y < fy; y++) for (let x = 0; x < fx; x++) {
        const k = (j * fy + y) * w + i * fx + x
        for (let ch = 0; ch < 3; ch++) s[ch] += rgba[4 * k + ch]
      }
      const o = 4 * (j * BACKDROP_W + i)
      bd.set([s[0] / (fx * fy), s[1] / (fx * fy), s[2] / (fx * fy), 1], o)
    }
  }
  return { key, anchor: key ? key.dir : pd, sh: Float32Array.from(sh), illuminance: eh, backdrop: { w: BACKDROP_W, h: BACKDROP_H, rgba: bd } }
}

export { ENVIRONMENTS }
