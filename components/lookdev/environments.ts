// Image-based lighting: equirectangular HDR environments from Poly Haven (CC0), fetched only when picked.
//
// Each file is Radiance RGBE (run-length encoded scanlines, top row first), linear Rec.709. On load it is
// converted to ACEScg and scaled so the brightest irradiance it gives any surface is pi, the key light's (the 18%
// gray ball's lit side then reads about middle gray under any of them, as under the softbox rig). For importance sampling,
// every pixel gets a weight of its luminance times sin(theta) (its share of the sphere) in a Walker alias table
// (Vose's construction), so the tracer draws a pixel with one texture fetch.

import { REC709_TO_AP1 } from './color'

export type Env = 'none' | 'studio' | 'midday' | 'overcast' | 'sunset'
export const ENV_ORDER: Env[] = ['none', 'studio', 'midday', 'overcast', 'sunset']

// Measured from the files as shipped, over the sky above the horizon (the floor hides the rest): gain, the scale
// that brings the brightest irradiance any surface receives (over 1,152 orientations) to pi, the key light's; turn,
// the turn (of a full circle) that brings that brightest direction to the key's default azimuth, front right, so
// each opens lit as the softbox rig is. The lab's turn control adds to it.
export const ENVIRONMENTS: Record<Exclude<Env, 'none'>, { label: string; file: string; gain: number; turn: number; title: string; authors: string; url: string }> = {
  studio: {
    label: 'Studio',
    file: 'brown_photostudio_02_1k.hdr',
    gain: 0.5537,
    turn: 0.725,
    title: 'Brown Photostudio 02',
    authors: 'Sergej Majboroda',
    url: 'https://polyhaven.com/a/brown_photostudio_02',
  },
  midday: {
    label: 'Midday sun',
    file: 'kloofendal_48d_partly_cloudy_puresky_1k.hdr',
    gain: 0.51,
    turn: 0.6972,
    title: 'Kloofendal 48d Partly Cloudy (Pure Sky)',
    authors: 'Greg Zaal, Jarod Guest',
    url: 'https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky',
  },
  overcast: {
    label: 'Overcast',
    file: 'overcast_soil_puresky_1k.hdr',
    gain: 0.6713,
    turn: 0.5444,
    title: 'Overcast Soil (Pure Sky)',
    authors: 'Jarod Guest, Sergej Majboroda',
    url: 'https://polyhaven.com/a/overcast_soil_puresky',
  },
  sunset: {
    label: 'Sunset',
    file: 'belfast_sunset_puresky_1k.hdr',
    gain: 0.6145,
    turn: 0.7667,
    title: 'Belfast Sunset (Pure Sky)',
    authors: 'Greg Zaal, Dimitrios Savva, Jarod Guest',
    url: 'https://polyhaven.com/a/belfast_sunset_puresky',
  },
}

export interface EnvData {
  w: number
  h: number
  rgba: Float32Array // ACEScg radiance, scaled by gain; alpha 1
  alias: Float32Array // per pixel: probability of keeping it, the alias pixel, its pdf constant, 0
}

// Radiance RGBE with new-style run-length encoding (every Poly Haven file is).
function decodeRGBE(buf: ArrayBuffer) {
  const b = new Uint8Array(buf)
  let i = 0
  const line = () => {
    let s = ''
    while (b[i] !== 0x0a) s += String.fromCharCode(b[i++])
    i++
    return s
  }
  if (!line().startsWith('#?')) throw new Error('not a Radiance HDR file')
  while (line() !== '');
  const res = line().split(' ')
  if (res[0] !== '-Y' || res[2] !== '+X') throw new Error('unsupported HDR orientation')
  const h = parseInt(res[1]), w = parseInt(res[3])
  const out = new Float32Array(w * h * 4)
  const scan = new Uint8Array(w * 4)
  for (let y = 0; y < h; y++) {
    if (b[i] !== 2 || b[i + 1] !== 2 || ((b[i + 2] << 8) | b[i + 3]) !== w) throw new Error('unsupported HDR encoding')
    i += 4
    for (let c = 0; c < 4; c++) {
      for (let x = 0; x < w; ) {
        let n = b[i++]
        if (n > 128) {
          n -= 128
          const v = b[i++]
          for (let k = 0; k < n; k++) scan[(x + k) * 4 + c] = v
        } else {
          for (let k = 0; k < n; k++) scan[(x + k) * 4 + c] = b[i++]
        }
        x += n
      }
    }
    for (let x = 0; x < w; x++) {
      const e = scan[x * 4 + 3]
      const f = e ? 2 ** (e - 136) : 0
      const o = (y * w + x) * 4
      out[o] = scan[x * 4] * f
      out[o + 1] = scan[x * 4 + 1] * f
      out[o + 2] = scan[x * 4 + 2] * f
      out[o + 3] = 1
    }
  }
  return { w, h, rgb: out }
}

const AP1_Y = [0.2722287, 0.6740818, 0.0536895] // ACEScg luminance weights

export async function loadEnvironment(env: Exclude<Env, 'none'>): Promise<EnvData> {
  const info = ENVIRONMENTS[env]
  const res = await fetch(`/lookdev/env/${info.file}`)
  if (!res.ok) throw new Error(`environment ${env}: HTTP ${res.status}`)
  const { w, h, rgb } = decodeRGBE(await res.arrayBuffer())
  const M = REC709_TO_AP1
  for (let o = 0; o < rgb.length; o += 4) {
    const r = rgb[o] * info.gain, g = rgb[o + 1] * info.gain, bl = rgb[o + 2] * info.gain
    rgb[o] = Math.max(0, M[0][0] * r + M[0][1] * g + M[0][2] * bl)
    rgb[o + 1] = Math.max(0, M[1][0] * r + M[1][1] * g + M[1][2] * bl)
    rgb[o + 2] = Math.max(0, M[2][0] * r + M[2][1] * g + M[2][2] * bl)
  }
  return withAliasTable(w, h, rgb)
}

// The white furnace's surround as an environment (radiance 1 everywhere), so the test also checks the
// environment's sampling and its weighting against the materials' own.
export function whiteEnvironment(): EnvData {
  const w = 64, h = 32
  return withAliasTable(w, h, new Float32Array(w * h * 4).fill(1))
}

function withAliasTable(w: number, h: number, rgb: Float32Array): EnvData {
  const n = w * h
  const weight = new Float64Array(n)
  let total = 0
  for (let y = 0; y < h; y++) {
    const sinT = Math.sin(((y + 0.5) / h) * Math.PI)
    for (let x = 0; x < w; x++) {
      const p = y * w + x
      const o = p * 4
      const wt = Math.max(0, AP1_Y[0] * rgb[o] + AP1_Y[1] * rgb[o + 1] + AP1_Y[2] * rgb[o + 2]) * sinT
      weight[p] = wt
      total += wt
    }
  }
  // Walker alias table, Vose's construction (numerically stable: small and large worklists).
  const alias = new Float32Array(n * 4)
  const scaled = new Float64Array(n)
  const small: number[] = []
  const large: number[] = []
  for (let p = 0; p < n; p++) {
    scaled[p] = total > 0 ? (weight[p] / total) * n : 1
    ;(scaled[p] < 1 ? small : large).push(p)
    // Solid-angle pdf of a direction in pixel p is this constant over sin(theta) at the direction: the pixel's
    // probability spread over its (2 pi / w) x (pi / h) of longitude and colatitude.
    alias[p * 4 + 2] = total > 0 ? ((weight[p] / total) * n) / (2 * Math.PI * Math.PI) : 1 / (2 * Math.PI * Math.PI)
  }
  while (small.length && large.length) {
    const s = small.pop()!
    const l = large.pop()!
    alias[s * 4] = scaled[s]
    alias[s * 4 + 1] = l
    scaled[l] = scaled[l] + scaled[s] - 1
    ;(scaled[l] < 1 ? small : large).push(l)
  }
  for (const p of large) { alias[p * 4] = 1; alias[p * 4 + 1] = p }
  for (const p of small) { alias[p * 4] = 1; alias[p * 4 + 1] = p }
  return { w, h, rgba: rgb, alias }
}
