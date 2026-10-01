// Hero models for the lab: what each one is, where it sits, and how its file becomes GPU textures. The files are
// built offline by tools/lookdev-models/build.mjs (reduced, BVH built, gzipped) and fetched only when picked.
import { OPENPBR_DEFAULTS, type OpenPBR } from './materials'

export type Model = 'spheres' | 'sportscar' | 'teapot'

export interface ModelInfo {
  label: string
  url: string
  // World size of the model's longest horizontal side (the files are normalized to 1).
  scale: number
  // Has parts in the glass slot (thin-walled), so shadow rays must pass through them.
  thinGlass: boolean
  credit: { text: string; source: string; license: string; licenseUrl: string }
}

export const MODEL_ORDER: Model[] = ['spheres', 'sportscar', 'teapot']

export const MODELS: Record<Exclude<Model, 'spheres'>, ModelInfo> = {
  sportscar: {
    label: 'Sports car',
    url: '/lookdev/models/sportscar.bin.gz?v=3', // bump with each rebuild so caches fetch the new file
    scale: 4.5,
    thinGlass: true,
    credit: {
      text: 'Sports Car by Yasutoshi Mori, from the pbrt-v4 scenes. Reduced to 184k triangles and rematerialed in OpenPBR for the lab.',
      source: 'https://github.com/mmp/pbrt-v4-scenes',
      license: 'CC BY 4.0',
      licenseUrl: 'https://creativecommons.org/licenses/by/4.0/',
    },
  },
  teapot: {
    label: 'Utah teapot',
    url: '/lookdev/models/teapot.bin.gz?v=2',
    scale: 3.6,
    thinGlass: false,
    credit: {
      // The model's terms ask that it be identified as the Utah Teapot and its origin at the University of Utah
      // acknowledged.
      text: 'The Utah Teapot, developed at the University of Utah (Martin Newell, 1975); the 2026 version by Cem Yuksel, with curvature continuity and chamfers. 80k triangles.',
      source: 'https://graphics.cs.utah.edu/teapot/',
      license: 'Free for any use',
      licenseUrl: 'https://graphics.cs.utah.edu/teapot/',
    },
  },
}

// With a model in the middle, the gray and chrome references shrink to the sides, as on a turntable plate.
// Each ball: center x, y, z and radius; the hero ball is absent (its material dresses the model).
export const MODEL_BALLS: [number, number, number, number][] = [
  [-2.95, 0.36, -0.2, 0.36],
  [2.95, 0.36, -0.2, 0.36],
  [0, 0, 0, 0],
]

const mat = (p: Partial<OpenPBR>): OpenPBR => ({ ...OPENPBR_DEFAULTS, ...p })

// Slots 4-9 (slots 1 and 2 are the scene's chrome and hero). All authored for the lab, in ACEScg.
export const MESH_MATERIALS: OpenPBR[] = [
  // 4 Glass: thin-walled (windows are sheets; light passes straight through), with the faint green of
  // automotive glass as a tint at each pass.
  mat({
    specular_roughness: 0,
    specular_ior: 1.5,
    transmission_weight: 1,
    transmission_color: [0.86, 0.92, 0.89],
    geometry_thin_walled: 1,
  }),
  // 5 Rubber: tires and seals.
  mat({ base_color: [0.025, 0.025, 0.026], base_diffuse_roughness: 0.6, specular_weight: 0.6, specular_roughness: 0.6 }),
  // 6 Gloss black trim.
  mat({ base_color: [0.012, 0.012, 0.012], specular_roughness: 0.12 }),
  // 7 Machined aluminum: wheels, brakes, suspension.
  mat({ base_color: [0.912, 0.914, 0.92], base_metalness: 1, specular_roughness: 0.22 }),
  // 8 Red lamp lens.
  mat({ base_color: [0.5, 0.012, 0.008], specular_roughness: 0.04, coat_weight: 1, coat_roughness: 0.01 }),
  // 9 Light gray plastic: lamp housings.
  mat({ base_color: [0.32, 0.32, 0.32], specular_roughness: 0.35 }),
]

export const TEX_W = 2048 // matches TEX_W_SHIFT in the shader

export interface MeshTextures {
  triCount: number
  bvh: { w: number; h: number; data: Float32Array }
  pos: { w: number; h: number; data: Float32Array }
  nrm: { w: number; h: number; data: Uint32Array }
  size: [number, number, number] // normalized bounds: x, y (height), z
}

const rows = (texels: number) => Math.max(1, Math.ceil(texels / TEX_W))

async function gunzip(bytes: Uint8Array) {
  // A host may already have decoded the gzip (Content-Encoding); only decompress what is still gzip.
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

export async function loadModel(model: Exclude<Model, 'spheres'>, signal?: AbortSignal): Promise<MeshTextures> {
  const res = await fetch(MODELS[model].url, { signal })
  if (!res.ok) throw new Error(`Model ${model}: HTTP ${res.status}`)
  const bytes = await gunzip(new Uint8Array(await res.arrayBuffer()))
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== 'LDM1') throw new Error(`Model ${model}: not an LDM1 file`)
  const headerLen = view.getUint32(4, true)
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLen)))
  if (header.version !== 2) throw new Error(`Model ${model}: file version ${header.version}, expected 2`)
  const base = bytes.byteOffset + 8 + headerLen
  // Sections are 4-byte aligned in the file (the header is padded), so typed views can sit on it directly.
  const section = <T>(name: string, Type: { new (b: ArrayBuffer, offset: number, length: number): T }) =>
    new Type(bytes.buffer as ArrayBuffer, base + header.layout[name].offset, header.layout[name].length)
  const nodes = section('nodes', Float32Array)
  const pos = section('pos', Float32Array)
  const nrm = section('nrm', Int16Array)
  const idx = section('idx', Uint32Array)
  const mat = section('mat', Uint8Array)
  const triCount: number = header.triCount

  // BVH: the file's 16 floats per inner node are exactly four texels.
  const bvhH = rows(header.nodeCount * 4)
  const bvh = new Float32Array(TEX_W * bvhH * 4)
  bvh.set(nodes)

  // Triangles: v0, e1, e2 (three texels), and one texel of packed normals plus the material slot.
  const posH = rows(triCount * 3)
  const posData = new Float32Array(TEX_W * posH * 4)
  const nrmH = rows(triCount)
  const nrmData = new Uint32Array(TEX_W * nrmH * 4)
  const oct = (v: number) => ((nrm[v * 2] & 0xffff) | ((nrm[v * 2 + 1] & 0xffff) << 16)) >>> 0
  for (let t = 0; t < triCount; t++) {
    const a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2]
    const o = t * 12
    for (let k = 0; k < 3; k++) {
      const v0 = pos[a * 3 + k]
      posData[o + k] = v0
      posData[o + 4 + k] = pos[b * 3 + k] - v0
      posData[o + 8 + k] = pos[c * 3 + k] - v0
    }
    nrmData.set([oct(a), oct(b), oct(c), mat[t]], t * 4)
  }
  return {
    triCount,
    bvh: { w: TEX_W, h: bvhH, data: bvh },
    pos: { w: TEX_W, h: posH, data: posData },
    nrm: { w: TEX_W, h: nrmH, data: nrmData },
    size: header.size,
  }
}
