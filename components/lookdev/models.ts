// Hero models for the lab: what each one is, where it sits, and how its file becomes GPU textures. The files are
// built offline by tools/lookdev-models/build.mjs (reduced, BVH built, gzipped) and fetched only when picked.
import { OPENPBR_DEFAULTS, type Hero, type OpenPBR } from './materials'

export type Model = 'spheres' | 'sportscar' | 'teapot' | 'victory' | 'fountain'

export interface ModelInfo {
  label: string
  url: string
  // World size of the model's longest horizontal side (the files are normalized to 1).
  scale: number
  // Has parts in the glass slot (thin-walled), so shadow rays must pass through them.
  thinGlass: boolean
  credit: { text: string; source: string; license: string; licenseUrl: string }
  // Its own materials for slots 4-9 (otherwise MESH_MATERIALS, the car's).
  materials?: OpenPBR[]
  // Height the camera aims at (default MODEL_CAM_TARGET's): tall models are framed higher.
  camTargetY?: number
  // The hero material picked with the model, if it has one it is shown in, and its turntable angle then.
  hero?: Hero
  yaw?: number
  // Real size: meters per scene unit (for lengths like subsurface radii).
  metersPerUnit: number
}

export const MODEL_ORDER: Model[] = ['spheres', 'sportscar', 'teapot', 'victory', 'fountain']

export const MODELS: Record<Exclude<Model, 'spheres'>, ModelInfo> = {
  sportscar: {
    label: 'Sports car',
    url: '/lookdev/models/sportscar.bin.gz?v=4', // bump with each rebuild so caches fetch the new file
    scale: 4.5,
    thinGlass: true,
    metersPerUnit: 1, // a car about 4.5 m long (an estimate; the scene does not state its size)
    credit: {
      text: 'Sports Car by Yasutoshi Mori, from the pbrt-v4 scenes; reduced and rematerialed in OpenPBR for the lab.',
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
    metersPerUnit: 0.075, // a teapot about 27 cm across (an estimate)
    credit: {
      // The model's terms ask that it be identified as the Utah Teapot and its origin at the University of Utah
      // acknowledged.
      text: 'The Utah Teapot, developed at the University of Utah (Martin Newell, 1975), in Cem Yuksel\'s 2026 version.',
      source: 'https://graphics.cs.utah.edu/teapot/',
      license: 'Free for any use',
      licenseUrl: 'https://graphics.cs.utah.edu/teapot/',
    },
  },
  victory: {
    label: 'Winged Victory',
    url: '/lookdev/models/victory.bin.gz?v=2',
    scale: 1.55,
    thinGlass: false,
    camTargetY: 1.0,
    hero: 'skin',
    yaw: 0.9,
    // The statue is 2.75 m tall with its wings (Louvre); the scan stands 2.13 units, its plinth included.
    metersPerUnit: 1.36,
    credit: {
      text: 'Based on "Winged Victory of Samothrace" by CosmoWenman, captured from the Skulpturhalle Basel\'s plaster cast; reduced and given one material for the lab.',
      source: 'https://sketchfab.com/3d-models/winged-victory-of-samothrace-4edd6459f2834e7ab0b395e71cee2513',
      license: 'CC BY 4.0',
      licenseUrl: 'https://creativecommons.org/licenses/by/4.0/',
    },
  },
  fountain: {
    label: 'Table fountain',
    url: '/lookdev/models/fountain.bin.gz?v=1',
    scale: 1.5,
    thinGlass: false,
    camTargetY: 1.0,
    hero: 'gold',
    metersPerUnit: 0.162, // 33.8 cm tall (the museum's record), standing 2.09 units
    credit: {
      text: 'Table Fountain, Paris, c. 1320-40, gilt silver and translucent enamels; Cleveland Museum of Art 1924.859, open access. Reduced for the lab.',
      source: 'https://sketchfab.com/3d-models/1924859-table-fountain-c03c9b6836aa42328803baeef085be40',
      license: 'CC0',
      licenseUrl: 'https://creativecommons.org/publicdomain/zero/1.0/',
    },
    // Slot 4: basse-taille enamel, translucent glass fused over engraved silver: a clear coat tinted blue-green
    // over the metal (OpenPBR's coat color tints what passes through it).
    materials: [
      {
        ...OPENPBR_DEFAULTS,
        base_metalness: 1,
        base_color: [0.94, 0.93, 0.9],
        specular_roughness: 0.2,
        coat_weight: 1,
        coat_color: [0.18, 0.42, 0.62],
        coat_roughness: 0.04,
        coat_ior: 1.55,
      },
    ],
  },
}

// Slots 4-9 for a model: its own, then the car's for any it does not set.
export const modelMaterials = (m: Exclude<Model, 'spheres'>): OpenPBR[] =>
  MESH_MATERIALS.map((d, i) => MODELS[m].materials?.[i] ?? d)

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
  bvh: { w: number; h: number; data: Uint32Array }
  pos: { w: number; h: number; data: Float32Array }
  nrm: { w: number; h: number; data: Uint32Array }
  size: [number, number, number] // normalized bounds: x, y (height), z
}

const rows = (texels: number) => Math.max(1, Math.ceil(texels / TEX_W))

// ---------------------------------------------------------------------------------------------------------------
// Four-wide BVH with compressed child boxes, built at load time from the file's binary one. Each binary node is
// collapsed by opening its largest-area inner child until it has four children (Wald, Benthin and Boulos 2008,
// "Getting rid of packets"); each child's box is stored as 8-bit offsets from the node's corner in power-of-two
// steps, rounded outward (as in Ylitie, Karras and Laine 2017, "Efficient Incoherent Ray Traversal on GPUs
// Through Compressed Wide BVHs"). A node is four RGBA32UI texels, the size of one binary node, for twice the
// children: a ray reads half the bytes per box and visits about half the nodes.
//   0: corner x, y, z (float bits); w: the three biased exponents of the step (x, y, z bytes)
//   1: lo.x, lo.y, lo.z, hi.x of the four children, one byte each (child i in byte i)
//   2: hi.y, hi.z of the four children; zw unused
//   3: the four children's refs: (index << 5) | count, count > 0 a leaf of count triangles from index, count 0
//      the four-wide node at index; EMPTY_REF for an unused slot (whose box bytes are lo 255, hi 0).
// ---------------------------------------------------------------------------------------------------------------
export const BVH4_STACK = 48 // must match BVH_STACK in the shader
const EMPTY_REF = 0xffffffff // must match the shader's test

interface Child { lo: number[]; hi: number[]; ref: number }

function collapseBvh4(nodes: Float32Array, nodeCount: number) {
  const f = Math.fround
  // The binary node's two children.
  const kids = (i: number): Child[] =>
    [0, 8].map(o => ({
      lo: [nodes[i * 16 + o], nodes[i * 16 + o + 1], nodes[i * 16 + o + 2]],
      ref: nodes[i * 16 + o + 3],
      hi: [nodes[i * 16 + o + 4], nodes[i * 16 + o + 5], nodes[i * 16 + o + 6]],
    })).filter(c => c.lo[0] <= c.hi[0])
  const area = (c: Child) => {
    const d = [0, 1, 2].map(k => c.hi[k] - c.lo[k])
    return d[0] * d[1] + d[1] * d[2] + d[2] * d[0]
  }
  const isInner = (c: Child) => (c.ref & 31) === 0
  // Collapse: four-wide node n gets the children list; inner children point at binary nodes until numbered.
  const out: { children: Child[] }[] = []
  let maxDepth = 0
  const build = (binary: number, depth: number): number => {
    maxDepth = Math.max(maxDepth, depth)
    const children = kids(binary)
    while (children.length < 4) {
      let best = -1
      children.forEach((c, i) => {
        if (isInner(c) && (best < 0 || area(c) > area(children[best]))) best = i
      })
      if (best < 0) break
      const open = children.splice(best, 1)[0]
      children.push(...kids(open.ref >> 5))
    }
    const index = out.length
    out.push({ children })
    for (const c of children) if (isInner(c)) c.ref = build(c.ref >> 5, depth + 1) * 32
    return index
  }
  if (nodeCount > 0) build(0, 1)
  if (3 * maxDepth > BVH4_STACK) console.warn(`BVH4 depth ${maxDepth} may overflow the traversal stack`)

  const data = new Uint32Array(Math.max(1, out.length) * 16)
  const fbits = new Float32Array(1)
  const ubits = new Uint32Array(fbits.buffer)
  const bitsOf = (x: number) => ((fbits[0] = x), ubits[0])
  out.forEach((node, n) => {
    const lo = [0, 1, 2].map(k => Math.min(...node.children.map(c => c.lo[k])))
    const hi = [0, 1, 2].map(k => Math.max(...node.children.map(c => c.hi[k])))
    const origin = lo.map(f)
    const exps: number[] = []
    const q = node.children.map(() => ({ lo: [0, 0, 0], hi: [0, 0, 0] }))
    for (let k = 0; k < 3; k++) {
      // The smallest power-of-two step that spans the node in 255 steps, every child box rounded outward and
      // checked in float32 arithmetic as the shader dequantizes it (corner + q * step).
      let e = Math.max(-126, Math.ceil(Math.log2(Math.max(hi[k] - origin[k], 1e-30) / 255)))
      for (;;) {
        const step = 2 ** e
        let fits = true
        node.children.forEach((c, i) => {
          let a = Math.max(0, Math.floor((c.lo[k] - origin[k]) / step))
          while (a > 0 && f(origin[k] + a * step) > c.lo[k]) a--
          let b = Math.ceil((c.hi[k] - origin[k]) / step)
          while (f(origin[k] + b * step) < c.hi[k]) b++
          if (b > 255) fits = false
          q[i].lo[k] = a
          q[i].hi[k] = b
        })
        if (fits) break
        e++
      }
      exps.push(e + 127)
    }
    const o = n * 16
    data[o] = bitsOf(origin[0])
    data[o + 1] = bitsOf(origin[1])
    data[o + 2] = bitsOf(origin[2])
    data[o + 3] = exps[0] | (exps[1] << 8) | (exps[2] << 16)
    // One byte per child slot (an empty slot: lo 255, hi 0; the traversal skips it by its ref).
    const used = node.children.length
    const packed = (get: (i: number) => number, emptyV: number) => {
      let v = 0
      for (let i = 0; i < 4; i++) v |= (i < used ? get(i) : emptyV) << (8 * i)
      return v >>> 0
    }
    data[o + 4] = packed(i => q[i].lo[0], 255)
    data[o + 5] = packed(i => q[i].lo[1], 255)
    data[o + 6] = packed(i => q[i].lo[2], 255)
    data[o + 7] = packed(i => q[i].hi[0], 0)
    data[o + 8] = packed(i => q[i].hi[1], 0)
    data[o + 9] = packed(i => q[i].hi[2], 0)
    for (let i = 0; i < 4; i++) data[o + 12 + i] = i < node.children.length ? node.children[i].ref : EMPTY_REF
  })
  return { data, nodeCount: Math.max(1, out.length), maxDepth }
}

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

  // BVH: the file's binary tree, collapsed to four-wide nodes of four texels each.
  const wide = collapseBvh4(nodes, header.nodeCount)
  const bvhH = rows(wide.nodeCount * 4)
  const bvh = new Uint32Array(TEX_W * bvhH * 4)
  bvh.set(wide.data)

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
