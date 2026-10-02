// Offline builder for the Lookdev lab's hero models. Reads a source model, maps its parts onto the lab's material
// slots, reduces it with meshoptimizer, builds a SAH bounding volume hierarchy, and writes one gzipped binary that
// the lab loads only when a visitor picks that model.
//
//   node tools/lookdev-models/build.mjs sportscar <path to pbrt-v4-scenes/sportscar>
//
// meshoptimizer (MIT) is resolved from node_modules, or from MESHOPT_DIR when set to a package directory.
import fs from 'node:fs'
import path from 'node:path'
import zlib from 'node:zlib'
import { pathToFileURL } from 'node:url'

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '../..')
const OUT_DIR = path.join(ROOT, 'public/lookdev/models')

const { MeshoptSimplifier } = await import(
  process.env.MESHOPT_DIR ? pathToFileURL(path.join(process.env.MESHOPT_DIR, 'index.js')).href : 'meshoptimizer'
)
await MeshoptSimplifier.ready

// Material slots shared with the renderer (components/lookdev/models.ts): 0-3 are the scene's own (gray ball,
// chrome, hero, floor); the mesh adds 4-9.
// A model may give 4-9 its own materials (MODELS[...].materials there); the names below are the car's, plus the
// fountain's enamel.
export const SLOT = { chrome: 1, hero: 2, glass: 4, rubber: 5, trim: 6, aluminum: 7, lamp: 8, plastic: 9, enamel: 4 }

// ---------------------------------------------------------------------------------------------------------------
// PLY (binary little endian, as exported for the pbrt-v4 scenes).
// ---------------------------------------------------------------------------------------------------------------
function readPly(file) {
  const buf = fs.readFileSync(file)
  const headerEnd = buf.indexOf('end_header\n') + 'end_header\n'.length
  const header = buf.subarray(0, headerEnd).toString('ascii').split('\n')
  if (!header[1].startsWith('format binary_little_endian')) throw new Error(`${file}: not binary little endian`)
  const elements = []
  for (const line of header) {
    const w = line.trim().split(/\s+/)
    if (w[0] === 'element') elements.push({ name: w[1], count: +w[2], props: [] })
    else if (w[0] === 'property') elements.at(-1).props.push(w[1] === 'list' ? { list: true, countType: w[2], type: w[3], name: w[4] } : { type: w[1], name: w[2] })
  }
  const size = { char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8 }
  const read = (t, o) => {
    switch (t) {
      case 'char': case 'int8': return buf.readInt8(o)
      case 'uchar': case 'uint8': return buf.readUInt8(o)
      case 'short': return buf.readInt16LE(o)
      case 'ushort': return buf.readUInt16LE(o)
      case 'int': case 'int32': return buf.readInt32LE(o)
      case 'uint': case 'uint32': return buf.readUInt32LE(o)
      case 'float': case 'float32': return buf.readFloatLE(o)
      case 'double': return buf.readDoubleLE(o)
    }
    throw new Error(`PLY type ${t}`)
  }
  let o = headerEnd
  let pos = null, nrm = null
  const tris = []
  for (const el of elements) {
    if (el.name === 'vertex') {
      pos = new Float32Array(el.count * 3)
      const names = el.props.map(p => p.name)
      const hasN = names.includes('nx')
      if (hasN) nrm = new Float32Array(el.count * 3)
      for (let i = 0; i < el.count; i++) {
        for (const p of el.props) {
          const v = read(p.type, o)
          o += size[p.type]
          const k = { x: 0, y: 1, z: 2 }[p.name]
          if (k !== undefined) pos[i * 3 + k] = v
          const kn = { nx: 0, ny: 1, nz: 2 }[p.name]
          if (kn !== undefined) nrm[i * 3 + kn] = v
        }
      }
    } else if (el.name === 'face') {
      for (let i = 0; i < el.count; i++) {
        for (const p of el.props) {
          if (!p.list) { o += size[p.type]; continue }
          const n = read(p.countType, o)
          o += size[p.countType]
          const ids = []
          for (let j = 0; j < n; j++) { ids.push(read(p.type, o)); o += size[p.type] }
          if (p.name !== 'vertex_indices' && p.name !== 'vertex_index') continue
          for (let j = 1; j + 1 < n; j++) tris.push(ids[0], ids[j], ids[j + 1]) // fan
        }
      }
    } else {
      throw new Error(`${file}: unexpected element ${el.name}`)
    }
  }
  return { pos, nrm, idx: new Uint32Array(tris) }
}

// ---------------------------------------------------------------------------------------------------------------
// Sports Car (Yasutoshi Mori, CC BY 4.0), from github.com/mmp/pbrt-v4-scenes/sportscar.
// ---------------------------------------------------------------------------------------------------------------
// Each pbrt material onto a lab slot, or null to leave the part out: the small lamp covers are dropped so the
// reflectors behind them read. The windows are thin-walled glass, so the cabin shows through them.
const CAR_MATERIALS = {
  BodyMat_phong_SG: SLOT.hero,
  WindowGlassMat_phong_SG: SLOT.glass,
  GlassMat_phong_SG: null, // headlight glass (one copy is authored in pbrt's z-up world frame, not the car's)
  HeadlightLens_phong_SG: null,
  StopLightCover_Mat_phong_SG: null,
  WinkerCover_Mat_phong_SG: null,
  Winker_Mat_phong_SG: SLOT.lamp,
  StopLightRed_Mat_phong_SG: SLOT.lamp,
  CamCover_phong_SG: SLOT.lamp,
  initialShadingGroup: SLOT.aluminum, // the brake calipers (red in the scene); silver to match the wheels
  MirrorMat_phong_SG: SLOT.chrome,
  LightReflecMat_phong_SG: SLOT.chrome,
  LightReflectInner_phong_SG: SLOT.chrome,
  Exhaust_Silver_phong_SG: SLOT.chrome,
  WheelHubColor_phong_SG: SLOT.aluminum,
  Wheel2Mat_phong_SG: SLOT.aluminum,
  LightSilverMat_phong_SG: SLOT.aluminum,
  BoltSilver_phong_SG: SLOT.aluminum,
  EngineSilver2_phong_SG: SLOT.aluminum,
  BrakeRotarSilver_phong_SG: SLOT.aluminum,
  SusArm_Silver2_phong_SG: SLOT.aluminum,
  GoldMat_phong_SG: SLOT.aluminum,
  TireMat_phong_SG: SLOT.rubber,
  suspension_silver_phong_SG: SLOT.rubber, // the tire itself, per the scene's own comment
  GomBlack_phong_SG: SLOT.rubber,
  MeshBlack_phong_SG: SLOT.rubber,
  Suspention_Black_phong_SG: SLOT.rubber,
  Suspention_Red_phong_SG: SLOT.trim,
  CarbonBlack_phong_SG: SLOT.trim,
  Chassis_Black_phong_SG: SLOT.trim,
  BodyGlossBlackMat_phong_SG: SLOT.trim,
  BodyMat_BK_phong_SG: SLOT.trim,
  StopLightBK_Mat_phong_SG: SLOT.plastic,
  HeadLightBK_Mat_phong_SG: SLOT.plastic,
  HeadLight_LED_phong_SG: SLOT.plastic,
  // Cabin
  Interior_White_phong_SG: SLOT.plastic,
  Interior_Silver_phong_SG: SLOT.chrome,
  Interior_Black_phong_SG: SLOT.trim,
  Interior_Red_phong_SG: SLOT.lamp,
  Interior_GlossBlack_phong_SG: SLOT.trim,
  Interior_Monitor_phong_SG: SLOT.trim,
  Interior_LineColor_phong_SG: SLOT.plastic,
  Interior_GomBlackq_phong_SG: SLOT.rubber,
  Seat_Black_phong_SG: SLOT.rubber,
  SeatColor_phong_SG: SLOT.plastic,
  PedalsSilver_mat_SG: SLOT.aluminum,
}

function loadSportsCar(dir) {
  const scene = fs.readFileSync(path.join(dir, 'geometry/geometry.pbrt'), 'utf8')
  const parts = []
  let material = null
  for (const line of scene.split('\n')) {
    const m = line.match(/NamedMaterial "([^"]+)"/)
    if (m) material = m[1]
    const s = line.match(/"string filename" \[ "geometry\/(OSD_CarPoly[^"]+\.ply)" \]/)
    if (!s) continue // the Plane_* shapes are the scene's ground and light cards
    if (!(material in CAR_MATERIALS)) throw new Error(`Unmapped material ${material}`)
    const slot = CAR_MATERIALS[material]
    if (slot === null) continue
    parts.push({ file: s[1], material, slot, ...readPly(path.join(dir, 'geometry', s[1])) })
  }
  return parts
}

// ---------------------------------------------------------------------------------------------------------------
// Utah Teapot, 2026 version (Cem Yuksel), from the University of Utah's generator (graphics.cs.utah.edu/teapot):
// teapot_generator.js and teapot_generator.wasm, run here in a sandbox. The options are the page's defaults: all
// parts, Blinn's 3/4 scale, circular, trimmed, chamfered, curvature continuous, round bottom, Yuksel interior,
// both sides, welded vertices. Every triangle takes the hero material.
// ---------------------------------------------------------------------------------------------------------------
async function loadTeapot(dir, resolution) {
  const { default: vm } = await import('node:vm')
  const { createRequire } = await import('node:module')
  const src = fs.readFileSync(path.join(dir, 'teapot_generator.js'), 'utf8')
  const ready = new Promise(resolve => {
    const sandbox = {
      require: createRequire(path.join(dir, 'teapot_generator.js')),
      process, console, Buffer, URL, TextDecoder, TextEncoder, WebAssembly, setTimeout, clearTimeout,
      __dirname: dir, __filename: path.join(dir, 'teapot_generator.js'),
      Module: { locateFile: p => path.join(dir, p), onRuntimeInitialized: () => resolve(sandbox) },
    }
    sandbox.globalThis = sandbox
    vm.createContext(sandbox)
    vm.runInContext(src, sandbox, { filename: 'teapot_generator.js' })
  })
  const sb = await ready
  const ex = sb.wasmExports
  let options = 0
  options |= 1 << 0 | 1 << 1 | 1 << 2 | 1 << 3 // handle, spout, lid, body
  options += 3 << 4 // bottom: round
  options += 3 << 6 // interior: Yuksel 2026
  options |= 1 << 8 | 1 << 9 | 1 << 10 | 1 << 11 | 1 << 12 // 3/4 scale, circular, trimmed, chamfer, curvature
  options += 2 << 14 // texture layout (unused here)
  options |= 1 << 20 | 1 << 21 | 1 << 22 // weld vertices, normals, texture coordinates
  options |= 1 << 23 | 1 << 24 // triangle tips, symmetric triangulation
  options |= 1 << 26 | 1 << 27 // both sides
  ex.tmesh_generate(options, resolution, -1)
  const nf = ex.tmesh_numfaces(), nv = ex.tmesh_numverts()
  const buf = sb.HEAPF32.buffer
  const faces = new Uint32Array(new Int32Array(buf, ex.tmesh_faces(), nf * 3))
  const verts = new Float32Array(buf, ex.tmesh_verts(), nv * 8) // position, normal, texture coordinate
  // The generator's mesh is z up; the lab is y up: (x, y, z) -> (x, z, -y).
  const pos = new Float32Array(nv * 3), nrm = new Float32Array(nv * 3)
  for (let i = 0; i < nv; i++) {
    const v = verts.subarray(i * 8, i * 8 + 8)
    pos.set([v[0], v[2], -v[1]], i * 3)
    nrm.set([v[3], v[5], -v[4]], i * 3)
  }
  ex.tmesh_clear()
  return [{ file: 'teapot', material: 'hero', slot: SLOT.hero, pos, nrm, idx: faces }]
}

// ---------------------------------------------------------------------------------------------------------------
// glTF 2.0 (.gltf + .bin, as Sketchfab exports it): every mesh primitive (triangles) under the default scene, with
// its node transforms applied, mapped onto a slot by its material's name. glTF is y up, like the lab.
// ---------------------------------------------------------------------------------------------------------------
function loadGltf(dir, slots) {
  const g = JSON.parse(fs.readFileSync(path.join(dir, 'scene.gltf'), 'utf8'))
  const bins = g.buffers.map(b => fs.readFileSync(path.join(dir, b.uri)))
  const read = (accessorIndex, Type) => {
    const a = g.accessors[accessorIndex]
    const v = g.bufferViews[a.bufferView]
    const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[a.type]
    if (v.byteStride && v.byteStride !== comps * Type.BYTES_PER_ELEMENT) throw new Error('interleaved buffers are not supported')
    const buf = bins[v.buffer]
    const offset = buf.byteOffset + (v.byteOffset ?? 0) + (a.byteOffset ?? 0)
    return new Type(buf.buffer.slice(offset, offset + a.count * comps * Type.BYTES_PER_ELEMENT))
  }
  const indexType = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array }
  const mul = (a, b) => { // column-major 4 x 4
    const r = new Array(16).fill(0)
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) r[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k]
    return r
  }
  const local = n => {
    if (n.matrix) return n.matrix
    const [x, y, z, w] = n.rotation ?? [0, 0, 0, 1]
    const [sx, sy, sz] = n.scale ?? [1, 1, 1]
    const [tx, ty, tz] = n.translation ?? [0, 0, 0]
    return [
      (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
      2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
      2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
      tx, ty, tz, 1,
    ]
  }
  const parts = []
  const visit = (ni, parent) => {
    const n = g.nodes[ni]
    const m = mul(parent, local(n))
    if (n.mesh !== undefined) {
      for (const p of g.meshes[n.mesh].primitives) {
        if ((p.mode ?? 4) !== 4) continue
        const name = g.materials[p.material]?.name ?? ''
        if (!(name in slots)) throw new Error(`Unmapped material ${name}`)
        const src = read(p.attributes.POSITION, Float32Array)
        const srcN = p.attributes.NORMAL !== undefined ? read(p.attributes.NORMAL, Float32Array) : null
        const nv = src.length / 3
        const pos = new Float32Array(nv * 3), nrm = srcN ? new Float32Array(nv * 3) : null
        for (let i = 0; i < nv; i++) {
          const [x, y, z] = src.subarray(i * 3, i * 3 + 3)
          for (let k = 0; k < 3; k++) pos[i * 3 + k] = m[k] * x + m[4 + k] * y + m[8 + k] * z + m[12 + k]
          if (srcN) {
            // Normals by the upper 3 x 3 (these files have rotation and uniform scale only), renormalized.
            const [a, b, c] = srcN.subarray(i * 3, i * 3 + 3)
            const v = [0, 1, 2].map(k => m[k] * a + m[4 + k] * b + m[8 + k] * c)
            const l = Math.hypot(...v) || 1
            nrm.set(v.map(e => e / l), i * 3)
          }
        }
        const idx = p.indices !== undefined
          ? Uint32Array.from(read(p.indices, indexType[g.accessors[p.indices].componentType]))
          : Uint32Array.from({ length: nv }, (_, i) => i)
        parts.push({ file: `node ${ni}`, material: name, slot: slots[name], pos, nrm, idx })
      }
    }
    for (const c of n.children ?? []) visit(c, m)
  }
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  for (const ni of g.scenes[g.scene ?? 0].nodes) visit(ni, I)
  return parts
}

// Winged Victory of Samothrace (CosmoWenman, CC BY 4.0, captured from the Skulpturhalle Basel's plaster cast): its
// nine texture tiles are one surface; all of it takes the hero material.
const VICTORY_SLOTS = Object.fromEntries(
  [...Array.from({ length: 9 }, (_, i) => `Texture_${i}.001`), 'Untextured.001'].map(n => [n, SLOT.hero]),
)
// Table Fountain, Cleveland Museum of Art 1924.859 (CC0): gilt silver (the hero) and translucent enamel.
const FOUNTAIN_SLOTS = { Fountain_Mats: SLOT.hero, Enamel_Mats: SLOT.enamel }

// ---------------------------------------------------------------------------------------------------------------
// Reduction: one triangle budget for the model, shared by the parts in proportion to their triangle counts, with
// part borders locked so neighbouring parts still meet.
// ---------------------------------------------------------------------------------------------------------------
function reduce(parts, budget) {
  const total = parts.reduce((n, p) => n + p.idx.length / 3, 0)
  const ratio = Math.min(1, budget / total)
  return parts.map(p => {
    const tris = p.idx.length / 3
    const target = Math.max(12, Math.floor(tris * ratio)) * 3
    if (target >= p.idx.length) return p
    const attrs = p.nrm ?? new Float32Array(p.pos.length)
    const [idx] = MeshoptSimplifier.simplifyWithAttributes(p.idx, p.pos, 3, attrs, 3, [0.5, 0.5, 0.5], null, target, 0.02, ['LockBorder'])
    return { ...p, idx }
  })
}

// Merge parts into one indexed mesh with a material per triangle, dropping unreferenced vertices.
function merge(parts) {
  let vcount = 0, tcount = 0
  const remaps = parts.map(p => {
    const remap = new Int32Array(p.pos.length / 3).fill(-1)
    for (const i of p.idx) if (remap[i] < 0) remap[i] = vcount++
    tcount += p.idx.length / 3
    return remap
  })
  const pos = new Float32Array(vcount * 3)
  const nrm = new Float32Array(vcount * 3)
  const idx = new Uint32Array(tcount * 3)
  const mat = new Uint8Array(tcount)
  let t = 0
  parts.forEach((p, k) => {
    const remap = remaps[k]
    for (let i = 0; i < remap.length; i++) {
      const j = remap[i]
      if (j < 0) continue
      pos.set(p.pos.subarray(i * 3, i * 3 + 3), j * 3)
      if (p.nrm) nrm.set(p.nrm.subarray(i * 3, i * 3 + 3), j * 3)
    }
    for (let i = 0; i < p.idx.length; i += 3) {
      idx[t * 3] = remap[p.idx[i]]
      idx[t * 3 + 1] = remap[p.idx[i + 1]]
      idx[t * 3 + 2] = remap[p.idx[i + 2]]
      mat[t++] = p.slot
    }
  })
  return { pos, nrm, idx, mat }
}

// Close a scan into a solid, for subsurface scattering (a random walk needs an inside): drop duplicated
// triangles, then fill each closed boundary loop with a fan around its centroid, wound to match the surface.
// Loops are found on positions (parts that meet at a seam share positions, not indices).
function closeHoles(mesh) {
  const { pos, idx, mat } = mesh
  const nv = pos.length / 3
  const byPos = new Map(), weld = new Uint32Array(nv)
  for (let v = 0; v < nv; v++) {
    const k = `${pos[v * 3]},${pos[v * 3 + 1]},${pos[v * 3 + 2]}`
    if (!byPos.has(k)) byPos.set(k, v)
    weld[v] = byPos.get(k)
  }
  const keep = [], seen = new Set(), dir = new Map()
  for (let t = 0; t < idx.length / 3; t++) {
    const w = [weld[idx[t * 3]], weld[idx[t * 3 + 1]], weld[idx[t * 3 + 2]]]
    const tk = [...w].sort((a, b) => a - b).join(',')
    if (seen.has(tk)) continue
    seen.add(tk)
    keep.push(t)
    for (let i = 0; i < 3; i++) dir.set(`${w[i]},${w[(i + 1) % 3]}`, mat[t])
  }
  // Boundary edges: a directed edge whose reverse no triangle uses.
  const next = new Map()
  for (const [k, slot] of dir) {
    const [a, b] = k.split(',').map(Number)
    if (dir.has(`${b},${a}`)) continue
    if (!next.has(a)) next.set(a, [])
    next.get(a).push({ b, slot })
  }
  const used = new Set(), fills = []
  for (const [a0, outs] of next) {
    for (const { b: b0, slot } of outs) {
      if (used.has(`${a0},${b0}`)) continue
      const loop = [a0]
      let a = a0, b = b0
      used.add(`${a},${b}`)
      while (b !== a0) {
        loop.push(b)
        const n = (next.get(b) ?? []).find(e => !used.has(`${b},${e.b}`))
        if (!n) break
        used.add(`${b},${n.b}`)
        a = b
        b = n.b
      }
      if (b === a0 && loop.length >= 3) fills.push({ loop, slot })
    }
  }
  const extraPos = [], newIdx = [], newMat = []
  for (const t of keep) { newIdx.push(idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]); newMat.push(mat[t]) }
  let v = nv
  for (const { loop, slot } of fills) {
    const c = [0, 1, 2].map(k => loop.reduce((s, i) => s + pos[i * 3 + k], 0) / loop.length)
    extraPos.push(...c)
    // The loop runs along the surface's boundary edges (a -> b); a fill triangle uses each edge reversed.
    for (let i = 0; i < loop.length; i++) {
      newIdx.push(loop[(i + 1) % loop.length], loop[i], v)
      newMat.push(slot)
    }
    v++
  }
  const pos2 = new Float32Array(pos.length + extraPos.length)
  pos2.set(pos)
  pos2.set(extraPos, pos.length)
  const nrm2 = new Float32Array(pos2.length)
  nrm2.set(mesh.nrm)
  // Fill vertices get no normal here: fillMissingNormals gives them the area-weighted face normal.
  console.error(`closeHoles: dropped ${idx.length / 3 - keep.length} duplicate triangles, filled ${fills.length} holes (${fills.reduce((s, f) => s + f.loop.length, 0)} edges)`)
  return { pos: pos2, nrm: nrm2, idx: Uint32Array.from(newIdx), mat: Uint8Array.from(newMat) }
}

// Smooth normals where a part has none: area-weighted face normals.
function fillMissingNormals(mesh) {
  const { pos, nrm, idx } = mesh
  const acc = new Float32Array(nrm.length)
  for (let i = 0; i < idx.length; i += 3) {
    const [a, b, c] = [idx[i] * 3, idx[i + 1] * 3, idx[i + 2] * 3]
    const e1 = [pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]]
    const e2 = [pos[c] - pos[a], pos[c + 1] - pos[a + 1], pos[c + 2] - pos[a + 2]]
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
    for (const v of [a, b, c]) for (let k = 0; k < 3; k++) acc[v + k] += n[k]
  }
  for (let v = 0; v < nrm.length; v += 3) {
    const l = Math.hypot(nrm[v], nrm[v + 1], nrm[v + 2])
    const src = l > 0.5 ? [nrm[v] / l, nrm[v + 1] / l, nrm[v + 2] / l] : (() => {
      const la = Math.hypot(acc[v], acc[v + 1], acc[v + 2]) || 1
      return [acc[v] / la, acc[v + 1] / la, acc[v + 2] / la]
    })()
    nrm.set(src, v)
  }
}

// Canonical frame: y up, centered on x and z, resting on y = 0, longest horizontal side 1.
function normalizeFrame(mesh) {
  const { pos } = mesh
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < pos.length; i += 3) for (let k = 0; k < 3; k++) {
    lo[k] = Math.min(lo[k], pos[i + k])
    hi[k] = Math.max(hi[k], pos[i + k])
  }
  const s = 1 / Math.max(hi[0] - lo[0], hi[2] - lo[2])
  const c = [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]
  for (let i = 0; i < pos.length; i += 3) for (let k = 0; k < 3; k++) pos[i + k] = (pos[i + k] - c[k]) * s
  return { size: [(hi[0] - lo[0]) * s, (hi[1] - lo[1]) * s, (hi[2] - lo[2]) * s] }
}

// ---------------------------------------------------------------------------------------------------------------
// BVH: binned SAH (Wald 2007), leaves of up to 4 triangles (16 where splitting does not pay). Written as inner
// nodes that hold both children's boxes (as in Aila and Laine 2009), so a ray only fetches nodes it enters.
// Node layout (16 floats): left bmin.xyz, left ref; left bmax.xyz, unused; the same for the right child.
// A ref packs (index << 5) | count: count > 0 is a leaf of that many triangles starting at index; count 0 is the
// inner node at index. An empty child has an inverted box.
// ---------------------------------------------------------------------------------------------------------------
function buildBvh(mesh) {
  const { pos, idx } = mesh
  const n = idx.length / 3
  const cen = new Float32Array(n * 3), bmin = new Float32Array(n * 3), bmax = new Float32Array(n * 3)
  for (let t = 0; t < n; t++) for (let k = 0; k < 3; k++) {
    const a = pos[idx[t * 3] * 3 + k], b = pos[idx[t * 3 + 1] * 3 + k], c = pos[idx[t * 3 + 2] * 3 + k]
    bmin[t * 3 + k] = Math.min(a, b, c)
    bmax[t * 3 + k] = Math.max(a, b, c)
    cen[t * 3 + k] = (bmin[t * 3 + k] + bmax[t * 3 + k]) / 2
  }
  const order = new Uint32Array(n).map((_, i) => i)
  const nodes = []
  const BINS = 16, LEAF = 4
  const area = (lo, hi) => {
    const d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
    return d[0] < 0 ? 0 : 2 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
  }
  const boundsOf = (start, count) => {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
    for (let i = start; i < start + count; i++) {
      const t = order[i]
      for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], bmin[t * 3 + k]); hi[k] = Math.max(hi[k], bmax[t * 3 + k]) }
    }
    return [lo, hi]
  }
  nodes.push({ start: 0, count: n })
  const work = [0]
  let maxDepth = 0
  const depth = [0]
  while (work.length) {
    const ni = work.pop()
    const d = depth.pop()
    maxDepth = Math.max(maxDepth, d)
    const node = nodes[ni]
    ;[node.lo, node.hi] = boundsOf(node.start, node.count)
    if (node.count <= LEAF) continue
    // Centroid bounds pick the binning range.
    const clo = [Infinity, Infinity, Infinity], chi = [-Infinity, -Infinity, -Infinity]
    for (let i = node.start; i < node.start + node.count; i++) {
      const t = order[i]
      for (let k = 0; k < 3; k++) { clo[k] = Math.min(clo[k], cen[t * 3 + k]); chi[k] = Math.max(chi[k], cen[t * 3 + k]) }
    }
    let best = { cost: Infinity, axis: -1, split: 0 }
    for (let axis = 0; axis < 3; axis++) {
      const ext = chi[axis] - clo[axis]
      if (ext <= 0) continue
      const cnt = new Int32Array(BINS)
      const blo = Array.from({ length: BINS }, () => [Infinity, Infinity, Infinity])
      const bhi = Array.from({ length: BINS }, () => [-Infinity, -Infinity, -Infinity])
      for (let i = node.start; i < node.start + node.count; i++) {
        const t = order[i]
        const b = Math.min(BINS - 1, Math.floor(((cen[t * 3 + axis] - clo[axis]) / ext) * BINS))
        cnt[b]++
        for (let k = 0; k < 3; k++) { blo[b][k] = Math.min(blo[b][k], bmin[t * 3 + k]); bhi[b][k] = Math.max(bhi[b][k], bmax[t * 3 + k]) }
      }
      // Sweep: cost of each of the BINS - 1 planes.
      const leftArea = new Float64Array(BINS), leftCnt = new Int32Array(BINS)
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], c = 0
      for (let b = 0; b < BINS - 1; b++) {
        c += cnt[b]
        for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], blo[b][k]); hi[k] = Math.max(hi[k], bhi[b][k]) }
        leftArea[b] = area(lo, hi)
        leftCnt[b] = c
      }
      lo = [Infinity, Infinity, Infinity]; hi = [-Infinity, -Infinity, -Infinity]; c = 0
      for (let b = BINS - 1; b > 0; b--) {
        c += cnt[b]
        for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], blo[b][k]); hi[k] = Math.max(hi[k], bhi[b][k]) }
        const cost = leftArea[b - 1] * leftCnt[b - 1] + area(lo, hi) * c
        if (leftCnt[b - 1] > 0 && c > 0 && cost < best.cost) best = { cost, axis, split: b }
      }
    }
    const leafCost = area(node.lo, node.hi) * node.count
    if (best.axis < 0 || (best.cost >= leafCost && node.count <= 16)) continue
    // Partition the triangle range around the chosen plane.
    const ext = chi[best.axis] - clo[best.axis]
    let i = node.start, j = node.start + node.count - 1
    while (i <= j) {
      const t = order[i]
      const b = Math.min(BINS - 1, Math.floor(((cen[t * 3 + best.axis] - clo[best.axis]) / ext) * BINS))
      if (b < best.split) i++
      else { order[i] = order[j]; order[j--] = t }
    }
    const leftCount = i - node.start
    if (leftCount === 0 || leftCount === node.count) continue
    const left = nodes.length
    nodes.push({ start: node.start, count: leftCount }, { start: i, count: node.count - leftCount })
    node.child = left
    node.axis = best.axis
    work.push(left, left + 1)
    depth.push(d + 1, d + 1)
  }
  // Number the inner nodes depth first, then write each with its children's boxes and refs.
  const inner = []
  const innerIndex = new Map()
  const stack = [0]
  while (stack.length) {
    const k = stack.pop()
    if (nodes[k].child === undefined) continue
    innerIndex.set(k, inner.length)
    inner.push(k)
    stack.push(nodes[k].child + 1, nodes[k].child)
  }
  const ref = k => {
    const nd = nodes[k]
    const r = nd.child === undefined ? nd.start * 32 + nd.count : innerIndex.get(k) * 32
    if (nd.count > 31 && nd.child === undefined) throw new Error('leaf too large to pack')
    if (r >= 2 ** 24) throw new Error('ref exceeds float precision')
    return r
  }
  const EMPTY = [1, 1, 1, 0, -1, -1, -1, 0] // inverted box: never entered
  const childData = k => [...nodes[k].lo, ref(k), ...nodes[k].hi, 0]
  let data
  if (!inner.length) {
    data = new Float32Array([...childData(0), ...EMPTY]) // the whole mesh is one leaf
  } else {
    data = new Float32Array(inner.length * 16)
    inner.forEach((k, i) => {
      data.set(childData(nodes[k].child), i * 16)
      data.set(childData(nodes[k].child + 1), i * 16 + 8)
    })
  }
  const innerCount = Math.max(1, inner.length)
  // Triangles in leaf order.
  const idx2 = new Uint32Array(idx.length), mat2 = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const t = order[i]
    idx2.set(idx.subarray(t * 3, t * 3 + 3), i * 3)
    mat2[i] = mesh.mat[t]
  }
  return { nodes: data, nodeCount: innerCount, idx: idx2, mat: mat2, maxDepth }
}

// Octahedral normal encoding (Cigolle et al. 2014, JCGT 3(2)), snorm16.
function octEncode(nrm) {
  const out = new Int16Array((nrm.length / 3) * 2)
  for (let i = 0, j = 0; i < nrm.length; i += 3, j += 2) {
    let [x, y, z] = [nrm[i], nrm[i + 1], nrm[i + 2]]
    const l = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1
    x /= l; y /= l; z /= l
    if (z < 0) {
      const ox = (1 - Math.abs(y)) * (x >= 0 ? 1 : -1)
      const oy = (1 - Math.abs(x)) * (y >= 0 ? 1 : -1)
      x = ox; y = oy
    }
    out[j] = Math.round(Math.max(-1, Math.min(1, x)) * 32767)
    out[j + 1] = Math.round(Math.max(-1, Math.min(1, y)) * 32767)
  }
  return out
}

// File: "LDM1", u32 header length, JSON header (space padded to 4 bytes), then the sections it lists.
function write(name, mesh, bvh, extra) {
  const nrm16 = octEncode(mesh.nrm)
  const sections = [
    ['nodes', bvh.nodes],
    ['pos', mesh.pos],
    ['nrm', nrm16],
    ['idx', bvh.idx],
    ['mat', bvh.mat],
  ]
  let offset = 0
  const layout = {}
  for (const [k, a] of sections) {
    layout[k] = { offset, length: a.length }
    offset += a.byteLength
    offset = (offset + 3) & ~3
  }
  const header = { version: 2, triCount: bvh.idx.length / 3, vertCount: mesh.pos.length / 3, nodeCount: bvh.nodeCount, layout, ...extra }
  let json = JSON.stringify(header)
  while ((json.length + 8) % 4) json += ' '
  const body = Buffer.alloc(offset)
  for (const [k, a] of sections) Buffer.from(a.buffer, a.byteOffset, a.byteLength).copy(body, layout[k].offset)
  const head = Buffer.alloc(8)
  head.write('LDM1', 0, 'ascii')
  head.writeUInt32LE(json.length, 4)
  const raw = Buffer.concat([head, Buffer.from(json, 'ascii'), body])
  const gz = zlib.gzipSync(raw, { level: 9 })
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const out = path.join(OUT_DIR, `${name}.bin.gz`)
  fs.writeFileSync(out, gz)
  return { out, raw: raw.length, gz: gz.length, header }
}

// ---------------------------------------------------------------------------------------------------------------
const [, , which, src, arg] = process.argv
const SOURCES = {
  sportscar: { load: () => loadSportsCar(src), budget: arg ? +arg : 100_000, credit: 'Sports Car by Yasutoshi Mori (CC BY 4.0)' },
  teapot: { load: () => loadTeapot(src, arg ? +arg : 24), budget: Infinity, credit: 'Utah Teapot, University of Utah (2026 version, Cem Yuksel)' },
  victory: { load: () => loadGltf(src, VICTORY_SLOTS), budget: arg ? +arg : 160_000, credit: 'Winged Victory of Samothrace by CosmoWenman (CC BY 4.0)', solid: true },
  fountain: { load: () => loadGltf(src, FOUNTAIN_SLOTS), budget: arg ? +arg : 180_000, credit: 'Table Fountain 1924.859, Cleveland Museum of Art (CC0)' },
}
if (!SOURCES[which] || !src) {
  console.error('usage: node tools/lookdev-models/build.mjs sportscar <pbrt-v4-scenes/sportscar dir> [triangle budget]')
  console.error('       node tools/lookdev-models/build.mjs teapot <dir with teapot_generator.js and .wasm> [resolution]')
  console.error('       node tools/lookdev-models/build.mjs victory|fountain <dir with scene.gltf> [triangle budget]')
  process.exit(1)
}
const parts = await SOURCES[which].load()
const before = parts.reduce((n, p) => n + p.idx.length / 3, 0)
const reduced = reduce(parts, SOURCES[which].budget)
let mesh = merge(reduced)
if (SOURCES[which].solid) mesh = closeHoles(mesh)
fillMissingNormals(mesh)
const frame = normalizeFrame(mesh)
const bvh = buildBvh(mesh)
const res = write(which, mesh, bvh, { size: frame.size, credit: SOURCES[which].credit })
console.log(JSON.stringify({ parts: parts.length, trianglesIn: before, trianglesOut: bvh.idx.length / 3, verts: mesh.pos.length / 3, nodes: bvh.nodeCount, maxDepth: bvh.maxDepth, size: frame.size, rawMB: +(res.raw / 1e6).toFixed(2), gzMB: +(res.gz / 1e6).toFixed(2), out: res.out }, null, 1))
