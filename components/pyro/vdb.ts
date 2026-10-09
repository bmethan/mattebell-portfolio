// OpenVDB files, written in the browser, so the lab's fields can be rendered in Houdini, Blender or any other
// program that reads them. Each grid is a standard Tree_float_5_4_3 (or Tree_vec3s_5_4_3 for a vector field): a root,
// internal nodes of 32^3 and 16^3 children, and leaves of 8^3 voxels, of which only the leaves that hold some active
// voxels are written. File format 224, values stored as 32-bit floats with OpenVDB's active-mask compression (only
// active voxels are stored; the rest read back as the background), no zip or Blosc.

export interface VdbGrid {
  name: string
  data: Float32Array // one value per voxel (three for a vector grid), x fastest, then y, then z
  vector?: boolean
  background?: number // what an inactive voxel reads as (0 by default)
  fogVolume?: boolean // the grid class: a fog volume (density), or unclassified
}

export interface VdbVolume {
  n: [number, number, number] // voxels along x, y, z
  voxelSize: number // m
  origin: [number, number, number] // the centre of voxel (0, 0, 0), m
  active: Uint8Array // per voxel, 1 where it holds data (the same for every grid)
  grids: VdbGrid[]
  creator?: string
}

const MAGIC = 0x56444220 // "VDB "
const FILE_VERSION = 224
const COMPRESS_ACTIVE_MASK = 0x2
const NO_MASK_OR_INACTIVE_VALS = 0 // inactive values are all the background: only active values are stored
const NO_MASK_AND_ALL_VALS = 6 // every value is stored

class Writer {
  private buf = new ArrayBuffer(1 << 20)
  private view = new DataView(this.buf)
  private bytes = new Uint8Array(this.buf)
  pos = 0

  private room(n: number) {
    if (this.pos + n <= this.buf.byteLength) return
    let size = this.buf.byteLength * 2
    while (size < this.pos + n) size *= 2
    const next = new ArrayBuffer(size)
    new Uint8Array(next).set(this.bytes)
    this.buf = next
    this.view = new DataView(next)
    this.bytes = new Uint8Array(next)
  }
  u8(v: number) {
    this.room(1)
    this.view.setUint8(this.pos, v)
    this.pos += 1
  }
  i32(v: number) {
    this.room(4)
    this.view.setInt32(this.pos, v, true)
    this.pos += 4
  }
  u32(v: number) {
    this.room(4)
    this.view.setUint32(this.pos, v, true)
    this.pos += 4
  }
  i64(v: number) {
    this.room(8)
    this.view.setBigInt64(this.pos, BigInt(v), true)
    this.pos += 8
  }
  f32(v: number) {
    this.room(4)
    this.view.setFloat32(this.pos, v, true)
    this.pos += 4
  }
  f64(v: number) {
    this.room(8)
    this.view.setFloat64(this.pos, v, true)
    this.pos += 8
  }
  ascii(s: string) {
    this.room(s.length)
    for (let i = 0; i < s.length; i++) this.bytes[this.pos + i] = s.charCodeAt(i) & 0x7f
    this.pos += s.length
  }
  // A string as OpenVDB writes one: its length (32 bits), then its characters.
  str(s: string) {
    this.u32(s.length)
    this.ascii(s)
  }
  // A node mask: its bits in 64-bit words, little-endian, bit i in word i >> 6.
  mask(bits: Uint8Array) {
    const words = bits.length / 64
    this.room(words * 8)
    for (let w = 0; w < words; w++) {
      let lo = 0
      let hi = 0
      for (let b = 0; b < 32; b++) if (bits[w * 64 + b]) lo |= 1 << b
      for (let b = 0; b < 32; b++) if (bits[w * 64 + 32 + b]) hi |= 1 << b
      this.view.setUint32(this.pos, lo >>> 0, true)
      this.view.setUint32(this.pos + 4, hi >>> 0, true)
      this.pos += 8
    }
  }
  patchI64(at: number, v: number) {
    this.view.setBigInt64(at, BigInt(v), true)
  }
  result() {
    return this.bytes.slice(0, this.pos)
  }
}

// Metadata entries: name, type name, then the value's size in bytes and the value.
type Meta = { name: string } & ({ type: 'string'; value: string } | { type: 'bool'; value: boolean } | { type: 'vec3i'; value: [number, number, number] } | { type: 'int64'; value: number })
function writeMeta(w: Writer, entries: Meta[]) {
  w.u32(entries.length)
  for (const m of entries) {
    w.str(m.name)
    w.str(m.type)
    if (m.type === 'string') {
      w.u32(m.value.length)
      w.ascii(m.value)
    } else if (m.type === 'bool') {
      w.u32(1)
      w.u8(m.value ? 1 : 0)
    } else if (m.type === 'vec3i') {
      w.u32(12)
      m.value.forEach(v => w.i32(v))
    } else {
      w.u32(8)
      w.i64(m.value)
    }
  }
}

const uuid = () => {
  const h = Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('')
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`
}

// The tree's shape, shared by every grid: which leaves (8^3 blocks) hold active voxels, grouped under the 16^3-leaf
// internal nodes (128 voxels a side) under the one 32^3 node at the origin (4096 a side, ample for the lab's grids).
// Children are written in the order of their bit in the parent's mask: x slowest, z fastest.
interface Leaf {
  origin: [number, number, number]
  mask: Uint8Array // 512, by offset (x << 6) | (y << 3) | z
  count: number
}
interface Node1 {
  origin: [number, number, number]
  leaves: { bit: number; leaf: Leaf }[]
}
function buildTree(vol: VdbVolume) {
  const [nx, ny, nz] = vol.n
  if (Math.max(nx, ny, nz) > 4096) throw new Error('grid too large for one root child')
  const node1s = new Map<number, Node1>()
  const bbox = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }
  let voxels = 0
  for (let lx = 0; lx < nx; lx += 8)
    for (let ly = 0; ly < ny; ly += 8)
      for (let lz = 0; lz < nz; lz += 8) {
        const mask = new Uint8Array(512)
        let count = 0
        for (let x = lx; x < Math.min(lx + 8, nx); x++)
          for (let y = ly; y < Math.min(ly + 8, ny); y++)
            for (let z = lz; z < Math.min(lz + 8, nz); z++) {
              if (!vol.active[(z * ny + y) * nx + x]) continue
              mask[((x - lx) << 6) | ((y - ly) << 3) | (z - lz)] = 1
              count++
              const c = [x, y, z]
              for (let k = 0; k < 3; k++) {
                bbox.min[k] = Math.min(bbox.min[k], c[k])
                bbox.max[k] = Math.max(bbox.max[k], c[k])
              }
            }
        if (!count) continue
        voxels += count
        const o1: [number, number, number] = [lx & ~127, ly & ~127, lz & ~127]
        const key = (o1[0] >> 7) * 1024 + (o1[1] >> 7) * 32 + (o1[2] >> 7)
        let node = node1s.get(key)
        if (!node) {
          node = { origin: o1, leaves: [] }
          node1s.set(key, node)
        }
        const bit = (((lx & 127) >> 3) << 8) | (((ly & 127) >> 3) << 4) | ((lz & 127) >> 3)
        node.leaves.push({ bit, leaf: { origin: [lx, ly, lz], mask, count } })
      }
  const children = [...node1s.values()]
    .map(n => ({ bit: ((n.origin[0] >> 7) << 10) | ((n.origin[1] >> 7) << 5) | (n.origin[2] >> 7), node: n }))
    .sort((a, b) => a.bit - b.bit)
  children.forEach(c => c.node.leaves.sort((a, b) => a.bit - b.bit))
  return { children, bbox, voxels }
}

export function writeVdb(vol: VdbVolume): Uint8Array<ArrayBuffer> {
  const w = new Writer()
  const tree = buildTree(vol)
  const [nx, ny] = vol.n
  // Header: magic (64 bits), file format, library version, has-grid-offsets, uuid.
  w.i64(MAGIC)
  w.u32(FILE_VERSION)
  w.u32(10)
  w.u32(0)
  w.u8(1)
  w.ascii(uuid())
  writeMeta(w, vol.creator ? [{ name: 'creator', type: 'string', value: vol.creator }] : [])
  w.u32(vol.grids.length)
  const empty = tree.children.length === 0
  for (const g of vol.grids) {
    const bg = g.background ?? 0
    const comps = g.vector ? 3 : 1
    // Grid descriptor: unique name, tree type, instance parent; then the grid's stream positions, filled in below.
    w.str(g.name)
    w.str(g.vector ? 'Tree_vec3s_5_4_3' : 'Tree_float_5_4_3')
    w.str('')
    const offsetsAt = w.pos
    w.i64(0)
    w.i64(0)
    w.i64(0)
    const gridPos = w.pos
    w.u32(COMPRESS_ACTIVE_MASK)
    const meta: Meta[] = [
      { name: 'class', type: 'string', value: g.fogVolume ? 'fog volume' : 'unknown' },
      { name: 'name', type: 'string', value: g.name },
      { name: 'is_local_space', type: 'bool', value: false },
      { name: 'is_saved_as_half_float', type: 'bool', value: false },
    ]
    if (g.vector) meta.push({ name: 'vector_type', type: 'string', value: 'invariant' })
    if (!empty) {
      meta.push({ name: 'file_bbox_min', type: 'vec3i', value: tree.bbox.min as [number, number, number] })
      meta.push({ name: 'file_bbox_max', type: 'vec3i', value: tree.bbox.max as [number, number, number] })
      meta.push({ name: 'file_voxel_count', type: 'int64', value: tree.voxels })
    }
    writeMeta(w, meta)
    // Transform: a uniform scale and a translation, so voxel (i, j, k) is centred at origin + (i, j, k) * voxelSize.
    const s = vol.voxelSize
    w.str('UniformScaleTranslateMap')
    const vec = (a: number, b: number, c: number) => {
      w.f64(a)
      w.f64(b)
      w.f64(c)
    }
    vec(...vol.origin) // translation
    vec(s, s, s) // scale
    vec(s, s, s) // voxel size
    vec(1 / s, 1 / s, 1 / s) // inverse scale
    vec(1 / (s * s), 1 / (s * s), 1 / (s * s)) // inverse scale squared
    vec(0.5 / s, 0.5 / s, 0.5 / s) // inverse twice scale
    // Topology: buffer count, the root (background, tiles, children), then each node's masks and tile values.
    w.i32(1)
    const value = (v: number) => {
      for (let k = 0; k < comps; k++) w.f32(v)
    }
    value(bg)
    w.u32(0)
    w.u32(empty ? 0 : 1)
    if (!empty) {
      w.i32(0)
      w.i32(0)
      w.i32(0)
      // The 32^3 node: which 16^3 nodes it holds; no active tiles, so no tile values are stored.
      const top = new Uint8Array(32768)
      tree.children.forEach(c => (top[c.bit] = 1))
      w.mask(top)
      w.mask(new Uint8Array(32768))
      w.u8(NO_MASK_OR_INACTIVE_VALS)
      for (const { node } of tree.children) {
        const kids = new Uint8Array(4096)
        node.leaves.forEach(l => (kids[l.bit] = 1))
        w.mask(kids)
        w.mask(new Uint8Array(4096))
        w.u8(NO_MASK_OR_INACTIVE_VALS)
        for (const { leaf } of node.leaves) w.mask(leaf.mask)
      }
    }
    // Buffers: per leaf, its value mask again and its active values, in offset order.
    const blockPos = w.pos
    for (const { node } of tree.children)
      for (const { leaf } of node.leaves) {
        w.mask(leaf.mask)
        w.u8(leaf.count === 512 ? NO_MASK_AND_ALL_VALS : NO_MASK_OR_INACTIVE_VALS)
        const [ox, oy, oz] = leaf.origin
        for (let i = 0; i < 512; i++) {
          if (!leaf.mask[i]) continue
          const v = ((oz + (i & 7)) * ny + oy + ((i >> 3) & 7)) * nx + ox + (i >> 6)
          for (let k = 0; k < comps; k++) w.f32(g.data[v * comps + k])
        }
      }
    const endPos = w.pos
    w.patchI64(offsetsAt, gridPos)
    w.patchI64(offsetsAt + 8, blockPos)
    w.patchI64(offsetsAt + 16, endPos)
  }
  return w.result()
}

// A zip archive, stored without compression (the VDBs are mostly floats, which deflate barely shrinks), built a file at
// a time: each goes into a Blob as it is added, which the browser may keep on disk, so a long sequence need not stay
// in memory.
const CRC = (() => {
  const t = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    t[n] = c >>> 0
  }
  return t
})()
const crc32 = (b: Uint8Array) => {
  let c = 0xffffffff
  for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}
export class ZipStore {
  private parts: Blob[] = []
  private central: Uint8Array<ArrayBuffer>[] = []
  private offset = 0
  private count = 0

  add(path: string, data: Uint8Array<ArrayBuffer>) {
    const name = new TextEncoder().encode(path)
    const crc = crc32(data)
    const local = new DataView(new ArrayBuffer(30))
    local.setUint32(0, 0x04034b50, true)
    local.setUint16(4, 20, true)
    local.setUint32(14, crc, true)
    local.setUint32(18, data.length, true)
    local.setUint32(22, data.length, true)
    local.setUint16(26, name.length, true)
    this.parts.push(new Blob([local.buffer, name, data]))
    const dir = new DataView(new ArrayBuffer(46))
    dir.setUint32(0, 0x02014b50, true)
    dir.setUint16(4, 20, true)
    dir.setUint16(6, 20, true)
    dir.setUint32(16, crc, true)
    dir.setUint32(20, data.length, true)
    dir.setUint32(24, data.length, true)
    dir.setUint16(28, name.length, true)
    dir.setUint32(42, this.offset, true)
    const entry = new Uint8Array(46 + name.length)
    entry.set(new Uint8Array(dir.buffer))
    entry.set(name, 46)
    this.central.push(entry)
    this.offset += 30 + name.length + data.length
    this.count++
  }

  finish(): Blob {
    const size = this.central.reduce((a, c) => a + c.length, 0)
    const end = new DataView(new ArrayBuffer(22))
    end.setUint32(0, 0x06054b50, true)
    end.setUint16(8, this.count, true)
    end.setUint16(10, this.count, true)
    end.setUint32(12, size, true)
    end.setUint32(16, this.offset, true)
    return new Blob([...this.parts, ...this.central, end.buffer], { type: 'application/zip' })
  }
}
