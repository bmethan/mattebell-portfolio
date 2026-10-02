// Uniform buffers as one list of 16-byte fields (vec4s, or arrays of them), from which both the WGSL struct and
// the host's packing come, so the two cannot drift apart. Every field is a full vec4, so WGSL's alignment rules
// never insert padding: field i (or array element j) sits at byte (offset + j) * 16.

export type FieldKind = 'vec4f' | 'vec4u' | 'vec4i'
export interface Field {
  name: string
  kind: FieldKind
  count?: number // an array of this many
}

export class UniformLayout {
  readonly offsets = new Map<string, { at: number; kind: FieldKind; count: number }>()
  readonly vec4s: number
  constructor(readonly structName: string, readonly fields: Field[]) {
    let at = 0
    for (const f of fields) {
      const count = f.count ?? 1
      this.offsets.set(f.name, { at, kind: f.kind, count })
      at += count
    }
    this.vec4s = at
  }

  get byteSize() {
    return this.vec4s * 16
  }

  wgsl() {
    const ty = (k: FieldKind) => (k === 'vec4f' ? 'vec4<f32>' : k === 'vec4u' ? 'vec4<u32>' : 'vec4<i32>')
    const body = this.fields.map(f => `  ${f.name}: ${f.count ? `array<${ty(f.kind)}, ${f.count}>` : ty(f.kind)},`).join('\n')
    return `struct ${this.structName} {\n${body}\n}\n`
  }

  // A host-side image of the buffer; set fields by name, then upload .data.
  writer() {
    return new UniformWriter(this)
  }
}

export class UniformWriter {
  readonly data: ArrayBuffer
  private f: Float32Array
  private u: Uint32Array
  private i: Int32Array
  constructor(private layout: UniformLayout) {
    this.data = new ArrayBuffer(layout.byteSize)
    this.f = new Float32Array(this.data)
    this.u = new Uint32Array(this.data)
    this.i = new Int32Array(this.data)
  }

  // Up to four components of field name (array element el).
  set(name: string, values: ArrayLike<number>, el = 0) {
    const o = this.layout.offsets.get(name)
    if (!o) throw new Error(`uniform field ${name} not in ${this.layout.structName}`)
    if (el >= o.count) throw new Error(`uniform field ${name}[${el}] out of range`)
    const base = (o.at + el) * 4
    const arr = o.kind === 'vec4f' ? this.f : o.kind === 'vec4u' ? this.u : this.i
    for (let k = 0; k < Math.min(4, values.length); k++) arr[base + k] = values[k]
    return this
  }

  // Raw 32-bit words from the start of field name (for whole arrays: the Sobol' table, the materials).
  setWords(name: string, words: Float32Array | Uint32Array) {
    const o = this.layout.offsets.get(name)
    if (!o) throw new Error(`uniform field ${name} not in ${this.layout.structName}`)
    if (words.length > o.count * 4) throw new Error(`uniform field ${name} overflows`)
    const arr = words instanceof Float32Array ? this.f : this.u
    arr.set(words as never, o.at * 4)
    return this
  }
}
