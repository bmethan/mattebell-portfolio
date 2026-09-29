import { VERT, E_TABLE_FRAG, TRACE_FRAG, DISPLAY_FRAG } from './shaders'
import { kelvinToACEScg } from './color'
import { SCENE_MATERIALS, HERO_PRESETS, MATERIAL_FIELDS, type OpenPBR, type Hero } from './materials'

export type Pass = 'beauty' | 'diffuse' | 'specular' | 'albedo' | 'normal'
export type View = 'aces' | 'agx' | 'neutral' | 'standard'
export type { Hero }

export interface LabState {
  keyAz: number
  keyEl: number
  key: boolean
  fill: boolean
  rim: boolean
  keyKelvin: number
  exposure: number
  hero: Hero
  heroRoughness: number | null
  pass: Pass
  view: View
  multiscatter: boolean
  furnace: boolean
}

export const DEFAULT_STATE: LabState = {
  keyAz: 0.62,
  keyEl: 0.52,
  key: true,
  fill: true,
  rim: true,
  keyKelvin: 4300,
  exposure: 0.5,
  hero: 'carpaint',
  heroRoughness: null,
  pass: 'beauty',
  view: 'aces',
  multiscatter: true,
  furnace: false,
}

export interface LabStatus {
  spp: number
  target: number
  converged: boolean
}

type Vec3 = [number, number, number]
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const length = (a: Vec3) => Math.sqrt(dot(a, a))
const normalize = (a: Vec3): Vec3 => scale(a, 1 / (length(a) || 1))

const PASS_ID: Record<Pass, number> = { beauty: 0, diffuse: 1, specular: 2, albedo: 3, normal: 4 }
const VIEW_ID: Record<View, number> = { aces: 0, agx: 1, neutral: 2, standard: 3 }

export const TARGET_SPP = 1024
const HALF_FLOAT_MAX_SPP = 256
const MAX_BOUNCES = 8

// Sobol' direction numbers for dimensions 1-3, verbatim from Burley 2020 ("Practical Hash-based Owen
// Scrambling", JCGT 9(4)) supplemental sobol.cpp. Packed one bit per uvec4: (dim1, dim2, dim3, 0).
const SOBOL_D1 = [
  0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
  0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
  0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
  0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
]
const SOBOL_D2 = [
  0x80000000, 0xc0000000, 0x60000000, 0x90000000, 0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
  0x68800000, 0x9cc00000, 0xee600000, 0x55900000, 0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
  0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000, 0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
  0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590, 0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,
]
const SOBOL_D3 = [
  0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
  0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000, 0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
  0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000, 0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
  0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050, 0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,
]
const SOBOL_UNIFORM = new Uint32Array(SOBOL_D1.flatMap((d1, i) => [d1, SOBOL_D2[i], SOBOL_D3[i], 0]))
const E_SIZE = 32 // mu and roughness resolution of the albedo tables
const E_LAYERS = 16 // IOR resolution
const LUT_SIZE = 65
const LUT_URL = '/lookdev/aces2-sdr-rec709-65.png'
const INDIRECT_CLAMP = 12

// Scene: three unit spheres resting on a floor at y = 0.
export const BALL_X = [-2.4, 0, 2.4]
const CENTROID: Vec3 = [0, 1, 0]

// Camera rig. The horizontal extent is fixed so the three balls always fit.
const CAM_POS: Vec3 = [0, 1.45, 10.5]
const CAM_TARGET: Vec3 = [0, 0.95, 0]
const H_HALF_EXTENT = 4.0

function direction(az: number, el: number): Vec3 {
  return [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)]
}

// Rectangular softbox facing the scene centroid; emits toward +normal.
function rectLight(az: number, el: number, dist: number, w: number, h: number) {
  const dir = direction(az, el)
  const center = add(CENTROID, scale(dir, dist))
  const n = scale(dir, -1)
  let uAxis = cross([0, 1, 0], n)
  if (length(uAxis) < 1e-3) uAxis = [1, 0, 0]
  uAxis = normalize(uAxis)
  const vAxis = normalize(cross(n, uAxis))
  const U = scale(uAxis, w)
  const V = scale(vAxis, h)
  const corner = sub(sub(center, scale(U, 0.5)), scale(V, 0.5))
  return { corner, U, V }
}

// KHR_parallel_shader_compile. The path tracer takes seconds to compile on some drivers (Direct3D's compiler
// inlines every function call), so compilation is started without asking for its status: any status query
// blocks the page until the compiler finishes. With the extension, completion is polled instead.
const COMPLETION_STATUS_KHR = 0x91b1

function startProgram(gl: WebGL2RenderingContext, fsSrc: string) {
  const p = gl.createProgram()!
  for (const [type, src] of [
    [gl.VERTEX_SHADER, VERT],
    [gl.FRAGMENT_SHADER, fsSrc],
  ] as const) {
    const sh = gl.createShader(type)!
    gl.shaderSource(sh, src)
    gl.compileShader(sh)
    gl.attachShader(p, sh)
  }
  gl.linkProgram(p)
  return p
}

function finishProgram(gl: WebGL2RenderingContext, p: WebGLProgram) {
  const shaders = gl.getAttachedShaders(p) ?? []
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const logs = shaders.map(s => gl.getShaderInfoLog(s)).filter(Boolean)
    throw new Error(`Shader program failed: ${[gl.getProgramInfoLog(p), ...logs].join('\n')}`)
  }
  shaders.forEach(s => {
    gl.detachShader(p, s)
    gl.deleteShader(s)
  })
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

export class LookdevEngine {
  static detect(): boolean {
    try {
      const c = document.createElement('canvas')
      const gl = c.getContext('webgl2')
      if (!gl) return false
      const ok = !!gl.getExtension('EXT_color_buffer_float') || !!gl.getExtension('EXT_color_buffer_half_float')
      gl.getExtension('WEBGL_lose_context')?.loseContext()
      return ok
    } catch {
      return false
    }
  }

  // Resolves true once the shaders are compiled and the albedo tables built (false if disposed first);
  // rejects if a shader fails to compile.
  readonly whenReady: Promise<boolean>
  private gl: WebGL2RenderingContext
  private vao: WebGLVertexArrayObject
  private progTrace: WebGLProgram
  private progDisplay: WebGLProgram
  private progTable: WebGLProgram
  private live = false
  private uniformCache = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>()
  private accumTex: WebGLTexture[] = []
  private accumFbo: WebGLFramebuffer[] = []
  private eTable: WebGLTexture
  private acesLut: WebGLTexture
  private acesReady = false
  private accumFmt: { internal: number; type: number }
  private width = 0
  private height = 0
  private spp = 0
  private target = TARGET_SPP
  private ping = 0
  private sppPerFrame = 1
  private rafId = 0
  private lastT = 0
  private visible = false
  private interacting = false
  private sceneDirty = true
  private disposed = false
  private state: LabState

  constructor(
    private canvas: HTMLCanvasElement,
    private onStatus: (s: LabStatus) => void,
    initial: LabState,
    private onLost?: () => void,
  ) {
    this.state = { ...initial }
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    })
    if (!gl) throw new Error('WebGL2 unavailable')
    this.gl = gl

    if (gl.getExtension('EXT_color_buffer_float')) {
      this.accumFmt = { internal: gl.RGBA32F, type: gl.FLOAT }
    } else if (gl.getExtension('EXT_color_buffer_half_float')) {
      // Half-float running means stop absorbing small updates once the blend weight drops below the format's
      // resolution, so cap the sample count on this path.
      this.accumFmt = { internal: gl.RGBA16F, type: gl.HALF_FLOAT }
      this.target = Math.min(this.target, HALF_FLOAT_MAX_SPP)
    } else {
      throw new Error('Float render targets unavailable')
    }

    this.vao = gl.createVertexArray()!
    const parallel = !!gl.getExtension('KHR_parallel_shader_compile')
    this.progTrace = startProgram(gl, TRACE_FRAG)
    this.progDisplay = startProgram(gl, DISPLAY_FRAG)
    this.progTable = startProgram(gl, E_TABLE_FRAG)

    // Placeholder until the ACES 2.0 LUT arrives, so the sampler always has a complete texture.
    this.acesLut = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_3D, this.acesLut)
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]))
    this.eTable = this.createAlbedoTexture()
    this.loadAcesLut()

    canvas.addEventListener('webglcontextlost', this.handleLost)
    this.whenReady = this.warmUp(parallel)
  }

  private async warmUp(parallel: boolean) {
    const gl = this.gl
    const progs = [this.progTrace, this.progDisplay, this.progTable]
    // Without the extension there is nothing to poll: yield once, then the status query below blocks.
    for (;;) {
      await sleep(parallel ? 50 : 0)
      if (this.disposed || gl.isContextLost()) return false
      if (!parallel || progs.every(p => gl.getProgramParameter(p, COMPLETION_STATUS_KHR))) break
    }
    progs.forEach(p => finishProgram(gl, p))
    this.buildAlbedoTables()
    gl.deleteProgram(this.progTable)
    this.live = true
    this.sceneDirty = true
    this.kick()
    return true
  }

  private handleLost = (e: Event) => {
    e.preventDefault()
    this.stop()
    this.onLost?.()
  }

  // Directional albedo tables for multiple-scattering compensation and layering (Kulla and Conty 2017,
  // Turquin 2019), integrated on the GPU once at startup: one draw per IOR slice of a 3D texture.
  private createAlbedoTexture() {
    const gl = this.gl
    const tex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_3D, tex)
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA16F, E_SIZE, E_SIZE, E_LAYERS, 0, gl.RGBA, gl.HALF_FLOAT, null)
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE)
    return tex
  }

  private buildAlbedoTables() {
    const gl = this.gl
    const tex = this.eTable
    const prog = this.progTable
    const fbo = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.viewport(0, 0, E_SIZE, E_SIZE)
    gl.useProgram(prog)
    gl.bindVertexArray(this.vao)
    gl.uniform2f(gl.getUniformLocation(prog, 'uSize'), E_SIZE, E_SIZE)
    gl.uniform1i(gl.getUniformLocation(prog, 'uSamples'), 2048)
    gl.uniform1i(gl.getUniformLocation(prog, 'uFresnelSamples'), 256)
    const uLayer = gl.getUniformLocation(prog, 'uLayer')
    for (let layer = 0; layer < E_LAYERS; layer++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, tex, 0, layer)
      gl.uniform1f(uLayer, layer / (E_LAYERS - 1))
      gl.drawArrays(gl.TRIANGLES, 0, 3)
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.deleteFramebuffer(fbo)
  }

  // ACES 2.0 SDR Output Transform, baked from OpenColorIO 2.5 into a 65^3 LUT stored as a PNG strip.
  private async loadAcesLut() {
    try {
      const blob = await (await fetch(LUT_URL)).blob()
      const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
      if (this.disposed) return
      if (bitmap.width !== LUT_SIZE || bitmap.height !== LUT_SIZE * LUT_SIZE) throw new Error('Unexpected LUT size')
      // Upload the bitmap straight to the GPU. A 2D-canvas readback would be randomized or blanked by
      // anti-fingerprinting browsers. With UNPACK_IMAGE_HEIGHT at its default, the N x N*N strip splits
      // into N slices of N rows, which is the layout the LUT was baked in.
      const gl = this.gl
      gl.bindTexture(gl.TEXTURE_3D, this.acesLut)
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE)
      gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, 0)
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, LUT_SIZE, LUT_SIZE, LUT_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, bitmap)
      bitmap.close()
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE)
      this.acesReady = true
      if (this.live && this.width && this.spp > 0) this.display()
    } catch (err) {
      console.warn('ACES 2.0 LUT unavailable; using AgX', err)
    }
  }

  private makeTex(w: number, h: number, internal: number, type: number) {
    const gl = this.gl
    const tex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, gl.RGBA, type, null)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    return tex
  }

  private loc(prog: WebGLProgram, name: string) {
    let m = this.uniformCache.get(prog)
    if (!m) {
      m = new Map()
      this.uniformCache.set(prog, m)
    }
    if (!m.has(name)) m.set(name, this.gl.getUniformLocation(prog, name))
    return m.get(name) ?? null
  }

  resize(cssW: number, cssH: number, dpr: number) {
    // Cap the internal resolution so the path tracer stays interactive on large or dense displays.
    const maxPixels = 1_400_000
    let scaleF = Math.min(dpr, 1.5)
    if (cssW * cssH * scaleF * scaleF > maxPixels) scaleF = Math.sqrt(maxPixels / (cssW * cssH))
    const w = Math.max(1, Math.round(cssW * scaleF))
    const h = Math.max(1, Math.round(cssH * scaleF))
    if (w === this.width && h === this.height) return
    this.width = w
    this.height = h
    this.canvas.width = w
    this.canvas.height = h

    const gl = this.gl
    this.accumTex.forEach(t => gl.deleteTexture(t))
    this.accumFbo.forEach(f => gl.deleteFramebuffer(f))
    this.accumTex = []
    this.accumFbo = []
    for (let i = 0; i < 2; i++) {
      const tex = this.makeTex(w, h, this.accumFmt.internal, this.accumFmt.type)
      const fbo = gl.createFramebuffer()!
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
      this.accumTex.push(tex)
      this.accumFbo.push(fbo)
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    this.sceneDirty = true
    this.reset()
  }

  setState(patch: Partial<LabState>) {
    const displayOnly = Object.keys(patch).every(k => k === 'exposure' || k === 'view')
    this.state = { ...this.state, ...patch }
    if (displayOnly && this.spp > 0) {
      // Exposure and view transform act on the accumulated radiance, so no re-render is needed.
      this.display()
      return
    }
    this.sceneDirty = true
    this.reset()
  }

  setInteracting(on: boolean) {
    this.interacting = on
    if (on) this.sppPerFrame = 1
  }

  setVisible(v: boolean) {
    this.visible = v
    if (v) this.kick()
    else this.stop()
  }

  // Sample count at which the progressive render stops. Raising it continues the current accumulation.
  setTarget(n: number) {
    const cap = this.accumFmt.type === this.gl.HALF_FLOAT ? HALF_FLOAT_MAX_SPP : Infinity
    this.target = Math.max(1, Math.min(cap, Math.floor(n)))
    this.kick()
  }

  // Normalized screen position (0..1, y down) of a world point, used for HTML labels.
  project(p: Vec3) {
    const { fwd, right, up, tanH, tanV } = this.camera()
    const d = sub(p, CAM_POS)
    const z = dot(d, fwd)
    const x = dot(d, right) / z / tanH
    const y = dot(d, up) / z / tanV
    return { x: (x + 1) / 2, y: 1 - (y + 1) / 2 }
  }

  private camera() {
    const fwd = normalize(sub(CAM_TARGET, CAM_POS))
    const right = normalize(cross(fwd, [0, 1, 0]))
    const up = cross(right, fwd)
    const tanH = H_HALF_EXTENT / length(sub(CAM_TARGET, CAM_POS))
    const aspect = this.width && this.height ? this.width / this.height : 2.39
    const tanV = tanH / aspect
    return { fwd, right, up, tanH, tanV }
  }

  private reset() {
    this.spp = 0
    this.kick()
  }

  private kick() {
    if (this.disposed || !this.live || !this.visible || this.rafId || !this.width) return
    this.lastT = 0
    this.rafId = requestAnimationFrame(this.step)
  }

  private stop() {
    if (this.rafId) cancelAnimationFrame(this.rafId)
    this.rafId = 0
  }

  private uploadScene() {
    const gl = this.gl
    const s = this.state
    const L = (n: string) => this.loc(this.progTrace, n)

    const cam = this.camera()
    gl.uniform3fv(L('uCamPos'), CAM_POS)
    gl.uniform3fv(L('uCamFwd'), cam.fwd)
    gl.uniform3fv(L('uCamRight'), cam.right)
    gl.uniform3fv(L('uCamUp'), cam.up)
    gl.uniform2f(L('uTanHalf'), cam.tanH, cam.tanV)

    // Three-point rig. Fill sits opposite the key; the rim strip sits behind, opposite the key.
    // Radiances are set from solid angle so the key alone lights the 18% gray card to about middle gray
    // (the key subtends ~0.16 sr, so L ~ pi / 0.16), with the fill near a 1:6 ratio and a hot rim.
    const side = s.keyAz >= 0 ? 1 : -1
    const keyRGB = kelvinToACEScg(s.keyKelvin)
    const fillRGB = kelvinToACEScg(8000)
    const rigs = [
      { on: s.key, rect: rectLight(s.keyAz, s.keyEl, 6.5, 2.6, 2.6), rgb: keyRGB, power: 20 },
      { on: s.fill, rect: rectLight(-side * 0.95, 0.18, 7.5, 4.5, 3.5), rgb: fillRGB, power: 1.8 },
      { on: s.rim, rect: rectLight(-side * 2.45, 0.45, 6.0, 0.9, 3.4), rgb: keyRGB, power: 36 },
    ]
    rigs.forEach((r, i) => {
      const on = r.on && !s.furnace
      gl.uniform3fv(L(`uLightCorner[${i}]`), r.rect.corner)
      gl.uniform3fv(L(`uLightU[${i}]`), r.rect.U)
      gl.uniform3fv(L(`uLightV[${i}]`), r.rect.V)
      gl.uniform3fv(L(`uLightRadiance[${i}]`), on ? scale(r.rgb, r.power) : [0, 0, 0])
    })

    gl.uniform4uiv(L('uSobol'), SOBOL_UNIFORM)
    gl.uniform1i(L('uMaxBounces'), MAX_BOUNCES)
    gl.uniform1i(L('uNumLights'), rigs.length)
    gl.uniform3fv(L('uEnv'), s.furnace ? [1, 1, 1] : [0.0012, 0.0013, 0.0016])
    gl.uniform1i(L('uFurnace'), s.furnace ? 1 : 0)
    gl.uniform1f(L('uIndirectClamp'), s.furnace ? 0 : INDIRECT_CLAMP)
    gl.uniform1i(L('uMultiScatter'), s.multiscatter ? 1 : 0)
    gl.uniform1i(L('uPass'), PASS_ID[s.pass])
    gl.uniform3fv(L('uBallX'), BALL_X)

    // Materials: 0 gray card, 1 chromium, 2 hero, 3 floor.
    const hero = HERO_PRESETS[s.hero]
    const heroMat: OpenPBR = { ...hero.params }
    if (s.heroRoughness !== null) heroMat[hero.roughnessParam] = s.heroRoughness
    const mats: OpenPBR[] = [SCENE_MATERIALS.gray, SCENE_MATERIALS.chrome, heroMat, SCENE_MATERIALS.floor]
    mats.forEach((m, i) => {
      for (const f of MATERIAL_FIELDS) {
        const v = m[f]
        const loc = L(`uMat[${i}].${f}`)
        if (Array.isArray(v)) gl.uniform3fv(loc, v)
        else gl.uniform1f(loc, v)
      }
    })
    this.sceneDirty = false
  }

  private step = (t: number) => {
    this.rafId = 0
    if (this.disposed || !this.visible || this.spp >= this.target) return

    const dt = this.lastT ? t - this.lastT : 16
    this.lastT = t
    if (this.interacting) {
      this.sppPerFrame = 1
    } else if (dt < 22 && this.sppPerFrame < 8) {
      this.sppPerFrame++
    } else if (dt > 40 && this.sppPerFrame > 1) {
      this.sppPerFrame--
    }
    if (!this.interacting && this.accumFmt.type === this.gl.HALF_FLOAT) this.sppPerFrame = Math.max(this.sppPerFrame, 4)
    this.traceBatch(Math.min(this.sppPerFrame, this.target - this.spp))
    this.display()
    this.onStatus({ spp: this.spp, target: this.target, converged: this.spp >= this.target })
    if (this.spp < this.target) this.rafId = requestAnimationFrame(this.step)
  }

  // Development aid: accumulate synchronously, independent of animation frames (e.g. in a hidden tab).
  // Each batch is flushed as its own submission and the queue is drained periodically, so no single GPU
  // submission runs long enough to trip the OS watchdog (TDR) and lose the context.
  renderNow(total: number) {
    if (!this.live) return
    let batches = 0
    while (this.spp < total) {
      this.traceBatch(Math.min(8, total - this.spp))
      this.gl.flush()
      if (++batches % 8 === 0) this.gl.finish()
    }
    this.display()
    this.onStatus({ spp: this.spp, target: this.target, converged: this.spp >= this.target })
  }

  // One trace pass: n new samples per pixel blended into the running mean (ping-pong accumulation).
  private traceBatch(n: number) {
    const gl = this.gl
    const src = this.ping
    const dst = 1 - this.ping
    gl.bindVertexArray(this.vao)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFbo[dst])
    gl.viewport(0, 0, this.width, this.height)
    gl.useProgram(this.progTrace)
    if (this.sceneDirty) this.uploadScene()
    const L = (name: string) => this.loc(this.progTrace, name)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.accumTex[src])
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_3D, this.eTable)
    gl.uniform1i(L('uPrev'), 0)
    gl.uniform1i(L('uETable'), 1)
    gl.uniform2f(L('uResolution'), this.width, this.height)
    gl.uniform1i(L('uSppDone'), this.spp)
    gl.uniform1i(L('uSppNew'), n)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
    this.ping = dst
    this.spp += n
  }

  private display() {
    if (!this.live) return
    const gl = this.gl
    const s = this.state
    const p = this.progDisplay
    gl.bindVertexArray(this.vao)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, this.width, this.height)
    gl.useProgram(p)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.accumTex[this.ping])
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_3D, this.acesLut)
    gl.uniform1i(this.loc(p, 'uAccum'), 0)
    gl.uniform1i(this.loc(p, 'uAcesLut'), 1)
    gl.uniform1i(this.loc(p, 'uAcesReady'), this.acesReady ? 1 : 0)
    gl.uniform1f(this.loc(p, 'uExposure'), s.furnace ? 0 : s.exposure)
    gl.uniform1i(this.loc(p, 'uView'), s.furnace ? VIEW_ID.standard : VIEW_ID[s.view])
    gl.uniform1i(this.loc(p, 'uPass'), PASS_ID[s.pass])
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }

  // Encoded image of the current frame (used to produce the static fallback poster).
  snapshot(type = 'image/png', quality?: number): Promise<Blob | null> {
    this.display()
    return new Promise(resolve => this.canvas.toBlob(resolve, type, quality))
  }

  // Raw accumulation buffer: linear ACEScg radiance, alpha = accumulated sphere coverage. Rows start at the bottom.
  readAccum(): { w: number; h: number; px: Float32Array } | null {
    if (this.accumFmt.type !== this.gl.FLOAT) return null
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFbo[this.ping])
    const px = new Float32Array(this.width * this.height * 4)
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, px)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    return { w: this.width, h: this.height, px }
  }

  // Mean linear radiance over interior sphere pixels. Used by the white furnace self-test.
  readSphereMean(): [number, number, number] | null {
    const buf = this.readAccum()
    if (!buf) return null
    const px = buf.px
    let r = 0, g = 0, b = 0, w = 0
    for (let i = 0; i < px.length; i += 4) {
      if (px[i + 3] < 0.999) continue
      r += px[i]
      g += px[i + 1]
      b += px[i + 2]
      w += 1
    }
    return w ? [r / w, g / w, b / w] : null
  }

  dispose() {
    this.disposed = true
    this.stop()
    this.canvas.removeEventListener('webglcontextlost', this.handleLost)
    const gl = this.gl
    this.accumTex.forEach(t => gl.deleteTexture(t))
    this.accumFbo.forEach(f => gl.deleteFramebuffer(f))
    gl.deleteTexture(this.eTable)
    gl.deleteTexture(this.acesLut)
    gl.deleteProgram(this.progTrace)
    gl.deleteProgram(this.progDisplay)
    gl.deleteProgram(this.progTable)
    gl.deleteVertexArray(this.vao)
  }
}
