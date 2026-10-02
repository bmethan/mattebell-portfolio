import { VERT, E_TABLE_FRAG, DISPLAY_FRAG, DENOISE_PREP_FRAG, ATROUS_FRAG, traceFrag } from './shaders'
import { kelvinToACEScg } from './color'
import { SCENE_MATERIALS, HERO_PRESETS, MATERIAL_FIELDS, STAGES, BALLS_METERS_PER_UNIT, COLOR_CHECKER, heroParams, subsurfaceScale, type OpenPBR, type Hero, type PaintFinish, type SkinTone, type Stage } from './materials'
import { MODELS, MODEL_BALLS, MESH_MATERIALS, loadModel, type Model } from './models'
import { ENVIRONMENTS, loadEnvironment, whiteEnvironment, type Env, type EnvData } from './environments'

export type Pass = 'beauty' | 'diffuse' | 'specular' | 'albedo' | 'normal'
export type View = 'aces' | 'agx' | 'neutral' | 'standard'
export type { Env, Hero, Model, PaintFinish, SkinTone, Stage }

export interface LabState {
  keyAz: number
  keyEl: number
  key: boolean
  fill: boolean
  rim: boolean
  keyKelvin: number
  fillKelvin: number
  rimKelvin: number
  keyGain: number // light mixer intensities, in stops over each light's base power
  fillGain: number
  rimGain: number
  exposure: number
  model: Model
  modelYaw: number // turntable angle of a model, radians
  stage: Stage
  chart: boolean // the color chart beside the reference balls
  env: Env // an HDR environment lighting the scene, or none
  envOn: boolean // the environment's light, in the mixer
  envGain: number // its intensity in stops, in the mixer
  envRot: number // its turn about the vertical axis, radians
  hero: Hero
  paint: PaintFinish // car paint's finish
  flakes: boolean // car paint flakes (metallic, pearl and iridescent finishes)
  skinTone: SkinTone // which official skin example the skin hero is
  heroRoughness: number | null
  heroAniso: number | null // anisotropy override for brushed heroes
  pass: Pass
  view: View
  multiscatter: boolean
  denoise: boolean // display the denoised image (only while rendering: the still is the raw 2048 spp render)
  furnace: boolean
}

export const DEFAULT_STATE: LabState = {
  keyAz: 0.62,
  keyEl: 0.52,
  key: true,
  fill: true,
  rim: true,
  keyKelvin: 4300,
  fillKelvin: 8000,
  rimKelvin: 4300,
  keyGain: 0,
  fillGain: 0,
  rimGain: 0,
  exposure: 0.5,
  model: 'spheres',
  modelYaw: 0.6,
  stage: 'void',
  chart: true,
  env: 'none',
  envOn: true,
  envGain: 0,
  envRot: 0,
  hero: 'carpaint',
  paint: 'solid',
  flakes: false,
  skinTone: 'iii',
  heroRoughness: null,
  heroAniso: null,
  pass: 'beauty',
  view: 'aces',
  multiscatter: true,
  denoise: true,
  furnace: false,
}

// Settings the display applies to the accumulated images (each light has its own), so they never re-render.
const DISPLAY_KEYS = new Set<string>([
  'exposure', 'view', 'key', 'fill', 'rim', 'keyKelvin', 'fillKelvin', 'rimKelvin', 'keyGain', 'fillGain', 'rimGain',
  'denoise', 'envOn', 'envGain',
])
export const isDisplayOnly = (patch: Partial<LabState>) => Object.keys(patch).every(k => DISPLAY_KEYS.has(k))

// poster.jpg is this renderer's converged (2048 spp) image of DEFAULT_STATE, so that state never needs a live
// render. Angles and roughness compare with a tolerance (arrow keys step in floats; a roughness equal to the
// preset's own value is the same material). The turntable angle only matters with a model on the plate.
export function isPosterState(s: LabState) {
  const d = DEFAULT_STATE
  const base = (st: LabState) => heroParams(st.hero, st.paint, st.flakes, st.skinTone)
  const rough = (st: LabState) => st.heroRoughness ?? base(st)[HERO_PRESETS[st.hero].roughnessParam]
  const aniso = (st: LabState) => st.heroAniso ?? base(st).specular_roughness_anisotropy
  return (Object.keys(d) as (keyof LabState)[]).every(k => {
    if (k === 'keyAz' || k === 'keyEl') return Math.abs(s[k] - d[k]) < 1e-6
    if (k === 'modelYaw') return s.model === 'spheres' || Math.abs(s.modelYaw - d.modelYaw) < 1e-6
    if (k === 'heroRoughness') return s.hero === d.hero && Math.abs(rough(s) - rough(d)) < 1e-6
    if (k === 'heroAniso') return s.hero === d.hero && Math.abs(aniso(s) - aniso(d)) < 1e-6
    return s[k] === d[k]
  })
}

// The hero material as rendered: its preset (car paint by finish and flakes), then the slider edits.
export function heroMaterial(s: LabState): OpenPBR {
  const m: OpenPBR = { ...heroParams(s.hero, s.paint, s.flakes, s.skinTone) }
  if (s.heroRoughness !== null) m[HERO_PRESETS[s.hero].roughnessParam] = s.heroRoughness
  if (s.heroAniso !== null) m.specular_roughness_anisotropy = s.heroAniso
  return m
}

interface EnvGPU {
  map: WebGLTexture
  alias: WebGLTexture
  w: number
  h: number
}

export interface LabStatus {
  spp: number
  target: number
  converged: boolean
  preview: boolean // a proxy-resolution preview is on screen while the scene changes
  ms: number // render time of the current image: time spent rendering since the last change, pauses excluded
  model: 'ready' | 'loading' | 'failed' // the picked model's file and the shader variant the scene needs
  waitingFor?: 'model' | 'environment' | 'shaders' // while loading: the model's file, the environment's, or only the variant's compile
}

// The tracer variant a scene needs: meshes for a model; full glass for a solid transmissive hero; the cheaper thin
// glass for thin-walled transmission (a model's windows). On the balls any glass hero takes the full variant,
// which is compiled ahead while the lab is in view.
type VariantKey = 'base' | 'base+glass' | 'base+sss' | 'mesh' | 'mesh+thin' | 'mesh+glass' | 'mesh+sss'
function variantOf(s: LabState): VariantKey {
  const hero = heroMaterial(s)
  // Subsurface scattering is a medium under a refracting surface: its own solid variant, with the walk.
  if (hero.subsurface_weight > 0 && hero.geometry_thin_walled < 0.5) return s.model === 'spheres' ? 'base+sss' : 'mesh+sss'
  const heroGlass = hero.transmission_weight > 0
  const heroSolid = heroGlass && hero.geometry_thin_walled < 0.5
  if (s.model === 'spheres') return heroGlass ? 'base+glass' : 'base'
  if (heroSolid) return 'mesh+glass'
  return heroGlass || MODELS[s.model].thinGlass ? 'mesh+thin' : 'mesh'
}
const VARIANT_GLASS: Record<VariantKey, 'none' | 'thin' | 'full' | 'sss'> = {
  base: 'none',
  'base+glass': 'full',
  'base+sss': 'sss',
  mesh: 'none',
  'mesh+thin': 'thin',
  'mesh+glass': 'full',
  'mesh+sss': 'sss',
}

interface MeshGPU {
  bvh: WebGLTexture
  pos: WebGLTexture
  nrm: WebGLTexture
  size: [number, number, number]
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
const MESH_TARGET_SPP = 512
const HALF_FLOAT_MAX_SPP = 256
const MAX_BOUNCES = 8
// A model scene costs far more per bounce; past the fifth bounce its paths add little a turntable shows.
const MESH_MAX_BOUNCES = 5
// The furnace test is a measurement, and light trapped in a model's cavities needs many bounces to escape: at 5
// the car reads 0.924, at 32 it reads 0.974 (the balls, with nothing to trap light, read 1.000 at 8).
export const MESH_FURNACE_BOUNCES = 32
// A solid glass hero on the balls. Very rough, high-IOR glass traps light longest (rough diamond at roughness 0.4:
// 0.949 at 32 bounces, 0.976 at 128); roulette ends most paths long before the cap.
const GLASS_MAX_BOUNCES = 64
const MESH_GLASS_MAX_BOUNCES = 16 // a solid glass hero on a model (each bounce costs a BVH traversal)
const DENOISE_LEVELS = 5 // a-trous levels: taps 1, 2, 4, 8 and 16 pixels apart
// The display shows the denoised image alone up to DENOISE_FULL_SPP samples, then hands over to the raw render
// (log-linearly) by the target: the finished image is unfiltered. (Fully converged, the filter would still
// soften the antialiased edges of the brightest highlights, whose samples are all or nothing.)
const DENOISE_FULL_SPP = 64
// Subsurface random walks: scattering events per path, beyond the bounces. Skin's red channel scatters some 70
// times before it is absorbed (albedo 0.986 for tone I), most walks far fewer before they leave.
const MAX_SCATTER = 256
// In the furnace nothing is absorbed, so walks only end by leaving: a longer cap, though on Winged Victory, whose
// skin's blue mean free path is under a millimeter at the statue's size, it still trims the longest walks (blue
// reads about 0.97; 8192 reads 0.98 at six times the cost).
const FURNACE_MAX_SCATTER = 2048
const MAX_NODE_VISITS = 4096 // per mesh ray; a typical one visits well under a hundred BVH nodes

// Frame pacing. The GPU is shared with the browser's compositor, and a single draw cannot be interrupted, so
// every frame submits at most one slice sized to the frame's GPU budget, and at most MAX_IN_FLIGHT frames of
// work may be queued: that queue is what a scroll that starts mid-render waits behind. (One sample per pixel
// at 1329 x 556 costs 30-80 ms on an integrated AMD GPU, depending on its clocks; an eight-sample draw held
// the whole page to ~10 fps. Measured there, 14 ms x 4 queued converged fastest but dropped a frame or two
// when a scroll began; 7.5 ms x 2 never did but ran 2-3x slower.)
const FRAME_BUDGET_MS = 10
const MAX_IN_FLIGHT = 2
const MAX_SAMPLES_PER_PASS = 8 // on fast GPUs, several samples per pass when they fit the budget twice over
// Enough to keep a four-sample half-float pass on a slow GPU near the budget, and a model scene (several times
// the balls' cost per sample) too: past height / SLICE_ROWS slices, the slices get thinner.
const MAX_SLICES = 128
const SLICE_ROWS = 8
const SCROLL_QUIET_MS = 200 // tracing pauses while the page scrolls
const SETTLE_MS = 120 // full-resolution refinement starts once the scene has stopped changing for this long
const PREVIEW_BUDGET_MS = 8
const PREVIEW_MAX_DIV = 6
// Before the first timing of a model scene arrives, plan as if a sample costs this much (a slow integrated GPU),
// so the first passes cannot stall the page while the estimate is still unknown.
const MESH_GUESS_MS = 400
const PREVIEW_HOLD_SPP = 4 // the preview stays up until the full-resolution image is at least this clean
// The preview's path length cap: while something is being dragged, short paths keep the preview sharp (a smaller
// proxy divisor fits the frame budget). Solid glass needs a few more to read as glass at all.
const PREVIEW_BOUNCES = 3
const PREVIEW_GLASS_BOUNCES = 8

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

// Base radiance of key, fill and rim (unit luminance color): the key alone lights the 18% gray ball to about middle
// gray (it subtends ~0.16 sr, so L ~ pi / 0.16), the fill near a 1:6 ratio, a hot rim. The mixer scales these.
const LIGHT_POWER = [20, 1.8, 36]

// Cyc: the floor ends 4 units behind the balls and sweeps up a 3-unit radius into a wall 7 units back, behind the
// rim light (its nearest corner sits about 4.7 back) so no softbox pokes through.
const CYC_Z = -4
const CYC_R = 3

// The camera sees an HDR environment softened, as a backdrop out of focus: this mip level of it (1k maps).
const ENV_BACKDROP_BLUR = 2.5

// Scene: three unit spheres resting on a floor at y = 0.
export const BALL_X = [-2.4, 0, 2.4]
const CENTROID: Vec3 = [0, 1, 0]

// Camera rig. The horizontal extent is fixed so the three balls always fit. A model is lower and wider than the
// balls, so its plate is framed tighter and aimed lower.
const CAM_POS: Vec3 = [0, 1.45, 10.5]
const CAM_TARGET: Vec3 = [0, 0.95, 0]
const H_HALF_EXTENT = 4.0
const MODEL_CAM_TARGET: Vec3 = [0, 0.62, 0]
const MODEL_H_HALF_EXTENT = 3.55

// The color chart stands at the left end of the row, beside the gray ball, as on a lookdev plate (chart, gray
// ball, chrome ball): on the floor, leaning back, turned to face the camera. Its width is in gray ball radii; the
// frame widens and moves left to take it in, keeping its margins.
const CHART_GAP = 0.18 // the tracer's CHART_GAP: between patches and around them, in patch widths
const CHART_ASPECT = (4 + 5 * CHART_GAP) / (6 + 7 * CHART_GAP) // height over width
const CHART_TILT = (15 * Math.PI) / 180

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
  // The denoiser needs the tracer's two extra images (six render targets at once).
  readonly canDenoise: boolean
  private progPrep: WebGLProgram | null = null
  private progAtrous: WebGLProgram | null = null
  private dnTex: WebGLTexture[] = []
  private dnFbo: WebGLFramebuffer[] = []
  private dnGuide: WebGLTexture | null = null
  private dnPrepFbo: WebGLFramebuffer | null = null
  private progTable: WebGLProgram
  // Tracer variants beyond the plain one (progTrace), compiled the first time a scene needs them, and each
  // model's textures.
  private variants = new Map<VariantKey, { prog: WebGLProgram; ready: boolean; failed: boolean }>()
  private meshes = new Map<Model, MeshGPU>()
  private meshLoading = new Set<Model>()
  private meshLoadFailed = new Set<Model>()
  // HDR environments, each fetched once and kept on the GPU: its radiance (mipmapped) and its alias table.
  private envs = new Map<Env, EnvGPU>()
  private envLoading = new Set<Env>()
  private envLoadFailed = new Set<Env>()
  private envBlank: WebGLTexture | null = null // bound in their place while no environment is in use
  private envWhite: EnvGPU | null = null // the white furnace's surround
  private parallel = false
  private uploadedFor: WebGLProgram | null = null // the trace program the scene uniforms were last set on
  private live = false
  private uniformCache = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>()
  private accumTex: WebGLTexture[][] = [] // [ping][key, fill, rim]
  private accumFbo: WebGLFramebuffer[] = []
  private eTable: WebGLTexture
  private eTableT: WebGLTexture // dielectric albedo for glass compensation
  private acesLut: WebGLTexture
  private acesReady = false
  private accumFmt: { internal: number; type: number }
  private width = 0
  private height = 0
  private spp = 0
  private target = TARGET_SPP
  private ping = 0
  // The pass in progress: passN new samples per pixel, drawn as passSlices slices (0 = no pass in progress).
  private passN = 0
  private passSlices = 1
  private passSlice = 0
  private passOverran = false
  private passAdaptive = false
  private cleanPasses = 0
  // GPU cost of one full-frame sample, from timer queries when available, per pass and furnace setting (the
  // albedo and normal passes stop at the first hit and cost a small fraction of a beauty sample).
  private msPerSpp = new Map<string, number>()
  // The same for the preview, scaled up by k^2 (a small render keeps the GPU less busy per pixel, so it is
  // tracked apart and never used to size full-resolution slices).
  private previewMsPerSpp = new Map<string, number>()
  private slicesGuess = 4 // without timer queries: adjusted from how often the GPU is still busy
  private timer: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number } | null = null
  // frac: the share of a full frame the timed draw covered.
  private queries: { q: WebGLQuery; n: number; frac: number; cost: string; preview: boolean; pass: number; slices: number }[] = []
  private passId = 0 // numbers trace passes, for timing a model scene pass by pass
  private passTiming = new Map<number, { ns: number; frac: number; peakMs: number }>()
  private meshSlices = new Map<string, number>() // slice count per model cost class, steered by slice GPU time
  private fences: WebGLSync[] = []
  private budgetCap = FRAME_BUDGET_MS
  private maxInFlight = MAX_IN_FLIGHT
  // Preview: while the scene is changing, one sample per pixel at a proxy resolution (1/previewK) is drawn
  // each frame, outside the accumulation, so dragging the light stays at the display rate.
  private previewTex: WebGLTexture[] = [] // key, fill, rim
  private previewFbo: WebGLFramebuffer | null = null
  private previewK = 0 // proxy divisor of the preview on screen; 0 = none
  private previewDirty = false
  private gen = 0 // counts scene changes; a preview only stands in for the state it was traced from
  private previewGen = -1
  private lastChange = -Infinity
  private lastScroll = -Infinity
  private lastFrameT = 0
  private renderMs = 0
  private frameTimes: number[] = []
  private frameMs = 1000 / 60
  private health = 1
  private rafId = 0
  private visible = false
  private active = false
  private interacting = false
  private bounces = MAX_BOUNCES // path length cap of the current scene
  private previewBounces = PREVIEW_BOUNCES // and of its preview
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
    this.timer = gl.getExtension('EXT_disjoint_timer_query_webgl2')
    const parallel = !!gl.getExtension('KHR_parallel_shader_compile')
    this.parallel = parallel
    this.canDenoise = gl.getParameter(gl.MAX_DRAW_BUFFERS) >= 6 && gl.getParameter(gl.MAX_COLOR_ATTACHMENTS) >= 6
    this.progTrace = startProgram(gl, traceFrag(false, 'none', this.canDenoise))
    if (this.canDenoise) {
      this.progPrep = startProgram(gl, DENOISE_PREP_FRAG)
      this.progAtrous = startProgram(gl, ATROUS_FRAG)
    }
    this.progDisplay = startProgram(gl, DISPLAY_FRAG)
    this.progTable = startProgram(gl, E_TABLE_FRAG)

    // Placeholder until the ACES 2.0 LUT arrives, so the sampler always has a complete texture.
    this.acesLut = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_3D, this.acesLut)
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]))
    this.eTable = this.createAlbedoTexture()
    this.eTableT = this.createAlbedoTexture()
    this.loadAcesLut()

    canvas.addEventListener('webglcontextlost', this.handleLost)
    // A wheel can arrive before the scroll it causes (or scroll nothing at the end of the page).
    window.addEventListener('scroll', this.handleScroll, { passive: true })
    window.addEventListener('wheel', this.handleScroll, { passive: true })
    this.whenReady = this.warmUp(parallel)
  }

  // ANGLE (Direct3D) may finish building a program for a particular render target at its first draw, which
  // would land on the visitor's first edit. Draw one pixel into each target now, while the page is idle.
  private prime() {
    if (!this.width || this.state.model !== 'spheres') return
    const gl = this.gl
    gl.enable(gl.SCISSOR_TEST)
    gl.scissor(0, 0, 1, 1)
    this.drawTrace(1, 1, 0) // into the idle accumulation buffer, which the next pass rewrites
    this.tracePreview(2, false) // untimed: a one-pixel draw says nothing about the cost of a frame
    this.previewK = 0
    this.previewDirty = true
    this.display() // one pixel of the canvas, under the still
    gl.disable(gl.SCISSOR_TEST)
    gl.flush()
  }

  private handleScroll = () => {
    this.lastScroll = performance.now()
  }

  private async warmUp(parallel: boolean) {
    const gl = this.gl
    const progs = [this.progTrace, this.progDisplay, this.progTable, this.progPrep, this.progAtrous].filter((p): p is WebGLProgram => !!p)
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
    this.prime()
    this.sceneDirty = true
    this.ensureModel()
    this.kick()
    return true
  }

  // ------------------------------------------------------------------------------------------------------------
  // Variants and models: a tracer variant compiles in the background the first time a scene needs it (polled,
  // like the main one), and each model's file is fetched once and kept on the GPU.
  // ------------------------------------------------------------------------------------------------------------
  private ensureModel() {
    if (this.disposed || !this.live) return
    this.ensureVariant(variantOf(this.state))
    const env = this.state.env
    if (env !== 'none' && !this.envs.has(env) && !this.envLoading.has(env) && !this.envLoadFailed.has(env)) this.loadEnv(env)
    const model = this.state.model
    if (model === 'spheres') return
    if (!this.meshes.has(model) && !this.meshLoading.has(model) && !this.meshLoadFailed.has(model)) this.loadMesh(model)
  }

  private async loadEnv(env: Exclude<Env, 'none'>) {
    this.envLoading.add(env)
    try {
      const d = await loadEnvironment(env)
      if (this.disposed) return
      this.envs.set(env, this.uploadEnv(d))
      this.sceneDirty = true
    } catch (err) {
      console.error(err)
      this.envLoadFailed.add(env)
    } finally {
      this.envLoading.delete(env)
    }
    this.kick()
    this.reportModel()
  }

  private uploadEnv(d: EnvData): EnvGPU {
    const gl = this.gl
    gl.activeTexture(gl.TEXTURE11) // its own unit, so no binding in use elsewhere changes
    const map = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, map)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, d.w, d.h, 0, gl.RGBA, gl.FLOAT, d.rgba)
    // Mipmaps for the backdrop need half floats to be renderable; without them it is sharp (no softening).
    for (let i = 0; i < 8 && gl.getError() !== gl.NO_ERROR; i++); // clear earlier errors
    gl.generateMipmap(gl.TEXTURE_2D)
    const mips = gl.getError() === gl.NO_ERROR
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mips ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const alias = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, alias)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, d.w, d.h, 0, gl.RGBA, gl.FLOAT, d.alias)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    return { map, alias, w: d.w, h: d.h }
  }

  // The environment the tracer uses: the white furnace's, or the picked one once loaded.
  private activeEnv(): EnvGPU | undefined {
    const s = this.state
    if (s.furnace) return (this.envWhite ??= this.uploadEnv(whiteEnvironment()))
    return s.env === 'none' ? undefined : this.envs.get(s.env)
  }

  private ensureVariant(key: VariantKey) {
    if (key === 'base' || this.variants.has(key) || this.disposed || !this.live) return
    const v = { prog: startProgram(this.gl, traceFrag(key.startsWith('mesh'), VARIANT_GLASS[key], this.canDenoise)), ready: false, failed: false }
    this.variants.set(key, v)
    this.compileVariant(v)
  }

  private async compileVariant(v: { prog: WebGLProgram; ready: boolean; failed: boolean }) {
    const gl = this.gl
    for (;;) {
      await sleep(this.parallel ? 50 : 0)
      if (this.disposed || gl.isContextLost()) return
      if (!this.parallel || gl.getProgramParameter(v.prog, COMPLETION_STATUS_KHR)) break
    }
    try {
      finishProgram(gl, v.prog)
      v.ready = true
    } catch (err) {
      console.error(err)
      v.failed = true
    }
    this.kick()
    this.reportModel()
  }

  private async loadMesh(model: Exclude<Model, 'spheres'>) {
    this.meshLoading.add(model)
    try {
      const m = await loadModel(model)
      if (this.disposed) return
      const gl = this.gl
      const tex = (w: number, h: number, internal: number, format: number, type: number, data: ArrayBufferView) => {
        const t = gl.createTexture()!
        gl.bindTexture(gl.TEXTURE_2D, t)
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
        gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        return t
      }
      this.meshes.set(model, {
        bvh: tex(m.bvh.w, m.bvh.h, gl.RGBA32UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT, m.bvh.data),
        pos: tex(m.pos.w, m.pos.h, gl.RGBA32F, gl.RGBA, gl.FLOAT, m.pos.data),
        nrm: tex(m.nrm.w, m.nrm.h, gl.RGBA32UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT, m.nrm.data),
        size: m.size,
      })
      this.sceneDirty = true
    } catch (err) {
      console.error(err)
      this.meshLoadFailed.add(model)
    } finally {
      this.meshLoading.delete(model)
    }
    this.kick()
    this.reportModel()
  }

  // What the scene still waits for: its model's file (which also covers a compile running alongside), or only
  // its tracer variant's compile.
  private waitingFor(): LabStatus['waitingFor'] {
    const model = this.state.model
    if (model !== 'spheres' && !this.meshes.has(model)) return 'model'
    if (this.state.env !== 'none' && !this.envs.has(this.state.env)) return 'environment'
    const key = variantOf(this.state)
    return key === 'base' || this.variants.get(key)?.ready ? undefined : 'shaders'
  }

  private modelStatus(): LabStatus['model'] {
    const model = this.state.model
    const v = this.variants.get(variantOf(this.state))
    if (v?.failed || (model !== 'spheres' && this.meshLoadFailed.has(model)) || this.envLoadFailed.has(this.state.env)) return 'failed'
    return this.waitingFor() ? 'loading' : 'ready'
  }

  // While a model loads or a variant compiles, say so (the frame keeps the previous image until the first preview
  // lands), and say when it is ready even if no frame has run since (the lab may be scrolled away).
  private reportModel() {
    if (!this.active) return
    const model = this.modelStatus()
    if (model !== 'ready') {
      this.onStatus({ spp: 0, target: this.goal, converged: false, preview: false, ms: 0, model, waitingFor: this.waitingFor() })
      return
    }
    const converged = this.spp >= this.goal
    this.onStatus({ spp: this.spp, target: this.goal, converged, preview: this.showingPreview(), ms: this.renderMs, model })
  }

  private traceProg() {
    const key = variantOf(this.state)
    return key === 'base' ? this.progTrace : this.variants.get(key)!.prog
  }

  // World points under each labelled object: the two reference balls, the hero (ball or model), then the chart.
  labelPoints(): Vec3[] {
    const chart = this.frame().chart
    return chart ? [...this.objectPoints(), chart.foot] : this.objectPoints()
  }

  private objectPoints(): Vec3[] {
    if (this.state.model === 'spheres') return BALL_X.map((x): Vec3 => [x, 0, 0])
    const mesh = this.meshes.get(this.state.model)
    const scaleW = MODELS[this.state.model].scale
    // The model's footprint, turned: label it at the point of the footprint nearest the camera.
    let front = 1.2
    if (mesh) {
      const [sx, , sz] = mesh.size
      const c = Math.cos(this.state.modelYaw), s = Math.sin(this.state.modelYaw)
      front = Math.max(...[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([i, j]) => (-i * sx * s + j * sz * c) * 0.5 * scaleW))
    }
    return [[MODEL_BALLS[0][0], 0, MODEL_BALLS[0][2]], [MODEL_BALLS[1][0], 0, MODEL_BALLS[1][2]], [0, 0, front]]
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
    const uMode = gl.getUniformLocation(prog, 'uMode')
    // Mode 0: the reflection tables; mode 1: the dielectric (reflection + refraction) albedo glass is compensated by.
    ;[tex, this.eTableT].forEach((target, mode) => {
      gl.uniform1i(uMode, mode)
      for (let layer = 0; layer < E_LAYERS; layer++) {
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, target, 0, layer)
        gl.uniform1f(uLayer, layer / (E_LAYERS - 1))
        gl.drawArrays(gl.TRIANGLES, 0, 3)
      }
    })
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

  // A render target of four images (the light mixer's key, fill, rim and environment), drawn together with draw
  // buffers, plus the denoiser's two (albedo, normal; see AUX in the shader) where the GPU can draw six.
  private makeTargets(w: number, h: number) {
    const gl = this.gl
    const tex = Array.from({ length: this.canDenoise ? 6 : 4 }, () => this.makeTex(w, h, this.accumFmt.internal, this.accumFmt.type))
    const fbo = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    tex.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0))
    gl.drawBuffers(tex.map((_, i) => gl.COLOR_ATTACHMENT0 + i))
    return { tex, fbo }
  }

  // Cap on the internal resolution, so the path tracer stays interactive on large or dense displays (and the light
  // mixer's three float images per buffer stay modest in GPU memory). The poster renders raise it.
  maxPixels = 1_200_000

  resize(cssW: number, cssH: number, dpr: number) {
    const maxPixels = this.maxPixels
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
    this.accumTex.flat().forEach(t => gl.deleteTexture(t))
    this.accumFbo.forEach(f => gl.deleteFramebuffer(f))
    this.accumTex = []
    this.accumFbo = []
    for (let i = 0; i < 2; i++) {
      const { tex, fbo } = this.makeTargets(w, h)
      this.accumTex.push(tex)
      this.accumFbo.push(fbo)
    }
    // Preview target: big enough for the finest proxy resolution (half); coarser ones use part of it.
    this.previewTex.forEach(t => gl.deleteTexture(t))
    if (this.previewFbo) gl.deleteFramebuffer(this.previewFbo)
    const preview = this.makeTargets(Math.ceil(w / 2), Math.ceil(h / 2))
    this.previewTex = preview.tex
    this.previewFbo = preview.fbo
    this.previewK = 0
    // The denoiser's ping-pong pair and guide, full size (a preview uses part of them), in half float: it only
    // feeds the display.
    this.dnTex.forEach(t => gl.deleteTexture(t))
    this.dnFbo.forEach(f => gl.deleteFramebuffer(f))
    if (this.dnGuide) gl.deleteTexture(this.dnGuide)
    if (this.dnPrepFbo) gl.deleteFramebuffer(this.dnPrepFbo)
    this.dnTex = []
    this.dnFbo = []
    if (this.canDenoise) {
      const half = () => this.makeTex(w, h, gl.RGBA16F, gl.HALF_FLOAT)
      for (let i = 0; i < 2; i++) {
        const t = half()
        const f = gl.createFramebuffer()!
        gl.bindFramebuffer(gl.FRAMEBUFFER, f)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0)
        this.dnTex.push(t)
        this.dnFbo.push(f)
      }
      this.dnGuide = half()
      this.dnPrepFbo = gl.createFramebuffer()!
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.dnPrepFbo)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.dnTex[0], 0)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.dnGuide, 0)
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    // Resizing the canvas clears it, so treat it like a scene change: a quick preview fills the frame at once
    // (and keeps up during a continuous window resize) while the new full-resolution passes are sliced.
    this.sceneDirty = true
    this.previewDirty = true
    this.gen++
    this.lastChange = performance.now()
    this.reset()
  }

  setState(patch: Partial<LabState>) {
    const displayOnly = isDisplayOnly(patch)
    this.state = { ...this.state, ...patch }
    if (displayOnly && (this.spp > 0 || (this.previewK > 0 && this.previewGen === this.gen))) {
      // Exposure, view transform and the light mixer act on the accumulated radiance, so no re-render is needed.
      this.display()
      return
    }
    this.sceneDirty = true
    this.previewDirty = true
    this.gen++
    this.lastChange = performance.now()
    // Any change may need another variant (a glass hero) or a model.
    this.ensureModel()
    this.reset()
    if (this.modelStatus() !== 'ready' || patch.model || patch.env) this.reportModel()
  }

  setInteracting(on: boolean) {
    this.interacting = on
  }

  setVisible(v: boolean) {
    this.visible = v
    // Someone is looking at the lab: compile the glass variant in the background (off the page's thread), so a
    // click on a glass hero usually finds it ready.
    if (v) this.ensureVariant('base+glass')
    if (v) this.kick()
    else this.stop()
  }

  // The component turns rendering on only while the settings differ from the still's (see isPosterState), so
  // visitors who only look, hover or scroll past cost no GPU time at all.
  setActive(on: boolean) {
    this.active = on
    if (!on) {
      this.stop()
      return
    }
    // An exposure or view edit keeps the accumulated image, which may already be converged, so no pass would
    // run to report it. Report the image already on the canvas so the still uncovers it.
    if (this.live && (this.spp > 0 || (this.previewK > 0 && this.previewGen === this.gen))) {
      const converged = this.spp >= this.goal
      this.onStatus({ spp: this.spp, target: this.goal, converged, preview: this.showingPreview(), ms: this.renderMs, model: 'ready' })
    }
    this.reportModel()
    this.kick()
  }

  // Sample count at which the progressive render stops. Raising it continues the current accumulation.
  setTarget(n: number) {
    const cap = this.accumFmt.type === this.gl.HALF_FLOAT ? HALF_FLOAT_MAX_SPP : Infinity
    this.target = Math.max(1, Math.min(cap, Math.floor(n)))
    this.kick()
  }

  // Where this scene's render stops: a model costs several times the balls per sample, so it finishes sooner.
  private get goal() {
    return this.state.model === 'spheres' ? this.target : Math.min(this.target, MESH_TARGET_SPP)
  }

  // Normalized screen position (0..1, y down) of a world point, used for HTML labels.
  project(p: Vec3) {
    const { pos, fwd, right, up, tanH, tanV } = this.camera()
    const d = sub(p, pos)
    const z = dot(d, fwd)
    const x = dot(d, right) / z / tanH
    const y = dot(d, up) / z / tanV
    return { x: (x + 1) / 2, y: 1 - (y + 1) / 2 }
  }

  // The framing (camera position and aim, horizontal half extent at the aim) and the chart's card, if shown.
  private frame() {
    const s = this.state
    const plate = s.model !== 'spheres'
    const y = s.model === 'spheres' ? CAM_TARGET[1] : MODELS[s.model].camTargetY ?? MODEL_CAM_TARGET[1]
    const half0 = plate ? MODEL_H_HALF_EXTENT : H_HALF_EXTENT
    if (!s.chart) return { pos: CAM_POS, target: [0, y, 0] as Vec3, half: half0, chart: null }
    // The row's right end (the hero ball, or the chrome ball beside a model) and the gray ball at its left.
    const right = plate ? MODEL_BALLS[1][0] + MODEL_BALLS[1][3] : BALL_X[2] + 1
    const [gx, gz, gr] = plate ? [MODEL_BALLS[0][0], MODEL_BALLS[0][2], MODEL_BALLS[0][3]] : [BALL_X[0], 0, 1]
    const w = gr * (plate ? 2.2 : 1.6)
    const h = w * CHART_ASPECT
    const cx = gx - gr * 1.3 - w / 2
    const left = cx - w / 2
    const mid = (left + right) / 2
    const pos: Vec3 = [mid, CAM_POS[1], CAM_POS[2]]
    // The card's bottom edge stands on the floor; its face turns toward the camera and leans back.
    const cz = gz + gr * 0.6 // a little forward, into the key's light
    const yaw = Math.atan2(pos[0] - cx, pos[2] - cz)
    const U: Vec3 = [Math.cos(yaw) * w, 0, -Math.sin(yaw) * w]
    const V: Vec3 = scale([-Math.sin(CHART_TILT) * Math.sin(yaw), Math.cos(CHART_TILT), -Math.sin(CHART_TILT) * Math.cos(yaw)], h)
    const foot: Vec3 = [cx, 0.002, cz]
    return { pos, target: [mid, y, 0] as Vec3, half: (right - left) / 2 + (half0 - right), chart: { O: sub(foot, scale(U, 0.5)), U, V, foot } }
  }

  private camera() {
    const { pos, target, half } = this.frame()
    const fwd = normalize(sub(target, pos))
    const right = normalize(cross(fwd, [0, 1, 0]))
    const up = cross(right, fwd)
    const tanH = half / length(sub(target, pos))
    const aspect = this.width && this.height ? this.width / this.height : 2.39
    const tanV = tanH / aspect
    return { pos, fwd, right, up, tanH, tanV }
  }

  // Start over from zero samples. A pass in progress is abandoned: its finished slices sit in the destination
  // buffer, which is only read after all of a pass's slices are written, and the next pass rewrites every pixel.
  private reset() {
    this.spp = 0
    this.passN = 0
    this.renderMs = 0
    this.kick()
  }

  private kick() {
    if (this.disposed || !this.live || !this.active || !this.visible || this.rafId || !this.width) return
    this.lastFrameT = 0
    this.rafId = requestAnimationFrame(this.step)
  }

  private stop() {
    if (this.rafId) cancelAnimationFrame(this.rafId)
    this.rafId = 0
  }

  private uploadScene() {
    const gl = this.gl
    const s = this.state
    const prog = this.traceProg()
    const L = (n: string) => this.loc(prog, n)

    const cam = this.camera()
    gl.uniform3fv(L('uCamPos'), cam.pos)
    gl.uniform3fv(L('uCamFwd'), cam.fwd)
    gl.uniform3fv(L('uCamRight'), cam.right)
    gl.uniform3fv(L('uCamUp'), cam.up)
    gl.uniform2f(L('uTanHalf'), cam.tanH, cam.tanV)

    // Three-point rig. Fill sits opposite the key; the rim strip sits behind, opposite the key.
    // Radiances are set from solid angle so the key alone lights the 18% gray card to about middle gray
    // (the key subtends ~0.16 sr, so L ~ pi / 0.16), with the fill near a 1:6 ratio and a hot rim.
    // Light mixer: every light is traced in white at its base power, always on, into its own image; its switch,
    // color temperature and intensity are applied in the display (mixWeights), so changing them is instant.
    const side = s.keyAz >= 0 ? 1 : -1
    const rigs = [
      { rect: rectLight(s.keyAz, s.keyEl, 6.5, 2.6, 2.6), power: LIGHT_POWER[0] },
      { rect: rectLight(-side * 0.95, 0.18, 7.5, 4.5, 3.5), power: LIGHT_POWER[1] },
      { rect: rectLight(-side * 2.45, 0.45, 6.0, 0.9, 3.4), power: LIGHT_POWER[2] },
    ]
    rigs.forEach((r, i) => {
      gl.uniform3fv(L(`uLightCorner[${i}]`), r.rect.corner)
      gl.uniform3fv(L(`uLightU[${i}]`), r.rect.U)
      gl.uniform3fv(L(`uLightV[${i}]`), r.rect.V)
      gl.uniform3fv(L(`uLightRadiance[${i}]`), s.furnace ? [0, 0, 0] : [r.power, r.power, r.power])
    })

    gl.uniform4uiv(L('uSobol'), SOBOL_UNIFORM)
    // Solid glass traps light by total internal reflection, so its paths need many more bounces to get out (rough
    // glass in the furnace: 0.91 at 8 bounces, 0.987 at 32). Russian roulette keeps the average path short.
    const solidGlass = variantOf(s).endsWith('+glass')
    const bounces =
      s.model === 'spheres'
        ? solidGlass ? GLASS_MAX_BOUNCES : MAX_BOUNCES
        : s.furnace ? MESH_FURNACE_BOUNCES : solidGlass ? MESH_GLASS_MAX_BOUNCES : MESH_MAX_BOUNCES
    gl.uniform1i(L('uMaxBounces'), bounces)
    this.bounces = bounces
    this.previewBounces = s.furnace ? bounces : Math.min(bounces, solidGlass ? PREVIEW_GLASS_BOUNCES : PREVIEW_BOUNCES)
    gl.uniform1i(L('uNumLights'), rigs.length)
    // An HDR environment replaces the rig's faint constant one; the furnace test's white surround is one too.
    const env = this.activeEnv()
    gl.uniform3fv(L('uEnv'), env ? [0, 0, 0] : [0.0012, 0.0013, 0.0016])
    gl.uniform1i(L('uEnvOn'), env ? 1 : 0)
    gl.uniform2i(L('uEnvSize'), env?.w ?? 1, env?.h ?? 1)
    gl.uniform1f(L('uEnvRot'), (s.env === 'none' || s.furnace ? 0 : ENVIRONMENTS[s.env].turn) + s.envRot / (2 * Math.PI))
    gl.uniform1f(L('uEnvBlur'), ENV_BACKDROP_BLUR)
    gl.uniform1i(L('uEnvMap'), 11)
    gl.uniform1i(L('uEnvAlias'), 12)
    gl.uniform1i(L('uFurnace'), s.furnace ? 1 : 0)
    gl.uniform1f(L('uIndirectClamp'), s.furnace ? 0 : INDIRECT_CLAMP)
    gl.uniform1i(L('uMultiScatter'), s.multiscatter ? 1 : 0)
    gl.uniform1i(L('uPass'), PASS_ID[s.pass])

    // Materials: 0 gray card, 1 chromium, 2 hero, 3 floor; with a model, 4-9 dress its other parts.
    const stage = STAGES[s.stage]
    // Smooth thin-walled glass in the scene (the hero, or the car's windows): shadow rays then pass through it
    // (shadowT in the tracer); otherwise they keep the cheaper any-hit test.
    const hero = heroMaterial(s)
    const smoothThin = (m: OpenPBR) => m.geometry_thin_walled > 0.5 && m.transmission_weight > 0 && m.specular_roughness <= 0.01 && m.coat_weight <= 0
    const thinGlass = smoothThin(hero) || (s.model !== 'spheres' && MODELS[s.model].thinGlass && MESH_MATERIALS.some(smoothThin))
    gl.uniform1i(L('uThinGlass'), thinGlass ? 1 : 0)
    gl.uniform1i(L('uETableT'), 5)
    gl.uniform1i(L('uCyc'), stage.cyc ? 1 : 0)
    gl.uniform1f(L('uCycZ'), CYC_Z)
    gl.uniform1f(L('uCycR'), CYC_R)
    const chart = this.frame().chart
    gl.uniform1i(L('uChart'), chart ? 1 : 0)
    if (chart) {
      gl.uniform3fv(L('uChartO'), chart.O)
      gl.uniform3fv(L('uChartU'), chart.U)
      gl.uniform3fv(L('uChartV'), chart.V)
    }
    gl.uniform3fv(L('uChartColor'), COLOR_CHECKER.flat())
    const mats: OpenPBR[] = [SCENE_MATERIALS.gray, SCENE_MATERIALS.chrome, hero, stage.material]

    if (s.model === 'spheres') {
      gl.uniform3fv(L('uBallX'), BALL_X)
    } else {
      mats.push(...MESH_MATERIALS)
      MODEL_BALLS.forEach((b, i) => gl.uniform4fv(L(`uBall[${i}]`), b))
      const c = Math.cos(s.modelYaw), sn = Math.sin(s.modelYaw)
      // Turntable: rotation about y, column major.
      gl.uniformMatrix3fv(L('uModelRot'), false, [c, 0, -sn, 0, 1, 0, sn, 0, c])
      gl.uniform3fv(L('uModelPos'), [0, 0, 0])
      gl.uniform1f(L('uModelScale'), MODELS[s.model].scale)
      gl.uniform1i(L('uMaxNodeVisits'), MAX_NODE_VISITS)
      // Slot 4 is the glass on a model that has it (thinGlass); shadow rays pass through it.
      gl.uniform1i(L('uThinSlot'), MODELS[s.model].thinGlass && smoothThin(MESH_MATERIALS[0]) ? 4 : -1)
      gl.uniform1i(L('uBvh'), 2)
      gl.uniform1i(L('uTriPos'), 3)
      gl.uniform1i(L('uTriNrm'), 4)
    }
    // Subsurface radii and transmission depths are given in centimeters (the presets'); the tracer needs scene
    // units.
    const sss = subsurfaceScale(s.model === 'spheres' ? BALLS_METERS_PER_UNIT : MODELS[s.model].metersPerUnit)
    gl.uniform1i(L('uMaxScatter'), s.furnace ? FURNACE_MAX_SCATTER : MAX_SCATTER)
    mats.forEach((m, i) => {
      for (const f of MATERIAL_FIELDS) {
        const v = f === 'subsurface_radius' || f === 'transmission_depth' ? m[f] * sss : m[f]
        const loc = L(`uMat[${i}].${f}`)
        if (Array.isArray(v)) gl.uniform3fv(loc, v)
        else gl.uniform1f(loc, v)
      }
    })
    this.sceneDirty = false
    this.uploadedFor = prog
  }

  // Binds the trace program with its scene uniforms current, and the model's textures when it has one.
  private useTrace() {
    const gl = this.gl
    const prog = this.traceProg()
    gl.useProgram(prog)
    if (this.sceneDirty || this.uploadedFor !== prog) this.uploadScene()
    gl.activeTexture(gl.TEXTURE5)
    gl.bindTexture(gl.TEXTURE_3D, this.eTableT)
    const env = this.activeEnv()
    if (!this.envBlank) this.envBlank = this.makeTex(1, 1, gl.RGBA32F, gl.FLOAT)
    gl.activeTexture(gl.TEXTURE11)
    gl.bindTexture(gl.TEXTURE_2D, env?.map ?? this.envBlank)
    gl.activeTexture(gl.TEXTURE12)
    gl.bindTexture(gl.TEXTURE_2D, env?.alias ?? this.envBlank)
    const mesh = this.state.model === 'spheres' ? null : this.meshes.get(this.state.model)
    if (mesh) {
      gl.activeTexture(gl.TEXTURE2)
      gl.bindTexture(gl.TEXTURE_2D, mesh.bvh)
      gl.activeTexture(gl.TEXTURE3)
      gl.bindTexture(gl.TEXTURE_2D, mesh.pos)
      gl.activeTexture(gl.TEXTURE4)
      gl.bindTexture(gl.TEXTURE_2D, mesh.nrm)
    }
    return (name: string) => this.loc(prog, name)
  }

  private step = (t: number) => {
    this.rafId = 0
    if (this.disposed || !this.active || !this.visible || this.spp >= this.goal) return
    // A model still loading (or its renderer still compiling): nothing to trace yet. Loading kicks this again.
    if (this.modelStatus() !== 'ready') {
      this.ensureModel()
      return
    }
    this.rafId = requestAnimationFrame(this.step)
    const gl = this.gl
    const dt = this.lastFrameT ? t - this.lastFrameT : 0
    this.trackFrame(dt)
    if (dt > 0 && dt < 250) this.renderMs += dt // the first frame after a pause adds nothing
    this.lastFrameT = t
    this.collectTimings()

    // Bound the queued GPU work. (Sync status only changes between tasks, so this reads the state at the
    // start of this animation frame.)
    while (this.fences.length && gl.getSyncParameter(this.fences[0], gl.SYNC_STATUS) === gl.SIGNALED) {
      gl.deleteSync(this.fences.shift()!)
    }
    if (this.fences.length >= this.maxInFlight) {
      this.passOverran = true
      return
    }
    // Leave the GPU to the compositor while the page scrolls.
    const now = performance.now()
    if (!this.interacting && now - this.lastScroll < SCROLL_QUIET_MS) return

    // While the scene is changing, show a proxy-resolution preview of each new state; refine once it settles.
    const k = this.previewDiv()
    if (k > 1 && (this.interacting || now - this.lastChange < SETTLE_MS)) {
      if (!this.previewDirty) return
      this.tracePreview(k)
      this.submitted()
      this.display()
      this.onStatus({ spp: 0, target: this.goal, converged: false, preview: true, ms: this.renderMs, model: 'ready' })
      return
    }

    if (this.passN === 0) this.planPass()
    const passDone = this.traceSlice()
    this.submitted()
    if (passDone) {
      this.display()
      const converged = this.spp >= this.goal
      this.onStatus({ spp: this.spp, target: this.goal, converged, preview: this.showingPreview(), ms: this.renderMs, model: 'ready' })
      if (converged) this.stop()
    }
  }

  // Cost class of the current settings, for the GPU time estimates (a model costs far more than the balls).
  private costKey() {
    return `${this.state.model}:${this.state.pass}:${this.state.furnace ? 1 : 0}`
  }

  // The preview stands in until the full-resolution image of the same state has PREVIEW_HOLD_SPP samples, so
  // the handoff never makes the image noisier.
  private showingPreview() {
    return this.previewK > 0 && this.previewGen === this.gen && this.spp < Math.min(PREVIEW_HOLD_SPP, this.goal)
  }

  // The display interval (a low percentile of recent frames, so frames this renderer slowed down do not
  // inflate it), and a health factor that shrinks the budget while frames are being dropped.
  private trackFrame(dt: number) {
    if (dt <= 0 || dt > 250) return // first frame after a pause
    this.frameTimes.push(dt)
    if (this.frameTimes.length > 32) this.frameTimes.shift()
    const sorted = [...this.frameTimes].sort((a, b) => a - b)
    this.frameMs = Math.min(34, Math.max(6.9, sorted[Math.floor(sorted.length / 4)]))
    this.health = dt > 1.5 * this.frameMs ? Math.max(0.4, this.health * 0.8) : Math.min(1, this.health + 0.01)
  }

  // GPU time one frame's slice may take: most of a frame at 60 Hz, since tracing already stops while the page
  // scrolls and the integrated GPUs this has to run on only raise their clocks under a steady load.
  private budgetMs() {
    return Math.min(this.budgetCap, 0.85 * this.frameMs) * this.health
  }

  private submitted() {
    const gl = this.gl
    const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
    if (fence) this.fences.push(fence)
    gl.flush()
  }

  // Proxy divisor for the preview: the smallest one whose single-sample frame fits PREVIEW_BUDGET_MS.
  private previewDiv() {
    const key = this.costKey()
    // The preview's own measured cost (its paths are shorter, see PREVIEW_BOUNCES); before it is known, the
    // full-resolution one, an upper bound; unmeasured, assume a mid-range GPU.
    const ms = this.previewMsPerSpp.get(key) ?? this.msPerSpp.get(key) ?? (this.state.model === 'spheres' ? 40 : MESH_GUESS_MS)
    const budget = Math.min(PREVIEW_BUDGET_MS, 0.5 * this.frameMs)
    return Math.max(1, Math.min(PREVIEW_MAX_DIV, Math.ceil(Math.sqrt(ms / budget))))
  }

  // One sample per pixel of the current state at 1/k resolution, into the preview target. It never touches
  // the accumulation buffers, so it has no effect on the converged image.
  private tracePreview(k: number, timed = true) {
    const gl = this.gl
    const cost = this.costKey()
    const q = this.timer && timed ? gl.createQuery() : null
    if (q) gl.beginQuery(this.timer!.TIME_ELAPSED_EXT, q)
    gl.bindVertexArray(this.vao)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.previewFbo)
    gl.viewport(0, 0, Math.ceil(this.width / k), Math.ceil(this.height / k))
    const L = this.useTrace()
    this.bindPrev(L) // not read when uSppDone is 0
    // Fractional size, so preview pixel i covers exactly full-resolution pixels [k i, k i + k) on screen.
    gl.uniform2f(L('uResolution'), this.width / k, this.height / k)
    gl.uniform1i(L('uSppDone'), 0)
    gl.uniform1i(L('uSppNew'), 1)
    gl.uniform1i(L('uMaxBounces'), this.previewBounces)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
    gl.uniform1i(L('uMaxBounces'), this.bounces)
    if (q) {
      gl.endQuery(this.timer!.TIME_ELAPSED_EXT)
      this.queries.push({ q, n: 1, frac: 1 / (k * k), cost, preview: true, pass: -1, slices: 1 })
    }
    this.previewK = k
    this.previewGen = this.gen
    this.previewDirty = false
  }

  // Size the next pass: new samples per pixel, and how many slices (frames) to spread them over.
  private planPass() {
    const halfFloat = this.accumFmt.type === this.gl.HALF_FLOAT
    const ms = this.msPerSpp.get(this.costKey()) ?? (this.state.model === 'spheres' ? undefined : MESH_GUESS_MS)
    // Half-float running means need increments of at least four samples to stay precise (HALF_FLOAT_MAX_SPP).
    let n = halfFloat ? 4 : 1
    if (ms !== undefined && !this.interacting && 2 * ms * n <= this.budgetMs()) {
      n = Math.max(n, Math.min(MAX_SAMPLES_PER_PASS, Math.floor(this.budgetMs() / (2 * ms))))
    }
    n = Math.min(n, this.goal - this.spp)
    let slices = this.slicesGuess
    this.passAdaptive = false
    if (this.interacting) {
      slices = 1 // no preview on this GPU (it is fast enough): keep up with the pointer
    } else if (this.meshSlices.has(this.costKey())) {
      slices = this.meshSlices.get(this.costKey())! * n // steered by measured slice times (collectTimings)
    } else if (ms !== undefined) {
      slices = Math.ceil((ms * n) / this.budgetMs())
    } else {
      this.passAdaptive = true
    }
    this.passN = n
    this.passSlices = Math.max(1, Math.min(MAX_SLICES, this.height, slices))
    this.passSlice = 0
    this.passId++
  }

  // Draws the next slice of the current pass; returns true when that completes the pass.
  private traceSlice() {
    const gl = this.gl
    const q = this.timer ? gl.createQuery() : null
    if (q) gl.beginQuery(this.timer!.TIME_ELAPSED_EXT, q)
    const rows = this.drawTrace(this.passN, this.passSlices, this.passSlice)
    if (q) {
      gl.endQuery(this.timer!.TIME_ELAPSED_EXT)
      this.queries.push({ q, n: this.passN, frac: rows / this.height, cost: this.costKey(), preview: false, pass: this.passId, slices: this.passSlices })
    }
    if (++this.passSlice < this.passSlices) return false

    // Every pixel of the destination now holds the new running mean.
    this.ping = 1 - this.ping
    this.spp += this.passN
    this.passN = 0
    if (this.passAdaptive) {
      // No timer queries: widen the split when the queue was full at some frame since the last decision
      // (including frames between passes), and narrow it slowly after several passes that kept up.
      if (this.passOverran) {
        this.slicesGuess = Math.min(MAX_SLICES, this.slicesGuess + 1)
        this.cleanPasses = 0
      } else if (++this.cleanPasses >= 4 && this.slicesGuess > 1) {
        this.slicesGuess--
        this.cleanPasses = 0
      }
    }
    this.passOverran = false
    return true
  }

  // GPU time of finished draws, scaled to one full-frame sample by the share of the frame each one covered.
  private collectTimings() {
    const gl = this.gl
    const t = this.timer
    if (!t || !this.queries.length) return
    if (gl.getParameter(t.GPU_DISJOINT_EXT)) {
      // A GPU clock discontinuity since the last check leaves every query in flight meaningless.
      this.queries.forEach(({ q }) => gl.deleteQuery(q))
      this.queries = []
      return
    }
    while (this.queries.length) {
      const { q, n, frac, cost, preview, pass, slices } = this.queries[0]
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break
      const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number
      gl.deleteQuery(q)
      this.queries.shift()
      let ms = ns / 1e6 / frac / n
      if (!preview && !cost.startsWith('spheres:')) {
        // A model's cost sits in the rows it covers, so one slice says little about a whole frame, and a thin
        // slice costs more per pixel than a full frame (too few pixels to keep the GPU busy). So: average whole
        // passes for the cost estimate, and steer the slice count by the slowest slice's actual GPU time, so the
        // slices through the model also fit the frame budget (passes abandoned by a reset are dropped).
        const acc = this.passTiming.get(pass) ?? { ns: 0, frac: 0, peakMs: 0 }
        acc.ns += ns
        acc.frac += frac
        acc.peakMs = Math.max(acc.peakMs, ns / 1e6)
        for (const k of this.passTiming.keys()) if (k < pass - 2) this.passTiming.delete(k)
        if (acc.frac < 0.999) {
          this.passTiming.set(pass, acc)
          continue
        }
        this.passTiming.delete(pass)
        ms = acc.ns / 1e6 / acc.frac / n
        const ratio = acc.peakMs / this.budgetMs()
        if (ratio > 1.15 || ratio < 0.6) {
          const now = this.meshSlices.get(cost) ?? slices
          this.meshSlices.set(cost, Math.max(1, Math.min(MAX_SLICES, Math.round(now * Math.min(1.6, Math.max(0.7, ratio))))))
        }
      }
      const model = preview ? this.previewMsPerSpp : this.msPerSpp
      const prev = model.get(cost)
      model.set(cost, prev === undefined ? ms : 0.7 * prev + 0.3 * ms)
    }
  }

  // Development aid: accumulate synchronously, independent of animation frames (e.g. in a hidden tab).
  // Chrome's gl.finish() does not wait for the GPU, so a one-pixel read drains the queue after each batch,
  // keeping every submission far below the OS watchdog (TDR) limit.
  renderNow(total: number) {
    if (!this.live || this.modelStatus() !== 'ready') return
    const gl = this.gl
    const px = new Float32Array(4)
    this.passN = 0
    while (this.spp < total) {
      const n = Math.min(4, total - this.spp)
      this.drawTrace(n, 1, 0)
      this.ping = 1 - this.ping
      this.spp += n
      if (this.accumFmt.type === gl.FLOAT) gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px)
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    this.display()
    this.onStatus({ spp: this.spp, target: this.goal, converged: this.spp >= this.goal, preview: false, ms: this.renderMs, model: 'ready' })
  }

  // The current running means (key, fill, rim, environment; and the denoiser's two) on units 0, 6, 7, 8, 9 and
  // 10, and the albedo table on unit 1.
  private bindPrev(L: (name: string) => WebGLUniformLocation | null) {
    const gl = this.gl
    const units = [0, 6, 7, 8, 9, 10]
    this.accumTex[this.ping].forEach((t, i) => {
      gl.activeTexture(gl.TEXTURE0 + units[i])
      gl.bindTexture(gl.TEXTURE_2D, t)
      gl.uniform1i(L(`uPrev${i}`), units[i])
    })
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_3D, this.eTable)
    gl.uniform1i(L('uETable'), 1)
  }

  // One slice of a trace pass: n new samples per pixel blended into the running mean, read from the current
  // buffer and written to the other (ping-pong). A slice is every sliceCount-th block of SLICE_ROWS rows, each
  // drawn under a scissor so pixels outside it are never shaded (a shader discard would still run the whole
  // path on Direct3D). Interleaving spreads the balls' cost evenly over the slices. Returns the rows drawn.
  private drawTrace(n: number, sliceCount: number, slice: number) {
    const gl = this.gl
    gl.bindVertexArray(this.vao)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFbo[1 - this.ping])
    gl.viewport(0, 0, this.width, this.height)
    const L = this.useTrace()
    this.bindPrev(L)
    gl.uniform2f(L('uResolution'), this.width, this.height)
    gl.uniform1i(L('uSppDone'), this.spp)
    gl.uniform1i(L('uSppNew'), n)
    if (sliceCount === 1) {
      gl.drawArrays(gl.TRIANGLES, 0, 3)
      return this.height
    }
    // Blocks of up to SLICE_ROWS rows; thinner when there are more slices than SLICE_ROWS blocks fill, so every
    // slice still draws something. (Much thinner slices are slower overall: a few thousand pixels cannot keep
    // the GPU busy while each path waits on its texture fetches.)
    const blockRows = Math.max(1, Math.min(SLICE_ROWS, Math.floor(this.height / sliceCount)))
    let rows = 0
    gl.enable(gl.SCISSOR_TEST)
    for (let y = slice * blockRows; y < this.height; y += sliceCount * blockRows) {
      gl.scissor(0, y, this.width, blockRows)
      gl.drawArrays(gl.TRIANGLES, 0, 3)
      rows += Math.min(blockRows, this.height - y)
    }
    gl.disable(gl.SCISSOR_TEST)
    return rows
  }

  private display() {
    if (!this.live) return
    const gl = this.gl
    const s = this.state
    const p = this.progDisplay
    gl.bindVertexArray(this.vao)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, this.width, this.height)
    const preview = this.showingPreview()
    const src = preview ? this.previewTex : this.accumTex[this.ping]
    const k = preview ? this.previewK : 1
    const mix = this.denoiseMix(preview)
    const denoised = mix > 0 ? this.denoise(src, Math.ceil(this.width / k), Math.ceil(this.height / k), preview ? 1 : this.spp) : null
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.viewport(0, 0, this.width, this.height)
    gl.useProgram(p)
    const units = [0, 2, 3, 5]
    src.slice(0, 4).forEach((t, i) => {
      gl.activeTexture(gl.TEXTURE0 + units[i])
      gl.bindTexture(gl.TEXTURE_2D, t)
      gl.uniform1i(this.loc(p, `uAccum${i}`), units[i])
    })
    gl.activeTexture(gl.TEXTURE4)
    gl.bindTexture(gl.TEXTURE_2D, denoised)
    gl.uniform1i(this.loc(p, 'uDenoised'), 4)
    gl.uniform1f(this.loc(p, 'uDenoiseMix'), mix)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_3D, this.acesLut)
    gl.uniform1i(this.loc(p, 'uAccumDiv'), preview ? this.previewK : 1)
    gl.uniform3fv(this.loc(p, 'uMix'), this.mixWeights().flat())
    gl.uniform1i(this.loc(p, 'uAcesLut'), 1)
    gl.uniform1i(this.loc(p, 'uAcesReady'), this.acesReady ? 1 : 0)
    gl.uniform1f(this.loc(p, 'uExposure'), s.furnace ? 0 : s.exposure)
    gl.uniform1i(this.loc(p, 'uView'), s.furnace ? VIEW_ID.standard : VIEW_ID[s.view])
    gl.uniform1i(this.loc(p, 'uPass'), PASS_ID[s.pass])
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }

  // The denoiser's share of the displayed image (see DENOISE_FULL_SPP). It runs on the beauty and lobe passes,
  // never on the AOVs or the furnace test (a measurement).
  private denoiseMix(preview: boolean) {
    const s = this.state
    if (!this.canDenoise || !s.denoise || s.furnace || !(s.pass === 'beauty' || s.pass === 'diffuse' || s.pass === 'specular')) return 0
    if (preview || this.spp <= DENOISE_FULL_SPP) return 1
    return Math.max(0, 1 - Math.log2(this.spp / DENOISE_FULL_SPP) / Math.log2(Math.max(this.goal, 2 * DENOISE_FULL_SPP) / DENOISE_FULL_SPP))
  }

  // Filters the mixed image of src (w x h of it) for display; returns the texture holding the result. Costs a
  // few milliseconds of GPU time at full resolution, once per displayed update.
  private denoise(src: WebGLTexture[], w: number, h: number, spp: number) {
    const gl = this.gl
    const prep = this.progPrep!
    const atrous = this.progAtrous!
    const bind = (prog: WebGLProgram, name: string, unit: number, tex: WebGLTexture) => {
      gl.activeTexture(gl.TEXTURE0 + unit)
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.uniform1i(this.loc(prog, name), unit)
    }
    gl.bindVertexArray(this.vao)
    gl.viewport(0, 0, w, h)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.dnPrepFbo)
    gl.useProgram(prep)
    bind(prep, 'uAccum0', 0, src[0])
    bind(prep, 'uAccum1', 2, src[1])
    bind(prep, 'uAccum2', 3, src[2])
    bind(prep, 'uAccum3', 7, src[3])
    bind(prep, 'uAux0', 5, src[4])
    bind(prep, 'uAux1', 6, src[5])
    gl.uniform3fv(this.loc(prep, 'uMix'), this.mixWeights().flat())
    gl.uniform2i(this.loc(prep, 'uSize'), w, h)
    gl.uniform1f(this.loc(prep, 'uSpp'), spp)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
    gl.useProgram(atrous)
    bind(atrous, 'uAux0', 5, src[4])
    bind(atrous, 'uGuide', 6, this.dnGuide!)
    gl.uniform2i(this.loc(atrous, 'uSize'), w, h)
    for (let i = 0; i < DENOISE_LEVELS; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.dnFbo[(i + 1) % 2])
      bind(atrous, 'uIn', 0, this.dnTex[i % 2])
      gl.uniform1i(this.loc(atrous, 'uStep'), 1 << i)
      gl.uniform1i(this.loc(atrous, 'uLast'), i === DENOISE_LEVELS - 1 ? 1 : 0)
      gl.drawArrays(gl.TRIANGLES, 0, 3)
    }
    return this.dnTex[DENOISE_LEVELS % 2]
  }

  // Light mixer: the weight of each light's image (key, fill, rim, environment). Light transport is linear in
  // emission, so a light traced in white and scaled here by its color (unit luminance) and 2^gain is the same image
  // as one traced in that color at that power. The rig's faint environment rides with the key's image; an HDR
  // environment keeps its own colors and takes only its intensity here.
  private mixWeights(): [number, number, number][] {
    const s = this.state
    if (s.furnace) return [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    const w = (on: boolean, kelvin: number, gain: number) => (on ? scale(kelvinToACEScg(kelvin), 2 ** gain) : [0, 0, 0]) as Vec3
    const e = s.envOn ? 2 ** s.envGain : 0
    return [w(s.key, s.keyKelvin, s.keyGain), w(s.fill, s.fillKelvin, s.fillGain), w(s.rim, s.rimKelvin, s.rimGain), [e, e, e]]
  }

  // Encoded image of the current frame (used to produce the static fallback poster).
  snapshot(type = 'image/png', quality?: number): Promise<Blob | null> {
    this.display()
    return new Promise(resolve => this.canvas.toBlob(resolve, type, quality))
  }

  // The image as displayed before the view transform: linear ACEScg radiance, the lights mixed as the mixer
  // is set; alpha = accumulated sphere coverage. Rows start at the bottom.
  readAccum(): { w: number; h: number; px: Float32Array } | null {
    if (this.accumFmt.type !== this.gl.FLOAT) return null
    const gl = this.gl
    const n = this.width * this.height * 4
    const px = new Float32Array(n)
    const layer = new Float32Array(n)
    const mix = this.mixWeights()
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFbo[this.ping])
    for (let k = 0; k < 4; k++) {
      gl.readBuffer(gl.COLOR_ATTACHMENT0 + k)
      gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, layer)
      for (let i = 0; i < n; i += 4) {
        px[i] += layer[i] * mix[k][0]
        px[i + 1] += layer[i + 1] * mix[k][1]
        px[i + 2] += layer[i + 2] * mix[k][2]
        if (k === 0) px[i + 3] = layer[i + 3]
      }
    }
    gl.readBuffer(gl.COLOR_ATTACHMENT0)
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
    window.removeEventListener('scroll', this.handleScroll)
    window.removeEventListener('wheel', this.handleScroll)
    const gl = this.gl
    this.fences.forEach(f => gl.deleteSync(f))
    this.fences = []
    this.queries.forEach(({ q }) => gl.deleteQuery(q))
    this.queries = []
    this.accumTex.flat().forEach(t => gl.deleteTexture(t))
    this.accumFbo.forEach(f => gl.deleteFramebuffer(f))
    gl.deleteTexture(this.eTable)
    gl.deleteTexture(this.eTableT)
    gl.deleteTexture(this.acesLut)
    gl.deleteProgram(this.progTrace)
    this.variants.forEach(v => gl.deleteProgram(v.prog))
    this.variants.clear()
    this.meshes.forEach(m => [m.bvh, m.pos, m.nrm].forEach(t => gl.deleteTexture(t)))
    this.meshes.clear()
    this.envs.forEach(e => [e.map, e.alias].forEach(t => gl.deleteTexture(t)))
    this.envs.clear()
    if (this.envWhite) [this.envWhite.map, this.envWhite.alias].forEach(t => gl.deleteTexture(t))
    if (this.envBlank) gl.deleteTexture(this.envBlank)
    gl.deleteProgram(this.progDisplay)
    gl.deleteProgram(this.progTable)
    if (this.progPrep) gl.deleteProgram(this.progPrep)
    if (this.progAtrous) gl.deleteProgram(this.progAtrous)
    this.dnTex.forEach(t => gl.deleteTexture(t))
    this.dnFbo.forEach(f => gl.deleteFramebuffer(f))
    if (this.dnGuide) gl.deleteTexture(this.dnGuide)
    if (this.dnPrepFbo) gl.deleteFramebuffer(this.dnPrepFbo)
    this.previewTex.forEach(t => gl.deleteTexture(t))
    if (this.previewFbo) gl.deleteFramebuffer(this.previewFbo)
    gl.deleteVertexArray(this.vao)
  }
}
