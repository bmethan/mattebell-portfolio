// The lab's scene, shared by its two renderers (WebGPU, and WebGL as the fallback): the state and its defaults,
// the tracer variants, the camera rig and lights, the materials as uploaded, and the light mixer. Pure functions
// of the state, so both renderers draw exactly the same scene.
import { kelvinToACEScg } from './color'
import { SCENE_MATERIALS, HERO_PRESETS, STAGES, BALLS_METERS_PER_UNIT, heroParams, subsurfaceScale, type OpenPBR, type Hero, type PaintFinish, type SkinTone, type Stage } from './materials'
import { MODELS, MODEL_BALLS, MESH_MATERIALS, type Model } from './models'
import { ENVIRONMENTS, type Env } from './environments'

export type Pass = 'beauty' | 'diffuse' | 'specular' | 'albedo' | 'normal'
export type KeyType = 'softbox' | 'blacklight'
// Split compare: two versions of the image either side of a draggable line (see LabState.split).
export type Compare = 'off' | 'denoise' | 'multiscatter' | 'view'
export type View = 'aces' | 'agx' | 'neutral' | 'standard'
export type { Env, Hero, Model, PaintFinish, SkinTone, Stage }

export interface LabState {
  keyAz: number
  keyEl: number
  key: boolean
  keyType: KeyType // the key's lamp: a white softbox, or a black light (ultraviolet)
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
  compare: Compare
  split: number // the compare line, as a fraction of the image's width
}

export const DEFAULT_STATE: LabState = {
  keyAz: 0.62,
  keyEl: 0.52,
  key: true,
  keyType: 'softbox',
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
  compare: 'off',
  split: 0.5,
}

// Settings the display applies to the accumulated images (each light has its own), so they never re-render.
export const DISPLAY_KEYS = new Set<string>([
  'exposure', 'view', 'key', 'fill', 'rim', 'keyKelvin', 'fillKelvin', 'rimKelvin', 'keyGain', 'fillGain', 'rimGain',
  'denoise', 'envOn', 'envGain',
])
// The compare and its line act on the display too, except the multiple-scattering compare, which the tracer
// renders (each pixel by its side of the line).
export const tracedCompare = (s: LabState) => s.compare === 'multiscatter'
export const isDisplayOnly = (patch: Partial<LabState>, prev: LabState) => {
  const next = { ...prev, ...patch }
  return Object.keys(patch).every(
    k => DISPLAY_KEYS.has(k) || ((k === 'compare' || k === 'split') && !tracedCompare(prev) && !tracedCompare(next)),
  )
}

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
    if (k === 'split') return s.compare === 'off' || Math.abs(s.split - d.split) < 1e-6
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


export interface LabStatus {
  spp: number
  target: number
  converged: boolean
  preview: boolean // a proxy-resolution preview is on screen while the scene changes
  ms: number // render time of the current image: time spent rendering since the last change, pauses excluded
  model: 'ready' | 'loading' | 'failed' // the picked model's file and the shader variant the scene needs
  waitingFor?: 'model' | 'environment' | 'shaders' // while loading: the model's file, the environment's, or only the variant's compile
  compiling?: string // while waiting for shaders: what they render, for the status line
  finishing?: string // the same, while its first draw (which may pause the page) is about to run
}

// The tracer variant a scene needs: meshes for a model; full glass for a solid transmissive hero; the cheaper thin
// glass for thin-walled transmission (a model's windows). On the balls any glass hero takes the full variant,
// which is compiled ahead while the lab is in view.
export type VariantKey =
  | 'base' | 'base+glass' | 'base+sss' | 'base+fluor' | 'base+fluor+glass'
  | 'mesh' | 'mesh+thin' | 'mesh+glass' | 'mesh+sss' | 'mesh+fluor' | 'mesh+fluor+glass'
export function variantOf(s: LabState): VariantKey {
  const hero = heroMaterial(s)
  // Fluorescence carries a fourth (ultraviolet) band along the path: its own variant, so nothing else pays for it.
  if (hero.lab_fluor_weight > 0) {
    // A fluorescent medium (uranium glass, tonic water) needs the solid glass as well.
    const medium = hero.transmission_weight > 0 && hero.geometry_thin_walled < 0.5
    return `${s.model === 'spheres' ? 'base' : 'mesh'}+fluor${medium ? '+glass' : ''}`
  }
  // Subsurface scattering is a medium under a refracting surface: its own solid variant, with the walk.
  if (hero.subsurface_weight > 0 && hero.geometry_thin_walled < 0.5) return s.model === 'spheres' ? 'base+sss' : 'mesh+sss'
  const heroGlass = hero.transmission_weight > 0
  const heroSolid = heroGlass && hero.geometry_thin_walled < 0.5
  if (s.model === 'spheres') return heroGlass ? 'base+glass' : 'base'
  if (heroSolid) return 'mesh+glass'
  return heroGlass || MODELS[s.model].thinGlass ? 'mesh+thin' : 'mesh'
}
// What each variant adds, for the status line while it compiles.
export const VARIANT_LABEL: Record<VariantKey, string> = {
  base: 'the renderer',
  'base+glass': 'the glass shader',
  'base+sss': 'the subsurface shader',
  'base+fluor': 'the fluorescence shader',
  'mesh+fluor': 'the model shader with fluorescence',
  'base+fluor+glass': 'the fluorescent glass shader',
  'mesh+fluor+glass': 'the model shader with fluorescent glass',
  mesh: 'the model shader',
  'mesh+thin': 'the model shader with glass',
  'mesh+glass': 'the model shader with solid glass',
  'mesh+sss': 'the model shader with subsurface',
}
export const VARIANT_GLASS: Record<VariantKey, 'none' | 'thin' | 'full' | 'sss' | 'fluor' | 'fluorglass'> = {
  base: 'none',
  'base+glass': 'full',
  'base+sss': 'sss',
  'base+fluor': 'fluor',
  'mesh+fluor': 'fluor',
  'base+fluor+glass': 'fluorglass',
  'mesh+fluor+glass': 'fluorglass',
  mesh: 'none',
  'mesh+thin': 'thin',
  'mesh+glass': 'full',
  'mesh+sss': 'sss',
}


export type Vec3 = [number, number, number]
export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
export const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
export const length = (a: Vec3) => Math.sqrt(dot(a, a))
export const normalize = (a: Vec3): Vec3 => scale(a, 1 / (length(a) || 1))

export const PASS_ID: Record<Pass, number> = { beauty: 0, diffuse: 1, specular: 2, albedo: 3, normal: 4 }
export const VIEW_ID: Record<View, number> = { aces: 0, agx: 1, neutral: 2, standard: 3 }

export const TARGET_SPP = 1024
export const MESH_TARGET_SPP = 512
export const HALF_FLOAT_MAX_SPP = 256
export const MAX_BOUNCES = 8
// A model scene costs far more per bounce; past the fifth bounce its paths add little a turntable shows.
export const MESH_MAX_BOUNCES = 5
// The furnace test is a measurement, and light trapped in a model's cavities needs many bounces to escape: at 5
// the car reads 0.924, at 32 it reads 0.974 (the balls, with nothing to trap light, read 1.000 at 8).
export const MESH_FURNACE_BOUNCES = 32
// A solid glass hero on the balls. Very rough, high-IOR glass traps light longest (rough diamond at roughness 0.4:
// 0.949 at 32 bounces, 0.976 at 128); roulette ends most paths long before the cap.
export const GLASS_MAX_BOUNCES = 64
export const MESH_GLASS_MAX_BOUNCES = 16 // a solid glass hero on a model (each bounce costs a BVH traversal)
export const DENOISE_LEVELS = 5 // a-trous levels: taps 1, 2, 4, 8 and 16 pixels apart
// The display shows the denoised image alone up to DENOISE_FULL_SPP samples, then hands over to the raw render
// (log-linearly) by the target: the finished image is unfiltered. (Fully converged, the filter would still
// soften the antialiased edges of the brightest highlights, whose samples are all or nothing.)
export const DENOISE_FULL_SPP = 64
// Subsurface random walks: scattering events per path, beyond the bounces. Skin's red channel scatters some 70
// times before it is absorbed (albedo 0.986 for tone I), most walks far fewer before they leave.
export const MAX_SCATTER = 256
// In the furnace nothing is absorbed, so walks only end by leaving: a longer cap, though on Winged Victory, whose
// skin's blue mean free path is under a millimeter at the statue's size, it still trims the longest walks (blue
// reads about 0.97; 8192 reads 0.98 at six times the cost).
export const FURNACE_MAX_SCATTER = 2048
export const MAX_NODE_VISITS = 4096 // per mesh ray; a typical one visits well under a hundred BVH nodes

// Frame pacing. The GPU is shared with the browser's compositor, and a single draw cannot be interrupted, so
// every frame submits at most one slice sized to the frame's GPU budget, and at most MAX_IN_FLIGHT frames of
// work may be queued: that queue is what a scroll that starts mid-render waits behind. (One sample per pixel
// at 1329 x 556 costs 30-80 ms on an integrated AMD GPU, depending on its clocks; an eight-sample draw held
// the whole page to ~10 fps. Measured there, 14 ms x 4 queued converged fastest but dropped a frame or two
// when a scroll began; 7.5 ms x 2 never did but ran 2-3x slower.)
export const FRAME_BUDGET_MS = 10
export const MAX_IN_FLIGHT = 2
export const MAX_SAMPLES_PER_PASS = 8 // on fast GPUs, several samples per pass when they fit the budget twice over
// Enough to keep a four-sample half-float pass on a slow GPU near the budget, and a model scene (several times
// the balls' cost per sample) too: past height / SLICE_ROWS slices, the slices get thinner.
export const MAX_SLICES = 128
export const SLICE_ROWS = 8
export const SCROLL_QUIET_MS = 200 // tracing pauses while the page scrolls
// The first draw of a tracer program can block the page for seconds. On Windows, Chrome and Edge run WebGL on
// Direct3D through ANGLE, whose background compile (KHR_parallel_shader_compile) builds a program's pixel shader for
// its first output only; a program writing several render targets, as the tracer does (one image per light, and
// the denoiser's two), gets its real pixel shader compiled at its first draw, on the GPU process's main thread
// (measured: 6 s for the plain tracer, 10 to 25 s for the variants, against 3 ms for the same shader with one
// output). ANGLE caches the result for later visits. Until the WebGPU port, the lab announces that draw and holds it
// until the notice has painted and the page is not scrolling, so the pause never lands mid-scroll unexplained.
export const FIRST_DRAW_NOTICE_MS = 250
export const SETTLE_MS = 120 // full-resolution refinement starts once the scene has stopped changing for this long
export const PREVIEW_BUDGET_MS = 8
export const PREVIEW_MAX_DIV = 6
// Before the first timing of a model scene arrives, plan as if a sample costs this much (a slow integrated GPU),
// so the first passes cannot stall the page while the estimate is still unknown.
export const MESH_GUESS_MS = 400
export const PREVIEW_HOLD_SPP = 4 // the preview stays up until the full-resolution image is at least this clean
// The preview's path length cap: while something is being dragged, short paths keep the preview sharp (a smaller
// proxy divisor fits the frame budget). Solid glass needs a few more to read as glass at all.
export const PREVIEW_BOUNCES = 3
export const PREVIEW_GLASS_BOUNCES = 8

// Sobol' direction numbers for dimensions 1-3, verbatim from Burley 2020 ("Practical Hash-based Owen
// Scrambling", JCGT 9(4)) supplemental sobol.cpp. Packed one bit per uvec4: (dim1, dim2, dim3, 0).
export const SOBOL_D1 = [
  0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
  0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
  0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
  0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
]
export const SOBOL_D2 = [
  0x80000000, 0xc0000000, 0x60000000, 0x90000000, 0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
  0x68800000, 0x9cc00000, 0xee600000, 0x55900000, 0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
  0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000, 0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
  0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590, 0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,
]
export const SOBOL_D3 = [
  0x80000000, 0xc0000000, 0x20000000, 0x50000000, 0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
  0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000, 0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
  0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000, 0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
  0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050, 0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,
]
export const SOBOL_UNIFORM = new Uint32Array(SOBOL_D1.flatMap((d1, i) => [d1, SOBOL_D2[i], SOBOL_D3[i], 0]))
export const E_SIZE = 32 // mu and roughness resolution of the albedo tables
export const E_LAYERS = 16 // IOR resolution
export const LUT_SIZE = 65
export const LUT_URL = '/lookdev/aces2-sdr-rec709-65.png'
export const INDIRECT_CLAMP = 12

// Base radiance of key, fill and rim (unit luminance color): the key alone lights the 18% gray ball to about middle
// gray (it subtends ~0.16 sr, so L ~ pi / 0.16), the fill near a 1:6 ratio, a hot rim. The mixer scales these.
export const LIGHT_POWER = [20, 1.8, 36]
// The key as a black light (a 365 nm tube): its power as ultraviolet, with the faint deep violet such tubes leak
// (ACEScg, about 2.5% of the softbox's luminance). The mixer leaves its color alone (no color temperature).
export const BLACKLIGHT_UV = 1
export const BLACKLIGHT_VISIBLE: Vec3 = [0.05, 0, 0.2]

// Cyc: the floor ends 4 units behind the balls and sweeps up a 3-unit radius into a wall 7 units back, behind the
// rim light (its nearest corner sits about 4.7 back) so no softbox pokes through.
export const CYC_Z = -4
export const CYC_R = 3

// The camera sees an HDR environment softened, as a backdrop out of focus: this mip level of it (1k maps).
export const ENV_BACKDROP_BLUR = 2.5
// Without an HDR environment, the rig's surroundings: a faint constant sky (ACEScg), riding with the key's image.
export const RIG_AMBIENT: Vec3 = [0.0012, 0.0013, 0.0016]

// Scene: three unit spheres resting on a floor at y = 0.
export const BALL_X = [-2.4, 0, 2.4]
export const CENTROID: Vec3 = [0, 1, 0]

// Camera rig. The horizontal extent is fixed so the three balls always fit. A model is lower and wider than the
// balls, so its plate is framed tighter and aimed lower.
export const CAM_POS: Vec3 = [0, 1.45, 10.5]
export const CAM_TARGET: Vec3 = [0, 0.95, 0]
export const H_HALF_EXTENT = 4.0
export const MODEL_CAM_TARGET: Vec3 = [0, 0.62, 0]
export const MODEL_H_HALF_EXTENT = 3.55

// The color chart stands at the left end of the row, beside the gray ball, as on a lookdev plate (chart, gray
// ball, chrome ball): on the floor, leaning back, turned to face the camera. Its width is in gray ball radii; the
// frame widens and moves left to take it in, keeping its margins.
export const CHART_GAP = 0.18 // the tracer's CHART_GAP: between patches and around them, in patch widths
export const CHART_ASPECT = (4 + 5 * CHART_GAP) / (6 + 7 * CHART_GAP) // height over width
export const CHART_TILT = (15 * Math.PI) / 180

export function direction(az: number, el: number): Vec3 {
  return [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)]
}

// Rectangular softbox facing the scene centroid; emits toward +normal.
export function rectLight(az: number, el: number, dist: number, w: number, h: number) {
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

// ------------------------------------------------------------------------------------------------------------
// The scene as both renderers upload it.
// ------------------------------------------------------------------------------------------------------------

// The framing (camera position and aim, horizontal half extent at the aim) and the chart's card, if shown.
export function frameOf(s: LabState) {
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

export function cameraOf(s: LabState, width: number, height: number) {
  const { pos, target, half } = frameOf(s)
  const fwd = normalize(sub(target, pos))
  const right = normalize(cross(fwd, [0, 1, 0]))
  const up = cross(right, fwd)
  const tanH = half / length(sub(target, pos))
  const aspect = width && height ? width / height : 2.39
  const tanV = tanH / aspect
  return { pos, fwd, right, up, tanH, tanV }
}

// Normalized screen position (0..1, y down) of a world point, used for HTML labels.
export function projectOf(s: LabState, width: number, height: number, p: Vec3) {
  const { pos, fwd, right, up, tanH, tanV } = cameraOf(s, width, height)
  const d = sub(p, pos)
  const z = dot(d, fwd)
  const x = dot(d, right) / z / tanH
  const y = dot(d, up) / z / tanV
  return { x: (x + 1) / 2, y: 1 - (y + 1) / 2 }
}

// World points under each labelled object: the two reference balls, the hero (ball or model), then the chart.
// meshSize: the model's bounding box, once loaded.
export function labelPointsOf(s: LabState, meshSize?: [number, number, number]): Vec3[] {
  const chart = frameOf(s).chart
  let points: Vec3[]
  if (s.model === 'spheres') {
    points = BALL_X.map((x): Vec3 => [x, 0, 0])
  } else {
    const scaleW = MODELS[s.model].scale
    // The model's footprint, turned: label it at the point of the footprint nearest the camera.
    let front = 1.2
    if (meshSize) {
      const [sx, , sz] = meshSize
      const c = Math.cos(s.modelYaw), sn = Math.sin(s.modelYaw)
      front = Math.max(...[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([i, j]) => (-i * sx * sn + j * sz * c) * 0.5 * scaleW))
    }
    points = [[MODEL_BALLS[0][0], 0, MODEL_BALLS[0][2]], [MODEL_BALLS[1][0], 0, MODEL_BALLS[1][2]], [0, 0, front]]
  }
  return chart ? [...points, chart.foot] : points
}

// Three-point rig. Fill sits opposite the key; the rim strip sits behind, opposite the key. Radiances are set from
// solid angle so the key alone lights the 18% gray card to about middle gray (the key subtends ~0.16 sr, so
// L ~ pi / 0.16), with the fill near a 1:6 ratio and a hot rim. Light mixer: every light is traced in white at its
// base power, always on, into its own image; its switch, color temperature and intensity are applied in the display
// (mixWeightsOf), so changing them is instant. uv: ultraviolet radiance (only the key as a black light has any).
export function lightRigOf(s: LabState) {
  const side = s.keyAz >= 0 ? 1 : -1
  const rects = [
    rectLight(s.keyAz, s.keyEl, 6.5, 2.6, 2.6),
    rectLight(-side * 0.95, 0.18, 7.5, 4.5, 3.5),
    rectLight(-side * 2.45, 0.45, 6.0, 0.9, 3.4),
  ]
  return rects.map((rect, i) => {
    const power = LIGHT_POWER[i]
    const black = i === 0 && s.keyType === 'blacklight'
    const radiance: Vec3 = s.furnace ? [0, 0, 0] : black ? scale(BLACKLIGHT_VISIBLE, power) : [power, power, power]
    return { ...rect, radiance, uv: !s.furnace && black ? power * BLACKLIGHT_UV : 0 }
  })
}

// Light mixer: the weight of each light's image (key, fill, rim, environment). Light transport is linear in
// emission, so a light traced in white and scaled here by its color (unit luminance) and 2^gain is the same image
// as one traced in that color at that power. The rig's faint environment rides with the key's image; an HDR
// environment keeps its own colors and takes only its intensity here.
export function mixWeightsOf(s: LabState): Vec3[] {
  if (s.furnace) return [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
  const w = (on: boolean, kelvin: number, gain: number) => (on ? scale(kelvinToACEScg(kelvin), 2 ** gain) : [0, 0, 0]) as Vec3
  const e = s.envOn ? 2 ** s.envGain : 0
  const g = s.key ? 2 ** s.keyGain : 0
  const key: Vec3 = s.keyType === 'blacklight' ? [g, g, g] : w(s.key, s.keyKelvin, s.keyGain)
  return [key, w(s.fill, s.fillKelvin, s.fillGain), w(s.rim, s.rimKelvin, s.rimGain), [e, e, e]]
}

// Path length caps: solid glass traps light by total internal reflection, so its paths need many more bounces to
// get out (rough glass in the furnace: 0.91 at 8 bounces, 0.987 at 32); Russian roulette keeps the average path
// short. The preview's cap is shorter still (see PREVIEW_BOUNCES).
export function bouncesOf(s: LabState) {
  const solidGlass = variantOf(s).endsWith('+glass')
  const bounces =
    s.model === 'spheres'
      ? solidGlass ? GLASS_MAX_BOUNCES : MAX_BOUNCES
      : s.furnace ? MESH_FURNACE_BOUNCES : solidGlass ? MESH_GLASS_MAX_BOUNCES : MESH_MAX_BOUNCES
  const preview = s.furnace ? bounces : Math.min(bounces, solidGlass ? PREVIEW_GLASS_BOUNCES : PREVIEW_BOUNCES)
  return { bounces, preview }
}

// Where a scene's render stops: a model costs several times the balls per sample, so it finishes sooner.
export const goalOf = (s: LabState, target: number) => (s.model === 'spheres' ? target : Math.min(target, MESH_TARGET_SPP))

const smoothThinGlass = (m: OpenPBR) => m.geometry_thin_walled > 0.5 && m.transmission_weight > 0 && m.specular_roughness <= 0.01 && m.coat_weight <= 0

// Smooth thin-walled glass in the scene (the hero, or the car's windows): shadow rays then pass through it;
// otherwise they keep the cheaper any-hit test. thinSlot: the model's material slot that is such glass, or -1.
export function thinGlassOf(s: LabState) {
  const thinGlass = smoothThinGlass(heroMaterial(s)) || (s.model !== 'spheres' && !!MODELS[s.model].thinGlass && MESH_MATERIALS.some(smoothThinGlass))
  const thinSlot = s.model !== 'spheres' && MODELS[s.model].thinGlass && smoothThinGlass(MESH_MATERIALS[0]) ? 4 : -1
  return { thinGlass, thinSlot }
}

// The environment's turn, in turns: its own alignment (the brightest light at the key's azimuth) plus the lab's.
export const envTurnOf = (s: LabState) => (s.env === 'none' || s.furnace ? 0 : ENVIRONMENTS[s.env].turn) + s.envRot / (2 * Math.PI)

// The materials as the tracer takes them: 0 gray card, 1 chromium, 2 hero, 3 floor; with a model, 4-9 dress its
// other parts. Subsurface radii and transmission depths are given in centimeters (the presets'), and a fluorescent
// medium's absorption per centimeter: the tracer needs scene units.
export function sceneMaterialsOf(s: LabState): OpenPBR[] {
  const mats: OpenPBR[] = [SCENE_MATERIALS.gray, SCENE_MATERIALS.chrome, heroMaterial(s), STAGES[s.stage].material]
  if (s.model !== 'spheres') mats.push(...MESH_MATERIALS)
  const sss = subsurfaceScale(s.model === 'spheres' ? BALLS_METERS_PER_UNIT : MODELS[s.model].metersPerUnit)
  return mats.map(m => {
    const medium = m.transmission_weight > 0
    return {
      ...m,
      subsurface_radius: m.subsurface_radius * sss,
      transmission_depth: m.transmission_depth * sss,
      lab_fluor_absorb: medium ? (m.lab_fluor_absorb.map(x => x / sss) as [number, number, number]) : m.lab_fluor_absorb,
      lab_fluor_uv: medium ? m.lab_fluor_uv / sss : m.lab_fluor_uv,
    }
  })
}
