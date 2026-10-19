// The pyro lab's engine: a WebGPU gas solver (see wgsl.ts) stepped at a fixed 30 steps per simulated second,
// and a volume renderer drawn after each step (or when a shading control changes, so a paused simulation can be
// relit). WebGPU only: browsers without it get the lab's still frame.
import { ADVECT_DETAIL, ADVECT_SCAL, ADVECT_VEL, BALL_HEAT, BIMOCQ_ACCUM, BIMOCQ_MAPS, BIMOCQ_RECON, LFM_CELLVEL, LFM_MARCH_BWD, LFM_MARCH_FWD, LFM_N, LFM_PULL, IVOCK_APPLY, IVOCK_CURL, IVOCK_SMOOTH, IVOCK_TARGET, VORT_BEFORE, PROJECT_REFLECT, RESOLVE, SKIN_N, SKIN_TEXELS, UPRES, UPRES_BLOCK, UPRES_COARSE, CURL_GEN, VEL_COMBINE, COMBINE, CONSERVE_RATIO, CONSERVE_SPREAD, CURL, DIFFUSE, DISPLAY, DIVERGENCE, CYC_BOUNCE, CYC_DIRECT, CYC_MAP, FLAME_NEAR, FLOOR_NEAR, ESCAPE_PHASES, FIRE_LIGHT_ENTRIES, FIRE_SUM, FIRE_SUM_TOTAL, FORCES, GLARE, GLARE_LEVELS, LIGHT, MG_PROLONG, MG_RESIDUAL, MG_RESTRICT, MG_SMOOTH, NOISE_GEN, PATH_TRACE, PROJECT, PT_COMPOSITE, RENDER, SOR, WOOD_CELLS, WOOD_LAYERS } from './wgsl'
import { planckTable, whiteBalance } from './blackbody'
import { BLUE_NOISE_SIZE, blueNoise } from './bluenoise'
import { FIRE_BANDS, FUEL, G, METER_AMBIENT, METER_FIRE_CD, METER_FIRE_EV, METER_FIRE_SLOPE, RHO_T, SOOT_CHROMA, SOOT_EXTINCTION, T_AMBIENT, exposureOfEV, fireTurbulence, luxOfEV } from './physics'
import { writeVdb, type VdbGrid } from './vdb'
import { fitWear, skinSolidAngles, skinTexel, woodLayers } from './wood'
import { DOMAIN_WIDTH, PYRO_DEFAULTS, RESOLUTIONS, RESOLUTION_ORDER, type PyroResolution, type PyroState } from './params'
import { REC709_TO_AP1, ap1Luminance, loadAcesLutGPU, mat3Apply, viewIndex } from '../lab/viewTransforms'
import { BACKDROP_H, BACKDROP_W, environmentLight, shBasis, type EnvLight } from './environment'
import { kelvinToACEScg } from '../lookdev/color'

export const SIM_DT = 1 / 30
// The wooden ball's wood steps once every this many steps of the gas (BALL_HEAT).
const WOOD_EVERY = 2
const MAX_STEPS_PER_FRAME = 2
// The fire's head start before its first frame (see preroll), simulated seconds, and the steps sent at a time.
const HEAD_START_S = 3
const PREROLL_BATCH = 3
const MAX_IN_FLIGHT = 2 // frames of GPU work queued at once: enough to keep the GPU busy, too few to stall the page
const SOR_OMEGA = 1.7
// Multigrid: Gauss-Seidel sweeps before and after each coarse correction, and on the coarsest level (over-relaxed
// there, where it is cheap); levels go down until a dimension would fall under 8 cells.
const MG_PRE = 2
const MG_POST = 2
const MG_COARSE = 16
// Red-black sweeps per step for each of its potentials (warm started).
const IVOCK_SWEEPS = 1
// Leapfrog flow maps' projection at each cycle's end, cold started: V-cycles.
const LFM_PROJECT_CYCLES = 4
// BiMocq2's steps between re-initialisations.
const BIMOCQ_REINIT = 16
const MG_COARSE_OMEGA = 1.6
const SPEED_LIMIT = 8 // m/s
const FOV_Y = (32 * Math.PI) / 180
// The close view (setView): a camera looking the lab camera's way, near enough that the ball's diameter takes this
// share of the frame's height, aimed this many ball radii above its centre (room for its flames); after a drag it aims
// again once the ball has gone this share of the way from the frame's middle to its edge; never lower than this (m).
const CLOSE_FILL = 0.5
const CLOSE_LIFT = 0.25
const CLOSE_REAIM = 1 / 3
const CLOSE_MIN_HEIGHT = 0.05
const RENDER_SCALE_MIN = 0.5
const RENDER_SCALE_START = 0.75
// The simulation falling behind real time while the page's frames still come (the GPU is the bottleneck, and the
// two-frames-in-flight gate keeps the page smooth by skipping steps): fewer steps than this a second, for a second of
// steady running, renders fewer pixels, so more of the GPU goes to the steps. Judged only after SIM_SETTLE_MS of
// running without a break (start-up, a restart, a change of scale), over the last second's steps, all at the current
// scale (so SIM_SETTLE_MS is at least a second). On the HP Envy's Radeon (RDNA 2) in a 1600-pixel-wide window, the
// bonfire's frame costs 21 ms at full scale and 10 at the lowest, its step 16: 24 steps a second at full scale, 29 at
// the lowest. (A smoke source's step there costs about 30 ms by itself, so a softer image gains it little.)
const SIM_BEHIND_RATE = 27
// (And raising the render scale past its start needs the simulation at real time: this many steps a second.)
const SIM_REAL_RATE = 29
const SIM_BEHIND_MS = 750
const SIM_SETTLE_MS = 1000
// A default resolution that cannot keep up steps down: judged on the steps taken between 3 and 6 s of running
// (after start-up compiles settle), against 18 per second.
const AUTO_RES_FROM_S = 3
const AUTO_RES_TO_S = 6
const AUTO_RES_MIN_RATE = 18
// The image is the display's own pixels (the canvas), up to DISPLAY_DPR_MAX per CSS pixel and MAX_DISPLAY_PIXELS in all,
// which the temporal upsampling (RESOLVE) builds up from the march's fewer. The march runs at the render scale of the
// frame's CSS pixels times their density (held to 1.5, as before), from RENDER_SCALE_START: down to RENDER_SCALE_MIN on
// a GPU that falls behind, and up to the display's own size on one with room. Paused, the march draws PAUSED_QUICK
// frames at the running size (so a control dragged while paused answers at the running frame rate), then settles at
// the display's size, held to PAUSED_MAX_PIXELS (beyond it, over proportionately more frames). On the HP Envy the march
// of the bonfire costs 20 ms at the half scale and 70 at the display's size; the passes after it, at the display's
// size, 1.9 ms (0.5 when the canvas was the march's size).
const DISPLAY_DPR_MAX = 2
const MAX_DISPLAY_PIXELS = 4_000_000
const PAUSED_QUICK = 3
const PAUSED_MAX_PIXELS = 2_200_000
// The upsampling's history: the samples a running pixel may count (enough to sharpen the still set; the moving flames'
// are rejected long before), and the sharpness of its gather of each frame's samples, 1 / (2 sigma^2) in display
// pixels: at half the display's size, where a display pixel finds a sample near it only every few frames; and at its
// full size, where each finds its own every frame and the gather need take in no neighbour's (as the settled stills
// always have).
const RUNNING_SAMPLES = 8
const GATHER_SHARPNESS = [3, 6]
// The share of the way the running history is relit toward the frame's light each frame (see RESOLVE): enough that
// neighbours do not hold different moments of the fire's flicker, little enough to keep the frame's noise out of it.
const RELIGHT = 0.5
const STATUS_MS = 250
const NO_WEAR = new Float32Array(12) // (the grey ball's: wornRadius's fit, all zero)
const STIR_RADIUS = 0.3 // m
const DETAIL_PERIOD = 1 // s of simulation between resets of each upres layer (half a period apart); the plume stretches them fast
const NOISE_SIZE = 64 // texels along each side of the tileable detail noise
// Spatiotemporal blue noise for the march's offsets: 64 x 64 x 16, made by void and cluster with Wolfe et al.'s energy
// (tools/stbn_gen.py) and served as a file.
const STBN_SLICES = 16
const STBN_URL = '/pyro/stbn-64x64x16.bin'
const TAA_SETTLE = 24 // frames a paused image accumulates (a running average) before it stops redrawing
// The path-traced reference (PATH_TRACE): paths per pixel it stops at, and scattering or surface events per path.
const PT_SAMPLES = 4096
const PT_BOUNCES = 32
// The sun's half-angle (its disc is 0.53 degrees across).
const SUN_HALF_ANGLE = (0.2665 * Math.PI) / 180
// The sun's and the sky's colours (authored in Rec.709, carried into ACEScg, the working space), each scaled to unit
// luminance (the EV controls set the illuminance).
const unitLum = (c709: Vec3): Vec3 => {
  const c = mat3Apply(REC709_TO_AP1, c709)
  return c.map(v => v / ap1Luminance(c)) as Vec3
}
const SUN_COLOUR = unitLum([1.0, 0.93, 0.84])
const SKY_COLOUR = unitLum([0.42, 0.52, 0.72])
// The Lookdev lab's three-point softbox rig (its scene.ts, lightRigOf), as directional lights: the key where the
// light azimuth and elevation put it, 4300 K; the fill on the far side from the key (0.95 rad round, 0.18 up), 8000 K;
// the rim strip behind (2.45 rad round, 0.45 up), 4300 K. Each at the Lookdev lab's ratio of its light to the key's,
// radiance times solid angle from the centre of that scene: the key 20 x 0.16 sr, the fill 1.8 x 0.28 sr (about 1:6),
// the rim 36 x 0.085 sr (about as strong as the key). The key's illuminance is the Key EV (the sun's).
const RIG = {
  key: { kelvin: 4300, omega: 0.16 },
  fill: { az: 0.95, el: 0.18, kelvin: 8000, ratio: (1.8 * 0.28) / (20 * 0.16), omega: 0.28 },
  rim: { az: 2.45, el: 0.45, kelvin: 4300, ratio: (36 * 0.085) / (20 * 0.16), omega: 0.085 },
}
// A cone of solid angle omega (sr): the cosine of its half-angle.
const coneCos = (omega: number) => 1 - omega / (2 * Math.PI)
// The on-set reference (referenceSet), in metres: positions along the floor are out from the bed's edge (never
// nearer the middle than `clear`), and the set takes `span` of room. Balls 16 cm across on stands 1.6 cm thick; the
// Lookdev lab's chart (its patches and gaps, CHART_GAP 0.18) 26 cm wide, leaning back 15 degrees.
const REF = {
  clear: 0.3,
  span: 0.75,
  margin: 0.05,
  ballR: 0.08,
  ballY: 0.45,
  ballZ: -0.12,
  greyAt: 0.38,
  chromeAt: 0.62,
  standR: 0.008,
  chartAt: 0.42,
  chartZ: 0.18,
  chartW: 0.26,
  chartAspect: (4 + 5 * 0.18) / (6 + 7 * 0.18),
  chartTilt: (15 * Math.PI) / 180,
  chartTurn: (25 * Math.PI) / 180,
}

// The fields at one moment, for export: per cell (x fastest, then y, then z), the extinction at 550 nm (1/m), the
// temperature (K), and for a fire the unburned fuel (mass fraction) and the heat release rate (W/m3); per cell,
// the velocity at its centre (m/s, three values). The cells are dx wide and cell (0, 0, 0) is centred at origin.
export interface PyroFields {
  n: Vec3
  dx: number
  origin: Vec3
  time: number
  fire: boolean
  density: Float32Array
  temperature: Float32Array
  fuel: Float32Array | null
  heat: Float32Array | null
  vel: Float32Array
}

export interface PyroStatus {
  ready: boolean
  running: boolean // stepping (not paused, on screen)
  simTime: number // s
  stepsPerSec: number
  renderScale: number
  dims: [number, number, number]
  resolution: PyroResolution // what the engine runs (it may step down from the default on a slow GPU)
  autoStepped: boolean // it did step down
  exposureEV: number // the camera's exposure now (metered when auto)
  ball: BallStatus | null // the wooden ball, when there is one
  pathTrace: { on: boolean; samples: number; compiling: boolean } // the path-traced reference (paused)
  view: PyroView // the camera: the lab's framing, or close on the ball (setView)
}

export type PyroView = 'whole' | 'close'

// The wooden ball, as read back a few times a second: its hottest surface (K), the heat its flames release (kW, from
// the volatiles it gives off fast enough to burn) and the share of its surface charred.
export interface BallStatus {
  surfaceK: number
  burningKW: number
  charred: number
}

type Vec3 = [number, number, number]
const halfToFloat = (h: number) => {
  const e = (h >> 10) & 31
  const f = h & 1023
  const v = e === 0 ? f * 2 ** -24 : e === 31 ? Infinity : (1 + f / 1024) * 2 ** (e - 15)
  return h & 0x8000 ? -v : v
}
// Floats to half floats (rounded to nearest, overflow to the largest half; the backdrop holds no negatives).
const toHalf = (src: Float32Array) => {
  const f = new Float32Array(1)
  const u = new Uint32Array(f.buffer)
  const out = new Uint16Array(src.length)
  for (let i = 0; i < src.length; i++) {
    f[0] = src[i]
    const x = u[0]
    const sign = (x >>> 16) & 0x8000
    const e = ((x >>> 23) & 0xff) - 127 + 15
    let m = x & 0x7fffff
    if (e >= 31) out[i] = sign | 0x7bff
    else if (e <= 0) {
      if (e < -10) out[i] = sign
      else {
        m |= 0x800000
        out[i] = sign | ((m >> (14 - e)) + ((m >> (13 - e)) & 1))
      }
    } else out[i] = sign | Math.min((e << 10) + (m >> 13) + ((m >> 12) & 1), 0x7bff)
  }
  return out
}
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}
type Camera = { pos: Vec3; fwd: Vec3; tanY: number; aspect: number }
// Whether a camera differs from another: its position, direction, field of view or aspect.
const camMoved = (a: Camera, b: Camera) => [...a.pos, ...a.fwd, a.tanY, a.aspect].some((v, i) => Math.abs(v - [...b.pos, ...b.fwd, b.tanY, b.aspect][i]) > 1e-6)

// The frame's targets and the bind groups of the passes over them (see ensureTargets, postGroups).
interface Targets {
  w: number
  h: number
  hdr: GPUTexture
  hist: [GPUTexture, GPUTexture]
  glare: GPUTexture
  glareViews: GPUTextureView[]
  resolve: [GPUBindGroup, GPUBindGroup]
  glareFirst: [GPUBindGroup, GPUBindGroup]
  glareDown: (GPUBindGroup | null)[]
  glareUp: (GPUBindGroup | null)[]
  display: [GPUBindGroup, GPUBindGroup]
}

interface Pipes {
  fireSum: GPUComputePipeline
  fireTotal: GPUComputePipeline
  noiseGen: GPUComputePipeline
  curlGen: GPUComputePipeline
  advDetail: GPUComputePipeline
  mgSmooth: GPUComputePipeline
  mgResidual: GPUComputePipeline
  mgRestrict: GPUComputePipeline
  mgProlong: GPUComputePipeline
  advectVel: GPUComputePipeline
  advectVelBack: GPUComputePipeline
  advectVelHalf: GPUComputePipeline
  projectReflect: GPUComputePipeline
  velCombine: GPUComputePipeline
  curl: GPUComputePipeline
  forces: GPUComputePipeline
  divergence: GPUComputePipeline
  sor: GPUComputePipeline
  project: GPUComputePipeline
  advFwd: GPUComputePipeline
  advBwd: GPUComputePipeline
  combine: GPUComputePipeline
  conserveSpread: GPUComputePipeline
  conserveRatio: GPUComputePipeline
  ballHeat: GPUComputePipeline
  vortBefore: GPUComputePipeline
  ivockTarget: GPUComputePipeline
  ivockCurl: GPUComputePipeline
  ivockSmooth: GPUComputePipeline
  ivockApply: GPUComputePipeline
  bimocqMaps: GPUComputePipeline
  bimocqRecon: GPUComputePipeline
  bimocqAccum: GPUComputePipeline
  advectVelDouble: GPUComputePipeline
  lfmCellVel: GPUComputePipeline
  lfmMarchFwd: GPUComputePipeline
  lfmMarchBwd: GPUComputePipeline
  lfmPull: [GPUComputePipeline, GPUComputePipeline, GPUComputePipeline]
  diffuse: GPUComputePipeline
  light: GPUComputePipeline
  cycDirect: GPUComputePipeline
  cycBounce: GPUComputePipeline
  flameNear: GPUComputePipeline
  floorNear: GPUComputePipeline
  render: GPURenderPipeline
  resolve: GPURenderPipeline
  display: GPURenderPipeline
  glareFirst: GPURenderPipeline
  glareDown: GPURenderPipeline
  glareUp: GPURenderPipeline
  upres: GPUComputePipeline
  upresCoarse: GPUComputePipeline
  ptComposite: GPURenderPipeline
}
// The pipelines only drawing needs (the light volume, the flames' and the floor's near light, the cyc's maps, the image
// pass and what follows it), and the solver's: everything else.
type FrameKey = 'light' | 'cycDirect' | 'cycBounce' | 'flameNear' | 'floorNear' | 'render' | 'resolve' | 'display' | 'glareFirst' | 'glareDown' | 'glareUp' | 'ptComposite'
type FramePipes = Pick<Pipes, FrameKey>
type SolverPipes = Omit<Pipes, FrameKey>

// One multigrid level: its cells, and bind groups for smoothing (by colour), the residual, restriction into the
// next coarser level, and prolongation from it.
interface Level {
  n: Vec3
  smooth: [GPUBindGroup, GPUBindGroup]
  residual: GPUBindGroup | null
  restrictTo: GPUBindGroup | null // into this level from the finer one (levels 1 and on)
  prolongFrom: GPUBindGroup | null // from this level into the finer one (levels 1 and on)
}

interface Grids {
  n: Vec3
  levels: Level[]
  textures: GPUTexture[]
  buffers: GPUBuffer[]
  vel: [GPUTexture, GPUTexture, GPUTexture, GPUTexture] // the step's velocity; projected; MacCormack's round trip; advected
  scal: [GPUTexture, GPUTexture, GPUTexture, GPUTexture]
  fireLight: GPUBuffer // the grid sums: fire light, its centre, the bed's area and the heat release rate
  hrr: GPUBuffer // heat release rate per cell, W/m3
  ratio: GPUBuffer // a fire's mass-keeping ratios (readable for development checks)
  spread: GPUBuffer // a fire's mass keeping: the volume over which each cell's gas was spread (fixed point)
  skin: GPUBuffer // the wooden ball's skin (SKIN_TEXELS texels, four floats each), and its next step
  skinNext: GPUBuffer
  ballSheet: GPUBuffer // the rest of the burning ball's surface, per skin texel (eight floats: the fuel flowing up it, ...)
  // The wood beneath the skin (BALL_HEAT): its cells' temperatures (this step's and the next's), their wood, char,
  // water and ash, and each column's state (this step's and the next's).
  woodTemps: GPUBuffer
  woodCells: GPUBuffer
  woodCols: GPUBuffer
  flame: GPUTexture // a fire's heat release rate per cell (MW/m3), for the renderer
  // IVOCK: the three components of the vector potential, each with its own finest multigrid level.
  // BiMocq2: the maps (the backward one's current level, ping-ponged; its previous level; the forward one,
  // ping-ponged), the accumulators, the snapshots, and an identity map to re-initialise from.
  bimocq: {
    x: [GPUBuffer, GPUBuffer]
    y: [GPUBuffer, GPUBuffer]
    xPrev: GPUBuffer
    dCurr: [GPUTexture, GPUTexture]
    dPrev: GPUTexture
    zero: GPUTexture
    identity: GPUBuffer
    phiCurr: GPUTexture
    phiPrev: GPUTexture
    phiStar: GPUTexture
    uniform: GPUBuffer
    maps: [GPUBindGroup, GPUBindGroup]
    recon: [GPUBindGroup, GPUBindGroup]
    accum: [GPUBindGroup, GPUBindGroup]
  }
  ivockRhs: GPUBuffer // its right-hand side, the vorticity lost, four floats a cell (readable for development checks)
  // Leapfrog flow maps: the cycle's half-step velocities (on the faces, and cell-centred for marching), its initial
  // velocity with the forces gathered in (ping-ponged), the step's forces, the pullbacks' scratch, the maps and their
  // Jacobian columns (four floats per face, three lattices), and the pressure kept aside while the re-initialisation
  // solves its own.
  lfm: {
    ring: GPUTexture[]
    ringC: GPUTexture[]
    u0: [GPUTexture, GPUTexture]
    delta: GPUTexture
    m: GPUTexture
    e: GPUTexture
    pressure: GPUBuffer
    pressureSave: GPUBuffer
    advect: GPUBindGroup[] // step k >= 2: the half-step velocity two back, through the last
    cellVel: GPUBindGroup[]
    marchFwd: GPUBindGroup[]
    marchBwd: GPUBindGroup
    pull: [GPUBindGroup, GPUBindGroup, GPUBindGroup]
  }
  detA: [GPUTexture, GPUTexture]
  detB: [GPUTexture, GPUTexture]
  lightDims: Vec3
  light: GPUTexture
  light2: GPUTexture
  cycDirect: GPUTexture // the light straight onto the cyc (CYC_DIRECT)
  cycBounce: [GPUTexture, GPUTexture] // the light the rest of the cyc sends it (CYC_BOUNCE), the render's the first
  emit: GPUTexture // each cell's glow and flame light per metre (FIRE_SUM, for FLAME_NEAR)
  flameNear: GPUTexture // the fire's light about each point of the flames (FLAME_NEAR)
  flameNearDir: GPUTexture // its first moment (FLAME_NEAR)
  floorNear: GPUTexture // the floor's light from the glowing gas within a bed's radius (FLOOR_NEAR)
  nearDims: Vec3
  bg: {
    advectVel: GPUBindGroup // semi-Lagrangian, straight to the advected texture
    advectVelFwd: GPUBindGroup
    advectVelBack: GPUBindGroup
    velCombine: GPUBindGroup
    arAdvect1: GPUBindGroup // advection-reflection: u^n through itself, half a step
    arAdvect2: GPUBindGroup // the reflected field through the halfway velocity, half a step
    projectReflect: GPUBindGroup
    curl: GPUBindGroup
    forces: GPUBindGroup
    divergence: GPUBindGroup
    sor: [GPUBindGroup, GPUBindGroup]
    project: GPUBindGroup
    advFwd: GPUBindGroup
    advBwd: GPUBindGroup
    combine: GPUBindGroup
    conserveSpread: GPUBindGroup
    conserveRatio: GPUBindGroup
    ballHeat: GPUBindGroup
    vortBefore: GPUBindGroup
    vortAdvect: GPUBindGroup
    vortBack: GPUBindGroup
    ivockTarget: GPUBindGroup
    ivockCurl: GPUBindGroup
    ivockSmooth: [GPUBindGroup, GPUBindGroup]
    ivockApply: GPUBindGroup
    diffuse: GPUBindGroup
    advDetail: GPUBindGroup
    fireSum: GPUBindGroup
    fireTotal: GPUBindGroup
  }
  // The frame's bind groups, made once its pipelines are (frameGroups).
  frame: FrameGroups | null
  frameGroups: (p: FramePipes) => FrameGroups
}
interface FrameGroups {
  light: GPUBindGroup
  cycDirect: GPUBindGroup
  cycBounce: [GPUBindGroup, GPUBindGroup] // (the first to the second, then back)
  flameNear: GPUBindGroup
  floorNear: GPUBindGroup
  render: GPUBindGroup
}

// The device and the compiled pipelines can be made ahead of time (prewarmPyro), while the page rests on the Lab:
// compiling the solver's programs can hold up the browser's frames for a moment, unnoticed when nothing moves, but a
// visible hitch right after a click. The first engine takes them.
interface Warm {
  adapter: GPUAdapter
  device: GPUDevice
  pipes: Pipes
  lost: boolean
}
let warm: Promise<Warm | null> | null = null
// Set once an engine has started: from then on there is nothing to make ahead. (The Lab's timer, armed while the page
// rests on it, could fire after a click on Pyro had started an engine, which takes warm and compiles its own when there
// is none: a second warm-up then compiled every pipeline again on a second device, alongside the engine's, and the
// two shared the browser's compiling threads. A visitor who clicked Pyro half a second after reaching the Lab waited
// 11.8 s for the solver, its image pass taking 8 to 11 s to compile instead of about 3.5.)
let engineStarted = false

async function requestDevice(): Promise<{ adapter: GPUAdapter; device: GPUDevice } | null> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) return null
  // Timestamps only for the development profiler (profileStep); the lab runs without them.
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [] })
  return { adapter, device }
}

async function compilePipes(d: GPUDevice, format: GPUTextureFormat, onSolver?: (p: SolverPipes) => void): Promise<Pipes> {
  const compute = (code: string, constants?: Record<string, number>) =>
    d.createComputePipelineAsync({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: 'main', constants } })
  const renderModule = d.createShaderModule({ code: RENDER })
  const resolveModule = d.createShaderModule({ code: RESOLVE })
  const displayModule = d.createShaderModule({ code: DISPLAY })
  const glareModule = d.createShaderModule({ code: GLARE })
  const compositeModule = d.createShaderModule({ code: PT_COMPOSITE })
  // The longest compiles are asked for first: the browser compiles a few at a time in the order asked, and the image
  // pass's alone takes a third of all the compiling, so started last it set the wait (a second more on the bonfire).
  const renderP = d.createRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs' },
    fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list' },
  })
  const flameNearP = compute(FLAME_NEAR)
  const forcesP = compute(FORCES)
  const floorNearP = compute(FLOOR_NEAR)
  const lightP = compute(LIGHT)
  const glarePipe = (entryPoint: string, add: boolean) =>
    d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: glareModule, entryPoint: 'vs' },
      fragment: {
        module: glareModule,
        entryPoint,
        targets: [{ format: 'rgba16float', blend: add ? { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } } : undefined }],
      },
      primitive: { topology: 'triangle-list' },
    })
  // (Asked for in the order they always were; the solver's are handed over as soon as they are all ready, about 1.4 s
  // after the request on an integrated GPU, the image pass's own taking some 4.5 s.)
  const solverP: Pending<SolverPipes> = {
    fireSum: compute(FIRE_SUM),
    fireTotal: compute(FIRE_SUM_TOTAL),
    noiseGen: compute(NOISE_GEN),
    curlGen: compute(CURL_GEN),
    advDetail: compute(ADVECT_DETAIL),
    mgSmooth: compute(MG_SMOOTH),
    mgResidual: compute(MG_RESIDUAL),
    mgRestrict: compute(MG_RESTRICT),
    mgProlong: compute(MG_PROLONG),
    advectVel: compute(ADVECT_VEL, { DIR: 1 }),
    advectVelBack: compute(ADVECT_VEL, { DIR: -1 }),
    advectVelHalf: compute(ADVECT_VEL, { DIR: 0.5 }),
    projectReflect: compute(PROJECT_REFLECT),
    velCombine: compute(VEL_COMBINE),
    curl: compute(CURL),
    forces: forcesP,
    divergence: compute(DIVERGENCE),
    sor: compute(SOR),
    project: compute(PROJECT),
    advFwd: compute(ADVECT_SCAL, { DIR: 1 }),
    advBwd: compute(ADVECT_SCAL, { DIR: -1 }),
    combine: compute(COMBINE),
    conserveSpread: compute(CONSERVE_SPREAD),
    conserveRatio: compute(CONSERVE_RATIO),
    ballHeat: compute(BALL_HEAT),
    vortBefore: compute(VORT_BEFORE),
    ivockTarget: compute(IVOCK_TARGET),
    ivockCurl: compute(IVOCK_CURL),
    ivockSmooth: compute(IVOCK_SMOOTH),
    ivockApply: compute(IVOCK_APPLY),
    bimocqMaps: compute(BIMOCQ_MAPS),
    bimocqRecon: compute(BIMOCQ_RECON),
    bimocqAccum: compute(BIMOCQ_ACCUM),
    advectVelDouble: compute(ADVECT_VEL, { DIR: 2 }),
    lfmCellVel: compute(LFM_CELLVEL),
    lfmMarchFwd: compute(LFM_MARCH_FWD),
    lfmMarchBwd: compute(LFM_MARCH_BWD),
    lfmPull: Promise.all([compute(LFM_PULL, { MODE: 1 }), compute(LFM_PULL, { MODE: 2 }), compute(LFM_PULL, { MODE: 3 })]),
    diffuse: compute(DIFFUSE),
    upres: compute(UPRES),
    upresCoarse: compute(UPRES_COARSE),
  }
  const frameP: Pending<FramePipes> = {
    light: lightP,
    render: renderP,
    resolve: d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: resolveModule, entryPoint: 'vs' },
      fragment: { module: resolveModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    }),
    display: d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: displayModule, entryPoint: 'vs' },
      fragment: { module: displayModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    }),
    glareFirst: glarePipe('fsDownFirst', false),
    glareDown: glarePipe('fsDown', false),
    glareUp: glarePipe('fsUp', true),
    ptComposite: d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: compositeModule, entryPoint: 'vs' },
      fragment: { module: compositeModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    }),
    cycDirect: compute(CYC_DIRECT),
    cycBounce: compute(CYC_BOUNCE),
    flameNear: flameNearP,
    floorNear: floorNearP,
  }
  const frame = settle(frameP)
  // (A failure among the frame's surfaces below, after the solver's; caught here too, so it is not reported early.)
  frame.catch(() => {})
  const solver = await settle(solverP)
  onSolver?.(solver)
  return { ...solver, ...(await frame) }
}

// An object of promises, settled into an object of their values.
type Pending<T> = { [K in keyof T]: Promise<T[K]> }
async function settle<T>(o: Pending<T>): Promise<T> {
  const keys = Object.keys(o) as (keyof T)[]
  const values = await Promise.all(keys.map(k => o[k]))
  return Object.fromEntries(keys.map((k, i) => [k, values[i]])) as T
}

export function prewarmPyro() {
  if (warm || engineStarted || typeof navigator === 'undefined' || !navigator.gpu) return
  warm = (async () => {
    try {
      const got = await requestDevice()
      if (!got) return null
      const w: Warm = { ...got, pipes: await compilePipes(got.device, navigator.gpu.getPreferredCanvasFormat()), lost: false }
      got.device.lost.then(() => {
        w.lost = true
      })
      return w
    } catch {
      return null
    }
  })()
}

export class PyroEngine {
  static detect(): boolean {
    return typeof navigator !== 'undefined' && !!navigator.gpu
  }

  // Takes the canvas's webgpu context first (or, if something else has taken that canvas, one of its own laid over
  // it, as the lookdev lab does), then an adapter and device. Resolves null where WebGPU is missing or refuses.
  static async create(canvas: HTMLCanvasElement, onStatus: (s: PyroStatus) => void, initial: PyroState, onLost?: () => void) {
    if (!PyroEngine.detect()) return null
    let own: HTMLCanvasElement | null = null
    try {
      let ctx = canvas.getContext('webgpu')
      if (!ctx) {
        own = document.createElement('canvas')
        ctx = own.getContext('webgpu')
        if (!ctx) {
          console.warn('Pyro: no webgpu context on the canvas, nor on one of its own')
          return null
        }
        own.className = 'lab-canvas-own'
        own.setAttribute('aria-hidden', 'true')
        canvas.after(own)
        console.warn('Pyro: the canvas gave no webgpu context; drawing on one of its own')
      }
      // A device and pipelines made ahead of time, if they are ready and still usable; else new ones. Taken at once, so
      // an engine started meanwhile (React's development double mount starts two) makes its own rather than sharing a
      // device the other destroys when it is thrown away.
      const pending = warm
      warm = null
      engineStarted = true
      const ahead = pending ? await pending : null
      const got = ahead && !ahead.lost ? ahead : await requestDevice()
      if (!got) {
        own?.remove()
        console.warn('Pyro: the browser offered no adapter')
        return null
      }
      const info = got.adapter.info
      const name = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ') || 'unnamed adapter'
      const engine = new PyroEngine(own ?? canvas, own, ctx, got.device, onStatus, initial, onLost, name, ahead && !ahead.lost ? ahead.pipes : null)
      engine.adapterFeatures = [...got.adapter.features].sort()
      const lim = got.adapter.limits
      engine.adapterLimits = {
        maxBufferSize: lim.maxBufferSize,
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxStorageBuffersPerShaderStage: lim.maxStorageBuffersPerShaderStage,
        maxStorageTexturesPerShaderStage: lim.maxStorageTexturesPerShaderStage,
        maxTextureDimension3D: lim.maxTextureDimension3D,
        maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup,
      }
      return engine
    } catch (err) {
      own?.remove()
      console.warn('Pyro: WebGPU unavailable', err)
      return null
    }
  }

  readonly adapterName: string
  // For the lab's device check (?check): what the adapter offers, and the WebGPU errors so far (the latest 50).
  adapterFeatures: string[] = []
  adapterLimits: Record<string, number> = {}
  readonly gpuErrors: string[] = []
  readonly whenReady: Promise<boolean>
  state: PyroState

  private format: GPUTextureFormat
  private sampler: GPUSampler
  private simBuf: GPUBuffer
  private renderBuf: GPUBuffer
  private parityBufs: [GPUBuffer, GPUBuffer]
  private simData = new ArrayBuffer(384)
  private renderData = new Float32Array(112)
  private wb: { kelvin: number; m: number[][] } | null = null
  private bbBuf: GPUBuffer
  private noiseTex: GPUTexture
  private curlTex: GPUTexture // the upres pass's turbulence, from the noise (see CURL_GEN)
  private curlMaxBuf: GPUBuffer // its largest magnitude (a float's bits)
  private fireLit = false // the fire's light was summed last step (a smoke source's is zero, cleared once)
  private blueTex: GPUTexture
  // Temporal accumulation: this frame's march (high dynamic range), the history it blends into (two, swapped each
  // frame), the blend's uniform, and how many frames have accumulated since anything changed.
  private targets: Targets | null = null
  private taaBuf: GPUBuffer
  private histIndex = 0
  private taaCount = 0
  private renderFrame = 0
  // The wooden ball, as last read back for the HUD.
  private ballStatus: BallStatus | null = null
  // BiMocq2: which map buffers are current, steps since the last re-initialisation, whether there is a previous
  // level, and whether it must start afresh (a new grid, or switched on).
  private bimocqParity = 0
  private bimocqSteps = 0
  private bimocqHasPrev = false
  private bimocqReset = true
  private bimocqWas = false
  // Leapfrog flow maps: the step within the cycle, and whether the last step used them.
  private lfmStep = 0
  private lfmWas = false
  private readingBall = false
  // The wood (BALL_HEAT): which half of its temperatures holds this step's, its clock (s), how many times faster than the
  // gas it runs (1 live), and, for development tests, the exposure it takes instead of the fire's (0: the fire; W/m2 of
  // radiant flux; -1: the standard furnace fire).
  private woodParity = 0
  private woodClock = 0
  private woodPhase = -1
  woodSpeed = 1
  woodTest = 0
  woodAir = 0 // (development: m/s of air blown over the wood under a test exposure; 0, the test's still air)
  // The wood's wear as the gas sees it (wornRadius; fitWear), fitted afresh from the wear map every fourth time the
  // ball is read back, about once a second (the wear moves a millimetre in a minute or two).
  private wear: Float32Array = new Float32Array(12)
  private wearReads = 0
  private balls = 0 // (fresh balls so far, so that a wear read back from the last is not fitted to this one)
  // Auto exposure: the fire's luminous intensity (cd, read back a few times a second) and the metered EV, eased.
  private fireCd = 0
  private fireResum = false // (paused: the fire's sums marched afresh before the next frame's light: resumFireNow)
  private metering = false
  private meterEV: number | null = null
  private meterAt = 0
  // The collision ball: its centre and velocity (from the visitor's drags, easing to rest), and its uniform for the
  // pressure solve's levels.
  private ballPos: Vec3 = [0.55, 0.7, 0]
  private ballVel: Vec3 = [0, 0, 0]
  private ballDragAt = 0
  private ballBuf: GPUBuffer
  // Wavelet-turbulence upres for a smoke source: its two textures (twice the grid's resolution), made when first
  // needed; the blocks of it that held smoke last step and this; the pass's bind group and uniform; the render bind
  // group that reads it; and whether it must refill from the coarse density on its next step.
  private upres: { tex: [GPUTexture, GPUTexture]; held: [GPUBuffer, GPUBuffer]; group: GPUBindGroup; render: GPUBindGroup; coarse: GPUTexture; coarseGroup: GPUBindGroup; light: GPUBindGroup } | null = null
  private upresBuf: GPUBuffer
  private upresInit = true
  private upresWasActive = false
  private dummy3d: GPUTexture
  // An HDR environment's light (environment.ts), once loaded, and its backdrop (cleared to black without one).
  private envLight: EnvLight | null = null
  private envTex: GPUTexture
  private envResTex: GPUTexture // the map without its key, for the path-traced reference
  // The path-traced reference: its pipeline (compiled the first time it is asked for), uniforms, lights, targets.
  private ptPipe: GPURenderPipeline | null = null
  private ptCompiling = false
  private ptBuf: GPUBuffer
  private wipeBuf: GPUBuffer
  private pt = { on: false, n: 0, idx: 0, wipe: 0.5, key: '', frame: 0 }
  // (Development: light the reference leaves out, to split its light by where it comes from: 1 the floor's but for the
  // camera's own sight of it, 2 all but single scattering in the gas, 4 the sky's, 8 the lights', 16 all scattering in
  // the gas (its glow alone, dimmed); set before tracing.)
  ptParts = 0
  private ptLights = new Float32Array(24)
  private ptT: { targets: Targets; upres: boolean; accum: [GPUTexture, GPUTexture]; comp: GPUTexture; trace: [GPUBindGroup, GPUBindGroup]; composite: GPUBindGroup[][]; display: GPUBindGroup } | null = null
  private stateVersion = 0
  // The ACES 2.0 view's LUT (a 1 x 1 stand-in until it has loaded; AgX shows meanwhile), shared with the Lookdev lab.
  private lut: GPUTexture
  private lutSampler: GPUSampler
  private lutReady = false
  private glareSampler: GPUSampler
  private noiseSampler: GPUSampler
  private pipes: Pipes | null = null
  private solver: SolverPipes | null = null // (the solver's alone, before the image pass's are ready; then all of them)
  private prerolled: Promise<void> = Promise.resolve()
  private grids: Grids | null = null
  private disposed = false
  private lost = false
  private visible = false
  private paused = false
  private raf = 0
  private lastTick = 0
  private acc = 0
  private simTime = 0
  private frame = 0
  private needsRender = true
  private lightDirty = true
  private stepTimes: number[] = []
  private frameTimes: number[] = []
  private renderScale = RENDER_SCALE_START
  private scaleCeiling = Infinity // the render scale the simulation keeps real time at (lowered when it falls behind)
  // The march's size while running (at the render scale), and this frame's (paused, the display's once settling);
  // frames drawn since the pause or the last change while paused; this frame's exposure and the last's (the history is
  // rescaled by their ratio as the auto exposure eases); the camera's version (a new framing starts the history
  // afresh); and the collision ball where the last frame drew it (the history under its old and new places starts
  // afresh as it is dragged, rather than the whole frame's).
  private marchW = 1
  private marchH = 1
  private frameMarch: [number, number] = [1, 1]
  private settle = 0
  private exposure = 1
  private lastExposure = 1
  private camVersion = 0
  private ballDrawn: Vec3 | null = null
  // (Development: a render scale pinned, which the adapting leaves alone; and a march size forced, for a still.)
  pinScale: number | null = null
  private forceMarch: [number, number] | null = null
  private steadySince = 0 // when the loop last began running without a break (ms)
  private behindMs = 0 // how long the simulation has run behind real time, without a break
  private lastAdapt = 0
  private inFlight = 0 // frames of GPU work submitted and not yet finished
  private runClock = 0 // seconds spent running (on screen, not paused) on the current grids
  private autoSteps = 0 // steps taken in the judging window
  // Until the visitor picks a resolution, a default that cannot keep up steps down (once per grid).
  autoResolution = true
  private autoStepped = false
  private cssW = 0
  private cssH = 0
  private dpr = 1
  private lastStatus = 0
  private stir: { centre: Vec3; vel: Vec3 } | null = null
  private cam = { pos: [0, 0, 0] as Vec3, fwd: [0, 0, -1] as Vec3, right: [1, 0, 0] as Vec3, up: [0, 1, 0] as Vec3, tanY: 0, aspect: 1 }
  // The lab's framing (layoutCamera): the camera's distance along z, the tangent of half its field of view, its aspect.
  private framing = { z: 0, tanY: 0, aspect: 1 }
  // The view (setView): the whole fire, or close on the ball where it was when the view last aimed at it, so the camera
  // moves only when the view does, never while the ball is dragged; and the frames drawn that callers wait for
  // (framesDrawn).
  private view: PyroView = 'whole'
  private closeOn: Vec3 = [0, 0, 0]
  private drawWaits: { at: number; done: () => void }[] = []

  private constructor(
    private canvas: HTMLCanvasElement,
    private ownCanvas: HTMLCanvasElement | null,
    private ctx: GPUCanvasContext,
    private device: GPUDevice,
    private onStatus: (s: PyroStatus) => void,
    initial: PyroState,
    private onLost: (() => void) | undefined,
    adapterName: string,
    prebuilt: Pipes | null,
  ) {
    this.adapterName = adapterName
    this.state = { ...PYRO_DEFAULTS, ...initial }
    const d = device
    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.configure()
    d.addEventListener('uncapturederror', ev => {
      const message = (ev as GPUUncapturedErrorEvent).error.message
      console.error(`Pyro WebGPU error: ${message}`)
      if (this.gpuErrors.length < 50) this.gpuErrors.push(message.slice(0, 300))
    })
    d.lost.then(info => {
      this.lost = true
      if (this.disposed) return
      console.warn('Pyro: WebGPU device lost', info.message)
      this.stop()
      this.onLost?.()
    })
    const U = GPUBufferUsage
    this.simBuf = d.createBuffer({ size: this.simData.byteLength, usage: U.UNIFORM | U.COPY_DST })
    this.renderBuf = d.createBuffer({ size: this.renderData.byteLength, usage: U.UNIFORM | U.COPY_DST })
    this.parityBufs = [0, 1].map(k => {
      const b = d.createBuffer({ size: 16, usage: U.UNIFORM, mappedAtCreation: true })
      new Uint32Array(b.getMappedRange()).set([k, 0, 0, 0])
      b.unmap()
      return b
    }) as [GPUBuffer, GPUBuffer]
    this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' })
    const bb = planckTable()
    this.bbBuf = d.createBuffer({ size: bb.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    d.queue.writeBuffer(this.bbBuf, 0, bb)
    this.noiseSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' })
    this.noiseTex = d.createTexture({
      size: [NOISE_SIZE, NOISE_SIZE, NOISE_SIZE],
      dimension: '3d',
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    })
    // The march's offsets: spatiotemporal blue noise, a slice a frame. Until the mask has loaded, the 2D blue noise
    // stepped by the golden ratio from slice to slice, as the renderer began.
    this.blueTex = d.createTexture({ size: [BLUE_NOISE_SIZE, BLUE_NOISE_SIZE, STBN_SLICES], format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    const tile = blueNoise()
    const slices = new Uint8Array(BLUE_NOISE_SIZE * BLUE_NOISE_SIZE * STBN_SLICES)
    for (let k = 0; k < STBN_SLICES; k++)
      for (let i = 0; i < tile.length; i++) slices[k * tile.length + i] = Math.floor((((tile[i] + 0.5) / 256 + k * 0.61803399) % 1) * 256)
    d.queue.writeTexture({ texture: this.blueTex }, slices, { bytesPerRow: BLUE_NOISE_SIZE, rowsPerImage: BLUE_NOISE_SIZE }, [BLUE_NOISE_SIZE, BLUE_NOISE_SIZE, STBN_SLICES])
    void fetch(STBN_URL)
      .then(r => (r.ok ? r.arrayBuffer() : null))
      .then(buf => {
        if (!buf || buf.byteLength !== slices.length || this.disposed) return
        d.queue.writeTexture({ texture: this.blueTex }, new Uint8Array(buf), { bytesPerRow: BLUE_NOISE_SIZE, rowsPerImage: BLUE_NOISE_SIZE }, [BLUE_NOISE_SIZE, BLUE_NOISE_SIZE, STBN_SLICES])
        this.needsRender = true
        this.taaCount = 0
      })
      .catch(() => {})
    this.taaBuf = d.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST })
    this.ballBuf = d.createBuffer({ size: 160, usage: U.UNIFORM | U.COPY_DST })
    this.upresBuf = d.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST })
    this.dummy3d = d.createTexture({ size: [1, 1, 1], dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING })
    this.envTex = d.createTexture({ size: [BACKDROP_W, BACKDROP_H], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    this.envResTex = d.createTexture({ size: [BACKDROP_W, BACKDROP_H], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    this.ptBuf = d.createBuffer({ size: 112, usage: U.UNIFORM | U.COPY_DST })
    this.wipeBuf = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST })
    this.syncEnv()
    this.lut = d.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING })
    this.lutSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    this.glareSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    void loadAcesLutGPU(d).then(lut => {
      if (!lut) return
      if (this.disposed) {
        lut.destroy()
        return
      }
      this.lut.destroy()
      this.lut = lut
      this.lutReady = true
      if (this.targets && this.pipes) Object.assign(this.targets, this.postGroups(this.targets.hdr, this.targets.hist, this.targets.glareViews))
      this.needsRender = true
      this.taaCount = 0
      this.kick()
    })
    this.curlTex = d.createTexture({
      size: [NOISE_SIZE, NOISE_SIZE, NOISE_SIZE],
      dimension: '3d',
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    })
    // (Copyable both ways for development checks: setting it huge turns the upres pass's skipping off, to compare.)
    this.curlMaxBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
    this.whenReady = this.build(prebuilt)
  }

  // Compiles every pipeline in the background. The grids and the solver start as soon as the solver's own pipelines are
  // ready, and the fire runs on (preroll) while the image pass's compile; its first frame is drawn once both are done.
  private async build(prebuilt: Pipes | null) {
    const pipes = prebuilt ?? (await compilePipes(this.device, this.format, solver => this.startSolver(solver)))
    if (this.disposed) return false
    if (!this.solver) this.startSolver(pipes)
    await this.prerolled
    if (this.disposed) return false
    // (Only now, so the frame loop, which waits for them, cannot start beside the head start.)
    this.pipes = pipes
    this.solver = pipes
    // (A resolution chosen meanwhile: its grids, from the start. Otherwise the frame's bind groups over these.)
    if (this.grids!.n.join() !== RESOLUTIONS[this.state.resolution].dims.join()) this.makeGrids()
    else this.grids!.frame = this.grids!.frameGroups(pipes)
    this.layoutCamera()
    this.needsRender = true
    this.kick()
    return true
  }

  // The fire run on to HEAD_START_S before its first frame, nothing drawn, while the image pass's pipelines compile (or,
  // with every pipeline made ahead, at once). A fire grows from its ignition for its first two seconds or so, a puff of
  // smoke rising off a low flame; the first frame a visitor sees is the fire as it burns. A few steps at a time, as
  // fast as the GPU takes them: about two simulated seconds a second on an integrated GPU, the image pass's compile
  // leaving some three seconds. (Not for smoke with its upres grid, whose pass needs the image pass's layout.)
  private async preroll() {
    if (this.upresActive()) return
    // (Some of the step's kernels read the image pass's uniforms: the fire's light is summed as it is drawn.)
    this.writeRender()
    while (!this.disposed && this.simTime < HEAD_START_S - 1e-6) {
      for (let i = 0; i < PREROLL_BATCH; i++) this.step()
      await this.device.queue.onSubmittedWorkDone()
    }
    if (this.disposed) return
    // The meter starts from the fire as it now burns, and the simulation's rate is judged from the first frame.
    await this.readFire()
    this.meterEV = null
    this.stepTimes = []
    this.steadySince = performance.now()
  }

  // The solver's pipelines ready: the detail noise, the grids, and the head start.
  private startSolver(solver: SolverPipes) {
    if (this.disposed) return
    this.solver = solver
    const pipes = solver
    // The detail noise, made once, and the upres pass's turbulence from it.
    const enc = this.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pipes.noiseGen)
    pass.setBindGroup(0, this.device.createBindGroup({ layout: pipes.noiseGen.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.noiseTex.createView() }] }))
    pass.dispatchWorkgroups(NOISE_SIZE / 4, NOISE_SIZE / 4, NOISE_SIZE / 4)
    pass.setPipeline(pipes.curlGen)
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipes.curlGen.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.noiseTex.createView() },
          { binding: 1, resource: this.noiseSampler },
          { binding: 2, resource: this.curlTex.createView() },
          { binding: 3, resource: { buffer: this.curlMaxBuf } },
        ],
      }),
    )
    pass.dispatchWorkgroups(NOISE_SIZE / 4, NOISE_SIZE / 4, NOISE_SIZE / 4)
    pass.end()
    this.device.queue.submit([enc.finish()])
    this.makeGrids()
    this.prerolled = this.preroll()
  }

  // The grids for the current resolution, zeroed, and every bind group over them.
  private makeGrids() {
    const d = this.device
    const p = this.solver!
    this.grids?.textures.forEach(t => t.destroy())
    this.upres?.tex.forEach(t => t.destroy())
    this.upres?.coarse.destroy()
    this.upres?.held.forEach(b => b.destroy())
    this.upres = null
    this.fireLit = true
    this.upresInit = true
    this.grids?.buffers.forEach(b => b.destroy())
    const n = RESOLUTIONS[this.state.resolution].dims
    const T = GPUTextureUsage
    const tex = (size: Vec3, copy = false) =>
      d.createTexture({
        size,
        dimension: '3d',
        format: 'rgba16float',
        usage: T.TEXTURE_BINDING | T.STORAGE_BINDING | (copy ? T.COPY_SRC | T.COPY_DST : 0),
      })
    const velSize: Vec3 = [n[0] + 1, n[1] + 1, n[2] + 1]
    const vel: [GPUTexture, GPUTexture, GPUTexture, GPUTexture] = [tex(velSize, true), tex(velSize, true), tex(velSize, true), tex(velSize, true)]
    const scal = [tex(n, true), tex(n, true), tex(n), tex(n)] as [GPUTexture, GPUTexture, GPUTexture, GPUTexture]
    const detA: [GPUTexture, GPUTexture] = [tex(n, true), tex(n, true)]
    const detB: [GPUTexture, GPUTexture] = [tex(n, true), tex(n, true)]
    const curl = tex(n)
    // IVOCK: the vorticity before advection, and advected.
    // IVOCK's partner: the momentum's eddy viscosity, as the diffusion step's weight per cell.
    const nut = tex(n)
    const vortA = tex(n)
    const vortB = tex(n)
    const vortC = tex(n)
    const psiCurl = tex(n)
    // A fire's heat release rate per cell (MW/m3), filterable, for rendering the flame sheets.
    const flame = tex(n)
    // Each cell's emitted light (FIRE_SUM), and the fire's light about the flames from it (FLAME_NEAR).
    const emit = tex(n)
    const lightDims: Vec3 = [Math.ceil(n[0] / 2), Math.ceil(n[1] / 2), Math.ceil(n[2] / 2)]
    const light = tex(lightDims)
    // The softbox rig's rim: its own optical depth (x), so the smoke shadows it too.
    const light2 = tex(lightDims)
    // The cyc's light (CYC_DIRECT, CYC_BOUNCE), maps of it.
    const cycTex = () => d.createTexture({ size: [CYC_MAP.w, CYC_MAP.h], format: 'rgba16float', usage: T.TEXTURE_BINDING | T.STORAGE_BINDING })
    const cycDirect = cycTex()
    const cycBounce: [GPUTexture, GPUTexture] = [cycTex(), cycTex()]
    const nearDims: Vec3 = lightDims
    const flameNear = tex(nearDims)
    const flameNearDir = tex(nearDims)
    const floorNear = d.createTexture({ size: [n[0], n[2]], format: 'rgba16float', usage: T.TEXTURE_BINDING | T.STORAGE_BINDING })
    const v = (t: GPUTexture) => t.createView()
    const group = (pipe: GPUComputePipeline | GPURenderPipeline, resources: GPUBindingResource[]) =>
      d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({ binding, resource })) })
    const cells = n[0] * n[1] * n[2]
    const div = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    const pressure = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
    const buffers: GPUBuffer[] = [div, pressure]
    const storage = (count: number) => {
      const b = d.createBuffer({ size: Math.max(16, count * 4), usage: GPUBufferUsage.STORAGE })
      buffers.push(b)
      return b
    }
    // Multigrid levels: level 0 solves for the pressure itself (right-hand side: the divergence buffer); each coarser
    // level holds a correction, its right-hand side and a residual.
    const dx = DOMAIN_WIDTH / n[0]
    const dims: Vec3[] = [n]
    while (dims.length < 6 && Math.min(...dims[dims.length - 1]) >= 8) {
      const f = dims[dims.length - 1]
      dims.push([Math.ceil(f[0] / 2), Math.ceil(f[1] / 2), Math.ceil(f[2] / 2)])
    }
    const xs: GPUBuffer[] = [pressure]
    const gs: GPUBuffer[] = [div]
    const rs: GPUBuffer[] = []
    dims.forEach((m, l) => {
      const count = m[0] * m[1] * m[2]
      if (l > 0) {
        xs.push(storage(count))
        gs.push(storage(count))
      }
      rs.push(storage(count))
    })
    const coarsest = dims.length - 1
    // Smoothing and residuals also see the collision ball (binding 5).
    const withBall = (pipe: GPUComputePipeline, resources: GPUBindingResource[]) =>
      d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [...resources.map((resource, binding) => ({ binding, resource })), { binding: 5, resource: { buffer: this.ballBuf } }],
      })
    const levelUniform = (l: number, colour: number) => {
      const b = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true })
      const m = dims[l]
      const f = dims[Math.max(0, l - 1)]
      const h = dx * 2 ** l
      new Uint32Array(b.getMappedRange(0, 32)).set([m[0], m[1], m[2], colour, f[0], f[1], f[2], 0])
      new Float32Array(b.getMappedRange(32, 16)).set([h * h, l === coarsest ? MG_COARSE_OMEGA : 1, 0, 0])
      b.unmap()
      buffers.push(b)
      return b
    }
    const levels: Level[] = dims.map((m, l) => {
      const u0 = levelUniform(l, 0)
      const u1 = levelUniform(l, 1)
      return {
        n: m,
        smooth: [withBall(p.mgSmooth, [{ buffer: u0 }, { buffer: gs[l] }, { buffer: xs[l] }]), withBall(p.mgSmooth, [{ buffer: u1 }, { buffer: gs[l] }, { buffer: xs[l] }])],
        residual: l < coarsest ? withBall(p.mgResidual, [{ buffer: u0 }, { buffer: gs[l] }, { buffer: xs[l] }, { buffer: rs[l] }]) : null,
        restrictTo: l > 0 ? group(p.mgRestrict, [{ buffer: u0 }, { buffer: rs[l - 1] }, { buffer: gs[l] }, { buffer: xs[l] }]) : null,
        prolongFrom: l > 0 ? group(p.mgProlong, [{ buffer: u0 }, { buffer: xs[l] }, { buffer: xs[l - 1] }]) : null,
      }
    })
    // IVOCK's vector potential, a component at a time: its right-hand side and solution at the finest level (warm
    // started from the last step), sharing the coarser levels with the pressure.
    const ivockRhs = d.createBuffer({ size: cells * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    buffers.push(ivockRhs)
    const ivockPsi = storage(cells * 4)
    // BiMocq2's buffers and bind groups (see 7a): parity k reads map k and writes map 1 - k.
    const mapBuffer = () => {
      const b = d.createBuffer({ size: cells * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
      buffers.push(b)
      return b
    }
    const bx: [GPUBuffer, GPUBuffer] = [mapBuffer(), mapBuffer()]
    const by: [GPUBuffer, GPUBuffer] = [mapBuffer(), mapBuffer()]
    const xPrev = mapBuffer()

    const identity = d.createBuffer({ size: cells * 16, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true })
    {
      const a = new Float32Array(identity.getMappedRange())
      for (let z = 0; z < n[2]; z++)
        for (let y = 0; y < n[1]; y++)
          for (let x = 0; x < n[0]; x++) a.set([x + 0.5, y + 0.5, z + 0.5, 1], ((z * n[1] + y) * n[0] + x) * 4)
      identity.unmap()
    }
    buffers.push(identity)
    const snap = (store = false) =>
      d.createTexture({
        size: n,
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | (store ? GPUTextureUsage.STORAGE_BINDING : 0),
      })
    const phiCurr = snap()
    const phiPrev = snap()
    // The accumulators as textures (the rebuild reads them filtered), the current one ping-ponged; and zeros to clear
    // them from.
    const dCurr: [GPUTexture, GPUTexture] = [snap(true), snap(true)]
    const dPrev = snap()
    const zero = snap()
    const phiStar = tex(n)
    const bimocqUniform = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    buffers.push(bimocqUniform)
    const bimocq: Grids['bimocq'] = {
      x: bx,
      y: by,
      xPrev,
      dCurr,
      dPrev,
      zero,
      identity,
      phiCurr,
      phiPrev,
      phiStar,
      uniform: bimocqUniform,
      maps: [0, 1].map(k => group(p.bimocqMaps, [{ buffer: this.simBuf }, this.sampler, v(vel[0]), { buffer: bx[k] }, { buffer: bx[1 - k] }, { buffer: by[k] }, { buffer: by[1 - k] }])) as [GPUBindGroup, GPUBindGroup],
      recon: [0, 1].map(k =>
        group(p.bimocqRecon, [{ buffer: this.simBuf }, this.sampler, v(vel[0]), { buffer: bx[1 - k] }, { buffer: xPrev }, v(phiCurr), v(phiPrev), v(dCurr[k]), v(dPrev), v(scal[0]), v(scal[1]), v(phiStar), { buffer: bimocqUniform }]),
      ) as [GPUBindGroup, GPUBindGroup],
      accum: [0, 1].map(k => group(p.bimocqAccum, [{ buffer: this.simBuf }, this.sampler, v(scal[0]), v(phiStar), { buffer: by[1 - k] }, v(dCurr[k]), v(dCurr[1 - k])])) as [GPUBindGroup, GPUBindGroup],
    }
    // Leapfrog flow maps (see 2c).
    const ring = Array.from({ length: LFM_N }, () => tex(velSize, true))
    const ringC = Array.from({ length: LFM_N }, () => tex(n))
    const u0: [GPUTexture, GPUTexture] = [tex(velSize, true), tex(velSize, true)]
    const delta = tex(velSize)
    const lm = tex(velSize)
    const le = tex(velSize)
    const faces = 3 * velSize[0] * velSize[1] * velSize[2]
    const faceBuffer = () => {
      const b = d.createBuffer({ size: faces * 16, usage: GPUBufferUsage.STORAGE })
      buffers.push(b)
      return b
    }
    const phi = faceBuffer()
    const fcol = faceBuffer()
    const psi = faceBuffer()
    const tcol = faceBuffer()
    // The impulse's gauge at each cell at the cycle's end: the pressure and |u|^2 / 2 gathered along its path.
    const lam = storage(cells * 2)
    const pressureSave = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
    buffers.push(pressureSave)
    const lfm: Grids['lfm'] = {
      ring,
      ringC,
      u0,
      delta,
      m: lm,
      e: le,
      pressure,
      pressureSave,
      advect: ring.map(r => group(p.advectVelDouble, [{ buffer: this.simBuf }, this.sampler, v(vel[0]), v(r), v(vel[3])])),
      cellVel: ringC.map(r => group(p.lfmCellVel, [{ buffer: this.simBuf }, v(vel[1]), v(r), { buffer: pressure }])),
      marchFwd: ringC.map(r => group(p.lfmMarchFwd, [{ buffer: this.simBuf }, this.sampler, v(r), v(delta), v(u0[0]), v(u0[1]), { buffer: phi }, { buffer: fcol }])),
      marchBwd: group(p.lfmMarchBwd, [{ buffer: this.simBuf }, this.sampler, ...ringC.map(v), { buffer: psi }, { buffer: tcol }, { buffer: lam }]),
      pull: [
        group(p.lfmPull[0], [{ buffer: this.simBuf }, this.sampler, { buffer: psi }, { buffer: tcol }, v(u0[0]), v(u0[0]), v(lm), { buffer: lam }]),
        group(p.lfmPull[1], [{ buffer: this.simBuf }, this.sampler, { buffer: phi }, { buffer: fcol }, v(lm), v(u0[0]), v(le), { buffer: lam }]),
        group(p.lfmPull[2], [{ buffer: this.simBuf }, this.sampler, { buffer: psi }, { buffer: tcol }, v(le), v(lm), v(vel[0]), { buffer: lam }]),
      ],
    }
    // The fire's light: per-workgroup partial sums, then the total (intensity and centre), read by the renderer.
    const groups = Math.ceil(n[0] / 4) * Math.ceil(n[1] / 4) * Math.ceil(n[2] / 4)
    const firePartial = storage(groups * 40)
    const fireLight = d.createBuffer({ size: FIRE_LIGHT_ENTRIES * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
    // The heat release rate in each cell (W/m3), from the step's burning: for the expansion, the sums and validation.
    const hrr = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    // A fire's mass keeping: the volume each cell's gas was spread over, and the ratio its advection is scaled by.
    const spread = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    const ratio = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    buffers.push(fireLight, hrr, spread, ratio)
    // The wooden ball's skin, fresh wood at the air's temperature, and the buffer its heating step writes; and the wood
    // beneath it (some 24 MB), cleared on a new ball (resetSkin).
    const skinBytes = SKIN_TEXELS * 16
    const skin = d.createBuffer({ size: skinBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
    const skinNext = d.createBuffer({ size: skinBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    const ballSheet = d.createBuffer({ size: 2 * skinBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
    const woodUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    const woodTemps = d.createBuffer({ size: 2 * WOOD_CELLS * 4, usage: woodUsage })
    const woodCells = d.createBuffer({ size: WOOD_CELLS * 16, usage: woodUsage })
    const woodCols = d.createBuffer({ size: 2 * SKIN_TEXELS * 32, usage: woodUsage })
    buffers.push(skin, skinNext, ballSheet, woodTemps, woodCells, woodCols)
    // The kernels that read the skin (binding 31: the ball's volatiles feed the fire, and the renderer draws it).
    const withSkin = (pipe: GPUComputePipeline | GPURenderPipeline, resources: GPUBindingResource[], extra: GPUBindGroupEntry[] = []) =>
      d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [...resources.map((resource, binding) => ({ binding, resource })), { binding: 31, resource: { buffer: skin } }, ...extra],
      })
    const fireCount = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true })
    new Uint32Array(fireCount.getMappedRange()).set([groups, Math.ceil(n[0] / 4), Math.ceil(n[1] / 4), Math.ceil(n[2] / 4)])
    fireCount.unmap()
    buffers.push(fireCount)
    const P = { buffer: this.simBuf }
    const R = { buffer: this.renderBuf }
    const s = this.sampler
    // The frame's bind groups, made when its pipelines are (with the grids, or once the image pass has compiled).
    const frameGroups = (p: FramePipes): FrameGroups => ({
      flameNear: d.createBindGroup({
        layout: p.flameNear.getBindGroupLayout(0),
        entries: [
          ...[P, R, s].map((resource, binding) => ({ binding, resource })),
          { binding: 4, resource: v(light) },
          { binding: 10, resource: { buffer: fireLight } },
          { binding: 40, resource: v(flameNear) },
          { binding: 41, resource: v(emit) },
          { binding: 42, resource: v(flameNearDir) },
        ],
      }),
      floorNear: d.createBindGroup({
        layout: p.floorNear.getBindGroupLayout(0),
        entries: [
          ...[P, R, s].map((resource, binding) => ({ binding, resource })),
          { binding: 10, resource: { buffer: fireLight } },
          { binding: 41, resource: v(emit) },
          { binding: 47, resource: v(floorNear) },
        ],
      }),
      light: group(p.light, [P, R, s, v(scal[0]), v(light), { buffer: fireLight }, v(light2), this.dummy3d.createView()]),
      cycDirect: d.createBindGroup({
        layout: p.cycDirect.getBindGroupLayout(0),
        entries: [
          ...[P, R, s].map((resource, binding) => ({ binding, resource })),
          { binding: 4, resource: v(light) },
          { binding: 10, resource: { buffer: fireLight } },
          { binding: 34, resource: v(light2) },
          { binding: 40, resource: v(cycDirect) },
          { binding: 46, resource: v(floorNear) },
        ],
      }),
      cycBounce: [0, 1].map(k =>
        d.createBindGroup({
          layout: p.cycBounce.getBindGroupLayout(0),
          entries: [
            ...[P, R, s, v(scal[0])].map((resource, binding) => ({ binding, resource })),
            { binding: 39, resource: v(cycBounce[k]) },
            { binding: 40, resource: v(cycBounce[1 - k]) },
            { binding: 41, resource: v(cycDirect) },
          ],
        }),
      ) as [GPUBindGroup, GPUBindGroup],
      render: withSkin(p.render, [P, R, s, v(scal[0]), v(light), v(detA[0]), v(detB[0]), this.noiseTex.createView(), this.noiseSampler, { buffer: this.bbBuf }, { buffer: fireLight }, this.blueTex.createView({ dimension: '2d-array' }), this.dummy3d.createView(), v(flame)], [{ binding: 32, resource: { buffer: ballSheet } }, { binding: 33, resource: this.envTex.createView() }, { binding: 34, resource: v(light2) }, { binding: 35, resource: v(vel[0]) }, { binding: 39, resource: v(cycBounce[0]) }, { binding: 43, resource: v(cycDirect) }, { binding: 44, resource: v(flameNear) }, { binding: 45, resource: v(flameNearDir) }, { binding: 46, resource: v(floorNear) }]),
    })
    this.grids = {
      n,
      textures: [...vel, ...scal, ...detA, ...detB, curl, light, cycDirect, ...cycBounce, emit, flameNear, flameNearDir, floorNear, flame, nut, vortA, vortB, vortC, psiCurl, ...ring, ...ringC, ...u0, delta, lm, le],
      buffers,
      levels,
      vel,
      scal,
      fireLight,
      hrr,
      ratio,
      spread,
      skin,
      skinNext,
      ballSheet,
      woodTemps,
      woodCells,
      woodCols,
      flame,
      bimocq,
      ivockRhs,
      lfm,
      detA,
      detB,
      lightDims,
      light,
      light2,
      cycDirect,
      cycBounce,
      emit,
      flameNear,
      flameNearDir,
      floorNear,
      nearDims,
      frame: this.pipes ? frameGroups(this.pipes) : null,
      frameGroups,
      bg: {
        advectVel: group(p.advectVel, [P, s, v(vel[0]), v(vel[0]), v(vel[3])]),
        advectVelFwd: group(p.advectVel, [P, s, v(vel[0]), v(vel[0]), v(vel[1])]),
        advectVelBack: group(p.advectVelBack, [P, s, v(vel[0]), v(vel[1]), v(vel[2])]),
        velCombine: group(p.velCombine, [P, s, v(vel[0]), v(vel[1]), v(vel[2]), v(vel[3])]),
        arAdvect1: group(p.advectVelHalf, [P, s, v(vel[0]), v(vel[0]), v(vel[3])]),
        arAdvect2: group(p.advectVelHalf, [P, s, v(vel[1]), v(vel[2]), v(vel[3])]),
        projectReflect: group(p.projectReflect, [P, v(vel[0]), { buffer: pressure }, v(vel[1]), v(vel[2])]),
        curl: group(p.curl, [P, v(vel[3]), v(curl), v(nut)]),
        vortBefore: group(p.vortBefore, [P, s, v(vel[0]), v(vortA)]),
        vortAdvect: group(p.advFwd, [P, s, v(vel[0]), v(vortA), v(vortB)]),
        vortBack: group(p.advBwd, [P, s, v(vel[0]), v(vortB), v(vortC)]),
        ivockTarget: d.createBindGroup({
          layout: p.ivockTarget.getBindGroupLayout(0),
          entries: [P, s, v(vel[0]), v(vel[3]), { buffer: ivockRhs }].map((resource, binding) => ({ binding, resource })).concat([7, 8, 9].map((binding, k) => ({ binding, resource: v([vortA, vortB, vortC][k]) }))),
        }),

        ivockSmooth: [0, 1].map(k =>
          group(p.ivockSmooth, [P, { buffer: this.parityBufs[k] }, { buffer: ivockRhs }, { buffer: ivockPsi }]),
        ) as [GPUBindGroup, GPUBindGroup],
        ivockCurl: d.createBindGroup({
          layout: p.ivockCurl.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: P },
            { binding: 1, resource: { buffer: ivockPsi } },
            { binding: 4, resource: v(psiCurl) },
          ],
        }),
        ivockApply: group(p.ivockApply, [P, v(vel[3]), v(psiCurl), v(vel[2])]),
        forces: group(p.forces, [P, v(vel[3]), v(scal[0]), v(curl), v(vel[0]), v(nut), v(delta)]),
        divergence: withSkin(p.divergence, [P, v(vel[0]), { buffer: div }, v(scal[0]), { buffer: fireLight }, { buffer: hrr }]),
        sor: [0, 1].map(k => group(p.sor, [P, { buffer: this.parityBufs[k] }, { buffer: div }, { buffer: pressure }])) as [GPUBindGroup, GPUBindGroup],
        project: group(p.project, [P, v(vel[0]), { buffer: pressure }, v(vel[1])]),
        advFwd: group(p.advFwd, [P, s, v(vel[0]), v(scal[0]), v(scal[1])]),
        advBwd: group(p.advBwd, [P, s, v(vel[0]), v(scal[1]), v(scal[2])]),
        combine: withSkin(p.combine, [P, s, v(vel[0]), v(scal[0]), v(scal[1]), v(scal[2]), v(scal[3]), { buffer: fireLight }, { buffer: hrr }, { buffer: ratio }, v(flame)]),
        conserveSpread: group(p.conserveSpread, [P, s, v(vel[0]), { buffer: spread }]),
        conserveRatio: withSkin(p.conserveRatio, [P, v(scal[0]), { buffer: fireLight }, { buffer: hrr }, { buffer: spread }, { buffer: ratio }, s, v(vel[0])]),
        ballHeat: withSkin(p.ballHeat, [P, s, v(scal[0]), { buffer: fireLight }, { buffer: skinNext }, { buffer: ballSheet }, this.noiseTex.createView(), this.noiseSampler, { buffer: woodTemps }, { buffer: woodCells }, { buffer: woodCols }, v(flame), v(vel[0])]),
        diffuse: group(p.diffuse, [P, v(vel[0]), v(scal[3]), v(scal[1])]),
        advDetail: group(p.advDetail, [P, s, v(vel[0]), v(detA[0]), v(detB[0]), v(detA[1]), v(detB[1])]),
        fireSum: d.createBindGroup({
          layout: p.fireSum.getBindGroupLayout(0),
          entries: [
            ...[P, R, s, v(scal[0])].map((resource, binding) => ({ binding, resource })),
            ...[v(detA[0]), v(detB[0]), this.noiseTex.createView(), this.noiseSampler, { buffer: this.bbBuf }].map((resource, k) => ({ binding: 5 + k, resource })),
            { binding: 10, resource: { buffer: fireLight } },
            { binding: 12, resource: this.dummy3d.createView() },
            { binding: 13, resource: v(flame) },
            { binding: 35, resource: v(vel[0]) },
            { binding: 40, resource: { buffer: firePartial } },
            { binding: 41, resource: { buffer: hrr } },
            { binding: 42, resource: v(emit) },
          ],
        }),
        fireTotal: group(p.fireTotal, [{ buffer: fireCount }, { buffer: firePartial }, { buffer: fireLight }]),
      },
    }
    this.resetSkin()
    this.bimocqReset = true
    this.lfmStep = 0
    this.simTime = 0
    this.frame = 0
    this.lightDirty = true
    this.needsRender = true
    this.runClock = 0
    this.autoSteps = 0
    this.stepTimes = []
    this.renderScale = RENDER_SCALE_START
    this.scaleCeiling = Infinity
    this.behindMs = 0
    this.steadySince = performance.now()
    this.applySize()
  }

  // ------------------------------------------------------------------------------------------------------------
  // Controls
  // ------------------------------------------------------------------------------------------------------------
  setState(patch: Partial<PyroState>) {
    const prev = this.state
    this.state = { ...prev, ...patch }
    if (patch.resolution && patch.resolution !== prev.resolution && this.pipes) {
      this.makeGrids()
      this.layoutCamera()
    }
    // (A ball of another size is a new ball: its cells are laid out by its radius.)
    if ((this.state.ball && !prev.ball) || (this.state.ballBurns && !prev.ballBurns) || this.state.ballRadius !== prev.ballRadius) this.resetSkin()
    // The close view frames the ball by its size, and without a ball there is nothing to be close on.
    if (this.view === 'close' && (!this.state.ball || this.state.ballRadius !== prev.ballRadius)) {
      if (!this.state.ball) this.view = 'whole'
      this.layoutCamera()
    }
    if (this.state.env !== prev.env) this.syncEnv()
    // Shading changes redraw (and relight) at once, even while paused, and accumulate afresh (traced too).
    this.stateVersion++
    if (this.paused) this.fireResum = true
    this.lightDirty = true
    this.needsRender = true
    this.taaCount = 0
    this.kick()
  }

  setPaused(paused: boolean) {
    this.paused = paused
    if (paused) {
      this.fireResum = true
      this.lightDirty = true
      this.needsRender = true
      // (The running history carries on, settling now at the display's own size.)
      this.settle = 0
    }
    if (!paused) this.pt.on = false
    this.acc = 0
    this.kick()
    this.report(true)
  }

  restart() {
    if (!this.pipes) return
    this.makeGrids()
    this.kick()
    this.report(true)
  }

  setVisible(visible: boolean) {
    this.visible = visible
    if (visible) this.kick()
    else this.stop()
  }

  // The canvas follows the frame, at its device pixels; the march, at a fraction of them (see DISPLAY_DPR_MAX).
  resize(cssW: number, cssH: number, dpr: number) {
    this.cssW = cssW
    this.cssH = cssH
    this.dpr = dpr
    this.applySize()
  }

  private applySize() {
    this.sizeTo()
    this.layoutCamera()
    this.needsRender = true
    this.kick()
  }
  // The canvas at the display's own pixels, and the running march at the render scale's (never past the display's).
  private sizeTo() {
    const [w, h] = this.fit(Math.min(this.dpr, DISPLAY_DPR_MAX), MAX_DISPLAY_PIXELS)
    if (this.pinScale != null) this.renderScale = this.pinScale
    this.renderScale = Math.min(this.renderScale, this.scaleMax())
    const [mw, mh] = this.fit(Math.min(this.dpr, 1.5) * this.renderScale, w * h)
    this.marchW = Math.min(mw, w)
    this.marchH = Math.min(mh, h)
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
    }
  }
  // The frame's CSS size at a density, held to a number of pixels.
  private fit(density: number, most: number): [number, number] {
    let w = Math.max(1, Math.round(this.cssW * density))
    let h = Math.max(1, Math.round(this.cssH * density))
    if (w * h > most) {
      const k = Math.sqrt(most / (w * h))
      w = Math.max(1, Math.round(w * k))
      h = Math.max(1, Math.round(h * k))
    }
    return [w, h]
  }
  // The render scale at which the march is the display's own size.
  private scaleMax() {
    return Math.min(this.dpr, DISPLAY_DPR_MAX) / Math.min(this.dpr, 1.5)
  }
  // The paused march: the display's own size, held to PAUSED_MAX_PIXELS; and the frames a paused image settles over,
  // TAA_SETTLE at the display's size, proportionately more at fewer pixels (each display pixel then gathers fewer).
  private pausedMarch(): [number, number] {
    if (this.forceMarch) return this.forceMarch
    const w = this.canvas.width
    const h = this.canvas.height
    const [mw, mh] = this.fit(Math.min(this.dpr, DISPLAY_DPR_MAX), PAUSED_MAX_PIXELS)
    return [Math.min(mw, w), Math.min(mh, h)]
  }
  private settleFrames() {
    const [mw, mh] = this.pausedMarch()
    return PAUSED_QUICK + Math.ceil(TAA_SETTLE * Math.max(1, (this.canvas.width * this.canvas.height) / (mw * mh)))
  }
  // This frame's march: the running size, or paused (after its first PAUSED_QUICK frames) the paused size.
  private chooseMarch() {
    const quick = !this.paused || (this.settle < PAUSED_QUICK && !this.forceMarch)
    this.frameMarch = quick ? [this.marchW, this.marchH] : this.pausedMarch()
  }

  private layoutCamera() {
    const next = this.cameraFor(this.closeOn)
    if (!next) return
    // A new camera starts the history afresh, and the path-traced reference (it no longer shows the same points).
    if (camMoved(next.cam, this.cam)) {
      this.camVersion++
      this.taaCount = 0
      this.needsRender = true
    }
    this.cam = next.cam
    this.framing = next.framing
  }

  // A camera a little below the plume's middle, looking along -z, as close as framing the domain's height allows, and
  // closer still on wide frames, until the domain spans the frame's width (smoke then leaves through the frame's
  // edges, never a side of the box in view). Close on the ball (aimed at `on`), the same camera moved in (closePos);
  // the lab's framing stays the lab's (the reference set is placed by it).
  private cameraFor(on: Vec3) {
    if (!this.grids) return null
    const [nx, ny] = this.grids.n
    const dx = DOMAIN_WIDTH / nx
    const h = ny * dx
    const aspect = this.canvas.width / Math.max(1, this.canvas.height)
    const tanY = Math.tan(FOV_Y / 2)
    const dist = Math.min((h * 0.525) / tanY, (DOMAIN_WIDTH * 0.49) / (tanY * aspect))
    const target: Vec3 = [0, h * 0.47, 0]
    const lab: Vec3 = [0, h * 0.3, dist]
    const fwd = norm(sub(target, lab))
    const right = norm(cross(fwd, [0, 1, 0]))
    const up = cross(right, fwd)
    const pos = this.closeUp() ? this.closePos(on, lab, fwd, right, up, tanY, aspect) : lab
    return { cam: { pos, fwd, right, up, tanY, aspect }, framing: { z: lab[2], tanY, aspect } }
  }

  private closeUp() {
    return this.view === 'close' && this.state.ball
  }

  // The close camera: back along the lab camera's direction from a little above `on`, where the ball was when it aimed,
  // far enough that the ball's diameter takes CLOSE_FILL of the frame (its height, or its width were the frame ever
  // taller than wide). Held where its frame shows no more of the box than the lab's does (where its corners meet the
  // box's middle plane: within 49% of its width either side, no higher than the lab frame's top), and above the floor.
  private closePos(on: Vec3, lab: Vec3, fwd: Vec3, right: Vec3, up: Vec3, tanY: number, aspect: number): Vec3 {
    const r = this.state.ballRadius
    const dist = r / (CLOSE_FILL * tanY * Math.min(aspect, 1))
    const from = (aim: Vec3): Vec3 => [aim[0] - fwd[0] * dist, aim[1] - fwd[1] * dist, aim[2] - fwd[2] * dist]
    // Where the frame's corners, seen from p, meet the plane z = 0 (x, y).
    const corners = (p: Vec3) =>
      [-1, 1].flatMap(sx =>
        [-1, 1].map(sy => {
          const d = [0, 1, 2].map(i => fwd[i] + right[i] * sx * tanY * aspect + up[i] * sy * tanY)
          const t = -p[2] / d[2]
          return [p[0] + d[0] * t, p[1] + d[1] * t]
        }),
      )
    const labTop = Math.max(...corners(lab).map(c => c[1]))
    // (The aim moves only in x and y, which moves the corners with it: they are found once, about an aim at the origin.)
    const own = corners(from([0, 0, 0]))
    const half = Math.max(...own.map(c => Math.abs(c[0])))
    const above = Math.max(...own.map(c => c[1]))
    const side = Math.max(DOMAIN_WIDTH * 0.49 - half, 0)
    const x = Math.max(-side, Math.min(side, on[0]))
    const y = Math.max(Math.min(on[1] + CLOSE_LIFT * r, labTop - above), CLOSE_MIN_HEIGHT + fwd[1] * dist)
    return from([x, y, 0])
  }

  // The view: the whole fire, or close on the ball where it is now (only with a ball). Whether the camera moved.
  setView(view: PyroView) {
    if (view === 'close' && !this.state.ball) return false
    const was = this.camVersion
    this.view = view
    if (view === 'close') this.closeOn = [...this.ballPos]
    this.layoutCamera()
    this.kick()
    this.report(true)
    return this.camVersion !== was
  }

  // Close up, after a drag or a run of nudges: whether the ball has gone more than CLOSE_REAIM of the way from the
  // frame's middle to its edge (or out of the frame) and aiming at it again would move the camera (held at the box's
  // limits, it may not). The lab asks first, so it fades the frame only for a cut that happens.
  wouldReframe() {
    if (!this.closeUp()) return false
    const c = this.cam
    const d = sub(this.ballPos, c.pos)
    const z = dot(d, c.fwd)
    if (z > 0 && Math.abs(dot(d, c.right) / (z * c.tanY * c.aspect)) <= CLOSE_REAIM && Math.abs(dot(d, c.up) / (z * c.tanY)) <= CLOSE_REAIM) return false
    const next = this.cameraFor(this.ballPos)
    return !!next && camMoved(next.cam, c)
  }

  // Aims again at the ball, if wouldReframe(). Whether the camera moved.
  reframe() {
    return this.wouldReframe() && this.setView('close')
  }

  // Resolves once this many more frames have been drawn, or after maxMs (the loop may be stopped, or paused and
  // settled): the lab's cut between views waits on it.
  framesDrawn(frames: number, maxMs: number) {
    return new Promise<void>(resolve => {
      const wait = { at: this.renderFrame + frames, done: resolve }
      this.drawWaits.push(wait)
      setTimeout(() => {
        this.drawWaits = this.drawWaits.filter(w => w !== wait)
        resolve()
      }, maxMs)
    })
  }

  // A drag on the image: ndc (x, y in -1..1, y up) and how far it moved (ndc units) in dtMs. The stir pushes the
  // air, at the drag's speed in the plane facing the camera, around the point where the pointer's ray passes
  // closest to the plume's axis (the vertical through the source).
  // The ray through a point of the frame (ndc, -1 to 1).
  private rayAt(x: number, y: number): { o: Vec3; dir: Vec3 } {
    const c = this.cam
    const dir = norm([
      c.fwd[0] + c.right[0] * x * c.tanY * c.aspect + c.up[0] * y * c.tanY,
      c.fwd[1] + c.right[1] * x * c.tanY * c.aspect + c.up[1] * y * c.tanY,
      c.fwd[2] + c.right[2] * x * c.tanY * c.aspect + c.up[2] * y * c.tanY,
    ])
    return { o: c.pos, dir }
  }

  // Moves the source along the floor to under the pointer; returns its new x (m), or null off the floor.
  sourceTo(x: number, y: number): number | null {
    if (!this.grids) return null
    const { o, dir } = this.rayAt(x, y)
    if (dir[1] > -1e-4) return null
    const t = -o[1] / dir[1]
    const half = DOMAIN_WIDTH / 2 - this.state.srcRadius - 0.15
    const sx = Math.max(-half, Math.min(half, o[0] + dir[0] * t))
    this.setState({ srcX: sx })
    return sx
  }

  // Moves the collision ball to under the pointer, on the plane through the domain's middle, and gives it the
  // velocity of the move, so it pushes the gas.
  // Sets the ball down beside the fire, ready to be dragged into it.
  // Puts the ball at (x, y) in the frame's plane, at rest (the device check holds it in the fire).
  placeBall(x: number, y: number) {
    this.ballPos = [x, y, 0]
    this.ballVel = [0, 0, 0]
    this.lightDirty = true
    this.needsRender = true
    this.kick()
  }

  ballBeside() {
    if (!this.grids) return
    const s = this.state
    const half = DOMAIN_WIDTH / 2 - s.ballRadius - 0.05
    const x = s.srcX + s.srcRadius + s.ballRadius + 0.3
    this.ballPos = [x <= half ? x : s.srcX - s.srcRadius - s.ballRadius - 0.3, s.ballRadius + 0.3, 0]
    this.ballVel = [0, 0, 0]
    this.lightDirty = true
    this.needsRender = true
    this.taaCount = 0
    this.kick()
  }

  ballTo(x: number, y: number) {
    if (!this.grids) return
    const { o, dir } = this.rayAt(x, y)
    if (Math.abs(dir[2]) < 1e-4) return
    const t = -o[2] / dir[2]
    const r = this.state.ballRadius
    const [nx, ny] = this.grids.n
    const height = (ny * DOMAIN_WIDTH) / nx
    const half = DOMAIN_WIDTH / 2 - r - 0.05
    const next: Vec3 = [Math.max(-half, Math.min(half, o[0] + dir[0] * t)), Math.max(r, Math.min(height - r - 0.1, o[1] + dir[1] * t)), 0]
    const now = performance.now()
    const secs = Math.max((now - this.ballDragAt) / 1000, 1 / 120)
    // A drag that has just begun has no speed yet; otherwise the move's, up to 6 m/s.
    let vel: Vec3 = [0, 0, 0]
    if (now - this.ballDragAt < 250) {
      vel = next.map((v, i) => (v - this.ballPos[i]) / secs) as Vec3
      const speed = Math.hypot(...vel)
      if (speed > 6) vel = vel.map(v => (v * 6) / speed) as Vec3
    }
    this.ballVel = vel
    this.ballPos = next
    this.ballDragAt = now
    this.lightDirty = true
    this.needsRender = true
    this.kick()
  }

  // Moves the ball by (dx, dy) metres in the frame's plane, as a drag would (the arrow keys).
  nudgeBall(dx: number, dy: number) {
    if (!this.grids) return
    const r = this.state.ballRadius
    const [nx, ny] = this.grids.n
    const height = (ny * DOMAIN_WIDTH) / nx
    const half = DOMAIN_WIDTH / 2 - r - 0.05
    const next: Vec3 = [Math.max(-half, Math.min(half, this.ballPos[0] + dx)), Math.max(r, Math.min(height - r - 0.1, this.ballPos[1] + dy)), 0]
    this.ballVel = next.map((v, i) => (v - this.ballPos[i]) / 0.1) as Vec3
    this.ballPos = next
    this.ballDragAt = performance.now()
    this.lightDirty = true
    this.needsRender = true
    this.kick()
  }

  stirAt(x: number, y: number, dxNdc: number, dyNdc: number, dtMs: number) {
    if (!this.grids) return
    const c = this.cam
    const dir = norm([
      c.fwd[0] + c.right[0] * x * c.tanY * c.aspect + c.up[0] * y * c.tanY,
      c.fwd[1] + c.right[1] * x * c.tanY * c.aspect + c.up[1] * y * c.tanY,
      c.fwd[2] + c.right[2] * x * c.tanY * c.aspect + c.up[2] * y * c.tanY,
    ])
    const o = c.pos
    const t = Math.max(0, -(o[0] * dir[0] + o[2] * dir[2]) / Math.max(dir[0] * dir[0] + dir[2] * dir[2], 1e-6))
    // World metres per ndc unit at the domain's centre plane (close up, at the stir's own distance).
    const [nx, ny] = this.grids.n
    const dist = this.closeUp() ? Math.max(t, 0.3) : Math.hypot(c.pos[0], c.pos[1] - (ny * DOMAIN_WIDTH) / nx * 0.47, c.pos[2])
    const mx = dxNdc * dist * c.tanY * c.aspect
    const my = dyNdc * dist * c.tanY
    const secs = Math.max(dtMs, 8) / 1000
    const gain = 0.45
    let vel: Vec3 = [(c.right[0] * mx + c.up[0] * my) / secs * gain, (c.right[1] * mx + c.up[1] * my) / secs * gain, (c.right[2] * mx + c.up[2] * my) / secs * gain]
    const speed = Math.hypot(...vel)
    if (speed > 3) vel = vel.map(v => (v * 3) / speed) as Vec3
    this.stir = { centre: [o[0] + dir[0] * t, o[1] + dir[1] * t, o[2] + dir[2] * t], vel }
    this.kick()
  }

  // ------------------------------------------------------------------------------------------------------------
  // The loop: fixed steps against the wall clock (at most two a frame; a slow device runs in slow motion rather
  // than taking bigger, less stable steps), and a render after each frame that stepped or changed.
  // ------------------------------------------------------------------------------------------------------------
  private kick() {
    if (this.raf || this.disposed || !this.visible || !this.pipes) return
    this.lastTick = performance.now()
    this.steadySince = this.lastTick
    this.behindMs = 0
    this.raf = requestAnimationFrame(this.tick)
  }

  private stop() {
    if (this.raf) cancelAnimationFrame(this.raf)
    this.raf = 0
    this.report(true)
  }

  private tick = (now: number) => {
    this.raf = 0
    if (this.disposed || !this.visible || !this.grids) return
    // A long gap (the page busy elsewhere) breaks the steady running the simulation's rate is judged on.
    if (now - this.lastTick > 250) this.steadySince = now
    const elapsed = Math.min(now - this.lastTick, 100)
    this.lastTick = now
    // The GPU still has two frames of work: queue nothing more (the page keeps its frame rate; a GPU that cannot
    // keep up runs the simulation in slow motion).
    if (this.inFlight >= MAX_IN_FLIGHT) {
      this.raf = requestAnimationFrame(this.tick)
      return
    }
    let steps = 0
    if (!this.paused) {
      this.runClock += elapsed / 1000
      this.acc += elapsed / 1000
      while (this.acc >= SIM_DT && steps < MAX_STEPS_PER_FRAME) {
        this.step()
        this.acc -= SIM_DT
        steps++
      }
      if (this.acc > SIM_DT * MAX_STEPS_PER_FRAME) this.acc = 0
    }
    if (steps > 0 || this.needsRender) {
      this.render()
      this.inFlight++
      this.device.queue.onSubmittedWorkDone().then(() => {
        this.inFlight--
      })
    }
    if (this.drawWaits.length) this.drawWaits = this.drawWaits.filter(w => (this.renderFrame >= w.at ? (w.done(), false) : true))
    if (!this.paused) {
      this.frameTimes.push(now)
      this.adapt()
    }
    this.report(false)
    // Keep going while simulating; when paused, until nothing changes.
    if (!this.paused || this.needsRender) this.raf = requestAnimationFrame(this.tick)
  }

  private step() {
    const g = this.grids!
    const p = this.solver!
    this.writeSim()
    const enc = this.device.createCommandEncoder()
    // BiMocq2 starts afresh when it has just been switched on, or on a new grid.
    const bimNow = this.bimocqOn()
    if (bimNow) this.bimocqBegin(enc)
    this.bimocqWas = bimNow
    // The wooden ball heats (or cools) first, so this step's fire takes in its volatiles: every WOOD_EVERY steps, a
    // step that long (its surface's thinnest cell takes a second or two to answer, and its solve is implicit).
    if (this.ballBurning() && this.woodDue()) {
      const skinPass = enc.beginComputePass()
      skinPass.setPipeline(p.ballHeat)
      skinPass.setBindGroup(0, g.bg.ballHeat)
      skinPass.dispatchWorkgroups(SKIN_TEXELS / 64)
      skinPass.end()
      enc.copyBufferToBuffer(g.skinNext, 0, g.skin, 0, SKIN_TEXELS * 16)
      this.woodStepped()
    }
    const [nx, ny, nz] = g.n
    const cv: Vec3 = [Math.ceil((nx + 1) / 4), Math.ceil((ny + 1) / 4), Math.ceil((nz + 1) / 4)]
    const cc: Vec3 = [Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)]
    const run = (pass: GPUComputePassEncoder, pipe: GPUComputePipeline, bg: GPUBindGroup, wg: Vec3) => {
      pass.setPipeline(pipe)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(wg[0], wg[1], wg[2])
    }
    // Velocity: advect (into vel[3]), forces (into vel[0]), project (into vel[1]). Advection-reflection advects half a
    // step, projects, reflects, advects the reflection the second half and projects again. Leapfrog flow maps advect
    // the half-step velocity two back through the last (2c), the cycle's first two steps over half a step and one.
    const scheme = this.velScheme()
    const lfm = scheme === 'lfm'
    // A cycle starts afresh from the current velocity when LFM has just been switched on.
    if (lfm && !this.lfmWas) this.lfmStep = 0
    const k = this.lfmStep
    if (lfm && k === 0) enc.copyTextureToTexture({ texture: g.vel[0] }, { texture: g.lfm.u0[0] }, [nx + 1, ny + 1, nz + 1])
    let pass = enc.beginComputePass()
    if (lfm) {
      if (k === 0) run(pass, p.advectVelHalf, g.bg.arAdvect1, cv)
      else if (k === 1) run(pass, p.advectVel, g.bg.advectVel, cv)
      else run(pass, p.advectVelDouble, g.lfm.advect[k - 2], cv)
    } else if (scheme === 'ar') {
      run(pass, p.advectVelHalf, g.bg.arAdvect1, cv)
    } else if (scheme === 'mac') {
      run(pass, p.advectVel, g.bg.advectVelFwd, cv)
      run(pass, p.advectVelBack, g.bg.advectVelBack, cv)
      run(pass, p.velCombine, g.bg.velCombine, cv)
    } else {
      run(pass, p.advectVel, g.bg.advectVel, cv)
    }
    if (this.ivockOn()) {
      this.ivock(pass, run, cv, cc)
      pass.end()
      enc.copyTextureToTexture({ texture: g.vel[2] }, { texture: g.vel[3] }, [nx + 1, ny + 1, nz + 1])
      pass = enc.beginComputePass()
    }
    run(pass, p.curl, g.bg.curl, cc)
    run(pass, p.forces, g.bg.forces, cv)
    run(pass, p.divergence, g.bg.divergence, cc)
    this.solvePressure(pass)
    if (scheme === 'ar') {
      run(pass, p.projectReflect, g.bg.projectReflect, cv)
      run(pass, p.advectVelHalf, g.bg.arAdvect2, cv)
      pass.end()
      enc.copyTextureToTexture({ texture: g.vel[3] }, { texture: g.vel[0] }, [nx + 1, ny + 1, nz + 1])
      pass = enc.beginComputePass()
      run(pass, p.divergence, g.bg.divergence, cc)
      this.solvePressure(pass)
    }
    run(pass, p.project, g.bg.project, cv)
    if (lfm) run(pass, p.lfmCellVel, g.lfm.cellVel[k], cc)
    pass.end()
    enc.copyTextureToTexture({ texture: g.vel[1] }, { texture: g.vel[0] }, [nx + 1, ny + 1, nz + 1])
    if (lfm) enc.copyTextureToTexture({ texture: g.vel[1] }, { texture: g.lfm.ring[k] }, [nx + 1, ny + 1, nz + 1])
    // A fire keeps its mass through the advection (the scalars' ratios, 7b).
    const fire = this.state.sourceType === 'fire'
    if (fire) enc.clearBuffer(g.spread)
    pass = enc.beginComputePass()
    if (fire) {
      run(pass, p.conserveSpread, g.bg.conserveSpread, cc)
      run(pass, p.conserveRatio, g.bg.conserveRatio, cc)
    }
    const bim = this.bimocqOn()
    if (bim) {
      run(pass, p.bimocqMaps, g.bimocq.maps[this.bimocqParity], cc)
      run(pass, p.bimocqRecon, g.bimocq.recon[this.bimocqParity], cc)
    } else {
      run(pass, p.advFwd, g.bg.advFwd, cc)
      if (this.macScalars()) run(pass, p.advBwd, g.bg.advBwd, cc)
    }
    run(pass, p.combine, g.bg.combine, cc)
    run(pass, p.diffuse, g.bg.diffuse, cc)
    // The upres grid refills from the coarse density whenever it starts (a new source, or switched on).
    const upActive = this.upresActive()
    if (upActive && !this.upresWasActive) this.upresInit = true
    this.upresWasActive = upActive
    const up = upActive ? this.ensureUpres() : null
    if (up) {
      run(pass, p.upres, up.group, [Math.ceil((2 * nx) / 4), Math.ceil((2 * ny) / 4), Math.ceil((2 * nz) / 4)])
      run(pass, p.upresCoarse, up.coarseGroup, cc) // (for the light pass)
    }
    run(pass, p.advDetail, g.bg.advDetail, cc)
    pass.end()
    if (up) {
      enc.copyTextureToTexture({ texture: up.tex[1] }, { texture: up.tex[0] }, [2 * nx, 2 * ny, 2 * nz])
      enc.copyBufferToBuffer(up.held[1], 0, up.held[0], 0, up.held[0].size)
      enc.clearBuffer(up.held[1])
    }
    enc.copyTextureToTexture({ texture: g.scal[1] }, { texture: g.scal[0] }, [nx, ny, nz])
    if (bim) this.bimocqAfter(enc, cc)
    enc.copyTextureToTexture({ texture: g.detA[1] }, { texture: g.detA[0] }, [nx, ny, nz])
    enc.copyTextureToTexture({ texture: g.detB[1] }, { texture: g.detB[0] }, [nx, ny, nz])
    // Leapfrog flow maps, once the scalars have moved with the step's half-step velocity: the forward map on, and at
    // the cycle's end the velocity rebuilt from the maps.
    if (lfm) {
      this.lfmMarch(enc, run, cv)
      if (k === LFM_N - 1) this.lfmReinit(enc, run, cv, cc)
      this.lfmStep = (k + 1) % LFM_N
    }
    this.lfmWas = lfm
    // The fire's light from the new scalars (zero without flames). A smoke source's sums to nothing (FIRE_SUM reads
    // only a fire's scalars), so it is cleared once rather than summed every step.
    if (fire) {
      pass = enc.beginComputePass()
      run(pass, p.fireSum, g.bg.fireSum, cc)
      run(pass, p.fireTotal, g.bg.fireTotal, [1 + FIRE_BANDS, 1, 1])
      pass.end()
    } else if (this.fireLit) {
      enc.clearBuffer(g.fireLight)
    }
    this.fireLit = fire
    this.device.queue.submit([enc.finish()])
    this.simTime += SIM_DT
    this.frame++
    this.stir = null
    this.lightDirty = true
    this.stepTimes.push(performance.now())
    if (this.runClock >= AUTO_RES_FROM_S && this.runClock < AUTO_RES_TO_S) this.autoSteps++
  }

  // The pressure solve: multigrid V-cycles (default) or red-black SOR sweeps, both warm started from the last step.
  private solvePressure(pass: GPUComputePassEncoder) {
    const g = this.grids!
    const p = this.solver!
    const [nx, ny, nz] = g.n
    const dispatch = (pipe: GPUComputePipeline, bg: GPUBindGroup, wg: Vec3) => {
      pass.setPipeline(pipe)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(wg[0], wg[1], wg[2])
    }
    if (this.state.pressure === 'sor') {
      const cs: Vec3 = [Math.ceil(nx / 2 / 16), Math.ceil(ny / 4), nz]
      for (let i = 0; i < this.state.pressureIters; i++) {
        dispatch(p.sor, g.bg.sor[0], cs)
        dispatch(p.sor, g.bg.sor[1], cs)
      }
      return
    }
    this.mgCycles(pass, this.state.vcycles)
  }

  // Multigrid V-cycles over the levels.
  private mgCycles(pass: GPUComputePassEncoder, count: number) {
    const g = this.grids!
    const p = this.solver!
    const dispatch = (pipe: GPUComputePipeline, bg: GPUBindGroup, wg: Vec3) => {
      pass.setPipeline(pipe)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(wg[0], wg[1], wg[2])
    }
    const L = g.levels
    const cells = (m: Vec3): Vec3 => [Math.ceil(m[0] / 4), Math.ceil(m[1] / 4), Math.ceil(m[2] / 4)]
    const smooth = (l: number, sweeps: number) => {
      const m = L[l].n
      const wg: Vec3 = [Math.ceil(m[0] / 2 / 16), Math.ceil(m[1] / 4), m[2]]
      const groups = L[l].smooth
      for (let i = 0; i < sweeps; i++) {
        dispatch(p.mgSmooth, groups[0], wg)
        dispatch(p.mgSmooth, groups[1], wg)
      }
    }
    const vcycle = (l: number) => {
      if (l === L.length - 1) {
        smooth(l, MG_COARSE)
        return
      }
      smooth(l, MG_PRE)
      dispatch(p.mgResidual, L[l].residual!, cells(L[l].n))
      dispatch(p.mgRestrict, L[l + 1].restrictTo!, cells(L[l + 1].n))
      vcycle(l + 1)
      dispatch(p.mgProlong, L[l + 1].prolongFrom!, cells(L[l].n))
      smooth(l, MG_POST)
    }
    for (let k = 0; k < count; k++) vcycle(0)
  }

  // The scalars' scheme: BiMocq2 for a smoke source when chosen; a fire given it uses MacCormack.
  private bimocqOn() {
    return this.state.advection === 'bimocq' && this.state.sourceType === 'smoke'
  }
  private macScalars() {
    return this.state.advection === 'mac' || (this.state.advection === 'bimocq' && !this.bimocqOn())
  }

  // BiMocq2 at the start of a step that uses it: afresh if need be (identity maps, the field as the snapshot, nothing
  // accumulated, no previous level).
  private bimocqBegin(enc: GPUCommandEncoder) {
    const g = this.grids!
    const b = g.bimocq
    const [nx, ny, nz] = g.n
    const bytes = nx * ny * nz * 16
    if (!this.bimocqReset && this.bimocqWas) return
    this.bimocqReset = false
    this.bimocqParity = 0
    this.bimocqSteps = 0
    this.bimocqHasPrev = false
    enc.copyBufferToBuffer(b.identity, 0, b.x[0], 0, bytes)
    enc.copyBufferToBuffer(b.identity, 0, b.y[0], 0, bytes)
    enc.copyTextureToTexture({ texture: b.zero }, { texture: b.dCurr[0] }, [nx, ny, nz])
    enc.copyTextureToTexture({ texture: g.scal[0] }, { texture: b.phiCurr }, [nx, ny, nz])
    this.device.queue.writeBuffer(b.uniform, 0, new Float32Array([0, 0, 0, 0]))
  }

  // BiMocq2 after the step's sources and mixing: what they changed, accumulated; the maps' parity flips; every
  // BIMOCQ_REINIT steps a re-initialisation (the current level becomes the previous one, the field the snapshot).
  private bimocqAfter(enc: GPUCommandEncoder, cc: Vec3) {
    const g = this.grids!
    const b = g.bimocq
    const p = this.solver!
    const [nx, ny, nz] = g.n
    const bytes = nx * ny * nz * 16
    const pass = enc.beginComputePass()
    pass.setPipeline(p.bimocqAccum)
    pass.setBindGroup(0, b.accum[this.bimocqParity])
    pass.dispatchWorkgroups(cc[0], cc[1], cc[2])
    pass.end()
    this.bimocqParity = 1 - this.bimocqParity
    if (++this.bimocqSteps < BIMOCQ_REINIT) return
    this.bimocqSteps = 0
    const cur = this.bimocqParity
    enc.copyBufferToBuffer(b.x[cur], 0, b.xPrev, 0, bytes)
    enc.copyBufferToBuffer(b.identity, 0, b.x[cur], 0, bytes)
    enc.copyBufferToBuffer(b.identity, 0, b.y[cur], 0, bytes)
    enc.copyTextureToTexture({ texture: b.phiCurr }, { texture: b.phiPrev }, [nx, ny, nz])
    enc.copyTextureToTexture({ texture: g.scal[0] }, { texture: b.phiCurr }, [nx, ny, nz])
    enc.copyTextureToTexture({ texture: b.dCurr[cur] }, { texture: b.dPrev }, [nx, ny, nz])
    enc.copyTextureToTexture({ texture: b.zero }, { texture: b.dCurr[cur] }, [nx, ny, nz])
    if (!this.bimocqHasPrev) {
      this.bimocqHasPrev = true
      this.device.queue.writeBuffer(b.uniform, 0, new Float32Array([1, 0, 0, 0]))
    }
  }

  // IVOCK runs for a smoke source (see wgsl.ts), with MacCormack or semi-Lagrangian velocity.
  private ivockOn() {
    const v = this.velScheme()
    return this.state.ivock && v !== 'ar' && v !== 'lfm' && this.state.sourceType === 'smoke'
  }

  // The velocity's scheme this step: leapfrog flow maps for a smoke source when chosen (a fire given them uses
  // MacCormack: the maps assume incompressible flow, and a fire's burning gas expands).
  private velScheme() {
    const v = this.state.velocity
    return v === 'lfm' && this.state.sourceType !== 'smoke' ? 'mac' : v
  }

  // Leapfrog flow maps after each step's projection: the forward map and its Jacobian columns marched through the
  // step's half-step velocity, gathering its forces into the cycle's initial velocity (2c).
  private lfmMarch(enc: GPUCommandEncoder, run: (pass: GPUComputePassEncoder, pipe: GPUComputePipeline, bg: GPUBindGroup, wg: Vec3) => void, cv: Vec3) {
    const g = this.grids!
    const [nx, ny, nz] = g.n
    const pass = enc.beginComputePass()
    run(pass, this.solver!.lfmMarchFwd, g.lfm.marchFwd[this.lfmStep], cv)
    pass.end()
    enc.copyTextureToTexture({ texture: g.lfm.u0[1] }, { texture: g.lfm.u0[0] }, [nx + 1, ny + 1, nz + 1])
  }

  // At the cycle's end: the backward map; the impulse pulled back along it; the round trip's error taken off; projected
  // (cold started, with LFM_PROJECT_CYCLES V-cycles, the step's pressure kept aside for the next step's warm start);
  // the result is the velocity, and the next cycle's start.
  private lfmReinit(enc: GPUCommandEncoder, run: (pass: GPUComputePassEncoder, pipe: GPUComputePipeline, bg: GPUBindGroup, wg: Vec3) => void, cv: Vec3, cc: Vec3) {
    const g = this.grids!
    const p = this.solver!
    const L = g.lfm
    const [nx, ny, nz] = g.n
    const bytes = nx * ny * nz * 4
    let pass = enc.beginComputePass()
    run(pass, p.lfmMarchBwd, L.marchBwd, cv)
    run(pass, p.lfmPull[0], L.pull[0], cv)
    run(pass, p.lfmPull[1], L.pull[1], cv)
    run(pass, p.lfmPull[2], L.pull[2], cv)
    pass.end()
    enc.copyBufferToBuffer(L.pressure, 0, L.pressureSave, 0, bytes)
    enc.clearBuffer(L.pressure)
    pass = enc.beginComputePass()
    run(pass, p.divergence, g.bg.divergence, cc)
    this.mgCycles(pass, LFM_PROJECT_CYCLES)
    run(pass, p.project, g.bg.project, cv)
    pass.end()
    enc.copyTextureToTexture({ texture: g.vel[1] }, { texture: g.vel[0] }, [nx + 1, ny + 1, nz + 1])
    enc.copyBufferToBuffer(L.pressureSave, 0, L.pressure, 0, bytes)
  }

  // IVOCK, after the velocity is advected (into vel[3]); the corrected velocity goes to vel[2] for the caller to copy
  // back. The vorticity before advection, advected; the loss against the advected velocity's curl; the three
  // potentials; their curl added.
  private ivock(pass: GPUComputePassEncoder, run: (pass: GPUComputePassEncoder, pipe: GPUComputePipeline, bg: GPUBindGroup, wg: Vec3) => void, cv: Vec3, cc: Vec3) {
    const g = this.grids!
    const p = this.solver!
    run(pass, p.vortBefore, g.bg.vortBefore, cc)
    run(pass, p.advFwd, g.bg.vortAdvect, cc)
    if (this.velScheme() === 'mac') run(pass, p.advBwd, g.bg.vortBack, cc)
    run(pass, p.ivockTarget, g.bg.ivockTarget, cc)
    const [nx, ny, nz] = g.n
    const sweep: Vec3 = [Math.ceil(nx / 2 / 16), Math.ceil(ny / 4), nz]
    for (let i = 0; i < IVOCK_SWEEPS; i++) {
      run(pass, p.ivockSmooth, g.bg.ivockSmooth[0], sweep)
      run(pass, p.ivockSmooth, g.bg.ivockSmooth[1], sweep)
    }
    run(pass, p.ivockCurl, g.bg.ivockCurl, cc)
    run(pass, p.ivockApply, g.bg.ivockApply, cv)
  }

  private writeSim() {
    const s = this.state
    const g = this.grids!
    const [nx, ny, nz] = g.n
    const dx = DOMAIN_WIDTH / nx
    const u = new Uint32Array(this.simData)
    const f = new Float32Array(this.simData)
    u.set([nx, ny, nz, this.frame], 0)
    f.set([dx, SIM_DT, this.simTime, this.velScheme() === 'mac' ? 1 : 0], 4)
    // A smoke source's dome rests on the floor (it is 2.5 times wider than tall); a fire's bed is a disc on the floor.
    f.set([s.srcX, s.srcRadius / 2.5 + dx, 0, s.srcRadius], 8)
    f.set([s.srcDensity, s.srcTemp, s.srcSpeed, s.srcNoise], 12)
    // Buoyancy: g (T - T_ambient) / T_ambient per kelvin of temperature rise, times the control (1 = physical).
    // Turbulence: m/s2 for a smoke source; for a fire, the turbulence model (physics.ts) times the control.
    const turb = s.sourceType === 'fire' ? s.turbulence * fireTurbulence(s.heatRelease, s.turbScale) : s.turbulence
    f.set([(s.buoyancy * G) / T_AMBIENT, this.woodClock, s.vorticity, turb], 16)
    f.set([s.turbScale, s.wind, 0, 0.05], 20)
    f.set([s.cooling, s.dissipation, SOR_OMEGA, SPEED_LIMIT], 24)
    const st = this.stir
    f.set([...(st ? st.centre : [0, 0, 0]), STIR_RADIUS], 28)
    f.set([...(st ? st.vel : [0, 0, 0]), st ? 1 : 0], 32)
    // Leapfrog flow maps: the step within the cycle, and how many steps its forces act over (2c).
    const lfm = this.velScheme() === 'lfm'
    const k = lfm && this.lfmWas ? this.lfmStep : 0
    f.set([lfm ? k : 0, lfm ? (k === 0 ? 0.5 : k === 1 ? 1 : 2) : 0, this.ivockOn() || lfm ? 1 : 0, this.macScalars() ? 1 : 0], 36)
    f.set([(-nx * dx) / 2, 0, (-nz * dx) / 2, 0], 40)
    // Each upres layer resets when the simulation clock crosses its period (layer B half a period later).
    const t0 = this.simTime / DETAIL_PERIOD
    const t1 = (this.simTime + SIM_DT) / DETAIL_PERIOD
    f.set([Math.floor(t1) > Math.floor(t0) ? 1 : 0, Math.floor(t1 + 0.5) > Math.floor(t0 + 0.5) ? 1 : 0, WOOD_EVERY * this.woodSpeed, this.woodTest], 44)
    // A fire bed gives off fuel at the rate that releases the chosen heat. Fuel and air mix (and so burn) in
    // sqrt(2 dx / g), the buoyant mixing time FDS uses at a cell's scale. The bed's nominal area (a disc, thinned by
    // the source noise) spreads the fuel until the area is measured.
    const fire = s.sourceType === 'fire'
    f.set([fire ? 1 : 0, fire ? (s.heatRelease * 1000) / FUEL.heatPerFuel : 0, s.sootYield, s.radiantFraction], 48)
    const bed = Math.PI * s.srcRadius ** 2 * (1 - 0.5 * s.srcNoise)
    f.set([s.expansion, Math.sqrt((2 * dx) / G), bed, this.woodParity], 52)
    // The collision ball, easing to rest once the visitor lets go.
    if (performance.now() - this.ballDragAt > 120) this.ballVel = this.ballVel.map(v => v * 0.6) as Vec3
    const radius = s.ball ? s.ballRadius : 0
    f.set([...this.ballPos, radius], 56)
    f.set([...this.ballVel, s.ball && s.ballBurns ? 1 : 0], 60)
    const set = this.referenceSet(s)
    f.set(set, 64)
    // The wooden ball as it has worn back (the grey ball keeps its shape).
    const wear = s.ball && s.ballBurns ? this.wear : NO_WEAR
    f.set(wear, 84)
    f[95] = this.woodAir
    this.device.queue.writeBuffer(this.simBuf, 0, this.simData)
    this.device.queue.writeBuffer(this.ballBuf, 0, new Float32Array([...this.ballPos, radius, (-nx * dx) / 2, 0, (-nz * dx) / 2, dx, ...set, ...wear]))
  }

  private writeRender() {
    const s = this.state
    const c = this.cam
    const az = (s.lightAz * Math.PI) / 180
    const el = (s.lightEl * Math.PI) / 180
    // Toward the sun; azimuth 0 puts it behind the camera (+z), positive swings it to camera right.
    let sun: Vec3 = [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)]
    let sunLux = SUN_COLOUR.map(v => v * luxOfEV(s.sunEV))
    // An HDR environment (environment.ts), turned so its light faces the light azimuth: its key takes the sun's place
    // (at the map's own elevation), the rest lights through spherical harmonics, all scaled so the environment's
    // illuminance on open ground, key included, meets the Sky EV.
    const e = this.envLight
    const envData = new Float32Array(40)
    if (e) {
      const k = luxOfEV(s.skyEV) / e.illuminance
      const rho = Math.atan2(e.anchor[0], e.anchor[2]) - az
      const cr = Math.cos(rho)
      const sr = Math.sin(rho)
      envData.set([1, cr, sr, k], 0)
      for (let m = 0; m < 9; m++) envData.set([e.sh[3 * m] * k, e.sh[3 * m + 1] * k, e.sh[3 * m + 2] * k, 0], 4 + 4 * m)
      if (e.key) {
        const [x, y, z] = e.key.dir
        sun = [x * cr - z * sr, y, z * cr + x * sr]
        sunLux = e.key.irradiance.map(v => v * k)
      } else sunLux = [0, 0, 0]
    }
    // The path-traced reference's lights (PATH_TRACE): each a cone (toward it, the cosine of its half-angle, 0 for
    // none) and its illuminance at normal incidence. The sun's disc, or an HDR map's key in the cone it was taken from;
    // with the rig (below), its three softboxes at their solid angles.
    const pl = this.ptLights.fill(0)
    const keyCos = e ? (e.key ? Math.cos((e.key.cone * Math.PI) / 180) : 0) : Math.cos(SUN_HALF_ANGLE)
    pl.set([...sun, keyCos, ...sunLux, 0], 0)
    // The floor's albedo: the lab's own, or the Lookdev lab's cyc in charcoal (its studio floor) or gray.
    const floorAlbedo = s.background === 'charcoal' ? 0.05 : s.background === 'gray' ? 0.18 : 0.14
    // The Lookdev lab's softbox rig (RIG): the key in the sun's place, where the light azimuth and elevation put it; the
    // fill through the spherical harmonics (soft and broad), with the floor's bounce of all three below the horizon;
    // the rim a second light behind, shadowed through its own light volume. No backdrop: the studio's black or a cyc.
    const rim = new Float32Array(8)
    if (s.env === 'rig') {
      const lux = luxOfEV(s.sunEV)
      const side = s.lightAz >= 0 ? 1 : -1
      const dirOf = (a: number, b: number): Vec3 => [Math.sin(a) * Math.cos(b), Math.sin(b), Math.cos(a) * Math.cos(b)]
      const tint = (kelvin: number, scale: number) => kelvinToACEScg(kelvin).map(v => v * scale) as Vec3
      sunLux = s.rigKey ? tint(RIG.key.kelvin, lux) : [0, 0, 0]
      const fillDir = dirOf(-side * RIG.fill.az, RIG.fill.el)
      const fillLux: Vec3 = s.rigFill ? tint(RIG.fill.kelvin, lux * RIG.fill.ratio) : [0, 0, 0]
      const rimDir = dirOf(-side * RIG.rim.az, RIG.rim.el)
      const rimLux: Vec3 = s.rigRim ? tint(RIG.rim.kelvin, lux * RIG.rim.ratio) : [0, 0, 0]
      rim.set([...rimDir, s.rigRim ? 1 : 0, ...rimLux, 0])
      pl.set([...sun, s.rigKey ? coneCos(RIG.key.omega) : 0, ...sunLux, 0], 0)
      pl.set([...fillDir, s.rigFill ? coneCos(RIG.fill.omega) : 0, ...fillLux, 0], 8)
      pl.set([...rimDir, s.rigRim ? coneCos(RIG.rim.omega) : 0, ...rimLux, 0], 16)
      // The fill as a small source (its irradiance times the basis toward it); the floor's bounce over the lower
      // hemisphere (where only the constant band and the vertical one survive).
      const sh = new Float64Array(27)
      const b = new Float64Array(9)
      shBasis(fillDir[0], fillDir[1], fillDir[2], b)
      for (let m = 0; m < 9; m++) for (let ch = 0; ch < 3; ch++) sh[3 * m + ch] += fillLux[ch] * b[m]
      for (let ch = 0; ch < 3; ch++) {
        const eh = sunLux[ch] * Math.max(sun[1], 0) + fillLux[ch] * Math.max(fillDir[1], 0) + rimLux[ch] * Math.max(rimDir[1], 0)
        const ground = (floorAlbedo * eh) / Math.PI
        sh[ch] += ground * 0.282095 * 2 * Math.PI
        sh[3 + ch] -= ground * 0.488603 * Math.PI
      }
      envData.set([1, 1, 0, 0], 0)
      for (let m = 0; m < 9; m++) envData.set([sh[3 * m], sh[3 * m + 1], sh[3 * m + 2], 0], 4 + 4 * m)
    }
    const fire = s.sourceType === 'fire'
    if (this.wb?.kelvin !== s.whiteBalance) this.wb = { kelvin: s.whiteBalance, m: whiteBalance(s.whiteBalance) }
    const wb = this.wb.m
    const r = this.renderData
    r.set([...c.pos, c.tanY], 0)
    r.set([...c.fwd, c.aspect], 4)
    r.set([...c.right, this.renderFrame % 4096], 8)
    this.lastExposure = this.exposure
    this.exposure = exposureOfEV(this.exposureEV())
    r.set([...c.up, this.exposure], 12)
    r.set([...sun, s.anisotropy], 16)
    r.set([...sunLux, s.densityScale], 20)
    r.set([...SKY_COLOUR.map(v => v * luxOfEV(s.skyEV)), s.albedo], 24)
    r.set([floorAlbedo, floorAlbedo, floorAlbedo, 3.2], 28)
    r.set([...this.frameMarch, this.simTime, fire ? s.flameDetail : 0], 32)
    // Cross-fade weights: each layer is weighted by a triangle of its age, 0 when it was just reset, 1 half a period on.
    const ageA = (this.simTime / DETAIL_PERIOD) % 1
    const ageB = (this.simTime / DETAIL_PERIOD + 0.5) % 1
    r.set([1 - Math.abs(2 * ageA - 1), 1 - Math.abs(2 * ageB - 1), s.detail, 1 / (8 * s.detailScale)], 36)
    // A flame's eddies rise at about half the speed of the gas in the flame, 1.9 Q^(1/5) m/s with Q in kW
    // (McCaffrey 1979).
    r.set([fire ? 1 : 0, fire ? SOOT_CHROMA[0] : 1, fire ? SOOT_CHROMA[2] : 1, 0.5 * 1.9 * s.heatRelease ** 0.2], 40)
    r.set([...wb[0], s.wind], 44)
    r.set([...wb[1], s.tricubic ? 1 : 0], 48)
    r.set([...wb[2], this.upresActive() && this.upres ? 1 : 0], 52)
    const cyc = s.background === 'charcoal' || s.background === 'gray'
    const stage = cyc ? 2 : s.background === 'sky' || e ? 1 : 0
    r.set([stage, fire ? SOOT_CHROMA[1] : 1, viewIndex(s.view), this.lutReady ? 1 : 0], 56)
    r.set(envData, 60)
    r.set(rim, 100)
    // The camera: a film camera's shutter at 24 frames a second (180 degrees is open 1/48 s), the glare's share and
    // levels, and whether the frame is paused. The motion blur only paused, a moment of the shutter a ray, which the
    // paused frame's accumulation converges over. While the fire runs, the few frames the history holds of a moment are
    // too few to average the shutter over: at a moment a sample, the largest share of the moving flames' grain (the
    // running bonfire on an integrated GPU, at its half render scale: 0.0075 with it, 0.0047 without; the subgrid
    // temperature's share 0.0006).
    r.set([this.paused ? (s.shutter / 360) / 24 : 0, s.glare, this.targets?.glareViews.length ?? GLARE_LEVELS, this.paused ? 1 : 0], 108)
    this.device.queue.writeBuffer(this.renderBuf, 0, r)
    // The reference set is the simulation's (a solid in it), but turned on or off it shows at once, paused or not.
    this.device.queue.writeBuffer(this.simBuf, 256, new Float32Array(this.referenceSet(s)))
  }

  // The on-set reference's layout (wgsl.ts refHit and setSolid), or zeros when it is off: the colour chart standing on
  // the floor, leaning back, turned to the camera and a little toward the fire; behind it the grey and chrome balls on
  // stands, at about the middle of the flames' height. Beside the source on its upwind side (the right, in still air),
  // clear of its bed and within the frame; if that side has no room in the frame, the other side, as far from the fire
  // as the frame allows when that is downwind (out of the flames that lean that way).
  private referenceSet(s: PyroState): number[] {
    if (!s.reference) return new Array(20).fill(0)
    const right = Math.max(s.srcX + s.srcRadius, REF.clear)
    const left = Math.min(s.srcX - s.srcRadius, -REF.clear)
    // The frame's half width where the balls stand (the camera looks along -z), and the domain's.
    const f = this.framing
    const seen = f.tanY > 0 ? (f.z - REF.ballZ) * f.tanY * f.aspect - REF.margin : Infinity
    const edge = Math.min(seen, DOMAIN_WIDTH / 2 - REF.margin)
    const room = (sd: number) => (sd > 0 ? edge - right : edge + left)
    const upwind = s.wind > 0.05 ? -1 : 1
    const side = room(upwind) >= REF.span || room(upwind) >= room(-upwind) ? upwind : -upwind
    // Pushed out along its side: downwind, to the frame's edge; otherwise only in, if the frame is too narrow.
    const downwind = side !== upwind && Math.abs(s.wind) > 0.05
    const push = downwind ? room(side) - REF.span : Math.min(room(side) - REF.span, 0)
    const at = (u: number) => (side > 0 ? right : left) + side * (u + push)
    const cx = at(REF.chartAt)
    const yaw = Math.atan2(-cx, f.z - REF.chartZ) - side * REF.chartTurn
    const w = REF.chartW
    const h = w * REF.chartAspect
    const U = [Math.cos(yaw) * w, 0, -Math.sin(yaw) * w]
    const V = [-Math.sin(REF.chartTilt) * Math.sin(yaw) * h, Math.cos(REF.chartTilt) * h, -Math.sin(REF.chartTilt) * Math.cos(yaw) * h]
    // (And the set's extent along x, for the simulation's quick test.)
    const xs = [at(REF.greyAt), at(REF.chromeAt)]
    const lo = Math.min(...xs, cx - Math.abs(U[0]) / 2 - Math.abs(V[0])) - REF.ballR
    const hi = Math.max(...xs, cx + Math.abs(U[0]) / 2 + Math.abs(V[0])) + REF.ballR
    return [
      ...[xs[0], REF.ballY, REF.ballZ, REF.ballR],
      ...[xs[1], REF.ballY, REF.ballZ, REF.ballR],
      ...[cx - U[0] / 2, 0.002, REF.chartZ - U[2] / 2, REF.standR],
      ...U, lo,
      ...V, hi,
    ]
  }

  // The frame's targets at the canvas's size, the display's (made again, and the history dropped, when it changes): the
  // march's image (drawn into its top left corner at the march's own size, which can change from frame to frame), the
  // accumulated history (two, swapped each frame), and the glare's chain of mips at half size.
  private ensureTargets() {
    const w = this.canvas.width
    const h = this.canvas.height
    if (this.targets && this.targets.w === w && this.targets.h === h) return this.targets
    if (this.targets) {
      this.targets.hdr.destroy()
      this.targets.hist.forEach(t => t.destroy())
      this.targets.glare.destroy()
    }
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    const tex = (more = 0) => this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: usage | more })
    // (The frame and its history are readable, for the probes that compare them with the path-traced reference.)
    const hdr = tex(GPUTextureUsage.COPY_SRC)
    const hist: [GPUTexture, GPUTexture] = [tex(GPUTextureUsage.COPY_SRC), tex(GPUTextureUsage.COPY_SRC)]
    const gw = Math.max(1, w >> 1)
    const gh = Math.max(1, h >> 1)
    const levels = Math.max(1, Math.min(GLARE_LEVELS, Math.floor(Math.log2(Math.min(gw, gh))) + 1))
    const glare = this.device.createTexture({ size: [gw, gh], format: 'rgba16float', usage, mipLevelCount: levels })
    const glareViews = Array.from({ length: levels }, (_, i) => glare.createView({ baseMipLevel: i, mipLevelCount: 1 }))
    this.targets = { w, h, hdr, hist, glare, glareViews, ...this.postGroups(hdr, hist, glareViews) }
    this.taaCount = 0
    return this.targets
  }
  // The bind groups of the passes after the march, for each history they read (made again when the ACES 2.0 LUT
  // arrives): the accumulation, the glare's first downsample (from the history just written), its levels each from the
  // one before (down) or after (up), and the display.
  private postGroups(hdr: GPUTexture, hist: [GPUTexture, GPUTexture], glareViews: GPUTextureView[]) {
    const p = this.pipes!
    const d = this.device
    const group = (pipe: GPURenderPipeline, entries: GPUBindGroupEntry[]) => d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries })
    const R = { binding: 1, resource: { buffer: this.renderBuf } }
    const src = (pipe: GPURenderPipeline, view: GPUTextureView) => group(pipe, [{ binding: 0, resource: view }, { binding: 1, resource: this.glareSampler }])
    const both = <T>(f: (k: number) => T): [T, T] => [f(0), f(1)]
    return {
      resolve: both(k => group(p.resolve, [{ binding: 2, resource: hdr.createView() }, { binding: 3, resource: hist[k].createView() }, { binding: 4, resource: { buffer: this.taaBuf } }, { binding: 5, resource: this.glareSampler }])),
      glareFirst: both(k => src(p.glareFirst, hist[k].createView())),
      glareDown: glareViews.map((_, i) => (i > 0 ? src(p.glareDown, glareViews[i - 1]) : null)),
      glareUp: glareViews.map((_, i) => (i < glareViews.length - 1 ? src(p.glareUp, glareViews[i + 1]) : null)),
      display: both(k =>
        group(p.display, [
          R,
          { binding: 2, resource: hist[k].createView() },
          { binding: 3, resource: glareViews[0] },
          { binding: 4, resource: this.glareSampler },
          { binding: 5, resource: this.lut.createView() },
          { binding: 6, resource: this.lutSampler },
        ]),
      ),
    }
  }
  // After the march: the accumulation into the next history, the glare's chain (if any), and the display to the
  // screen. stamps: for the profiler, GPU timestamps around each of them in turn from query base on.
  private post(enc: GPUCommandEncoder, screen: GPUTexture, t: Targets, stamps?: { qs: GPUQuerySet; base: number }) {
    const p = this.pipes!
    const next = 1 - this.histIndex
    const black = { r: 0, g: 0, b: 0, a: 1 }
    const draw = (view: GPUTextureView, pipe: GPURenderPipeline, group: GPUBindGroup, load: boolean, timestampWrites?: GPURenderPassTimestampWrites) => {
      const rp = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: load ? 'load' : 'clear', storeOp: 'store', clearValue: black }], timestampWrites })
      rp.setPipeline(pipe)
      rp.setBindGroup(0, group)
      rp.draw(3)
      rp.end()
    }
    let q = stamps?.base ?? 0
    const stamp = (begin: boolean, end: boolean): GPURenderPassTimestampWrites | undefined => {
      if (!stamps || (!begin && !end)) return undefined
      const w: GPURenderPassTimestampWrites = { querySet: stamps.qs }
      if (begin) w.beginningOfPassWriteIndex = q
      if (end) w.endOfPassWriteIndex = q + 1
      if (end) q += 2
      return w
    }
    draw(t.hist[next].createView(), p.resolve, t.resolve[this.histIndex], false, stamp(true, true))
    if (this.state.glare > 0) {
      const L = t.glareViews.length
      draw(t.glareViews[0], p.glareFirst, t.glareFirst[next], false, stamp(true, L === 1))
      for (let i = 1; i < L; i++) draw(t.glareViews[i], p.glareDown, t.glareDown[i]!, false)
      for (let i = L - 2; i >= 0; i--) draw(t.glareViews[i], p.glareUp, t.glareUp[i]!, true, stamp(false, i === 0))
    }
    draw(screen.createView(), p.display, t.display[next], false, stamp(true, true))
    this.histIndex = next
  }

  // (Development: resumFireNow, waited on, for the probes, which read the sums at a paused frame.)
  async resumFire() {
    this.resumFireNow()
    await this.device.queue.onSubmittedWorkDone()
  }

  // The fire's sums again for the frame as it stands, every direction of its escape tables marched for it: each step
  // marches a twenty-fourth of them (ESCAPE_PHASES), so a paused frame's tables were a blend of the last 0.8 s, and a
  // bonfire's soot about its flames moves the share that gets out each way by a seventh from one step to the next and a
  // quarter over the blend; and they march with the shading's albedo and anisotropy (the similarity relation), which a
  // paused frame can change. (Left as the steps left them, the bonfire's wall took from under half to twice the fire's
  // light it takes with every direction marched for the frame.) The step's own phase last, the frame's word of the
  // uniforms set for each.
  private resumFireNow() {
    const g = this.grids!
    const p = this.solver!
    const [nx, ny, nz] = g.n
    for (let k = 1; k <= ESCAPE_PHASES; k++) {
      this.device.queue.writeBuffer(this.simBuf, 12, new Uint32Array([this.frame + k]))
      const enc = this.device.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(p.fireSum)
      pass.setBindGroup(0, g.bg.fireSum)
      pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4))
      pass.setPipeline(p.fireTotal)
      pass.setBindGroup(0, g.bg.fireTotal)
      pass.dispatchWorkgroups(1 + FIRE_BANDS, 1, 1)
      pass.end()
      this.device.queue.submit([enc.finish()])
    }
    this.device.queue.writeBuffer(this.simBuf, 12, new Uint32Array([this.frame]))
    this.fireResum = false
    this.lightDirty = true
  }

  private upresActive() {
    return this.state.sourceType === 'smoke' && this.state.upres
  }

  // The upres grid's textures and bind groups, made on first use; writes the pass's uniform each step.
  private ensureUpres() {
    const g = this.grids!
    const p = this.pipes!
    const d = this.device
    if (!this.upres) {
      const [nx, ny, nz] = g.n
      const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST
      const make = () => d.createTexture({ size: [2 * nx, 2 * ny, 2 * nz], dimension: '3d', format: 'rgba16float', usage })
      const tex: [GPUTexture, GPUTexture] = [make(), make()]
      const blocks = [2 * nx, 2 * ny, 2 * nz].reduce((a, k) => a * Math.ceil(k / UPRES_BLOCK), 1)
      const holds = () => d.createBuffer({ size: 4 * blocks, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
      const held: [GPUBuffer, GPUBuffer] = [holds(), holds()]
      const v = (t: GPUTexture) => t.createView()
      const entries = (res: GPUBindingResource[]) => res.map((resource, binding) => ({ binding, resource }))
      const group = d.createBindGroup({
        layout: p.upres.getBindGroupLayout(0),
        entries: [
          ...entries([{ buffer: this.simBuf }, this.sampler, v(g.vel[0]), v(tex[0]), v(tex[1]), v(g.detA[0]), v(g.detB[0]), this.curlTex.createView(), this.noiseSampler, v(g.scal[0]), { buffer: this.upresBuf }]),
          { binding: 11, resource: { buffer: this.curlMaxBuf } },
          { binding: 12, resource: { buffer: held[0] } },
          { binding: 13, resource: { buffer: held[1] } },
        ],
      })
      const render = d.createBindGroup({
        layout: p.render.getBindGroupLayout(0),
        entries: [
          ...entries([{ buffer: this.simBuf }, { buffer: this.renderBuf }, this.sampler, v(g.scal[0]), v(g.light), v(g.detA[0]), v(g.detB[0]), this.noiseTex.createView(), this.noiseSampler, { buffer: this.bbBuf }, { buffer: g.fireLight }, this.blueTex.createView({ dimension: '2d-array' }), v(tex[0]), v(g.flame)]),
          { binding: 31, resource: { buffer: g.skin } },
          { binding: 32, resource: { buffer: g.ballSheet } },
          { binding: 33, resource: this.envTex.createView() },
          { binding: 34, resource: v(g.light2) },
          { binding: 35, resource: v(g.vel[0]) },
          { binding: 39, resource: v(g.cycBounce[0]) },
          { binding: 43, resource: v(g.cycDirect) },
          { binding: 44, resource: v(g.flameNear) },
          { binding: 45, resource: v(g.flameNearDir) },
          { binding: 46, resource: v(g.floorNear) },
        ],
      })
      // (The light volume shadows the same smoke, averaged over each coarse cell.)
      const coarse = d.createTexture({ size: [nx, ny, nz], dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING })
      const coarseGroup = d.createBindGroup({ layout: p.upresCoarse.getBindGroupLayout(0), entries: entries([v(tex[1]), v(coarse)]) })
      const light = d.createBindGroup({
        layout: p.light.getBindGroupLayout(0),
        entries: entries([{ buffer: this.simBuf }, { buffer: this.renderBuf }, this.sampler, v(g.scal[0]), v(g.light), { buffer: g.fireLight }, v(g.light2), v(coarse)]),
      })
      this.upres = { tex, held, group, render, coarse, coarseGroup, light }
      this.upresInit = true
    }
    const ageA = (this.simTime / DETAIL_PERIOD) % 1
    const ageB = (this.simTime / DETAIL_PERIOD + 0.5) % 1
    const dx = DOMAIN_WIDTH / g.n[0]
    // Eddies from four cells across down to half a cell (the fbm's three octaves), at the strength of the flow's
    // next band in Kolmogorov's spectrum, 2^(-5/6) of the resolved flow's, times the control.
    d.queue.writeBuffer(this.upresBuf, 0, new Float32Array([2 ** (-5 / 6) * this.state.upresStrength, 1 / (8 * 4 * dx), 1 - Math.abs(2 * ageA - 1), 1 - Math.abs(2 * ageB - 1), this.upresInit ? 1 : 0, 0, 0, 0]))
    this.upresInit = false
    return this.upres
  }

  private configure() {
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' })
  }

  // The canvas's texture for this frame, or null when there is none to draw to (disposed, or the device lost). Another
  // engine on the same canvas can leave the context unconfigured, or configured for its own device: React's development
  // double mount starts two, and the one it throws away unconfigures the shared context as it goes. Take it back.
  private canvasTexture(): GPUTexture | null {
    if (this.disposed || this.lost) return null
    if (typeof this.ctx.getConfiguration === 'function' && this.ctx.getConfiguration()?.device !== this.device) this.configure()
    try {
      return this.ctx.getCurrentTexture()
    } catch {
      // Unconfigured, in a browser that cannot say so: configure, and try once more.
      try {
        this.configure()
        return this.ctx.getCurrentTexture()
      } catch (err) {
        console.warn('Pyro: no canvas texture to draw to', err)
        return null
      }
    }
  }

  private render() {
    // Nothing to draw to: skip the frame rather than throw.
    const screen = this.canvasTexture()
    if (!screen) return
    const g = this.grids!
    const f = g.frame!
    const p = this.pipes!
    this.needsRender = false
    const t = this.ensureTargets()
    // (A history started afresh, paused: the settling starts again, at the running size for its first frames.)
    if (this.taaCount === 0) this.settle = 0
    this.chooseMarch()
    this.writeRender()
    if (this.pt.on && this.paused && this.ptPipe && this.settle >= this.settleFrames()) {
      this.renderTraced(screen, t)
      return
    }
    // The resolve's uniform (RESOLVE): whether there is a history, the exposure's change since it was drawn, this frame's
    // offset within the march's pixels (the R2 sequence, as the image pass draws it), the march's pixels per display
    // pixel, the samples the history may count (paused, as many as it settles over), the gather's sharpness, the march's
    // size, and the collision ball's footprint if it moved since the last frame.
    const [mw, mh] = this.frameMarch
    const n = this.renderFrame % 4096
    const frac = (v: number) => v - Math.floor(v)
    const sub = [frac(0.7548777 * n + 0.5) - 0.5, frac(0.5698403 * n + 0.5) - 0.5]
    const most = this.paused ? TAA_SETTLE : RUNNING_SAMPLES
    const exposureChange = this.exposure / Math.max(this.lastExposure, 1e-30)
    const full = Math.min(Math.max((Math.min(mw / t.w, mh / t.h) - 0.5) / 0.5, 0), 1)
    const sharpness = GATHER_SHARPNESS[0] + (GATHER_SHARPNESS[1] - GATHER_SHARPNESS[0]) * full
    const relight = this.paused ? 0 : RELIGHT
    this.device.queue.writeBuffer(this.taaBuf, 0, new Float32Array([this.taaCount > 0 ? 1 : 0, exposureChange, ...sub, mw / t.w, mh / t.h, most, sharpness, mw, mh, relight, 0, ...this.ballFootprint(t)]))
    // Paused, the fire's escape tables marched afresh for the frame (after its uniforms, which they read). (Not before
    // a restart's first step, which reads the sums, the bed's area among them, as the restart left them, empty: summed
    // then, they changed the whole run.)
    if (this.fireResum && this.paused && this.fireLit && this.frame > 0) this.resumFireNow()
    const enc = this.device.createCommandEncoder()
    if (this.lightDirty) {
      const pass = enc.beginComputePass()
      pass.setPipeline(p.light)
      pass.setBindGroup(0, this.upresActive() && this.upres ? this.upres.light : f.light)
      pass.dispatchWorkgroups(Math.ceil(g.lightDims[0] / 4), Math.ceil(g.lightDims[1] / 4), Math.ceil(g.lightDims[2] / 4))
      pass.setPipeline(p.flameNear)
      pass.setBindGroup(0, f.flameNear)
      pass.dispatchWorkgroups(Math.ceil(g.nearDims[0] / 4), Math.ceil(g.nearDims[1] / 4), Math.ceil(g.nearDims[2] / 4))
      pass.setPipeline(p.floorNear)
      pass.setBindGroup(0, f.floorNear)
      pass.dispatchWorkgroups(Math.ceil(g.n[0] / 8), Math.ceil(g.n[2] / 8))
      if (this.state.background === 'gray' || this.state.background === 'charcoal') {
        const cg: [number, number] = [Math.ceil(CYC_MAP.w / 8), Math.ceil(CYC_MAP.h / 8)]
        pass.setPipeline(p.cycDirect)
        pass.setBindGroup(0, f.cycDirect)
        pass.dispatchWorkgroups(...cg)
        pass.setPipeline(p.cycBounce)
        for (const k of [0, 1]) {
          pass.setBindGroup(0, f.cycBounce[k])
          pass.dispatchWorkgroups(...cg)
        }
      }
      pass.end()
      this.lightDirty = false
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: t.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    })
    pass.setViewport(0, 0, mw, mh, 0, 1)
    pass.setScissorRect(0, 0, mw, mh)
    pass.setPipeline(p.render)
    pass.setBindGroup(0, this.upresActive() && this.upres ? this.upres.render : f.render)
    pass.draw(3)
    pass.end()
    this.post(enc, screen, t)
    this.device.queue.submit([enc.finish()])
    this.taaCount++
    this.renderFrame++
    this.ballDrawn = this.state.ball ? [...this.ballPos] : null
    // Paused, keep drawing until the history has settled (and then, tracing, go on to the traced frames).
    if (this.paused) {
      this.settle++
      if (this.settle < this.settleFrames() || (this.pt.on && this.ptPipe)) this.needsRender = true
    }
  }

  // The collision ball's footprint in the display's pixels, when it has moved since the last frame drew it (a drag):
  // a circle around where it was and where it is, with a margin for its soft edge (centre x, y, radius, and 1), which
  // the resolve starts afresh. (The flames and the floor's shadow that move with it the history's clip deals with.)
  private ballFootprint(t: Targets): number[] {
    const was = this.ballDrawn
    if (!this.state.ball || !was || was.every((v, i) => v === this.ballPos[i])) return [0, 0, 0, 0]
    const c = this.cam
    const at = (p: Vec3) => {
      const d = sub(p, c.pos)
      const z = Math.max(dot(d, c.fwd), 1e-3)
      const x = dot(d, c.right) / (z * c.tanY * c.aspect)
      const y = dot(d, c.up) / (z * c.tanY)
      return { x: ((x + 1) / 2) * t.w, y: ((1 - y) / 2) * t.h, r: (this.state.ballRadius / (z * c.tanY)) * (t.h / 2) }
    }
    const a = at(was)
    const b = at(this.ballPos)
    const r = Math.hypot(a.x - b.x, a.y - b.y) / 2 + 1.25 * Math.max(a.r, b.r) + 4
    return [(a.x + b.x) / 2, (a.y + b.y) / 2, r, 1]
  }

  // The path-traced reference, while paused: one more path per pixel into the running average, then the wipe (the
  // settled live frame left of it, the traced right), the glare and the display. Anything changed since the last
  // traced frame starts the average afresh.
  private renderTraced(screen: GPUTexture, t: Targets) {
    const p = this.pipes!
    const pt = this.ensurePt(t)
    const key = `${this.stateVersion}|${this.simTime}|${this.ballPos.join(',')}|${t.w}x${t.h}|${this.camVersion}|${this.upresActive()}|${this.envLight ? 1 : 0}`
    // (Traced at the display's own size, whatever the march's.)
    this.device.queue.writeBuffer(this.renderBuf, 32 * 4, new Float32Array([t.w, t.h]))
    if (key !== this.pt.key) {
      this.pt.key = key
      this.pt.n = 0
    }
    const head = new Uint32Array([this.pt.frame++, this.pt.n, PT_BOUNCES, this.ptParts]) // (the seed, the count, the cap, the parts left out)
    const buf = new ArrayBuffer(112)
    new Uint32Array(buf, 0, 4).set(head)
    new Float32Array(buf, 16, 24).set(this.ptLights)
    this.device.queue.writeBuffer(this.ptBuf, 0, buf)
    this.device.queue.writeBuffer(this.wipeBuf, 0, new Float32Array([Math.round(this.pt.wipe * t.w), 0, 0, 0]))
    const enc = this.device.createCommandEncoder()
    const black = { r: 0, g: 0, b: 0, a: 1 }
    const draw = (view: GPUTextureView, pipe: GPURenderPipeline, group: GPUBindGroup, load = false) => {
      const rp = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: load ? 'load' : 'clear', storeOp: 'store', clearValue: black }] })
      rp.setPipeline(pipe)
      rp.setBindGroup(0, group)
      rp.draw(3)
      rp.end()
    }
    const next = 1 - this.pt.idx
    draw(pt.accum[next].createView(), this.ptPipe!, pt.trace[this.pt.idx])
    draw(pt.comp.createView(), p.ptComposite, pt.composite[next][this.histIndex])
    if (this.state.glare > 0) {
      const L = t.glareViews.length
      // (The glare from the live frame, on both sides: a traced frame's early outliers would spread over the whole.)
      draw(t.glareViews[0], p.glareFirst, t.glareFirst[this.histIndex])
      for (let i = 1; i < L; i++) draw(t.glareViews[i], p.glareDown, t.glareDown[i]!)
      for (let i = L - 2; i >= 0; i--) draw(t.glareViews[i], p.glareUp, t.glareUp[i]!, true)
    }
    draw(screen.createView(), p.display, pt.display)
    this.device.queue.submit([enc.finish()])
    this.pt.idx = next
    this.pt.n++
    this.renderFrame++
    this.needsRender = this.pt.n < PT_SAMPLES
  }

  // The traced frame's targets and bind groups (made again with the frame's own): the running average (two, swapped
  // each frame, full float; readable, for the probes), the wipe's composite, and the passes that read them.
  private ensurePt(t: Targets) {
    const upres = this.upresActive() && !!this.upres
    if (this.ptT && this.ptT.targets === t && this.ptT.upres === upres) return this.ptT
    if (this.ptT) {
      this.ptT.accum.forEach(a => a.destroy())
      this.ptT.comp.destroy()
    }
    const d = this.device
    const p = this.pipes!
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    const accum: [GPUTexture, GPUTexture] = [0, 1].map(() => d.createTexture({ size: [t.w, t.h], format: 'rgba32float', usage: usage | GPUTextureUsage.COPY_SRC })) as [GPUTexture, GPUTexture]
    const comp = d.createTexture({ size: [t.w, t.h], format: 'rgba16float', usage })
    const v = (x: GPUTexture) => x.createView()
    const trace = [0, 1].map(k => d.createBindGroup({ layout: this.ptPipe!.getBindGroupLayout(0), entries: this.traceEntries(accum[k], upres) })) as [GPUBindGroup, GPUBindGroup]
    const composite = [0, 1].map(k =>
      [0, 1].map(h =>
        d.createBindGroup({
          layout: p.ptComposite.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: v(t.hist[h]) },
            { binding: 1, resource: v(accum[k]) },
            { binding: 2, resource: { buffer: this.wipeBuf } },
          ],
        }),
      ),
    )
    const display = d.createBindGroup({
      layout: p.display.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderBuf } },
        { binding: 2, resource: v(comp) },
        { binding: 3, resource: t.glareViews[0] },
        { binding: 4, resource: this.glareSampler },
        { binding: 5, resource: this.lut.createView() },
        { binding: 6, resource: this.lutSampler },
      ],
    })
    this.ptT = { targets: t, upres, accum, comp, trace, composite, display }
    this.pt.key = ''
    return this.ptT
  }

  // The path tracer's bindings (the image pass's, then the average it reads, its uniforms, the key-free map).
  private traceEntries(prev: GPUTexture, upres: boolean): GPUBindGroupEntry[] {
    const g = this.grids!
    const v = (x: GPUTexture) => x.createView()
    const res: GPUBindingResource[] = [{ buffer: this.simBuf }, { buffer: this.renderBuf }, this.sampler, v(g.scal[0]), v(g.light), v(g.detA[0]), v(g.detB[0]), this.noiseTex.createView(), this.noiseSampler, { buffer: this.bbBuf }, { buffer: g.fireLight }, this.blueTex.createView({ dimension: '2d-array' }), upres ? v(this.upres!.tex[0]) : this.dummy3d.createView(), v(g.flame)]
    return [
      ...res.map((resource, binding) => ({ binding, resource })),
      { binding: 31, resource: { buffer: g.skin } },
      { binding: 32, resource: { buffer: g.ballSheet } },
      { binding: 33, resource: this.envTex.createView() },
      { binding: 34, resource: v(g.light2) },
      { binding: 35, resource: v(g.vel[0]) },
      { binding: 36, resource: v(prev) },
      { binding: 37, resource: { buffer: this.ptBuf } },
      { binding: 38, resource: this.envResTex.createView() },
    ]
  }

  // Development check of the fire's light (PATH_TRACE's fluence): per cell of the grid's middle plane along z, the
  // image pass's fire light (its light volume), the reference's sampling of the flames' own (lux of luminance), and
  // the soot's extinction there (1/m), four floats a cell, row 0 at the top.
  async fireFieldCheck(): Promise<{ n: Vec3; data: Float32Array } | null> {
    if (!this.grids) return null
    const d = this.device
    const [nx, ny] = this.grids.n
    const m = d.createShaderModule({ code: PATH_TRACE })
    const pipe = await d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: m, entryPoint: 'vs' },
      fragment: { module: m, entryPoint: 'fluence', targets: [{ format: 'rgba32float' }] },
      primitive: { topology: 'triangle-list' },
    })
    const target = d.createTexture({ size: [nx, ny], format: 'rgba32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC })
    const prev = d.createTexture({ size: [1, 1], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING })
    const group = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: this.traceEntries(prev, this.upresActive() && !!this.upres) })
    const bpr = Math.ceil((nx * 16) / 256) * 256
    const read = d.createBuffer({ size: bpr * ny, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] })
    rp.setPipeline(pipe)
    rp.setBindGroup(0, group)
    rp.draw(3)
    rp.end()
    enc.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: bpr }, [nx, ny])
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const raw = new Float32Array(read.getMappedRange())
    const data = new Float32Array(nx * ny * 4)
    for (let y = 0; y < ny; y++) data.set(raw.subarray((y * bpr) / 4, (y * bpr) / 4 + nx * 4), y * nx * 4)
    read.unmap()
    read.destroy()
    target.destroy()
    prev.destroy()
    return { n: this.grids.n, data }
  }

  // The path-traced reference (PATH_TRACE), while paused: on, traced frames accumulate behind the wipe; off (or
  // resuming), the live frame alone. Its pipeline compiles the first time it is asked for.
  setPathTrace(on: boolean) {
    this.pt.on = on && this.paused
    this.pt.key = ''
    this.pt.n = 0
    // Both sides at one exposure: the meter settles at once (as for a still), and the live frame settles again at it
    // before the traced frames begin.
    if (this.pt.on) {
      this.meterEV = this.targetEV()
      this.taaCount = 0
    }
    if (this.pt.on && !this.ptPipe && !this.ptCompiling) {
      this.ptCompiling = true
      const m = this.device.createShaderModule({ code: PATH_TRACE })
      this.device
        .createRenderPipelineAsync({
          layout: 'auto',
          vertex: { module: m, entryPoint: 'vs' },
          fragment: { module: m, entryPoint: 'pt', targets: [{ format: 'rgba32float' }] },
          primitive: { topology: 'triangle-list' },
        })
        .then(pipe => {
          this.ptPipe = pipe
          this.needsRender = true
          this.kick()
        })
        .catch(err => {
          console.warn('Pyro: the path tracer did not compile', err)
          this.pt.on = false
        })
        .finally(() => {
          this.ptCompiling = false
          this.report(true)
        })
    }
    this.needsRender = true
    this.kick()
    this.report(true)
  }
  // The wipe's place, as a share of the frame's width from the left (the live frame left of it).
  setWipe(share: number) {
    this.pt.wipe = Math.min(Math.max(share, 0), 1)
    if (this.pt.on) {
      this.needsRender = true
      this.kick()
    }
  }


  // When the simulation falls behind real time, or the display's frames come slower than 40 a second, for a second
  // while simulating (the GPU is the bottleneck), march fewer pixels, down to RENDER_SCALE_MIN. Frames back above 55 a
  // second raise it again, but never above the scale the simulation last fell behind at (until a restart: a new
  // starting point or resolution judges afresh), and past the starting scale, toward the display's own size, only while
  // the simulation keeps real time too (SIM_REAL_RATE): a GPU with room to spare. (Whether a step down helped is not
  // judged from the rate it gives: a starting fire grows for its first ten seconds or so, costing more to draw as it
  // does, so its rate falls whatever the scale.)
  private adapt() {
    const now = this.frameTimes[this.frameTimes.length - 1]
    const since = Math.min(now - this.lastAdapt, 100)
    this.lastAdapt = now
    if (!this.paused && now - this.steadySince > SIM_SETTLE_MS) {
      let rate = 0
      for (const t of this.stepTimes) if (now - t <= 1000) rate++
      this.behindMs = rate < SIM_BEHIND_RATE ? this.behindMs + since : 0
      if (this.behindMs >= SIM_BEHIND_MS && this.renderScale > RENDER_SCALE_MIN && this.pinScale == null) {
        this.renderScale = Math.max(RENDER_SCALE_MIN, this.renderScale * 0.85)
        this.scaleCeiling = this.renderScale
        this.behindMs = 0
        this.steadySince = now
        this.frameTimes = []
        this.applySize()
        return
      }
    }
    while (this.frameTimes.length && now - this.frameTimes[0] > 1000) this.frameTimes.shift()
    if (this.paused || this.frameTimes.length < 10) return
    const spanMs = now - this.frameTimes[0]
    const fps = ((this.frameTimes.length - 1) * 1000) / Math.max(spanMs, 1)
    if (this.pinScale != null) {
      // (A pinned scale: no adapting.)
    } else if (fps < 40 && this.renderScale > RENDER_SCALE_MIN) {
      this.renderScale = Math.max(RENDER_SCALE_MIN, this.renderScale * 0.85)
      this.frameTimes = []
      this.applySize()
    } else if (fps > 55) {
      let rate = 0
      for (const t of this.stepTimes) if (now - t <= 1000) rate++
      const realTime = now - this.steadySince > SIM_SETTLE_MS && rate >= SIM_REAL_RATE
      const top = Math.min(this.scaleCeiling, realTime ? this.scaleMax() : Math.max(RENDER_SCALE_START, this.renderScale))
      if (this.renderScale < top) {
        this.renderScale = Math.min(top, this.renderScale / 0.85)
        this.frameTimes = []
        this.applySize()
      }
    }
    // A default resolution too heavy for this GPU: one step down, judged once (it restarts the simulation).
    if (this.autoResolution && this.runClock >= AUTO_RES_TO_S) {
      this.autoResolution = false
      const i = RESOLUTION_ORDER.indexOf(this.state.resolution)
      if (i > 0 && this.autoSteps < AUTO_RES_MIN_RATE * (AUTO_RES_TO_S - AUTO_RES_FROM_S)) {
        this.autoStepped = true
        this.setState({ resolution: RESOLUTION_ORDER[i - 1] })
        this.report(true)
      }
    }
  }

  // Auto exposure (see physics.ts): the ambient light metered, but no brighter than the fire allows. Eased over about
  // a second, so the fire's puffing does not pump the exposure.
  private targetEV() {
    const s = this.state
    // (An HDR environment's illuminance on open ground, key included, is the Sky EV's; the softbox rig's, the key's
    // and the fill's as they face the camera.)
    const rig = s.env === 'rig' ? luxOfEV(s.sunEV) * Math.max((s.rigKey ? 1 : 0) + (s.rigFill ? RIG.fill.ratio : 0), 0.05) : 0
    const lux = s.env === 'rig' ? rig : this.envLight ? luxOfEV(s.skyEV) : luxOfEV(s.sunEV) + luxOfEV(s.skyEV)
    const ambient = Math.log2((METER_AMBIENT * lux) / 2.5)
    if (s.sourceType !== 'fire' || this.fireCd <= 0) return ambient
    return Math.max(ambient, METER_FIRE_EV + METER_FIRE_SLOPE * Math.log2(this.fireCd / METER_FIRE_CD))
  }
  private exposureEV() {
    const s = this.state
    if (s.exposureMode === 'manual') return s.exposure
    // (Tracing, the meter holds where it settled, so the live frame and the traced one share one exposure.)
    if (this.pt.on && this.meterEV !== null) return this.meterEV - s.exposureComp
    const target = this.targetEV()
    const now = performance.now()
    if (this.meterEV === null) this.meterEV = target
    else this.meterEV += (target - this.meterEV) * (1 - Math.exp(-(now - this.meterAt) / 800))
    this.meterAt = now
    return this.meterEV - s.exposureComp
  }
  // The fire's luminous intensity, read back from the grid sums.
  private async readFire() {
    if (this.metering || !this.grids || this.state.sourceType !== 'fire') return
    this.metering = true
    const d = this.device
    const read = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.grids.fireLight, 0, read, 0, 16)
    d.queue.submit([enc.finish()])
    try {
      await read.mapAsync(GPUMapMode.READ)
      const dx = DOMAIN_WIDTH / this.grids.n[0]
      this.fireCd = new Float32Array(read.getMappedRange())[3] * dx * dx * dx
      read.unmap()
    } catch {}
    read.destroy()
    this.metering = false
  }

  private ballBurning() {
    return this.state.ball && this.state.ballBurns
  }

  // The chosen HDR environment's light, loaded and prepared (environment.ts, cached per map), then its backdrop
  // uploaded; until then, and without one, the sun and the flat sky light the scene.
  private syncEnv() {
    const env = this.state.env
    if (env === 'none' || env === 'rig') {
      this.envLight = null
      return
    }
    void environmentLight(env)
      .then(e => {
        if (this.disposed || this.state.env !== env) return
        this.envLight = e
        this.device.queue.writeTexture({ texture: this.envTex }, toHalf(e.backdrop.rgba), { bytesPerRow: BACKDROP_W * 8 }, [BACKDROP_W, BACKDROP_H])
        this.device.queue.writeTexture({ texture: this.envResTex }, toHalf(e.residual.rgba), { bytesPerRow: BACKDROP_W * 8 }, [BACKDROP_W, BACKDROP_H])
        this.lightDirty = true
        this.needsRender = true
        this.taaCount = 0
        this.kick()
      })
      .catch(err => console.warn(`Pyro environment ${env} unavailable`, err))
  }

  // A fresh wooden ball: unburned, at the air's temperature. (Its cells are set up on its first step, BALL_HEAT, from
  // their columns cleared here.)
  private resetSkin() {
    if (!this.grids) return
    const g = this.grids
    const fresh = new Float32Array(SKIN_TEXELS * 4)
    for (let i = 0; i < SKIN_TEXELS; i++) fresh.set([T_AMBIENT, 0, 0, 0], i * 4)
    this.device.queue.writeBuffer(g.skin, 0, fresh)
    const enc = this.device.createCommandEncoder()
    for (const b of [g.ballSheet, g.woodTemps, g.woodCells, g.woodCols]) enc.clearBuffer(b)
    this.device.queue.submit([enc.finish()])
    this.woodParity = 0
    this.woodClock = 0
    this.woodPhase = -1
    this.wear = new Float32Array(12)
    this.balls++
    this.ballStatus = null
    this.needsRender = true
    this.taaCount = 0
  }

  // Whether the wood steps with this step of the gas (every WOOD_EVERY), and after it does: its halves trade places,
  // and its clock runs on.
  private woodDue() {
    this.woodPhase = (this.woodPhase + 1) % WOOD_EVERY
    return this.woodPhase === 0
  }
  private woodStepped() {
    this.woodParity = 1 - this.woodParity
    this.woodClock += SIM_DT * WOOD_EVERY * this.woodSpeed
  }

  // Swaps the charred ball for a fresh one.
  newBall() {
    this.resetSkin()
    this.kick()
    this.report(true)
  }

  // The wooden ball's state for the HUD, as the solver works it out (BALL_HEAT and ballFuelFlux in wgsl.ts): its hottest
  // surface, the power its volatiles burn at once they come off fast enough, and how much of it is charred.
  private async readBall() {
    if (this.readingBall || !this.grids || !this.ballBurning()) {
      if (!this.ballBurning()) this.ballStatus = null
      return
    }
    this.readingBall = true
    const d = this.device
    const bytes = SKIN_TEXELS * 16
    const read = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.grids.skin, 0, read, 0, bytes)
    // (And now and then the rest of the surface, for its wear: two vec4s to a texel, the wear in mm the first's y.)
    const sheet = this.wearReads++ % 4 === 0 ? d.createBuffer({ size: 2 * bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null
    const ball = this.balls
    if (sheet) enc.copyBufferToBuffer(this.grids.ballSheet, 0, sheet, 0, 2 * bytes)
    d.queue.submit([enc.finish()])
    try {
      if (sheet) {
        await sheet.mapAsync(GPUMapMode.READ)
        const b = new Float32Array(sheet.getMappedRange())
        const worn = new Float32Array(SKIN_TEXELS)
        for (let i = 0; i < SKIN_TEXELS; i++) worn[i] = b[8 * i + 1] * 0.001
        sheet.unmap()
        if (this.ballBurning() && ball === this.balls) this.wear = fitWear(worn)
      }
      await read.mapAsync(GPUMapMode.READ)
      const a = new Float32Array(read.getMappedRange())
      const r = this.state.ballRadius
      let hottest = 0
      let fuel = 0
      let charredArea = 0
      let area = 0
      const smooth = (e0: number, e1: number, x: number) => {
        const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1)
        return t * t * (3 - 2 * t)
      }
      const solid = skinSolidAngles()
      for (let i = 0; i < SKIN_TEXELS; i++) {
        const k = i * 4
        const texel = r * r * solid[i % (SKIN_N * SKIN_N)]
        hottest = Math.max(hottest, a[k])
        fuel += a[k + 1] * smooth(0.0015, 0.0025, a[k + 1]) * texel
        if (a[k + 2] > 0.3) charredArea += texel
        area += texel
      }
      read.unmap()
      if (this.ballBurning()) this.ballStatus = { surfaceK: hottest, burningKW: (fuel * FUEL.heatPerFuel) / 1000, charred: charredArea / area }
    } catch {}
    read.destroy()
    sheet?.destroy()
    this.readingBall = false
  }

  // Development: one column of the wooden ball's wood, the one under the surface facing dir (from the ball's centre):
  // each cell's depth below the surface as made (mm), thickness (mm), temperature (K), wood left, char, water left and
  // ash; the skin there (surface temperature K, volatiles kg/m2/s, char depth mm, ash cover), the rest of the surface
  // (fuel flow up it, wear mm, char halfway down and at its base K), the column's patch of the grain (its wood's density
  // against the mean, before the rings), the share of it under cracks, the soot on it (kg/m2) and how much of it is
  // covered, how much the fire's flame lies against it, and the wear's fit as the gas sees it (fitWear).
  async readWood(dir: Vec3 = [0, 0, 1]) {
    if (!this.grids) return null
    const g = this.grids
    const d = this.device
    const t = skinTexel(dir)
    const half = this.woodParity
    const read = d.createBuffer({ size: WOOD_LAYERS * 20 + 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    for (let k = 0; k < WOOD_LAYERS; k++) {
      enc.copyBufferToBuffer(g.woodTemps, (half * WOOD_CELLS + k * SKIN_TEXELS + t) * 4, read, k * 4, 4)
      enc.copyBufferToBuffer(g.woodCells, (k * SKIN_TEXELS + t) * 16, read, WOOD_LAYERS * 4 + k * 16, 16)
    }
    enc.copyBufferToBuffer(g.skin, t * 16, read, WOOD_LAYERS * 20, 16)
    enc.copyBufferToBuffer(g.ballSheet, t * 32, read, WOOD_LAYERS * 20 + 16, 16)
    enc.copyBufferToBuffer(g.ballSheet, t * 32 + 16, read, WOOD_LAYERS * 20 + 48, 16)
    enc.copyBufferToBuffer(g.woodCols, ((half * SKIN_TEXELS + t) * 2 + 1) * 16, read, WOOD_LAYERS * 20 + 32, 16)
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const a = new Float32Array(read.getMappedRange().slice(0))
    read.unmap()
    read.destroy()
    const r = this.state.ballRadius
    const layers = woodLayers(r)
    return {
      texel: t,
      clock: this.woodClock,
      skin: Array.from(a.slice(WOOD_LAYERS * 5, WOOD_LAYERS * 5 + 4)),
      sheet: Array.from(a.slice(WOOD_LAYERS * 5 + 4, WOOD_LAYERS * 5 + 8)),
      density: a[WOOD_LAYERS * 5 + 8],
      cracks: a[WOOD_LAYERS * 5 + 9],
      soot: a[WOOD_LAYERS * 5 + 10],
      flameAgainst: a[WOOD_LAYERS * 5 + 11],
      sootCover: a[WOOD_LAYERS * 5 + 12],
      airSpeed: a[WOOD_LAYERS * 5 + 13],
      heatTransfer: a[WOOD_LAYERS * 5 + 14],
      crackTemp: a[WOOD_LAYERS * 5 + 15],
      wearFit: Array.from(this.wear),
      cells: layers.map((l, k) => ({ top: l.top * 1000, thick: l.thick * 1000, T: a[k], wood: a[WOOD_LAYERS + k * 4], char: a[WOOD_LAYERS + k * 4 + 1], water: a[WOOD_LAYERS + k * 4 + 2], ash: a[WOOD_LAYERS + k * 4 + 3] })),
    }
  }

  private report(force: boolean) {
    const now = performance.now()
    if (!force && now - this.lastStatus < STATUS_MS) return
    this.lastStatus = now
    void this.readFire()
    void this.readBall()
    while (this.stepTimes.length && now - this.stepTimes[0] > 1000) this.stepTimes.shift()
    this.onStatus({
      ready: !!this.grids,
      running: !this.paused && this.visible && !!this.grids,
      simTime: this.simTime,
      stepsPerSec: this.stepTimes.length,
      renderScale: this.renderScale,
      dims: this.grids ? this.grids.n : RESOLUTIONS[this.state.resolution].dims,
      resolution: this.state.resolution,
      autoStepped: this.autoStepped,
      exposureEV: this.state.exposureMode === 'manual' ? this.state.exposure : (this.meterEV ?? this.targetEV()) - this.state.exposureComp,
      ball: this.ballBurning() ? this.ballStatus : null,
      pathTrace: { on: this.pt.on, samples: this.pt.on && this.ptPipe ? this.pt.n : 0, compiling: this.ptCompiling },
      view: this.closeUp() ? 'close' : 'whole',
    })
  }

  // Development profiler: one step and one frame, each kernel in its own pass with GPU timestamps; milliseconds per
  // kernel (all SOR sweeps together). Needs the timestamp-query feature. The frame's march at a given size, or this
  // frame's.
  async profileStep(march?: [number, number]): Promise<Record<string, number> | null> {
    if (!this.grids || !this.device.features.has('timestamp-query')) return null
    const screen = this.canvasTexture()
    if (!screen) return null
    const d = this.device
    const g = this.grids
    const p = this.pipes!
    const labels: string[] = []
    const slots = 64
    const qs = d.createQuerySet({ type: 'timestamp', count: slots * 2 })
    const resolve = d.createBuffer({ size: slots * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    const read = d.createBuffer({ size: slots * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    this.writeSim()
    if (march) this.frameMarch = march
    else this.chooseMarch()
    this.writeRender()
    const enc = d.createCommandEncoder()
    const [nx, ny, nz] = g.n
    const cv: Vec3 = [Math.ceil((nx + 1) / 4), Math.ceil((ny + 1) / 4), Math.ceil((nz + 1) / 4)]
    const cc: Vec3 = [Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)]
    const timed = (label: string, body: (pass: GPUComputePassEncoder) => void) => {
      const k = labels.length
      labels.push(label)
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } })
      body(pass)
      pass.end()
    }
    const one = (pipe: GPUComputePipeline, bg: GPUBindGroup, wg: Vec3) => (pass: GPUComputePassEncoder) => {
      pass.setPipeline(pipe)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(wg[0], wg[1], wg[2])
    }
    const scheme = this.velScheme()
    const solveLabel = this.state.pressure === 'sor' ? `sor x${this.state.pressureIters}` : `multigrid x${this.state.vcycles}`
    if (this.ballBurning()) {
      timed('ballHeat', one(p.ballHeat, g.bg.ballHeat, [SKIN_TEXELS / 64, 1, 1]))
      enc.copyBufferToBuffer(g.skinNext, 0, g.skin, 0, SKIN_TEXELS * 16)
      this.woodStepped()
    }
    if (scheme === 'lfm') {
      timed('advectVel', one(p.advectVelDouble, g.lfm.advect[0], cv))
    } else if (scheme === 'ar') {
      timed('advectVel', one(p.advectVelHalf, g.bg.arAdvect1, cv))
    } else if (scheme === 'mac') {
      timed('advectVel', pass => {
        one(p.advectVel, g.bg.advectVelFwd, cv)(pass)
        one(p.advectVelBack, g.bg.advectVelBack, cv)(pass)
        one(p.velCombine, g.bg.velCombine, cv)(pass)
      })
    } else {
      timed('advectVel', one(p.advectVel, g.bg.advectVel, cv))
    }
    if (this.ivockOn()) {
      // Each part of IVOCK on its own.
      const [ix, iy, iz] = g.n
      const sweep: Vec3 = [Math.ceil(ix / 2 / 16), Math.ceil(iy / 4), iz]
      timed('ivock: vorticity', one(p.vortBefore, g.bg.vortBefore, cc))
      timed('ivock: advect', pass => {
        one(p.advFwd, g.bg.vortAdvect, cc)(pass)
        if (scheme === 'mac') one(p.advBwd, g.bg.vortBack, cc)(pass)
      })
      timed('ivock: loss', one(p.ivockTarget, g.bg.ivockTarget, cc))
      timed('ivock: potentials', pass => {
        for (let i = 0; i < IVOCK_SWEEPS; i++) {
          one(p.ivockSmooth, g.bg.ivockSmooth[0], sweep)(pass)
          one(p.ivockSmooth, g.bg.ivockSmooth[1], sweep)(pass)
        }
      })
      timed('ivock: correction', pass => {
        one(p.ivockCurl, g.bg.ivockCurl, cc)(pass)
        one(p.ivockApply, g.bg.ivockApply, cv)(pass)
      })
      enc.copyTextureToTexture({ texture: g.vel[2] }, { texture: g.vel[3] }, [nx + 1, ny + 1, nz + 1])
    }
    timed('curl', one(p.curl, g.bg.curl, cc))
    timed('forces', one(p.forces, g.bg.forces, cv))
    timed('divergence', one(p.divergence, g.bg.divergence, cc))
    timed(solveLabel, pass => this.solvePressure(pass))
    if (scheme === 'ar') {
      timed('reflect', pass => {
        one(p.projectReflect, g.bg.projectReflect, cv)(pass)
        one(p.advectVelHalf, g.bg.arAdvect2, cv)(pass)
      })
      enc.copyTextureToTexture({ texture: g.vel[3] }, { texture: g.vel[0] }, [nx + 1, ny + 1, nz + 1])
      timed('divergence 2', one(p.divergence, g.bg.divergence, cc))
      timed(`${solveLabel} (2)`, pass => this.solvePressure(pass))
    }
    timed('project', one(p.project, g.bg.project, cv))
    if (scheme === 'lfm') timed('lfm: ring', one(p.lfmCellVel, g.lfm.cellVel[2], cc))
    enc.copyTextureToTexture({ texture: g.vel[1] }, { texture: g.vel[0] }, [nx + 1, ny + 1, nz + 1])
    if (scheme === 'lfm') {
      // The march every step; the re-initialisation's map and pullbacks once a cycle (a quarter of them per step).
      // (Development only: the extra march disturbs the cycle's maps for the rest of it.)
      timed('lfm: march', one(p.lfmMarchFwd, g.lfm.marchFwd[2], cv))
      timed('lfm: backward map', one(p.lfmMarchBwd, g.lfm.marchBwd, cv))
      timed('lfm: pullbacks', pass => {
        one(p.lfmPull[0], g.lfm.pull[0], cv)(pass)
        one(p.lfmPull[1], g.lfm.pull[1], cv)(pass)
      })
    }
    if (this.state.sourceType === 'fire') {
      enc.clearBuffer(g.spread)
      timed('massKeeping', pass => {
        one(p.conserveSpread, g.bg.conserveSpread, cc)(pass)
        one(p.conserveRatio, g.bg.conserveRatio, cc)(pass)
      })
    }
    const bim = this.bimocqOn()
    if (bim) {
      this.bimocqBegin(enc)
      timed('bimocq: maps', one(p.bimocqMaps, g.bimocq.maps[this.bimocqParity], cc))
      timed('bimocq: rebuild', one(p.bimocqRecon, g.bimocq.recon[this.bimocqParity], cc))
    } else {
      timed('advectScalars', one(p.advFwd, g.bg.advFwd, cc))
      timed('advectBack', one(p.advBwd, g.bg.advBwd, cc))
    }
    timed('combine', one(p.combine, g.bg.combine, cc))
    timed('mixing', one(p.diffuse, g.bg.diffuse, cc))
    if (bim) timed('bimocq: accumulate', one(p.bimocqAccum, g.bimocq.accum[this.bimocqParity], cc))
    if (this.upresActive()) {
      const up = this.ensureUpres()
      timed('upres', one(p.upres, up.group, [Math.ceil((2 * nx) / 4), Math.ceil((2 * ny) / 4), Math.ceil((2 * nz) / 4)]))
      timed('upres: coarse', one(p.upresCoarse, up.coarseGroup, cc))
      enc.copyTextureToTexture({ texture: up.tex[1] }, { texture: up.tex[0] }, [2 * nx, 2 * ny, 2 * nz])
    }
    timed('detail', one(p.advDetail, g.bg.advDetail, cc))
    timed('fireLight', pass => {
      one(p.fireSum, g.bg.fireSum, cc)(pass)
      one(p.fireTotal, g.bg.fireTotal, [1 + FIRE_BANDS, 1, 1])(pass)
    })
    enc.copyTextureToTexture({ texture: g.scal[1] }, { texture: g.scal[0] }, [nx, ny, nz])
    timed('light', one(p.light, this.upresActive() && this.upres ? this.upres.light : g.frame!.light, [Math.ceil(g.lightDims[0] / 4), Math.ceil(g.lightDims[1] / 4), Math.ceil(g.lightDims[2] / 4)]))
    timed('flameNear', one(p.flameNear, g.frame!.flameNear, [Math.ceil(g.nearDims[0] / 4), Math.ceil(g.nearDims[1] / 4), Math.ceil(g.nearDims[2] / 4)]))
    timed('floorNear', one(p.floorNear, g.frame!.floorNear, [Math.ceil(g.n[0] / 8), Math.ceil(g.n[2] / 8), 1]))
    timed('cyc', pass => {
      const cg: [number, number, number] = [Math.ceil(CYC_MAP.w / 8), Math.ceil(CYC_MAP.h / 8), 1]
      one(p.cycDirect, g.frame!.cycDirect, cg)(pass)
      one(p.cycBounce, g.frame!.cycBounce[0], cg)(pass)
      one(p.cycBounce, g.frame!.cycBounce[1], cg)(pass)
    })
    const k = labels.length
    labels.push('render')
    const t = this.ensureTargets()
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: t.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 },
    })
    rp.setViewport(0, 0, ...this.frameMarch, 0, 1)
    rp.setScissorRect(0, 0, ...this.frameMarch)
    rp.setPipeline(p.render)
    rp.setBindGroup(0, g.frame!.render)
    rp.draw(3)
    rp.end()
    labels.push('accumulate')
    if (this.state.glare > 0) labels.push('glare')
    labels.push('display')
    this.post(enc, screen, t, { qs, base: 2 * k + 2 })
    enc.resolveQuerySet(qs, 0, labels.length * 2, resolve, 0)
    enc.copyBufferToBuffer(resolve, 0, read, 0, labels.length * 16)
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const stamps = new BigInt64Array(read.getMappedRange())
    const out: Record<string, number> = {}
    labels.forEach((l, i) => {
      out[l] = Number(stamps[2 * i + 1] - stamps[2 * i]) / 1e6
    })
    read.unmap()
    qs.destroy()
    resolve.destroy()
    read.destroy()
    return out
  }

  // Development check of the pressure solve: one step, then the divergence of the velocity before the solve (left in
  // the right-hand side buffer) and after projection, as root-mean-square values in 1/s.
  async measureDivergence(): Promise<{ before: number; after: number; bad: number } | null> {
    if (!this.grids) return null
    const d = this.device
    const g = this.grids
    const p = this.pipes!
    const [nx, ny, nz] = g.n
    const bytes = nx * ny * nz * 4
    const rhs = g.buffers[0]
    const readBack = async () => {
      const read = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
      const enc = d.createCommandEncoder()
      enc.copyBufferToBuffer(rhs, 0, read, 0, bytes)
      d.queue.submit([enc.finish()])
      await read.mapAsync(GPUMapMode.READ)
      const a = new Float32Array(read.getMappedRange())
      let s = 0
      let k = 0
      for (let i = 0; i < a.length; i++) {
        if (Number.isFinite(a[i])) {
          s += a[i] * a[i]
          k++
        } else bad++
      }
      read.unmap()
      read.destroy()
      return Math.sqrt(s / Math.max(k, 1)) * SIM_DT // rhs = -div / dt
    }
    let bad = 0
    this.step()
    const before = await readBack()
    const enc = d.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(p.divergence)
    pass.setBindGroup(0, g.bg.divergence)
    pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4))
    pass.end()
    d.queue.submit([enc.finish()])
    const after = await readBack()
    return { before, after, bad }
  }

  // Development probe: the scalar fields (soot, temperature, reaction, flame) up the vertical line through the source.
  // For validation: run exactly `steps` simulation steps now (whether or not paused), then wait for the GPU.
  // For the device check: the values the frame and the light textures could not hold. Over the current frame and each
  // light texture, the texels holding a value that is not finite (infinity or NaN: a store past half floats' ceiling on
  // a GPU that does not clamp it, or a division gone wrong) and those held at the ceiling (65504), read by their bits so
  // no shader compiler can fold the test away.
  private checkPipes: { p2: GPUComputePipeline; p3: GPUComputePipeline } | null = null
  async checkValues(): Promise<{ frame: number; textures: number; ceiling: number; where: string[] } | null> {
    const g = this.grids
    const t = this.targets
    if (!g || !t) return null
    const d = this.device
    const code = (three: boolean) => `
@group(0) @binding(0) var tex: ${three ? 'texture_3d<f32>' : 'texture_2d<f32>'};
@group(0) @binding(1) var<storage, read_write> count: array<atomic<u32>, 2>;
@compute @workgroup_size(${three ? '4, 4, 4' : '8, 8'})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = textureDimensions(tex);
  if (any(${three ? 'id' : 'id.xy'} >= n)) { return; }
  let v = textureLoad(tex, ${three ? 'vec3<i32>(id)' : 'vec2<i32>(id.xy)'}, 0);
  var bad = false;
  var high = false;
  for (var c = 0u; c < 4u; c++) {
    let b = bitcast<u32>(v[c]);
    if ((b & 0x7f800000u) == 0x7f800000u) { bad = true; } else if (abs(v[c]) >= 65504.0) { high = true; }
  }
  if (bad) { atomicAdd(&count[0], 1u); }
  if (high) { atomicAdd(&count[1], 1u); }
}`
    if (!this.checkPipes) {
      const make = (three: boolean) => d.createComputePipelineAsync({ layout: 'auto', compute: { module: d.createShaderModule({ code: code(three) }), entryPoint: 'main' } })
      this.checkPipes = { p2: await make(false), p3: await make(true) }
    }
    const list: [string, GPUTexture][] = [
      ['frame', t.hdr],
      ['emitted light', g.emit],
      ["flames' near light", g.flameNear],
      ['its direction', g.flameNearDir],
      ["floor's near light", g.floorNear],
      ['light volume', g.light],
      ['light volume (rim, dome, floor)', g.light2],
      ["cyc's direct light", g.cycDirect],
      ["cyc's bounce", g.cycBounce[0]],
    ]
    const STRIDE = 256 // (storage bindings start at multiples of 256)
    const counts = d.createBuffer({ size: STRIDE * list.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    const read = d.createBuffer({ size: STRIDE * list.length, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    const pass = enc.beginComputePass()
    list.forEach(([, tex], i) => {
      const three = tex.dimension === '3d'
      const pipe = three ? this.checkPipes!.p3 : this.checkPipes!.p2
      pass.setPipeline(pipe)
      pass.setBindGroup(0, d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: { buffer: counts, offset: STRIDE * i, size: 8 } }] }))
      if (three) pass.dispatchWorkgroups(Math.ceil(tex.width / 4), Math.ceil(tex.height / 4), Math.ceil(tex.depthOrArrayLayers / 4))
      else pass.dispatchWorkgroups(Math.ceil(tex.width / 8), Math.ceil(tex.height / 8))
    })
    pass.end()
    enc.copyBufferToBuffer(counts, 0, read, 0, STRIDE * list.length)
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const u = new Uint32Array(read.getMappedRange().slice(0))
    read.unmap()
    read.destroy()
    counts.destroy()
    let textures = 0
    let ceiling = 0
    const where: string[] = []
    list.forEach(([name], i) => {
      const bad = u[(STRIDE / 4) * i]
      const high = u[(STRIDE / 4) * i + 1]
      if (i > 0) textures += bad
      ceiling += high
      if (bad) where.push(`${name} ${bad}`)
    })
    return { frame: u[0], textures, ceiling, where }
  }

  async advance(steps: number): Promise<number> {
    if (!this.grids) return 0
    for (let i = 0; i < steps; i++) this.step()
    await this.device.queue.onSubmittedWorkDone()
    this.needsRender = true
    return this.simTime
  }

  // For validation: the whole scalar grid as float32 (x fastest, then y, then z; four channels per cell), and the
  // grid sums (fire light, its centre, the bed's volume and the heat release rate).
  async readScalars(): Promise<{ n: Vec3; dx: number; t: number; data: Float32Array; sums: number[]; hrr: Float32Array } | null> {
    if (!this.grids) return null
    const d = this.device
    const [nx, ny, nz] = this.grids.n
    const rowBytes = Math.ceil((nx * 8) / 256) * 256
    const read = d.createBuffer({ size: rowBytes * ny * nz, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const sums = d.createBuffer({ size: 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const hrrRead = d.createBuffer({ size: nx * ny * nz * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.grids.hrr, 0, hrrRead, 0, nx * ny * nz * 4)
    enc.copyTextureToBuffer({ texture: this.grids.scal[0] }, { buffer: read, bytesPerRow: rowBytes, rowsPerImage: ny }, [nx, ny, nz])
    enc.copyBufferToBuffer(this.grids.fireLight, 0, sums, 0, 48)
    d.queue.submit([enc.finish()])
    await Promise.all([read.mapAsync(GPUMapMode.READ), sums.mapAsync(GPUMapMode.READ), hrrRead.mapAsync(GPUMapMode.READ)])
    const u16 = new Uint16Array(read.getMappedRange())
    const data = new Float32Array(nx * ny * nz * 4)
    for (let z = 0; z < nz; z++)
      for (let y = 0; y < ny; y++) {
        const o = (z * ny * rowBytes + y * rowBytes) / 2
        const out = ((z * ny + y) * nx) * 4
        for (let k = 0; k < nx * 4; k++) data[out + k] = halfToFloat(u16[o + k])
      }
    const out = { n: this.grids.n, dx: DOMAIN_WIDTH / nx, t: this.simTime, data, sums: [...new Float32Array(sums.getMappedRange())], hrr: new Float32Array(hrrRead.getMappedRange().slice(0)) }
    read.unmap()
    sums.unmap()
    hrrRead.unmap()
    read.destroy()
    sums.destroy()
    hrrRead.destroy()
    return out
  }

  // Development check of IVOCK: the vorticity lost in the last step, four floats a cell (x, y, z, unused).
  // Development check of the mass keeping: the ratio each cell's gas was scaled by in the last step (negative where it
  // expands).
  async readRatio(): Promise<Float32Array | null> {
    if (!this.grids) return null
    const d = this.device
    const bytes = this.grids.n[0] * this.grids.n[1] * this.grids.n[2] * 4
    const read = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.grids.ratio, 0, read, 0, bytes)
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(read.getMappedRange().slice(0))
    read.unmap()
    read.destroy()
    return out
  }

  async readIvock(): Promise<Float32Array | null> {
    if (!this.grids) return null
    const d = this.device
    const bytes = this.grids.n[0] * this.grids.n[1] * this.grids.n[2] * 16
    const read = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.grids.ivockRhs, 0, read, 0, bytes)
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(read.getMappedRange().slice(0))
    read.unmap()
    read.destroy()
    return out
  }

  async readFields(): Promise<PyroFields | null> {
    const sc = await this.readScalars()
    if (!sc || !this.grids) return null
    const d = this.device
    const [nx, ny, nz] = sc.n
    const cells = nx * ny * nz
    // The velocity, staggered on the cells' faces: (n + 1) per axis.
    const [vx, vy, vz] = [nx + 1, ny + 1, nz + 1]
    const rowBytes = Math.ceil((vx * 8) / 256) * 256
    const read = d.createBuffer({ size: rowBytes * vy * vz, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyTextureToBuffer({ texture: this.grids.vel[0] }, { buffer: read, bytesPerRow: rowBytes, rowsPerImage: vy }, [vx, vy, vz])
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const u16 = new Uint16Array(read.getMappedRange())
    const face = (x: number, y: number, z: number, k: number) => halfToFloat(u16[(z * vy * rowBytes + y * rowBytes) / 2 + x * 4 + k])
    const fire = this.state.sourceType === 'fire'
    const density = new Float32Array(cells)
    const temperature = new Float32Array(cells)
    const fuel = fire ? new Float32Array(cells) : null
    const heat = fire ? new Float32Array(cells) : null
    const vel = new Float32Array(cells * 3)
    const rhoA = RHO_T / T_AMBIENT
    // (The hottest a cell gets, as the solver caps it.)
    const rMin = T_AMBIENT / (T_AMBIENT + 2200)
    for (let z = 0; z < nz; z++)
      for (let y = 0; y < ny; y++)
        for (let x = 0; x < nx; x++) {
          const i = (z * ny + y) * nx + x
          const [soot, dy, , f] = sc.data.subarray(i * 4, i * 4 + 4)
          if (fire) {
            // As the renderer reads them: soot's partial density, the density deficit (T0 / T = 1 - deficit).
            density[i] = (SOOT_EXTINCTION * rhoA * Math.max(soot, 0)) / 1000
            temperature[i] = T_AMBIENT / Math.max(1 - Math.max(dy, 0), rMin)
            fuel![i] = Math.max(f, 0) / Math.max(1 - Math.max(dy, 0), rMin)
            heat![i] = sc.hrr[i]
          } else {
            density[i] = this.state.densityScale * Math.max(soot, 0)
            temperature[i] = T_AMBIENT + Math.max(dy, 0)
          }
          vel[i * 3] = 0.5 * (face(x, y, z, 0) + face(x + 1, y, z, 0))
          vel[i * 3 + 1] = 0.5 * (face(x, y, z, 1) + face(x, y + 1, z, 1))
          vel[i * 3 + 2] = 0.5 * (face(x, y, z, 2) + face(x, y, z + 1, 2))
        }
    read.unmap()
    read.destroy()
    const dx = sc.dx
    return { n: sc.n, dx, origin: [(-nx * dx) / 2 + dx / 2, dx / 2, (-nz * dx) / 2 + dx / 2], time: sc.t, fire, density, temperature, fuel, heat, vel }
  }

  // The simulation at this moment as an OpenVDB file (vdb.ts), y up, in metres: density (the extinction at 550 nm,
  // 1/m, a fog volume), temperature (K), vel (m/s) and, for a fire, fuel (the unburned fuel's mass fraction) and
  // heat_release (W/m3). Every grid stores the voxels where there is smoke, heat or fuel.
  async vdbFrame(): Promise<Uint8Array<ArrayBuffer> | null> {
    const f = await this.readFields()
    if (!f) return null
    const active = new Uint8Array(f.density.length)
    for (let i = 0; i < active.length; i++)
      active[i] = f.density[i] > 1e-4 || Math.abs(f.temperature[i] - T_AMBIENT) > 1 || (f.fuel !== null && f.fuel[i] > 1e-4) ? 1 : 0
    const grids: VdbGrid[] = [
      { name: 'density', data: f.density, fogVolume: true },
      { name: 'temperature', data: f.temperature, background: T_AMBIENT },
      { name: 'vel', data: f.vel, vector: true },
    ]
    if (f.fuel) grids.push({ name: 'fuel', data: f.fuel })
    if (f.heat) grids.push({ name: 'heat_release', data: f.heat })
    return writeVdb({ n: f.n, voxelSize: f.dx, origin: f.origin, active, grids, creator: 'mattebell.xyz pyro lab' })
  }

  // A run of frames as OpenVDB files, one per simulation step (1/30 s), each handed on as it is made. The simulation
  // holds still while each frame is read back, then carries on as it was. Stops early if cancelled; returns the
  // number of frames made.
  async vdbSequence(frames: number, onFrame: (file: Uint8Array<ArrayBuffer>, index: number) => void, cancelled?: () => boolean): Promise<number> {
    const paused = this.paused
    this.paused = true
    let made = 0
    try {
      for (let i = 0; i < frames && !cancelled?.(); i++) {
        if (i > 0) await this.advance(1)
        const file = await this.vdbFrame()
        if (!file) break
        onFrame(file, i)
        made++
      }
    } finally {
      this.paused = paused
      this.acc = 0
      this.kick()
    }
    return made
  }

  async probeColumn(): Promise<number[][] | null> {
    if (!this.grids) return null
    const d = this.device
    const [nx, ny, nz] = this.grids.n
    const rowBytes = Math.ceil((nx * 8) / 256) * 256
    const read = d.createBuffer({ size: rowBytes * ny * nz, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyTextureToBuffer({ texture: this.grids.scal[0] }, { buffer: read, bytesPerRow: rowBytes, rowsPerImage: ny }, [nx, ny, nz])
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const u16 = new Uint16Array(read.getMappedRange())
    const half = (h: number) => {
      const e = (h >> 10) & 31
      const f = h & 1023
      const v = e === 0 ? f * 2 ** -24 : e === 31 ? Infinity : (1 + f / 1024) * 2 ** (e - 15)
      return h & 0x8000 ? -v : v
    }
    const x = Math.floor(nx / 2)
    const z = Math.floor(nz / 2)
    const out: number[][] = []
    for (let y = 0; y < ny; y++) {
      const o = (z * ny * rowBytes + y * rowBytes) / 2 + x * 4
      out.push([0, 1, 2, 3].map(k => +half(u16[o + k]).toFixed(3)))
    }
    read.unmap()
    read.destroy()
    return out
  }

  // For the lab's stills: the current frame drawn at an exact size, as an image file.
  async snapshot(w: number, h: number, type = 'image/jpeg', quality = 0.9): Promise<Blob | null> {
    if (!this.grids) return null
    this.canvas.width = w
    this.canvas.height = h
    this.layoutCamera()
    // A still meters at once (no easing) and accumulates as a paused frame does, from a fresh history, marched at its
    // own size from the first frame.
    await this.readFire()
    // Disposed, or the device lost, while metering: no still.
    if (this.disposed || this.lost) return null
    this.meterEV = this.targetEV()
    const paused = this.paused
    this.paused = true
    this.forceMarch = [w, h]
    this.taaCount = 0
    for (let i = 0; i < TAA_SETTLE; i++) this.render()
    this.forceMarch = null
    this.paused = paused
    const blob = await new Promise<Blob | null>(resolve => this.canvas.toBlob(resolve, type, quality))
    this.applySize()
    return blob
  }

  // (Development: the running view, exactly. Frames of `steps` solver steps and a render each, as the running loop draws
  // them (the running history, the running render scale), metered at once as a still is; then the canvas as an image
  // file, taken before the frame is presented, and the engine left paused.)
  async runView(frames: number, steps = 1, type = 'image/png', quality = 0.95): Promise<Blob | null> {
    if (!this.grids) return null
    // (A reading already under way may be of the grids before a restart: wait it out, and read afresh.)
    while (this.metering) await new Promise(r => setTimeout(r, 10))
    await this.readFire()
    if (this.disposed || this.lost) return null
    this.meterEV = this.targetEV()
    this.paused = false
    this.sizeTo()
    this.layoutCamera()
    this.taaCount = 0
    for (let i = 0; i < frames; i++) {
      for (let k = 0; k < steps; k++) this.step()
      this.render()
    }
    // (The canvas is copied as toBlob is called; paused again before its encoding is awaited, so the frame loop takes no
    // steps meanwhile.)
    const blob = new Promise<Blob | null>(resolve => this.canvas.toBlob(resolve, type, quality))
    this.lastView = { canvas: [this.canvas.width, this.canvas.height], march: [this.marchW, this.marchH], t: this.simTime, ev: this.meterEV ?? 0 }
    this.paused = true
    this.applySize()
    return blob
  }
  lastView: { canvas: number[]; march: number[]; t: number; ev: number } | null = null

  // (Development: the paused view as a visitor sees it once it has settled. Paused where it stands, every frame drawn as
  // the frame loop would draw them until it stops; then the canvas. Frames drawn, for the probes.)
  async pausedView(type = 'image/png', quality = 0.95): Promise<Blob | null> {
    if (!this.grids) return null
    this.setPaused(true)
    let frames = 0
    while (this.needsRender && frames < 1000) {
      this.render()
      frames++
    }
    this.lastView = { canvas: [this.canvas.width, this.canvas.height], march: [...this.frameMarch], t: this.simTime, ev: frames }
    return new Promise<Blob | null>(resolve => this.canvas.toBlob(resolve, type, quality))
  }

  dispose() {
    this.disposed = true
    if (this.raf) cancelAnimationFrame(this.raf)
    this.raf = 0
    // The context, unless another engine on the canvas has since configured it for its own device.
    try {
      const cfg = typeof this.ctx.getConfiguration === 'function' ? this.ctx.getConfiguration() : undefined
      if (cfg === undefined || cfg?.device === this.device) this.ctx.unconfigure()
    } catch {}
    this.device.destroy()
    this.ownCanvas?.remove()
  }
}
