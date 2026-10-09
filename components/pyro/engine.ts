// The pyro lab's engine: a WebGPU gas solver (see wgsl.ts) stepped at a fixed 30 steps per simulated second,
// and a volume renderer drawn after each step (or when a shading control changes, so a paused simulation can be
// relit). WebGPU only: browsers without it get the lab's still frame.
import { ADVECT_DETAIL, ADVECT_SCAL, ADVECT_VEL, BALL_HEAT, PROJECT_REFLECT, RESOLVE, SKIN_H, SKIN_W, UPRES, VEL_COMBINE, COMBINE, CONSERVE_RATIO, CONSERVE_SPREAD, CURL, DIFFUSE, DIVERGENCE, FIRE_SUM, FIRE_SUM_TOTAL, FORCES, LIGHT, MG_PROLONG, MG_RESIDUAL, MG_RESTRICT, MG_SMOOTH, NOISE_GEN, PROJECT, RENDER, SOR } from './wgsl'
import { planckTable, whiteBalance } from './blackbody'
import { BLUE_NOISE_SIZE, blueNoise } from './bluenoise'
import { FIRE_BANDS, FUEL, G, METER_AMBIENT, METER_FIRE_CD, METER_FIRE_EV, METER_FIRE_SLOPE, RHO_T, SOOT_CHROMA, SOOT_EXTINCTION, T_AMBIENT, exposureOfEV, fireTurbulence, luxOfEV } from './physics'
import { writeVdb, type VdbGrid } from './vdb'
import { DOMAIN_WIDTH, PYRO_DEFAULTS, RESOLUTIONS, RESOLUTION_ORDER, type PyroResolution, type PyroState } from './params'

export const SIM_DT = 1 / 30
const MAX_STEPS_PER_FRAME = 2
const MAX_IN_FLIGHT = 2 // frames of GPU work queued at once: enough to keep the GPU busy, too few to stall the page
const SOR_OMEGA = 1.7
// Multigrid: Gauss-Seidel sweeps before and after each coarse correction, and on the coarsest level (over-relaxed
// there, where it is cheap); levels go down until a dimension would fall under 8 cells.
const MG_PRE = 2
const MG_POST = 2
const MG_COARSE = 16
const MG_COARSE_OMEGA = 1.6
const SPEED_LIMIT = 8 // m/s
const FOV_Y = (32 * Math.PI) / 180
const RENDER_SCALE_MIN = 0.5
const RENDER_SCALE_MAX = 0.75
// A default resolution that cannot keep up steps down: judged on the steps taken between 3 and 6 s of running
// (after start-up compiles settle), against 18 per second.
const AUTO_RES_FROM_S = 3
const AUTO_RES_TO_S = 6
const AUTO_RES_MIN_RATE = 18
const MAX_RENDER_PIXELS = 900_000
const STATUS_MS = 250
const STIR_RADIUS = 0.3 // m
const DETAIL_PERIOD = 1 // s of simulation between resets of each upres layer (half a period apart); the plume stretches them fast
const NOISE_SIZE = 64 // texels along each side of the tileable detail noise
const TAA_WEIGHT = 0.2 // the current frame's share of the accumulated image while simulating
const TAA_SETTLE = 24 // frames a paused image accumulates (a running average) before it stops redrawing
// The sun's and the sky's colours, each scaled to unit luminance (the EV controls set the illuminance).
const unitLum = (c: Vec3): Vec3 => c.map(v => v / (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])) as Vec3
const SUN_COLOUR = unitLum([1.0, 0.93, 0.84])
const SKY_COLOUR = unitLum([0.42, 0.52, 0.72])

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
}

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
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

interface Pipes {
  fireSum: GPUComputePipeline
  fireTotal: GPUComputePipeline
  noiseGen: GPUComputePipeline
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
  diffuse: GPUComputePipeline
  light: GPUComputePipeline
  render: GPURenderPipeline
  resolve: GPURenderPipeline
  upres: GPUComputePipeline
}

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
  spread: GPUBuffer // a fire's mass keeping: the volume over which each cell's gas was spread (fixed point)
  skin: GPUBuffer // the wooden ball's skin (SKIN_W x SKIN_H texels, four floats each), and its next step
  skinNext: GPUBuffer
  flame: GPUTexture // a fire's heat release rate per cell (MW/m3), for the renderer
  detA: [GPUTexture, GPUTexture]
  detB: [GPUTexture, GPUTexture]
  lightDims: Vec3
  light: GPUTexture
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
    diffuse: GPUBindGroup
    advDetail: GPUBindGroup
    fireSum: GPUBindGroup
    fireTotal: GPUBindGroup
    light: GPUBindGroup
    render: GPUBindGroup
  }
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

async function requestDevice(): Promise<{ adapter: GPUAdapter; device: GPUDevice } | null> {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) return null
  // Timestamps only for the development profiler (profileStep); the lab runs without them.
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [] })
  return { adapter, device }
}

async function compilePipes(d: GPUDevice, format: GPUTextureFormat): Promise<Pipes> {
  const compute = (code: string, constants?: Record<string, number>) =>
    d.createComputePipelineAsync({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: 'main', constants } })
  const renderModule = d.createShaderModule({ code: RENDER })
  const resolveModule = d.createShaderModule({ code: RESOLVE })
  const [fireSum, fireTotal, noiseGen, advDetail, mgSmooth, mgResidual, mgRestrict, mgProlong, advectVel, advectVelBack, advectVelHalf, projectReflect, velCombine, curl, forces, divergence, sor, project, advFwd, advBwd, combine, conserveSpread, conserveRatio, ballHeat, diffuse, light, upres, render, resolve] = await Promise.all([
    compute(FIRE_SUM),
    compute(FIRE_SUM_TOTAL),
    compute(NOISE_GEN),
    compute(ADVECT_DETAIL),
    compute(MG_SMOOTH),
    compute(MG_RESIDUAL),
    compute(MG_RESTRICT),
    compute(MG_PROLONG),
    compute(ADVECT_VEL, { DIR: 1 }),
    compute(ADVECT_VEL, { DIR: -1 }),
    compute(ADVECT_VEL, { DIR: 0.5 }),
    compute(PROJECT_REFLECT),
    compute(VEL_COMBINE),
    compute(CURL),
    compute(FORCES),
    compute(DIVERGENCE),
    compute(SOR),
    compute(PROJECT),
    compute(ADVECT_SCAL, { DIR: 1 }),
    compute(ADVECT_SCAL, { DIR: -1 }),
    compute(COMBINE),
    compute(CONSERVE_SPREAD),
    compute(CONSERVE_RATIO),
    compute(BALL_HEAT),
    compute(DIFFUSE),
    compute(LIGHT),
    compute(UPRES),
    d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: renderModule, entryPoint: 'vs' },
      fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    }),
    d.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: resolveModule, entryPoint: 'vs' },
      fragment: { module: resolveModule, entryPoint: 'fs', targets: [{ format }, { format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    }),
  ])
  return { fireSum, fireTotal, noiseGen, advDetail, mgSmooth, mgResidual, mgRestrict, mgProlong, advectVel, advectVelBack, advectVelHalf, projectReflect, velCombine, curl, forces, divergence, sor, project, advFwd, advBwd, combine, conserveSpread, conserveRatio, ballHeat, diffuse, light, upres, render, resolve }
}

export function prewarmPyro() {
  if (warm || typeof navigator === 'undefined' || !navigator.gpu) return
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
      // A device and pipelines made ahead of time, if they are ready and still usable; else new ones.
      const ahead = warm ? await warm : null
      warm = null
      const got = ahead && !ahead.lost ? ahead : await requestDevice()
      if (!got) {
        own?.remove()
        console.warn('Pyro: the browser offered no adapter')
        return null
      }
      const info = got.adapter.info
      const name = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ') || 'unnamed adapter'
      return new PyroEngine(own ?? canvas, own, ctx, got.device, onStatus, initial, onLost, name, ahead && !ahead.lost ? ahead.pipes : null)
    } catch (err) {
      own?.remove()
      console.warn('Pyro: WebGPU unavailable', err)
      return null
    }
  }

  readonly adapterName: string
  readonly whenReady: Promise<boolean>
  state: PyroState

  private format: GPUTextureFormat
  private sampler: GPUSampler
  private simBuf: GPUBuffer
  private renderBuf: GPUBuffer
  private parityBufs: [GPUBuffer, GPUBuffer]
  private simData = new ArrayBuffer(256)
  private renderData = new Float32Array(60)
  private wb: { kelvin: number; m: number[][] } | null = null
  private bbBuf: GPUBuffer
  private noiseTex: GPUTexture
  private blueTex: GPUTexture
  // Temporal accumulation: this frame's march (high dynamic range), the history it blends into (two, swapped each
  // frame), the blend's uniform, and how many frames have accumulated since anything changed.
  private targets: { w: number; h: number; hdr: GPUTexture; hist: [GPUTexture, GPUTexture]; groups: [GPUBindGroup, GPUBindGroup] } | null = null
  private taaBuf: GPUBuffer
  private histIndex = 0
  private taaCount = 0
  private renderFrame = 0
  // The wooden ball, as last read back for the HUD.
  private ballStatus: BallStatus | null = null
  private readingBall = false
  // Auto exposure: the fire's luminous intensity (cd, read back a few times a second) and the metered EV, eased.
  private fireCd = 0
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
  // needed; the pass's bind group and uniform; the render bind group that reads it; and whether it must refill from
  // the coarse density on its next step.
  private upres: { tex: [GPUTexture, GPUTexture]; group: GPUBindGroup; render: GPUBindGroup } | null = null
  private upresBuf: GPUBuffer
  private upresInit = true
  private upresWasActive = false
  private dummy3d: GPUTexture
  private noiseSampler: GPUSampler
  private pipes: Pipes | null = null
  private grids: Grids | null = null
  private disposed = false
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
  private renderScale = RENDER_SCALE_MAX
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
    ctx.configure({ device: d, format: this.format, alphaMode: 'opaque' })
    d.addEventListener('uncapturederror', ev => console.error(`Pyro WebGPU error: ${(ev as GPUUncapturedErrorEvent).error.message}`))
    d.lost.then(info => {
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
    this.blueTex = d.createTexture({ size: [BLUE_NOISE_SIZE, BLUE_NOISE_SIZE], format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })
    d.queue.writeTexture({ texture: this.blueTex }, blueNoise(), { bytesPerRow: BLUE_NOISE_SIZE }, [BLUE_NOISE_SIZE, BLUE_NOISE_SIZE])
    this.taaBuf = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST })
    this.ballBuf = d.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST })
    this.upresBuf = d.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST })
    this.dummy3d = d.createTexture({ size: [1, 1, 1], dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING })
    this.whenReady = this.build(prebuilt)
  }

  // Compiles every pipeline in the background, then makes the grids.
  private async build(prebuilt: Pipes | null) {
    const pipes = prebuilt ?? (await compilePipes(this.device, this.format))
    if (this.disposed) return false
    this.pipes = pipes
    // The detail noise, made once.
    const enc = this.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pipes.noiseGen)
    pass.setBindGroup(0, this.device.createBindGroup({ layout: pipes.noiseGen.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.noiseTex.createView() }] }))
    pass.dispatchWorkgroups(NOISE_SIZE / 4, NOISE_SIZE / 4, NOISE_SIZE / 4)
    pass.end()
    this.device.queue.submit([enc.finish()])
    this.makeGrids()
    this.layoutCamera()
    this.needsRender = true
    this.kick()
    return true
  }

  // The grids for the current resolution, zeroed, and every bind group over them.
  private makeGrids() {
    const d = this.device
    const p = this.pipes!
    this.grids?.textures.forEach(t => t.destroy())
    this.upres?.tex.forEach(t => t.destroy())
    this.upres = null
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
    const vel: [GPUTexture, GPUTexture, GPUTexture, GPUTexture] = [tex(velSize, true), tex(velSize, true), tex(velSize), tex(velSize, true)]
    const scal = [tex(n, true), tex(n, true), tex(n), tex(n)] as [GPUTexture, GPUTexture, GPUTexture, GPUTexture]
    const detA: [GPUTexture, GPUTexture] = [tex(n, true), tex(n, true)]
    const detB: [GPUTexture, GPUTexture] = [tex(n, true), tex(n, true)]
    const curl = tex(n)
    // A fire's heat release rate per cell (MW/m3), filterable, for rendering the flame sheets.
    const flame = tex(n)
    const lightDims: Vec3 = [Math.ceil(n[0] / 2), Math.ceil(n[1] / 2), Math.ceil(n[2] / 2)]
    const light = tex(lightDims)
    const v = (t: GPUTexture) => t.createView()
    const group = (pipe: GPUComputePipeline | GPURenderPipeline, resources: GPUBindingResource[]) =>
      d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({ binding, resource })) })
    const cells = n[0] * n[1] * n[2]
    const div = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    const pressure = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE })
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
    // The fire's light: per-workgroup partial sums, then the total (intensity and centre), read by the renderer.
    const groups = Math.ceil(n[0] / 4) * Math.ceil(n[1] / 4) * Math.ceil(n[2] / 4)
    const firePartial = storage(groups * 12)
    const fireLight = d.createBuffer({ size: (3 + 2 * FIRE_BANDS) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    // The heat release rate in each cell (W/m3), from the step's burning: for the expansion, the sums and validation.
    const hrr = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    // A fire's mass keeping: the volume each cell's gas was spread over, and the ratio its advection is scaled by.
    const spread = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    const ratio = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE })
    buffers.push(fireLight, hrr, spread, ratio)
    // The wooden ball's skin, fresh wood at the air's temperature, and the buffer its heating step writes.
    const skinBytes = SKIN_W * SKIN_H * 16
    const skin = d.createBuffer({ size: skinBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
    const skinNext = d.createBuffer({ size: skinBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    buffers.push(skin, skinNext)
    // The kernels that read the skin (binding 31: the ball's volatiles feed the fire, and the renderer draws it).
    const withSkin = (pipe: GPUComputePipeline | GPURenderPipeline, resources: GPUBindingResource[]) =>
      d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [...resources.map((resource, binding) => ({ binding, resource })), { binding: 31, resource: { buffer: skin } }],
      })
    const fireCount = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true })
    new Uint32Array(fireCount.getMappedRange()).set([groups, Math.ceil(n[0] / 4), Math.ceil(n[1] / 4), Math.ceil(n[2] / 4)])
    fireCount.unmap()
    buffers.push(fireCount)
    const P = { buffer: this.simBuf }
    const R = { buffer: this.renderBuf }
    const s = this.sampler
    this.grids = {
      n,
      textures: [...vel, ...scal, ...detA, ...detB, curl, light, flame],
      buffers,
      levels,
      vel,
      scal,
      fireLight,
      hrr,
      spread,
      skin,
      skinNext,
      flame,
      detA,
      detB,
      lightDims,
      light,
      bg: {
        advectVel: group(p.advectVel, [P, s, v(vel[0]), v(vel[0]), v(vel[3])]),
        advectVelFwd: group(p.advectVel, [P, s, v(vel[0]), v(vel[0]), v(vel[1])]),
        advectVelBack: group(p.advectVelBack, [P, s, v(vel[0]), v(vel[1]), v(vel[2])]),
        velCombine: group(p.velCombine, [P, s, v(vel[0]), v(vel[1]), v(vel[2]), v(vel[3])]),
        arAdvect1: group(p.advectVelHalf, [P, s, v(vel[0]), v(vel[0]), v(vel[3])]),
        arAdvect2: group(p.advectVelHalf, [P, s, v(vel[1]), v(vel[2]), v(vel[3])]),
        projectReflect: group(p.projectReflect, [P, v(vel[0]), { buffer: pressure }, v(vel[1]), v(vel[2])]),
        curl: group(p.curl, [P, v(vel[3]), v(curl)]),
        forces: group(p.forces, [P, v(vel[3]), v(scal[0]), v(curl), v(vel[0])]),
        divergence: withSkin(p.divergence, [P, v(vel[0]), { buffer: div }, v(scal[0]), { buffer: fireLight }, { buffer: hrr }]),
        sor: [0, 1].map(k => group(p.sor, [P, { buffer: this.parityBufs[k] }, { buffer: div }, { buffer: pressure }])) as [GPUBindGroup, GPUBindGroup],
        project: group(p.project, [P, v(vel[0]), { buffer: pressure }, v(vel[1])]),
        advFwd: group(p.advFwd, [P, s, v(vel[0]), v(scal[0]), v(scal[1])]),
        advBwd: group(p.advBwd, [P, s, v(vel[0]), v(scal[1]), v(scal[2])]),
        combine: withSkin(p.combine, [P, s, v(vel[0]), v(scal[0]), v(scal[1]), v(scal[2]), v(scal[3]), { buffer: fireLight }, { buffer: hrr }, { buffer: ratio }, v(flame)]),
        conserveSpread: group(p.conserveSpread, [P, s, v(vel[0]), { buffer: spread }]),
        conserveRatio: withSkin(p.conserveRatio, [P, v(scal[0]), { buffer: fireLight }, { buffer: hrr }, { buffer: spread }, { buffer: ratio }, s, v(vel[0])]),
        ballHeat: withSkin(p.ballHeat, [P, s, v(scal[0]), { buffer: fireLight }, { buffer: skinNext }]),
        diffuse: group(p.diffuse, [P, v(vel[0]), v(scal[3]), v(scal[1])]),
        advDetail: group(p.advDetail, [P, s, v(vel[0]), v(detA[0]), v(detB[0]), v(detA[1]), v(detB[1])]),
        fireSum: group(p.fireSum, [P, R, v(scal[0]), { buffer: this.bbBuf }, { buffer: firePartial }, { buffer: hrr }]),
        fireTotal: group(p.fireTotal, [{ buffer: fireCount }, { buffer: firePartial }, { buffer: fireLight }]),
        light: group(p.light, [P, R, s, v(scal[0]), v(light), { buffer: fireLight }]),
        render: withSkin(p.render, [P, R, s, v(scal[0]), v(light), v(detA[0]), v(detB[0]), this.noiseTex.createView(), this.noiseSampler, { buffer: this.bbBuf }, { buffer: fireLight }, this.blueTex.createView(), this.dummy3d.createView(), v(flame)]),
      },
    }
    this.resetSkin()
    this.simTime = 0
    this.frame = 0
    this.lightDirty = true
    this.needsRender = true
    this.runClock = 0
    this.autoSteps = 0
    this.stepTimes = []
    this.renderScale = RENDER_SCALE_MAX
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
    if ((this.state.ball && !prev.ball) || (this.state.ballBurns && !prev.ballBurns)) this.resetSkin()
    // Shading changes redraw (and relight) at once, even while paused, and accumulate afresh.
    this.lightDirty = true
    this.needsRender = true
    this.taaCount = 0
    this.kick()
  }

  setPaused(paused: boolean) {
    this.paused = paused
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

  // The canvas follows the frame; the image renders at a fraction of its device pixels (smoke is soft).
  resize(cssW: number, cssH: number, dpr: number) {
    this.cssW = cssW
    this.cssH = cssH
    this.dpr = dpr
    this.applySize()
  }

  private applySize() {
    const base = Math.min(this.dpr, 1.5) * this.renderScale
    let w = Math.max(1, Math.round(this.cssW * base))
    let h = Math.max(1, Math.round(this.cssH * base))
    if (w * h > MAX_RENDER_PIXELS) {
      const k = Math.sqrt(MAX_RENDER_PIXELS / (w * h))
      w = Math.round(w * k)
      h = Math.round(h * k)
    }
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
    }
    this.layoutCamera()
    this.needsRender = true
    this.kick()
  }

  // A camera a little below the plume's middle, looking along -z, as close as framing the domain's height allows, and
  // closer still on wide frames, until the domain spans the frame's width (smoke then leaves through the frame's
  // edges, never a side of the box in view).
  private layoutCamera() {
    if (!this.grids) return
    const [nx, ny] = this.grids.n
    const dx = DOMAIN_WIDTH / nx
    const h = ny * dx
    const aspect = this.canvas.width / Math.max(1, this.canvas.height)
    const tanY = Math.tan(FOV_Y / 2)
    const dist = Math.min((h * 0.525) / tanY, (DOMAIN_WIDTH * 0.49) / (tanY * aspect))
    const target: Vec3 = [0, h * 0.47, 0]
    const pos: Vec3 = [0, h * 0.3, dist]
    const fwd = norm(sub(target, pos))
    const right = norm(cross(fwd, [0, 1, 0]))
    const up = cross(right, fwd)
    this.cam = { pos, fwd, right, up, tanY, aspect }
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
    this.taaCount = 0
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
    this.taaCount = 0
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
    // World metres per ndc unit at the domain's centre plane.
    const [nx, ny] = this.grids.n
    const dist = Math.hypot(c.pos[0], c.pos[1] - (ny * DOMAIN_WIDTH) / nx * 0.47, c.pos[2])
    const mx = dxNdc * dist * c.tanY * c.aspect
    const my = dyNdc * dist * c.tanY
    const secs = Math.max(dtMs, 8) / 1000
    const gain = 0.45
    let vel: Vec3 = [(c.right[0] * mx + c.up[0] * my) / secs * gain, (c.right[1] * mx + c.up[1] * my) / secs * gain, (c.right[2] * mx + c.up[2] * my) / secs * gain]
    const speed = Math.hypot(...vel)
    if (speed > 3) vel = vel.map(v => (v * 3) / speed) as Vec3
    const o = c.pos
    const t = Math.max(0, -(o[0] * dir[0] + o[2] * dir[2]) / Math.max(dir[0] * dir[0] + dir[2] * dir[2], 1e-6))
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
    const p = this.pipes!
    this.writeSim()
    const enc = this.device.createCommandEncoder()
    // The wooden ball's skin heats (or cools) first, so this step's fire takes in its volatiles.
    if (this.ballBurning()) {
      const skinPass = enc.beginComputePass()
      skinPass.setPipeline(p.ballHeat)
      skinPass.setBindGroup(0, g.bg.ballHeat)
      skinPass.dispatchWorkgroups(SKIN_W / 8, SKIN_H / 8, 1)
      skinPass.end()
      enc.copyBufferToBuffer(g.skinNext, 0, g.skin, 0, SKIN_W * SKIN_H * 16)
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
    // step, projects, reflects, advects the reflection the second half and projects again.
    const scheme = this.state.velocity
    let pass = enc.beginComputePass()
    if (scheme === 'ar') {
      run(pass, p.advectVelHalf, g.bg.arAdvect1, cv)
    } else if (scheme === 'mac') {
      run(pass, p.advectVel, g.bg.advectVelFwd, cv)
      run(pass, p.advectVelBack, g.bg.advectVelBack, cv)
      run(pass, p.velCombine, g.bg.velCombine, cv)
    } else {
      run(pass, p.advectVel, g.bg.advectVel, cv)
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
    pass.end()
    enc.copyTextureToTexture({ texture: g.vel[1] }, { texture: g.vel[0] }, [nx + 1, ny + 1, nz + 1])
    // A fire keeps its mass through the advection (the scalars' ratios, 7b).
    const fire = this.state.sourceType === 'fire'
    if (fire) enc.clearBuffer(g.spread)
    pass = enc.beginComputePass()
    if (fire) {
      run(pass, p.conserveSpread, g.bg.conserveSpread, cc)
      run(pass, p.conserveRatio, g.bg.conserveRatio, cc)
    }
    run(pass, p.advFwd, g.bg.advFwd, cc)
    if (this.state.advection === 'mac') run(pass, p.advBwd, g.bg.advBwd, cc)
    run(pass, p.combine, g.bg.combine, cc)
    run(pass, p.diffuse, g.bg.diffuse, cc)
    // The upres grid refills from the coarse density whenever it starts (a new source, or switched on).
    const upActive = this.upresActive()
    if (upActive && !this.upresWasActive) this.upresInit = true
    this.upresWasActive = upActive
    const up = upActive ? this.ensureUpres() : null
    if (up) {
      run(pass, p.upres, up.group, [Math.ceil((2 * nx) / 4), Math.ceil((2 * ny) / 4), Math.ceil((2 * nz) / 4)])
    }
    run(pass, p.advDetail, g.bg.advDetail, cc)
    pass.end()
    if (up) enc.copyTextureToTexture({ texture: up.tex[1] }, { texture: up.tex[0] }, [2 * nx, 2 * ny, 2 * nz])
    enc.copyTextureToTexture({ texture: g.scal[1] }, { texture: g.scal[0] }, [nx, ny, nz])
    enc.copyTextureToTexture({ texture: g.detA[1] }, { texture: g.detA[0] }, [nx, ny, nz])
    enc.copyTextureToTexture({ texture: g.detB[1] }, { texture: g.detB[0] }, [nx, ny, nz])
    // The fire's light from the new scalars (zero without flames).
    pass = enc.beginComputePass()
    run(pass, p.fireSum, g.bg.fireSum, cc)
    run(pass, p.fireTotal, g.bg.fireTotal, [1 + FIRE_BANDS, 1, 1])
    pass.end()
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
    const p = this.pipes!
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
    const L = g.levels
    const cells = (m: Vec3): Vec3 => [Math.ceil(m[0] / 4), Math.ceil(m[1] / 4), Math.ceil(m[2] / 4)]
    const smooth = (l: number, sweeps: number) => {
      const m = L[l].n
      const wg: Vec3 = [Math.ceil(m[0] / 2 / 16), Math.ceil(m[1] / 4), m[2]]
      for (let i = 0; i < sweeps; i++) {
        dispatch(p.mgSmooth, L[l].smooth[0], wg)
        dispatch(p.mgSmooth, L[l].smooth[1], wg)
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
    for (let k = 0; k < this.state.vcycles; k++) vcycle(0)
  }

  private writeSim() {
    const s = this.state
    const g = this.grids!
    const [nx, ny, nz] = g.n
    const dx = DOMAIN_WIDTH / nx
    const u = new Uint32Array(this.simData)
    const f = new Float32Array(this.simData)
    u.set([nx, ny, nz, this.frame], 0)
    f.set([dx, SIM_DT, this.simTime, 0], 4)
    // A smoke source's dome rests on the floor (it is 2.5 times wider than tall); a fire's bed is a disc on the floor.
    f.set([s.srcX, s.srcRadius / 2.5 + dx, 0, s.srcRadius], 8)
    f.set([s.srcDensity, s.srcTemp, s.srcSpeed, s.srcNoise], 12)
    // Buoyancy: g (T - T_ambient) / T_ambient per kelvin of temperature rise, times the control (1 = physical).
    // Turbulence: m/s2 for a smoke source; for a fire, the turbulence model (physics.ts) times the control.
    const turb = s.sourceType === 'fire' ? s.turbulence * fireTurbulence(s.heatRelease, s.turbScale) : s.turbulence
    f.set([(s.buoyancy * G) / T_AMBIENT, 0, s.vorticity, turb], 16)
    f.set([s.turbScale, s.wind, 0, 0.05], 20)
    f.set([s.cooling, s.dissipation, SOR_OMEGA, SPEED_LIMIT], 24)
    const st = this.stir
    f.set([...(st ? st.centre : [0, 0, 0]), STIR_RADIUS], 28)
    f.set([...(st ? st.vel : [0, 0, 0]), st ? 1 : 0], 32)
    f.set([0, 0, 0, s.advection === 'mac' ? 1 : 0], 36)
    f.set([(-nx * dx) / 2, 0, (-nz * dx) / 2, 0], 40)
    // Each upres layer resets when the simulation clock crosses its period (layer B half a period later).
    const t0 = this.simTime / DETAIL_PERIOD
    const t1 = (this.simTime + SIM_DT) / DETAIL_PERIOD
    f.set([Math.floor(t1) > Math.floor(t0) ? 1 : 0, Math.floor(t1 + 0.5) > Math.floor(t0 + 0.5) ? 1 : 0, 0, 0], 44)
    // A fire bed gives off fuel at the rate that releases the chosen heat. Fuel and air mix (and so burn) in
    // sqrt(2 dx / g), the buoyant mixing time FDS uses at a cell's scale. The bed's nominal area (a disc, thinned by
    // the source noise) spreads the fuel until the area is measured.
    const fire = s.sourceType === 'fire'
    f.set([fire ? 1 : 0, fire ? (s.heatRelease * 1000) / FUEL.heatPerFuel : 0, s.sootYield, s.radiantFraction], 48)
    const bed = Math.PI * s.srcRadius ** 2 * (1 - 0.5 * s.srcNoise)
    f.set([s.expansion, Math.sqrt((2 * dx) / G), bed, 0], 52)
    // The collision ball, easing to rest once the visitor lets go.
    if (performance.now() - this.ballDragAt > 120) this.ballVel = this.ballVel.map(v => v * 0.6) as Vec3
    const radius = s.ball ? s.ballRadius : 0
    f.set([...this.ballPos, radius], 56)
    f.set([...this.ballVel, s.ball && s.ballBurns ? 1 : 0], 60)
    this.device.queue.writeBuffer(this.simBuf, 0, this.simData)
    this.device.queue.writeBuffer(this.ballBuf, 0, new Float32Array([...this.ballPos, radius, (-nx * dx) / 2, 0, (-nz * dx) / 2, dx]))
  }

  private writeRender() {
    const s = this.state
    const c = this.cam
    const az = (s.lightAz * Math.PI) / 180
    const el = (s.lightEl * Math.PI) / 180
    // Toward the sun; azimuth 0 puts it behind the camera (+z), positive swings it to camera right.
    const sun: Vec3 = [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)]
    const fire = s.sourceType === 'fire'
    if (this.wb?.kelvin !== s.whiteBalance) this.wb = { kelvin: s.whiteBalance, m: whiteBalance(s.whiteBalance) }
    const wb = this.wb.m
    const r = this.renderData
    r.set([...c.pos, c.tanY], 0)
    r.set([...c.fwd, c.aspect], 4)
    r.set([...c.right, this.renderFrame % 4096], 8)
    r.set([...c.up, exposureOfEV(this.exposureEV())], 12)
    r.set([...sun, s.anisotropy], 16)
    r.set([...SUN_COLOUR.map(v => v * luxOfEV(s.sunEV)), s.densityScale], 20)
    r.set([...SKY_COLOUR.map(v => v * luxOfEV(s.skyEV)), s.albedo], 24)
    r.set([0.14, 0.14, 0.14, 3.2], 28)
    r.set([this.canvas.width, this.canvas.height, this.simTime, fire ? s.flameDetail : 0], 32)
    // Cross-fade weights: each layer is weighted by a triangle of its age, 0 when it was just reset, 1 half a period on.
    const ageA = (this.simTime / DETAIL_PERIOD) % 1
    const ageB = (this.simTime / DETAIL_PERIOD + 0.5) % 1
    r.set([1 - Math.abs(2 * ageA - 1), 1 - Math.abs(2 * ageB - 1), s.detail, 1 / (8 * s.detailScale)], 36)
    // A flame's eddies rise at about half the speed of the gas in the flame, 1.9 Q^(1/5) m/s with Q in kW
    // (McCaffrey 1979).
    r.set([fire ? 1 : 0, fire ? SOOT_CHROMA[0] : 1, fire ? SOOT_CHROMA[1] : 1, 0.5 * 1.9 * s.heatRelease ** 0.2], 40)
    r.set([...wb[0], s.wind], 44)
    r.set([...wb[1], s.tricubic ? 1 : 0], 48)
    r.set([...wb[2], this.upresActive() && this.upres ? 1 : 0], 52)
    r.set([s.background === 'sky' ? 1 : 0, 0, 0, 0], 56)
    this.device.queue.writeBuffer(this.renderBuf, 0, r)
  }

  // The accumulation targets at the canvas's size (made again, and the history dropped, when it changes).
  private ensureTargets() {
    const w = this.canvas.width
    const h = this.canvas.height
    if (this.targets && this.targets.w === w && this.targets.h === h) return this.targets
    if (this.targets) {
      this.targets.hdr.destroy()
      this.targets.hist.forEach(t => t.destroy())
    }
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    const tex = () => this.device.createTexture({ size: [w, h], format: 'rgba16float', usage })
    const hdr = tex()
    const hist: [GPUTexture, GPUTexture] = [tex(), tex()]
    const p = this.pipes!
    const group = (k: number) =>
      this.device.createBindGroup({
        layout: p.resolve.getBindGroupLayout(0),
        entries: [
          { binding: 1, resource: { buffer: this.renderBuf } },
          { binding: 2, resource: hdr.createView() },
          { binding: 3, resource: hist[k].createView() },
          { binding: 4, resource: { buffer: this.taaBuf } },
        ],
      })
    this.targets = { w, h, hdr, hist, groups: [group(0), group(1)] }
    this.taaCount = 0
    return this.targets
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
      const v = (t: GPUTexture) => t.createView()
      const entries = (res: GPUBindingResource[]) => res.map((resource, binding) => ({ binding, resource }))
      const group = d.createBindGroup({
        layout: p.upres.getBindGroupLayout(0),
        entries: entries([{ buffer: this.simBuf }, this.sampler, v(g.vel[0]), v(tex[0]), v(tex[1]), v(g.detA[0]), v(g.detB[0]), this.noiseTex.createView(), this.noiseSampler, v(g.scal[0]), { buffer: this.upresBuf }]),
      })
      const render = d.createBindGroup({
        layout: p.render.getBindGroupLayout(0),
        entries: [
          ...entries([{ buffer: this.simBuf }, { buffer: this.renderBuf }, this.sampler, v(g.scal[0]), v(g.light), v(g.detA[0]), v(g.detB[0]), this.noiseTex.createView(), this.noiseSampler, { buffer: this.bbBuf }, { buffer: g.fireLight }, this.blueTex.createView(), v(tex[0]), v(g.flame)]),
          { binding: 31, resource: { buffer: g.skin } },
        ],
      })
      this.upres = { tex, group, render }
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

  private render() {
    const g = this.grids!
    const p = this.pipes!
    this.needsRender = false
    this.writeRender()
    const t = this.ensureTargets()
    // The current frame's weight: a running average while nothing changes (paused), a fixed share while simulating.
    const weight = this.taaCount === 0 ? 1 : this.paused ? Math.max(1 / (this.taaCount + 1), 1 / TAA_SETTLE) : TAA_WEIGHT
    this.device.queue.writeBuffer(this.taaBuf, 0, new Float32Array([this.taaCount > 0 ? 1 : 0, weight, 0, 0]))
    const enc = this.device.createCommandEncoder()
    if (this.lightDirty) {
      const pass = enc.beginComputePass()
      pass.setPipeline(p.light)
      pass.setBindGroup(0, g.bg.light)
      pass.dispatchWorkgroups(Math.ceil(g.lightDims[0] / 4), Math.ceil(g.lightDims[1] / 4), Math.ceil(g.lightDims[2] / 4))
      pass.end()
      this.lightDirty = false
    }
    let pass = enc.beginRenderPass({
      colorAttachments: [{ view: t.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    })
    pass.setPipeline(p.render)
    pass.setBindGroup(0, this.upresActive() && this.upres ? this.upres.render : g.bg.render)
    pass.draw(3)
    pass.end()
    // Read the last history, write the next.
    const next = 1 - this.histIndex
    pass = enc.beginRenderPass({
      colorAttachments: [
        { view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        { view: t.hist[next].createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    })
    pass.setPipeline(p.resolve)
    pass.setBindGroup(0, t.groups[this.histIndex])
    pass.draw(3)
    pass.end()
    this.device.queue.submit([enc.finish()])
    this.histIndex = next
    this.taaCount++
    this.renderFrame++
    // Paused, keep drawing until the history has settled.
    if (this.paused && this.taaCount < TAA_SETTLE) this.needsRender = true
  }


  // When the display's frames come slower than 40 a second for a second while simulating (the GPU is the
  // bottleneck), render fewer pixels, down to half.
  private adapt() {
    const now = this.frameTimes[this.frameTimes.length - 1]
    while (this.frameTimes.length && now - this.frameTimes[0] > 1000) this.frameTimes.shift()
    if (this.paused || this.frameTimes.length < 10) return
    const spanMs = now - this.frameTimes[0]
    const fps = ((this.frameTimes.length - 1) * 1000) / Math.max(spanMs, 1)
    if (fps < 40 && this.renderScale > RENDER_SCALE_MIN) {
      this.renderScale = Math.max(RENDER_SCALE_MIN, this.renderScale * 0.85)
      this.frameTimes = []
      this.applySize()
    } else if (fps > 55 && this.renderScale < RENDER_SCALE_MAX) {
      this.renderScale = Math.min(RENDER_SCALE_MAX, this.renderScale / 0.85)
      this.frameTimes = []
      this.applySize()
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
    const ambient = Math.log2((METER_AMBIENT * (luxOfEV(s.sunEV) + luxOfEV(s.skyEV))) / 2.5)
    if (s.sourceType !== 'fire' || this.fireCd <= 0) return ambient
    return Math.max(ambient, METER_FIRE_EV + METER_FIRE_SLOPE * Math.log2(this.fireCd / METER_FIRE_CD))
  }
  private exposureEV() {
    const s = this.state
    if (s.exposureMode === 'manual') return s.exposure
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

  // A fresh wooden ball: unburned, at the air's temperature.
  private resetSkin() {
    if (!this.grids) return
    const fresh = new Float32Array(SKIN_W * SKIN_H * 4)
    for (let i = 0; i < SKIN_W * SKIN_H; i++) fresh.set([T_AMBIENT, T_AMBIENT, 0, 0], i * 4)
    this.device.queue.writeBuffer(this.grids.skin, 0, fresh)
    this.ballStatus = null
    this.needsRender = true
    this.taaCount = 0
  }

  // Swaps the charred ball for a fresh one.
  newBall() {
    this.resetSkin()
    this.kick()
    this.report(true)
  }

  // The wooden ball's state for the HUD, as the solver works it out (see charRate and ballFuelFlux in wgsl.ts).
  private async readBall() {
    if (this.readingBall || !this.grids || !this.ballBurning()) {
      if (!this.ballBurning()) this.ballStatus = null
      return
    }
    this.readingBall = true
    const d = this.device
    const bytes = SKIN_W * SKIN_H * 16
    const read = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.grids.skin, 0, read, 0, bytes)
    d.queue.submit([enc.finish()])
    try {
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
      for (let j = 0; j < SKIN_H; j++) {
        const texel = r * r * Math.sin((Math.PI * (j + 0.5)) / SKIN_H) * (Math.PI / SKIN_H) * ((2 * Math.PI) / SKIN_W)
        for (let i = 0; i < SKIN_W; i++) {
          const k = (j * SKIN_W + i) * 4
          const [t1, t2, char] = [a[k], a[k + 1], a[k + 2]]
          hottest = Math.max(hottest, t1)
          const front = t1 + (t2 - t1) * Math.min(Math.max(char / 4, 0), 1)
          const m = 500 * (1 - 0.16) * Math.min(9.5 * Math.exp(-8230 / Math.max(front, 300)), 6e-5)
          fuel += m * smooth(0.0015, 0.0025, m) * texel
          if (char > 0.3) charredArea += texel
          area += texel
        }
      }
      read.unmap()
      if (this.ballBurning()) this.ballStatus = { surfaceK: hottest, burningKW: (fuel * FUEL.heatPerFuel) / 1000, charred: charredArea / area }
    } catch {}
    read.destroy()
    this.readingBall = false
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
    })
  }

  // Development profiler: one step and one frame, each kernel in its own pass with GPU timestamps; milliseconds per
  // kernel (all SOR sweeps together). Needs the timestamp-query feature.
  async profileStep(): Promise<Record<string, number> | null> {
    if (!this.grids || !this.device.features.has('timestamp-query')) return null
    const d = this.device
    const g = this.grids
    const p = this.pipes!
    const labels: string[] = []
    const slots = 64
    const qs = d.createQuerySet({ type: 'timestamp', count: slots * 2 })
    const resolve = d.createBuffer({ size: slots * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    const read = d.createBuffer({ size: slots * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    this.writeSim()
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
    const scheme = this.state.velocity
    const solveLabel = this.state.pressure === 'sor' ? `sor x${this.state.pressureIters}` : `multigrid x${this.state.vcycles}`
    if (this.ballBurning()) {
      timed('ballHeat', one(p.ballHeat, g.bg.ballHeat, [SKIN_W / 8, SKIN_H / 8, 1]))
      enc.copyBufferToBuffer(g.skinNext, 0, g.skin, 0, SKIN_W * SKIN_H * 16)
    }
    if (scheme === 'ar') {
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
    enc.copyTextureToTexture({ texture: g.vel[1] }, { texture: g.vel[0] }, [nx + 1, ny + 1, nz + 1])
    if (this.state.sourceType === 'fire') {
      enc.clearBuffer(g.spread)
      timed('massKeeping', pass => {
        one(p.conserveSpread, g.bg.conserveSpread, cc)(pass)
        one(p.conserveRatio, g.bg.conserveRatio, cc)(pass)
      })
    }
    timed('advectScalars', one(p.advFwd, g.bg.advFwd, cc))
    timed('advectBack', one(p.advBwd, g.bg.advBwd, cc))
    timed('combine', one(p.combine, g.bg.combine, cc))
    timed('mixing', one(p.diffuse, g.bg.diffuse, cc))
    if (this.upresActive()) {
      const up = this.ensureUpres()
      timed('upres', one(p.upres, up.group, [Math.ceil((2 * nx) / 4), Math.ceil((2 * ny) / 4), Math.ceil((2 * nz) / 4)]))
      enc.copyTextureToTexture({ texture: up.tex[1] }, { texture: up.tex[0] }, [2 * nx, 2 * ny, 2 * nz])
    }
    timed('detail', one(p.advDetail, g.bg.advDetail, cc))
    timed('fireLight', pass => {
      one(p.fireSum, g.bg.fireSum, cc)(pass)
      one(p.fireTotal, g.bg.fireTotal, [1 + FIRE_BANDS, 1, 1])(pass)
    })
    enc.copyTextureToTexture({ texture: g.scal[1] }, { texture: g.scal[0] }, [nx, ny, nz])
    timed('light', one(p.light, g.bg.light, [Math.ceil(g.lightDims[0] / 4), Math.ceil(g.lightDims[1] / 4), Math.ceil(g.lightDims[2] / 4)]))
    const k = labels.length
    labels.push('render')
    const t = this.ensureTargets()
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: t.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 },
    })
    rp.setPipeline(p.render)
    rp.setBindGroup(0, g.bg.render)
    rp.draw(3)
    rp.end()
    labels.push('accumulate')
    const ap = enc.beginRenderPass({
      colorAttachments: [
        { view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        { view: t.hist[1 - this.histIndex].createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
      timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k + 2, endOfPassWriteIndex: 2 * k + 3 },
    })
    ap.setPipeline(p.resolve)
    ap.setBindGroup(0, t.groups[this.histIndex])
    ap.draw(3)
    ap.end()
    this.histIndex = 1 - this.histIndex
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
    // A still meters at once (no easing) and accumulates as a paused frame does.
    await this.readFire()
    this.meterEV = this.targetEV()
    const paused = this.paused
    this.paused = true
    for (let i = 0; i < TAA_SETTLE; i++) this.render()
    this.paused = paused
    const blob = await new Promise<Blob | null>(resolve => this.canvas.toBlob(resolve, type, quality))
    this.applySize()
    return blob
  }

  dispose() {
    this.disposed = true
    if (this.raf) cancelAnimationFrame(this.raf)
    this.raf = 0
    try {
      this.ctx.unconfigure()
    } catch {}
    this.device.destroy()
    this.ownCanvas?.remove()
  }
}
