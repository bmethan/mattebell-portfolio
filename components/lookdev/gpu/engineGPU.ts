/// <reference types="@webgpu/types" />
// The lab's renderer on WebGPU: the same path tracer as the WebGL engine (engine.ts), as compute shaders.
//
// Why: WebGL on Windows runs through ANGLE on Direct3D 11, which compiles a program that writes several render
// targets (as the tracer does: one image per light, and the denoiser's two) a second time at its first draw, on
// the GPU process's main thread, freezing the page for 6 to 25 seconds per shader variant (see FIRST_DRAW_NOTICE_MS).
// WebGPU compiles pipelines in the background (createComputePipelineAsync) and the compute tracer writes its
// running means straight into storage buffers, in place.
//
// The scene (camera, lights, materials, variants, mixer) comes from scene.ts, shared with the WebGL engine; frame
// pacing follows the same plan (slices of rows sized to the frame's GPU budget from measured GPU time, a proxy-
// resolution preview while the scene changes).
import {
  isDisplayOnly, variantOf, frameOf, projectOf, labelPointsOf, cameraOf, lightRigOf, mixWeightsOf,
  bouncesOf, goalOf, thinGlassOf, envTurnOf, sceneMaterialsOf, VARIANT_LABEL, VARIANT_GLASS, PASS_ID, VIEW_ID,
  TARGET_SPP, DENOISE_LEVELS, DENOISE_FULL_SPP, MAX_SCATTER, FURNACE_MAX_SCATTER, MAX_NODE_VISITS,
  MAX_IN_FLIGHT, MAX_SAMPLES_PER_PASS, SLICE_ROWS, SCROLL_QUIET_MS, SETTLE_MS, PREVIEW_BUDGET_MS,
  PREVIEW_MAX_DIV, MESH_GUESS_MS, PREVIEW_HOLD_SPP, SOBOL_UNIFORM, E_SIZE, E_LAYERS, LUT_SIZE, LUT_URL, INDIRECT_CLAMP,
  CYC_Z, CYC_R, ENV_BACKDROP_BLUR, BALL_X, RIG_AMBIENT,
  type LabState, type LabStatus, type VariantKey, type Vec3,
} from '../scene'
import { COLOR_CHECKER, STAGES } from '../materials'
import { MODELS, MODEL_BALLS, loadModel, type Model } from '../models'
import { loadEnvironment, whiteEnvironment, type Env, type EnvData } from '../environments'
import { traceWGSL } from './wgslTrace'
import { E_TABLE_WGSL, MIP_WGSL, DISPLAY_WGSL, DISPLAY, DENOISE_PREP_WGSL, ATROUS_WGSL, DENOISE, DENOISE_STRIDE } from './wgslAux'
import { SCENE, PASS, PASS_STRIDE, matWords, MAX_MATS, MAT_VEC4S } from './sceneLayout'

const MAX_DISPATCHES = 64 // pass-uniform slots per frame (a frame issues a slice or two)
const TIMED_PER_FRAME = 16 // timestamp pairs per frame (a frame can issue several slices)
const MAX_SLICES_PER_FRAME = 16
const TIMING_SLOTS = 4 // frames whose timestamps can be in flight at once
// Slices are at least this many rows tall. Thinner ones bound nothing (a single row through a uranium glass model
// can take 80 ms) and keep too little of the GPU busy: on the car, 512 one-row slices rendered 3 samples in 15 s
// where 64 rendered 15, with no worse frames. Several slices go into one frame when they fit (see traceSlice).
const MIN_SLICE_ROWS = 8
// The GPU time a frame may spend tracing (the WebGL engine's FRAME_BUDGET_MS is 10). Measured on an RDNA 2 iGPU:
// about 45% more samples than 10 ms, frames as smooth while dragging and scrolling; 'health' shrinks it when frames
// run long.
const GPU_FRAME_BUDGET_MS = 14
// Above this cost per full-frame sample (tonic water, uranium glass, skin), the coarse preview gives way to the
// full-resolution image (denoised) after its first sample rather than PREVIEW_HOLD_SPP: those take seconds each.
const HEAVY_SAMPLE_MS = 200

// Variants compiled ahead once the plain tracer is ready and the lab is in view, one at a time, in the order
// visitors tend to need them. On WebGPU a pipeline compiles in the background and needs nothing more at its first
// use, so a first pick of a model or a glass hero usually finds it ready.
const PREWARM: VariantKey[] = ['base+glass', 'mesh', 'mesh+sss', 'mesh+thin', 'base+sss', 'base+fluor', 'mesh+glass', 'base+fluor+glass', 'mesh+fluor', 'mesh+fluor+glass']

interface EnvGPU {
  map: GPUTexture
  alias: GPUBuffer
  w: number
  h: number
}
interface MeshGPU {
  bvh: GPUBuffer
  pos: GPUBuffer
  nrm: GPUBuffer
  size: [number, number, number]
}
interface Target {
  w: number // allocated size (a preview uses part of it)
  h: number
  accum: GPUBuffer // key, fill, rim, environment planes
  aux: GPUBuffer // albedo, normal planes
}
interface TimedDispatch {
  n: number
  frac: number
  cost: string
  preview: boolean
  pass: number
  slices: number
  slice: number
}

// float32 to IEEE half, round to nearest (for rgba16float uploads).
const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)
function toHalf(v: number) {
  f32[0] = v
  const x = u32[0]
  const sign = (x >>> 16) & 0x8000
  const e = ((x >>> 23) & 0xff) - 127 + 15
  let m = x & 0x7fffff
  if (e >= 31) return sign | 0x7c00 | ((x & 0x7fffffff) > 0x7f800000 ? 0x200 : 0)
  if (e <= 0) {
    if (e < -10) return sign
    m = (m | 0x800000) >> (1 - e)
    return sign | ((m + 0x1000) >> 13)
  }
  const h = sign | (e << 10) | (m >> 13)
  return h + ((m >> 12) & 1) // round half up on the dropped bit
}
function halves(src: Float32Array) {
  const out = new Uint16Array(src.length)
  for (let i = 0; i < src.length; i++) out[i] = toHalf(src[i])
  return out
}

export class LookdevEngineGPU {
  static detect(): boolean {
    return typeof navigator !== 'undefined' && !!navigator.gpu
  }

  // Asks for an adapter and device before touching the canvas (a canvas takes only one kind of context, and the
  // WebGL engine is the fallback). Resolves null where WebGPU is missing or refuses.
  static async create(canvas: HTMLCanvasElement, onStatus: (s: LabStatus) => void, initial: LabState, onLost?: () => void) {
    if (!LookdevEngineGPU.detect()) return null
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      if (!adapter) return null
      const features: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : []
      const lim = adapter.limits
      const device = await adapter.requestDevice({
        requiredFeatures: features,
        requiredLimits: {
          maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 512 * 1024 * 1024),
          maxBufferSize: Math.min(lim.maxBufferSize, 512 * 1024 * 1024),
          maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 8),
        },
      })
      const ctx = canvas.getContext('webgpu')
      if (!ctx) {
        device.destroy()
        return null
      }
      return new LookdevEngineGPU(canvas, ctx, device, onStatus, initial, onLost)
    } catch (err) {
      console.warn('WebGPU unavailable; using WebGL', err)
      return null
    }
  }

  readonly kind = 'webgpu'
  readonly canDenoise = true
  readonly whenReady: Promise<boolean>
  maxPixels = 1_200_000
  state: LabState
  spp = 0

  private format: GPUTextureFormat
  private sceneBuf: GPUBuffer
  private passBuf: GPUBuffer
  private passSlot = 0
  private sceneWriter = SCENE.writer()
  private traceLayout: GPUBindGroupLayout
  private tracePipeLayout: GPUPipelineLayout
  private variants = new Map<VariantKey, { pipeline: GPUComputePipeline | null; ready: boolean; failed: boolean }>()
  private displayPipeline: GPURenderPipeline | null = null
  private displayBuf: GPUBuffer
  private prepPipeline: GPUComputePipeline | null = null
  private atrousPipeline: GPUComputePipeline | null = null
  private denoiseBuf: GPUBuffer
  private eTable: GPUTexture
  private eTableT: GPUTexture
  private eSampler: GPUSampler
  private envSampler: GPUSampler
  private lut: GPUTexture
  private lutSampler: GPUSampler
  private acesReady = false
  private envBlank: GPUTexture
  private envBlankAlias: GPUBuffer
  private dummy: GPUBuffer
  private envs = new Map<Env, EnvGPU>()
  private envLoading = new Set<Env>()
  private envLoadFailed = new Set<Env>()
  private envWhite: EnvGPU | null = null
  private mipPipeline: GPUComputePipeline | null = null
  private meshes = new Map<Model, MeshGPU>()
  private meshLoading = new Set<Model>()
  private meshLoadFailed = new Set<Model>()
  private full: Target | null = null
  private preview: Target | null = null
  private dnA: GPUBuffer | null = null
  private dnB: GPUBuffer | null = null
  private dnGuide: GPUBuffer | null = null
  private bindCache = new Map<string, GPUBindGroup>()
  private timing: { querySet: GPUQuerySet; resolve: GPUBuffer; read: GPUBuffer; busy: boolean }[] = []
  private live = false
  private disposed = false
  private width = 0
  private height = 0
  private target = TARGET_SPP
  private passN = 0
  private passSlices = 1
  private passSlice = 0
  private passOverran = false
  private passAdaptive = false
  private cleanPasses = 0
  private msPerSpp = new Map<string, number>()
  private previewMsPerSpp = new Map<string, number>()
  private slicesGuess = 4
  private passId = 0
  private passTiming = new Map<number, { ns: number; frac: number; peakMs: number }>()
  private meshSlices = new Map<string, number>()
  // Each row's GPU time per sample (from the timed slices through it), so that a frame issues as many slices as fit
  // its budget. The slice count only bounds the costliest slice; on a model most rows cost far less (open
  // background), and one slice per frame left the GPU mostly idle (the car: 512 frames per sample).
  private rowCost = new Float32Array(0)
  private rowCostKey = ''
  private maxSlices = 64 // set from the height (MIN_SLICE_ROWS)
  private inFlight = 0
  private budgetCap = GPU_FRAME_BUDGET_MS
  private maxInFlight = MAX_IN_FLIGHT
  private previewK = 0
  private previewDirty = false
  private gen = 0
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
  private bounces = 8
  private previewBounces = 3
  private sceneDirty = true

  private constructor(
    private canvas: HTMLCanvasElement,
    private ctx: GPUCanvasContext,
    private device: GPUDevice,
    private onStatus: (s: LabStatus) => void,
    initial: LabState,
    private onLost?: () => void,
  ) {
    this.state = { ...initial }
    const d = device
    this.format = navigator.gpu.getPreferredCanvasFormat()
    ctx.configure({ device: d, format: this.format, alphaMode: 'opaque' })
    d.lost.then(info => {
      if (this.disposed) return
      console.warn('WebGPU device lost', info.message)
      this.stop()
      this.onLost?.()
    })

    const U = GPUBufferUsage
    this.sceneBuf = d.createBuffer({ size: SCENE.byteSize, usage: U.UNIFORM | U.COPY_DST })
    this.passBuf = d.createBuffer({ size: PASS_STRIDE * MAX_DISPATCHES, usage: U.UNIFORM | U.COPY_DST })
    this.displayBuf = d.createBuffer({ size: DISPLAY.byteSize, usage: U.UNIFORM | U.COPY_DST })
    this.denoiseBuf = d.createBuffer({ size: DENOISE_STRIDE * (DENOISE_LEVELS + 1), usage: U.UNIFORM | U.COPY_DST })
    this.dummy = d.createBuffer({ size: 16, usage: U.STORAGE })
    this.envBlankAlias = d.createBuffer({ size: 16, usage: U.STORAGE })
    this.sceneWriter.setWords('sobol', SOBOL_UNIFORM)

    const tex3 = () =>
      d.createTexture({ size: [E_SIZE, E_SIZE, E_LAYERS], dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING })
    this.eTable = tex3()
    this.eTableT = tex3()
    const clampLinear: GPUSamplerDescriptor = { magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' }
    this.eSampler = d.createSampler(clampLinear)
    this.lutSampler = d.createSampler(clampLinear)
    this.envSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'clamp-to-edge' })
    this.envBlank = d.createTexture({ size: [1, 1], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING })
    this.lut = d.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING })

    const C = GPUShaderStage.COMPUTE
    this.traceLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 2, visibility: C, buffer: { type: 'storage' } },
        { binding: 3, visibility: C, buffer: { type: 'storage' } },
        { binding: 4, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 5, visibility: C, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 6, visibility: C, sampler: { type: 'filtering' } },
        { binding: 7, visibility: C, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 8, visibility: C, sampler: { type: 'filtering' } },
        { binding: 9, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 10, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 11, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 12, visibility: C, buffer: { type: 'read-only-storage' } },
      ],
    })
    this.tracePipeLayout = d.createPipelineLayout({ bindGroupLayouts: [this.traceLayout] })

    if (d.features.has('timestamp-query')) {
      for (let i = 0; i < TIMING_SLOTS; i++) {
        this.timing.push({
          querySet: d.createQuerySet({ type: 'timestamp', count: 2 * TIMED_PER_FRAME }),
          resolve: d.createBuffer({ size: 16 * TIMED_PER_FRAME, usage: U.QUERY_RESOLVE | U.COPY_SRC }),
          read: d.createBuffer({ size: 16 * TIMED_PER_FRAME, usage: U.MAP_READ | U.COPY_DST }),
          busy: false,
        })
      }
    }

    this.loadAcesLut()
    window.addEventListener('scroll', this.handleScroll, { passive: true })
    window.addEventListener('wheel', this.handleScroll, { passive: true })
    this.whenReady = this.warmUp()
  }

  private handleScroll = () => {
    this.lastScroll = performance.now()
  }

  // ------------------------------------------------------------------------------------------------------------
  // Start-up: the plain tracer, display, denoiser and albedo-table pipelines compile in the background; then the
  // tables are integrated once on the GPU.
  // ------------------------------------------------------------------------------------------------------------
  private async warmUp() {
    const d = this.device
    const shader = (code: string) => d.createShaderModule({ code })
    const tracer = this.ensureVariant('base')
    const compute = (code: string) => d.createComputePipelineAsync({ layout: 'auto', compute: { module: shader(code), entryPoint: 'main' } })
    const displayModule = shader(DISPLAY_WGSL)
    const [table, mip, display, prep, atrous] = await Promise.all([
      compute(E_TABLE_WGSL),
      compute(MIP_WGSL),
      d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: displayModule, entryPoint: 'vs' },
        fragment: { module: displayModule, entryPoint: 'fs', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      }),
      compute(DENOISE_PREP_WGSL),
      compute(ATROUS_WGSL),
    ])
    await tracer
    if (this.disposed) return false
    if (this.variants.get('base')?.failed) throw new Error('The WebGPU tracer failed to compile')
    this.mipPipeline = mip
    this.displayPipeline = display
    this.prepPipeline = prep
    this.atrousPipeline = atrous
    // Albedo tables: once, at start-up.
    const enc = d.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(table)
    pass.setBindGroup(0, d.createBindGroup({ layout: table.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.eTable.createView() }, { binding: 1, resource: this.eTableT.createView() }] }))
    pass.dispatchWorkgroups(E_SIZE / 8, E_SIZE / 8, E_LAYERS)
    pass.end()
    d.queue.submit([enc.finish()])
    await d.queue.onSubmittedWorkDone()
    if (this.disposed) return false
    this.live = true
    this.sceneDirty = true
    this.ensureModel()
    this.reportModel()
    this.kick()
    this.prewarmNext()
    return true
  }

  private async loadAcesLut() {
    try {
      const blob = await (await fetch(LUT_URL)).blob()
      const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
      if (this.disposed) return
      if (bitmap.width !== LUT_SIZE || bitmap.height !== LUT_SIZE * LUT_SIZE) throw new Error('Unexpected LUT size')
      const lut = this.device.createTexture({
        size: [bitmap.width, bitmap.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: lut, premultipliedAlpha: false }, [bitmap.width, bitmap.height])
      bitmap.close()
      this.lut.destroy()
      this.lut = lut
      this.acesReady = true
      this.bindCache.clear()
      if (this.live && this.width && this.spp > 0) this.display()
    } catch (err) {
      console.warn('ACES 2.0 LUT unavailable; using AgX', err)
    }
  }

  // ------------------------------------------------------------------------------------------------------------
  // Variants, models and environments.
  // ------------------------------------------------------------------------------------------------------------
  private ensureVariant(key: VariantKey): Promise<void> | undefined {
    if (this.variants.has(key) || this.disposed) return
    const code = traceWGSL(key.startsWith('mesh'), VARIANT_GLASS[key])
    const v = { pipeline: null as GPUComputePipeline | null, ready: false, failed: false }
    this.variants.set(key, v)
    const shaderModule = this.device.createShaderModule({ code })
    return this.device
      .createComputePipelineAsync({ layout: this.tracePipeLayout, compute: { module: shaderModule, entryPoint: 'main' } })
      .then(
        p => {
          v.pipeline = p
          v.ready = true
        },
        err => {
          console.error(`WebGPU tracer ${key} failed`, err)
          v.failed = true
        },
      )
      .finally(() => {
        this.kick()
        this.reportModel()
        this.prewarmNext()
      })
  }

  private prewarmNext() {
    if (!this.live || !this.visible || this.disposed) return
    if ([...this.variants.values()].some(v => !v.ready && !v.failed)) return
    const key = PREWARM.find(k => !this.variants.has(k))
    if (key) this.ensureVariant(key)
  }

  private ensureModel() {
    if (this.disposed || !this.live) return
    this.ensureVariant(variantOf(this.state))
    const env = this.state.env
    if (env !== 'none' && !this.envs.has(env) && !this.envLoading.has(env) && !this.envLoadFailed.has(env)) this.loadEnv(env)
    const model = this.state.model
    if (model === 'spheres') return
    if (!this.meshes.has(model) && !this.meshLoading.has(model) && !this.meshLoadFailed.has(model)) this.loadMesh(model)
  }

  private storage(data: ArrayBufferView) {
    const buf = this.device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    this.device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength)
    return buf
  }

  private async loadEnv(env: Exclude<Env, 'none'>) {
    this.envLoading.add(env)
    try {
      const d = await loadEnvironment(env)
      if (this.disposed) return
      this.envs.set(env, this.uploadEnv(d))
      this.sceneDirty = true
      this.bindCache.clear()
    } catch (err) {
      console.error(err)
      this.envLoadFailed.add(env)
    } finally {
      this.envLoading.delete(env)
    }
    this.kick()
    this.reportModel()
  }

  // The environment's radiance as half floats with its mipmaps (built on the GPU), and its alias table.
  private uploadEnv(e: EnvData): EnvGPU {
    const d = this.device
    const levels = Math.floor(Math.log2(Math.max(e.w, e.h))) + 1
    const map = d.createTexture({
      size: [e.w, e.h],
      format: 'rgba16float',
      mipLevelCount: levels,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    })
    d.queue.writeTexture({ texture: map }, halves(e.rgba), { bytesPerRow: e.w * 8 }, [e.w, e.h])
    if (this.mipPipeline) {
      const enc = d.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(this.mipPipeline)
      for (let l = 1; l < levels; l++) {
        const w = Math.max(1, e.w >> l), h = Math.max(1, e.h >> l)
        pass.setBindGroup(0, d.createBindGroup({
          layout: this.mipPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: map.createView({ baseMipLevel: l - 1, mipLevelCount: 1 }) },
            { binding: 1, resource: map.createView({ baseMipLevel: l, mipLevelCount: 1 }) },
          ],
        }))
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8))
      }
      pass.end()
      d.queue.submit([enc.finish()])
    }
    return { map, alias: this.storage(e.alias), w: e.w, h: e.h }
  }

  private activeEnv(): EnvGPU | undefined {
    const s = this.state
    if (s.furnace) return (this.envWhite ??= this.uploadEnv(whiteEnvironment()))
    return s.env === 'none' ? undefined : this.envs.get(s.env)
  }

  private async loadMesh(model: Exclude<Model, 'spheres'>) {
    this.meshLoading.add(model)
    try {
      const m = await loadModel(model)
      if (this.disposed) return
      // The model's textures are row-major 2048 texels wide, so texel i of the GLSL tracer's texAt(i) is element i.
      this.meshes.set(model, { bvh: this.storage(m.bvh.data), pos: this.storage(m.pos.data), nrm: this.storage(m.nrm.data), size: m.size })
      this.sceneDirty = true
      this.bindCache.clear()
    } catch (err) {
      console.error(err)
      this.meshLoadFailed.add(model)
    } finally {
      this.meshLoading.delete(model)
    }
    this.kick()
    this.reportModel()
  }

  private waitingFor(): LabStatus['waitingFor'] {
    const model = this.state.model
    if (model !== 'spheres' && !this.meshes.has(model)) return 'model'
    if (this.state.env !== 'none' && !this.envs.has(this.state.env)) return 'environment'
    return this.variants.get(variantOf(this.state))?.ready ? undefined : 'shaders'
  }

  modelStatus(): LabStatus['model'] {
    const model = this.state.model
    const v = this.variants.get(variantOf(this.state))
    if (v?.failed || (model !== 'spheres' && this.meshLoadFailed.has(model)) || this.envLoadFailed.has(this.state.env)) return 'failed'
    return this.waitingFor() ? 'loading' : 'ready'
  }

  private reportModel() {
    if (!this.active) return
    const model = this.modelStatus()
    if (model !== 'ready') {
      const waitingFor = this.waitingFor()
      const compiling = waitingFor === 'shaders' ? VARIANT_LABEL[variantOf(this.state)] : undefined
      this.onStatus({ spp: 0, target: this.goal, converged: false, preview: false, ms: 0, model, waitingFor, compiling })
      return
    }
    const converged = this.spp >= this.goal
    this.onStatus({ spp: this.spp, target: this.goal, converged, preview: this.showingPreview(), ms: this.renderMs, model })
  }

  // ------------------------------------------------------------------------------------------------------------
  // The page's interface (as the WebGL engine's).
  // ------------------------------------------------------------------------------------------------------------
  labelPoints(): Vec3[] {
    return labelPointsOf(this.state, this.state.model === 'spheres' ? undefined : this.meshes.get(this.state.model)?.size)
  }

  project(p: Vec3) {
    return projectOf(this.state, this.width, this.height, p)
  }

  private get goal() {
    return goalOf(this.state, this.target)
  }

  resize(cssW: number, cssH: number, dpr: number) {
    let scaleF = Math.min(dpr, 1.5)
    if (cssW * cssH * scaleF * scaleF > this.maxPixels) scaleF = Math.sqrt(this.maxPixels / (cssW * cssH))
    const w = Math.max(1, Math.round(cssW * scaleF))
    const h = Math.max(1, Math.round(cssH * scaleF))
    if (w === this.width && h === this.height) return
    this.width = w
    this.height = h
    this.maxSlices = Math.max(1, Math.floor(h / MIN_SLICE_ROWS))
    this.canvas.width = w
    this.canvas.height = h
    const d = this.device
    const target = (tw: number, th: number): Target => ({
      w: tw,
      h: th,
      accum: d.createBuffer({ size: tw * th * 16 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      aux: d.createBuffer({ size: tw * th * 16 * 2, usage: GPUBufferUsage.STORAGE }),
    })
    for (const t of [this.full, this.preview]) {
      t?.accum.destroy()
      t?.aux.destroy()
    }
    this.full = target(w, h)
    this.preview = target(Math.ceil(w / 2), Math.ceil(h / 2))
    for (const b of [this.dnA, this.dnB, this.dnGuide]) b?.destroy()
    const dn = () => d.createBuffer({ size: w * h * 16, usage: GPUBufferUsage.STORAGE })
    this.dnA = dn()
    this.dnB = dn()
    this.dnGuide = dn()
    this.bindCache.clear()
    this.previewK = 0
    this.sceneDirty = true
    this.previewDirty = true
    this.gen++
    this.lastChange = performance.now()
    this.reset()
  }

  setState(patch: Partial<LabState>) {
    const displayOnly = isDisplayOnly(patch, this.state)
    this.state = { ...this.state, ...patch }
    if (displayOnly && (this.spp > 0 || (this.previewK > 0 && this.previewGen === this.gen))) {
      this.display()
      return
    }
    this.sceneDirty = true
    this.previewDirty = true
    this.gen++
    this.lastChange = performance.now()
    this.ensureModel()
    this.reset()
    if (this.modelStatus() !== 'ready' || patch.model || patch.env) this.reportModel()
  }

  setInteracting(on: boolean) {
    this.interacting = on
  }

  setVisible(v: boolean) {
    this.visible = v
    if (v) this.prewarmNext()
    if (v) this.kick()
    else this.stop()
  }

  setActive(on: boolean) {
    this.active = on
    if (!on) {
      this.stop()
      return
    }
    if (this.live && (this.spp > 0 || (this.previewK > 0 && this.previewGen === this.gen))) {
      const converged = this.spp >= this.goal
      this.onStatus({ spp: this.spp, target: this.goal, converged, preview: this.showingPreview(), ms: this.renderMs, model: 'ready' })
    }
    this.reportModel()
    this.kick()
  }

  setTarget(n: number) {
    this.target = Math.max(1, Math.floor(n))
    this.kick()
  }

  reset() {
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

  // ------------------------------------------------------------------------------------------------------------
  // The scene uniform.
  // ------------------------------------------------------------------------------------------------------------
  private uploadScene() {
    const s = this.state
    const W = this.sceneWriter
    const cam = cameraOf(s, this.width, this.height)
    W.set('camPos', cam.pos).set('camFwd', cam.fwd).set('camRight', cam.right).set('camUp', cam.up).set('tanHalf', [cam.tanH, cam.tanV])
    const rig = lightRigOf(s)
    rig.forEach((r, i) => {
      W.set('lightCorner', r.corner, i).set('lightU', r.U, i).set('lightV', r.V, i).set('lightRad', [...r.radiance, r.uv], i)
    })
    const b = bouncesOf(s)
    this.bounces = b.bounces
    this.previewBounces = b.preview
    const env = this.activeEnv()
    W.set('ambient', [...(env ? [0, 0, 0] : RIG_AMBIENT), ENV_BACKDROP_BLUR])
    const chart = frameOf(s).chart
    if (chart) W.set('chartO', chart.O).set('chartU', chart.U).set('chartV', chart.V)
    W.set('fparams', [CYC_Z, CYC_R, s.furnace ? 0 : INDIRECT_CLAMP, envTurnOf(s)])
    W.set('fparams2', [s.split])
    W.set('iparams0', [s.furnace ? 1 : 0, PASS_ID[s.pass], s.multiscatter ? 1 : 0, s.compare === 'multiscatter' && !s.furnace ? 1 : 0])
    const { thinGlass, thinSlot } = thinGlassOf(s)
    W.set('iparams1', [STAGES[s.stage].cyc ? 1 : 0, chart ? 1 : 0, thinGlass ? 1 : 0, rig.length])
    W.set('iparams2', [env ? 1 : 0, s.furnace ? FURNACE_MAX_SCATTER : MAX_SCATTER, thinSlot, MAX_NODE_VISITS])
    W.set('iparams3', [env?.w ?? 1, env?.h ?? 1])
    COLOR_CHECKER.forEach((c, i) => W.set('chartColor', c, i))
    if (s.model === 'spheres') {
      W.set('ballX', BALL_X)
    } else {
      MODEL_BALLS.forEach((ball, i) => W.set('balls', ball, i))
      const c = Math.cos(s.modelYaw), sn = Math.sin(s.modelYaw)
      W.set('modelRot', [c, 0, -sn], 0).set('modelRot', [0, 1, 0], 1).set('modelRot', [sn, 0, c], 2)
      W.set('modelPos', [0, 0, 0, MODELS[s.model].scale])
    }
    const mats = sceneMaterialsOf(s)
    const words = new Float32Array(MAX_MATS * MAT_VEC4S * 4)
    mats.forEach((m, i) => words.set(matWords(m), i * MAT_VEC4S * 4))
    W.setWords('mats', words)
    this.device.queue.writeBuffer(this.sceneBuf, 0, W.data)
    this.sceneDirty = false
  }

  private writePass(dims: [number, number, number, number], slice: [number, number, number], bounces: number, res: [number, number]) {
    const slot = this.passSlot++ % MAX_DISPATCHES
    const w = PASS.writer()
    w.set('dims', dims).set('slice', slice).set('iparams', [bounces]).set('res', res)
    this.device.queue.writeBuffer(this.passBuf, slot * PASS_STRIDE, w.data)
    return slot * PASS_STRIDE
  }

  // The tracer's bind group for a target: the scene, the target's images, the tables, the environment, the model.
  private traceBindGroup(t: Target) {
    const env = this.activeEnv()
    const mesh = this.state.model === 'spheres' ? undefined : this.meshes.get(this.state.model)
    const key = `trace:${t === this.full ? 'full' : 'preview'}:${this.state.furnace ? 'white' : this.state.env}:${env ? 1 : 0}:${this.state.model}:${mesh ? 1 : 0}`
    let bg = this.bindCache.get(key)
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.traceLayout,
        entries: [
          { binding: 0, resource: { buffer: this.sceneBuf } },
          { binding: 1, resource: { buffer: this.passBuf, size: PASS.byteSize } },
          { binding: 2, resource: { buffer: t.accum } },
          { binding: 3, resource: { buffer: t.aux } },
          { binding: 4, resource: this.eTable.createView() },
          { binding: 5, resource: this.eTableT.createView() },
          { binding: 6, resource: this.eSampler },
          { binding: 7, resource: (env?.map ?? this.envBlank).createView() },
          { binding: 8, resource: this.envSampler },
          { binding: 9, resource: { buffer: env?.alias ?? this.envBlankAlias } },
          { binding: 10, resource: { buffer: mesh?.bvh ?? this.dummy } },
          { binding: 11, resource: { buffer: mesh?.pos ?? this.dummy } },
          { binding: 12, resource: { buffer: mesh?.nrm ?? this.dummy } },
        ],
      })
      this.bindCache.set(key, bg)
    }
    return bg
  }

  private tracePipeline() {
    return this.variants.get(variantOf(this.state))?.pipeline ?? null
  }

  // A slice's rows: every slice-count-th block of rows from block 'slice' (as the kernel lays them out).
  private sliceRows(h: number, sliceCount: number, slice: number, fn: (y: number) => void) {
    const blockRows = sliceCount === 1 ? h : Math.max(1, Math.min(SLICE_ROWS, Math.floor(h / sliceCount)))
    for (let b = slice * blockRows; b < h; b += sliceCount * blockRows) {
      for (let y = b; y < Math.min(h, b + blockRows); y++) fn(y)
    }
  }

  // The predicted GPU time of a slice of the current pass, or 0 while any of its rows is unmeasured.
  private predictSlice(slice: number) {
    if (this.rowCostKey !== this.rowKey()) return 0
    let ms = 0
    let unknown = false
    this.sliceRows(this.height, this.passSlices, slice, y => {
      const c = this.rowCost[y]
      if (c > 0) ms += c
      else unknown = true
    })
    return unknown ? 0 : ms * this.passN
  }

  private rowKey() {
    return `${this.costKey()}:${this.width}x${this.height}`
  }

  // One dispatch of the tracer into target t: new samples per pixel, over one slice of its rows.
  private encodeTrace(
    enc: GPUCommandEncoder,
    t: Target,
    w: number,
    h: number,
    res: [number, number],
    sppDone: number,
    n: number,
    sliceCount: number,
    slice: number,
    bounces: number,
    timed: TimedDispatch | null,
    times: { slot: number; list: TimedDispatch[] } | null,
  ) {
    const pipeline = this.tracePipeline()
    if (!pipeline) return 0
    if (this.sceneDirty) this.uploadScene()
    // Blocks of up to SLICE_ROWS rows; thinner when there are more slices than SLICE_ROWS blocks fill.
    const blockRows = sliceCount === 1 ? h : Math.max(1, Math.min(SLICE_ROWS, Math.floor(h / sliceCount)))
    const blocks = Math.ceil(h / blockRows)
    const sliceBlocks = Math.max(0, Math.ceil((blocks - slice) / sliceCount))
    let rows = 0
    for (let b = 0; b < sliceBlocks; b++) rows += Math.min(blockRows, h - (slice + b * sliceCount) * blockRows)
    if (sliceBlocks === 0) return 0
    const offset = this.writePass([w, h, sppDone, n], [slice, sliceCount, blockRows], bounces, res)
    let desc: GPUComputePassDescriptor = {}
    if (timed && times && times.list.length < TIMED_PER_FRAME) {
      const i = times.list.length
      desc = { timestampWrites: { querySet: this.timing[times.slot].querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } }
      times.list.push({ ...timed, frac: timed.frac * rows / h }) // a preview counts as 1/k^2 of a frame
    }
    const pass = enc.beginComputePass(desc)
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, this.traceBindGroup(t), [offset])
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil((sliceBlocks * blockRows) / 8)) // the kernel's 8 x 8
    pass.end()
    return rows
  }

  // ------------------------------------------------------------------------------------------------------------
  // The frame loop (as the WebGL engine's step).
  // ------------------------------------------------------------------------------------------------------------
  private step = (t: number) => {
    this.rafId = 0
    if (this.disposed || !this.active || !this.visible || this.spp >= this.goal) return
    if (this.modelStatus() !== 'ready') {
      this.ensureModel()
      return
    }
    this.rafId = requestAnimationFrame(this.step)
    const dt = this.lastFrameT ? t - this.lastFrameT : 0
    this.trackFrame(dt)
    if (dt > 0 && dt < 250) this.renderMs += dt
    this.lastFrameT = t
    if (this.inFlight >= this.maxInFlight) {
      this.passOverran = true
      return
    }
    const now = performance.now()
    if (!this.interacting && now - this.lastScroll < SCROLL_QUIET_MS) return

    const k = this.previewDiv()
    if (k > 1 && (this.interacting || now - this.lastChange < SETTLE_MS)) {
      if (!this.previewDirty) return
      this.tracePreview(k)
      this.onStatus({ spp: 0, target: this.goal, converged: false, preview: true, ms: this.renderMs, model: 'ready' })
      return
    }

    if (this.passN === 0) this.planPass()
    const passDone = this.traceSlice()
    if (passDone) {
      const converged = this.spp >= this.goal
      this.onStatus({ spp: this.spp, target: this.goal, converged, preview: this.showingPreview(), ms: this.renderMs, model: 'ready' })
      if (converged) this.stop()
    }
  }

  // A cost class for the pacing's measurements. Beyond the WebGL engine's (model, pass, furnace), the material
  // and lighting: uranium glass under the black light costs some 20 times plastic on the same model, and planning
  // its first passes from plastic's timings packed whole frames with slices of it.
  private costKey() {
    const s = this.state
    return `${s.model}:${s.pass}:${s.furnace ? 1 : 0}:${variantOf(s)}:${s.hero}:${s.keyType}:${s.env}`
  }

  private showingPreview() {
    const hold = (this.fullSampleMs() ?? 0) > HEAVY_SAMPLE_MS ? 1 : PREVIEW_HOLD_SPP
    return this.previewK > 0 && this.previewGen === this.gen && this.spp < Math.min(hold, this.goal)
  }

  // A full-frame sample's GPU time: measured, or before that the preview's (shorter paths, so scaled up).
  private fullSampleMs() {
    const previewMs = this.previewMsPerSpp.get(this.costKey())
    return this.msPerSpp.get(this.costKey()) ?? (previewMs !== undefined ? 1.5 * previewMs : undefined)
  }

  private trackFrame(dt: number) {
    if (dt <= 0 || dt > 250) return
    this.frameTimes.push(dt)
    if (this.frameTimes.length > 32) this.frameTimes.shift()
    const sorted = [...this.frameTimes].sort((a, b) => a - b)
    this.frameMs = Math.min(34, Math.max(6.9, sorted[Math.floor(sorted.length / 4)]))
    this.health = dt > 1.5 * this.frameMs ? Math.max(0.4, this.health * 0.8) : Math.min(1, this.health + 0.01)
  }

  private budgetMs() {
    return Math.min(this.budgetCap, 0.85 * this.frameMs) * this.health
  }

  private previewDiv() {
    const key = this.costKey()
    const ms = this.previewMsPerSpp.get(key) ?? this.msPerSpp.get(key) ?? (this.state.model === 'spheres' ? 40 : MESH_GUESS_MS)
    const budget = Math.min(PREVIEW_BUDGET_MS, 0.5 * this.frameMs)
    return Math.max(1, Math.min(PREVIEW_MAX_DIV, Math.ceil(Math.sqrt(ms / budget))))
  }

  // Submits a frame's commands, with its timestamps when there are any, and counts it in flight until done.
  private submit(enc: GPUCommandEncoder, times: { slot: number; list: TimedDispatch[] } | null) {
    const tm = times && times.list.length ? this.timing[times.slot] : null
    if (tm && times) {
      enc.resolveQuerySet(tm.querySet, 0, 2 * times.list.length, tm.resolve, 0)
      enc.copyBufferToBuffer(tm.resolve, 0, tm.read, 0, 16 * times.list.length)
    }
    this.device.queue.submit([enc.finish()])
    this.inFlight++
    this.device.queue.onSubmittedWorkDone().then(() => {
      this.inFlight--
    })
    if (tm && times) {
      const list = times.list
      tm.read.mapAsync(GPUMapMode.READ).then(
        () => {
          const ts = new BigUint64Array(tm.read.getMappedRange().slice(0))
          tm.read.unmap()
          tm.busy = false
          list.forEach((d, i) => this.recordTiming(d, Number(ts[2 * i + 1] - ts[2 * i])))
        },
        () => {
          tm.busy = false
        },
      )
    } else if (times) {
      this.timing[times.slot].busy = false
    }
  }

  private timesSlot() {
    const i = this.timing.findIndex(t => !t.busy)
    if (i < 0) return null
    this.timing[i].busy = true
    return { slot: i, list: [] as TimedDispatch[] }
  }

  private tracePreview(k: number) {
    const t = this.preview!
    const w = Math.ceil(this.width / k), h = Math.ceil(this.height / k)
    const enc = this.device.createCommandEncoder()
    const times = this.timesSlot()
    this.encodeTrace(enc, t, w, h, [this.width / k, this.height / k], 0, 1, 1, 0, this.previewBounces, { n: 1, frac: 1 / (k * k), cost: this.costKey(), preview: true, pass: -1, slices: 1, slice: 0 }, times)
    this.previewK = k
    this.previewGen = this.gen
    this.previewDirty = false
    this.encodeDisplay(enc)
    this.submit(enc, times)
  }

  private planPass() {
    // Before a full pass is measured, the preview's measured cost beats a guess.
    const ms = this.fullSampleMs()
    let n = 1
    if (ms !== undefined && !this.interacting && 2 * ms * n <= this.budgetMs()) {
      n = Math.max(n, Math.min(MAX_SAMPLES_PER_PASS, Math.floor(this.budgetMs() / (2 * ms))))
    }
    n = Math.min(n, this.goal - this.spp)
    let slices = this.slicesGuess
    this.passAdaptive = false
    if (this.interacting) {
      slices = 1
    } else if (this.meshSlices.has(this.costKey())) {
      slices = this.meshSlices.get(this.costKey())! * n
    } else if (ms !== undefined) {
      slices = Math.ceil((ms * n) / this.budgetMs())
    } else if (this.state.model !== 'spheres') {
      slices = this.maxSlices // a model scene not yet measured: the thinnest slices until its first timings
    } else {
      this.passAdaptive = true
    }
    this.passN = n
    this.passSlices = Math.max(1, Math.min(this.maxSlices, this.height, slices))
    this.passSlice = 0
    this.passId++
  }

  // Encodes and submits the next slice of the current pass (and the display when that completes it).
  private traceSlice() {
    const enc = this.device.createCommandEncoder()
    const times = this.timesSlot()
    const budget = this.budgetMs()
    const predict = (i: number) => this.predictSlice(i) // 0: not yet measured
    let used = 0
    let issued = 0
    let done = false
    do {
      const p = predict(this.passSlice)
      this.encodeTrace(enc, this.full!, this.width, this.height, [this.width, this.height], this.spp, this.passN, this.passSlices, this.passSlice, this.bounces,
        { n: this.passN, frac: 1, cost: this.costKey(), preview: false, pass: this.passId, slices: this.passSlices, slice: this.passSlice }, times)
      used += p > 0 ? p : budget
      issued++
      done = ++this.passSlice >= this.passSlices
    } while (!done && issued < MAX_SLICES_PER_FRAME && predict(this.passSlice) > 0 && used + predict(this.passSlice) <= budget)
    if (done) {
      this.spp += this.passN
      this.passN = 0
      if (this.passAdaptive) {
        if (this.passOverran) {
          this.slicesGuess = Math.min(this.maxSlices, this.slicesGuess + 1)
          this.cleanPasses = 0
        } else if (++this.cleanPasses >= 4 && this.slicesGuess > 1) {
          this.slicesGuess--
          this.cleanPasses = 0
        }
      }
      this.passOverran = false
      this.encodeDisplay(enc)
    }
    this.submit(enc, times)
    return done
  }

  // GPU time of a finished dispatch, scaled to one full-frame sample (as the WebGL engine's collectTimings).
  private recordTiming(d: TimedDispatch, ns: number) {
    if (this.disposed || !(ns > 0)) return
    if (!d.preview && d.slices > 1) {
      const key = `${d.cost}:${this.width}x${this.height}`
      if (key !== this.rowCostKey) {
        this.rowCost = new Float32Array(this.height)
        this.rowCostKey = key
      }
      let rows = 0
      this.sliceRows(this.height, d.slices, d.slice, () => rows++)
      const perRow = ns / 1e6 / d.n / Math.max(rows, 1)
      this.sliceRows(this.height, d.slices, d.slice, y => { this.rowCost[y] = perRow })
    }
    let ms = ns / 1e6 / d.frac / d.n
    if (!d.preview && !d.cost.startsWith('spheres:')) {
      const acc = this.passTiming.get(d.pass) ?? { ns: 0, frac: 0, peakMs: 0 }
      acc.ns += ns
      acc.frac += d.frac
      acc.peakMs = Math.max(acc.peakMs, ns / 1e6)
      for (const k of this.passTiming.keys()) if (k < d.pass - 2) this.passTiming.delete(k)
      if (acc.frac < 0.999) {
        this.passTiming.set(d.pass, acc)
        return
      }
      this.passTiming.delete(d.pass)
      ms = acc.ns / 1e6 / acc.frac / d.n
      const ratio = acc.peakMs / this.budgetMs()
      if (ratio > 1.15 || ratio < 0.6) {
        const now = this.meshSlices.get(d.cost) ?? d.slices
        this.meshSlices.set(d.cost, Math.max(1, Math.min(this.maxSlices, Math.round(now * Math.min(3, Math.max(0.7, ratio))))))
      }
    }
    const model = d.preview ? this.previewMsPerSpp : this.msPerSpp
    const prev = model.get(d.cost)
    model.set(d.cost, prev === undefined ? ms : 0.7 * prev + 0.3 * ms)
  }

  // ------------------------------------------------------------------------------------------------------------
  // Display and denoiser.
  // ------------------------------------------------------------------------------------------------------------
  private displayCompare() {
    const s = this.state
    if (s.furnace) return 0
    if (s.compare === 'denoise') return s.pass === 'beauty' || s.pass === 'diffuse' || s.pass === 'specular' ? 1 : 0
    return s.compare === 'view' ? 2 : 0
  }

  private denoiseMix(preview: boolean) {
    const s = this.state
    if (!s.denoise || s.furnace || !(s.pass === 'beauty' || s.pass === 'diffuse' || s.pass === 'specular')) return 0
    if (preview || this.spp <= DENOISE_FULL_SPP) return 1
    return Math.max(0, 1 - Math.log2(this.spp / DENOISE_FULL_SPP) / Math.log2(Math.max(this.goal, 2 * DENOISE_FULL_SPP) / DENOISE_FULL_SPP))
  }

  private display() {
    if (!this.live || !this.width || !this.full) return
    const enc = this.device.createCommandEncoder()
    this.encodeDisplay(enc)
    this.device.queue.submit([enc.finish()])
  }

  private encodeDisplay(enc: GPUCommandEncoder) {
    if (!this.displayPipeline || !this.full || !this.preview) return
    const d = this.device
    const s = this.state
    const preview = this.showingPreview()
    const src = preview ? this.preview : this.full
    const k = preview ? this.previewK : 1
    const w = Math.ceil(this.width / k), h = Math.ceil(this.height / k)
    const mix = this.denoiseMix(preview)
    const compare = this.displayCompare()
    const mixW = mixWeightsOf(s)
    let denoised = this.dnA!
    if (mix > 0 || compare === 1) denoised = this.encodeDenoise(enc, src, w, h, preview ? 1 : this.spp, mixW)
    const u = DISPLAY.writer()
    mixW.forEach((m, i) => u.set('mixW', m, i))
    u.set('params', [s.furnace ? 0 : s.exposure, mix, s.split * this.width])
    u.set('iparams', [s.furnace ? VIEW_ID.standard : VIEW_ID[s.view], PASS_ID[s.pass], compare, this.acesReady ? 1 : 0])
    u.set('dims', [w, h, k, this.height])
    d.queue.writeBuffer(this.displayBuf, 0, u.data)
    const key = `display:${preview ? 'p' : 'f'}:${denoised === this.dnA ? 'a' : 'b'}:${this.acesReady ? 1 : 0}`
    let bg = this.bindCache.get(key)
    if (!bg) {
      bg = d.createBindGroup({
        layout: this.displayPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.displayBuf } },
          { binding: 1, resource: { buffer: src.accum } },
          { binding: 2, resource: { buffer: denoised } },
          { binding: 3, resource: this.lut.createView() },
          { binding: 4, resource: this.lutSampler },
        ],
      })
      this.bindCache.set(key, bg)
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    })
    pass.setPipeline(this.displayPipeline)
    pass.setBindGroup(0, bg)
    pass.draw(3)
    pass.end()
  }

  // Filters the mixed image of src (w x h of it); returns the buffer holding the result.
  private encodeDenoise(enc: GPUCommandEncoder, src: Target, w: number, h: number, spp: number, mixW: Vec3[]) {
    const d = this.device
    const prep = this.prepPipeline!
    const atrous = this.atrousPipeline!
    const which = src === this.full ? 'f' : 'p'
    for (let i = 0; i <= DENOISE_LEVELS; i++) {
      const u = DENOISE.writer()
      mixW.forEach((m, j) => u.set('mixW', m, j))
      u.set('params', [spp]).set('iparams', [w, h, i === 0 ? 0 : 1 << (i - 1), i === DENOISE_LEVELS ? 1 : 0])
      d.queue.writeBuffer(this.denoiseBuf, i * DENOISE_STRIDE, u.data)
    }
    const bg = (key: string, make: () => GPUBindGroup) => {
      let g = this.bindCache.get(key)
      if (!g) {
        g = make()
        this.bindCache.set(key, g)
      }
      return g
    }
    const uniform = (i: number) => ({ buffer: this.denoiseBuf, offset: i * DENOISE_STRIDE, size: DENOISE.byteSize })
    const pass = enc.beginComputePass()
    pass.setPipeline(prep)
    pass.setBindGroup(0, bg(`prep:${which}`, () => d.createBindGroup({
      layout: prep.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: uniform(0) },
        { binding: 1, resource: { buffer: src.accum } },
        { binding: 2, resource: { buffer: src.aux } },
        { binding: 3, resource: { buffer: this.dnA! } },
        { binding: 4, resource: { buffer: this.dnGuide! } },
      ],
    })))
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8))
    pass.setPipeline(atrous)
    for (let i = 0; i < DENOISE_LEVELS; i++) {
      const from = i % 2 === 0 ? this.dnA! : this.dnB!
      const to = i % 2 === 0 ? this.dnB! : this.dnA!
      pass.setBindGroup(0, bg(`atrous:${which}:${i}`, () => d.createBindGroup({
        layout: atrous.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: uniform(i + 1) },
          { binding: 1, resource: { buffer: from } },
          { binding: 2, resource: { buffer: this.dnGuide! } },
          { binding: 3, resource: { buffer: src.aux } },
          { binding: 4, resource: { buffer: to } },
        ],
      })))
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8))
    }
    pass.end()
    return DENOISE_LEVELS % 2 === 1 ? this.dnB! : this.dnA!
  }

  // ------------------------------------------------------------------------------------------------------------
  // Development aids and readbacks (asynchronous on WebGPU).
  // ------------------------------------------------------------------------------------------------------------
  // Accumulates synchronously with the GPU, a few samples per submission (each waited on, far below any GPU
  // watchdog), independent of animation frames.
  async renderNow(total: number) {
    if (!this.live || this.modelStatus() !== 'ready') return
    this.passN = 0
    while (this.spp < total) {
      const n = Math.min(4, total - this.spp)
      const enc = this.device.createCommandEncoder()
      this.encodeTrace(enc, this.full!, this.width, this.height, [this.width, this.height], this.spp, n, 1, 0, this.bounces, null, null)
      this.device.queue.submit([enc.finish()])
      await this.device.queue.onSubmittedWorkDone()
      this.spp += n
    }
    this.display()
    await this.device.queue.onSubmittedWorkDone()
    this.onStatus({ spp: this.spp, target: this.goal, converged: this.spp >= this.goal, preview: false, ms: this.renderMs, model: 'ready' })
  }

  async snapshot(type = 'image/png', quality?: number): Promise<Blob | null> {
    this.display()
    await this.device.queue.onSubmittedWorkDone()
    return new Promise(resolve => this.canvas.toBlob(resolve, type, quality))
  }

  // The image before the view transform: linear ACEScg radiance mixed as the mixer is set; alpha = coverage.
  async readAccum(): Promise<{ w: number; h: number; px: Float32Array } | null> {
    if (!this.full) return null
    const d = this.device
    const n = this.width * this.height
    const read = d.createBuffer({ size: n * 16 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = d.createCommandEncoder()
    enc.copyBufferToBuffer(this.full.accum, 0, read, 0, n * 16 * 4)
    d.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const planes = new Float32Array(read.getMappedRange().slice(0))
    read.unmap()
    read.destroy()
    const mix = mixWeightsOf(this.state)
    const px = new Float32Array(n * 4)
    for (let k = 0; k < 4; k++) {
      const base = k * n * 4
      for (let i = 0; i < n * 4; i += 4) {
        px[i] += planes[base + i] * mix[k][0]
        px[i + 1] += planes[base + i + 1] * mix[k][1]
        px[i + 2] += planes[base + i + 2] * mix[k][2]
        if (k === 0) px[i + 3] = planes[base + i + 3]
      }
    }
    return { w: this.width, h: this.height, px }
  }

  async readSphereMean(): Promise<[number, number, number] | null> {
    const buf = await this.readAccum()
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
    window.removeEventListener('scroll', this.handleScroll)
    window.removeEventListener('wheel', this.handleScroll)
    try {
      this.ctx.unconfigure()
    } catch {}
    this.device.destroy()
  }
}

