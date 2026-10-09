// The pyro lab's engine: a WebGPU gas solver (see wgsl.ts) stepped at a fixed 30 steps per simulated second,
// and a volume renderer drawn after each step (or when a shading control changes, so a paused simulation can be
// relit). WebGPU only: browsers without it get the lab's still frame.
import { ADVECT_SCAL, ADVECT_VEL, COMBINE, CURL, DIVERGENCE, FORCES, LIGHT, PROJECT, RENDER, SOR } from './wgsl'
import { DOMAIN_HEIGHT, PYRO_DEFAULTS, RESOLUTIONS, RESOLUTION_ORDER, type PyroResolution, type PyroState } from './params'

export const SIM_DT = 1 / 30
const MAX_STEPS_PER_FRAME = 2
const MAX_IN_FLIGHT = 2 // frames of GPU work queued at once: enough to keep the GPU busy, too few to stall the page
const SOR_OMEGA = 1.7
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

export interface PyroStatus {
  ready: boolean
  running: boolean // stepping (not paused, on screen)
  simTime: number // s
  stepsPerSec: number
  renderScale: number
  dims: [number, number, number]
  resolution: PyroResolution // what the engine runs (it may step down from the default on a slow GPU)
  autoStepped: boolean // it did step down
}

type Vec3 = [number, number, number]
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

interface Pipes {
  advectVel: GPUComputePipeline
  curl: GPUComputePipeline
  forces: GPUComputePipeline
  divergence: GPUComputePipeline
  sor: GPUComputePipeline
  project: GPUComputePipeline
  advFwd: GPUComputePipeline
  advBwd: GPUComputePipeline
  combine: GPUComputePipeline
  light: GPUComputePipeline
  render: GPURenderPipeline
}

interface Grids {
  n: Vec3
  textures: GPUTexture[]
  buffers: GPUBuffer[]
  vel: [GPUTexture, GPUTexture]
  scal: [GPUTexture, GPUTexture, GPUTexture, GPUTexture]
  lightDims: Vec3
  bg: {
    advectVel: GPUBindGroup
    curl: GPUBindGroup
    forces: GPUBindGroup
    divergence: GPUBindGroup
    sor: [GPUBindGroup, GPUBindGroup]
    project: GPUBindGroup
    advFwd: GPUBindGroup
    advBwd: GPUBindGroup
    combine: GPUBindGroup
    light: GPUBindGroup
    render: GPUBindGroup
  }
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
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      if (!adapter) {
        own?.remove()
        console.warn('Pyro: the browser offered no adapter')
        return null
      }
      const device = await adapter.requestDevice()
      const info = adapter.info
      const name = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ') || 'unnamed adapter'
      return new PyroEngine(own ?? canvas, own, ctx, device, onStatus, initial, onLost, name)
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
  private simData = new ArrayBuffer(176)
  private renderData = new Float32Array(36)
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
    this.whenReady = this.build()
  }

  // Compiles every pipeline in the background, then makes the grids.
  private async build() {
    const d = this.device
    const compute = (code: string, constants?: Record<string, number>) =>
      d.createComputePipelineAsync({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: 'main', constants } })
    const renderModule = d.createShaderModule({ code: RENDER })
    const [advectVel, curl, forces, divergence, sor, project, advFwd, advBwd, combine, light, render] = await Promise.all([
      compute(ADVECT_VEL),
      compute(CURL),
      compute(FORCES),
      compute(DIVERGENCE),
      compute(SOR),
      compute(PROJECT),
      compute(ADVECT_SCAL, { DIR: 1 }),
      compute(ADVECT_SCAL, { DIR: -1 }),
      compute(COMBINE),
      compute(LIGHT),
      d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: renderModule, entryPoint: 'vs' },
        fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      }),
    ])
    if (this.disposed) return false
    this.pipes = { advectVel, curl, forces, divergence, sor, project, advFwd, advBwd, combine, light, render }
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
    const vel: [GPUTexture, GPUTexture] = [tex(velSize, true), tex(velSize, true)]
    const scal = [tex(n, true), tex(n), tex(n), tex(n, true)] as [GPUTexture, GPUTexture, GPUTexture, GPUTexture]
    const curl = tex(n)
    const lightDims: Vec3 = [Math.ceil(n[0] / 2), Math.ceil(n[1] / 2), Math.ceil(n[2] / 2)]
    const light = tex(lightDims)
    const cells = n[0] * n[1] * n[2]
    const div = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE })
    const pressure = d.createBuffer({ size: cells * 4, usage: GPUBufferUsage.STORAGE })
    const v = (t: GPUTexture) => t.createView()
    const group = (pipe: GPUComputePipeline | GPURenderPipeline, resources: GPUBindingResource[]) =>
      d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({ binding, resource })) })
    const P = { buffer: this.simBuf }
    const R = { buffer: this.renderBuf }
    const s = this.sampler
    this.grids = {
      n,
      textures: [...vel, ...scal, curl, light],
      buffers: [div, pressure],
      vel,
      scal,
      lightDims,
      bg: {
        advectVel: group(p.advectVel, [P, s, v(vel[0]), v(vel[1])]),
        curl: group(p.curl, [P, v(vel[1]), v(curl)]),
        forces: group(p.forces, [P, v(vel[1]), v(scal[0]), v(curl), v(vel[0])]),
        divergence: group(p.divergence, [P, v(vel[0]), { buffer: div }]),
        sor: [0, 1].map(k => group(p.sor, [P, { buffer: this.parityBufs[k] }, { buffer: div }, { buffer: pressure }])) as [GPUBindGroup, GPUBindGroup],
        project: group(p.project, [P, v(vel[0]), { buffer: pressure }, v(vel[1])]),
        advFwd: group(p.advFwd, [P, s, v(vel[0]), v(scal[0]), v(scal[1])]),
        advBwd: group(p.advBwd, [P, s, v(vel[0]), v(scal[1]), v(scal[2])]),
        combine: group(p.combine, [P, s, v(vel[0]), v(scal[0]), v(scal[1]), v(scal[2]), v(scal[3])]),
        light: group(p.light, [P, R, s, v(scal[0]), v(light)]),
        render: group(p.render, [P, R, s, v(scal[0]), v(light)]),
      },
    }
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
    // Shading changes redraw (and relight) at once, even while paused.
    this.lightDirty = true
    this.needsRender = true
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

  // A camera a little below the plume's middle, framing the domain's height with some room, looking along -z.
  private layoutCamera() {
    if (!this.grids) return
    const [, ny, nz] = this.grids.n
    const dx = DOMAIN_HEIGHT / ny
    const h = ny * dx
    const depth = nz * dx
    const aspect = this.canvas.width / Math.max(1, this.canvas.height)
    const tanY = Math.tan(FOV_Y / 2)
    const dist = (h * 0.56) / tanY + depth / 2
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
  stirAt(x: number, y: number, dxNdc: number, dyNdc: number, dtMs: number) {
    if (!this.grids) return
    const c = this.cam
    const dir = norm([
      c.fwd[0] + c.right[0] * x * c.tanY * c.aspect + c.up[0] * y * c.tanY,
      c.fwd[1] + c.right[1] * x * c.tanY * c.aspect + c.up[1] * y * c.tanY,
      c.fwd[2] + c.right[2] * x * c.tanY * c.aspect + c.up[2] * y * c.tanY,
    ])
    // World metres per ndc unit at the domain's centre plane.
    const dist = Math.hypot(c.pos[0], c.pos[1] - DOMAIN_HEIGHT * 0.47, c.pos[2])
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
    const [nx, ny, nz] = g.n
    const cv: Vec3 = [Math.ceil((nx + 1) / 4), Math.ceil((ny + 1) / 4), Math.ceil((nz + 1) / 4)]
    const cc: Vec3 = [Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4)]
    const run = (pass: GPUComputePassEncoder, pipe: GPUComputePipeline, bg: GPUBindGroup, wg: Vec3) => {
      pass.setPipeline(pipe)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(wg[0], wg[1], wg[2])
    }
    let pass = enc.beginComputePass()
    run(pass, p.advectVel, g.bg.advectVel, cv)
    run(pass, p.curl, g.bg.curl, cc)
    run(pass, p.forces, g.bg.forces, cv)
    run(pass, p.divergence, g.bg.divergence, cc)
    for (let i = 0; i < this.state.pressureIters; i++) {
      run(pass, p.sor, g.bg.sor[0], cc)
      run(pass, p.sor, g.bg.sor[1], cc)
    }
    run(pass, p.project, g.bg.project, cv)
    pass.end()
    enc.copyTextureToTexture({ texture: g.vel[1] }, { texture: g.vel[0] }, [nx + 1, ny + 1, nz + 1])
    pass = enc.beginComputePass()
    run(pass, p.advFwd, g.bg.advFwd, cc)
    if (this.state.advection === 'mac') run(pass, p.advBwd, g.bg.advBwd, cc)
    run(pass, p.combine, g.bg.combine, cc)
    pass.end()
    enc.copyTextureToTexture({ texture: g.scal[3] }, { texture: g.scal[0] }, [nx, ny, nz])
    this.device.queue.submit([enc.finish()])
    this.simTime += SIM_DT
    this.frame++
    this.stir = null
    this.lightDirty = true
    this.stepTimes.push(performance.now())
    if (this.runClock >= AUTO_RES_FROM_S && this.runClock < AUTO_RES_TO_S) this.autoSteps++
  }

  private writeSim() {
    const s = this.state
    const g = this.grids!
    const [nx, ny, nz] = g.n
    const dx = DOMAIN_HEIGHT / ny
    const u = new Uint32Array(this.simData)
    const f = new Float32Array(this.simData)
    u.set([nx, ny, nz, this.frame], 0)
    f.set([dx, SIM_DT, this.simTime, 0], 4)
    f.set([0, s.srcRadius + dx, 0, s.srcRadius], 8)
    f.set([s.srcDensity, s.srcTemp, s.srcSpeed, s.srcNoise], 12)
    f.set([s.buoyancy, 0, s.vorticity, s.turbulence], 16)
    f.set([s.turbScale, s.wind, 0, 0.05], 20)
    f.set([s.cooling, s.dissipation, SOR_OMEGA, SPEED_LIMIT], 24)
    const st = this.stir
    f.set([...(st ? st.centre : [0, 0, 0]), STIR_RADIUS], 28)
    f.set([...(st ? st.vel : [0, 0, 0]), st ? 1 : 0], 32)
    f.set([0, 0, 0, s.advection === 'mac' ? 1 : 0], 36)
    f.set([(-nx * dx) / 2, 0, (-nz * dx) / 2, 0], 40)
    this.device.queue.writeBuffer(this.simBuf, 0, this.simData)
  }

  private writeRender() {
    const s = this.state
    const c = this.cam
    const az = (s.lightAz * Math.PI) / 180
    const el = (s.lightEl * Math.PI) / 180
    // Toward the sun; azimuth 0 puts it behind the camera (+z), positive swings it to camera right.
    const sun: Vec3 = [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)]
    const sunCol: Vec3 = [1.0, 0.93, 0.84]
    const skyCol: Vec3 = [0.42, 0.52, 0.72]
    const r = this.renderData
    r.set([...c.pos, c.tanY], 0)
    r.set([...c.fwd, c.aspect], 4)
    r.set([...c.right, this.frame % 1024], 8)
    r.set([...c.up, s.exposure], 12)
    r.set([...sun, s.anisotropy], 16)
    r.set([...sunCol.map(v => v * s.sunIntensity), s.densityScale], 20)
    r.set([...skyCol.map(v => v * s.skyIntensity), s.albedo], 24)
    r.set([0.14, 0.14, 0.14, 2.6], 28)
    r.set([this.canvas.width, this.canvas.height, 0, 0], 32)
    this.device.queue.writeBuffer(this.renderBuf, 0, r)
  }

  private render() {
    const g = this.grids!
    const p = this.pipes!
    this.needsRender = false
    this.writeRender()
    const enc = this.device.createCommandEncoder()
    if (this.lightDirty) {
      const pass = enc.beginComputePass()
      pass.setPipeline(p.light)
      pass.setBindGroup(0, g.bg.light)
      pass.dispatchWorkgroups(Math.ceil(g.lightDims[0] / 4), Math.ceil(g.lightDims[1] / 4), Math.ceil(g.lightDims[2] / 4))
      pass.end()
      this.lightDirty = false
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    })
    pass.setPipeline(p.render)
    pass.setBindGroup(0, g.bg.render)
    pass.draw(3)
    pass.end()
    this.device.queue.submit([enc.finish()])
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

  private report(force: boolean) {
    const now = performance.now()
    if (!force && now - this.lastStatus < STATUS_MS) return
    this.lastStatus = now
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
    })
  }

  // For the lab's stills: the current frame drawn at an exact size, as an image file.
  async snapshot(w: number, h: number, type = 'image/jpeg', quality = 0.9): Promise<Blob | null> {
    if (!this.grids) return null
    this.canvas.width = w
    this.canvas.height = h
    this.layoutCamera()
    this.render()
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
