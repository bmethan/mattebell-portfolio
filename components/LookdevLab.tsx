'use client'
import { useEffect, useId, useRef, useState, useSyncExternalStore } from 'react'
import {
  LookdevEngine,
  DEFAULT_STATE,
  isDisplayOnly,
  isPosterState,
  TARGET_SPP,
  MESH_FURNACE_BOUNCES,
  type Compare,
  type KeyType,
  type LabState,
  type LabStatus,
  type Pass,
  type View,
} from './lookdev/engine'
import { LookdevEngineGPU } from './lookdev/gpu/engineGPU'
import { HERO_PRESETS, HERO_FAMILIES, familyOf, type Hero, type HeroFamily, PAINT_FINISHES, PAINT_ORDER, SKIN_ORDER, SKIN_TONES, STAGES, STAGE_ORDER, heroParams, paintHasFlakes } from './lookdev/materials'
import { MODELS, MODEL_ORDER, type Model } from './lookdev/models'
import { CITATIONS, CITATION_GROUPS, citeAuthors } from './lookdev/citations'
import { ENVIRONMENTS, ENV_ORDER, type Env } from './lookdev/environments'
import WorkIndicator from './WorkIndicator'

// The renderer: WebGPU where the browser has it (compute shaders, compiled in the background), WebGL otherwise.
// ?renderer=webgl forces the WebGL one (for comparing the two).
type Engine = LookdevEngine | LookdevEngineGPU
const noSubscribe = () => () => {}
const stamped = (line: string) => `${line} (at ${(performance.now() / 1000).toFixed(1)} s)`
// The ?debug readout: the renderer in use, the GPU it reports, and on WebGL why WebGPU was not used.
function describeRenderer(e: LookdevEngine | LookdevEngineGPU, gpuFailed: boolean): string {
  if (e instanceof LookdevEngineGPU) return `WebGPU, ${e.adapterName}`
  const gl = (e as unknown as { gl?: WebGL2RenderingContext }).gl
  const ext = gl?.getExtension('WEBGL_debug_renderer_info')
  const gpu = gl ? String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) : 'unknown GPU'
  const why = forcedWebGL() ? 'forced by ?renderer=webgl'
    : gpuFailed ? 'WebGPU failed on this device'
    : !navigator.gpu ? 'this browser has no WebGPU'
    : 'no WebGPU adapter'
  return `WebGL2, ${gpu}, ${e.targetsNote} (${why})`
}
const forcedWebGL = () => typeof location !== 'undefined' && new URLSearchParams(location.search).get('renderer') === 'webgl'

const PASSES: { id: Pass; label: string }[] = [
  { id: 'beauty', label: 'Beauty' },
  { id: 'diffuse', label: 'Diffuse' },
  { id: 'specular', label: 'Specular' },
  { id: 'albedo', label: 'Albedo' },
  { id: 'normal', label: 'Normal' },
]

const VIEWS: { id: View; label: string }[] = [
  { id: 'aces', label: 'ACES 2.0' },
  { id: 'agx', label: 'AgX' },
  { id: 'neutral', label: 'PBR Neutral' },
  { id: 'standard', label: 'Standard' },
]

const LIGHT_HINT_KEY = 'lab-light-moved'
const ENV_HINT_KEY = 'lab-env-turned'
const TOUCH_SLOP = 8 // px a finger must travel sideways before it drags the light
const POSTER_SPP = 2048 // samples per pixel of poster.jpg, the converged render of the default settings
const START_QUIET_MS = 400 // WebGPU starts once the lab is on screen and the page has not scrolled for this long

const AZ_RANGE = 5.2
const EL_MIN = 0.05
const EL_MAX = 1.35
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v))
const deg = (r: number) => Math.round((r * 180) / Math.PI)
// Render time as m:ss, as an IPR shows it.
const clock = (ms: number) => {
  const s = Math.floor(ms / 1000)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

// A slider on the skill meters' track: --pct drives the teal fill (see .lab-range).
function Range({
  min,
  max,
  step,
  value,
  label,
  onChange,
  disabled,
}: {
  min: number
  max: number
  step: number
  value: number
  label: string
  onChange: (v: number) => void
  disabled?: boolean
}) {
  return (
    <input
      type="range"
      className="lab-range"
      disabled={disabled}
      min={min}
      max={max}
      step={step}
      value={value}
      aria-label={label}
      style={{ '--pct': `${((value - min) / (max - min)) * 100}%` } as React.CSSProperties}
      onChange={e => onChange(+e.target.value)}
    />
  )
}

function Pill({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" className={`lab-pill${on ? ' on' : ''}`} aria-pressed={on} onClick={onClick}>
      {children}
    </button>
  )
}

// The renderer's own 2048 spp image of the default settings, one per frame shape (the frame is 4:3 on narrow
// screens). A <picture> downloads only the one that is shown; the files are served exactly as rendered.
// object-fit contain: the camera keeps a fixed horizontal extent, so a still lines up with the live frame.
function Still({ alt }: { alt: string }) {
  return (
    <picture>
      <source media="(max-width: 768px)" srcSet="/lookdev/poster-4x3.jpg" />
      <img className="lab-still-img" src="/lookdev/poster.jpg" alt={alt} decoding="async" />
    </picture>
  )
}

// Picking a model also dresses it in its own hero material, if it has one (the statue in skin).
function modelPatch(m: Model): Partial<LabState> {
  if (m === 'spheres') return { model: m }
  const { hero, yaw } = MODELS[m]
  return { model: m, ...(hero ? { hero, heroRoughness: null, heroAniso: null } : {}), ...(yaw !== undefined ? { modelYaw: yaw } : {}) }
}

// The lab's technical notes, one short entry per topic (shown under "Technical notes and references").
const TECH_NOTES: [string, string][] = [
  [
    'Renderer',
    'A progressive path tracer written for the browser: compute shaders on WebGPU where the browser has it, WebGL2 otherwise. Both draw the same samples from the same sequences and render the same image.',
  ],
  ['Color', 'Rendered in ACEScg, shown through ACES 2.0 (baked from OpenColorIO 2.5), AgX, Khronos PBR Neutral or a plain sRGB curve.'],
  [
    'Color chart',
    "A ColorChecker Classic: X-Rite's published values for its 24 patches, converted to ACEScg reflectances and shaded like the gray card, so every view transform can be judged against known colors.",
  ],
  [
    'Validation',
    'In a white furnace test, with multiple-scattering compensation on, every OpenPBR preset averages within 0.4% of 1.0, and rough glass stays within about 0.6% through a solid ball. Paint flakes are a lab extension, not part of OpenPBR, and lose under 1%.',
  ],
  ['Glass', 'Dispersion is spectral: each light path through dispersive glass is traced at one wavelength between 380 and 780 nm.'],
  [
    'Subsurface',
    "Skin and marble use OpenPBR subsurface: light refracts in through the surface, walks a scattering medium and leaves through a diffuse exit, as in Cycles' random walk. The official presets give no length unit; the lab reads them, and the depth of its honey, as centimeters, at each scene's real size (Winged Victory at the statue's 2.75 m).",
  ],
  ['Light mixer', "Each light renders into its own image, like a production renderer's light groups, so switching, dimming or recoloring a light needs no new render."],
  [
    'Fluorescence',
    'A lab extension, as OpenPBR has none. Light carries a fourth band, ultraviolet, which only the black light emits. A fluorescent base absorbs part of the ultraviolet and visible light reaching it and re-emits it in its own color, never more energy than it took in. In uranium glass and tonic water the dye fills the medium and glows throughout: there, direct light is sampled along straight lines through the surface, losing its reflection and the absorption on the way but not its focusing (no caustics inside), as production renderers do. The presets are authored from how their dyes behave, not measured, and the environment maps carry no ultraviolet.',
  ],
  [
    'Environments',
    "Four HDR environments from Poly Haven light the scene as a light of their own in the mixer. Each is importance sampled through an alias table and weighted against the materials' own sampling, scaled so its brightest light matches the key, and turned so that light starts where the key does. The white furnace test runs through the same code.",
  ],
  [
    'Denoiser',
    "While it renders, the image is shown through an edge-avoiding wavelet filter guided by albedo, normals and each pixel's noise, labelled in the frame. It hands over to the raw render as samples build up: the finished image is never filtered.",
  ],
]

// Whole seconds since the current wait began (key: what is awaited, or null), ticking while it lasts.
function useWaitSeconds(key: string | null) {
  const [secs, setSecs] = useState(0)
  useEffect(() => {
    setSecs(0)
    if (!key) return
    const t0 = performance.now()
    const id = setInterval(() => setSecs(Math.floor((performance.now() - t0) / 1000)), 1000)
    return () => clearInterval(id)
  }, [key])
  return secs
}
// The count reads after the first two seconds: shorter waits need no clock.
const waitClock = (secs: number) => (secs >= 2 ? `  ${secs} s` : '')

const KEY_TYPES: { id: KeyType; label: string }[] = [
  { id: 'softbox', label: 'Softbox' },
  { id: 'blacklight', label: 'Black light' },
]

const COMPARES: { id: Compare; label: string; sides?: [string, string] }[] = [
  { id: 'off', label: 'Off' },
  { id: 'denoise', label: 'Denoiser', sides: ['Raw', 'Denoised'] },
  { id: 'multiscatter', label: 'Multiple scattering', sides: ['Compensation off', 'Compensation on'] },
  { id: 'view', label: 'View transform', sides: ['ACES 2.0', 'AgX'] },
]

// Split compare: a line across the image with a version of it either side, dragged by its handle (or moved with
// the arrow keys once the handle has focus).
function SplitLine({ split, sides, frameRef, onSplit, onDrag }: {
  split: number
  sides: [string, string]
  frameRef: React.RefObject<HTMLDivElement | null>
  onSplit: (v: number) => void
  onDrag: (on: boolean) => void
}) {
  const drag = useRef<number | null>(null)
  const fromPointer = (e: React.PointerEvent) => {
    const r = frameRef.current?.getBoundingClientRect()
    if (r) onSplit(clamp((e.clientX - r.left) / r.width, 0.02, 0.98))
  }
  return (
    <div className="lab-split" style={{ left: `${split * 100}%` }}>
      <span className="lab-split-label left">{sides[0]}</span>
      <span className="lab-split-label right">{sides[1]}</span>
      <div
        className="lab-split-handle"
        role="slider"
        tabIndex={0}
        aria-label={`Compare line: ${sides[0]} left, ${sides[1]} right`}
        aria-valuemin={2}
        aria-valuemax={98}
        aria-valuenow={Math.round(split * 100)}
        aria-valuetext={`${Math.round(split * 100)}% across`}
        onPointerDown={e => {
          drag.current = e.pointerId
          e.currentTarget.setPointerCapture(e.pointerId)
          onDrag(true)
        }}
        onPointerMove={e => {
          if (drag.current === e.pointerId) fromPointer(e)
        }}
        onPointerUp={e => {
          if (drag.current !== e.pointerId) return
          drag.current = null
          onDrag(false)
        }}
        onPointerCancel={() => {
          drag.current = null
          onDrag(false)
        }}
        onKeyDown={e => {
          if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
          e.preventDefault()
          onSplit(clamp(split + (e.key === 'ArrowLeft' ? -0.02 : 0.02), 0.02, 0.98))
        }}
      />
    </div>
  )
}

type ControlTab = 'look' | 'light' | 'render'
const CONTROL_TABS: { id: ControlTab; label: string }[] = [
  { id: 'look', label: 'Look' },
  { id: 'light', label: 'Light' },
  { id: 'render', label: 'Render' },
]

// The controls in three tabs by intent: what is shot, how it is lit, how it is rendered and shown. WAI-ARIA tabs:
// the arrow keys, Home and End move between them.
function LabTabs({ id, tab, onTab }: { id: string; tab: ControlTab; onTab: (t: ControlTab) => void }) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const onKeyDown = (e: React.KeyboardEvent, i: number) => {
    const n = CONTROL_TABS.length
    const j = e.key === 'ArrowRight' ? (i + 1) % n : e.key === 'ArrowLeft' ? (i + n - 1) % n : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : -1
    if (j < 0) return
    e.preventDefault()
    onTab(CONTROL_TABS[j].id)
    refs.current[j]?.focus()
  }
  return (
    <div className="lab-tabs" role="tablist" aria-label="Lab controls">
      {CONTROL_TABS.map((t, i) => (
        <button
          key={t.id}
          ref={el => {
            refs.current[i] = el
          }}
          type="button"
          role="tab"
          id={`${id}-${t.id}`}
          aria-selected={tab === t.id}
          aria-controls={`${id}-panel`}
          tabIndex={tab === t.id ? 0 : -1}
          className={`lab-tab${tab === t.id ? ' on' : ''}`}
          onClick={() => onTab(t.id)}
          onKeyDown={e => onKeyDown(e, i)}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

const MIXER_LIGHTS = [
  { id: 'key', label: 'Key', gain: 'keyGain', kelvin: 'keyKelvin' },
  { id: 'fill', label: 'Fill', gain: 'fillGain', kelvin: 'fillKelvin' },
  { id: 'rim', label: 'Rim', gain: 'rimGain', kelvin: 'rimKelvin' },
] as const

// Light mixer: each light renders into its own image, so switching one, its intensity and its color temperature
// recombine the images on the spot instead of starting a new render. Moving the key still re-renders.
function LightMixer({ lab, update }: { lab: LabState; update: (patch: Partial<LabState>) => void }) {
  const id = useId()
  return (
    <div className="lab-mixer" role="group" aria-labelledby={id}>
      <span id={id} className="lab-label lab-mixer-title">
        Light mixer
      </span>
      <span className="lab-label lab-mixer-head" aria-hidden="true">
        Intensity
      </span>
      <span className="lab-label lab-mixer-head" aria-hidden="true">
        Color temp
      </span>
      {MIXER_LIGHTS.map(l => (
        <div key={l.id} className="lab-mixer-row">
          <Pill on={lab[l.id]} onClick={() => update({ [l.id]: !lab[l.id] })}>
            {l.label}
          </Pill>
          <Range
            min={-3}
            max={3}
            step={0.1}
            value={lab[l.gain]}
            label={`${l.label} intensity, stops`}
            onChange={v => update({ [l.gain]: v })}
          />
          <span className="lab-readout">
            {lab[l.gain] >= 0 ? '+' : ''}
            {lab[l.gain].toFixed(1)}
          </span>
          <Range
            min={2500}
            max={9000}
            step={100}
            value={lab[l.kelvin]}
            label={`${l.label} color temperature, kelvin`}
            onChange={v => update({ [l.kelvin]: v })}
            disabled={l.id === 'key' && lab.keyType === 'blacklight'}
          />
          <span className="lab-readout">{l.id === 'key' && lab.keyType === 'blacklight' ? 'UV' : `${lab[l.kelvin]}K`}</span>
        </div>
      ))}
      {lab.env !== 'none' && (
        <div className="lab-mixer-row">
          <Pill on={lab.envOn} onClick={() => update({ envOn: !lab.envOn })}>
            HDRI
          </Pill>
          <Range
            min={-3}
            max={3}
            step={0.1}
            value={lab.envGain}
            label="Environment intensity, stops"
            onChange={v => update({ envGain: v })}
          />
          <span className="lab-readout">
            {lab.envGain >= 0 ? '+' : ''}
            {lab.envGain.toFixed(1)}
          </span>
          {/* An HDR environment keeps its own colors: no color temperature. */}
          <span />
          <span />
        </div>
      )}
    </div>
  )
}

function Group({ label, children }: { label: string; children: React.ReactNode }) {
  const id = useId()
  return (
    <div className="lab-group" role="group" aria-labelledby={id}>
      <span id={id} className="lab-label">
        {label}
      </span>
      {children}
    </div>
  )
}

export default function LookdevLab() {
  const sectionRef = useRef<HTMLElement>(null)
  const frameRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  // WebGPU failed after taking the canvas (a canvas keeps its first kind of context): a fresh canvas, on WebGL.
  const [gpuFailed, setGpuFailed] = useState(false)
  const labRef = useRef<LabState>(DEFAULT_STATE)
  const dragging = useRef<number | null>(null) // pointerId of the drag moving the light
  const touchStart = useRef<{ id: number; x: number; y: number } | null>(null)
  const envGrab = useRef<{ u: number; rot: number } | null>(null) // where a drag turning the environment began

  const [lab, setLab] = useState<LabState>(DEFAULT_STATE)
  const [status, setStatus] = useState<LabStatus>({ spp: 0, target: TARGET_SPP, converged: false, preview: false, ms: 0, model: 'ready' })
  const [mode, setMode] = useState<'idle' | 'live' | 'fallback'>('idle')
  const [labels, setLabels] = useState<{ x: number; y: number }[]>([])
  const [furnaceMean, setFurnaceMean] = useState<number | null>(null)
  const [paused, setPaused] = useState(false)
  const [ready, setReady] = useState(false)
  // On WebGPU the controls and the light drag work before the renderer is ready (it starts late, see the effect
  // below): the settings wait in labRef and the first frame renders them. WebGL keeps them locked until ready,
  // its first draw can freeze the page. (Server render: locked, as without WebGPU.)
  const gpuInput = useSyncExternalStore(noSubscribe, () => !!navigator.gpu && !forcedWebGL(), () => false) && !gpuFailed
  const inputOn = ready || gpuInput
  // ?debug shows which renderer runs and on what GPU (for checking devices: an iPhone, a friend's laptop).
  const debugOn = useSyncExternalStore(noSubscribe, () => new URLSearchParams(location.search).has('debug'), () => false)
  const [rendererInfo, setRendererInfo] = useState('')
  // Startup stages, warnings and errors, oldest first (the last fourteen), each stamped with seconds since the page
  // opened. It outlives a switch to a fresh canvas, so a WebGPU failure and the WebGL attempt after it both show.
  const [debugLog, setDebugLog] = useState<string[]>([])
  useEffect(() => {
    if (!debugOn) return
    const add = (line: string) => setDebugLog(l => [...l.slice(-13), stamped(line)])
    const text = (args: unknown[]) => args.map(x => (x instanceof Error ? x.message : String(x))).join(' ').slice(0, 240)
    const onError = (ev: ErrorEvent) => add(`error: ${text([ev.error ?? ev.message])}`)
    const onRejection = (ev: PromiseRejectionEvent) => add(`error: ${text([ev.reason])}`)
    const consoleError = console.error
    const consoleWarn = console.warn
    console.error = (...args: unknown[]) => {
      add(`error: ${text(args)}`)
      consoleError(...args)
    }
    console.warn = (...args: unknown[]) => {
      add(`warning: ${text(args)}`)
      consoleWarn(...args)
    }
    window.addEventListener('error', onError)
    window.addEventListener('unhandledrejection', onRejection)
    return () => {
      console.error = consoleError
      console.warn = consoleWarn
      window.removeEventListener('error', onError)
      window.removeEventListener('unhandledrejection', onRejection)
    }
  }, [debugOn])
  const lastInFamily = useRef<Partial<Record<HeroFamily, Hero>>>({}) // each family's last picked material
  const [tab, setTab] = useState<ControlTab>('look')
  const startSecs = useWaitSeconds(mode === 'live' && !ready && !paused ? 'start' : null)
  const loadSecs = useWaitSeconds(status.model === 'loading' ? `${status.waitingFor}:${status.compiling}` : null)
  const tabsId = useId()
  const [canDenoise, setCanDenoise] = useState(false) // the GPU can draw the denoiser's extra images
  // The still is the renderer's own converged image of the default settings, so the live render (IPR) only
  // runs once something differs: the first edit (a click or drag on the frame, an arrow key, any control)
  // starts it, and putting every setting back shows the still again with the GPU idle. Hover and scrolling
  // past never cost anything.
  const pristine = isPosterState(lab)
  const [liveShown, setLiveShown] = useState(false) // the live render has drawn a frame since it (re)started
  // While it renders, the live image is shown through the denoiser, handing over to the raw render by the time
  // it converges (the still is never filtered); the HUD says so.
  const denoised =
    canDenoise && lab.denoise && !lab.furnace && !pristine && !status.converged &&
    (lab.pass === 'beauty' || lab.pass === 'diffuse' || lab.pass === 'specular')

  const update = (patch: Partial<LabState>) => {
    const prev = labRef.current
    const next = { ...prev, ...patch }
    labRef.current = next
    setLab(next)
    // Anything beyond exposure, view transform and the light mixer re-renders from zero samples, so the previous
    // count and furnace reading no longer apply.
    if (!isDisplayOnly(patch, prev)) {
      setFurnaceMean(null)
      setStatus(s => ({ ...s, spp: 0, converged: false, preview: false, ms: 0 }))
    }
    const toStill = isPosterState(next)
    if (toStill) setLiveShown(false)
    engineRef.current?.setState(patch)
    engineRef.current?.setActive(!toStill)
  }

  // Create the renderer once the page is idle (or earlier, if the section comes into view first), so its
  // shaders compile in the background before anyone scrolls here. It only renders while on screen.
  useEffect(() => {
    const section = sectionRef.current
    const frame = frameRef.current
    const canvas = canvasRef.current
    if (!section || !frame || !canvas) return
    let engine: Engine | null = null
    let unmounted = false

    const layout = () => {
      if (!engine) return
      const r = frame.getBoundingClientRect()
      engine.resize(r.width, r.height, window.devicePixelRatio || 1)
      const e = engine
      setLabels(e.labelPoints().map(p => e.project(p)))
    }

    // When the white furnace test converges, measure the mean radiance over the spheres.
    const handleStatus = (s: LabStatus) => {
      setStatus(s)
      if (s.spp > 0 || s.preview) setLiveShown(true)
      if (s.converged && labRef.current.furnace && engineRef.current) {
        const e = engineRef.current
        Promise.resolve(e.readSphereMean()).then(m => {
          if (engineRef.current === e && labRef.current.furnace) setFurnaceMean(m ? (m[0] + m[1] + m[2]) / 3 : null)
        })
      }
    }

    let visible = false
    let started = false
    let io: IntersectionObserver | null = null

    const fallback = () => {
      setRendererInfo('none: this browser offers neither WebGPU nor WebGL2')
      engine?.dispose()
      engine = null
      engineRef.current = null
      setMode('fallback')
      io?.disconnect()
    }

    const create = async () => {
      try {
        let e: Engine | null = null
        if (!gpuFailed && !forcedWebGL() && LookdevEngineGPU.detect()) {
          stage(`starting WebGPU, on ${startedBy}`)
          e = await LookdevEngineGPU.create(canvas, handleStatus, labRef.current, onDeviceLost)
          if (unmounted) {
            e?.dispose()
            return
          }
          if (!e) {
            // WebGPU may have taken this canvas before failing (a canvas keeps its first kind of context):
            // WebGL goes on a fresh one.
            stage('WebGPU did not start; WebGL next, on a fresh canvas')
            setGpuFailed(true)
            return
          }
        }
        if (!e) stage('starting WebGL')
        e ??= new LookdevEngine(canvas, handleStatus, labRef.current, onContextLost)
        engine = e
        setRendererInfo(describeRenderer(e, gpuFailed))
        stage('compiling shaders')
        engineRef.current = e
        if (process.env.NODE_ENV !== 'production') {
          ;(window as unknown as { __lookdev?: Engine }).__lookdev = e
        }
        setMode('live')
        setCanDenoise(e.canDenoise)
        layout()
        e.setVisible(visible)
        e.setActive(!isPosterState(labRef.current))
        e.whenReady.then(
          ok => {
            if (ok && engine === e) setReady(true)
            if (ok) stage('ready')
          },
          err => {
            stage(`failed: ${((x: unknown) => (x instanceof Error ? x.message : String(x)).slice(0, 240))(err)}`)
            console.error(err)
            if (engine !== e) return
            if (e instanceof LookdevEngineGPU) setGpuFailed(true)
            else fallback()
          },
        )
      } catch (err) {
        stage(`failed to start: ${((x: unknown) => (x instanceof Error ? x.message : String(x)).slice(0, 240))(err)}`)
        console.error(err)
        fallback()
      }
    }
    // The ?debug readout's stage line: what startup is doing, stamped with seconds since the page opened.
    const stage = (what: string) => setDebugLog(l => [...l.slice(-13), stamped(what)])

    let startedBy = ''
    const start = (why: string) => {
      if (started) return
      started = true
      startedBy = why
      if (LookdevEngineGPU.detect() || LookdevEngine.detect()) create()
      else fallback()
    }

    // GPU resets (driver updates, sleep and wake) lose the WebGL context. Pause behind the still frame and
    // rebuild everything when the browser restores it.
    const onContextLost = () => {
      engine?.dispose()
      engine = null
      engineRef.current = null
      setReady(false)
      setLiveShown(false)
      setPaused(true)
    }
    const onContextRestored = () => {
      setPaused(false)
      create()
    }
    // WebGPU has no restore event: a lost device is replaced with a new one, twice at most; a device that keeps
    // being lost hands over to WebGL (on a fresh canvas).
    let devicesLost = 0
    const onDeviceLost = () => {
      onContextLost()
      if (++devicesLost > 2) {
        stage('WebGPU device lost three times; WebGL next, on a fresh canvas')
        setPaused(false)
        setGpuFailed(true)
        return
      }
      setTimeout(() => !unmounted && onContextRestored(), 1000)
    }
    canvas.addEventListener('webglcontextrestored', onContextRestored)

    // When to start. WebGPU starts when the page comes to rest within a screen of the lab (a visitor reading the
    // section above, or looking at the still), or as soon as a visitor points at or uses the lab: getting a GPU
    // adapter and compiling the first shaders holds up the browser's frames (measured: up to 0.75 s). Started at
    // load it stuttered the scroll down from the top; started on arrival it stalled the first drag. Over a page
    // at rest nothing moves. WebGL starts at idle, as before: its compile cannot be hidden either way (see
    // FIRST_DRAW_NOTICE_MS).
    const gpuPath = !gpuFailed && !forcedWebGL() && LookdevEngineGPU.detect()
    let cancelIdle = () => {}
    let quietTimer = 0
    let near = false
    const armQuiet = () => {
      window.clearTimeout(quietTimer)
      quietTimer = window.setTimeout(() => near && start('the page resting near the lab'), START_QUIET_MS)
    }
    const nearby = new IntersectionObserver(entries => {
      near = entries.some(en => en.isIntersecting)
      if (gpuPath && near) armQuiet()
    }, { rootMargin: '100% 0px' })
    nearby.observe(section)
    const startNow = (ev: Event) => start(ev.type)
    if (gpuPath) {
      window.addEventListener('scroll', armQuiet, { passive: true })
      frame.addEventListener('pointerenter', startNow)
      section.addEventListener('pointerdown', startNow)
      section.addEventListener('focusin', startNow)
    } else {
      if ('requestIdleCallback' in window) {
        const id = requestIdleCallback(() => start('idle'), { timeout: 4000 })
        cancelIdle = () => cancelIdleCallback(id)
      } else {
        const id = setTimeout(() => start('idle'), 2000)
        cancelIdle = () => clearTimeout(id)
      }
      // Approaching the section starts the renderer early if the idle callback has not yet.
      io = new IntersectionObserver(entries => entries.some(en => en.isIntersecting) && start('approach'), {
        rootMargin: '200px 0px',
      })
      io.observe(section)
    }
    // Only a frame that is actually on screen renders.
    const onScreen = new IntersectionObserver(entries => {
      visible = entries.some(en => en.isIntersecting)
      engine?.setVisible(visible)
    })
    onScreen.observe(frame)
    const ro = new ResizeObserver(layout)
    ro.observe(frame)
    return () => {
      unmounted = true
      cancelIdle()
      window.clearTimeout(quietTimer)
      window.removeEventListener('scroll', armQuiet)
      nearby.disconnect()
      frame.removeEventListener('pointerenter', startNow)
      section.removeEventListener('pointerdown', startNow)
      section.removeEventListener('focusin', startNow)
      io?.disconnect()
      onScreen.disconnect()
      ro.disconnect()
      canvas.removeEventListener('webglcontextrestored', onContextRestored)
      engine?.dispose()
      engineRef.current = null
      // Fast Refresh keeps state across a remount, but the next engine starts cold.
      setReady(false)
      setLiveShown(false)
    }
  }, [gpuFailed])

  // Labels follow the plate: a model moves the reference balls aside, and its own label tracks the turntable; the
  // chart widens the frame.
  useEffect(() => {
    const e = engineRef.current
    if (e) setLabels(e.labelPoints().map(p => e.project(p)))
  }, [lab.model, lab.modelYaw, lab.chart, status.model, ready])

  // "Drag me" by the key light's marker until the visitor first moves the light, and "Drag to turn" by the
  // environment's handle until they first turn it (each remembered on this device).
  const [hintsDone, setHintsDone] = useState({ light: true, env: true }) // until read below: no flash for returners
  useEffect(() => {
    const seen = (key: string) => {
      try {
        return localStorage.getItem(key) === '1'
      } catch {
        return false
      }
    }
    setHintsDone({ light: seen(LIGHT_HINT_KEY), env: seen(ENV_HINT_KEY) })
  }, [])
  const lightMoved = hintsDone.light
  const noteHintDone = (which: 'light' | 'env') => {
    if (hintsDone[which]) return
    setHintsDone(h => ({ ...h, [which]: true }))
    try {
      localStorage.setItem(which === 'light' ? LIGHT_HINT_KEY : ENV_HINT_KEY, '1')
    } catch {}
  }

  // With the key off and an environment lighting the scene, a drag turns the environment instead: grabbed where
  // the drag starts, a full turn across the image's width.
  const turnsEnv = !lab.key && lab.env !== 'none'
  const setFromPointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    noteHintDone(turnsEnv ? 'env' : 'light')
    const r = e.currentTarget.getBoundingClientRect()
    const u = clamp((e.clientX - r.left) / r.width, 0, 1)
    const v = clamp((e.clientY - r.top) / r.height, 0, 1)
    if (turnsEnv) {
      const g = (envGrab.current ??= { u, rot: labRef.current.envRot })
      const rot = g.rot - (u - g.u) * 2 * Math.PI // the backdrop follows the pointer
      update({ envRot: Math.atan2(Math.sin(rot), Math.cos(rot)) })
    } else update({ keyAz: (u - 0.5) * AZ_RANGE, keyEl: clamp(EL_MAX - v * (EL_MAX - EL_MIN) * 1.1, EL_MIN, EL_MAX) })
  }

  const beginDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    envGrab.current = null
    dragging.current = e.pointerId
    e.currentTarget.setPointerCapture(e.pointerId)
    engineRef.current?.setInteracting(true)
    setFromPointer(e)
  }
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (lab.furnace || !inputOn || dragging.current !== null) return
    if (e.pointerType !== 'touch') {
      if (e.button === 0) beginDrag(e) // mouse or pen: press places the light, a drag moves it
      return
    }
    // Touch: only a sideways drag moves the light. A vertical move is a page scroll (touch-action: pan-y hands
    // it to the browser, which cancels the pointer), and a tap is often just stopping a scroll fling, so
    // neither changes anything.
    touchStart.current = { id: e.pointerId, x: e.clientX, y: e.clientY }
  }
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragging.current !== null) {
      if (e.pointerId === dragging.current) setFromPointer(e) // one pointer drives the light
      return
    }
    const s = touchStart.current
    if (!s || s.id !== e.pointerId) return
    const dx = e.clientX - s.x
    const dy = e.clientY - s.y
    if (Math.abs(dx) >= TOUCH_SLOP && Math.abs(dx) > Math.abs(dy)) {
      touchStart.current = null
      beginDrag(e)
    }
  }
  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (touchStart.current?.id === e.pointerId) touchStart.current = null
    if (dragging.current !== e.pointerId) return
    dragging.current = null
    engineRef.current?.setInteracting(false)
  }
  const onKeyDown = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (lab.furnace || !inputOn) return
    const s = 0.08
    let { keyAz, keyEl } = labRef.current // held keys repeat faster than React re-renders
    if (turnsEnv && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      e.preventDefault()
      noteHintDone('env')
      const r = labRef.current.envRot + (e.key === 'ArrowLeft' ? -s : s)
      update({ envRot: Math.atan2(Math.sin(r), Math.cos(r)) })
      return
    }
    if (e.key === 'ArrowLeft') keyAz -= s
    else if (e.key === 'ArrowRight') keyAz += s
    else if (e.key === 'ArrowUp') keyEl += s
    else if (e.key === 'ArrowDown') keyEl -= s
    else return
    e.preventDefault()
    noteHintDone('light')
    update({ keyAz: clamp(keyAz, -AZ_RANGE / 2, AZ_RANGE / 2), keyEl: clamp(keyEl, EL_MIN, EL_MAX) })
  }

  const hero = HERO_PRESETS[lab.hero]
  const family = familyOf(lab.hero)
  lastInFamily.current[family.id] = lab.hero
  const pickHero = (h: Hero) => update({ hero: h, heroRoughness: null, heroAniso: null })
  // The black light is judged alone, as in a dark room: picking it turns the fill, rim and environment off (the
  // mixer brings them back); back to the softbox, they return.
  const pickKeyType = (t: KeyType) => {
    if (t === lab.keyType) return
    const on = t === 'softbox'
    update({ keyType: t, key: true, fill: on, rim: on, envOn: on })
  }
  // An environment first lights the scene alone (the softboxes switch off; the mixer brings them back), as it is
  // judged in lookdev. Back to none, the softbox rig returns.
  const pickEnv = (en: Env) => {
    if (en === lab.env) return
    const rig = en === 'none' ? { key: true, fill: true, rim: true } : lab.env === 'none' ? { key: false, fill: false, rim: false } : {}
    update({ env: en, envOn: true, ...rig })
  }
  const heroBase = heroParams(lab.hero, lab.paint, lab.flakes, lab.skinTone)
  const heroRough = lab.heroRoughness ?? heroBase[hero.roughnessParam]
  const brushed = heroBase.specular_roughness_anisotropy > 0
  const heroAniso = lab.heroAniso ?? heroBase.specular_roughness_anisotropy
  const flaked = lab.hero === 'carpaint' && paintHasFlakes(lab.paint, lab.flakes)
  const heroName =
    lab.hero === 'carpaint' && lab.paint !== 'solid'
      ? `${PAINT_FINISHES[lab.paint].label}${flaked ? ' flake' : ''} car paint`
      : hero.label
  const markerX = (lab.keyAz / AZ_RANGE + 0.5) * 100
  const markerY = ((EL_MAX - lab.keyEl) / ((EL_MAX - EL_MIN) * 1.1)) * 100
  const model = lab.model === 'spheres' ? null : MODELS[lab.model]
  const ballNames = ['18% gray', 'Chrome', model ? `${model.label}, ${heroName.toLowerCase()}` : heroName, 'Color chart']

  return (
    <section ref={sectionRef} id="lab" className="section-pad" style={{ borderBottom: '0.5px solid var(--border)' }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
          Lookdev lab
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
        </div>
        <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em', marginBottom: 12 }}>
          Light, rendered live
        </h2>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, maxWidth: 640 }}>
          A progressive path tracer running a subset of OpenPBR Surface on the references every lighting department
          shoots: a color chart, an 18% gray ball, a chrome ball, and a hero material.
        </p>
      </div>

      <p className="lab-howto">
        <span className="lab-howto-pointer">Drag anywhere on the image</span>
        <span className="lab-howto-touch">Swipe sideways on the image</span>
        {turnsEnv ? (
          ' to turn the environment'
        ) : (
          <>
            {' '}
            to move the key light (
            <span className="lab-howto-marker" aria-hidden="true" />
            <span className="sr-only">the small circle</span> marks it)
          </>
        )}
        <span aria-hidden="true"> · </span>
        <span className="sr-only">. </span>Every control below is live
      </p>

      <div ref={frameRef} className="lab-frame">
        {mode === 'fallback' ? (
          <Still alt="Path traced lookdev references: a color chart, an 18% gray card ball, a chromium ball and a clear coated car paint ball under a three point softbox rig" />
        ) : (
          <canvas
            key={gpuFailed ? 'webgl' : 'gpu'}
            ref={canvasRef}
            tabIndex={0}
            role="application"
            aria-roledescription={turnsEnv ? 'environment control' : 'key light control'}
            aria-label={`Path traced render of the lookdev references. Drag, or use the arrow keys, to ${turnsEnv ? 'turn the environment' : 'move the key light'}.`}
            className="lab-canvas"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onKeyDown={onKeyDown}
          />
        )}

        {mode !== 'fallback' && (
          // The still covers the canvas for the default settings, and after an edit until the live render has
          // drawn its first frame.
          <div className={`lab-still${ready && !pristine && liveShown ? ' gone' : ''}`}>
            <Still alt="" />
          </div>
        )}
        {mode !== 'fallback' && !ready && (
          <div className="lab-hud lab-hud-left">
            {paused ? (
              'Renderer paused. It resumes when the browser restores graphics.'
            ) : (
              <>
                <WorkIndicator className="lab-work" />
                {status.finishing
                  ? 'Still frame. Finishing the live renderer, first time only; the page may pause while it finishes.'
                  : `Still frame. The live renderer is compiling its shaders, first time only.${waitClock(startSecs)}`}
              </>
            )}
          </div>
        )}
        {mode === 'live' && ready && lab.compare !== 'off' && !lab.furnace && (
          <SplitLine
            split={lab.split}
            sides={COMPARES.find(c => c.id === lab.compare)!.sides!}
            frameRef={frameRef}
            onSplit={v => update({ split: v })}
            onDrag={on => engineRef.current?.setInteracting(on)}
          />
        )}
        {mode === 'live' &&
          labels.map((p, i) => (
            <div key={i} className="lab-ball-label" style={{ left: `${p.x * 100}%`, top: `calc(${p.y * 100}% + 12px)` }}>
              {ballNames[i]}
            </div>
          ))}
        {mode === 'live' && ready && (
          <>
            <div className="sr-only" aria-live="polite">
              {turnsEnv
                ? `Environment turned ${deg(lab.envRot)} degrees`
                : `Key light at ${deg(lab.keyAz)} degrees azimuth, ${deg(lab.keyEl)} degrees elevation`}
            </div>
            <div className="lab-hud lab-hud-left">
              {lab.furnace
                ? 'White furnace'
                : turnsEnv
                  ? `${ENVIRONMENTS[lab.env as Exclude<Env, 'none'>].label} environment  ${deg(lab.envRot)}°`
                  : lab.keyType === 'blacklight'
                    ? `Black light ${deg(lab.keyAz)}° az  ${deg(lab.keyEl)}° el`
                    : `Key ${deg(lab.keyAz)}° az  ${deg(lab.keyEl)}° el  ${lab.keyKelvin}K`}
            </div>
            <div className="lab-hud lab-hud-right">
              {status.model === 'failed' ? (
                model ? `The ${model.label.toLowerCase()} could not load` : 'The shaders for this scene failed to compile'
              ) : status.finishing ? (
                <>
                  <WorkIndicator className="lab-work" />
                  {`Finishing ${status.finishing}, first time only; the page may pause while it finishes`}
                </>
              ) : status.model === 'loading' ? (
                <>
                  <WorkIndicator className="lab-work" />
                  {status.waitingFor === 'model' && model
                    ? `Loading the ${model.label.toLowerCase()}`
                    : status.waitingFor === 'environment'
                      ? 'Loading the environment'
                      : `Compiling ${status.compiling ?? 'shaders'}, first time only`}
                  {waitClock(loadSecs)}
                </>
              ) : (
                <>
                  {!pristine && !status.converged && <WorkIndicator className="lab-work" />}
                  {PASSES.find(p => p.id === lab.pass)?.label}
                  {denoised && ', denoised'}
                  {'  '}
                  {pristine
                    ? `${POSTER_SPP} spp`
                    : status.preview
                      ? 'Preview'
                      : `${clock(status.ms)}  ${status.converged ? `${status.spp} spp` : `${status.spp} / ${status.target} spp`}`}
                </>
              )}
            </div>
            <div
              className="lab-progress"
              style={{ transform: `scaleX(${pristine ? 1 : status.spp / status.target})` }}
            />
            {lab.key && !lab.furnace && (
              <div
                className={`lab-marker${lightMoved ? '' : ' hinting'}`}
                style={{ left: `${markerX}%`, top: `${clamp(markerY, 0, 100)}%` }}
              />
            )}
            {/* With the key off and no environment, the marker stays where the key would be, faded and named, so
                it is clear why nothing lights up. */}
            {!lab.key && !turnsEnv && !lab.furnace && (
              <>
                <div className="lab-marker off" style={{ left: `${markerX}%`, top: `${clamp(markerY, 0, 100)}%` }} />
                <div
                  className={`lab-drag-hint off${markerX > 70 ? ' left' : ''}`}
                  style={{ left: `${markerX}%`, top: `${clamp(markerY, 0, 100)}%` }}
                  aria-hidden="true"
                >
                  Key off
                </div>
              </>
            )}
            {/* An environment lighting the scene alone: a drag turns it, and a two-way handle says so where the
                circle was (the key light, and its marker, are off). */}
            {turnsEnv && !lab.furnace && (
              <>
                <div className={`lab-env-handle${hintsDone.env ? '' : ' hinting'}`} aria-hidden="true">
                  <svg viewBox="0 0 28 12" width="28" height="12">
                    <path d="M1.5 6h25M5.5 2 1.5 6l4 4M22.5 2l4 4-4 4" />
                  </svg>
                </div>
                {!hintsDone.env && (
                  <div className="lab-drag-hint lab-env-hint" aria-hidden="true">
                    <span className="lab-howto-pointer">Drag to turn</span>
                    <span className="lab-howto-touch">Swipe to turn</span>
                  </div>
                )}
              </>
            )}
            {lab.key && !lab.furnace && !lightMoved && (
              <div
                className={`lab-drag-hint${markerX > 70 ? ' left' : ''}`}
                style={{ left: `${markerX}%`, top: `${clamp(markerY, 0, 100)}%` }}
                aria-hidden="true"
              >
                Drag me
              </div>
            )}
          </>
        )}
        {debugOn && (
          <div className="lab-hud lab-hud-bottom">
            {`Renderer: ${rendererInfo || 'not started yet (it starts when the page rests near the lab)'}`}
            {debugLog.map(line => `\n${line}`).join('')}
          </div>
        )}
        {mode === 'fallback' && (
          <div className="lab-hud lab-hud-left">Still frame. Your browser lacks the WebGPU or WebGL2 features the live renderer needs.</div>
        )}
      </div>

      {mode !== 'fallback' && (
        <fieldset className="lab-controls" disabled={!inputOn}>
          <LabTabs id={tabsId} tab={tab} onTab={setTab} />
          <div className="lab-panel" role="tabpanel" id={`${tabsId}-panel`} aria-labelledby={`${tabsId}-${tab}`}>
            {tab === 'look' && (
              <>
                <div className="lab-row">
                  <Group label="Model">
                    {MODEL_ORDER.map(m => (
                      <Pill key={m} on={lab.model === m} onClick={() => update(modelPatch(m))}>
                        {m === 'spheres' ? 'Reference balls' : MODELS[m].label}
                      </Pill>
                    ))}
                  </Group>
                  <Group label="Stage">
                    {STAGE_ORDER.map(st => (
                      <Pill key={st} on={lab.stage === st} onClick={() => update({ stage: st })}>
                        {STAGES[st].label}
                      </Pill>
                    ))}
                  </Group>
                  <Group label="Chart">
                    <Pill on={lab.chart} onClick={() => update({ chart: !lab.chart })}>
                      Color chart {lab.chart ? 'on' : 'off'}
                    </Pill>
                  </Group>
                  {model && (
                    <Group label="Turntable">
                      <Range
                        min={-180}
                        max={180}
                        step={1}
                        value={deg(lab.modelYaw)}
                        label="Turntable angle, degrees"
                        onChange={v => update({ modelYaw: (v * Math.PI) / 180 })}
                      />
                      <span className="lab-readout">{deg(lab.modelYaw)}°</span>
                    </Group>
                  )}
                </div>

                <div className="lab-row">
                  <Group label="Hero">
                    {model?.heroes
                      ? model.heroes.map(h => (
                          <Pill key={h} on={lab.hero === h} onClick={() => pickHero(h)}>
                            {HERO_PRESETS[h].label}
                          </Pill>
                        ))
                      : HERO_FAMILIES.map(f => (
                          <Pill key={f.id} on={f.id === family.id} onClick={() => pickHero(lastInFamily.current[f.id] ?? f.heroes[0])}>
                            {f.label}
                          </Pill>
                        ))}
                  </Group>
                </div>

                {!model?.heroes && family.heroes.length > 1 && (
                  <div className="lab-row">
                    <Group label={family.label}>
                      {family.heroes.map(h => (
                        <Pill key={h} on={lab.hero === h} onClick={() => pickHero(h)}>
                          {HERO_PRESETS[h].label}
                        </Pill>
                      ))}
                    </Group>
                  </div>
                )}

                {lab.hero === 'carpaint' && (
                  <div className="lab-row">
                    <Group label="Finish">
                      {PAINT_ORDER.map(f => (
                        <Pill key={f} on={lab.paint === f} onClick={() => update({ paint: f, heroRoughness: null })}>
                          {PAINT_FINISHES[f].label}
                        </Pill>
                      ))}
                    </Group>
                    {lab.paint !== 'solid' && (
                      <Group label="Flakes">
                        <Pill on={lab.flakes} onClick={() => update({ flakes: !lab.flakes, heroRoughness: null })}>
                          Flakes {lab.flakes ? 'on' : 'off'}
                        </Pill>
                      </Group>
                    )}
                  </div>
                )}

                {lab.hero === 'skin' && (
                  <div className="lab-row">
                    <Group label="Tone">
                      {SKIN_ORDER.map(t => (
                        <Pill key={t} on={lab.skinTone === t} onClick={() => update({ skinTone: t, heroRoughness: null })}>
                          {SKIN_TONES[t].label}
                        </Pill>
                      ))}
                    </Group>
                  </div>
                )}

                <div className="lab-row">
                  <Group label={`${hero.label} roughness`}>
                    <Range
                      min={0}
                      max={1}
                      step={0.01}
                      value={heroRough}
                      label={`${hero.label} roughness`}
                      onChange={v => update({ heroRoughness: v })}
                    />
                    <span className="lab-readout">{heroRough.toFixed(2)}</span>
                  </Group>
                  {brushed && (
                    <Group label="Brushing">
                      <Range
                        min={0}
                        max={1}
                        step={0.01}
                        value={heroAniso}
                        label="Brushing, specular roughness anisotropy"
                        onChange={v => update({ heroAniso: v })}
                      />
                      <span className="lab-readout">{heroAniso.toFixed(2)}</span>
                    </Group>
                  )}
                </div>
              </>
            )}
            {tab === 'light' && (
              <>
                <div className="lab-row">
                  <Group label="Key lamp">
                    {KEY_TYPES.map(t => (
                      <Pill key={t.id} on={lab.keyType === t.id} onClick={() => pickKeyType(t.id)}>
                        {t.label}
                      </Pill>
                    ))}
                  </Group>
                </div>

                <LightMixer lab={lab} update={update} />

                <div className="lab-row">
                  <Group label="Environment">
                    {ENV_ORDER.map(en => (
                      <Pill key={en} on={lab.env === en} onClick={() => pickEnv(en)}>
                        {en === 'none' ? 'None' : ENVIRONMENTS[en].label}
                      </Pill>
                    ))}
                  </Group>
                  {lab.env !== 'none' && (
                    <Group label="Turn">
                      <Range
                        min={-180}
                        max={180}
                        step={1}
                        value={deg(lab.envRot)}
                        label="Environment turn, degrees"
                        onChange={v => update({ envRot: (v * Math.PI) / 180 })}
                      />
                      <span className="lab-readout">{deg(lab.envRot)}°</span>
                    </Group>
                  )}
                </div>
              </>
            )}
            {tab === 'render' && (
              <>
                <div className="lab-row">
                  <Group label="Pass">
                    {PASSES.map(p => (
                      <Pill key={p.id} on={lab.pass === p.id} onClick={() => update({ pass: p.id })}>
                        {p.label}
                      </Pill>
                    ))}
                  </Group>
                  {canDenoise && (
                    <Group label="Denoiser">
                      <Pill on={lab.denoise} onClick={() => update({ denoise: !lab.denoise })}>
                        Denoiser {lab.denoise ? 'on' : 'off'}
                      </Pill>
                    </Group>
                  )}
                </div>

                <div className="lab-row">
                  <Group label="View">
                    {VIEWS.map(v => (
                      <Pill key={v.id} on={lab.view === v.id} onClick={() => update({ view: v.id })}>
                        {v.label}
                      </Pill>
                    ))}
                  </Group>
                  <Group label="Exposure">
                    <Range
                      min={-3}
                      max={3}
                      step={0.1}
                      value={lab.exposure}
                      label="Exposure, stops"
                      onChange={v => update({ exposure: v })}
                    />
                    <span className="lab-readout">
                      {lab.exposure >= 0 ? '+' : ''}
                      {lab.exposure.toFixed(1)}
                    </span>
                  </Group>
                </div>

                <div className="lab-row">
                  <Group label="Compare">
                    {COMPARES.filter(c => c.id !== 'denoise' || canDenoise).map(c => (
                      <Pill
                        key={c.id}
                        on={lab.compare === c.id}
                        // Turned on, the line starts through the hero, where the difference matters.
                        onClick={() => update({ compare: c.id, ...(lab.compare === 'off' && labels[2] ? { split: clamp(labels[2].x, 0.02, 0.98) } : {}) })}
                      >
                        {c.label}
                      </Pill>
                    ))}
                  </Group>
                </div>

                <div className="lab-row">
                  <Group label="Validation">
                    <Pill on={lab.multiscatter} onClick={() => update({ multiscatter: !lab.multiscatter })}>
                      Multiple scattering {lab.multiscatter ? 'on' : 'off'}
                    </Pill>
                    <Pill on={lab.furnace} onClick={() => update({ furnace: !lab.furnace })}>
                      Furnace test
                    </Pill>
                  </Group>
                  {lab.furnace && (
                    <span className="lab-note">
                      Uniform white light, no floor, every albedo at 1. An energy conserving material disappears into the
                      background.
                      {furnaceMean !== null &&
                        ` Measured mean over the ${model ? 'balls and model' : 'spheres'}: ${furnaceMean.toFixed(3)} (ideal 1.000).`}
                      {model &&
                        ` A model reads a little under 1: light caught in its cavities needs dozens of bounces to escape, and the renderer stops at ${MESH_FURNACE_BOUNCES}.`}
                      {lab.hero === 'skin' &&
                        ' Skin reads a little under 1 in blue: with nothing absorbed, a walk ends only by leaving, and the lab stops the longest walks.'}
                    </span>
                  )}
                </div>
              </>
            )}
          </div>
        </fieldset>
      )}

      {model && (
        <p className="lab-credit">
          <span className="lab-label">Model</span>
          <span>
            {model.credit.text}{' '}
            <a href={model.credit.source} target="_blank" rel="noopener noreferrer">
              Source
            </a>
            {' · '}
            <a href={model.credit.licenseUrl} target="_blank" rel="noopener noreferrer">
              {model.credit.license}
            </a>
          </span>
        </p>
      )}
      {lab.env !== 'none' && (
        <p className="lab-credit">
          <span className="lab-label">Environment</span>
          <span>
            {ENVIRONMENTS[lab.env].title} by {ENVIRONMENTS[lab.env].authors}, Poly Haven.{' '}
            <a href={ENVIRONMENTS[lab.env].url} target="_blank" rel="noopener noreferrer">
              Source
            </a>
            {' · '}
            <a href="https://creativecommons.org/publicdomain/zero/1.0/" target="_blank" rel="noopener noreferrer">
              CC0
            </a>
          </span>
        </p>
      )}

      <details className="lab-tech">
        <summary>Technical notes and references</summary>
        <div className="lab-tech-body">
          <dl className="lab-tech-notes">
            {TECH_NOTES.map(([term, text]) => (
              <div key={term}>
                <dt>{term}</dt>
                <dd>{text}</dd>
              </div>
            ))}
          </dl>
          <div className="lab-refs">
            {CITATION_GROUPS.map(g => (
              <div key={g} className="lab-refs-group">
                <h3>{g}</h3>
                <ol>
                  {CITATIONS.filter(c => c.group === g).map(c => (
                    <li key={c.id}>
                      {citeAuthors(c)} ({c.year}).{' '}
                      <a href={c.url} target="_blank" rel="noopener noreferrer">
                        {c.title}
                      </a>
                      . <span className="lab-refs-venue">{c.venue}.</span>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
        </div>
      </details>
    </section>
  )
}
