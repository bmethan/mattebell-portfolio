'use client'
// The lab's pyro mode: a 3D smoke solver and volume renderer on the GPU (see pyro/engine.ts), stirred by dragging
// across the image, with the solver's controls below. It mounts when the Lab switch first selects it, and starts
// at once (choosing it is the visitor's go-ahead).
import { useEffect, useId, useRef, useState, useSyncExternalStore } from 'react'
import { PyroEngine, type PyroStatus } from './pyro/engine'
import { PRESETS, PYRO_DEFAULTS, RESOLUTIONS, RESOLUTION_ORDER, SOURCE_LOOKS, type PyroState, type SourceType } from './pyro/params'
import { PYRO_CITATIONS, PYRO_CITATION_GROUPS } from './pyro/citations'
import { citeAuthors } from './lookdev/citations'
import { Group, Pill, Range, Tabs } from './lab/ui'

const TOUCH_SLOP = 8 // px a finger must travel sideways before it stirs (a vertical swipe scrolls the page)
const noSubscribe = () => () => {}

type PyroTab = 'source' | 'forces' | 'fire' | 'shading' | 'solver'
const PYRO_TABS = [
  { id: 'source', label: 'Source' },
  { id: 'forces', label: 'Forces' },
  { id: 'fire', label: 'Fire' },
  { id: 'shading', label: 'Shading' },
  { id: 'solver', label: 'Solver' },
] as const

const TECH_NOTES: [string, string][] = [
  [
    'Solver',
    'Air, smoke and fire on a staggered grid, the layout production solvers use: velocity on the cell faces, everything else at the cell centres. Each step, 30 per simulated second, advects the velocity, adds the forces, then projects the flow with a pressure solve: incompressible, except where burning gas expands.',
  ],
  [
    'Pressure',
    'Geometric multigrid, as production solvers use: each V-cycle smooths with red-black Gauss-Seidel on the full grid, hands the remaining error down through coarser grids (each half the resolution of the last), where the broad, slow-to-fix part of it is cheap to remove, and brings the corrections back up. Two V-cycles a step, started from the previous step’s pressure, leave about 1% of the flow’s divergence, the floor set by storing velocity at half precision (measured on this solver). The Solver tab also offers plain red-black SOR for comparison: 40 sweeps reach the same floor at four to eight times the GPU time, and a shorter solve shows as smoke that swells and loses some of its swirl.',
  ],
  [
    'Advection',
    'Velocity and scalars move with Modified MacCormack: a forward and a backward semi-Lagrangian step, half the round-trip error added back, then clamped to the values the forward step read, so detail survives without new extremes. The Solver tab also offers semi-Lagrangian alone, the softer, more diffusive choice, and for the velocity advection-reflection (Zehnder, Narain and Thomaszewski 2018): it advects half a step, projects, reflects the velocity through the projection and advects the reflection the second half, which loses far less of the flow’s energy to the projection. It needs two full pressure solves a step (with one V-cycle each, the reflection amplifies the solver’s error until it blows up). Measured against fire experiments it mixes the plume no differently from MacCormack at this resolution, where the grid itself, not the scheme, limits the turbulence, so MacCormack, the cheaper of the two, is the default.',
  ],
  [
    'Forces',
    'Buoyancy at its physical strength: gas hotter than the air around it rises at g (T − T₀) / T₀. Vorticity confinement, which restores the small swirls a coarse grid smooths away. Divergence-free turbulence from the curl of a noise potential, rising with the plume and churning as it rises (held still, the updraft would draw its pushes out into steady vertical streaks): for a smoke source its strength is the Turbulence control, in m/s²; for a fire it is a turbulence model, calibrated against measurements (see Validation). Wind, and a smoke source’s push. Cooling and dissipation default to 0: the plume cools by mixing with the air it draws in.',
  ],
  [
    'Combustion',
    'The fire follows the fast-chemistry model of NIST’s Fire Dynamics Simulator (FDS), the standard tool of fire-safety engineering. The bed is a disc on the floor that gives off wood gas (Douglas-fir volatiles, C3.4H6.2O2.5) through its surface at the rate that releases the chosen heat, as FDS models a burning surface, the vapour pushing out at its own small blowing speed. Two fields track it: the mixture fraction, the share of the gas that came from the fuel, and the fuel still unburned. In each cell, fuel burns with the oxygen mixed in with it as fast as turbulence mixes them (the eddy dissipation concept, with FDS’s mixing time: the turnover time of the eddies finer than the grid, from the same Smagorinsky model the Mixing note describes, and never slower than the buoyant time √(2Δx / g)), so the flame sits where fuel meets air, a sheet around a fuel-rich core, and its height comes out of the simulation rather than a control. Burning a kilogram of the fuel takes 1.36 kg of oxygen and releases 17.8 MJ (13.1 MJ per kilogram of oxygen, Huggett’s constant); 30% of the heat is radiated away and the rest heats the gas, and 1.5% of the fuel’s mass becomes soot, wood’s measured yield. Heated gas expands at (dT/dt) / T, which the pressure solve turns into the flame’s outward push.',
  ],
  [
    'Mixing',
    'Three corrections make a coarse grid mix the way gas does. A fire’s fields are kept per unit volume (the density deficit and partial densities) rather than as a temperature and mass fractions, because the solver mixes by volume and gas mixes by mass: equal volumes of 1300 K gas and 293 K air make 478 K, not the 796 K an average of temperatures gives. Without this the plume stayed three times too hot. And mixing finer than the grid, as large-eddy simulations model it: Smagorinsky’s eddy diffusivity over a turbulent Schmidt number (0.2 and 0.5, FDS’s values), on the scalars. And mass: semi-Lagrangian advection carries values along the flow, not masses, and where burning gas swells to several times its volume in a step it made too little room for it, so a quarter to a third of the fire’s products and heat, and a tenth of its fuel, simply vanished. A pass before the advection measures how widely each cell’s gas was spread and scales it to the room it should fill (Lentine, Aanjaneya and Fedkiw’s mass-conserving semi-Lagrangian advection); MacCormack’s sharpening, which is not mass-neutral where gas expands, stands aside in those cells. The fire now releases 98 to 102% of the heat asked for.',
  ],
  [
    'Flames',
    'Nothing glows but soot. Hot soot emits what it absorbs times the blackbody radiance of its temperature (Kirchhoff’s law), so a flame’s colour and brightness follow from the simulated temperature and soot alone: Planck’s law through the CIE colour matching functions, in absolute units (a blackbody at 2042 K comes out at 600,000 cd/m², the old definition of the candle). The soot’s extinction is the measured 8.7 m²/g of flame smoke and its albedo 0.25, and it absorbs blue more than red (as 1/λ), so it reddens what shines through it. Below about 1000 K its glow is too faint to see, which is where the visible flame ends. A cell’s temperature is an average, and a blackbody’s glow climbs so steeply with temperature that the average glow is far from the glow at the average temperature (the turbulence-radiation interaction), so the renderer adds the temperature fluctuation finer than the grid: its size from the resolved temperature gradient, as large-eddy simulations model a subgrid variance (Pierce and Moin), only its shape from a noise that rises with the flames and churns (Flame detail scales it; 0 renders the cell averages). The fire lights its own smoke and the floor: its emitted light is summed on the GPU every step in eight slices of its height, each a point light at its brightness-weighted centre, and each point of the smoke gathers them through the soot in the way, so dense smoke shadows the fire’s light as it does the sun’s. (The floor takes the same eight lights unshadowed: the smoke rises away from the light between them.)',
  ],
  [
    'Detail',
    'Wisps finer than the grid, in the spirit of production upres: two layers of texture coordinates ride with the flow, and the renderer reads a fine noise at them, so the detail moves with the smoke rather than sliding through it. The flow stretches each layer as it shears it, so each resets on its own schedule, half a period after the other, and the two cross-fade: a layer is invisible at the moment it resets. A stretched layer would streak its noise along the stretch, so each layer also tracks how far the flow has stretched it (after Neyret 2003) and counts for less the more it has; where both are stretched, as in the fast updraft over a fire, the detail fades rather than streaks. The detail works hardest in thin smoke, at the edges and in the wisps, and leaves the faintest haze smooth, where fine noise would read as grain. It shapes what the camera sees; the light still travels through the simulated density.',
  ],
  [
    'Upres',
    'A smoke source’s density also runs on a grid twice as fine, in the manner of wavelet turbulence (Kim, Thürey, James and Gross): carried by the solver’s velocity plus turbulence too fine for the coarse grid, the curl of a noise, so it stays divergence free, read at the texture coordinates the detail layers carry with the flow (its eddies travel with the smoke and fade where the flow has stretched them), and as strong as the local flow times 2^(−5/6), the share of the next band of eddies in Kolmogorov’s spectrum. The camera sees this finer smoke; the lighting still comes from the coarse grid. It takes the place of the rendered detail for smoke, and the Solver tab turns it off. A fire keeps the rendered detail: its soot and temperature come from the combustion on the coarse grid.',
  ],
  ['Boundaries', 'A solid floor, and open sides and top: smoke leaves the box, and the air that flows in carries none. And the collision ball, when there is one (see Interaction).'],
  [
    'Interaction',
    'Drag to stir the gas, move the source along the floor, or move a solid ball through the flow; the arrow keys do the same. The ball is a solid in the solver: the pressure solve treats its cells as walls at every multigrid level (a cell is solid when its centre is inside it, so the coarse levels see it too), and the faces around it move with its velocity, so a moving ball pushes the gas and the gas flows around it. It casts soft shadows from the sun on the floor and the smoke, hides its share of the sky (its view factor from the floor, (r/d)² cos θ), and is lit by the fire’s shadowed light. Each starting point sets a source, its look and the wind, and starts again.',
  ],
  [
    'Rendering',
    'Each pixel marches through the volume and integrates each step in closed form, in photometric units (cd/m²), each colour channel on its own. Sunlight reaches every point through the optical depth toward the sun, recomputed each frame on a half-resolution grid; multiple-scattering octaves stand in for light bouncing inside the smoke; the sky lights it from above through the smoke overhead; the floor takes its shadow and the fire’s light. Between cells the smoke and soot are sampled with a cubic B-spline (eight trilinear fetches placed so the hardware’s blend gives the spline’s weights, after Sigg and Hadwiger), smooth where trilinear interpolation creases at every cell face; the Shading tab switches back to trilinear, which renders in about two thirds of the time. Each frame starts its march at a blue-noise offset stepped by the golden ratio, and frames accumulate: each blends into a running history, first clamped to the range of the new frame’s neighbourhood so what moves is not smeared behind it, which averages the sampling noise away. A paused frame keeps accumulating until it settles.',
  ],
  [
    'Camera',
    'Exposed like a real camera. Light is set as a light meter reads it, the sun and sky as incident readings (2.5 × 2^EV lux), and the exposure is the camera’s EV at ISO 100, saturating at 1.2 × 2^EV cd/m² (Frostbite’s physically based camera). The white balance is a Bradford chromatic adaptation from a blackbody of the chosen temperature to the display’s white, the step Nguyen and colleagues took for fire; then ACES filmic tone mapping through a fitted curve. Auto exposure meters the scene as a camera in program mode would: the sun and sky as an incident meter reads them, but never brighter than the fire allows (EV 7 for an 80 kW campfire, half a stop more for each doubling of its light), eased over a second so the flames’ puffing does not pump it; its brightest flames saturate to yellow-white, as they do in photographs. The background is a studio’s black, or the sky (its radiance the sky’s illuminance over π, a little brighter toward the horizon) over a floor that runs into the horizon’s haze.',
  ],
  [
    'Validation',
    'The fire was checked against fire-science measurements, on a uniform 0.6 m bed (the renderer plays no part). At this grid the solver resolves too little of the turbulence that mixes air into a plume (with none, the plume stays three to four times too hot), so the fire’s turbulence is a model: eddies of the fire’s own velocity, u² / l with u = 1.9 Q^(1/5) m/s (McCaffrey’s velocity scale), times one constant, calibrated on an 80 kW fire against McCaffrey’s measured centreline temperatures, as FDS calibrates its mixing constant against flame heights. With that one constant, the time-averaged centreline temperature of the 80 kW fire comes within about 20% of McCaffrey’s measurements over the intermittent flame and within about 30% over the plume above it. The size scaling is weaker than measured: the 40 kW fire runs 10 to 40% cooler than McCaffrey’s values, and the 150 kW fire up to 65% hotter in its intermittent flame (its flames reach the top of the 1.9 m domain). The flames come out taller than Heskestad’s correlation: the height below which 99% of the heat is released is 1.5 to 1.7 times his flame height, and the height where the flame (gas 500 K above the air) is present half the time about twice it. In tests, a constant strong enough to shorten them much left the plume far too cool, so the constant is set by the temperatures. The fire’s heat release puffs at 2.3 to 2.7 Hz, faster than the 1.9 Hz of Cetegen and Ahmed’s correlation for this bed. It releases 98 to 102% of the heat asked for (see Mixing).',
  ],
  [
    'Limits',
    'A real-time study rather than a production simulation: the grid is coarse (at most 200 × 80 × 40 cells over a box 4.8 m wide), velocity is stored at half precision, the pressure solve treats the gas as one density (FDS solves for a variable one), and a GPU that cannot keep up runs the simulation in slow motion. The chemistry is infinitely fast (no ignition, extinction or smouldering), soot forms at a fixed yield rather than forming and burning out within the flame, and the fire’s light reaches the floor unshadowed. Production pyro adds sparse and adaptive grids, far larger solves and collisions.',
  ],
]

// The solver's own frame of the default settings, one per frame shape (the frame is 4:3 on narrow screens).
function PyroStill({ alt }: { alt: string }) {
  return (
    <picture>
      <source media="(max-width: 768px)" srcSet="/pyro/poster-4x3.jpg" />
      <img className="lab-still-img" src="/pyro/poster.jpg" alt={alt} decoding="async" />
    </picture>
  )
}

// A labelled slider with its value.
function Slider({
  label,
  min,
  max,
  step,
  value,
  digits = 2,
  unit = '',
  onChange,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  digits?: number
  unit?: string
  onChange: (v: number) => void
}) {
  return (
    <Group label={label}>
      <Range min={min} max={max} step={step} value={value} label={unit ? `${label}, ${unit.trim()}` : label} onChange={onChange} />
      <span className="lab-readout">
        {value.toFixed(digits)}
        {unit}
      </span>
    </Group>
  )
}

export default function PyroLab() {
  const frameRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<PyroEngine | null>(null)
  const [pyro, setPyro] = useState<PyroState>(PYRO_DEFAULTS)
  const pyroRef = useRef(pyro)
  const [status, setStatus] = useState<PyroStatus | null>(null)
  // Mounted only on the client (the Lab switch shows it), so WebGPU can be checked here.
  const [mode, setMode] = useState<'starting' | 'live' | 'fallback'>(() => (PyroEngine.detect() ? 'starting' : 'fallback'))
  const [paused, setPaused] = useState(false)
  const [tab, setTab] = useState<PyroTab>('source')
  // What a drag on the image does: stir the gas, move the source along the floor, or move the collision ball.
  const [tool, setTool] = useState<'stir' | 'source' | 'ball'>('stir')
  const [renderer, setRenderer] = useState('')
  const tabsId = useId()
  const debugOn = useSyncExternalStore(noSubscribe, () => new URLSearchParams(location.search).has('debug'), () => false)
  const shown = mode === 'live' && !!status && status.simTime > 0

  useEffect(() => {
    pyroRef.current = pyro
  }, [pyro])
  // The engine's reports; when it steps down from the default resolution on a slow GPU, the Solver tab follows.
  const onStatus = (s: PyroStatus) => {
    setStatus(s)
    if (s.autoStepped) setPyro(p => (p.resolution === s.resolution ? p : { ...p, resolution: s.resolution }))
  }

  useEffect(() => {
    const canvas = canvasRef.current
    const frame = frameRef.current
    if (!canvas || !frame || !PyroEngine.detect()) return
    let engine: PyroEngine | null = null
    let unmounted = false
    let visible = false
    let losses = 0
    const fit = () => {
      const r = frame.getBoundingClientRect()
      engine?.resize(r.width, r.height, window.devicePixelRatio || 1)
    }
    const start = async () => {
      const e = await PyroEngine.create(canvas, onStatus, pyroRef.current, onLost)
      if (unmounted) {
        e?.dispose()
        return
      }
      if (!e) {
        setMode('fallback')
        return
      }
      engine = e
      engineRef.current = e
      if (process.env.NODE_ENV !== 'production') {
        ;(window as unknown as { __pyro?: PyroEngine }).__pyro = e
      }
      setRenderer(`WebGPU, ${e.adapterName}`)
      fit()
      e.setVisible(visible)
      const ok = await e.whenReady.catch(err => {
        console.error(err)
        return false
      })
      if (unmounted || engine !== e) return
      if (!ok) {
        e.dispose()
        engine = null
        engineRef.current = null
        setMode('fallback')
        return
      }
      setMode('live')
    }
    // A lost device is replaced, twice at most.
    const onLost = () => {
      engine?.dispose()
      engine = null
      engineRef.current = null
      if (++losses > 2) {
        setMode('fallback')
        return
      }
      setTimeout(() => !unmounted && start(), 1000)
    }
    const io = new IntersectionObserver(entries => {
      visible = entries.some(en => en.isIntersecting)
      engine?.setVisible(visible)
    })
    io.observe(frame)
    const ro = new ResizeObserver(fit)
    ro.observe(frame)
    start()
    return () => {
      unmounted = true
      io.disconnect()
      ro.disconnect()
      engine?.dispose()
      engineRef.current = null
    }
  }, [])

  // A new source brings the look it reads best with (see SOURCE_LOOKS).
  const applyPreset = (state: Partial<PyroState>) => {
    update(state)
    engineRef.current?.restart()
  }
  const pickTool = (t: 'stir' | 'source' | 'ball') => {
    setTool(t)
    if (t === 'ball' && !pyro.ball) update({ ball: true })
  }
  const pickSource = (t: SourceType) => {
    if (t !== pyro.sourceType) update({ sourceType: t, ...SOURCE_LOOKS[t] })
  }
  const update = (patch: Partial<PyroState>) => {
    setPyro(p => ({ ...p, ...patch }))
    if (patch.resolution && engineRef.current) engineRef.current.autoResolution = false
    engineRef.current?.setState(patch)
  }
  const togglePause = () => {
    setPaused(!paused)
    engineRef.current?.setPaused(!paused)
  }

  // Stirring: a mouse or pen stirs at once; a finger must first travel sideways (a vertical swipe scrolls).
  const drag = useRef<{ id: number; x: number; y: number; t: number; live: boolean } | null>(null)
  // Moving the source or the ball: to the point under the pointer.
  const moveTo = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - r.left) / r.width) * 2 - 1
    const y = 1 - ((e.clientY - r.top) / r.height) * 2
    const engine = engineRef.current
    if (!engine) return
    if (tool === 'ball') engine.ballTo(x, y)
    else {
      const sx = engine.sourceTo(x, y)
      if (sx !== null) setPyro(p => ({ ...p, srcX: sx }))
    }
  }
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return
    const moving = tool !== 'stir'
    const live = moving || e.pointerType !== 'touch'
    drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY, t: e.timeStamp, live }
    if (live) e.currentTarget.setPointerCapture(e.pointerId)
    if (moving) moveTo(e)
  }
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current
    if (!d || d.id !== e.pointerId) return
    if (!d.live) {
      const mx = Math.abs(e.clientX - d.x)
      const my = Math.abs(e.clientY - d.y)
      if (mx < TOUCH_SLOP && my < TOUCH_SLOP) return
      if (my > mx) {
        drag.current = null
        return
      }
      d.live = true
      e.currentTarget.setPointerCapture(e.pointerId)
    }
    if (tool !== 'stir') {
      moveTo(e)
      return
    }
    const r = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - r.left) / r.width) * 2 - 1
    const y = 1 - ((e.clientY - r.top) / r.height) * 2
    engineRef.current?.stirAt(x, y, ((e.clientX - d.x) / r.width) * 2, (-(e.clientY - d.y) / r.height) * 2, e.timeStamp - d.t)
    d.x = e.clientX
    d.y = e.clientY
    d.t = e.timeStamp
  }
  const endDrag = () => {
    drag.current = null
  }
  // The arrow keys push the smoke sideways or up from the middle of the frame.
  const onKeyDown = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    const push: Record<string, [number, number]> = { ArrowLeft: [-0.05, 0], ArrowRight: [0.05, 0], ArrowUp: [0, 0.05], ArrowDown: [0, -0.05] }
    const p = push[e.key]
    if (!p) return
    e.preventDefault()
    if (tool === 'ball') engineRef.current?.nudgeBall(p[0] * 2, p[1] * 2)
    else if (tool === 'source') {
      const sx = Math.max(-2, Math.min(2, pyro.srcX + p[0] * 2))
      update({ srcX: sx })
    } else engineRef.current?.stirAt(0, 0, p[0], p[1], 33)
  }

  const dims = status?.dims ?? RESOLUTIONS[pyro.resolution].dims
  const inputOn = mode === 'live'

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em', marginBottom: 12 }}>Fire and smoke, simulated live</h2>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, maxWidth: 640 }}>
          A 3D fire and smoke solver running on your GPU, built the way production FX tools are: combustion, buoyancy,
          vorticity, turbulence and a pressure solve every step, lit with physically based volume rendering.
        </p>
      </div>

      <p className="lab-howto">
        {tool === 'stir' ? (
          <>
            <span className="lab-howto-pointer">Drag across the image</span>
            <span className="lab-howto-touch">Swipe sideways on the image</span> to stir the {pyro.sourceType === 'fire' ? 'fire' : 'smoke'}
          </>
        ) : tool === 'source' ? (
          <>Drag along the floor to move the {pyro.sourceType === 'fire' ? 'fire' : 'smoke'}</>
        ) : (
          <>Drag the ball through the {pyro.sourceType === 'fire' ? 'flames' : 'smoke'}</>
        )}
        <span aria-hidden="true"> · </span>
        <span className="sr-only">. </span>Every control below is live
      </p>

      <div ref={frameRef} className="lab-frame">
        {mode === 'fallback' ? (
          <PyroStill alt="A fire burning on the floor, its flames yellow at the base and red at the tips, its smoke rising, rendered by the pyro lab's solver" />
        ) : (
          <canvas
            ref={canvasRef}
            tabIndex={0}
            role="application"
            aria-roledescription="fire and smoke simulation"
            aria-label={
              tool === 'stir'
                ? 'Live fire and smoke simulation. Drag, or use the arrow keys, to stir it.'
                : tool === 'source'
                  ? 'Live fire and smoke simulation. Drag, or use the left and right arrow keys, to move the source.'
                  : 'Live fire and smoke simulation. Drag, or use the arrow keys, to move the ball.'
            }
            className={`lab-canvas${tool === 'stir' ? '' : ' lab-canvas-move'}`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onKeyDown={onKeyDown}
          />
        )}
        {mode !== 'fallback' && (
          <div className={`lab-still${shown ? ' gone' : ''}`}>
            <PyroStill alt="" />
          </div>
        )}
        <div className="lab-hud lab-hud-left">
          {mode === 'fallback'
            ? 'Still frame. The live solver needs WebGPU, which this browser does not offer.'
            : mode === 'starting'
              ? 'Starting the solver'
              : `${pyro.sourceType === 'fire' ? 'Fire' : 'Smoke'}  ${dims.join(' × ')}`}
        </div>
        {mode === 'live' && status && (
          <div className="lab-hud lab-hud-right">
            {paused ? `Paused  ${status.simTime.toFixed(1)} s` : `${status.simTime.toFixed(1)} s  ${status.stepsPerSec} steps/s`}
          </div>
        )}
        {debugOn && (
          <div className="lab-hud lab-hud-bottom">
            {`Renderer: ${renderer || (mode === 'fallback' ? 'none (no WebGPU)' : 'starting')}`}
            {status && `\nRender scale ${status.renderScale.toFixed(2)}`}
          </div>
        )}
      </div>

      {mode !== 'fallback' && (
        <fieldset className="lab-controls" disabled={!inputOn}>
          <div className="lab-row">
            <Pill on={paused} onClick={togglePause}>
              {paused ? 'Resume' : 'Pause'}
            </Pill>
            <Pill on={false} onClick={() => engineRef.current?.restart()}>
              Restart
            </Pill>
            <Pill
              on={false}
              onClick={() => {
                setPyro(PYRO_DEFAULTS)
                engineRef.current?.setState(PYRO_DEFAULTS)
                setTool('stir')
              }}
            >
              Reset controls
            </Pill>
          </div>
          <div className="lab-row">
            <Group label="Drag to">
              <Pill on={tool === 'stir'} onClick={() => pickTool('stir')}>
                Stir
              </Pill>
              <Pill on={tool === 'source'} onClick={() => pickTool('source')}>
                {pyro.sourceType === 'fire' ? 'Move the fire' : 'Move the smoke'}
              </Pill>
              <Pill on={tool === 'ball'} onClick={() => pickTool('ball')}>
                Move the ball
              </Pill>
            </Group>
            {pyro.ball && (
              <Pill
                on={false}
                onClick={() => {
                  update({ ball: false })
                  if (tool === 'ball') setTool('stir')
                }}
              >
                Remove the ball
              </Pill>
            )}
          </div>
          <Tabs id={tabsId} label="Pyro controls" tabs={PYRO_TABS} tab={tab} onTab={setTab} />
          <div className="lab-panel" role="tabpanel" id={`${tabsId}-panel`} aria-labelledby={`${tabsId}-${tab}`}>
            {tab === 'source' && (
              <>
                <div className="lab-row">
                  <Group label="Start from">
                    {PRESETS.map(p => (
                      <Pill key={p.id} on={false} onClick={() => applyPreset(p.state)}>
                        {p.label}
                      </Pill>
                    ))}
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Type">
                    <Pill on={pyro.sourceType === 'smoke'} onClick={() => pickSource('smoke')}>
                      Smoke
                    </Pill>
                    <Pill on={pyro.sourceType === 'fire'} onClick={() => pickSource('fire')}>
                      Fire
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Slider label="Radius" min={0.08} max={0.4} step={0.01} value={pyro.srcRadius} unit=" m" onChange={v => update({ srcRadius: v })} />
                  {pyro.sourceType === 'fire' ? (
                    <Slider label="Heat release" min={10} max={400} step={5} value={pyro.heatRelease} digits={0} unit=" kW" onChange={v => update({ heatRelease: v })} />
                  ) : (
                    <Slider label="Density" min={0.2} max={2} step={0.05} value={pyro.srcDensity} onChange={v => update({ srcDensity: v })} />
                  )}
                </div>
                {pyro.sourceType === 'smoke' && (
                  <div className="lab-row">
                    <Slider label="Temperature" min={10} max={300} step={5} value={pyro.srcTemp} digits={0} unit=" K" onChange={v => update({ srcTemp: v })} />
                    <Slider label="Speed" min={0} max={2} step={0.05} value={pyro.srcSpeed} unit=" m/s" onChange={v => update({ srcSpeed: v })} />
                  </div>
                )}
                <div className="lab-row">
                  <Slider label="Noise" min={0} max={1} step={0.05} value={pyro.srcNoise} onChange={v => update({ srcNoise: v })} />
                  <Slider label="Position" min={-2} max={2} step={0.05} value={pyro.srcX} unit=" m" onChange={v => update({ srcX: v })} />
                </div>
                {pyro.ball && (
                  <div className="lab-row">
                    <Slider label="Ball radius" min={0.1} max={0.4} step={0.01} value={pyro.ballRadius} unit=" m" onChange={v => update({ ballRadius: v })} />
                  </div>
                )}
              </>
            )}
            {tab === 'forces' && (
              <>
                <div className="lab-row">
                  <Slider label="Buoyancy" min={0} max={2} step={0.05} value={pyro.buoyancy} onChange={v => update({ buoyancy: v })} />
                  <Slider label="Vorticity" min={0} max={1} step={0.01} value={pyro.vorticity} onChange={v => update({ vorticity: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Turbulence" min={0} max={3} step={0.05} value={pyro.turbulence} onChange={v => update({ turbulence: v })} />
                  <Slider label="Turbulence scale" min={0.1} max={1} step={0.01} value={pyro.turbScale} unit=" m" onChange={v => update({ turbScale: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Wind" min={-2} max={2} step={0.05} value={pyro.wind} unit=" m/s" onChange={v => update({ wind: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Cooling" min={0} max={2} step={0.05} value={pyro.cooling} unit=" /s" onChange={v => update({ cooling: v })} />
                  <Slider label="Dissipation" min={0} max={0.5} step={0.01} value={pyro.dissipation} unit=" /s" onChange={v => update({ dissipation: v })} />
                </div>
                <p className="lab-note">
                  {pyro.sourceType === 'fire'
                    ? 'Buoyancy 1 is physical, Turbulence 1 the calibrated turbulence model, Cooling and Dissipation 0 physical.'
                    : 'Turbulence in m/s². Buoyancy 1 is physical.'}
                </p>
              </>
            )}
            {tab === 'fire' && (
              <>
                <div className="lab-row">
                  <Slider label="Soot yield" min={0} max={0.2} step={0.001} value={pyro.sootYield} digits={3} unit=" kg/kg" onChange={v => update({ sootYield: v })} />
                  <Slider label="Radiant fraction" min={0.1} max={0.5} step={0.01} value={pyro.radiantFraction} onChange={v => update({ radiantFraction: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Expansion" min={0} max={1.5} step={0.05} value={pyro.expansion} onChange={v => update({ expansion: v })} />
                  <Slider label="Flame detail" min={0} max={2} step={0.05} value={pyro.flameDetail} onChange={v => update({ flameDetail: v })} />
                </div>
                <p className="lab-note">
                  {pyro.sourceType === 'fire'
                    ? 'Wood burns with a soot yield of 0.015 and radiates 0.3 of its heat. Expansion and buoyancy 1 are physical. Flame detail 0 shows the flame at the grid\u2019s resolution.'
                    : 'These act on a fire source: choose Fire under Source.'}
                </p>
              </>
            )}
            {tab === 'shading' && (
              <>
                <div className="lab-row">
                  {pyro.sourceType === 'smoke' && (
                    <Slider label="Smoke density" min={2} max={60} step={1} value={pyro.densityScale} digits={0} unit=" /m" onChange={v => update({ densityScale: v })} />
                  )}
                  <Slider label="Albedo" min={0.1} max={1} step={0.01} value={pyro.albedo} onChange={v => update({ albedo: v })} />
                  <Slider label="Anisotropy" min={-0.5} max={0.9} step={0.01} value={pyro.anisotropy} onChange={v => update({ anisotropy: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Light azimuth" min={-180} max={180} step={1} value={pyro.lightAz} digits={0} unit={'°'} onChange={v => update({ lightAz: v })} />
                  <Slider label="Light elevation" min={5} max={85} step={1} value={pyro.lightEl} digits={0} unit={'°'} onChange={v => update({ lightEl: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Sun" min={-4} max={16} step={0.5} value={pyro.sunEV} digits={1} unit=" EV" onChange={v => update({ sunEV: v })} />
                  <Slider label="Sky" min={-4} max={14} step={0.5} value={pyro.skyEV} digits={1} unit=" EV" onChange={v => update({ skyEV: v })} />
                </div>
                <div className="lab-row">
                  <Group label="Exposure">
                    <Pill on={pyro.exposureMode === 'auto'} onClick={() => update({ exposureMode: 'auto' })}>
                      Auto
                    </Pill>
                    <Pill on={pyro.exposureMode === 'manual'} onClick={() => update({ exposureMode: 'manual', exposure: Math.round((status?.exposureEV ?? pyro.exposure) * 4) / 4 })}>
                      Manual
                    </Pill>
                  </Group>
                  {pyro.exposureMode === 'auto' ? (
                    <Slider label="Compensation" min={-3} max={3} step={0.25} value={pyro.exposureComp} digits={2} unit=" EV" onChange={v => update({ exposureComp: v })} />
                  ) : (
                    <Slider label="Exposure" min={-2} max={16} step={0.25} value={pyro.exposure} digits={2} unit=" EV" onChange={v => update({ exposure: v })} />
                  )}
                  <Slider label="White balance" min={2000} max={10000} step={100} value={pyro.whiteBalance} digits={0} unit=" K" onChange={v => update({ whiteBalance: v })} />
                </div>
                <div className="lab-row">
                  <Group label="Background">
                    <Pill on={pyro.background === 'studio'} onClick={() => update({ background: 'studio' })}>
                      Studio
                    </Pill>
                    <Pill on={pyro.background === 'sky'} onClick={() => update({ background: 'sky' })}>
                      Sky
                    </Pill>
                  </Group>
                  <Group label="Sampling">
                    <Pill on={!pyro.tricubic} onClick={() => update({ tricubic: false })}>
                      Trilinear
                    </Pill>
                    <Pill on={pyro.tricubic} onClick={() => update({ tricubic: true })}>
                      Tricubic
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Slider label="Detail" min={0} max={1.5} step={0.05} value={pyro.detail} onChange={v => update({ detail: v })} />
                  <Slider label="Detail scale" min={0.02} max={0.2} step={0.005} value={pyro.detailScale} digits={3} unit=" m" onChange={v => update({ detailScale: v })} />
                </div>
                <p className="lab-note">
                  Light as a meter reads it: the sun and sky are incident readings (2.5 × 2^EV lux), the exposure is the camera’s EV at ISO 100.
                  Pause the simulation to relight a frozen frame: shading changes redraw at once.
                </p>
              </>
            )}
            {tab === 'solver' && (
              <>
                <div className="lab-row">
                  <Group label="Resolution">
                    {RESOLUTION_ORDER.map(r => (
                      <Pill key={r} on={pyro.resolution === r} onClick={() => update({ resolution: r })}>
                        {`${RESOLUTIONS[r].label} ${RESOLUTIONS[r].dims.join('×')}`}
                      </Pill>
                    ))}
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Velocity">
                    <Pill on={pyro.velocity === 'mac'} onClick={() => update({ velocity: 'mac' })}>
                      Modified MacCormack
                    </Pill>
                    <Pill on={pyro.velocity === 'ar'} onClick={() => update({ velocity: 'ar' })}>
                      Advection-reflection
                    </Pill>
                    <Pill on={pyro.velocity === 'sl'} onClick={() => update({ velocity: 'sl' })}>
                      Semi-Lagrangian
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Scalars">
                    <Pill on={pyro.advection === 'mac'} onClick={() => update({ advection: 'mac' })}>
                      Modified MacCormack
                    </Pill>
                    <Pill on={pyro.advection === 'sl'} onClick={() => update({ advection: 'sl' })}>
                      Semi-Lagrangian
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Smoke upres">
                    <Pill on={!pyro.upres} onClick={() => update({ upres: false })}>
                      Off
                    </Pill>
                    <Pill on={pyro.upres} onClick={() => update({ upres: true })}>
                      Wavelet turbulence, 2×
                    </Pill>
                  </Group>
                  {pyro.upres && (
                    <Slider label="Upres turbulence" min={0} max={2} step={0.05} value={pyro.upresStrength} onChange={v => update({ upresStrength: v })} />
                  )}
                </div>
                <div className="lab-row">
                  <Group label="Pressure">
                    <Pill on={pyro.pressure === 'mg'} onClick={() => update({ pressure: 'mg' })}>
                      Multigrid
                    </Pill>
                    <Pill on={pyro.pressure === 'sor'} onClick={() => update({ pressure: 'sor' })}>
                      Red-black SOR
                    </Pill>
                  </Group>
                  {pyro.pressure === 'mg' ? (
                    <Slider label="V-cycles" min={1} max={4} step={1} value={pyro.vcycles} digits={0} onChange={v => update({ vcycles: v })} />
                  ) : (
                    <Slider label="Iterations" min={5} max={120} step={5} value={pyro.pressureIters} digits={0} onChange={v => update({ pressureIters: v })} />
                  )}
                </div>
                <p className="lab-note">A new resolution restarts the simulation.</p>
              </>
            )}
          </div>
        </fieldset>
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
            {PYRO_CITATION_GROUPS.map(g => (
              <div key={g} className="lab-refs-group">
                <h3>{g}</h3>
                <ol>
                  {PYRO_CITATIONS.filter(c => c.group === g).map(c => (
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
    </div>
  )
}
