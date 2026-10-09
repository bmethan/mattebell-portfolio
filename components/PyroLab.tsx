'use client'
// The lab's pyro mode: a 3D smoke solver and volume renderer on the GPU (see pyro/engine.ts), stirred by dragging
// across the image, with the solver's controls below. It mounts when the Lab switch first selects it, and starts
// at once (choosing it is the visitor's go-ahead).
import { useEffect, useId, useRef, useState, useSyncExternalStore } from 'react'
import { PyroEngine, type PyroStatus } from './pyro/engine'
import { PYRO_DEFAULTS, RESOLUTIONS, RESOLUTION_ORDER, SOURCE_LOOKS, type PyroState, type SourceType } from './pyro/params'
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
    'Air and smoke on a staggered grid, the layout production solvers use: velocity on the cell faces, density and temperature at the cell centres. Each step, 30 per simulated second, advects the velocity (semi-Lagrangian, with a second-order backtrace), adds the forces, then makes the flow incompressible with a pressure solve and projection.',
  ],
  [
    'Pressure',
    'Geometric multigrid, as production solvers use: each V-cycle smooths with red-black Gauss-Seidel on the full grid, hands the remaining error down through coarser grids (each half the resolution of the last), where the broad, slow-to-fix part of it is cheap to remove, and brings the corrections back up. Two V-cycles a step, started from the previous step’s pressure, leave about 1% of the flow’s divergence, the floor set by storing velocity at half precision (measured on this solver). The Solver tab also offers plain red-black SOR for comparison: 40 sweeps reach the same floor at four to eight times the GPU time, and a shorter solve shows as smoke that swells and loses some of its swirl.',
  ],
  [
    'Advection',
    'Density and temperature move with Modified MacCormack: a forward and a backward semi-Lagrangian step, half the round-trip error added back, then clamped to the values the forward step read, so detail survives without new extremes. Semi-Lagrangian alone is the softer, more diffusive choice.',
  ],
  [
    'Forces',
    'Buoyancy from temperature; vorticity confinement, which restores the small swirls a coarse grid smooths away; divergence-free turbulence from the curl of a noise potential, only where there is smoke; wind; and the source’s upward push. Hot air cools and smoke dissipates at the rates in the Forces tab.',
  ],
  [
    'Combustion',
    'A fire source emits fuel instead of smoke, tracked as a reaction coordinate: the share of the reaction still to come, 1 as the gas leaves the source, running down at the burn rate while the gas is hotter than its ignition point. As it burns it releases heat, which drives the buoyancy, and soot, mostly toward the end of the reaction, where a flame turns to smoke. Burning gas expands: where there is flame, the pressure solve makes the flow diverge by the expansion rate instead of not at all, which gives flames their outward push and flicker.',
  ],
  [
    'Flames',
    'Flames emit light at the blackbody colour and brightness of their temperature: Planck\u2019s law integrated against the CIE colour matching functions. The radiating temperature follows the reaction, at the flame\u2019s peak (1900 K by default) where it is fresh at the base, cooling toward the tips as it runs out, which gives the yellow-to-orange-to-red of a real flame. Real flames span a huge range of brightness, a 2000 K core about 60 times brighter than a 1500 K edge; the contrast control compresses it, 1 being physical. The fire also lights the smoke and the floor, as a point light at its brightness-weighted centre carrying its total emitted light, summed on the GPU every step; the smoke does not shadow that light.',
  ],
  [
    'Detail',
    'Wisps finer than the grid, in the spirit of production upres: two layers of texture coordinates ride with the flow, and the renderer reads a fine noise at them, so the detail moves with the smoke rather than sliding through it. The flow stretches each layer as it shears it, so each resets on its own schedule, half a period after the other, and the two cross-fade: a layer is invisible at the moment it resets. The detail works hardest in thin smoke, at the edges and in the wisps. It shapes what the camera sees; the light still travels through the simulated density.',
  ],
  ['Boundaries', 'A solid floor, and open sides and top: smoke leaves the box, and the air that flows in carries none.'],
  [
    'Rendering',
    'Each pixel marches through the volume and integrates the light over each step in closed form. Sunlight reaches every point through the smoke’s optical depth toward the sun, recomputed each frame on a half-resolution grid. Multiple-scattering octaves stand in for light bouncing inside the smoke, the sky lights it from above through the smoke overhead, and the floor takes its shadow.',
  ],
  [
    'Display',
    'ACES filmic tone mapping through a fitted curve (the lookdev lab applies the full ACES 2.0 transform; the fit keeps the per-pixel cost small).',
  ],
  [
    'Limits',
    'A real-time study rather than a production simulation: the grid is coarse (at most 200 × 80 × 40 cells over a box 4.8 m wide), velocity is stored at half precision, and a GPU that cannot keep up runs the simulation in slow motion rather than taking longer, less stable steps. Production pyro adds sparse and adaptive grids, far larger multigrid solves, collisions and combustion.',
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
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return
    const live = e.pointerType !== 'touch'
    drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY, t: e.timeStamp, live }
    if (live) e.currentTarget.setPointerCapture(e.pointerId)
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
    engineRef.current?.stirAt(0, 0, p[0], p[1], 33)
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
        <span className="lab-howto-pointer">Drag across the image</span>
        <span className="lab-howto-touch">Swipe sideways on the image</span> to stir the fire
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
            aria-label="Live fire and smoke simulation. Drag, or use the arrow keys, to stir it."
            className="lab-canvas"
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
              }}
            >
              Reset controls
            </Pill>
          </div>
          <Tabs id={tabsId} label="Pyro controls" tabs={PYRO_TABS} tab={tab} onTab={setTab} />
          <div className="lab-panel" role="tabpanel" id={`${tabsId}-panel`} aria-labelledby={`${tabsId}-${tab}`}>
            {tab === 'source' && (
              <>
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
                    <Slider label="Fuel" min={0.2} max={2} step={0.05} value={pyro.srcFuel} onChange={v => update({ srcFuel: v })} />
                  ) : (
                    <Slider label="Density" min={0.2} max={2} step={0.05} value={pyro.srcDensity} onChange={v => update({ srcDensity: v })} />
                  )}
                </div>
                <div className="lab-row">
                  <Slider label="Temperature" min={0.2} max={2} step={0.05} value={pyro.srcTemp} onChange={v => update({ srcTemp: v })} />
                  <Slider label="Speed" min={0} max={2} step={0.05} value={pyro.srcSpeed} unit=" m/s" onChange={v => update({ srcSpeed: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Noise" min={0} max={1} step={0.05} value={pyro.srcNoise} onChange={v => update({ srcNoise: v })} />
                </div>
              </>
            )}
            {tab === 'forces' && (
              <>
                <div className="lab-row">
                  <Slider label="Buoyancy" min={0} max={5} step={0.1} value={pyro.buoyancy} digits={1} onChange={v => update({ buoyancy: v })} />
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
              </>
            )}
            {tab === 'fire' && (
              <>
                <div className="lab-row">
                  <Slider label="Burn rate" min={0.2} max={3} step={0.05} value={pyro.burnRate} unit=" /s" onChange={v => update({ burnRate: v })} />
                  <Slider label="Heat" min={0} max={4} step={0.05} value={pyro.heat} onChange={v => update({ heat: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Expansion" min={0} max={4} step={0.05} value={pyro.expansion} unit=" /s" onChange={v => update({ expansion: v })} />
                  <Slider label="Soot" min={0} max={1.5} step={0.05} value={pyro.soot} onChange={v => update({ soot: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Flame intensity" min={0} max={4} step={0.05} value={pyro.flameIntensity} onChange={v => update({ flameIntensity: v })} />
                  <Slider label="Flame temperature" min={1200} max={2600} step={10} value={pyro.flameKelvin} digits={0} unit=" K" onChange={v => update({ flameKelvin: v })} />
                  <Slider label="Flame contrast" min={0.2} max={1} step={0.05} value={pyro.flameContrast} onChange={v => update({ flameContrast: v })} />
                </div>
                <p className="lab-note">
                  {pyro.sourceType === 'fire'
                    ? 'Flame temperature: the gas at its hottest, where burning a full measure of fuel takes it. Flame contrast 1 is physical blackbody brightness.'
                    : 'These act on a fire source: choose Fire under Source.'}
                </p>
              </>
            )}
            {tab === 'shading' && (
              <>
                <div className="lab-row">
                  <Slider label="Smoke density" min={2} max={60} step={1} value={pyro.densityScale} digits={0} unit=" /m" onChange={v => update({ densityScale: v })} />
                  <Slider label="Albedo" min={0.1} max={1} step={0.01} value={pyro.albedo} onChange={v => update({ albedo: v })} />
                  <Slider label="Anisotropy" min={-0.5} max={0.9} step={0.01} value={pyro.anisotropy} onChange={v => update({ anisotropy: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Light azimuth" min={-180} max={180} step={1} value={pyro.lightAz} digits={0} unit={'°'} onChange={v => update({ lightAz: v })} />
                  <Slider label="Light elevation" min={5} max={85} step={1} value={pyro.lightEl} digits={0} unit={'°'} onChange={v => update({ lightEl: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Sun" min={0} max={8} step={0.1} value={pyro.sunIntensity} digits={1} onChange={v => update({ sunIntensity: v })} />
                  <Slider label="Sky" min={0} max={2} step={0.05} value={pyro.skyIntensity} onChange={v => update({ skyIntensity: v })} />
                  <Slider label="Exposure" min={-3} max={3} step={0.1} value={pyro.exposure} digits={1} unit=" EV" onChange={v => update({ exposure: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Detail" min={0} max={1.5} step={0.05} value={pyro.detail} onChange={v => update({ detail: v })} />
                  <Slider label="Detail scale" min={0.02} max={0.2} step={0.005} value={pyro.detailScale} digits={3} unit=" m" onChange={v => update({ detailScale: v })} />
                </div>
                <p className="lab-note">Pause the simulation to relight a frozen frame: shading changes redraw at once.</p>
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
                  <Group label="Advection">
                    <Pill on={pyro.advection === 'mac'} onClick={() => update({ advection: 'mac' })}>
                      Modified MacCormack
                    </Pill>
                    <Pill on={pyro.advection === 'sl'} onClick={() => update({ advection: 'sl' })}>
                      Semi-Lagrangian
                    </Pill>
                  </Group>
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
