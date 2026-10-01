'use client'
import { useEffect, useId, useRef, useState } from 'react'
import {
  LookdevEngine,
  DEFAULT_STATE,
  isPosterState,
  TARGET_SPP,
  MESH_FURNACE_BOUNCES,
  type LabState,
  type LabStatus,
  type Pass,
  type View,
} from './lookdev/engine'
import { HERO_PRESETS, HERO_ORDER, PAINT_FINISHES, PAINT_ORDER, STAGES, STAGE_ORDER, heroParams, paintHasFlakes } from './lookdev/materials'
import { MODELS, MODEL_ORDER } from './lookdev/models'
import { CITATIONS } from './lookdev/citations'
import WorkIndicator from './WorkIndicator'

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

const TOUCH_SLOP = 8 // px a finger must travel sideways before it drags the light
const POSTER_SPP = 2048 // samples per pixel of poster.jpg, the converged render of the default settings

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
}: {
  min: number
  max: number
  step: number
  value: number
  label: string
  onChange: (v: number) => void
}) {
  return (
    <input
      type="range"
      className="lab-range"
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
  const engineRef = useRef<LookdevEngine | null>(null)
  const labRef = useRef<LabState>(DEFAULT_STATE)
  const dragging = useRef<number | null>(null) // pointerId of the drag moving the light
  const touchStart = useRef<{ id: number; x: number; y: number } | null>(null)

  const [lab, setLab] = useState<LabState>(DEFAULT_STATE)
  const [status, setStatus] = useState<LabStatus>({ spp: 0, target: TARGET_SPP, converged: false, preview: false, ms: 0, model: 'ready' })
  const [mode, setMode] = useState<'idle' | 'live' | 'fallback'>('idle')
  const [labels, setLabels] = useState<{ x: number; y: number }[]>([])
  const [furnaceMean, setFurnaceMean] = useState<number | null>(null)
  const [paused, setPaused] = useState(false)
  const [ready, setReady] = useState(false)
  // The still is the renderer's own converged image of the default settings, so the live render (IPR) only
  // runs once something differs: the first edit (a click or drag on the frame, an arrow key, any control)
  // starts it, and putting every setting back shows the still again with the GPU idle. Hover and scrolling
  // past never cost anything.
  const pristine = isPosterState(lab)
  const [liveShown, setLiveShown] = useState(false) // the live render has drawn a frame since it (re)started

  const update = (patch: Partial<LabState>) => {
    const next = { ...labRef.current, ...patch }
    labRef.current = next
    setLab(next)
    // Anything beyond exposure or view transform re-renders from zero samples, so the previous count and
    // furnace reading no longer apply.
    if (Object.keys(patch).some(k => k !== 'exposure' && k !== 'view')) {
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
    let engine: LookdevEngine | null = null

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
        const m = engineRef.current.readSphereMean()
        setFurnaceMean(m ? (m[0] + m[1] + m[2]) / 3 : null)
      }
    }

    let visible = false
    let started = false
    let io: IntersectionObserver | null = null

    const fallback = () => {
      engine?.dispose()
      engine = null
      engineRef.current = null
      setMode('fallback')
      io?.disconnect()
    }

    const create = () => {
      try {
        const e = new LookdevEngine(canvas, handleStatus, labRef.current, onContextLost)
        engine = e
        engineRef.current = e
        if (process.env.NODE_ENV !== 'production') {
          ;(window as unknown as { __lookdev?: LookdevEngine }).__lookdev = e
        }
        setMode('live')
        layout()
        e.setVisible(visible)
        e.setActive(!isPosterState(labRef.current))
        e.whenReady.then(
          ok => {
            if (ok && engine === e) setReady(true)
          },
          err => {
            console.error(err)
            if (engine === e) fallback()
          },
        )
      } catch (err) {
        console.error(err)
        fallback()
      }
    }

    const start = () => {
      if (started) return
      started = true
      if (LookdevEngine.detect()) create()
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
    canvas.addEventListener('webglcontextrestored', onContextRestored)

    let cancelIdle: () => void
    if ('requestIdleCallback' in window) {
      const id = requestIdleCallback(start, { timeout: 4000 })
      cancelIdle = () => cancelIdleCallback(id)
    } else {
      const id = setTimeout(start, 2000)
      cancelIdle = () => clearTimeout(id)
    }

    // Approaching the section starts the renderer early if the idle callback has not yet; only a frame that is
    // actually on screen renders.
    io = new IntersectionObserver(entries => entries.some(en => en.isIntersecting) && start(), {
      rootMargin: '200px 0px',
    })
    io.observe(section)
    const onScreen = new IntersectionObserver(entries => {
      visible = entries.some(en => en.isIntersecting)
      engine?.setVisible(visible)
    })
    onScreen.observe(frame)
    const ro = new ResizeObserver(layout)
    ro.observe(frame)
    return () => {
      cancelIdle()
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
  }, [])

  // Labels follow the plate: a model moves the reference balls aside, and its own label tracks the turntable.
  useEffect(() => {
    const e = engineRef.current
    if (e) setLabels(e.labelPoints().map(p => e.project(p)))
  }, [lab.model, lab.modelYaw, status.model, ready])

  const setFromPointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const u = clamp((e.clientX - r.left) / r.width, 0, 1)
    const v = clamp((e.clientY - r.top) / r.height, 0, 1)
    update({ keyAz: (u - 0.5) * AZ_RANGE, keyEl: clamp(EL_MAX - v * (EL_MAX - EL_MIN) * 1.1, EL_MIN, EL_MAX) })
  }

  const beginDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    dragging.current = e.pointerId
    e.currentTarget.setPointerCapture(e.pointerId)
    engineRef.current?.setInteracting(true)
    setFromPointer(e)
  }
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (lab.furnace || !ready || dragging.current !== null) return
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
    if (lab.furnace || !ready) return
    const s = 0.08
    let { keyAz, keyEl } = labRef.current // held keys repeat faster than React re-renders
    if (e.key === 'ArrowLeft') keyAz -= s
    else if (e.key === 'ArrowRight') keyAz += s
    else if (e.key === 'ArrowUp') keyEl += s
    else if (e.key === 'ArrowDown') keyEl -= s
    else return
    e.preventDefault()
    update({ keyAz: clamp(keyAz, -AZ_RANGE / 2, AZ_RANGE / 2), keyEl: clamp(keyEl, EL_MIN, EL_MAX) })
  }

  const hero = HERO_PRESETS[lab.hero]
  const heroBase = heroParams(lab.hero, lab.paint, lab.flakes)
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
  const ballNames = ['18% gray', 'Chrome', model ? `${model.label}, ${heroName.toLowerCase()}` : heroName]

  return (
    <section ref={sectionRef} id="lab" className="section-pad" style={{ borderBottom: '0.5px solid var(--border)' }}>
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
          Lookdev lab
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
        </div>
        <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em', marginBottom: 12 }}>
          Light, rendered live
        </h2>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, maxWidth: 640 }}>
          A progressive path tracer running a subset of OpenPBR Surface on the reference balls every lighting department
          shoots: an 18% gray ball, a chrome ball, and a hero material. Drag the frame to move the key light.
        </p>
      </div>

      <div ref={frameRef} className="lab-frame">
        {mode === 'fallback' ? (
          <Still alt="Path traced lookdev reference balls: an 18% gray card ball, a chromium ball and a clear coated car paint ball under a three point softbox rig" />
        ) : (
          <canvas
            ref={canvasRef}
            tabIndex={0}
            role="application"
            aria-roledescription="key light control"
            aria-label="Path traced render of three lookdev reference balls. Drag, or use the arrow keys, to move the key light."
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
                Still frame. The live renderer is compiling its shaders.
              </>
            )}
          </div>
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
              {`Key light at ${deg(lab.keyAz)} degrees azimuth, ${deg(lab.keyEl)} degrees elevation`}
            </div>
            <div className="lab-hud lab-hud-left">
              {lab.furnace ? 'White furnace' : `Key ${deg(lab.keyAz)}° az  ${deg(lab.keyEl)}° el  ${lab.keyKelvin}K`}
            </div>
            <div className="lab-hud lab-hud-right">
              {status.model === 'failed' ? (
                model ? `The ${model.label.toLowerCase()} could not load` : 'The shaders for this scene failed to compile'
              ) : status.model === 'loading' ? (
                <>
                  <WorkIndicator className="lab-work" />
                  {status.waitingFor === 'model' && model ? `Loading the ${model.label.toLowerCase()}` : 'Compiling shaders'}
                </>
              ) : (
                <>
                  {!pristine && !status.converged && <WorkIndicator className="lab-work" />}
                  {PASSES.find(p => p.id === lab.pass)?.label}
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
              <div className="lab-marker" style={{ left: `${markerX}%`, top: `${clamp(markerY, 0, 100)}%` }} />
            )}
          </>
        )}
        {mode === 'fallback' && (
          <div className="lab-hud lab-hud-left">Still frame. Your browser lacks the WebGL2 features the live renderer needs.</div>
        )}
      </div>

      {mode !== 'fallback' && (
        <fieldset className="lab-controls" disabled={!ready}>
          <div className="lab-row">
            <Group label="Model">
              {MODEL_ORDER.map(m => (
                <Pill key={m} on={lab.model === m} onClick={() => update({ model: m })}>
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
            <Group label="Lights">
              <Pill on={lab.key} onClick={() => update({ key: !lab.key })}>Key</Pill>
              <Pill on={lab.fill} onClick={() => update({ fill: !lab.fill })}>Fill</Pill>
              <Pill on={lab.rim} onClick={() => update({ rim: !lab.rim })}>Rim</Pill>
            </Group>
            <Group label="Hero">
              {HERO_ORDER.map(h => (
                <Pill key={h} on={lab.hero === h} onClick={() => update({ hero: h, heroRoughness: null, heroAniso: null })}>
                  {HERO_PRESETS[h].label}
                </Pill>
              ))}
            </Group>
          </div>

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

          <div className="lab-row">
            <Group label="Key temp">
              <Range
                min={2500}
                max={9000}
                step={100}
                value={lab.keyKelvin}
                label="Key temp, kelvin"
                onChange={v => update({ keyKelvin: v })}
              />
              <span className="lab-readout">{lab.keyKelvin}K</span>
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

          <div className="lab-row">
            <Group label="Pass">
              {PASSES.map(p => (
                <Pill key={p.id} on={lab.pass === p.id} onClick={() => update({ pass: p.id })}>
                  {p.label}
                </Pill>
              ))}
            </Group>
            <Group label="View">
              {VIEWS.map(v => (
                <Pill key={v.id} on={lab.view === v.id} onClick={() => update({ view: v.id })}>
                  {v.label}
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
              </span>
            )}
          </div>
        </fieldset>
      )}

      <div className="lab-methods">
        {model && (
          <p className="lab-note" style={{ width: '100%', margin: '0 0 8px' }}>
            {model.credit.text}{' '}
            <a href={model.credit.source} target="_blank" rel="noopener noreferrer">
              Source
            </a>
            {', '}
            <a href={model.credit.licenseUrl} target="_blank" rel="noopener noreferrer">
              {model.credit.license}
            </a>
            .
          </p>
        )}
        <p className="lab-note" style={{ width: '100%', margin: '0 0 4px' }}>
          Rendered in ACEScg. With multiple-scattering compensation on, every OpenPBR preset averages within 0.4% of
          1.0 in a white furnace test, and rough glass, compensated for multiple scattering, stays within about 1.5%
          through a solid ball. Paint flakes are a lab extension, not part of OpenPBR, and lose about 2%. Dispersion is
          spectral: each light path through dispersive glass is traced at one wavelength between 380 and 780 nm.
        </p>
        <span className="lab-label">Methods</span>
        <ul>
          {CITATIONS.map(c => (
            <li key={c.id}>
              <a href={c.url} target="_blank" rel="noopener noreferrer" title={`${c.authors}. ${c.title}. ${c.venue}, ${c.year}.`}>
                {c.short}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
