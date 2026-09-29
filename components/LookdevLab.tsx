'use client'
import { useEffect, useId, useRef, useState } from 'react'
import Image from 'next/image'
import {
  LookdevEngine,
  DEFAULT_STATE,
  BALL_X,
  TARGET_SPP,
  type LabState,
  type LabStatus,
  type Pass,
  type View,
} from './lookdev/engine'
import { HERO_PRESETS, HERO_ORDER } from './lookdev/materials'
import { CITATIONS } from './lookdev/citations'

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

const AZ_RANGE = 5.2
const EL_MIN = 0.05
const EL_MAX = 1.35
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v))
const deg = (r: number) => Math.round((r * 180) / Math.PI)

function Pill({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" className={`lab-pill${on ? ' on' : ''}`} aria-pressed={on} onClick={onClick}>
      {children}
    </button>
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
  const dragging = useRef(false)

  const [lab, setLab] = useState<LabState>(DEFAULT_STATE)
  const [status, setStatus] = useState<LabStatus>({ spp: 0, target: TARGET_SPP, converged: false })
  const [mode, setMode] = useState<'idle' | 'live' | 'fallback'>('idle')
  const [labels, setLabels] = useState<{ x: number; y: number }[]>([])
  const [furnaceMean, setFurnaceMean] = useState<number | null>(null)
  const [paused, setPaused] = useState(false)

  useEffect(() => {
    labRef.current = lab
  }, [lab])

  const update = (patch: Partial<LabState>) => {
    setLab(prev => ({ ...prev, ...patch }))
    // Anything beyond exposure or view transform re-renders, so a previous furnace reading no longer applies.
    if (Object.keys(patch).some(k => k !== 'exposure' && k !== 'view')) setFurnaceMean(null)
    engineRef.current?.setState(patch)
  }

  // Create the renderer only when the section approaches the viewport; pause it when offscreen.
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
      setLabels(BALL_X.map(x => e.project([x, 0, 0])))
    }

    // When the white furnace test converges, measure the mean radiance over the spheres.
    const handleStatus = (s: LabStatus) => {
      setStatus(s)
      if (s.converged && labRef.current.furnace && engineRef.current) {
        const m = engineRef.current.readSphereMean()
        setFurnaceMean(m ? (m[0] + m[1] + m[2]) / 3 : null)
      }
    }

    let visible = false
    let contextLost = false
    let io: IntersectionObserver | null = null

    const create = () => {
      try {
        engine = new LookdevEngine(canvas, handleStatus, labRef.current, onContextLost)
        engineRef.current = engine
        if (process.env.NODE_ENV !== 'production') {
          ;(window as unknown as { __lookdev?: LookdevEngine }).__lookdev = engine
        }
        setMode('live')
        layout()
        engine.setVisible(visible)
      } catch (err) {
        console.error(err)
        engine = null
        engineRef.current = null
        setMode('fallback')
        io?.disconnect()
      }
    }

    // GPU resets (driver updates, sleep and wake) lose the WebGL context. Pause behind the still frame and
    // rebuild everything when the browser restores it.
    const onContextLost = () => {
      contextLost = true
      engine?.dispose()
      engine = null
      engineRef.current = null
      setPaused(true)
    }
    const onContextRestored = () => {
      contextLost = false
      setPaused(false)
      if (visible) create()
    }
    canvas.addEventListener('webglcontextrestored', onContextRestored)

    io = new IntersectionObserver(
      entries => {
        visible = entries.some(en => en.isIntersecting)
        if (visible && !engine && !contextLost) {
          if (!LookdevEngine.detect()) {
            setMode('fallback')
            io?.disconnect()
            return
          }
          create()
          return
        }
        engine?.setVisible(visible)
      },
      { rootMargin: '200px 0px' },
    )
    io.observe(section)
    const ro = new ResizeObserver(layout)
    ro.observe(frame)
    return () => {
      io?.disconnect()
      ro.disconnect()
      canvas.removeEventListener('webglcontextrestored', onContextRestored)
      engine?.dispose()
      engineRef.current = null
    }
  }, [])

  const setFromPointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const u = clamp((e.clientX - r.left) / r.width, 0, 1)
    const v = clamp((e.clientY - r.top) / r.height, 0, 1)
    update({ keyAz: (u - 0.5) * AZ_RANGE, keyEl: clamp(EL_MAX - v * (EL_MAX - EL_MIN) * 1.1, EL_MIN, EL_MAX) })
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (lab.furnace) return
    dragging.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    engineRef.current?.setInteracting(true)
    setFromPointer(e)
  }
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragging.current) setFromPointer(e)
  }
  const endDrag = () => {
    if (!dragging.current) return
    dragging.current = false
    engineRef.current?.setInteracting(false)
  }
  const onKeyDown = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (lab.furnace) return
    const s = 0.08
    let { keyAz, keyEl } = lab
    if (e.key === 'ArrowLeft') keyAz -= s
    else if (e.key === 'ArrowRight') keyAz += s
    else if (e.key === 'ArrowUp') keyEl += s
    else if (e.key === 'ArrowDown') keyEl -= s
    else return
    e.preventDefault()
    update({ keyAz: clamp(keyAz, -AZ_RANGE / 2, AZ_RANGE / 2), keyEl: clamp(keyEl, EL_MIN, EL_MAX) })
  }

  const hero = HERO_PRESETS[lab.hero]
  const heroRough = lab.heroRoughness ?? hero.params[hero.roughnessParam]
  const markerX = (lab.keyAz / AZ_RANGE + 0.5) * 100
  const markerY = ((EL_MAX - lab.keyEl) / ((EL_MAX - EL_MIN) * 1.1)) * 100
  const ballNames = ['18% gray', 'Chrome', hero.label]

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
          <Image
            src="/lookdev/poster.jpg"
            alt="Path traced lookdev reference balls: an 18% gray card ball, a chromium ball and a clear coated car paint ball under a three point softbox rig"
            fill
            sizes="(max-width: 768px) 100vw, 1200px"
            style={{ objectFit: 'cover' }}
          />
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

        {mode === 'live' && paused && (
          <>
            <Image src="/lookdev/poster.jpg" alt="" fill sizes="(max-width: 768px) 100vw, 1200px" style={{ objectFit: 'cover' }} />
            <div className="lab-hud lab-hud-left">Renderer paused. It resumes when the browser restores graphics.</div>
          </>
        )}
        {mode === 'live' && !paused && (
          <>
            <div className="sr-only" aria-live="polite">
              {`Key light at ${deg(lab.keyAz)} degrees azimuth, ${deg(lab.keyEl)} degrees elevation`}
            </div>
            <div className="lab-hud lab-hud-left">
              {lab.furnace ? 'White furnace' : `Key ${deg(lab.keyAz)}° az  ${deg(lab.keyEl)}° el  ${lab.keyKelvin}K`}
            </div>
            <div className="lab-hud lab-hud-right">
              {PASSES.find(p => p.id === lab.pass)?.label}
              {'  '}
              {status.converged ? `${status.spp} spp` : `${status.spp} / ${status.target} spp`}
            </div>
            <div className="lab-progress" style={{ transform: `scaleX(${status.spp / status.target})` }} />
            {lab.key && !lab.furnace && (
              <div className="lab-marker" style={{ left: `${markerX}%`, top: `${clamp(markerY, 0, 100)}%` }} />
            )}
            {labels.map((p, i) => (
              <div key={i} className="lab-ball-label" style={{ left: `${p.x * 100}%`, top: `calc(${p.y * 100}% + 12px)` }}>
                {ballNames[i]}
              </div>
            ))}
          </>
        )}
        {mode === 'fallback' && (
          <div className="lab-hud lab-hud-left">Still frame. Your browser lacks the WebGL2 features the live renderer needs.</div>
        )}
      </div>

      {mode !== 'fallback' && (
        <div className="lab-controls">
          <div className="lab-row">
            <Group label="Lights">
              <Pill on={lab.key} onClick={() => update({ key: !lab.key })}>Key</Pill>
              <Pill on={lab.fill} onClick={() => update({ fill: !lab.fill })}>Fill</Pill>
              <Pill on={lab.rim} onClick={() => update({ rim: !lab.rim })}>Rim</Pill>
            </Group>
            <Group label="Hero">
              {HERO_ORDER.map(h => (
                <Pill key={h} on={lab.hero === h} onClick={() => update({ hero: h, heroRoughness: null })}>
                  {HERO_PRESETS[h].label}
                </Pill>
              ))}
            </Group>
          </div>

          <div className="lab-row">
            <Group label="Key temp">
              <input
                type="range"
                className="lab-range"
                min={2500}
                max={9000}
                step={100}
                value={lab.keyKelvin}
                aria-label="Key temp, kelvin"
                onChange={e => update({ keyKelvin: +e.target.value })}
              />
              <span className="lab-readout">{lab.keyKelvin}K</span>
            </Group>
            <Group label="Exposure">
              <input
                type="range"
                className="lab-range"
                min={-3}
                max={3}
                step={0.1}
                value={lab.exposure}
                aria-label="Exposure, stops"
                onChange={e => update({ exposure: +e.target.value })}
              />
              <span className="lab-readout">
                {lab.exposure >= 0 ? '+' : ''}
                {lab.exposure.toFixed(1)}
              </span>
            </Group>
            <Group label={`${hero.label} roughness`}>
              <input
                type="range"
                className="lab-range"
                min={0}
                max={1}
                step={0.01}
                value={heroRough}
                aria-label={`${hero.label} roughness`}
                onChange={e => update({ heroRoughness: +e.target.value })}
              />
              <span className="lab-readout">{heroRough.toFixed(2)}</span>
            </Group>
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
                {furnaceMean !== null && ` Measured mean over the spheres: ${furnaceMean.toFixed(3)} (ideal 1.000).`}
              </span>
            )}
          </div>
        </div>
      )}

      <div className="lab-methods">
        <p className="lab-note" style={{ width: '100%', margin: '0 0 4px' }}>
          Rendered in ACEScg. With multiple-scattering compensation on, every preset averages within 0.02% of 1.0 in a
          white furnace test.
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
