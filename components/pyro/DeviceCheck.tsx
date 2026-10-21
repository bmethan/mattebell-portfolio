'use client'

// The lab's device check, shown only with ?check in the address: it runs each starting point for a few seconds, then
// a smoke source on leapfrog flow maps and the wooden ball held in the fire, and writes a report to copy: the device,
// what its GPU adapter offers, the steps and frames a second each case managed, any WebGPU errors, and any values the
// frame or the light textures could not hold (not finite, or held at half floats' ceiling: an iPad once drew an oil
// fire's base teal from one, with no error to show for it). For testing the lab on phones, tablets and other
// computers; visitors never see it.
import { useState, type RefObject } from 'react'
import type { PyroEngine, PyroStatus } from './engine'
import { PRESETS, PYRO_DEFAULTS, type PyroState } from './params'
import { Pill } from '../lab/ui'

interface Props {
  engineRef: RefObject<PyroEngine | null>
  statusRef: RefObject<PyroStatus | null>
  frameRef: RefObject<HTMLDivElement | null>
  applyPreset: (state: Partial<PyroState>) => void
  ensureRunning: () => void
  fallback: boolean
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))
const WARM_S = 5 // long enough for the render scale to settle on a GPU that falls behind
const MEASURE_S = 5

function deviceLine() {
  const s = window.screen
  return `${navigator.userAgent}\nScreen ${s.width}x${s.height} at ${window.devicePixelRatio}x, window ${window.innerWidth}x${window.innerHeight}`
}

export default function DeviceCheck({ engineRef, statusRef, frameRef, applyPreset, ensureRunning, fallback }: Props) {
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [report, setReport] = useState('')
  const [copied, setCopied] = useState(false)

  const measure = async (e: PyroEngine) => {
    const errors0 = e.gpuErrors.length
    const steps: number[] = []
    let frames = 0
    let counting = true
    const tick = () => {
      frames++
      if (counting) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
    const t0 = performance.now()
    for (let i = 0; i < MEASURE_S; i++) {
      await sleep(1000)
      const s = statusRef.current
      if (s) steps.push(s.stepsPerSec)
    }
    counting = false
    const v = await e.checkValues().catch(() => null)
    const wall = (performance.now() - t0) / 1000
    steps.sort((a, b) => a - b)
    const s = statusRef.current
    return {
      steps: steps.length ? steps[steps.length >> 1] : 0,
      fps: Math.round(frames / wall),
      scale: s ? s.renderScale.toFixed(2) : '?',
      resolution: s ? `${s.resolution}${s.autoStepped ? ' (stepped down)' : ''}` : '?',
      errors: e.gpuErrors.length - errors0,
      values: v,
    }
  }

  const start = async () => {
    if (running) return
    setRunning(true)
    setReport('')
    setCopied(false)
    const lines: string[] = [`Pyro lab device check, ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`, deviceLine()]
    const engine = engineRef.current
    try {
      if (fallback || !engine) {
        lines.push('WebGPU: not available here, so the lab shows its still frame.')
        return
      }
      const e = engine
      lines.push(`Adapter: ${e.adapterName}`)
      lines.push(`Features: ${e.adapterFeatures.join(', ') || 'none listed'}`)
      lines.push(`Limits: ${Object.entries(e.adapterLimits).map(([k, v]) => `${k.replace(/^max/, '')} ${v}`).join(', ')}`)
      lines.push('', 'Case | resolution | steps/s | frames/s | render scale | errors | not finite (frame, light) | held at 65504')
      // Centred, clear of the sticky header, with the progress line below it in view.
      frameRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' })
      const cases: { name: string; state: Partial<PyroState> }[] = [
        ...PRESETS.map(p => ({ name: p.label, state: p.state })),
        { name: 'Smoke plume, flow maps', state: { ...PRESETS.find(p => p.id === 'plume')!.state, velocity: 'lfm' } },
      ]
      let notFinite = 0
      const badWhere = new Set<string>()
      for (const [i, c] of cases.entries()) {
        setProgress(`${i + 1} of ${cases.length + 1}: ${c.name}`)
        applyPreset({ ball: false, velocity: PYRO_DEFAULTS.velocity, ...c.state })
        ensureRunning()
        await sleep(WARM_S * 1000)
        const m = await measure(e)
        const v = m.values
        if (v) {
          notFinite += v.frame + v.textures
          v.where.forEach(w => badWhere.add(`${c.name}: ${w}`))
        }
        lines.push(`${c.name} | ${m.resolution} | ${m.steps} | ${m.fps} | ${m.scale} | ${m.errors} | ${v ? `${v.frame}, ${v.textures}` : '?'} | ${v ? v.ceiling : '?'}`)
      }
      // The wooden ball held in the bonfire's flames until 16 simulated seconds (it catches at 10 to 12).
      setProgress(`${cases.length + 1} of ${cases.length + 1}: the wooden ball in the bonfire`)
      applyPreset({ velocity: PYRO_DEFAULTS.velocity, ...PRESETS.find(p => p.id === 'bonfire')!.state, ball: true, ballBurns: true })
      ensureRunning()
      await sleep(300)
      e.placeBall(0, 0.45)
      const t0 = performance.now()
      while ((statusRef.current?.simTime ?? 0) < 16 && performance.now() - t0 < 60000) await sleep(500)
      const s = statusRef.current
      const b = s?.ball
      lines.push(
        '',
        b
          ? `Burning ball at ${s!.simTime.toFixed(1)} s: ${Math.round(b.surfaceK - 273)} C, burning ${b.burningKW.toFixed(1)} kW, ${Math.round(b.charred * 100)}% charred: ${b.burningKW >= 0.5 ? 'caught, as it should' : 'not alight yet (it should be by about 12 s)'}`
          : `Burning ball: no reading by ${(s?.simTime ?? 0).toFixed(1)} s`,
      )
      // Its sparks: thrown once it flames, some in the air, none with a value not finite.
      const sp = await e.readSparks().catch(() => null)
      if (sp) {
        const n = sp.sparks.length / 10
        let glowing = 0
        let hottest = 0
        let bad = 0
        for (let k = 0; k < n; k++) {
          const f = sp.sparks.subarray(k * 10, k * 10 + 10)
          if (!f.every(Number.isFinite)) bad++
          else {
            if (f[7] > 798) glowing++
            hottest = Math.max(hottest, f[7])
          }
        }
        lines.push(`Sparks by then: ${sp.thrown} thrown, ${n} in the air, ${glowing} glowing${n ? `, hottest ${Math.round(hottest - 273)} C` : ''}, ${bad} not finite: ${sp.thrown > 0 && bad === 0 ? 'as they should be' : bad ? 'values not finite' : 'none thrown yet'}`)
      }
      lines.push(`WebGPU errors in all: ${e.gpuErrors.length}${e.gpuErrors.length ? `; first: ${e.gpuErrors[0]}` : ''}`)
      lines.push(`Values not finite in all: ${notFinite}${badWhere.size ? ` (${[...badWhere].slice(0, 6).join('; ')})` : ''}`)
    } finally {
      setProgress('')
      setReport(lines.join('\n'))
      setRunning(false)
    }
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(report)
      setCopied(true)
    } catch {
      const area = document.getElementById('pyro-check-report') as HTMLTextAreaElement | null
      area?.select()
    }
  }

  return (
    <section className="lab-check" aria-label="Device check">
      <p className="lab-note">
        <b>Device check.</b> Runs each starting point for {WARM_S + MEASURE_S} seconds, then a smoke source on flow maps and
        the wooden ball held in the fire, and writes a report to copy. About a minute and a half; keep this page open and on
        screen while it runs.
      </p>
      <div className="lab-row">
        <Pill on={running} onClick={start}>
          {running ? 'Running' : report ? 'Run again' : 'Start'}
        </Pill>
        {report && (
          <Pill on={false} onClick={copy}>
            {copied ? 'Copied' : 'Copy report'}
          </Pill>
        )}
      </div>
      {progress && (
        <p className="lab-note" aria-live="polite">
          {progress}
        </p>
      )}
      {report && <textarea id="pyro-check-report" className="lab-check-report" readOnly value={report} rows={16} onFocus={ev => ev.currentTarget.select()} />}
    </section>
  )
}
