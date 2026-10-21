// The burning ball's sparks (burn-6): held in the fire (and pulled out, if the scenario pulls it), then followed step by
// step for a few seconds: how many the wood throws a second (all, and over a millimetre, against the ball's flames'
// power: Petersen and Banerjee's pile threw some 130 over a millimetre a second per MW), how many are in the air and
// glowing, their temperatures, how long each glows (over 798 K: their pile's averaged 0.4 s, at most about 2), which
// way they go a quarter second out, and what ends them. Optionally stirred beside the ball, or the ball dragged, while
// followed.
import fs from 'node:fs'
import { launch, list, openLab, options, outDir, runTo } from './lib.mjs'

const o = options(
  {
    at: { type: 'string', default: '45' },
    follow: { type: 'string', default: '5' },
    'wood-speed': { type: 'string', default: '4' },
    stir: { type: 'string' },
    drag: { type: 'string' },
  },
  `
sparks.mjs: the burning ball's sparks, followed step by step.
  --at 45            simulated second to start following at (the wood run --wood-speed times faster until then)
  --follow 5         seconds to follow them
  --wood-speed 4     how many times faster the wood runs until --at (1 thereafter)
  --stir 1,2.5       stir beside the ball (upstream 0.25 m, pushing along +x at 2 m/s) from 1 s to 2.5 s into the follow
  --drag 0.5,1.4,1.3,0.6   drag the ball from where it is to (1.3, 0.6) from 0.5 s to 1.4 s into the follow
Defaults for this probe: --preset "Bonfire at dusk" --ball-at 0,0.45.
Example: node tools/pyro-probes/sparks.mjs --at 75 --drag 0.5,1.4,1.3,0.6`,
)
if (!o.scenario.ballAt) o.scenario.ballAt = [0, 0.45]
if (o.preset === 'Smoke plume') o.scenario.preset = 'Bonfire at dusk'
const dir = outDir(o.out)
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  await page.evaluate(speed => {
    window.__pyro.woodSpeed = speed
  }, Number(o['wood-speed']))
  await runTo(page, Number(o.at), o.scenario)
  const res = await page.evaluate(
    async ([secs, stir, drag]) => {
      const e = window.__pyro
      e.woodSpeed = 1
      const steps = Math.round(secs * 30)
      const seen = new Map() // id -> { glow (s), d0 (mm), angle (deg at 0.25 s), last }
      const perSecond = []
      let thrown0 = (await e.readSparks()).thrown
      let kw = 0
      let n = 0
      const temps = []
      const from = [...e.ballPos]
      for (let i = 0; i < steps; i++) {
        const t = i / 30
        if (stir && t >= stir[0] && t < stir[1]) {
          const b = e.ballPos
          e.stir = { centre: [b[0] - 0.25, b[1], b[2]], vel: [2, 0, 0] }
        }
        if (drag && t >= drag[0] && t < drag[1]) {
          const ease = f => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(Math.max(f, 0), 1))
          const at = f => [from[0] + (drag[2] - from[0]) * ease(f), from[1] + (drag[3] - from[1]) * ease(f), 0]
          const f0 = (t - drag[0]) / (drag[1] - drag[0])
          const f1 = (t + 1 / 30 - drag[0]) / (drag[1] - drag[0])
          const p0 = at(f0)
          const p1 = at(f1)
          e.ballPos = p1
          e.ballVel = p1.map((x, k) => (x - p0[k]) * 30)
          e.ballDragAt = performance.now()
        }
        await e.advance(1)
        const r = await e.readSparks()
        const live = new Set()
        let glowing = 0
        for (let k = 0; k < r.sparks.length; k += 10) {
          const id = r.sparks[k + 9]
          const T = r.sparks[k + 7]
          live.add(id)
          let rec = seen.get(id)
          if (!rec) {
            rec = { glow: 0, d0: r.sparks[k + 3] * 1000, angle: null, gone: false }
            seen.set(id, rec)
          }
          if (T > 798) {
            rec.glow += 1 / 30
            glowing++
          }
          if (rec.angle === null && r.sparks[k + 8] >= 0.25) {
            const v = [r.sparks[k + 4], r.sparks[k + 5], r.sparks[k + 6]]
            const sp = Math.hypot(...v)
            rec.angle = sp > 0.05 ? (Math.acos(Math.max(-1, Math.min(1, v[1] / sp))) * 180) / Math.PI : null
          }
          if (i % 10 === 0) temps.push(T)
        }
        for (const [id, rec] of seen) if (!live.has(id)) rec.gone = true
        if (i % 30 === 29) {
          const thrown = r.thrown
          const st = e.ballStatus
          perSecond.push({ t: +(e.simTime).toFixed(1), thrown: thrown - thrown0, inAir: r.sparks.length / 10, glowing, ballKW: st ? +st.burningKW.toFixed(1) : 0 })
          thrown0 = thrown
          if (st) {
            kw += st.burningKW
            n++
          }
          await e.readBall()
        }
      }
      const done = [...seen.values()].filter(r => r.gone)
      const glows = done.map(r => r.glow).filter(g => g > 0)
      const big = done.filter(r => r.d0 >= 1).map(r => r.glow).filter(g => g > 0)
      const angles = [...seen.values()].map(r => r.angle).filter(a => a !== null)
      const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0)
      const median = a => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : 0)
      return {
        perSecond,
        ballKW: n ? kw / n : 0,
        followed: seen.size,
        glowLife: { n: glows.length, mean: mean(glows), median: median(glows), max: glows.length ? Math.max(...glows) : 0, over1mm: mean(big), over1mmN: big.length },
        tempMedian: median(temps),
        hotterThan1500: temps.length ? temps.filter(t => t > 1500).length / temps.length : 0,
        within33: angles.length ? angles.filter(a => a <= 33).length / angles.length : 0,
        upward: angles.length ? angles.filter(a => a < 90).length / angles.length : 0,
      }
    },
    [Number(o.follow), o.stir ? list(o.stir) : null, o.drag ? list(o.drag) : null],
  )
  const thrownPerS = res.perSecond.reduce((a, p) => a + p.thrown, 0) / Math.max(res.perSecond.length, 1)
  const lines = [
    `sparks from ${o.at} s for ${o.follow} s (${JSON.stringify(o.scenario)}${o.stir ? `, stirred ${o.stir}` : ''}${o.drag ? `, dragged ${o.drag}` : ''})`,
    ...res.perSecond.map(p => `  ${p.t} s: thrown ${p.thrown}/s, ${p.inAir} in the air, ${p.glowing} glowing, ball's flames ${p.ballKW} kW`),
    `thrown ${thrownPerS.toFixed(0)} a second, ${(thrownPerS * 0.115).toFixed(1)} over 1 mm; ball's flames ${res.ballKW.toFixed(0)} kW: ${((thrownPerS * 0.115) / Math.max(res.ballKW / 1000, 1e-6)).toFixed(0)} over 1 mm per MW (pile: some 130)`,
    `glowing life (over 798 K), ${res.glowLife.n} sparks ended: mean ${res.glowLife.mean.toFixed(2)} s, median ${res.glowLife.median.toFixed(2)}, longest ${res.glowLife.max.toFixed(2)}; over 1 mm (${res.glowLife.over1mmN}): mean ${res.glowLife.over1mm.toFixed(2)} s (pile: 0.4 s, at most about 2)`,
    `temperature: median ${res.tempMedian.toFixed(0)} K, ${(res.hotterThan1500 * 100).toFixed(0)}% over 1500 K (embers in air: 1020 to 1340 K)`,
    `a quarter second out: ${(res.within33 * 100).toFixed(0)}% within 33 degrees of vertical, ${(res.upward * 100).toFixed(0)}% rising (pile: 68% within 33)`,
  ]
  console.log(lines.join('\n'))
  fs.writeFileSync(`${dir}/sparks-${o.at}.txt`, lines.join('\n') + '\n')
} finally {
  await browser.close()
}
