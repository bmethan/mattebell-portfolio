// Whether a run stays sane: the fastest gas and where it is, the kinetic energy and the total smoke, every few
// seconds, optionally while the probe stirs the gas or sweeps the ball through it. A run that blows up shows here
// as a fastest speed of hundreds or thousands of m/s.
import { launch, openLab, options } from './lib.mjs'

const o = options(
  {
    until: { type: 'string', default: '30' },
    every: { type: 'string', default: '5' },
    action: { type: 'string', default: 'none' },
  },
  `
stability.mjs: speed, energy and smoke over a run.
  --until 30 --every 5    last simulated second, and seconds between readings
  --action none|stir|ball  stir the gas every third step, or sweep the ball back and forth, while it runs
Example: node tools/pyro-probes/stability.mjs --patch '{"velocity":"lfm","resolution":"high"}' --action stir`,
)
if (o.action === 'ball' && !o.scenario.ballAt) o.scenario.ballAt = [0, 0.45]
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  const every = Number(o.every)
  let step = 0
  for (let t = every; t <= Number(o.until) + 1e-9; t += every) {
    const r = await page.evaluate(
      async ([action, n, step0]) => {
        const e = window.__pyro
        for (let s = 0; s < n; s++) {
          const k = step0 + s
          if (action === 'ball') e.nudgeBall(k % 120 < 60 ? 0.04 : -0.04, k % 40 < 20 ? 0.01 : -0.01)
          if (action === 'stir' && k % 3 === 0) e.stirAt(0.05 * Math.sin(k / 10), 0.1, 0.02 * Math.cos(k / 7), 0.015, 16)
          await e.advance(1)
        }
        const f = await e.readFields()
        const [nx, ny] = f.n
        let mx = 0, at = 0, ke = 0, nan = 0, smoke = 0
        for (let i = 0; i < f.vel.length / 3; i++) {
          const a = f.vel[3 * i], b = f.vel[3 * i + 1], c = f.vel[3 * i + 2]
          if (!Number.isFinite(a + b + c)) { nan++; continue }
          const m = Math.hypot(a, b, c)
          ke += m * m
          if (m > mx) { mx = m; at = i }
          smoke += f.density[i]
        }
        return { max: +mx.toFixed(3), at: [at % nx, Math.floor(at / nx) % ny, Math.floor(at / nx / ny)], ke: +(0.5 * ke * f.dx ** 3).toFixed(4), smoke: +(smoke * f.dx ** 3).toFixed(3), nan }
      },
      [o.action, Math.round(every * 30), step],
    )
    step += Math.round(every * 30)
    console.log(`${+t.toFixed(2)} s`, JSON.stringify(r))
  }
} finally {
  await browser.close()
}
