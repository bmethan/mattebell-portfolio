// The burning ball held in the fire and pulled out: its hottest surface, the power it burns at and how much of it is
// charred, every few seconds, and when it first flames. The documented hold-and-pull results come from this with the
// defaults (the ball 0.45 m up in the 180 kW bonfire, pulled to 1.3, 0.6) and --pull-at 8, 25 and 60.
import { launch, openLab, options, runTo } from './lib.mjs'

const o = options(
  {
    until: { type: 'string', default: '240' },
    every: { type: 'string', default: '5' },
  },
  `
ballburn.mjs: the ball's burning over time.
  --until 240       last simulated second
  --every 5         seconds between readings (use 1 to time the catch)
Defaults for this probe: --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-to 1.3,0.6 (pass --pull-at to pull it out).
Example: node tools/pyro-probes/ballburn.mjs --pull-at 60`,
)
if (o.preset === 'Smoke plume') o.scenario.preset = 'Bonfire at dusk'
if (!o.scenario.ballAt) o.scenario.ballAt = [0, 0.45]
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  const rows = []
  let caught = null
  for (let t = Number(o.every); t <= Number(o.until) + 1e-9; t += Number(o.every)) {
    await runTo(page, t, o.scenario)
    const st = await page.evaluate(async () => {
      const e = window.__pyro
      await e.readBall()
      await new Promise(r => setTimeout(r, 150))
      return e.ballStatus
    })
    if (caught === null && st.burningKW >= 0.5) caught = t
    rows.push(`${+t.toFixed(2)}s ${Math.round(st.surfaceK - 273)}C ${st.burningKW.toFixed(1)}kW ${(st.charred * 100).toFixed(0)}%`)
  }
  console.log(`held at ${o.scenario.ballAt}${o.scenario.pullAt != null ? `, pulled at ${o.scenario.pullAt} s to ${o.scenario.pullTo}` : ''}: first flaming (0.5 kW) by ${caught ?? 'never'} s`)
  console.log(rows.join(' | '))
} finally {
  await browser.close()
}
