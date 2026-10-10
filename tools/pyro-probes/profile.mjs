// GPU time per kernel for one step and one frame (the engine's profileStep, which needs timestamp queries), the
// median of several, after the scenario runs a while; and the live steps a second the HUD reports. Timings on an
// integrated GPU vary by up to two times from run to run: compare like with like, in one sitting.
import { launch, list, openLab, options, runTo } from './lib.mjs'

const o = options(
  {
    at: { type: 'string', default: '6' },
    runs: { type: 'string', default: '7' },
    resolutions: { type: 'string', default: 'medium' },
    live: { type: 'boolean', default: false },
  },
  `
profile.mjs: where a step's GPU time goes.
  --at 6               simulated second to profile at
  --runs 7             profiles to take the median of
  --resolutions medium,high
  --live               also let it run live for 8 s and read the HUD's steps a second
Example: node tools/pyro-probes/profile.mjs --patch '{"velocity":"lfm"}' --live`,
)
const browser = await launch()
try {
  for (const res of o.resolutions.split(',')) {
    const page = await openLab(browser, { ...o.scenario, patch: { ...o.scenario.patch, resolution: res } })
    await runTo(page, Number(o.at), o.scenario)
    const runs = []
    for (let i = 0; i < Number(o.runs); i++) runs.push(await page.evaluate(() => window.__pyro.profileStep()))
    if (!runs[0]) {
      console.log('no timestamp queries on this GPU or browser')
      break
    }
    const med = {}
    for (const k of Object.keys(runs[0])) {
      const v = runs.map(r => r[k]).sort((a, b) => a - b)
      med[k] = +v[v.length >> 1].toFixed(2)
    }
    const total = Object.values(med).reduce((a, b) => a + b, 0)
    console.log(res, JSON.stringify(med), 'total', total.toFixed(1), 'ms')
    if (o.live) {
      await page.evaluate(() => window.__pyro.setPaused(false))
      await page.waitForTimeout(8000)
      const reads = []
      for (let i = 0; i < 4; i++) {
        await page.waitForTimeout(1000)
        reads.push((await page.locator('#lab-panel-pyro .lab-hud-right').innerText()).replace(/\s+/g, ' '))
      }
      console.log(res, 'live:', reads.join(' | '))
    }
    await page.close()
  }
} finally {
  await browser.close()
}
