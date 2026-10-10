// Shared plumbing for the Pyro lab probes: a headless Edge on the dev server's lab (where the engine is exposed as
// window.__pyro, in development builds only), the scenarios the probes run, and their command-line options.
import { createRequire } from 'node:module'
import fs from 'node:fs'
import { parseArgs } from 'node:util'

export const LAB_URL = process.env.PYRO_URL ?? 'http://localhost:4622/?lab=pyro'
const BROWSER = process.env.PYRO_BROWSER ?? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
// Playwright is not a dependency of the site. The probes use this repo's copy if one is installed, else the one in
// the node_modules beside the package.json that PLAYWRIGHT_FROM names (on Matthew's machine, magpie's).
const PLAYWRIGHT_FROM = [process.env.PLAYWRIGHT_FROM, 'C:/Users/leahp/Claude_Local/magpie/package.json'].filter(Boolean)

async function playwright() {
  for (const name of ['playwright', 'playwright-core']) {
    try {
      return await import(name)
    } catch {}
  }
  for (const from of PLAYWRIGHT_FROM) {
    if (!fs.existsSync(from)) continue
    try {
      return createRequire(from)('playwright')
    } catch {}
  }
  throw new Error('No Playwright found: install playwright-core here (npm i -D playwright-core) or set PLAYWRIGHT_FROM to a package.json whose node_modules has playwright.')
}

// The options every probe takes, plus its own; `help` is printed for --help.
export function options(own, help) {
  const { values } = parseArgs({
    options: {
      preset: { type: 'string', default: 'Smoke plume' },
      patch: { type: 'string', default: '{}' },
      'ball-at': { type: 'string' },
      'pull-at': { type: 'string' },
      'pull-to': { type: 'string', default: '1.3,0.6' },
      out: { type: 'string', default: 'probe-out' },
      help: { type: 'boolean', default: false },
      ...own,
    },
  })
  if (values.help) {
    console.log(help.trim())
    console.log(`
Scenario options (every probe):
  --preset "Smoke plume"   the starting point to click (its button label; "" for none)
  --patch '{"velocity":"lfm"}'   lab state to apply on top (JSON)
  --ball-at 0,0.45         put the collision ball here (x, y in metres) at the start
  --pull-at 60 --pull-to 1.3,0.6   move it there at that simulated second
  --out probe-out          where files go
Environment: PYRO_URL (default ${LAB_URL}), PYRO_BROWSER, PLAYWRIGHT_FROM.`)
    process.exit(0)
  }
  const nums = s => (s ? s.split(',').map(Number) : null)
  values.scenario = {
    preset: values.preset,
    patch: JSON.parse(values.patch),
    ballAt: nums(values['ball-at']),
    pullAt: values['pull-at'] != null ? Number(values['pull-at']) : null,
    pullTo: nums(values['pull-to']),
  }
  return values
}
export const list = s => s.split(',').map(Number)

export async function launch() {
  const { chromium } = await playwright()
  return chromium.launch({
    executablePath: BROWSER,
    headless: true,
    args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-dawn-features=allow_unsafe_apis'],
  })
}

// Opens the lab, clicks the scenario's starting point, applies its patch and restarts, paused: from here the
// simulation moves only when a probe steps it, so every run of a scenario is the same run.
export async function openLab(browser, scenario) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 } })
  page.on('console', m => {
    if (m.type() === 'error' && !/401|font-size|Turnstile|ERR_INTERNET/.test(m.text())) console.error('page error:', m.text().slice(0, 300))
  })
  page.on('pageerror', e => console.error('page exception:', e.message))
  await page.goto(LAB_URL, { waitUntil: 'load', timeout: 120000 })
  await page.locator('#lab-panel-pyro .lab-frame').scrollIntoViewIfNeeded()
  await page.waitForFunction(() => window.__pyro && document.querySelector('#lab-panel-pyro .lab-hud-right'), null, { timeout: 60000 })
  await page.waitForTimeout(1500)
  if (scenario.preset) {
    await page.locator('#lab-panel-pyro .lab-panel button', { hasText: scenario.preset }).first().click()
    await page.waitForTimeout(300)
  }
  await page.evaluate(async s => {
    const e = window.__pyro
    e.autoResolution = false
    e.setPaused(true)
    e.setState(s.ballAt ? { ...s.patch, ball: true } : s.patch)
    e.restart()
    await new Promise(r => setTimeout(r, 300))
    if (s.ballAt) e.ballPos = [s.ballAt[0], s.ballAt[1], 0]
  }, scenario)
  return page
}

// Steps the lab to simulated time t (s), moving the ball to its pull position on the way if the scenario pulls it.
export async function runTo(page, t, scenario) {
  return page.evaluate(
    async ([t, s]) => {
      const e = window.__pyro
      const stepTo = async to => {
        const n = Math.round((to - e.simTime) * 30)
        if (n > 0) await e.advance(n)
      }
      if (s.pullAt != null && e.simTime < s.pullAt - 1e-6 && t >= s.pullAt) {
        await stepTo(s.pullAt)
        e.ballPos = [s.pullTo[0], s.pullTo[1], 0]
      }
      await stepTo(t)
      return e.simTime
    },
    [t, scenario],
  )
}

export function outDir(dir) {
  fs.mkdirSync(dir, { recursive: true })
  return dir
}
export function saveBase64(file, b64) {
  fs.writeFileSync(file, Buffer.from(b64, 'base64'))
}

// In-page helpers, as a string to evaluate once per page: the gas state of a cell from the raw scalars (a fire's are
// per unit volume: soot, density deficit, mixture fraction and fuel as partial densities; a smoke source's are
// density and temperature rise), and a cell's world position.
export const PAGE_HELPERS = `
window.__probe = {
  gas(s, i, fire) {
    const d = s.data
    if (!fire) return { density: d[i * 4], rise: d[i * 4 + 1] }
    const r = Math.max(1 - d[i * 4 + 1], 0.1)
    return { soot: Math.max(d[i * 4], 0) / r, rise: 293 * Math.max(d[i * 4 + 1], 0) / r, z: Math.max(d[i * 4 + 2], 0) / r, fuel: Math.max(d[i * 4 + 3], 0) / r, hrr: s.hrr[i] }
  },
  world(s, x, y, z) {
    const [nx, , nz] = s.n
    return [(-nx * s.dx) / 2 + (x + 0.5) * s.dx, (y + 0.5) * s.dx, (-nz * s.dx) / 2 + (z + 0.5) * s.dx]
  },
}`
