// The wooden ball's embers: held in the fire, pulled out, left in still air, stirred beside it, and left again. Each
// second: its hottest surface, the hottest 5% of its faces and of its cracks' walls, how much of it glows (above 500 C,
// a dull red, and above 700 C, a bright one), and the power its flames burn at.
import { launch, openLab, options, runTo } from './lib.mjs'

const o = options(
  {
    still: { type: 'string', default: '30' },
    stir: { type: 'string', default: '8' },
    after: { type: 'string', default: '20' },
    'stir-speed': { type: 'string', default: '2' },
    'stir-from': { type: 'string', default: '0.25' },
  },
  `
embers.mjs: the ball's embers after it is pulled out, still and stirred.
  --still 30          seconds left in still air after the pull
  --stir 8            seconds stirred beside it
  --after 20          seconds left again after the stir
  --stir-speed 2      the stir's push (m/s a step, as a drag gives it, up to 3)
  --stir-from 0.25    how far upstream of the ball the stir is centred (m), pushing along +x
Defaults for this probe: --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-at 300 --pull-to 1.3,0.6.
Example: node tools/pyro-probes/embers.mjs --pull-at 90 --stir-speed 0.5`,
)
if (o.preset === 'Smoke plume') o.scenario.preset = 'Bonfire at dusk'
if (!o.scenario.ballAt) o.scenario.ballAt = [0, 0.45]
if (o.scenario.pullAt == null) o.scenario.pullAt = 300
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  await runTo(page, o.scenario.pullAt + 0.05, o.scenario)
  const out = await page.evaluate(
    async ([still, stirS, after, speed, from]) => {
      const e = window.__pyro
      const d = e.device
      const N = 64
      const T = 6 * N * N
      const solid = new Float32Array(N * N)
      const a = Math.PI / (2 * N)
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const u = Math.tan((((i + 0.5) * 2) / N - 1) * (Math.PI / 4))
        const v = Math.tan((((j + 0.5) * 2) / N - 1) * (Math.PI / 4))
        const q = 1 + u * u + v * v
        solid[j * N + i] = (a * a * (1 + u * u) * (1 + v * v)) / (q * Math.sqrt(q))
      }
      const grab = async (buf, size) => {
        const r = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
        const enc = d.createCommandEncoder()
        enc.copyBufferToBuffer(buf, 0, r, 0, size)
        d.queue.submit([enc.finish()])
        await r.mapAsync(GPUMapMode.READ)
        const x = new Float32Array(r.getMappedRange().slice(0))
        r.unmap()
        r.destroy()
        return x
      }
      const top5 = xs => {
        xs.sort((p, q) => q - p)
        return Math.round(xs[Math.round(xs.length * 0.05)] - 273)
      }
      const read = async label => {
        const skin = await grab(e.grids.skin, T * 16)
        const sheet = await grab(e.grids.ballSheet, T * 32)
        const r = e.state.ballRadius
        const faces = []
        const cracks = []
        let a500 = 0
        let a700 = 0
        let fuel = 0
        for (let t = 0; t < T; t++) {
          const ts = skin[4 * t]
          const area = r * r * solid[t % (N * N)]
          faces.push(ts)
          cracks.push(sheet[8 * t + 7])
          if (ts > 773) a500 += area
          if (ts > 973) a700 += area
          if (skin[4 * t + 1] > 0.002) fuel += skin[4 * t + 1] * area
        }
        const hottest = Math.round(Math.max(...faces) - 273)
        return `${label.padEnd(6)} ${e.simTime.toFixed(0).padStart(4)} s  hottest ${hottest} C  faces' top 5% ${top5(faces)} C, cracks' ${top5(cracks)} C  above 500 C ${Math.round(a500 * 1e4)} cm2, above 700 C ${Math.round(a700 * 1e4)} cm2  flaming ${(fuel * 17.8e3).toFixed(1)} kW`
      }
      const lines = [await read('pulled')]
      const run = async (secs, label, stir) => {
        for (let s = 0; s < secs; s++) {
          for (let k = 0; k < 30; k++) {
            if (stir) {
              const b = e.ballPos
              e.stir = { centre: [b[0] - from, b[1], b[2]], vel: [speed, 0, 0] }
            }
            await e.advance(1)
          }
          lines.push(await read(label))
        }
      }
      await run(still, 'still', false)
      await run(stirS, 'stir', true)
      await run(after, 'after', false)
      return lines.join('\n')
    },
    [Number(o.still), Number(o.stir), Number(o.after), Number(o['stir-speed']), Number(o['stir-from'])],
  )
  console.log(out)
} finally {
  await browser.close()
}
