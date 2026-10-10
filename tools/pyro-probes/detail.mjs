// How much detail a smoke run keeps, at chosen moments: the smoke's sharpness (mean |grad density| over the mean
// density where there is smoke), the flow's enstrophy (sum |curl u|^2 dV) and kinetic energy, where the smoke sits
// (its mean height, rise speed and spread from the axis), and a still of each moment. The numbers behind the solver
// comparisons on the physics review page (BiMocq2, IVOCK, leapfrog flow maps).
import { launch, list, openLab, options, outDir, runTo, saveBase64 } from './lib.mjs'

const o = options(
  { times: { type: 'string', default: '4,8,12' }, tag: { type: 'string', default: 'run' } },
  `
detail.mjs: smoke detail metrics and stills.
  --times 4,8,12    simulated seconds to measure
  --tag run         name for the stills, <out>/detail-<tag>-<t>s.png
Example: node tools/pyro-probes/detail.mjs --patch '{"velocity":"lfm"}' --tag lfm`,
)
const dir = outDir(o.out)
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  for (const t of list(o.times)) {
    await runTo(page, t, o.scenario)
    const r = await page.evaluate(async () => {
      const e = window.__pyro
      const f = await e.readFields()
      const [nx, ny, nz] = f.n
      const dx = f.dx
      const idx = (x, y, z) => (z * ny + y) * nx + x
      let gsum = 0, dsum = 0, cnt = 0, ens = 0, ke = 0, total = 0, dy = 0, dvy = 0, dr = 0
      for (let z = 1; z < nz - 1; z++)
        for (let y = 1; y < ny - 1; y++)
          for (let x = 1; x < nx - 1; x++) {
            const i = idx(x, y, z)
            const d = f.density[i]
            total += d * dx ** 3
            dy += d * (y + 0.5) * dx
            dvy += d * f.vel[i * 3 + 1]
            dr += d * Math.hypot((x + 0.5) * dx - 0.5 * nx * dx, (z + 0.5) * dx - 0.5 * nz * dx)
            const v = k => f.vel[i * 3 + k]
            ke += 0.5 * (v(0) ** 2 + v(1) ** 2 + v(2) ** 2) * dx ** 3
            const V = (xx, yy, zz, k) => f.vel[idx(xx, yy, zz) * 3 + k]
            const wx = (V(x, y + 1, z, 2) - V(x, y - 1, z, 2) - V(x, y, z + 1, 1) + V(x, y, z - 1, 1)) / (2 * dx)
            const wy = (V(x, y, z + 1, 0) - V(x, y, z - 1, 0) - V(x + 1, y, z, 2) + V(x - 1, y, z, 2)) / (2 * dx)
            const wz = (V(x + 1, y, z, 1) - V(x - 1, y, z, 1) - V(x, y + 1, z, 0) + V(x, y - 1, z, 0)) / (2 * dx)
            ens += (wx * wx + wy * wy + wz * wz) * dx ** 3
            if (d > 0.05) {
              const gx = (f.density[idx(x + 1, y, z)] - f.density[idx(x - 1, y, z)]) / (2 * dx)
              const gy = (f.density[idx(x, y + 1, z)] - f.density[idx(x, y - 1, z)]) / (2 * dx)
              const gz = (f.density[idx(x, y, z + 1)] - f.density[idx(x, y, z - 1)]) / (2 * dx)
              gsum += Math.hypot(gx, gy, gz)
              dsum += d
              cnt++
            }
          }
      const cells = total / dx ** 3
      const blob = await e.snapshot(1184, 497, 'image/png')
      const buf = new Uint8Array(await blob.arrayBuffer())
      let b = ''
      for (let i = 0; i < buf.length; i += 0x8000) b += String.fromCharCode(...buf.subarray(i, i + 0x8000))
      return { sharpness_per_m: +(gsum / Math.max(dsum, 1e-9)).toFixed(2), smokeCells: cnt, smoke: +total.toFixed(3), enstrophy: +ens.toFixed(3), ke: +ke.toFixed(3), height_m: +(dy / cells).toFixed(3), rise_ms: +(dvy / cells).toFixed(3), spread_m: +(dr / cells).toFixed(3), png: btoa(b) }
    })
    saveBase64(`${dir}/detail-${o.tag}-${t}s.png`, r.png)
    delete r.png
    console.log(JSON.stringify({ tag: o.tag, t, ...r }))
  }
} finally {
  await browser.close()
}
