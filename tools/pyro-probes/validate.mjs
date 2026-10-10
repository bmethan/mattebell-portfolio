// The fire against fire science: for each case a fresh simulation, stepped exactly (3 steps per sample, 10 samples a
// simulated second), spun up, then sampled for the heat release actually released, the flame height (the height
// below which 99% of the time-averaged heat release happens, against Heskestad's L = 0.235 Q^(2/5) - 1.02 D; and
// where gas 500 or 700 K over ambient is present half the time), the time-averaged centreline temperature rise
// against McCaffrey's measured fits, and the puffing frequency (the strongest peaks of the centreline temperature
// and of the heat release, against Cetegen and Ahmed's f = 1.5 D^-1/2). The physics review page's results come from
// this with the defaults: node tools/pyro-probes/validate.mjs --cases '[{},{"heatRelease":40},{"heatRelease":150}]'
import fs from 'node:fs'
import { launch, openLab, options, outDir } from './lib.mjs'

const o = options(
  {
    cases: { type: 'string', default: '[{}]' },
    spin: { type: 'string', default: '4' },
    span: { type: 'string', default: '30' },
  },
  `
validate.mjs: the fire against McCaffrey, Heskestad and Cetegen and Ahmed.
  --cases '[{},{"heatRelease":40}]'   state patches, each over the validation fire (a uniform 0.6 m bed of wood gas)
  --spin 4 --span 30                  seconds of spin-up, then of sampling
Prints a JSON line per case and writes them all to <out>/validate.json. About 8 minutes a case.`,
)
// The validation fire: a uniform 0.6 m bed of wood gas with the calibrated turbulence (the lab's defaults may differ).
const BASE = { sourceType: 'fire', srcNoise: 0, srcRadius: 0.3, heatRelease: 80, sootYield: 0.015, radiantFraction: 0.3, turbulence: 1, turbScale: 0.2, vorticity: 0.35, buoyancy: 1, expansion: 1, cooling: 0, dissipation: 0, wind: 0, srcX: 0, velocity: 'mac', advection: 'mac', resolution: 'medium', ball: false }
const cases = JSON.parse(o.cases).map(c => ({ ...BASE, ...c }))
const spin = Number(o.spin)
const span = Number(o.span)
const dir = outDir(o.out)
const browser = await launch()
const results = []
try {
  for (const patch of cases) {
    const page = await openLab(browser, { preset: '', patch })
    const r = await page.evaluate(
      async ([spin, span]) => {
        const e = window.__pyro
        await e.advance(Math.round(spin * 30))
        const st = e.state
        const D = 2 * st.srcRadius * Math.sqrt(1 - 0.5 * st.srcNoise) // equivalent diameter of the bed's mean area
        const samples = Math.round(span * 10)
        const rise = d => (293 * Math.max(d, 0)) / Math.max(1 - d, 0.1)
        let n, dx
        const hrrLayer = [], tCol = [], series = []
        const THR = [500, 700] // flame intermittency: per layer, the share of samples in which some cell is this hot
        const inter = THR.map(() => [])
        let qSum = 0
        for (let i = 0; i < samples; i++) {
          await e.advance(3)
          const s = await e.readScalars()
          n = s.n
          dx = s.dx
          const [nx, ny, nz] = n
          qSum += s.sums[9]
          if (!hrrLayer.length) for (let y = 0; y < ny; y++) { hrrLayer.push(0); tCol.push(0); inter.forEach(a => a.push(0)) }
          for (let y = 0; y < ny; y++) {
            let hot = 0
            for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) hot = Math.max(hot, s.data[((z * ny + y) * nx + x) * 4 + 1])
            const r = rise(hot)
            THR.forEach((t, k) => { if (r >= t) inter[k][y] += 1 / samples })
          }
          const cx = Math.floor(nx / 2), cz = Math.floor(nz / 2)
          for (let y = 0; y < ny; y++) {
            let h = 0 // the heat release rate the step's burning recorded, W/m3, summed over the layer
            for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) h += s.hrr[(z * ny + y) * nx + x] * dx * dx * dx
            hrrLayer[y] += h / samples
            let t = 0 // centreline: the mean of the 3 x 3 cells around the axis
            for (let dz = -1; dz <= 1; dz++) for (let ddx = -1; ddx <= 1; ddx++) t += rise(s.data[(((cz + dz) * ny + y) * nx + cx + ddx) * 4 + 1])
            tCol[y] += t / 9 / samples
          }
          // puffing signals: the centreline temperature averaged over heights D/2 to 3D/2, and the total heat release
          let t = 0, cnt = 0
          for (let y = Math.round(D / 2 / dx); y <= Math.min(ny - 1, Math.round((1.5 * D) / dx)); y++)
            for (let dz = -1; dz <= 1; dz++) for (let ddx = -1; ddx <= 1; ddx++) { t += rise(s.data[(((cz + dz) * ny + y) * nx + cx + ddx) * 4 + 1]); cnt++ }
          let qt = 0
          for (let k = 0; k < s.hrr.length; k++) qt += s.hrr[k]
          series.push([t / cnt, qt])
        }
        return { n, dx, D, Q: qSum / samples / 1000, hrrLayer, tCol, series, inter }
      },
      [spin, span],
    )
    const total = r.hrrLayer.reduce((a, b) => a + b, 0)
    const quant = q => {
      let acc = 0
      for (let y = 0; y < r.hrrLayer.length; y++) {
        acc += r.hrrLayer[y]
        if (acc >= q * total) return +((y + 1) * r.dx).toFixed(2)
      }
      return null
    }
    const L99 = quant(0.99)
    // the height where the flame is present half the time (between layers, linearly)
    const halfHeight = a => {
      for (let y = 1; y < a.length; y++) if (a[y] < 0.5 && a[y - 1] >= 0.5) return +((y - 0.5 + (a[y - 1] - 0.5) / (a[y - 1] - a[y]) - 0.5) * r.dx).toFixed(2)
      return null
    }
    const Qk = r.Q
    const Lh = 0.235 * Qk ** 0.4 - 1.02 * r.D
    // McCaffrey's measured centreline temperature rise (his fits): 797 K in the flame, then 62.9 (z/Q^0.4)^-1 and
    // 21.6 (z/Q^0.4)^-5/3 in the intermittent region and the plume
    const mcc = zs => {
      const x = zs / Qk ** 0.4
      return x < 0.08 ? 797 : x < 0.2 ? 62.9 / x : 21.6 * x ** (-5 / 3)
    }
    const prof = []
    for (let y = 1; y < r.tCol.length; y += 4) {
      const z = (y + 0.5) * r.dx
      prof.push([+z.toFixed(2), +(z / Qk ** 0.4).toFixed(3), Math.round(r.tCol[y]), Math.round(mcc(z))])
    }
    const peaksOf = sig => {
      const m = sig.reduce((a, b) => a + b, 0) / sig.length
      const xs = sig.map(v => v - m)
      const spec = []
      for (let f = 0.5; f <= 4; f += 0.05) {
        let re = 0, im = 0
        xs.forEach((v, i) => { re += v * Math.cos((2 * Math.PI * f * i) / 10); im -= v * Math.sin((2 * Math.PI * f * i) / 10) })
        spec.push([f, re * re + im * im])
      }
      return spec.filter((p, i) => i > 0 && i < spec.length - 1 && p[1] > spec[i - 1][1] && p[1] >= spec[i + 1][1]).sort((a, b) => b[1] - a[1])
    }
    const pT = peaksOf(r.series.map(v => v[0]))
    const pQ = peaksOf(r.series.map(v => v[1]))
    const res = {
      case: patch,
      D: +r.D.toFixed(3),
      Q_kW: +Qk.toFixed(1),
      flameHeight50pct_dT500_dT700: r.inter.map(halfHeight),
      flameHeight_m: L99,
      hrrHeights_50_90_99: [quant(0.5), quant(0.9), L99],
      heskestad_m: +Lh.toFixed(2),
      puff_Hz: +(pT[0] ?? [0])[0].toFixed(2),
      puffPeaks_Hz: { temperature: pT.slice(0, 3).map(p => +p[0].toFixed(2)), heatRelease: pQ.slice(0, 3).map(p => +p[0].toFixed(2)) },
      cetegenAhmed_Hz: +(1.5 / Math.sqrt(r.D)).toFixed(2),
      zukoski_Hz: +(1.57 / Math.sqrt(r.D)).toFixed(2),
      profile_z_zQ_dT_mccaffrey: prof,
    }
    results.push(res)
    console.log(JSON.stringify(res))
    await page.close()
  }
  fs.writeFileSync(`${dir}/validate.json`, JSON.stringify(results, null, 1))
} finally {
  await browser.close()
}
