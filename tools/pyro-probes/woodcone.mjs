// The wooden ball's wood under a fixed exposure, as fire scientists test wood: a radiant flux on the whole ball in still
// air as a cone calorimeter heats a sample, flaming (its flame taking the oxygen) or with its flame out (the char in
// air: air12.4), or the standard furnace fire (ISO 834). The wood's clock runs fast (the
// engine's woodSpeed, development only), and one column is read as it goes: when it first gives off enough fuel to
// flame, its surface temperature, how deep the char line (the 300 C isotherm) has gone, the char's own depth, and the
// temperature below the char line, against which Janssens and White's profile, 20 + 280 (1 - x/a)^2, gives a.
// Measured, for southern pine at 8 to 9% moisture (Tran and White 1992; White and Dietenberger, Wood Handbook ch. 18):
// 0.38 to 0.49 mm/min at 17.4 kW/m2, 0.54 at 25.2, 0.66 to 0.75 at 38.5, 0.77 to 0.85 at 55.5 (averaged to 36 mm), a
// about 38 mm at 18 kW/m2 and 26.5 at 55; in the furnace about 0.8 mm/min (0.65 in Eurocode 5) and a about 33 mm.
import { launch, openLab, options } from './lib.mjs'

const o = options(
  {
    cases: { type: 'string', default: '18,25,38,55,iso' },
    minutes: { type: 'string', default: '40' },
    speed: { type: 'string', default: '30' },
    every: { type: 'string', default: '5' },
    flame: { type: 'string', default: '0' },
    radius: { type: 'string', default: '0.22' },
    dir: { type: 'string', default: '0,0,1' },
    after: { type: 'string', default: '12.4' },
    'air-speed': { type: 'string', default: '0' },
  },
  `
woodcone.mjs: the ball's wood under a fixed exposure.
  --cases 18,25,38,55,iso   radiant fluxes (kW/m2, the sample flaming), air<flux> (its flame out), iso (the furnace),
                            chatani (as Chatani and Harada: flaming at 20 kW/m2 for 12 min, then 12.4 with the flame out)
  --after 12.4              the chatani case's flux once its flame is out (they measured 12.4 and 8.4)
  --air-speed 0             m/s of air blown over the sample with its flame out (air cases, and chatani's second part)
  --minutes 40              how long, in the wood's own time
  --speed 30                how many times faster than the gas the wood's clock runs
  --every 5                 minutes between readings
  --flame 0                 kW/m2 the sample's own flame adds while it flames (cone cases): while it gives off 1 g/m2/s,
                            the critical mass flux for flaming at 20 to 30 kW/m2 (McAllister and colleagues 2010)
  --radius 0.22             the ball's radius (m)
  --dir 0,0,1               the column read: the surface facing this way from the ball's centre
Example: node tools/pyro-probes/woodcone.mjs --cases 25,55 --minutes 30`,
)
const cases = o.cases.split(',')
const browser = await launch()
try {
  for (const c of cases) {
    const page = await openLab(browser, { preset: 'Smoke plume', patch: { resolution: 'low', ballRadius: Number(o.radius), ballBurns: true }, ballAt: [1.4, 0.7], pullAt: null, pullTo: null })
    const out = await page.evaluate(
      async ([c, minutes, speed, every, flame, dir, after, airSpeed]) => {
        const e = window.__pyro
        const iso = c === 'iso'
        const chatani = c === 'chatani'
        let air = c.startsWith('air')
        let flux = iso ? 0 : chatani ? 20000 : Number(air ? c.slice(3) : c) * 1000
        e.woodSpeed = speed
        e.woodTest = iso ? -1 : air ? -flux : flux
        e.woodAir = air ? airSpeed : 0
        e.newBall()
        const rows = []
        let flamedAt = null
        let at644 = null
        let reached36 = null
        // Wood seconds per step of the gas.
        const perStep = speed / 30
        const readEvery = Math.max(1, Math.round(10 / perStep))
        let nextRow = 0
        for (;;) {
          await e.advance(readEvery)
          const w = await e.readWood(dir)
          const tMin = w.clock / 60
          const [ts, fuel, charMm] = w.skin
          if (flamedAt === null && fuel >= 0.002) flamedAt = w.clock
          if (at644 === null && ts >= 644) at644 = w.clock
          if (chatani && !air && w.clock >= 720) {
            air = true
            flux = after * 1000
            e.woodTest = -flux
            e.woodAir = airSpeed
          }
          if (!iso && !air) e.woodTest = flux + (fuel >= 0.001 ? flame * 1000 : 0)
          // The char line: the 300 C isotherm, between cell centres, by depth below the surface as made.
          const cells = w.cells.filter(x => x.T > 1)
          const mid = cells.map(x => ({ d: x.top + x.thick / 2, T: x.T }))
          let line = 0
          for (let i = 0; i + 1 < mid.length; i++) {
            if (mid[i].T >= 573 && mid[i + 1].T < 573) {
              line = mid[i].d + ((mid[i].T - 573) / (mid[i].T - mid[i + 1].T)) * (mid[i + 1].d - mid[i].d)
              break
            }
          }
          if (mid.length && mid[0].T < 573) line = 0
          if (reached36 === null && line >= 36) reached36 = w.clock
          // The profile below it: where the temperature falls to 40 C, (1 - x/a)^2 = 20/280, so a = x / 0.7327.
          let t40 = null
          for (let i = 0; i + 1 < mid.length; i++) {
            if (mid[i].d > line && mid[i].T >= 313 && mid[i + 1].T < 313) {
              t40 = mid[i].d + ((mid[i].T - 313) / (mid[i].T - mid[i + 1].T)) * (mid[i + 1].d - mid[i].d)
              break
            }
          }
          const at6 = (() => {
            const d = line + 6
            for (let i = 0; i + 1 < mid.length; i++) if (mid[i].d <= d && mid[i + 1].d > d) return mid[i].T + ((d - mid[i].d) / (mid[i + 1].d - mid[i].d)) * (mid[i + 1].T - mid[i].T)
            return null
          })()
          if (tMin >= nextRow - 1e-9) {
            rows.push({
              min: +tMin.toFixed(1),
              surfC: Math.round(ts - 273),
              fuel: +(fuel * 1000).toFixed(2),
              lineMm: +line.toFixed(1),
              rate: tMin > 0 ? +(line / tMin).toFixed(2) : 0,
              charMm: +charMm.toFixed(1),
              wearMm: +w.sheet[1].toFixed(3),
              density: +w.density.toFixed(3),
              at6mmC: at6 === null ? null : Math.round(at6 - 273),
              a: t40 === null || line === 0 ? null : +((t40 - line) / 0.7327).toFixed(1),
            })
            nextRow += every
          }
          if (tMin >= minutes) break
        }
        return { flamedAt, at644, reached36, rows }
      },
      [c, Number(o.minutes), Number(o.speed), Number(o.every), Number(o.flame), o.dir.split(',').map(Number), Number(o.after), Number(o['air-speed'])],
    )
    const label = c === 'iso' ? 'ISO 834 furnace' : c === 'chatani' ? `Chatani and Harada (20 kW/m2 flaming 12 min, then ${o.after} flame out${Number(o['air-speed']) ? `, air at ${o['air-speed']} m/s` : ''})` : c.startsWith('air') ? `${c.slice(3)} kW/m2, flame out` : `${c} kW/m2 flaming${Number(o.flame) ? ` (+${o.flame} its flame)` : ''}`
    const avg36 = out.reached36 ? (36 / (out.reached36 / 60)).toFixed(2) : 'not reached'
    console.log(`${label}: flaming fuel (2 g/m2/s) at ${out.flamedAt?.toFixed(0) ?? 'never'} s, surface 644 K at ${out.at644?.toFixed(0) ?? 'never'} s, char line to 36 mm: ${avg36} mm/min`)
    console.log(`  (the column read: its wood ${out.rows[0]?.density} times the mean density)`)
    for (const r of out.rows) console.log(`  ${r.min} min  surface ${r.surfC} C  fuel ${r.fuel} g/m2/s  char line ${r.lineMm} mm (${r.rate} mm/min)  char ${r.charMm} mm  worn ${r.wearMm} mm  6 mm under the line ${r.at6mmC} C  a ${r.a} mm`)
    await page.close()
  }
} finally {
  await browser.close()
}
