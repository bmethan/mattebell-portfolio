// The cells where a field is highest, and where they are: how far from the collision ball's surface, and the rest of
// the gas's state there. The tool that found soot running away in still air beside the ball (2026-10-09).
import { PAGE_HELPERS, launch, list, openLab, options, runTo } from './lib.mjs'

const o = options(
  {
    times: { type: 'string', default: '4,8,12' },
    field: { type: 'string', default: 'soot' },
    top: { type: 'string', default: '6' },
    over: { type: 'string', default: '0' },
  },
  `
hotcells.mjs: the highest cells of one field, with their place and state.
  --times 120,140,150   simulated seconds to look at
  --field soot          fire: soot (g/kg), rise (K), z, fuel, hrr (W/m3); smoke: density, rise
  --top 6               how many cells to list
  --over 5              also count the cells above this value
Example: node tools/pyro-probes/hotcells.mjs --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-at 60 --times 130,138,141`,
)
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  await page.evaluate(PAGE_HELPERS)
  for (const t of list(o.times)) {
    await runTo(page, t, o.scenario)
    const r = await page.evaluate(
      async ([field, top, over]) => {
        const e = window.__pyro
        const s = await e.readScalars()
        const fire = e.state.sourceType === 'fire'
        const [nx, ny, nz] = s.n
        const cells = []
        let count = 0
        for (let z = 0; z < nz; z++)
          for (let y = 0; y < ny; y++)
            for (let x = 0; x < nx; x++) {
              const g = window.__probe.gas(s, (z * ny + y) * nx + x, fire)
              const v = g[field]
              if (v > over) count++
              if (cells.length < top || v > cells[cells.length - 1].v) {
                const w = window.__probe.world(s, x, y, z)
                const off = e.state.ball ? Math.hypot(w[0] - e.ballPos[0], w[1] - e.ballPos[1], w[2] - e.ballPos[2]) - e.state.ballRadius : null
                const round = q => (Math.abs(q) >= 100 ? Math.round(q) : +q.toPrecision(3))
                cells.push({ v, cell: [x, y, z], at: w.map(q => +q.toFixed(2)), fromBallCm: off == null ? null : +(off * 100).toFixed(1), ...Object.fromEntries(Object.entries(g).map(([k, q]) => [k, round(q)])) })
                cells.sort((a, b) => b.v - a.v)
                if (cells.length > top) cells.pop()
              }
            }
        return { t: +s.t.toFixed(2), count, cells: cells.map(({ v, ...c }) => c) }
      },
      [o.field, Number(o.top), Number(o.over)],
    )
    console.log(`${r.t} s: ${r.count} cells with ${o.field} over ${o.over}`)
    for (const c of r.cells) console.log('   ', JSON.stringify(c))
  }
} finally {
  await browser.close()
}
