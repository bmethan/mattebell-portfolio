// One cell and its six neighbours, step by step: the raw scalars, the gas state, the fire's mass-keeping ratio and
// the velocity. For watching a misbehaving cell (pick it with hotcells.mjs) to the step where it goes wrong.
import { PAGE_HELPERS, launch, list, openLab, options, runTo } from './lib.mjs'

const o = options(
  {
    cell: { type: 'string' },
    from: { type: 'string', default: '0' },
    steps: { type: 'string', default: '30' },
    every: { type: 'string', default: '1' },
    field: { type: 'string', default: 'soot' },
  },
  `
cellprobe.mjs: a cell and its neighbours, step by step.
  --cell 116,16,22    the cell (indices; from hotcells.mjs)
  --from 136          simulated second to start at
  --steps 60 --every 3   how many solver steps to watch, and how many between readings
  --field soot        the field shown for the neighbours
Each line: time, the cell's state, its mass-keeping ratio (fire; negative where the gas expands), its velocity, and
the neighbours (-x +x -y +y -z +z) as field/ratio.
Example: node tools/pyro-probes/cellprobe.mjs --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-at 60 --cell 116,16,22 --from 136 --steps 60 --every 3`,
)
if (!o.cell) throw new Error('--cell x,y,z is required (see --help)')
const cell = list(o.cell)
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  await page.evaluate(PAGE_HELPERS)
  await runTo(page, Number(o.from), o.scenario)
  const every = Number(o.every)
  for (let k = 0; k <= Number(o.steps); k += every) {
    const r = await page.evaluate(
      async ([c, every, k, field]) => {
        const e = window.__pyro
        if (k > 0) await e.advance(every)
        const s = await e.readScalars()
        const fire = e.state.sourceType === 'fire'
        const ratio = fire ? await e.readRatio() : null
        const f = await e.readFields()
        const [nx, ny] = s.n
        const idx = (x, y, z) => (z * ny + y) * nx + x
        const round = q => (Math.abs(q) >= 100 ? Math.round(q) : +q.toPrecision(3))
        const at = (x, y, z) => {
          const i = idx(x, y, z)
          const g = window.__probe.gas(s, i, fire)
          return { g: Object.fromEntries(Object.entries(g).map(([n, q]) => [n, round(q)])), r: ratio ? +ratio[i].toFixed(3) : null }
        }
        const i0 = idx(...c)
        const here = at(...c)
        const nb = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]].map(o => {
          const q = at(c[0] + o[0], c[1] + o[1], c[2] + o[2])
          return `${q.g[field]}${q.r != null ? '/' + q.r : ''}`
        })
        return { t: +s.t.toFixed(2), here, vel: [0, 1, 2].map(j => +f.vel[i0 * 3 + j].toFixed(3)), nb }
      },
      [cell, every, k, o.field],
    )
    console.log(r.t, JSON.stringify(r.here.g), 'ratio', r.here.r, 'vel', JSON.stringify(r.vel), `neighbours ${o.field}${r.here.r != null ? '/ratio' : ''}:`, r.nb.join(' '))
  }
} finally {
  await browser.close()
}
