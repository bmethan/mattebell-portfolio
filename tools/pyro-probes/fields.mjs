// Heat maps of the gas: for each field, a vertical slice (front view, at one depth) and its extreme over the depth,
// at chosen moments. The quickest way to see what the solver is actually doing under a render.
import { PAGE_HELPERS, launch, list, openLab, options, outDir, runTo, saveBase64 } from './lib.mjs'

const o = options(
  {
    times: { type: 'string', default: '4,8,12' },
    fields: { type: 'string' },
    slice: { type: 'string' },
    scale: { type: 'string', default: '3' },
  },
  `
fields.mjs: heat maps of the gas fields.
  --times 4,8,12      simulated seconds to capture
  --fields soot,rise  fields to draw (fire: soot, rise, z, fuel, hrr; smoke: density, rise); default all
  --slice 16          the depth (cell index) of the slice; default the ball's, else the middle
  --scale 3           pixels per cell
Writes <out>/fields-<t>s.png: a row per field, the slice and then its extreme over the depth (the sum, for hrr).
Example: node tools/pyro-probes/fields.mjs --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-at 60 --times 120,140,150`,
)
const dir = outDir(o.out)
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  await page.evaluate(PAGE_HELPERS)
  for (const t of list(o.times)) {
    await runTo(page, t, o.scenario)
    const r = await page.evaluate(
      async ([want, slice, scale]) => {
        const e = window.__pyro
        const s = await e.readScalars()
        const fire = e.state.sourceType === 'fire'
        const [nx, ny, nz] = s.n
        const names = want ? want.split(',') : fire ? ['soot', 'rise', 'z', 'fuel', 'hrr'] : ['density', 'rise']
        const zs = slice != null ? Number(slice) : e.state.ball ? Math.round((e.ballPos[2] - -nz * s.dx * 0.5) / s.dx - 0.5) : nz >> 1
        const sl = {}, ex = {}
        for (const k of names) { sl[k] = new Float32Array(nx * ny); ex[k] = new Float32Array(nx * ny) }
        for (let z = 0; z < nz; z++)
          for (let y = 0; y < ny; y++)
            for (let x = 0; x < nx; x++) {
              const g = window.__probe.gas(s, (z * ny + y) * nx + x, fire)
              const j = y * nx + x
              for (const k of names) {
                const v = g[k]
                if (z === zs) sl[k][j] = v
                ex[k][j] = k === 'hrr' ? ex[k][j] + (v * s.dx ** 3) / 1000 : Math.max(ex[k][j], v)
              }
            }
        // Black, red, yellow, white, each row scaled to its own top (99.5th percentile of the two maps).
        const W = nx * scale, H = ny * scale, gap = 18
        const canvas = new OffscreenCanvas(2 * W + gap, names.length * (H + gap))
        const ctx = canvas.getContext('2d')
        ctx.fillStyle = '#141414'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        const tops = {}
        names.forEach((k, row) => {
          const all = [...sl[k], ...ex[k]].filter(v => v > 0).sort((a, b) => a - b)
          const top = all.length ? all[Math.floor(all.length * 0.995)] : 1
          tops[k] = top
          ;[sl[k], ex[k]].forEach((a, col) => {
            const img = ctx.createImageData(nx, ny)
            for (let y = 0; y < ny; y++)
              for (let x = 0; x < nx; x++) {
                const v = Math.min(Math.max(a[y * nx + x] / top, 0), 1)
                const p = ((ny - 1 - y) * nx + x) * 4
                img.data[p] = 255 * Math.min(3 * v, 1)
                img.data[p + 1] = 255 * Math.min(Math.max(3 * v - 1, 0), 1)
                img.data[p + 2] = 255 * Math.min(Math.max(3 * v - 2, 0), 1)
                img.data[p + 3] = 255
              }
            const tile = new OffscreenCanvas(nx, ny)
            tile.getContext('2d').putImageData(img, 0, 0)
            ctx.imageSmoothingEnabled = false
            ctx.drawImage(tile, col * (W + gap), row * (H + gap) + gap, W, H)
          })
          ctx.fillStyle = '#ffe680'
          ctx.font = '12px sans-serif'
          const unit = { soot: 'g/kg', rise: 'K', z: '', fuel: 'kg/kg', hrr: 'kW per column (projection), W/m3 (slice)', density: '' }[k]
          ctx.fillText(`${k} ${unit}: slice at depth ${zs} | extreme over the depth   (white = ${top.toPrecision(3)})`, 4, row * (H + gap) + 13)
        })
        const blob = await canvas.convertToBlob({ type: 'image/png' })
        const buf = new Uint8Array(await blob.arrayBuffer())
        let b = ''
        for (let i = 0; i < buf.length; i += 0x8000) b += String.fromCharCode(...buf.subarray(i, i + 0x8000))
        return { png: btoa(b), t: s.t, n: s.n, dx: s.dx, slice: zs, tops }
      },
      [o.fields ?? null, o.slice ?? null, Number(o.scale)],
    )
    saveBase64(`${dir}/fields-${t}s.png`, r.png)
    delete r.png
    console.log(`${t} s -> ${dir}/fields-${t}s.png`, JSON.stringify(r.tops))
  }
} finally {
  await browser.close()
}
