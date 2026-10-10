// Rendered stills at chosen moments, each settled (the frame accumulated as a paused frame does), optionally one per
// solver step for a clip (--clip seconds). Same scenario options as the other probes, so a still can be taken of
// exactly the run a measurement came from.
import { launch, list, openLab, options, outDir, runTo, saveBase64 } from './lib.mjs'

const o = options(
  {
    times: { type: 'string', default: '4,8,12' },
    size: { type: 'string', default: '1184,497' },
    clip: { type: 'string' },
    tag: { type: 'string', default: 'still' },
  },
  `
stills.mjs: rendered frames.
  --times 30,90,150   simulated seconds to render
  --size 1184,497     frame size in pixels
  --clip 12           instead, every step from the start to this second, as <out>/<tag>-0001.jpg... (30 a second)
  --tag still         file names: <out>/<tag>-<t>s.png
Example: node tools/pyro-probes/stills.mjs --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-at 60 --times 30,90,150,220`,
)
const dir = outDir(o.out)
const [w, h] = list(o.size)
const shot = (page, type) =>
  page.evaluate(
    async ([w, h, type]) => {
      const blob = await window.__pyro.snapshot(w, h, type, 0.95)
      const buf = new Uint8Array(await blob.arrayBuffer())
      let b = ''
      for (let i = 0; i < buf.length; i += 0x8000) b += String.fromCharCode(...buf.subarray(i, i + 0x8000))
      return btoa(b)
    },
    [w, h, type],
  )
const browser = await launch()
try {
  const page = await openLab(browser, o.scenario)
  if (o.clip) {
    const frames = Math.round(Number(o.clip) * 30)
    for (let f = 1; f <= frames; f++) {
      await runTo(page, f / 30, o.scenario)
      saveBase64(`${dir}/${o.tag}-${String(f).padStart(4, '0')}.jpg`, await shot(page, 'image/jpeg'))
      if (f % 60 === 0) console.log(`frame ${f} of ${frames}`)
    }
  } else {
    for (const t of list(o.times)) {
      await runTo(page, t, o.scenario)
      saveBase64(`${dir}/${o.tag}-${t}s.png`, await shot(page, 'image/png'))
      const ball = await page.evaluate(async () => {
        const e = window.__pyro
        if (!e.state.ball) return null
        await e.readBall()
        await new Promise(r => setTimeout(r, 150))
        return e.ballStatus
      })
      console.log(`${t} s -> ${dir}/${o.tag}-${t}s.png`, ball ? JSON.stringify(ball) : '')
    }
  }
} finally {
  await browser.close()
}
