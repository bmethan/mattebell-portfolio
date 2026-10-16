// (Prototype, for the full-resolution plan.) The running lab as a visitor sees it, and what each way of drawing it
// sharper costs. Four modes:
//   emulate  the running view exactly (the engine's runView: frames of a step and a render each, the running blend),
//            for each configuration, at one simulated moment, beside the settled full-size still of the same moment
//            (no motion blur) as the reference; and a second still a third of a second later, to tell what is still
//            from what moves.
//   live     the lab as a visitor runs it, in real time: screenshots of the frame, and its render scale, canvas, march,
//            frames and steps a second.
//   cost     GPU time of the image pass and the passes after it (the engine's profileStep), for each configuration.
//   compile  how long the image pass and the two resolves take to compile, each on its own, on a fresh device.
// A configuration is today (the canvas at the render scale) or up (the prototype's temporal upsampling: the canvas at
// the display's pixels, the march at the render scale), with the render scale pinned: today:0.5, up:0.5, today:1.
//   loop     frames for a pre-rendered loop: settled full-size frames, one a step, as the stills are.
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'
import { LAB_URL, launch, list, openLab, options, outDir, runTo, saveBase64 } from './lib.mjs'

const o = options(
  {
    mode: { type: 'string', default: 'emulate' },
    window: { type: 'string', default: '1600,765,1' },
    at: { type: 'string', default: '10' },
    frames: { type: 'string', default: '45' },
    configs: { type: 'string', default: 'today:0.5,up:0.5' },
    seconds: { type: 'string', default: '14' },
    runs: { type: 'string', default: '7' },
    tag: { type: 'string', default: 'envy' },
    size: { type: 'string', default: '1829,765' },
    fade: { type: 'string', default: '1' },
  },
  `
fullres.mjs: the running view and its cost, today and with the prototype's temporal upsampling.
  --mode emulate|live|cost|compile|loop
  --window 1600,765,1     the window's CSS width and height and its device pixels per CSS pixel
  --at 10                 simulated second to capture or profile at (emulate, cost)
  --frames 45             running frames drawn before the capture (emulate), a step each
  --configs today:0.5,up:0.5   what to compare (today or up, with the render scale pinned, "auto" leaving it adapting; up
                          takes a third value, the samples a running pixel may count, 8 by default, and a fourth, the
                          narrow gather's 1 / (2 sigma^2), 3 by default: up:0.5:12:2)
  --seconds 14            how long to let it run before the screenshots (live)
  --runs 7                profiles to take the median of (cost)
  --tag envy              file names: <out>/<mode>-<tag>-<config>.png
  --size 1829,765 --fade 1   (loop) frame size, and seconds rendered on past --seconds for the closing cross-fade
Example: node tools/pyro-probes/fullres.mjs --mode emulate --configs today:0.5,up:0.5,today:1 --preset "Bonfire at dusk"`,
)
const [ww, wh, dpr] = list(o.window)
const view = { width: ww, height: wh, dpr }
const dir = outDir(o.out)
const configs = o.configs.split(',').map(c => {
  const [kind, scale, most, sharp] = c.split(':')
  return { name: c.replaceAll(':', '-'), up: kind === 'up', scale: scale === 'auto' ? null : Number(scale), most: most ? Number(most) : 8, sharp: sharp ? Number(sharp) : 3 }
})
const log = []
const note = line => {
  console.log(line)
  log.push(line)
}
const toB64 = `async blob => {
  const buf = new Uint8Array(await blob.arrayBuffer())
  let b = ''
  for (let i = 0; i < buf.length; i += 0x8000) b += String.fromCharCode(...buf.subarray(i, i + 0x8000))
  return btoa(b)
}`
const sizes = page =>
  page.evaluate(() => {
    const e = window.__pyro
    return { canvas: [e.canvas.width, e.canvas.height], march: [e.marchW, e.marchH], css: [e.cssW, e.cssH], scale: +e.renderScale.toFixed(3) }
  })

const browser = await launch()
try {
  if (o.mode === 'emulate') {
    const page = await openLab(browser, o.scenario, view)
    const at = Number(o.at)
    const frames = Number(o.frames)
    for (const c of configs) {
      await page.evaluate(([up, scale, most, sharp]) => {
        const e = window.__pyro
        e.upsample = up
        e.pinScale = scale
        e.upMost = most
        e.upSharp = sharp
        e.restart()
      }, [c.up, c.scale, c.most, c.sharp])
      await page.waitForTimeout(300)
      await runTo(page, at - frames / 30, o.scenario)
      const b64 = await page.evaluate(async ([frames, toB64]) => (0, eval)(`(${toB64})`)(await window.__pyro.runView(frames, 1)), [frames, toB64])
      const v = await page.evaluate(() => ({ ...window.__pyro.lastView, now: window.__pyro.simTime }))
      saveBase64(`${dir}/emulate-${o.tag}-${c.name}.png`, b64)
      note(`${c.name}: at ${v.t.toFixed(3)} s (now ${v.now.toFixed(3)}), canvas ${v.canvas.join('x')}, march ${v.march.join('x')}, metered EV ${v.ev.toFixed(2)}`)
    }
    // The references: settled, full size (the frame's device pixels), no motion blur; then a third of a second on.
    const ref = async name => {
      const b64 = await page.evaluate(async toB64 => {
        const e = window.__pyro
        e.upsample = false
        e.pinScale = null
        const w = Math.round(e.cssW * e.dpr)
        const h = Math.round(e.cssH * e.dpr)
        return (0, eval)(`(${toB64})`)(await e.snapshot(w, h, 'image/png'))
      }, toB64)
      saveBase64(`${dir}/emulate-${o.tag}-${name}.png`, b64)
    }
    await page.evaluate(() => window.__pyro.setState({ shutter: 0 }))
    await ref('reference')
    // Today's paused view: settled, but at the half render scale it ran at (the canvas a paused frame keeps).
    const paused = await page.evaluate(async toB64 => {
      const e = window.__pyro
      const w = Math.round(e.cssW * Math.min(e.dpr, 1.5) * 0.5)
      const h = Math.round(e.cssH * Math.min(e.dpr, 1.5) * 0.5)
      return (0, eval)(`(${toB64})`)(await e.snapshot(w, h, 'image/png'))
    }, toB64)
    saveBase64(`${dir}/emulate-${o.tag}-paused-today.png`, paused)
    await page.evaluate(() => window.__pyro.advance(10))
    await ref('reference-later')
    note(`references at ${(await page.evaluate(() => window.__pyro.simTime)) - 1 / 3} s and a third of a second on`)
  } else if (o.mode === 'live') {
    for (const c of configs) {
      const page = await browser.newPage({ viewport: { width: ww, height: wh }, deviceScaleFactor: dpr })
      page.on('pageerror', e => console.error('page exception:', e.message))
      await page.goto(LAB_URL, { waitUntil: 'load', timeout: 120000 })
      const frame = page.locator('#lab-panel-pyro .lab-frame')
      await frame.scrollIntoViewIfNeeded()
      await page.waitForFunction(() => window.__pyro && document.querySelector('#lab-panel-pyro .lab-hud-right'), null, { timeout: 90000 })
      const t0 = Date.now()
      await page.evaluate(([up, scale, most, sharp]) => {
        const e = window.__pyro
        e.upsample = up
        e.pinScale = scale
        e.upMost = most
        e.upSharp = sharp
        e.resize(e.cssW, e.cssH, e.dpr)
        // Frames a second, counted from the page's own animation frames.
        window.__frames = []
        const tick = now => {
          window.__frames.push(now)
          requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
      }, [c.up, c.scale, c.most, c.sharp])
      const reads = []
      for (let s = 0; s < Number(o.seconds); s++) {
        await page.waitForTimeout(1000)
        reads.push(
          await page.evaluate(() => {
            const e = window.__pyro
            const now = performance.now()
            const fps = window.__frames.filter(t => now - t <= 1000).length
            return { sim: +e.simTime.toFixed(1), steps: document.querySelector('#lab-panel-pyro .lab-hud-right').textContent.trim(), scale: +e.renderScale.toFixed(3), canvas: `${e.canvas.width}x${e.canvas.height}`, march: `${e.marchW}x${e.marchH}`, fps }
          }),
        )
      }
      await frame.screenshot({ path: `${dir}/live-${o.tag}-${c.name}.png` })
      await page.waitForTimeout(500)
      await frame.screenshot({ path: `${dir}/live-${o.tag}-${c.name}-b.png` })
      note(`${c.name} (live, ${((Date.now() - t0) / 1000).toFixed(0)} s after the first frame):`)
      for (const r of reads) note(`  ${JSON.stringify(r)}`)
      await page.close()
    }
  } else if (o.mode === 'cost') {
    const page = await openLab(browser, o.scenario, view)
    await runTo(page, Number(o.at), o.scenario)
    for (const c of configs) {
      const runs = []
      for (let i = 0; i < Number(o.runs); i++) {
        runs.push(
          await page.evaluate(async ([up, scale]) => {
            const e = window.__pyro
            e.upsample = up
            e.pinScale = scale
            // (The running sizes, while it stays paused.)
            e.paused = false
            e.sizeTo()
            e.paused = true
            return e.profileStep()
          }, [c.up, c.scale]),
        )
      }
      const sz = await sizes(page)
      const med = {}
      for (const k of ['render', 'accumulate', 'glare', 'display']) {
        const v = runs.map(r => r[k] ?? 0).sort((a, b) => a - b)
        med[k] = +v[v.length >> 1].toFixed(2)
      }
      const frame = med.render + med.accumulate + med.glare + med.display
      note(`${c.name}: canvas ${sz.canvas.join('x')}, march ${sz.march.join('x')}: ${JSON.stringify(med)} frame ${frame.toFixed(2)} ms`)
    }
    const step = await page.evaluate(() => window.__pyro.profileStep())
    const stepMs = Object.entries(step)
      .filter(([k]) => !['render', 'accumulate', 'glare', 'display'].includes(k))
      .reduce((a, [, v]) => a + v, 0)
    note(`the step's own kernels (one profile): ${stepMs.toFixed(1)} ms`)
  } else if (o.mode === 'loop') {
    // Frames for a pre-rendered loop: from --at, every step for --seconds plus --fade (rendered on, for the cross-fade
    // that closes the loop), each settled at --size as the stills are (motion blur included).
    const [w, h] = list(o.size)
    const page = await openLab(browser, o.scenario, view)
    await runTo(page, Number(o.at), o.scenario)
    const n = Math.round((Number(o.seconds) + Number(o.fade)) * 30)
    const t0 = Date.now()
    for (let f = 0; f < n; f++) {
      if (f > 0) await page.evaluate(() => window.__pyro.advance(1))
      const b64 = await page.evaluate(async ([w, h, toB64]) => (0, eval)(`(${toB64})`)(await window.__pyro.snapshot(w, h, 'image/png')), [w, h, toB64])
      saveBase64(`${dir}/loop-${String(f).padStart(4, '0')}.png`, b64)
      if (f % 30 === 29) console.log(`frame ${f + 1} of ${n}, ${((Date.now() - t0) / (f + 1) / 1000).toFixed(2)} s a frame`)
    }
    note(`${n} frames at ${w}x${h} from ${o.at} s: ${((Date.now() - t0) / 1000).toFixed(0)} s, ${((Date.now() - t0) / n / 1000).toFixed(2)} s a frame`)
  } else if (o.mode === 'compile') {
    // The shaders' source, from wgsl.ts and what it imports, transpiled into the output folder.
    const require = createRequire(import.meta.url)
    const ts = require('typescript')
    const root = path.resolve('components')
    const out = path.resolve(dir, 'wgsl-src')
    const done = new Set()
    const emit = file => {
      if (done.has(file)) return
      done.add(file)
      let src = fs.readFileSync(file, 'utf8')
      const js = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText
      const rel = path.relative(root, file).replace(/\.tsx?$/, '.mjs')
      const fixed = js.replace(/from '(\.[^']+)'/g, (m, spec) => {
        const base = path.resolve(path.dirname(file), spec)
        const found = ['.ts', '.tsx'].map(x => base + x).find(f => fs.existsSync(f))
        if (!found) return m
        emit(found)
        return `from '${spec}.mjs'`
      })
      fs.mkdirSync(path.dirname(path.join(out, rel)), { recursive: true })
      fs.writeFileSync(path.join(out, rel), fixed)
    }
    emit(path.join(root, 'pyro', 'wgsl.ts'))
    const W = await import(pathToFileURL(path.join(out, 'pyro', 'wgsl.mjs')).href)
    const page = await browser.newPage()
    // (A secure context, for WebGPU: a plain file of the dev server's.)
    await page.goto(new URL('/robots.txt', LAB_URL).href)
    const shaders = { render: W.RENDER, resolve: W.RESOLVE, resolveUp: W.RESOLVE_UP }
    for (const [name, code] of Object.entries(shaders)) {
      const times = []
      for (let i = 0; i < 3; i++) {
        times.push(
          await page.evaluate(async ([code, nonce]) => {
            const adapter = await navigator.gpu.requestAdapter()
            const d = await adapter.requestDevice()
            const t0 = performance.now()
            const m = d.createShaderModule({ code: `${code}\n// ${nonce}` })
            await d.createRenderPipelineAsync({ layout: 'auto', vertex: { module: m, entryPoint: 'vs' }, fragment: { module: m, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] }, primitive: { topology: 'triangle-list' } })
            const ms = performance.now() - t0
            d.destroy()
            return ms
          }, [code, `${name}-${i}-${Date.now()}`]),
        )
      }
      note(`${name}: compiled in ${times.map(t => (t / 1000).toFixed(2)).join(', ')} s`)
    }
  }
} finally {
  await browser.close()
  fs.writeFileSync(`${dir}/${o.mode}-${o.tag}.txt`, log.join('\n') + '\n')
}
