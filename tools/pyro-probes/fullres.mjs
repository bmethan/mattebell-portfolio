// The running lab as a visitor sees it, and what drawing it costs: the temporal upsampling's checks. Modes:
//   emulate  the running view exactly (the engine's runView: frames of a step and a render each, the running history)
//            at each render scale, at one simulated moment, then the paused view a visitor gets on pausing there (the
//            engine's pausedView); beside them the settled full-size still of the same moment, without motion blur (for
//            the running view) and with it (for the paused), and a still a third of a second later, to tell what is
//            still from what moves.
//   live     the lab as a visitor runs it, in real time: screenshots of the frame, and its render scale, canvas, march,
//            frames and steps a second.
//   cost     GPU time of the image pass and the passes after it (the engine's profileStep) at each render scale, and
//            paused (the march at the display's size).
//   compile  how long the image pass and the resolve take to compile, each on its own, on a fresh device.
//   loop     frames for a pre-rendered loop: settled full-size frames, one a step, as the stills are.
// A configuration is a render scale, pinned (0.5, 0.75), or "auto" (left to adapt, live only).
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
    steps: { type: 'string', default: '1' },
    configs: { type: 'string', default: '0.5' },
    seconds: { type: 'string', default: '14' },
    runs: { type: 'string', default: '7' },
    shots: { type: 'string', default: '2' },
    tag: { type: 'string', default: 'envy' },
    size: { type: 'string', default: '1829,765' },
    fade: { type: 'string', default: '1' },
  },
  `
fullres.mjs: the running and paused views and their cost.
  --mode emulate|live|cost|compile|loop|drag|trace
  --window 1600,765,1     the window's CSS width and height and its device pixels per CSS pixel
  --at 10                 simulated second to capture or profile at (emulate, cost)
  --frames 45 --steps 1   running frames drawn before the capture (emulate), and solver steps a frame (0 holds the fire
                          still: the running view's sampling alone, without the flames' motion and flicker)
  --configs 0.5,0.75      render scales to pin ("auto" leaves it adapting, live only)
  --seconds 14 --shots 2  how long to let it run before the screenshots, and how many, half a second apart (live)
  --runs 7                profiles to take the median of (cost)
  --tag envy              file names: <out>/<mode>-<tag>-<config>.png
  --size 1829,765 --fade 1   (loop) frame size, and seconds rendered on past --seconds for the closing cross-fade
Example: node tools/pyro-probes/fullres.mjs --mode emulate --configs 0.5,0.75 --preset "Bonfire at dusk"`,
)
const [ww, wh, dpr] = list(o.window)
const view = { width: ww, height: wh, dpr }
const dir = outDir(o.out)
const configs = o.configs.split(',').map(c => ({ name: c, scale: c === 'auto' ? null : Number(c) }))
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
      await page.evaluate(scale => {
        const e = window.__pyro
        e.pinScale = scale
        e.restart()
      }, c.scale)
      await page.waitForTimeout(300)
      await runTo(page, at - (frames * Number(o.steps)) / 30, o.scenario)
      const b64 = await page.evaluate(async ([frames, steps, toB64]) => (0, eval)(`(${toB64})`)(await window.__pyro.runView(frames, steps)), [frames, Number(o.steps), toB64])
      const v = await page.evaluate(() => ({ ...window.__pyro.lastView, now: window.__pyro.simTime }))
      saveBase64(`${dir}/emulate-${o.tag}-${c.name}.png`, b64)
      note(`${c.name}: at ${v.t.toFixed(3)} s (now ${v.now.toFixed(3)}), canvas ${v.canvas.join('x')}, march ${v.march.join('x')}`)
      // Paused there: the running history settling, as the visitor sees it.
      const p64 = await page.evaluate(async toB64 => (0, eval)(`(${toB64})`)(await window.__pyro.pausedView()), toB64)
      const pv = await page.evaluate(() => window.__pyro.lastView)
      saveBase64(`${dir}/emulate-${o.tag}-paused-${c.name}.png`, p64)
      note(`  paused: ${pv.ev} frames to settle, last march ${pv.march.join('x')}`)
    }
    // The references: settled, full size (the frame's device pixels), without motion blur and with it; then a third of
    // a second on.
    const ref = async name => {
      const b64 = await page.evaluate(async toB64 => {
        const e = window.__pyro
        e.pinScale = null
        const w = Math.round(e.cssW * Math.min(e.dpr, 2))
        const h = Math.round(e.cssH * Math.min(e.dpr, 2))
        return (0, eval)(`(${toB64})`)(await e.snapshot(w, h, 'image/png'))
      }, toB64)
      saveBase64(`${dir}/emulate-${o.tag}-${name}.png`, b64)
    }
    const shutter = await page.evaluate(() => window.__pyro.state.shutter)
    await ref('reference-blur')
    await page.evaluate(() => window.__pyro.setState({ shutter: 0 }))
    await ref('reference')
    await page.evaluate(() => window.__pyro.advance(10))
    await ref('reference-later')
    await page.evaluate(sh => window.__pyro.setState({ shutter: sh }), shutter)
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
      await page.evaluate(scale => {
        const e = window.__pyro
        e.pinScale = scale
        e.resize(e.cssW, e.cssH, e.dpr)
        // Frames a second, counted from the page's own animation frames.
        window.__frames = []
        const tick = now => {
          window.__frames.push(now)
          requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
      }, c.scale)
      const reads = []
      for (let s = 0; s < Number(o.seconds); s++) {
        await page.waitForTimeout(1000)
        reads.push(
          await page.evaluate(() => {
            const e = window.__pyro
            const now = performance.now()
            const fps = window.__frames.filter(t => now - t <= 1000).length
            return { sim: +e.simTime.toFixed(1), steps: document.querySelector('#lab-panel-pyro .lab-hud-right').textContent.trim(), scale: +e.renderScale.toFixed(3), canvas: `${e.canvas.width}x${e.canvas.height}`, march: `${e.marchW}x${e.marchH}`, fps, history: e.taaCount }
          }),
        )
      }
      for (let k = 0; k < Number(o.shots); k++) {
        if (k > 0) await page.waitForTimeout(500)
        const before = await page.evaluate(() => window.__pyro.taaCount)
        await frame.screenshot({ path: `${dir}/live-${o.tag}-${c.name}${k ? `-${k}` : ''}.png` })
        const after = await page.evaluate(() => window.__pyro.taaCount)
        note(`  shot ${k}: history ${before} frames before, ${after} after`)
      }
      note(`${c.name} (live, ${((Date.now() - t0) / 1000).toFixed(0)} s after the first frame):`)
      for (const r of reads) note(`  ${JSON.stringify(r)}`)
      await page.close()
    }
  } else if (o.mode === 'cost') {
    const page = await openLab(browser, o.scenario, view)
    await runTo(page, Number(o.at), o.scenario)
    for (const c of [...configs, { name: 'paused', scale: null }]) {
      const runs = []
      let march = null
      for (let i = 0; i < Number(o.runs); i++) {
        const r = await page.evaluate(async scale => {
          const e = window.__pyro
          // (A running march at the scale, or the paused one.)
          let march = e.pausedMarch()
          if (scale != null) {
            e.pinScale = scale
            e.sizeTo()
            march = [e.marchW, e.marchH]
          }
          return { march, t: await e.profileStep(march) }
        }, c.scale)
        runs.push(r.t)
        march = r.march
      }
      const sz = { ...(await sizes(page)), march }
      const med = {}
      for (const k of ['render', 'accumulate', 'glare', 'display']) {
        const v = runs.map(r => r[k] ?? 0).sort((a, b) => a - b)
        med[k] = +v[v.length >> 1].toFixed(2)
      }
      const frame = med.render + med.accumulate + med.glare + med.display
      note(`${c.name}: canvas ${sz.canvas.join('x')}, march ${sz.march.join('x')}: ${JSON.stringify(med)} frame ${frame.toFixed(2)} ms`)
    }
    const step = await page.evaluate(() => window.__pyro.profileStep([1, 1]))
    const stepMs = Object.entries(step)
      .filter(([k]) => !['render', 'accumulate', 'glare', 'display'].includes(k))
      .reduce((a, [, v]) => a + v, 0)
    note(`the step's own kernels (one profile): ${stepMs.toFixed(1)} ms`)
  } else if (o.mode === 'drag') {
    // The collision ball dragged through the running view: --frames frames of a step, a nudge of the ball (2 cm along
    // x) and a render each, then the canvas; the set beside it should stay as sharp as it was, the ball leave no trail.
    const page = await openLab(browser, o.scenario, view)
    const frames = Number(o.frames)
    await runTo(page, Number(o.at) - frames / 30, o.scenario)
    const b64 = await page.evaluate(
      async ([frames, toB64]) => {
        const e = window.__pyro
        e.paused = false
        e.sizeTo()
        e.layoutCamera()
        e.taaCount = 0
        for (let i = 0; i < frames; i++) {
          e.step()
          e.nudgeBall(0.02, 0)
          e.render()
        }
        const blob = new Promise(r => e.canvas.toBlob(r, 'image/png'))
        e.paused = true
        e.applySize()
        return (0, eval)(`(${toB64})`)(await blob)
      },
      [frames, toB64],
    )
    saveBase64(`${dir}/drag-${o.tag}.png`, b64)
    note(`drag: ${frames} frames, the ball now at ${JSON.stringify(await page.evaluate(() => window.__pyro.ballPos.map(v => +v.toFixed(3))))}`)
  } else if (o.mode === 'trace') {
    // The path-traced reference, paused at --at: its pipeline compiled, then --seconds of tracing; the paths a pixel it
    // reached, and a screenshot of the frame with the wipe in the middle.
    const page = await openLab(browser, o.scenario, view)
    await runTo(page, Number(o.at), o.scenario)
    const r = await page.evaluate(async secs => {
      const e = window.__pyro
      e.setPathTrace(true)
      const t0 = performance.now()
      while ((e.ptCompiling || !e.ptPipe) && performance.now() - t0 < 120000) await new Promise(r => setTimeout(r, 250))
      const compileS = (performance.now() - t0) / 1000
      // (The live frame settles first; the traced paths start after it.)
      while (e.pt.n === 0 && performance.now() - t0 < 180000) await new Promise(r => setTimeout(r, 100))
      const n0 = e.pt.n
      const t1 = performance.now()
      await new Promise(r => setTimeout(r, secs * 1000))
      return { compileS, size: [e.canvas.width, e.canvas.height], paths: e.pt.n, rate: (e.pt.n - n0) / ((performance.now() - t1) / 1000), on: e.pt.on }
    }, Number(o.seconds))
    await page.locator('#lab-panel-pyro .lab-frame').screenshot({ path: `${dir}/trace-${o.tag}.png` })
    note(`trace: compiled in ${r.compileS.toFixed(1)} s, at ${r.size.join('x')}: ${r.paths} paths a pixel, ${r.rate.toFixed(1)} a second, on ${r.on}`)
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
    const shaders = { render: W.RENDER, resolve: W.RESOLVE }
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
