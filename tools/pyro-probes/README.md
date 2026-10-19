# Pyro lab probes

Scripts that drive the Pyro lab's engine in a headless Edge and measure what it is doing: the gas fields, single
cells step by step, the burning ball, stability, smoke detail, the fire's validation against fire science, GPU time.
They run against the **development server** (the engine is exposed as `window.__pyro` only in development builds),
paused and stepped exactly, so a given scenario gives the same run every time.

## Setup

- Start the dev server (port 4622 by default; set `PYRO_URL` otherwise).
- `npm install` brings Playwright: `playwright-core`, a development dependency only (nothing of it ships with the
  site), which drives the installed browser and downloads none. `PLAYWRIGHT_FROM` can point at another copy (a
  package.json whose node_modules has `playwright`).
- They drive the installed Edge (`PYRO_BROWSER` to change it), with its GPU.
- Run one probe at a time: two sharing the GPU can lose the device.

## Scenarios (every probe)

| Option | Meaning |
|---|---|
| `--preset "Smoke plume"` | the starting point to click, by its button label (`""` for none) |
| `--patch '{"velocity":"lfm"}'` | lab state to apply on top, as JSON |
| `--ball-at 0,0.45` | put the collision ball here (x, y in metres) at the start |
| `--pull-at 60 --pull-to 1.3,0.6` | move it there at that simulated second |
| `--out probe-out` | where files go |

`--help` on any probe lists its own options.

## The probes

| Probe | What it tells you |
|---|---|
| `fields.mjs` | Heat maps of each field (fire: soot, temperature rise, mixture fraction, fuel, heat release; smoke: density, temperature rise), a front slice and the extreme over the depth, at chosen moments. Start here when a render looks wrong. |
| `hotcells.mjs` | The highest cells of a field, where they are (and how far from the ball's surface), and the rest of the gas's state there. |
| `cellprobe.mjs` | One cell and its six neighbours, step by step: the gas, the fire's mass-keeping ratio (`readRatio`), the velocity. |
| `ballburn.mjs` | The burning ball held in the bonfire and pulled out: its surface temperature, burning power and char every few seconds, and when it first flames. |
| `embers.mjs` | The ball's embers once pulled out: still, stirred beside it and still again, each second its hottest faces and cracks, how much of it glows, and its flames' power. |
| `woodcone.mjs` | The ball's wood as fire scientists test wood: a cone calorimeter's flux (flaming, or with its flame out, optionally blown), or the standard furnace fire, one column read as its char line advances. |
| `charsphere.mjs` | The char's surface model solved for a small char sphere burning in air, against measured 1 cm wood-char spheres. No GPU. |
| `stability.mjs` | The fastest gas and where, the kinetic energy and the total smoke over a run, optionally while stirring or sweeping the ball. A blow-up shows as speeds in the hundreds. |
| `detail.mjs` | Smoke detail: sharpness, enstrophy, kinetic energy, where the smoke sits, and stills. |
| `validate.mjs` | The fire against McCaffrey's centreline temperatures, Heskestad's flame height and Cetegen and Ahmed's puffing frequency. About 8 minutes a case. |
| `stills.mjs` | Settled renders at chosen moments, or every step for a clip. |
| `profile.mjs` | GPU time per kernel (median of several), and the live steps a second. |

## Recipes

The soot runaway beside the ball (fixed 2026-10-09), found in three steps:

```bash
node tools/pyro-probes/fields.mjs --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-at 60 --times 120,140,150
```
```bash
node tools/pyro-probes/hotcells.mjs --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-at 60 --times 130,138,141
```
```bash
node tools/pyro-probes/cellprobe.mjs --preset "Bonfire at dusk" --ball-at 0,0.45 --pull-at 60 --cell 116,16,22 --from 136 --steps 60 --every 3
```

The physics review page's numbers:

```bash
node tools/pyro-probes/validate.mjs --cases '[{},{"heatRelease":40},{"heatRelease":150}]'
```
```bash
node tools/pyro-probes/ballburn.mjs --pull-at 60
```
```bash
node tools/pyro-probes/detail.mjs --patch '{"velocity":"lfm"}' --tag lfm
```

## The engine's readouts

The probes read the engine through its development readouts, which can also be called from the browser console on
the dev server: `readScalars()` (the raw fields and heat release), `readFields()` (density, temperature, velocity
per cell), `readBall()` then `ballStatus`, `readRatio()` (the mass-keeping ratios), `readIvock()`,
`measureDivergence()`, `profileStep()`, `snapshot(w, h)`, and `advance(steps)` to step it while paused.
