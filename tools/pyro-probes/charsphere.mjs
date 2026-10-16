// The wooden ball's char surface model (BALL_HEAT's charFire, charKinetics, charBurn and the surface's heat transfer,
// surfaceH) solved for a small char sphere burning in air with no other heat, its burning's heat balanced against its
// radiation and convection, at a range of air speeds. Varunkumar, Rajan and Mukunda (2011) measured 1 cm wood-char
// spheres burning at 983 K in still air and 1023 K at 0.13 to 0.15 m/s; Yan and colleagues (2024), wood spheres at 950
// to 1150 K from still air to 8 m/s. No GPU: the same arithmetic as the shader, in JavaScript.
// Usage: node tools/pyro-probes/charsphere.mjs [--d 0.01] [--speeds 0,0.14,0.5,1,2,4,8]
import { parseArgs } from 'node:util'

const { values: o } = parseArgs({ options: { d: { type: 'string', default: '0.01' }, speeds: { type: 'string', default: '0,0.14,0.5,1,2,4,8' } } })
const SIGMA = 5.670374e-8
const RG = 8.314
const Y_O2 = 0.233
const T_AMB = 293
const EPS = 0.838
const smooth = (a, b, x) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1)
  return t * t * (3 - 2 * t)
}
const charFire = t => {
  const co = 2512 * Math.exp(-6240 / t)
  const away = (co / (1 + co)) * (1 - smooth(850, 1000, t))
  return { cPerO2: 0.375 / (1 - 0.5 * away), heat: 3.28e7 + (9.2e6 - 3.28e7) * away }
}
const charKinetics = t => {
  const measured = 0.248 * Math.exp(-39784 / (RG * t))
  const cone = (13 / 1100) * Y_O2 * charFire(t).cPerO2
  return measured / Math.max(1 - measured / cone, 0.05)
}
const charBurn = (k, s) => 1 / (1 / Math.max(k, 1e-12) + 1 / Math.max(s, 1e-12))
// (As surfaceH, the sphere uncrazed: forced and natural convection over its own diameter.)
const surfaceH = (u, ts, tg, d) => {
  const tf = 0.5 * (ts + tg)
  const nu = 1.5e-5 * (tf / 293) ** 1.7
  const k = 0.0257 * (tf / 293) ** 0.8
  const re = (u * d) / nu
  const hF = ((2 + (0.4 * Math.sqrt(re) + 0.06 * re ** (2 / 3)) * 0.872) * k) / d
  const ra = (((9.81 * Math.abs(ts - tg)) / tf) * d ** 3 * 0.71) / nu ** 2
  const hN = ((2 + (0.589 * ra ** 0.25) / 1.296) * k) / d
  return Math.cbrt(hF ** 3 + hN ** 3)
}
const d = Number(o.d)
console.log(`${d * 1000} mm char sphere burning in air (Varunkumar 2011: 983 K still, 1023 K at 0.13 to 0.15 m/s; Yan 2024: 950 to 1150 K over 0 to 8 m/s)`)
for (const u of o.speeds.split(',').map(Number)) {
  // (The upper root, a burning sphere's: down from 1800 K to the first temperature its burning can hold.)
  let t = 1800
  let h = 0
  let r = 0
  for (; t > 400; t -= 1) {
    h = surfaceH(u, t, T_AMB, d)
    const f = charFire(t)
    r = charBurn(charKinetics(t), (h / 1100) * Y_O2 * f.cPerO2)
    if (r * f.heat - EPS * SIGMA * (t ** 4 - T_AMB ** 4) - h * (t - T_AMB) >= 0) break
  }
  console.log(t > 400 ? `  ${u.toFixed(2)} m/s: ${t} K, burning ${(r * 1e3).toFixed(2)} g/m2/s (heat transfer ${h.toFixed(1)} W/m2 K)` : `  ${u.toFixed(2)} m/s: does not keep burning`)
}
