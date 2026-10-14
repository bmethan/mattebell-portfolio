import { BB_MAX_K, BB_MIN_K, chemiluminescenceColour } from './blackbody'
import { AP1_WGSL, VIEW_TRANSFORMS_WGSL } from '../lab/viewTransforms'
import { FUZZ } from '../lookdev/gpu/wgslBsdf'
import { COLOR_CHECKER, SCENE_MATERIALS } from '../lookdev/materials'
import { FIRE_BANDS } from './physics'
import { CHEMI_LM_PER_W, CP, FUEL, RHO_T, SCHMIDT_T, SMAGORINSKY, SOOT_DELAY, SOOT_EXTINCTION, T_AMBIENT, Y_O2_AIR, wf } from './physics'

// The fire's escape table's rings (degrees of elevation; RENDER_COMMON's escapeDir) and their sines, by which the
// table is read; and the share of the sphere each stands for, with straight down first and straight up last (for the
// table's mean over the sphere). Between rings the table is read linearly in the sine (a hat function in the sine,
// whose measure over the sphere is even: each end takes half its span, over 2); toward a pole, linearly in the
// angle from the pole, as the root of 1 - |sine| (the pole a third, the ring two thirds of its span, over 2).
const ESCAPE_RINGS = [-30, -15, 0, 15, 22.5, 30, 37.5, 45, 60, 75]
const ESCAPE_SINES = ESCAPE_RINGS.map(d => Math.sin((d * Math.PI) / 180))
// The directions (sixteen a ring, then straight up and straight down), the table's entries (four to an entry, with the
// mean over the sphere after the directions: up, down and the mean in the last), and the steps a table takes to march.
const ESCAPE_AZ = 16
const ESCAPE_NDIRS = ESCAPE_AZ * ESCAPE_RINGS.length + 2
const ESCAPE_ENTRIES = Math.ceil((ESCAPE_NDIRS + 1) / 4)
const ESCAPE_POLES = (ESCAPE_AZ / 4) * ESCAPE_RINGS.length
export const ESCAPE_PHASES = 24
// The fire's light and its tables in the sums (FIRE_SUM_TOTAL), vec4 entries: the totals, two a slice, then each
// slice's escape table, scattered light's table, spread and escape table by height.
export const FIRE_LIGHT_ENTRIES = 3 + (3 + 3 * ESCAPE_ENTRIES) * FIRE_BANDS
// The share of the turn each of a quadrant's azimuths stands for, at even steps of the diamond angle (half the gaps
// either side), for the table's mean.
const AZ_SHARES = (() => {
  const q = ESCAPE_AZ / 4
  const a = Array.from({ length: q + 1 }, (_, i) => Math.atan2(i / q, 1 - i / q))
  return Array.from({ length: q }, (_, i) => {
    const before = i === 0 ? a[1] - a[0] : a[i] - a[i - 1]
    return ((a[i + 1] - a[i]) / 2 + before / 2) / (2 * Math.PI)
  })
})()
const ESCAPE_WEIGHTS = (() => {
  const m = [-1, ...ESCAPE_SINES, 1]
  const w = m.map(() => 0)
  for (let i = 0; i + 1 < m.length; i++) {
    const half = (m[i + 1] - m[i]) / 2
    const toLower = i === 0 ? 1 / 3 : i + 2 === m.length ? 2 / 3 : 1 / 2
    w[i] += half * toLower
    w[i + 1] += half * (1 - toLower)
  }
  return w
})()

// The share of slice b's light that gets out of its own soot toward p (bandOut), from the escape tables in the fire's
// sums (FIRE_SUM): read linearly in the sine of the elevation between the rings (and straight up and down), and in
// azimuth between the sixteen by the diamond angle (|z| over |x| + |z| round the quadrants, exact at the sixteen and
// with no inverse trigonometry: each surface reads the tables thirty-two times a pixel). With the light the soot
// scatters back out: it scatters what it stops by its albedo, and mostly onward (the lab's anisotropy). By the
// similarity relation the escape table counts the onward share as never stopped (FIRE_SUM), and the rest is scattered
// evenly every way, at the albedo a (1 - g) / (1 - a g). Of the share the soot stops (1 - the table's mean over the
// sphere), that much is scattered from where it was stopped, and gets out by the second table, the share of such light
// that leaves each way; what the soot stops of that, again, and so on: albedo (1 - mean) / (1 - albedo (1 - its mean))
// of the light, each way by the second table. (Scattered light starts nearer the soot's edge than the flames that made
// it, and gets out more easily and more evenly. Let out as the flames' own light is, the floor beside a bonfire had
// seven tenths of the scattered light the path-traced reference gives it, and the wall and smoke beside the fire were
// 9% dark.) Within a bed's radius of the slice's centre, Beer's law over the share of it crossed. (For the light pass,
// whose binding is fireSums, and the renderers, fireLight.)
// The slices' spread geometry (bandSpread, RENDER_LIB), for the renderers (fireLight) and the light pass (fireSums).
function bandSpreadWgsl(sums: string) {
  return /* wgsl */ `
fn bandSpread(b: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
  let c = ${sums}[4u + 2u * b].xyz;
  let m = ${sums}[SPREAD_BASE + b];
  let mid = 0.5 * (m.x + m.z);
  let dev = sqrt(max(0.25 * (m.x - m.z) * (m.x - m.z) + m.w * m.w, 0.0));
  let ang = 0.5 * atan2(2.0 * m.w, m.x - m.z);
  let e1 = vec3<f32>(cos(ang), 0.0, sin(ang)) * sqrt(max(mid + dev, 0.0) * 3.0);
  let e2 = vec3<f32>(-sin(ang), 0.0, cos(ang)) * sqrt(max(mid - dev, 0.0) * 3.0);
  let e3 = vec3<f32>(0.0, min(sqrt(m.y * 3.0), max(c.y - 0.5 * P.grid.x, 0.0)), 0.0);
  var g = 0.0;
  for (var k = 0u; k < 6u; k++) {
    let o = select(select(e3, e2, k / 2u == 1u), e1, k / 2u == 0u) * select(1.0, -1.0, (k & 1u) == 1u);
    let d = c + o - p;
    let r2 = dot(d, d) + BAND_R2 / 6.0;
    g += max(dot(d, n), 0.0) / (sqrt(r2) * r2);
  }
  return g / 6.0;
}
`
}

function bandOutWgsl(sums: string) {
  return /* wgsl */ `
fn escTab(base: u32, k: u32) -> f32 { return ${sums}[base + k / 4u][k % 4u]; }
fn escRing(base: u32, r: u32, a0: u32, a1: u32, fa: f32) -> f32 {
  return mix(escTab(base, ${ESCAPE_AZ}u * r + a0), escTab(base, ${ESCAPE_AZ}u * r + a1), fa);
}
fn bandEscape(base: u32, u: vec3<f32>) -> f32 {
  let ax = abs(u.x);
  let az2 = abs(u.z);
  let q = select(az2 / (ax + az2), 0.0, ax + az2 < 1e-6);
  let dia = select(select(3.0 + 1.0 - q, 2.0 + q, u.x < 0.0), select(1.0 + 1.0 - q, q, u.x >= 0.0), u.z >= 0.0);
  let az = ${ESCAPE_AZ / 4}.0 * dia;
  let a0 = u32(floor(az)) % ${ESCAPE_AZ}u;
  let a1 = (a0 + 1u) % ${ESCAPE_AZ}u;
  let fa = az - floor(az);
  let m = u.y;
  // (Toward a pole, by the angle from it, the root of 1 - |sine|, within 1% to 30 degrees from the pole: straight up
  // looks along the whole flame column, and the escape falls toward it evenly in the angle; read in the sine, which
  // bunches near the pole, the upper slices' light toward 70 to 85 degrees came out a quarter short.)
  if (m <= ESCAPE_SINES[0]) {
    return mix(escTab(base, ESCAPE_DIRS - 1u), escRing(base, 0u, a0, a1, fa), sqrt(max(m + 1.0, 0.0) / (ESCAPE_SINES[0] + 1.0)));
  }
  if (m >= ESCAPE_SINES[ESCAPE_NRINGS - 1u]) {
    return mix(escRing(base, ESCAPE_NRINGS - 1u, a0, a1, fa), escTab(base, ESCAPE_DIRS - 2u), 1.0 - sqrt(max(1.0 - m, 0.0) / (1.0 - ESCAPE_SINES[ESCAPE_NRINGS - 1u])));
  }
  var r = 0u;
  for (var q = 1u; q < ESCAPE_NRINGS - 1u; q++) { if (m >= ESCAPE_SINES[q]) { r = q; } }
  let f = (m - ESCAPE_SINES[r]) / (ESCAPE_SINES[r + 1u] - ESCAPE_SINES[r]);
  return mix(escRing(base, r, a0, a1, fa), escRing(base, r + 1u, a0, a1, fa), f);
}
fn bandOut(b: u32, p: vec3<f32>) -> f32 {
  let d = p - ${sums}[4u + 2u * b].xyz;
  let dist = max(length(d), 1e-6);
  let albedo = R.skyCol.w * (1.0 - R.sunDir.w) / max(1.0 - R.skyCol.w * R.sunDir.w, 1e-3);
  let base = ESCAPE_BASE + ESCAPE_STRIDE * b;
  let base2 = ESCAPE_BASE2 + ESCAPE_STRIDE * b;
  let mean = ${sums}[base + ESCAPE_POLES].z;
  let meanS = ${sums}[base2 + ESCAPE_POLES].z;
  let scat = albedo * (1.0 - mean) / (1.0 - albedo * (1.0 - meanS));
  let e = clamp(bandEscape(base, d / dist) + scat * bandEscape(base2, d / dist), 1e-6, 1.0);
  let rad = max(P.src.w, 0.1);
  return select(e, pow(e, dist / rad), dist < rad);
}
// The same for a point p of the floor or the cyc facing n. The floor sees each glowing cell by Lambert's cosine, its
// height over the distance cubed, so whatever the distance the slice's upper cells weigh more in the light it takes
// than in the slice's own; and above the bed's soot, they let out more of it. So the floor reads the escape by height
// (ESCAPE_BASE3); the scattered light, the same by the ratio of the two. (Weighted by the cells' light alone, the escape
// toward a campfire's floor a metre or more off on the side its flames lean to was 12% under the floor's own for its
// lowest slice, and 4% for the next, against every glowing cell's light marched through the soot to each point.) Any
// surface weighs a cell by its distance in front of it along n, the slice's centre's (off) and the cell's height over
// the centre's times n.y (the cells' spread across n besides their height taken as even about the centre), so its
// escape is e0 + n.y y (eh - e0) / off, y the centre's height: the floor's by height, the wall's the slice's own, the
// cyc's sweep between by its own cosine. (Read toward the height table the more the more it faces up instead, the
// sweep took up to a fifth too much of a bonfire's light half way up it, where its cosine weighs the cells mostly by
// how far in front of it they stand.)
fn bandOutFloor(b: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
  let d = p - ${sums}[4u + 2u * b].xyz;
  let dist = max(length(d), 1e-6);
  let albedo = R.skyCol.w * (1.0 - R.sunDir.w) / max(1.0 - R.skyCol.w * R.sunDir.w, 1e-3);
  let base = ESCAPE_BASE + ESCAPE_STRIDE * b;
  let base2 = ESCAPE_BASE2 + ESCAPE_STRIDE * b;
  let mean = ${sums}[base + ESCAPE_POLES].z;
  let meanS = ${sums}[base2 + ESCAPE_POLES].z;
  let scat = albedo * (1.0 - mean) / (1.0 - albedo * (1.0 - meanS));
  let e0 = bandEscape(base, d / dist);
  let eh = bandEscape(ESCAPE_BASE3 + ESCAPE_STRIDE * b, d / dist);
  let off = -dot(d, n);
  let r = select(1.0, clamp(1.0 + n.y * max(${sums}[4u + 2u * b].y, 0.0) * (eh / max(e0, 1e-6) - 1.0) / off, 0.5, 2.0), off > 1e-3);
  let e = clamp((e0 + scat * bandEscape(base2, d / dist)) * r, 1e-6, 1.0);
  let rad = max(P.src.w, 0.1);
  return select(e, pow(e, dist / rad), dist < rad);
}
`
}

// The cyc's floor ends this far behind the fire (m), past the back of the gas's domain (about 0.48 m), and sweeps up
// this radius into a wall 2.3 m back.
export const CYC_Z = -1.1
export const CYC_R = 1.2
// The map of the cyc's light (CYC_DIRECT, CYC_BOUNCE): across the view from x0, and along the cyc from the floor at
// front (m from the fire) back to the sweep, up it, and up the wall to wallTop; texels dx (m) square.
export const CYC_MAP = { x0: -4.8, front: 4.0, wallTop: 5.0, dx: 0.2, w: 48, h: 54 }

// WGSL for the pyro lab: a gas solver on a staggered (MAC) grid, as production FX solvers lay it out, and a
// volume renderer for it.
//
// Grid: nx x ny x nz cells of size dx, the floor at y = 0, centred on x and z. Scalars (density, temperature)
// sit at cell centres; each velocity component sits on the faces normal to it, all three in one texture of
// (nx+1, ny+1, nz+1) texels: u at texel (i,j,k) is the x face at grid point (i, j+1/2, k+1/2), v the y face at
// (i+1/2, j, k+1/2), w the z face at (i+1/2, j+1/2, k). Texels past a component's own range are never read.
//
// Boundaries: the floor is solid (v = 0 there, pressure Neumann); the sides and top are open (pressure 0 just
// outside, smoke leaves, and air flowing in carries none).

// The wooden ball's skin map, texels around and from pole to pole (see SIM_COMMON).
// Leapfrog flow maps' cycle, in steps (the paper's fire-ball setting).
export const LFM_N = 4
export const SKIN_W = 128
export const SKIN_H = 64

// The on-set reference as a solid (engine.ts referenceSet), from a uniform S holding its fields: its two balls, and
// its chart as a card one cell (h) thick, which a thinner card would let the gas through between cell centres. Its
// stands, thinner than a cell, are left out.
const setSolid = (S: string) => /* wgsl */ `
fn inSet(w: vec3<f32>, h: f32) -> bool {
  // (Most cells are nowhere near it: first, the set's extent, along x, up to the balls' tops, back to their backs.)
  if (${S}.refGrey.w <= 0.0 || w.x < ${S}.refChartU.w - h || w.x > ${S}.refChartV.w + h) { return false; }
  if (w.y > ${S}.refGrey.y + ${S}.refGrey.w + h || w.z < ${S}.refGrey.z - ${S}.refGrey.w - h) { return false; }
  if (distance(w, ${S}.refGrey.xyz) < ${S}.refGrey.w || distance(w, ${S}.refChrome.xyz) < ${S}.refChrome.w) { return true; }
  let U = ${S}.refChartU.xyz;
  let V = ${S}.refChartV.xyz;
  let N = normalize(cross(U, V));
  let q = w - ${S}.refChartO.xyz;
  let a = dot(q, U) / dot(U, U);
  let b = dot(q, V) / dot(V, V);
  return abs(dot(q, N)) < 0.5 * h * (abs(N.x) + abs(N.y) + abs(N.z)) && a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0;
}`

export const SIM_COMMON = /* wgsl */ `
struct Params {
  dims: vec4<u32>,      // nx, ny, nz, frame
  grid: vec4<f32>,      // dx (m), dt (s), time (s), whether the velocity is advected with MacCormack (0 or 1)
  src: vec4<f32>,       // source centre (m), radius (m)
  srcProps: vec4<f32>,  // smoke density, smoke temperature rise (K), upward speed (m/s), noise (0..1)
  forces: vec4<f32>,    // buoyancy (m/s2 per K of temperature rise), unused, vorticity, turbulence (m/s2)
  forces2: vec4<f32>,   // turbulence scale (m), wind x (m/s), wind z (m/s), drag (1/s)
  decay: vec4<f32>,     // cooling (1/s), dissipation (1/s), SOR relaxation, speed limit (m/s)
  stirO: vec4<f32>,     // stir centre (m), radius (m)
  stirD: vec4<f32>,     // stir velocity change (m/s), on (0 or 1)
  stirR: vec4<f32>,     // LFM's step within its cycle, the forces' time step in steps (0 = one; LFM's leapfrog), the eddy viscosity on (IVOCK or LFM; 0 or 1), the scalars' MacCormack on (0 or 1)
  origin: vec4<f32>,    // domain min corner (m)
  detail: vec4<f32>,    // reset detail layer A this step (0 or 1), layer B
  fire: vec4<f32>,      // source is a fire bed (0 or 1), fuel it gives off (kg/s), soot yield (kg/kg), radiant fraction
  fire2: vec4<f32>,     // expansion (1 = physical), mixing time (s), the bed's nominal area (m2), unused
  ball: vec4<f32>,      // the collision ball's centre (m), radius (m; 0 = none)
  ballVel: vec4<f32>,   // its velocity (m/s), whether it is wood that burns (0 or 1)
  refGrey: vec4<f32>,   // the on-set reference (engine.ts): the grey ball's centre and radius (0: no reference)
  refChrome: vec4<f32>, // the chrome ball's
  refChartO: vec4<f32>, // the colour chart's lower corner; the stands' radius
  refChartU: vec4<f32>, // the chart's width, along its lower edge; the set's least x
  refChartV: vec4<f32>, // its height, up its face; the set's greatest x
}
@group(0) @binding(0) var<uniform> P: Params;

// The solids: the collision ball, a sphere, and the on-set reference when it stands. A cell (or a face) is solid when
// its centre is inside one. The ball's faces move with it; the set's stand still.
fn inBall(w: vec3<f32>) -> bool { return P.ball.w > 0.0 && distance(w, P.ball.xyz) < P.ball.w; }
${setSolid('P')}
fn solidAt(w: vec3<f32>) -> bool { return inBall(w) || inSet(w, P.grid.x); }
fn cellSolid(c: vec3<i32>) -> bool { return solidAt(P.origin.xyz + (vec3<f32>(c) + 0.5) * P.grid.x); }
fn anySolid() -> bool { return P.ball.w > 0.0 || P.refGrey.w > 0.0; }
// The velocity of a face between cells a and b, one of them solid: the collision ball's, or none (the set's).
fn wallVel(a: vec3<i32>, b: vec3<i32>) -> vec3<f32> {
  let wa = P.origin.xyz + (vec3<f32>(a) + 0.5) * P.grid.x;
  let wb = P.origin.xyz + (vec3<f32>(b) + 0.5) * P.grid.x;
  return select(vec3<f32>(0.0), P.ballVel.xyz, inBall(wa) || inBall(wb));
}

// The ball as wood (a pine, 500 kg/m3) that heats, chars and burns. Its skin is a map of its surface, equirectangular
// (u around the vertical from +x, v down from the top), and holds per texel: x the surface's temperature (K), y the
// wood's a few millimetres in (K), z the char's depth (mm), w how much of the char has turned to ash (0 to 1). It
// moves with the ball and does not turn.
const SKIN_W = ${SKIN_W}u;
const SKIN_H = ${SKIN_H}u;
@group(0) @binding(31) var<storage, read> skin: array<vec4<f32>>;
fn ballBurns() -> bool { return P.ball.w > 0.0 && P.ballVel.w > 0.5; }
fn skinDir(u: f32, v: f32) -> vec3<f32> {
  let phi = 6.2831853 * u;
  let th = 3.1415927 * v;
  return vec3<f32>(sin(th) * cos(phi), cos(th), sin(th) * sin(phi));
}
// The skin where the outward normal n meets it, bilinear between texels (wrapping around, clamped at the poles).
fn skinAt(n: vec3<f32>) -> vec4<f32> {
  let x = fract(atan2(n.z, n.x) * 0.15915494 + 1.0) * f32(SKIN_W) - 0.5;
  let y = clamp(acos(clamp(n.y, -1.0, 1.0)) * 0.31830989 * f32(SKIN_H) - 0.5, 0.0, f32(SKIN_H) - 1.0);
  let x0 = floor(x);
  let y0 = floor(y);
  let i0 = u32((i32(x0) + i32(SKIN_W)) % i32(SKIN_W));
  let i1 = (i0 + 1u) % SKIN_W;
  let j0 = u32(y0);
  let j1 = min(j0 + 1u, SKIN_H - 1u);
  let a = mix(skin[j0 * SKIN_W + i0], skin[j0 * SKIN_W + i1], x - x0);
  let b = mix(skin[j1 * SKIN_W + i0], skin[j1 * SKIN_W + i1], x - x0);
  return mix(a, b, y - y0);
}
// How fast the char front eats into the wood (m/s): Arrhenius in the temperature at the front (the surface's while
// the char is thin, the wood's below as it thickens, which is how char's insulation slows a fire down), set so that
// wood chars at about 2 mm a minute at 650 K and ten times slower at 550 K.
fn charRate(s: vec4<f32>) -> f32 {
  let front = mix(s.x, s.y, clamp(s.z / 4.0, 0.0, 1.0));
  return min(9.5 * exp(-8230.0 / max(front, 300.0)), 6e-5);
}
const WOOD_RHO = 500.0;
const CHAR_YIELD = 0.16; // pine's, at fire heating rates (Rinta-Paavola and Hostikka 2022)
// The volatiles the charring drives off (kg / m2 / s): the wood that does not stay behind as char.
fn pyrolysisOf(s: vec4<f32>) -> f32 { return WOOD_RHO * (1.0 - CHAR_YIELD) * charRate(s); }
// What reaches the fire from the ball's surface: its volatiles, once they come off fast enough to burn (about 2 g per
// m2 a second, wood's critical mass flux for flaming); slower, they only scorch it.
fn ballFuelFlux(n: vec3<f32>) -> f32 {
  let m = pyrolysisOf(skinAt(n));
  return m * smoothstep(0.0015, 0.0025, m);
}

// Scalars, at each cell. A smoke source: x smoke density, y the temperature rise over ambient (K). A fire: quantities
// per unit volume, because the solver mixes by volume (interpolating, and mixing finer than the grid) while
// temperature and mass fractions mix by mass, and hot gas is light: kept as they are, a volume of 1300 K gas mixing
// with a volume of 293 K air would come out at 796 K instead of 478 K, and a fire's plume would stay far too hot.
// So y is the density deficit, 1 - rho / rho_ambient (0 in ambient air), and x, z, w are partial densities relative
// to ambient air's: of soot (g per kg), of the gas that came from the fuel bed (the mixture fraction times the relative
// density) and of the fuel still unburned. gasOf() turns them back into a temperature and mass fractions.
const T_AMB = ${wf(T_AMBIENT)}; // K
const RHO_T = ${wf(RHO_T)}; // air density times temperature, kg K / m3
const Y_O2_AIR = ${wf(Y_O2_AIR)};
const S_O2 = ${wf(FUEL.oxygenPerFuel)}; // kg of oxygen burning a kg of fuel takes
const HEAT = ${wf(FUEL.heatPerFuel)}; // J per kg of fuel burned
const CP = ${wf(CP)}; // J / (kg K)
const SOOT_G = 1000.0; // soot is kept in g per kg (fp16 keeps more of its precision there)

struct Gas {
  rise: f32, // temperature over ambient, K
  z: f32, // mixture fraction
  fuel: f32, // unburned fuel, mass fraction
  soot: f32, // g per kg
}
// The hottest a cell's gas gets (K over ambient), well above a wood flame's; reading and writing a cell use the same
// cap, so a cell at it keeps its masses.
const RISE_MAX = 2200.0;
fn gasOf(s: vec4<f32>) -> Gas {
  let r = max(1.0 - s.y, T_AMB / (T_AMB + RISE_MAX)); // rho / rho_ambient = T_ambient / T
  return Gas(T_AMB * max(s.y, 0.0) / r, max(s.z, 0.0) / r, max(s.w, 0.0) / r, max(s.x, 0.0) / r);
}
fn storeGas(g: Gas) -> vec4<f32> {
  let r = T_AMB / (T_AMB + g.rise);
  return vec4<f32>(g.soot * r, 1.0 - r, g.z * r, g.fuel * r);
}
// The temperature rise over ambient (K), from either source's scalars.
fn riseOf(s: vec4<f32>) -> f32 {
  if (P.fire.x > 0.5) { return T_AMB * max(s.y, 0.0) / max(1.0 - s.y, 0.1); }
  return s.y;
}

// The resolved strain rate at a cell, |S| = sqrt(2 S_ij S_ij), from central differences of the cell-centred velocity.
const C_S = ${wf(SMAGORINSKY)};
fn cellVel(vt: texture_3d<f32>, c0: vec3<i32>) -> vec3<f32> {
  let c = clamp(c0, vec3<i32>(0), nCells() - 1);
  let a = textureLoad(vt, c, 0);
  return 0.5 * vec3<f32>(a.x + textureLoad(vt, c + X, 0).x, a.y + textureLoad(vt, c + Y, 0).y, a.z + textureLoad(vt, c + Z, 0).z);
}
fn strainRate(vt: texture_3d<f32>, c: vec3<i32>) -> f32 {
  let h = 2.0 * P.grid.x;
  let gx = (cellVel(vt, c + X) - cellVel(vt, c - X)) / h;
  let gy = (cellVel(vt, c + Y) - cellVel(vt, c - Y)) / h;
  let gz = (cellVel(vt, c + Z) - cellVel(vt, c - Z)) / h;
  let sxy = 0.5 * (gy.x + gx.y);
  let sxz = 0.5 * (gz.x + gx.z);
  let syz = 0.5 * (gz.y + gy.z);
  return sqrt(2.0 * (gx.x * gx.x + gy.y * gy.y + gz.z * gz.z + 2.0 * (sxy * sxy + sxz * sxz + syz * syz)));
}
// How long fuel and air take to mix in a cell, as FDS models it: the faster of the subgrid turbulence's time,
// C_u dx / sqrt(2/3 k_sgs) (C_u = 0.4, which FDS calibrated against Heskestad's flame heights; k_sgs from the eddy
// viscosity, (C_s dx)^2 |S| = 0.1 dx sqrt(k_sgs), Deardorff's relation) and the buoyant time sqrt(2 dx / g). No
// faster than a turbulent flame's 100 m/s crossing the cell.
fn mixTime(strain: f32) -> f32 {
  let dx = P.grid.x;
  let sqrtK = C_S * C_S * dx * strain / 0.1;
  let tauU = 0.4 * dx / max(0.8164966 * sqrtK, 1e-6);
  return max(min(tauU, P.fire2.y), dx / 100.0);
}

// Fast chemistry, mixing-limited (the eddy dissipation concept, as NIST's Fire Dynamics Simulator uses it): in each
// cell, fuel burns with the oxygen mixed in with it as fast as turbulence mixes them, at min(Y_fuel, Y_O2 / s) /
// tau_mix. The oxygen is what the air brought (the mixture fraction says how much of the gas is air) less what burning
// has used. Returns the mass fraction of fuel burned in dt.
fn fuelBurned(g: Gas, dt: f32, tau: f32) -> f32 {
  let yO2 = Y_O2_AIR * (1.0 - g.z) - S_O2 * (g.z - g.fuel);
  return min(g.fuel, max(yO2, 0.0) / S_O2) * (1.0 - exp(-dt / tau));
}
// The fuel given off into a cell in dt, per unit of the cell's mass. The bed's: the rate that releases the chosen heat,
// spread over the bed's surface in proportion to its mask (the mask's area as measured last step; before that, its
// nominal area), into the bottom layer of cells. A burning ball's: into the layer of cells just outside it, each of
// which holds about dx^2 of its surface.
fn fuelAdded(g: Gas, w: vec3<f32>, measuredArea: f32, dt: f32) -> f32 {
  var flux = 0.0; // kg / m2 / s
  let m = bedMask(w);
  if (m > 0.0) {
    let area = select(P.fire2.z, measuredArea, measuredArea > 1e-6);
    flux = P.fire.y * m / area;
  }
  if (ballBurns()) {
    let d = w - P.ball.xyz;
    let gap = length(d) - P.ball.w;
    if (gap > 0.0 && gap <= P.grid.x) { flux += ballFuelFlux(d / length(d)); }
  }
  if (flux <= 0.0) { return 0.0; }
  return flux * dt / (RHO_T / (T_AMB + g.rise) * P.grid.x);
}
// The temperature rise from burning that much fuel, net of the share radiated away.
fn heatRise(burned: f32) -> f32 { return burned * HEAT * (1.0 - P.fire.w) / CP; }
// The rate (1/s) at which a cell's gas grows in volume. At constant pressure, gas heated by dT in dt grows at
// (dT / dt) / T: the heat released last step, which is what the gas here is still taking up. The bed's fuel vapour
// takes up room too (its blowing velocity, a few centimetres a second).
fn expansionOf(g: Gas, hrrHere: f32, w: vec3<f32>, measuredArea: f32) -> f32 {
  let heating = hrrHere * (1.0 - P.fire.w) / (RHO_T / (T_AMB + g.rise) * CP);
  return P.fire2.x * heating / (T_AMB + g.rise) + fuelAdded(g, w, measuredArea, P.grid.y) / P.grid.y;
}
// Fixed point for the mass-keeping pass's atomic sums of trilinear weights.
const SPREAD_ONE = 1048576.0;

const X = vec3<i32>(1, 0, 0);
const Y = vec3<i32>(0, 1, 0);
const Z = vec3<i32>(0, 0, 1);

fn nCells() -> vec3<i32> { return vec3<i32>(P.dims.xyz); }
fn nF() -> vec3<f32> { return vec3<f32>(P.dims.xyz); }
fn worldOf(g: vec3<f32>) -> vec3<f32> { return P.origin.xyz + g * P.grid.x; }
fn cellIndex(c: vec3<i32>) -> u32 { let n = nCells(); return u32(c.x + n.x * (c.y + n.y * c.z)); }

fn hash3(c: vec3<i32>) -> f32 {
  var h = (bitcast<u32>(c.x) * 0x8da6b343u) ^ (bitcast<u32>(c.y) * 0xd8163841u) ^ (bitcast<u32>(c.z) * 0xcb1ab31fu);
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return f32(h) * (1.0 / 4294967296.0);
}

// Value noise in 0..1 with quintic interpolation.
fn vnoise(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let c = vec3<i32>(i);
  let x00 = mix(hash3(c), hash3(c + X), u.x);
  let x10 = mix(hash3(c + Y), hash3(c + X + Y), u.x);
  let x01 = mix(hash3(c + Z), hash3(c + X + Z), u.x);
  let x11 = mix(hash3(c + Y + Z), hash3(c + X + Y + Z), u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
fn fbm(p: vec3<f32>) -> f32 {
  return 0.6 * vnoise(p) + 0.3 * vnoise(p * 2.03 + 17.1) + 0.1 * vnoise(p * 4.07 + 31.7);
}

// The source, its edge one cell soft, broken into puffs by fast-drifting noise: a squat dome on the floor, 2.5 times
// wider than tall (a fire bed, or the mouth of a smoke pot). A taller source fills with gas that has only just left
// it, all of it as dense or as hot as the source makes it, and reads as a solid ball.
fn sourceMask(w: vec3<f32>) -> f32 {
  let r = P.src.w;
  let q = w - P.src.xyz;
  let d = length(vec3<f32>(q.x, q.y * 2.5, q.z));
  let m = 1.0 - smoothstep(r - 2.0 * P.grid.x, r, d);
  if (m <= 0.0) { return 0.0; }
  // Coarse noise drifting in every direction: a pattern held still in x and z would emit the same thin columns
  // step after step, which the updraft stretches into streaks.
  let n = fbm(w * 3.5 + vec3<f32>(0.9, -2.8, 0.7) * P.grid.z);
  return m * mix(1.0, smoothstep(0.35, 0.75, n), P.srcProps.w);
}

// A fire's bed: a disc on the floor that gives off fuel vapour through its surface (into the bottom layer of cells), as
// a pool fire or a burning bed of wood does and as FDS models one, with the same drifting noise as a smoke source.
// Fuel given off inside a volume would have the air the plume draws in sweep straight through it, so the fire would
// burn lean, without a flame's fuel-rich core.
fn bedMask(w: vec3<f32>) -> f32 {
  if (w.y > P.grid.x) { return 0.0; }
  let r = P.src.w;
  let m = 1.0 - smoothstep(r - 2.0 * P.grid.x, r, length(w.xz - P.src.xz));
  if (m <= 0.0) { return 0.0; }
  let n = fbm(w * 3.5 + vec3<f32>(0.9, -2.8, 0.7) * P.grid.z);
  return m * mix(1.0, smoothstep(0.35, 0.75, n), P.srcProps.w);
}

// Staggered velocity, trilinear on each component's own lattice, clamped to that component's range.
fn velDims() -> vec3<f32> { return nF() + 1.0; }
fn comp(t: texture_3d<f32>, s: sampler, tc: vec3<f32>, hi: vec3<f32>) -> vec4<f32> {
  return textureSampleLevel(t, s, (clamp(tc, vec3<f32>(0.0), hi) + 0.5) / velDims(), 0.0);
}
fn uAt(t: texture_3d<f32>, s: sampler, g: vec3<f32>) -> f32 {
  let n = nF();
  return comp(t, s, vec3<f32>(g.x, g.y - 0.5, g.z - 0.5), vec3<f32>(n.x, n.y - 1.0, n.z - 1.0)).x;
}
fn vAt(t: texture_3d<f32>, s: sampler, g: vec3<f32>) -> f32 {
  let n = nF();
  return comp(t, s, vec3<f32>(g.x - 0.5, g.y, g.z - 0.5), vec3<f32>(n.x - 1.0, n.y, n.z - 1.0)).y;
}
fn wAt(t: texture_3d<f32>, s: sampler, g: vec3<f32>) -> f32 {
  let n = nF();
  return comp(t, s, vec3<f32>(g.x - 0.5, g.y - 0.5, g.z), vec3<f32>(n.x - 1.0, n.y - 1.0, n.z)).z;
}
fn velAt(t: texture_3d<f32>, s: sampler, g: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(uAt(t, s, g), vAt(t, s, g), wAt(t, s, g));
}
// Where the fluid at grid point g came from one step ago (sgn 1), or goes to (sgn -1): second-order Runge-Kutta.
fn backtrace(t: texture_3d<f32>, s: sampler, g: vec3<f32>, sgn: f32) -> vec3<f32> {
  let k = sgn * P.grid.y / P.grid.x;
  let v1 = velAt(t, s, g);
  let v2 = velAt(t, s, g - 0.5 * k * v1);
  return g - k * v2;
}

fn scalAt(t: texture_3d<f32>, s: sampler, g: vec3<f32>) -> vec4<f32> {
  let n = nF();
  return textureSampleLevel(t, s, (clamp(g - 0.5, vec3<f32>(0.0), n - 1.0) + 0.5) / n, 0.0);
}
// Inside the domain, or below the floor (clamped to it); past an open side or the top, the air carries no smoke.
fn inDomain(g: vec3<f32>) -> bool {
  let n = nF();
  return g.x >= 0.0 && g.x <= n.x && g.z >= 0.0 && g.z <= n.z && g.y <= n.y;
}
`

// 1. Advect the velocity by itself: semi-Lagrangian with an RK2 backtrace through the flow field, DIR 1 tracing back (the
// forward step), -1 tracing forward (MacCormack's reverse step, below).
export const ADVECT_VEL = /* wgsl */ `
${SIM_COMMON}
override DIR: f32 = 1.0;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var flow: texture_3d<f32>;
@group(0) @binding(3) var velSrc: texture_3d<f32>;
@group(0) @binding(4) var velDst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  let f = vec3<f32>(id);
  var o = vec3<f32>(0.0);
  if (id.y < n.y && id.z < n.z) {
    o.x = uAt(velSrc, samp, backtrace(flow, samp, vec3<f32>(f.x, f.y + 0.5, f.z + 0.5), DIR));
  }
  if (id.x < n.x && id.z < n.z && id.y > 0u) {
    o.y = vAt(velSrc, samp, backtrace(flow, samp, vec3<f32>(f.x + 0.5, f.y, f.z + 0.5), DIR));
  }
  if (id.x < n.x && id.y < n.y) {
    o.z = wAt(velSrc, samp, backtrace(flow, samp, vec3<f32>(f.x + 0.5, f.y + 0.5, f.z), DIR));
  }
  textureStore(velDst, vec3<i32>(id), vec4<f32>(o, 0.0));
}
`

// 1b. Modified MacCormack for the velocity (Selle et al. 2008), as for the scalars: the forward result plus half the
// round-trip error, each component clamped to the values on its own lattice that the forward step interpolated
// between. Semi-Lagrangian alone damps the eddies that draw air into a plume, so a fire's plume would mix too
// slowly with the air around it: too hot, and its flame too tall.
export const VEL_COMBINE = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var v0: texture_3d<f32>;
@group(0) @binding(3) var v1: texture_3d<f32>;
@group(0) @binding(4) var v2: texture_3d<f32>;
@group(0) @binding(5) var dst: texture_storage_3d<rgba16float, write>;

// The value clamped to the range of component k of v0 at the 8 lattice points around tc (that lattice's texel
// coordinates; hi its last texel).
fn clampFoot(tc: vec3<f32>, hi: vec3<i32>, k: i32, value: f32) -> f32 {
  let b = vec3<i32>(floor(clamp(tc, vec3<f32>(0.0), vec3<f32>(hi))));
  var lo = 1e9;
  var up = -1e9;
  for (var j = 0; j < 8; j++) {
    let q = textureLoad(v0, min(b + vec3<i32>(j & 1, (j >> 1) & 1, (j >> 2) & 1), hi), 0)[k];
    lo = min(lo, q);
    up = max(up, q);
  }
  return clamp(value, lo, up);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  let c = vec3<i32>(id);
  let ni = vec3<i32>(n);
  let f = vec3<f32>(id);
  let a1 = textureLoad(v1, c, 0).xyz;
  let corrected = a1 + 0.5 * (textureLoad(v0, c, 0).xyz - textureLoad(v2, c, 0).xyz);
  var o = vec3<f32>(0.0);
  if (id.y < n.y && id.z < n.z) {
    let g = backtrace(v0, samp, vec3<f32>(f.x, f.y + 0.5, f.z + 0.5), 1.0);
    o.x = clampFoot(vec3<f32>(g.x, g.y - 0.5, g.z - 0.5), vec3<i32>(ni.x, ni.y - 1, ni.z - 1), 0, corrected.x);
  }
  if (id.x < n.x && id.z < n.z && id.y > 0u) {
    let g = backtrace(v0, samp, vec3<f32>(f.x + 0.5, f.y, f.z + 0.5), 1.0);
    o.y = clampFoot(vec3<f32>(g.x - 0.5, g.y, g.z - 0.5), vec3<i32>(ni.x - 1, ni.y, ni.z - 1), 1, corrected.y);
  }
  if (id.x < n.x && id.y < n.y) {
    let g = backtrace(v0, samp, vec3<f32>(f.x + 0.5, f.y + 0.5, f.z), 1.0);
    o.z = clampFoot(vec3<f32>(g.x - 0.5, g.y - 0.5, g.z), vec3<i32>(ni.x - 1, ni.y - 1, ni.z), 2, corrected.z);
  }
  textureStore(dst, c, vec4<f32>(o, 0.0));
}
`

// 2. Vorticity at cell centres (from cell-centred velocity, central differences), and its magnitude. And, when IVOCK
// or leapfrog flow maps run, the momentum's subgrid viscosity: Smagorinsky's eddy viscosity (C_s dx)^2 |S|, as FDS uses
// for momentum, kept as the explicit diffusion step's weight nu dt / dx^2 (over LFM's leapfrog steps, their length; at
// most 0.15, inside its stability limit). Otherwise the solver's own numerical dissipation plays that part; both take
// the numerical dissipation away, and the flow then piled energy into the smallest eddies until it ran away.
export const CURL = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var curlOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var nutOut: texture_storage_3d<rgba16float, write>;

fn vc(c0: vec3<i32>) -> vec3<f32> {
  let c = clamp(c0, vec3<i32>(0), nCells() - 1);
  let a = textureLoad(vel, c, 0);
  return 0.5 * vec3<f32>(a.x + textureLoad(vel, c + X, 0).x, a.y + textureLoad(vel, c + Y, 0).y, a.z + textureLoad(vel, c + Z, 0).z);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let dx = vc(c + X) - vc(c - X);
  let dy = vc(c + Y) - vc(c - Y);
  let dz = vc(c + Z) - vc(c - Z);
  let w = vec3<f32>(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) / (2.0 * P.grid.x);
  textureStore(curlOut, c, vec4<f32>(w, length(w)));
  var k = 0.0;
  if (P.stirR.z > 0.5) {
    k = min(C_S * C_S * strainRate(vel, c) * P.grid.y * select(1.0, P.stirR.y, P.stirR.y > 0.0), 0.15);
  }
  textureStore(nutOut, c, vec4<f32>(k, 0.0, 0.0, 0.0));
}
`

// 2b. IVOCK (Zhang, Bridson and Greif 2015): restoring the vorticity that advecting the velocity loses. The vorticity
// of the step's velocity is stretched by the flow, (w . grad u) dt, taken as the velocity's difference across a
// vortex segment one cell long, h |w| / h (the paper's form), and for a fire scaled by exp(-div u dt), as burning gas
// expands (its fire variant). It is then advected with the velocity's own scheme (with MacCormack's, when the velocity
// uses it: paired with plain semi-Lagrangian vorticity, the correction ran away), and what the advected velocity's
// curl falls short of it is the loss, dw. A vector potential with (-lap + 1 / l^2) psi = dw (three screened Poisson
// solves, one red-black Gauss-Seidel sweep of all three at once each step, warm started: the screening makes the equation
// local, so it needs no multigrid) gives the velocity that restores it, curl psi, added before the forces and the projection. The
// screening, l four cells, is this lab's: the loss advection causes is at small scales. Simplifications: vorticity and potential at cell centres rather than the paper's cell edges, and
// psi = 0 past the open sides rather than its open-space values. Masked out within two cells of the floor, the open
// sides and the collision ball, where shearing boundary layers make it unstable. Two more of this lab's: only lost
// magnitude along the advected vorticity's own direction is restored (unrestricted, the correction fed on itself and
// ran away), and it runs for a smoke source only: in a fire it mixed the plume far too fast (its centreline 10% of
// McCaffrey's temperatures), most likely from the fire's strong expansion meeting this cell-centred form.
export const VORT_BEFORE = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let h = P.grid.x;
  let dx = cellVel(vel, c + X) - cellVel(vel, c - X);
  let dy = cellVel(vel, c + Y) - cellVel(vel, c - Y);
  let dz = cellVel(vel, c + Z) - cellVel(vel, c - Z);
  var w = vec3<f32>(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) / (2.0 * h);
  let mag = length(w);
  if (mag > 1e-6) {
    let g = vec3<f32>(id) + 0.5;
    let half = 0.5 * w / mag;
    let du = velAt(vel, samp, g + half) - velAt(vel, samp, g - half);
    w += P.grid.y * mag * du / h;
  }
  if (P.fire.x > 0.5) {
    let a = textureLoad(vel, c, 0);
    let div = (textureLoad(vel, c + X, 0).x - a.x + textureLoad(vel, c + Y, 0).y - a.y + textureLoad(vel, c + Z, 0).z - a.z) / h;
    // (Only expansion: a pressure solve's leftover convergence would otherwise grow it exponentially.)
    w *= exp(-clamp(div * P.grid.y, 0.0, 5.0));
  }
  textureStore(dst, c, vec4<f32>(w, 0.0));
}
`

export const IVOCK_TARGET = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var velOld: texture_3d<f32>; // the step's velocity, before advection
@group(0) @binding(3) var velAdv: texture_3d<f32>; // the advected velocity
@group(0) @binding(4) var<storage, read_write> rhs: array<vec4<f32>>;
@group(0) @binding(7) var vort0: texture_3d<f32>; // the stretched vorticity, before advection
@group(0) @binding(8) var vort1: texture_3d<f32>; // advected forward
@group(0) @binding(9) var vort2: texture_3d<f32>; // and back (MacCormack's round trip)

fn ivockMask(c: vec3<i32>) -> f32 {
  let n = nCells();
  // Cells from the floor, and from the nearest open side or the top: the correction fades in from 2 to 6 of them.
  let side = f32(min(min(min(c.x, n.x - 1 - c.x), min(c.z, n.z - 1 - c.z)), n.y - 1 - c.y));
  var m = smoothstep(2.0, 6.0, side) * smoothstep(1.0, 3.0, f32(c.y));
  if (P.ball.w > 0.0) {
    let gap = distance(worldOf(vec3<f32>(c) + 0.5), P.ball.xyz) - P.ball.w;
    m *= smoothstep(1.0, 2.5, gap / P.grid.x);
  }
  // (And near the reference set's balls.)
  if (P.refGrey.w > 0.0) {
    let wc = worldOf(vec3<f32>(c) + 0.5);
    let gap = min(distance(wc, P.refGrey.xyz) - P.refGrey.w, distance(wc, P.refChrome.xyz) - P.refChrome.w);
    m *= smoothstep(1.0, 2.5, gap / P.grid.x);
  }
  return m;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  let n = nCells();
  if (any(c >= n)) { return; }
  let i = cellIndex(c);
  // The vorticity advected: semi-Lagrangian, or with MacCormack's correction clamped to the values it interpolated.
  var w = textureLoad(vort1, c, 0).xyz;
  if (P.grid.w > 0.5) {
    let g = backtrace(velOld, samp, vec3<f32>(id) + 0.5, 1.0);
    if (inDomain(g)) {
      let b = vec3<i32>(floor(clamp(g - 0.5, vec3<f32>(0.0), vec3<f32>(n - 1))));
      var lo = vec3<f32>(1e9);
      var hi = vec3<f32>(-1e9);
      for (var k = 0; k < 8; k++) {
        let q = textureLoad(vort0, min(b + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1), n - 1), 0).xyz;
        lo = min(lo, q);
        hi = max(hi, q);
      }
      w = clamp(w + 0.5 * (textureLoad(vort0, c, 0).xyz - textureLoad(vort2, c, 0).xyz), lo, hi);
    }
  }
  // The advected velocity's own curl.
  let h = 2.0 * P.grid.x;
  let ax = cellVel(velAdv, c + X) - cellVel(velAdv, c - X);
  let ay = cellVel(velAdv, c + Y) - cellVel(velAdv, c - Y);
  let az = cellVel(velAdv, c + Z) - cellVel(velAdv, c - Z);
  let have = vec3<f32>(ay.z - az.y, az.x - ax.z, ax.y - ay.x) / h;
  // (Where burning gas expands strongly the correction is unreliable, so it fades out there.)
  let a0 = textureLoad(velOld, c, 0);
  let div = (textureLoad(velOld, c + X, 0).x - a0.x + textureLoad(velOld, c + Y, 0).y - a0.y + textureLoad(velOld, c + Z, 0).z - a0.z) / P.grid.x;
  let swell = div * P.grid.y / 0.02;
  // Only lost magnitude along the advected vorticity's own direction is restored: the correction never adds vorticity
  // the flow did not carry, and never more than it carried.
  let wl = length(w);
  var dw = vec3<f32>(0.0);
  if (wl > 1e-4) {
    let dir = w / wl;
    dw = dir * clamp(dot(w - have, dir), 0.0, wl) * ivockMask(c) / (1.0 + swell * swell);
  }
  rhs[i] = vec4<f32>(dw, 0.0);
}
`

// The potentials' screened Poisson equation, (n + h^2 / l^2) psi - sum of neighbours = h^2 dw, all three components in
// one red-black Gauss-Seidel sweep: psi = 0 past the open sides and the top, nothing below the floor or inside the
// collision ball (as the pressure's).
export const IVOCK_SMOOTH = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var<uniform> S: vec4<u32>; // x: the colour updated in this pass
@group(0) @binding(2) var<storage, read> rhs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> psi: array<vec4<f32>>;
const SCREEN = ${wf(1 / 16)}; // (h / l)^2, l = 4 cells

fn psiAt(i: i32) -> vec3<f32> { return psi[i].xyz; }

@compute @workgroup_size(16, 4, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nCells();
  let c = vec3<i32>(i32(2u * id.x + ((id.y + id.z + S.x) & 1u)), i32(id.y), i32(id.z));
  if (any(c >= n)) { return; }
  let i = i32(cellIndex(c));
  if (cellSolid(c)) {
    psi[i] = vec4<f32>(0.0);
    return;
  }
  let sy = n.x;
  let sz = n.x * n.y;
  var sum = vec3<f32>(0.0);
  var cnt = SCREEN;
  if (c.x > 0) { if (!cellSolid(c - X)) { sum += psiAt(i - 1); cnt += 1.0; } } else { cnt += 1.0; }
  if (c.x < n.x - 1) { if (!cellSolid(c + X)) { sum += psiAt(i + 1); cnt += 1.0; } } else { cnt += 1.0; }
  if (c.z > 0) { if (!cellSolid(c - Z)) { sum += psiAt(i - sz); cnt += 1.0; } } else { cnt += 1.0; }
  if (c.z < n.z - 1) { if (!cellSolid(c + Z)) { sum += psiAt(i + sz); cnt += 1.0; } } else { cnt += 1.0; }
  if (c.y > 0 && !cellSolid(c - Y)) { sum += psiAt(i - sy); cnt += 1.0; }
  if (c.y < n.y - 1) { if (!cellSolid(c + Y)) { sum += psiAt(i + sy); cnt += 1.0; } } else { cnt += 1.0; }
  let h2 = P.grid.x * P.grid.x;
  psi[i] = vec4<f32>((sum + rhs[i].xyz * h2) / cnt, 0.0);
}
`

// The correction, curl psi, at each cell centre (central differences; psi = 0 past the domain), then on each face
// the mean of the two cells it divides, added to the advected velocity. The domain's own boundary faces are left alone.
export const IVOCK_CURL = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var<storage, read> psi: array<vec4<f32>>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;

fn psiAt(c: vec3<i32>) -> vec3<f32> {
  if (any(c < vec3<i32>(0)) || any(c >= nCells())) { return vec3<f32>(0.0); }
  return psi[cellIndex(c)].xyz;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let h = 2.0 * P.grid.x;
  let gx = (psiAt(c + X) - psiAt(c - X)) / h;
  let gy = (psiAt(c + Y) - psiAt(c - Y)) / h;
  let gz = (psiAt(c + Z) - psiAt(c - Z)) / h;
  textureStore(dst, c, vec4<f32>(gy.z - gz.y, gz.x - gx.z, gx.y - gy.x, 0.0));
}
`

export const IVOCK_APPLY = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var velAdv: texture_3d<f32>;
@group(0) @binding(2) var curlPsi: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nCells();
  let f = vec3<i32>(id);
  if (any(f > n)) { return; }
  var u = textureLoad(velAdv, f, 0);
  // Face f.x sits between cells f - X and f along x; likewise y and z.
  if (f.x > 0 && f.x < n.x && f.y < n.y && f.z < n.z) { u.x += 0.5 * (textureLoad(curlPsi, f - X, 0).x + textureLoad(curlPsi, f, 0).x); }
  if (f.y > 0 && f.y < n.y && f.x < n.x && f.z < n.z) { u.y += 0.5 * (textureLoad(curlPsi, f - Y, 0).y + textureLoad(curlPsi, f, 0).y); }
  if (f.z > 0 && f.z < n.z && f.x < n.x && f.y < n.y) { u.z += 0.5 * (textureLoad(curlPsi, f - Z, 0).z + textureLoad(curlPsi, f, 0).z); }
  textureStore(dst, f, u);
}
`

// 2c. Leapfrog flow maps for a smoke source's velocity (Sun, Li, Wang, Wang, Li, van Bloemen Waanders and Zhu 2025).
// The velocity is carried as impulse along flow maps over a cycle of LFM_N steps, rather than re-sampled each step.
// Within a cycle, each step makes a half-step velocity by leapfrog: the one two half-steps back, plus the forces,
// advected over 2 dt by the one between (the first two steps start it over dt/2 and dt), then projected. These
// half-step velocities are kept in a ring. The forward map, and the column of its Jacobian each face needs, is marched
// through them step by step (RK4, with the Jacobian marched alongside), and the step's forces are gathered back along
// it into the cycle's initial velocity (the paper's path integral). At the cycle's end the backward map is marched
// back through the ring the same way, and the new velocity is the initial one's impulse pulled back along it, m =
// T^T u0(Psi), with the error of a round trip through both maps taken off (BFECC), clamped, and projected. It then
// starts the next cycle.
// This lab's version differs from the paper in four ways. It marches through a cell-centred copy of each ring velocity,
// sampled trilinearly, and takes the Jacobian's product with a column as a central difference along it, where the paper
// samples with a 27-tap quadratic B-spline: the paper's way costs some 80 loads a sample, past this GPU's budget. Its
// domain has open sides, which hold the pressure at zero, where the paper's examples are closed boxes: the impulse
// differs from the velocity by the gradient of a gauge, which grows along each path at p - |u|^2 / 2, and an open side
// does not respect it, so the gauge is marched back along the same maps and its gradient taken off. Near an open side,
// on a path that came in through one, and where a map has distorted too far within a cycle, the maps carry the velocity
// itself, less the gradient of the pressure gathered along the path. And, as with IVOCK, once the numerical dissipation
// is gone the flow needs a real sink for its smallest eddies: Smagorinsky's eddy viscosity (2). Without any one of
// these, wind, a stir or the finer grid ran away within seconds. Its projections are this solver's multigrid. Map
// positions are in grid units (cell centres at i + 0.5), velocities in m/s, Jacobian columns dimensionless.
const LFM_COMMON = /* wgsl */ `
const LFM_N = ${LFM_N}u;
// Face texel id's position on lattice a (0: x faces, 1: y, 2: z), in grid units.
fn facePos(id: vec3<u32>, a: u32) -> vec3<f32> {
  let f = vec3<f32>(id) + 0.5;
  if (a == 0u) { return vec3<f32>(f.x - 0.5, f.y, f.z); }
  if (a == 1u) { return vec3<f32>(f.x, f.y - 0.5, f.z); }
  return vec3<f32>(f.x, f.y, f.z - 0.5);
}
fn faceValid(id: vec3<u32>, a: u32) -> bool {
  let n = P.dims.xyz;
  if (a == 0u) { return id.x <= n.x && id.y < n.y && id.z < n.z; }
  if (a == 1u) { return id.x < n.x && id.y <= n.y && id.z < n.z; }
  return id.x < n.x && id.y < n.y && id.z <= n.z;
}
fn faceIndex(id: vec3<u32>, a: u32) -> u32 {
  let m = P.dims.xyz + 1u;
  return a * m.x * m.y * m.z + id.x + m.x * (id.y + m.y * id.z);
}
fn unitCol(a: u32) -> vec3<f32> { return select(select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), a == 1u), vec3<f32>(1.0, 0.0, 0.0), a == 0u); }
// How far a map's Jacobian column counts: from the two ends of its path, fully when both are 6 cells or more inside
// the open sides and the top, not at all when either is within 2 (or outside); and from the column itself, fully while
// it has stretched or shrunk by less than 2 times, not at all past 3 (the distortion the paper re-initialises to
// avoid, which a hard stir on the finer grid reaches within a cycle). Elsewhere the maps carry the velocity itself.
fn impulseWeight(g: vec3<f32>) -> f32 {
  let n = nF();
  let d = min(min(min(g.x, n.x - g.x), min(g.z, n.z - g.z)), n.y - g.y);
  return smoothstep(2.0, 6.0, d);
}
fn colWeight(c: vec3<f32>, g: vec3<f32>, end: vec3<f32>) -> f32 {
  let l = length(c);
  return impulseWeight(g) * impulseWeight(end) * (1.0 - smoothstep(2.0, 3.0, max(l, 1.0 / max(l, 1e-3))));
}
fn weightedCol(c: vec3<f32>, a: u32, g: vec3<f32>, end: vec3<f32>) -> vec3<f32> { return mix(unitCol(a), c, colWeight(c, g, end)); }
`

// The ring's cell-centred copies, for marching: each cell the mean of its faces, and the step's pressure (solved for
// its forces' span of steps, so divided by it), which the impulse's gauge gathers.
export const LFM_CELLVEL = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> pressure: array<f32>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let u = cellVel(vel, c);
  textureStore(dst, c, vec4<f32>(u, pressure[cellIndex(c)] / max(P.stirR.y, 0.5)));
}
`

// One RK4 step of a map and its Jacobian column through a cell-centred velocity (h > 0 marches back in time).
const LFM_RK4 = /* wgsl */ `
fn velC(t: texture_3d<f32>, s: sampler, g: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(t, s, clamp(g, vec3<f32>(0.5), nF() - 0.5) / nF(), 0.0).xyz;
}
// (c . grad) u, per second: a central difference half a cell either way along c.
fn gradDot(t: texture_3d<f32>, s: sampler, g: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
  return (velC(t, s, g + 0.5 * c) - velC(t, s, g - 0.5 * c)) / P.grid.x;
}
struct March { x: vec3<f32>, c: vec3<f32> }
fn rk4(t: texture_3d<f32>, s: sampler, m0: March, h: f32) -> March {
  let k = h / P.grid.x;
  let u1 = velC(t, s, m0.x);
  let a1 = gradDot(t, s, m0.x, m0.c);
  let x2 = m0.x - 0.5 * k * u1;
  let c2 = m0.c - 0.5 * h * a1;
  let u2 = velC(t, s, x2);
  let a2 = gradDot(t, s, x2, c2);
  let x3 = m0.x - 0.5 * k * u2;
  let c3 = m0.c - 0.5 * h * a2;
  let u3 = velC(t, s, x3);
  let a3 = gradDot(t, s, x3, c3);
  let x4 = m0.x - k * u3;
  let c4 = m0.c - h * a3;
  let u4 = velC(t, s, x4);
  let a4 = gradDot(t, s, x4, c4);
  return March(m0.x - k / 6.0 * (u1 + 2.0 * u2 + 2.0 * u3 + u4), m0.c - h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4));
}
`

// Each step: the forward map and its Jacobian columns one step on through the step's half-step velocity, and the
// step's forces gathered back along it into the cycle's initial velocity, u0 += dt F^T f, at the step's midpoint.
// The forces acted over k_f steps (the leapfrog's), so dt f = delta / k_f.
export const LFM_MARCH_FWD = /* wgsl */ `
${SIM_COMMON}
${LFM_COMMON}
${LFM_RK4}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var mid: texture_3d<f32>;
@group(0) @binding(3) var delta: texture_3d<f32>;
@group(0) @binding(4) var u0In: texture_3d<f32>;
@group(0) @binding(5) var u0Out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var<storage, read_write> phi: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> fcol: array<vec4<f32>>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  var u0 = textureLoad(u0In, vec3<i32>(id), 0);
  let kf = max(P.stirR.y, 0.5);
  for (var a = 0u; a < 3u; a++) {
    if (!faceValid(id, a)) { continue; }
    let i = faceIndex(id, a);
    var m0 = March(facePos(id, a), unitCol(a));
    if (P.stirR.x > 0.5) { m0 = March(phi[i].xyz, fcol[i].xyz); }
    // The force at the step's midpoint, gathered with the midpoint's column.
    let um = velC(mid, samp, m0.x);
    let xm = m0.x + 0.5 * P.grid.y / P.grid.x * um;
    let cm = m0.c + 0.5 * P.grid.y * gradDot(mid, samp, m0.x, m0.c);
    u0[a] += dot(weightedCol(cm, a, facePos(id, a), xm), velAt(delta, samp, xm)) / kf;
    let m1 = rk4(mid, samp, m0, -P.grid.y);
    phi[i] = vec4<f32>(m1.x, 1.0);
    fcol[i] = vec4<f32>(m1.c, 0.0);
  }
  textureStore(u0Out, vec3<i32>(id), u0);
}
`

// At the cycle's end: the backward map and its Jacobian columns from each face back through the ring.
export const LFM_MARCH_BWD = /* wgsl */ `
${SIM_COMMON}
${LFM_COMMON}
${LFM_RK4}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var mid0: texture_3d<f32>;
@group(0) @binding(3) var mid1: texture_3d<f32>;
@group(0) @binding(4) var mid2: texture_3d<f32>;
@group(0) @binding(5) var mid3: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> psi: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> tcol: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> lam: array<vec2<f32>>;
// The gauge: one RK4 step back of a cell's path, the pressure and |u|^2 / 2 gathered at the same stages.
struct Gauge { x: vec3<f32>, l: vec2<f32> }
fn ringAt(t: texture_3d<f32>, s: sampler, g: vec3<f32>) -> vec4<f32> {
  return textureSampleLevel(t, s, clamp(g, vec3<f32>(0.5), nF() - 0.5) / nF(), 0.0);
}
fn gaugeStep(t: texture_3d<f32>, s: sampler, g0: Gauge, h: f32) -> Gauge {
  let k = h / P.grid.x;
  let a1 = ringAt(t, s, g0.x);
  let a2 = ringAt(t, s, g0.x - 0.5 * k * a1.xyz);
  let a3 = ringAt(t, s, g0.x - 0.5 * k * a2.xyz);
  let a4 = ringAt(t, s, g0.x - k * a3.xyz);
  let f = vec4<f32>(dot(a1.xyz, a1.xyz), dot(a2.xyz, a2.xyz), dot(a3.xyz, a3.xyz), dot(a4.xyz, a4.xyz)) * 0.5;
  let ke = f.x + 2.0 * f.y + 2.0 * f.z + f.w;
  return Gauge(g0.x - k / 6.0 * (a1.xyz + 2.0 * a2.xyz + 2.0 * a3.xyz + a4.xyz), g0.l + h / 6.0 * vec2<f32>(a1.w + 2.0 * a2.w + 2.0 * a3.w + a4.w, ke));
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  if (all(id < n)) {
    var q = Gauge(vec3<f32>(id) + 0.5, vec2<f32>(0.0));
    q = gaugeStep(mid3, samp, q, P.grid.y);
    q = gaugeStep(mid2, samp, q, P.grid.y);
    q = gaugeStep(mid1, samp, q, P.grid.y);
    q = gaugeStep(mid0, samp, q, P.grid.y);
    lam[cellIndex(vec3<i32>(id))] = q.l;
  }
  for (var a = 0u; a < 3u; a++) {
    if (!faceValid(id, a)) { continue; }
    var m = March(facePos(id, a), unitCol(a));
    m = rk4(mid3, samp, m, P.grid.y);
    m = rk4(mid2, samp, m, P.grid.y);
    m = rk4(mid1, samp, m, P.grid.y);
    m = rk4(mid0, samp, m, P.grid.y);
    let i = faceIndex(id, a);
    psi[i] = vec4<f32>(m.x, 1.0);
    tcol[i] = vec4<f32>(m.c, 0.0);
  }
}
`

// The pullbacks at the cycle's end. MODE 1: the impulse, m = T^T u0(Psi). MODE 2: the round trip's error, e = F^T
// m(Phi) - u0. MODE 3: m - T^T e(Psi) / 2, clamped to the range of m at the face and its six neighbours on its
// lattice (the paper's optional clamp), and a smoke source's push applied again: it sets the air there to its speed,
// and the next cycle's first step, half as long, counts what it changes twice (its forces act over half a step), so
// the cycle should start at that speed already.
export const LFM_PULL = /* wgsl */ `
${SIM_COMMON}
${LFM_COMMON}
override MODE: u32 = 1u;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> mapPos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> mapCol: array<vec4<f32>>;
@group(0) @binding(4) var src: texture_3d<f32>;
@group(0) @binding(5) var aux: texture_3d<f32>;
@group(0) @binding(6) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var<storage, read> lam: array<vec2<f32>>;
fn lamAt(c: vec3<i32>) -> vec2<f32> { return lam[cellIndex(clamp(c, vec3<i32>(0), nCells() - 1))]; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  let c = vec3<i32>(id);
  let hi = vec3<i32>(n);
  let here = textureLoad(aux, c, 0);
  let lamHere = lamAt(c);
  var out = vec4<f32>(0.0);
  for (var a = 0u; a < 3u; a++) {
    if (!faceValid(id, a)) { continue; }
    let i = faceIndex(id, a);
    let wgt = colWeight(mapCol[i].xyz, facePos(id, a), mapPos[i].xyz);
    let col = mix(unitCol(a), mapCol[i].xyz, wgt);
    let v = dot(col, velAt(src, samp, mapPos[i].xyz));
    if (MODE == 1u) { out[a] = v; }
    if (MODE == 2u) { out[a] = v - here[a]; }
    if (MODE == 3u) {
      var lo = here[a];
      var up = here[a];
      for (var k = 0; k < 6; k++) {
        var o = vec3<i32>(0);
        o[k / 2] = select(-1, 1, (k & 1) == 1);
        let q = textureLoad(aux, clamp(c + o, vec3<i32>(0), hi), 0)[a];
        lo = min(lo, q);
        up = max(up, q);
      }
      var r = clamp(here[a] - 0.5 * v, lo, up);
      // The gauge off. The impulse less the gradient of what it gathered, p - |u|^2 / 2, is the velocity; where the
      // maps carry the velocity itself, it lacks only the pressure's push, the gradient of the pressure gathered.
      var o = vec3<i32>(0);
      o[a] = 1;
      let gl = (lamHere - lamAt(c - o)) / P.grid.x;
      r -= gl.x - wgt * gl.y;
      out[a] = r;
    }
  }
  if (MODE == 3u && P.fire.x < 0.5 && faceValid(id, 1u) && id.y > 0u) {
    out.y = mix(out.y, P.srcProps.z, sourceMask(worldOf(facePos(id, 1u))));
  }
  textureStore(dst, c, out);
}
`

// 3. Forces on each face: vorticity confinement (Fedkiw, Stam and Jensen 2001), buoyancy from temperature,
// curl-noise turbulence where there is smoke, wind, the source's upward push, the visitor's
// stir, and drag. The floor face stays at rest.
export const FORCES = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var velIn: texture_3d<f32>;
@group(0) @binding(2) var scal: texture_3d<f32>;
@group(0) @binding(3) var curlT: texture_3d<f32>;
@group(0) @binding(4) var velOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var nutT: texture_3d<f32>;
// What this step changed, for LFM's path integral.
@group(0) @binding(6) var deltaOut: texture_storage_3d<rgba16float, write>;

const WIND_DRAG = 0.4; // 1/s: how quickly the air takes up the wind

fn cellCurl(c: vec3<i32>) -> vec4<f32> { return textureLoad(curlT, clamp(c, vec3<i32>(0), nCells() - 1), 0); }
fn cellScal(c: vec3<i32>) -> vec4<f32> { return textureLoad(scal, clamp(c, vec3<i32>(0), nCells() - 1), 0); }

// Confinement force at a cell centre: eps dx (N x w), N the unit gradient of the vorticity's magnitude.
fn confine(c0: vec3<i32>) -> vec3<f32> {
  let c = clamp(c0, vec3<i32>(0), nCells() - 1);
  let w = cellCurl(c);
  let g = vec3<f32>(cellCurl(c + X).w - cellCurl(c - X).w, cellCurl(c + Y).w - cellCurl(c - Y).w, cellCurl(c + Z).w - cellCurl(c - Z).w);
  let l = length(g);
  if (l < 1e-5) { return vec3<f32>(0.0); }
  return P.forces.z * P.grid.x * cross(g / l, w.xyz);
}

// Turbulence: the curl of a noise vector potential (divergence free by construction), one component at a time. The
// potential rises with the plume and churns as it rises (value noise in time as well as space, a new slice every
// TURB_CHURN seconds): held still, it would push the gas the same way at the same places step after step, and the
// updraft would draw those pushes out into steady vertical lanes.
const TURB_RISE = 1.0; // m/s
const TURB_CHURN = 0.35; // s
fn psi(q: vec3<f32>, k: f32) -> f32 {
  let s = P.grid.z / TURB_CHURN;
  let i = floor(s);
  let f = s - i;
  let o = vec3<f32>(k * 37.1, k * 11.7, k * 23.3);
  let a = vnoise(q + o + vec3<f32>(i * 17.31, i * 7.13, i * 29.71));
  let b = vnoise(q + o + vec3<f32>(i * 17.31 + 17.31, i * 7.13 + 7.13, i * 29.71 + 29.71));
  return mix(a, b, f * f * (3.0 - 2.0 * f)) - 0.5;
}
const H = 0.25;
fn curlX(q: vec3<f32>) -> f32 {
  return ((psi(q + vec3<f32>(0.0, H, 0.0), 2.0) - psi(q - vec3<f32>(0.0, H, 0.0), 2.0)) - (psi(q + vec3<f32>(0.0, 0.0, H), 1.0) - psi(q - vec3<f32>(0.0, 0.0, H), 1.0))) / (2.0 * H);
}
fn curlY(q: vec3<f32>) -> f32 {
  return ((psi(q + vec3<f32>(0.0, 0.0, H), 0.0) - psi(q - vec3<f32>(0.0, 0.0, H), 0.0)) - (psi(q + vec3<f32>(H, 0.0, 0.0), 2.0) - psi(q - vec3<f32>(H, 0.0, 0.0), 2.0))) / (2.0 * H);
}
fn curlZ(q: vec3<f32>) -> f32 {
  return ((psi(q + vec3<f32>(H, 0.0, 0.0), 1.0) - psi(q - vec3<f32>(H, 0.0, 0.0), 1.0)) - (psi(q + vec3<f32>(0.0, H, 0.0), 0.0) - psi(q - vec3<f32>(0.0, H, 0.0), 0.0))) / (2.0 * H);
}
fn noiseSpace(w: vec3<f32>) -> vec3<f32> {
  return (w - vec3<f32>(0.0, TURB_RISE * P.grid.z, 0.0)) / max(P.forces2.x, 0.01);
}
// Turbulence acts where there is smoke, or, in a fire, hot gas.
fn turbMask(a: vec4<f32>, b: vec4<f32>) -> f32 {
  let s = 0.5 * (a + b);
  return clamp(select(3.0 * s.x, riseOf(s) / 300.0, P.fire.x > 0.5), 0.0, 1.0);
}

// The eddy viscosity's step at a face: its weight (the two cells' mean) times the face lattice's Laplacian of each
// component (each component on its own lattice; past the lattice's edge, its value at the edge).
fn nutAt(c: vec3<i32>) -> f32 { return textureLoad(nutT, clamp(c, vec3<i32>(0), nCells() - 1), 0).x; }
fn faceVel(c: vec3<i32>) -> vec3<f32> { return textureLoad(velIn, clamp(c, vec3<i32>(0), vec3<i32>(P.dims.xyz)), 0).xyz; }
fn eddyDiffusion(c: vec3<i32>) -> vec3<f32> {
  let u = faceVel(c);
  let lap = faceVel(c + X) + faceVel(c - X) + faceVel(c + Y) + faceVel(c - Y) + faceVel(c + Z) + faceVel(c - Z) - 6.0 * u;
  let k = vec3<f32>(0.5 * (nutAt(c - X) + nutAt(c)), 0.5 * (nutAt(c - Y) + nutAt(c)), 0.5 * (nutAt(c - Z) + nutAt(c)));
  return k * lap;
}

// The stir: a velocity change along the visitor's drag, falling off from where the pointer's ray passes the plume.
fn stir(w: vec3<f32>) -> vec3<f32> {
  if (P.stirD.w <= 0.0) { return vec3<f32>(0.0); }
  let q = w - P.stirO.xyz;
  let r = P.stirO.w;
  return P.stirD.xyz * exp(-dot(q, q) / (r * r));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  let c = vec3<i32>(id);
  let f = vec3<f32>(id);
  // (LFM's leapfrog applies a step's forces over a half, one or two steps.)
  let dt = P.grid.y * select(1.0, P.stirR.y, P.stirR.y > 0.0);
  let vmax = P.decay.w;
  let keep = exp(-P.forces2.w * dt);
  let windT = 1.0 - exp(-WIND_DRAG * dt);
  let vIn = textureLoad(velIn, c, 0).xyz;
  var v = vIn;
  // The momentum's eddy viscosity (2), when IVOCK or LFM runs: an explicit diffusion of each face's velocity.
  if (P.stirR.z > 0.5) { v += eddyDiffusion(c); }

  if (id.y < n.y && id.z < n.z) {
    let w = worldOf(vec3<f32>(f.x, f.y + 0.5, f.z + 0.5));
    var a = 0.5 * (confine(c - X).x + confine(c).x);
    a += P.forces.w * turbMask(cellScal(c - X), cellScal(c)) * curlX(noiseSpace(w));
    var u = v.x + a * dt;
    u += (P.forces2.y - u) * windT;
    u = (u + stir(w).x * dt / P.grid.y) * keep;
    v.x = clamp(u, -vmax, vmax);
  }
  if (id.x < n.x && id.z < n.z) {
    if (id.y == 0u) {
      v.y = 0.0;
    } else {
      let w = worldOf(vec3<f32>(f.x + 0.5, f.y, f.z + 0.5));
      let s0 = cellScal(c - Y);
      let s1 = cellScal(c);
      // Buoyancy: gas hotter than the air around it is lighter, and rises at g (T - T_ambient) / T_ambient.
      var a = 0.5 * (confine(c - Y).y + confine(c).y) + P.forces.x * 0.5 * (riseOf(s0) + riseOf(s1));
      a += P.forces.w * turbMask(s0, s1) * curlY(noiseSpace(w));
      var vy = v.y + a * dt;
      vy -= vy * windT;
      // A smoke source pushes its smoke out; a fire bed only gives off fuel, and buoyancy does the rest.
      if (P.fire.x < 0.5) { vy = mix(vy, P.srcProps.z, sourceMask(w)); }
      vy = (vy + stir(w).y * dt / P.grid.y) * keep;
      v.y = clamp(vy, -vmax, vmax);
    }
  }
  if (id.x < n.x && id.y < n.y) {
    let w = worldOf(vec3<f32>(f.x + 0.5, f.y + 0.5, f.z));
    var a = 0.5 * (confine(c - Z).z + confine(c).z);
    a += P.forces.w * turbMask(cellScal(c - Z), cellScal(c)) * curlZ(noiseSpace(w));
    var wz = v.z + a * dt;
    wz += (P.forces2.z - wz) * windT;
    wz = (wz + stir(w).z * dt / P.grid.y) * keep;
    v.z = clamp(wz, -vmax, vmax);
  }
  // Faces inside the collision ball move with it; inside the reference set they stand still.
  if (anySolid()) {
    let fx = worldOf(vec3<f32>(f.x, f.y + 0.5, f.z + 0.5));
    let fy = worldOf(vec3<f32>(f.x + 0.5, f.y, f.z + 0.5));
    let fz = worldOf(vec3<f32>(f.x + 0.5, f.y + 0.5, f.z));
    if (solidAt(fx)) { v.x = select(0.0, P.ballVel.x, inBall(fx)); }
    if (solidAt(fy)) { v.y = select(0.0, P.ballVel.y, inBall(fy)); }
    if (solidAt(fz)) { v.z = select(0.0, P.ballVel.z, inBall(fz)); }
  }
  textureStore(velOut, c, vec4<f32>(v, 0.0));
  textureStore(deltaOut, c, vec4<f32>(v - vIn, 0.0));
}
`

// 4. Divergence of the face velocities at each cell (the floor face is solid), written as the pressure equation's
// right-hand side: lap(p) = div / dt with unit density, so g = -div / dt in the form (n p - sum of neighbours) / h2 = g.
export const DIVERGENCE = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(3) var scal: texture_3d<f32>;
@group(0) @binding(4) var<storage, read> fireSums: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> hrr: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  if (cellSolid(c)) {
    rhs[cellIndex(c)] = 0.0;
    return;
  }
  let a = textureLoad(vel, c, 0);
  var x0 = a.x;
  var x1 = textureLoad(vel, c + X, 0).x;
  var y0 = select(a.y, 0.0, c.y == 0);
  var y1 = textureLoad(vel, c + Y, 0).y;
  var z0 = a.z;
  var z1 = textureLoad(vel, c + Z, 0).z;
  // A face against a solid moves with it (the collision ball) or stands still (the reference set).
  if (anySolid()) {
    if (cellSolid(c - X)) { x0 = wallVel(c - X, c - X).x; }
    if (cellSolid(c + X)) { x1 = wallVel(c + X, c + X).x; }
    if (cellSolid(c - Y)) { y0 = wallVel(c - Y, c - Y).y; }
    if (cellSolid(c + Y)) { y1 = wallVel(c + Y, c + Y).y; }
    if (cellSolid(c - Z)) { z0 = wallVel(c - Z, c - Z).z; }
    if (cellSolid(c + Z)) { z1 = wallVel(c + Z, c + Z).z; }
  }
  let div = ((x1 - x0) + (y1 - y0) + (z1 - z0)) / P.grid.x;
  // Burning gas expands. At constant pressure, gas heated by dT in dt grows in volume at (dT / dt) / T, so where
  // fuel burns the projection makes the flow diverge at that rate (the low-Mach-number limit); elsewhere the flow is
  // incompressible.
  var expansion = 0.0;
  if (P.fire.x > 0.5) {
    expansion = expansionOf(gasOf(textureLoad(scal, c, 0)), hrr[cellIndex(c)], worldOf(vec3<f32>(id) + 0.5), fireSums[2].x);
  }
  rhs[cellIndex(c)] = -(div - expansion) / P.grid.y;
}
`

// 5. Pressure, the comparison solver: red-black Gauss-Seidel with over-relaxation on the 7-point Laplacian. Open sides
// and top: p = 0 outside. Solid floor: no flow through it, so the floor neighbour drops out.
export const SOR = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var<uniform> S: vec4<u32>; // x: the colour updated in this pass
@group(0) @binding(2) var<storage, read> rhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> p: array<f32>;

// One thread per cell of this pass's colour (half the grid), in runs along x so neighbouring threads read
// neighbouring memory.
@compute @workgroup_size(16, 4, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nCells();
  let c = vec3<i32>(i32(2u * id.x + ((id.y + id.z + S.x) & 1u)), i32(id.y), i32(id.z));
  if (any(c >= n)) { return; }
  let i = i32(cellIndex(c));
  let sy = n.x;
  let sz = n.x * n.y;
  if (cellSolid(c)) {
    p[i] = 0.0;
    return;
  }
  // Each neighbour inside the domain counts unless it is solid (the collision ball: no flow through its faces); past
  // an open side or the top, p = 0 counts; below the floor nothing does.
  var sum = 0.0;
  var cnt = 0.0;
  if (c.x > 0) { if (!cellSolid(c - X)) { sum += p[i - 1]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.x < n.x - 1) { if (!cellSolid(c + X)) { sum += p[i + 1]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.z > 0) { if (!cellSolid(c - Z)) { sum += p[i - sz]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.z < n.z - 1) { if (!cellSolid(c + Z)) { sum += p[i + sz]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.y > 0 && !cellSolid(c - Y)) { sum += p[i - sy]; cnt += 1.0; }
  if (c.y < n.y - 1) { if (!cellSolid(c + Y)) { sum += p[i + sy]; cnt += 1.0; } } else { cnt += 1.0; }
  let pn = (sum + rhs[i] * P.grid.x * P.grid.x) / max(cnt, 1.0);
  p[i] = mix(p[i], pn, P.decay.z);
}
`

// 5b. Pressure, the default solver: geometric multigrid V-cycles (cell-centred levels, each half the resolution of the
// last; red-black Gauss-Seidel smoothing; residuals restricted by averaging the eight children; corrections
// prolonged by trilinear interpolation). Every level keeps the fine grid's boundaries: open sides and top (p = 0 just
// outside), a solid floor. Equation on each level: (n p - sum of neighbours) / h2 = g.
const MG_COMMON = /* wgsl */ `
struct Level {
  n: vec4<u32>,  // this level's cells; the colour updated (smoothing)
  nf: vec4<u32>, // the next finer level's cells (restriction, prolongation)
  h: vec4<f32>,  // cell size squared (m2), relaxation factor
}
@group(0) @binding(0) var<uniform> L: Level;
fn nL() -> vec3<i32> { return vec3<i32>(L.n.xyz); }
fn at(c: vec3<i32>, n: vec3<i32>) -> i32 { return c.x + n.x * (c.y + n.y * c.z); }
// The collision ball (centre, radius; radius 0 = none), the grid's origin and finest cell size, and the on-set
// reference (Params' fields): a cell of any level is solid when its centre is inside one, so the coarse levels see
// them too.
struct Ball {
  s: vec4<f32>,
  o: vec4<f32>,
  refGrey: vec4<f32>,
  refChrome: vec4<f32>,
  refChartO: vec4<f32>,
  refChartU: vec4<f32>,
  refChartV: vec4<f32>,
}
@group(0) @binding(5) var<uniform> B: Ball;
${setSolid('B')}
fn solidL(c: vec3<i32>) -> bool {
  let h = sqrt(L.h.x);
  let w = B.o.xyz + (vec3<f32>(c) + 0.5) * h;
  return (B.s.w > 0.0 && distance(w, B.s.xyz) < B.s.w) || inSet(w, h);
}
`

export const MG_SMOOTH = /* wgsl */ `
${MG_COMMON}
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

// The 7-point operator's neighbour sum and count at a fluid cell: a neighbour inside the domain counts unless solid;
// past an open side or the top, p = 0 counts; below the floor nothing does.
fn gather(c: vec3<i32>, n: vec3<i32>) -> vec2<f32> {
  let i = at(c, n);
  let sy = n.x;
  let sz = n.x * n.y;
  var sum = 0.0;
  var cnt = 0.0;
  if (c.x > 0) { if (!solidL(c - vec3<i32>(1, 0, 0))) { sum += x[i - 1]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.x < n.x - 1) { if (!solidL(c + vec3<i32>(1, 0, 0))) { sum += x[i + 1]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.z > 0) { if (!solidL(c - vec3<i32>(0, 0, 1))) { sum += x[i - sz]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.z < n.z - 1) { if (!solidL(c + vec3<i32>(0, 0, 1))) { sum += x[i + sz]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.y > 0 && !solidL(c - vec3<i32>(0, 1, 0))) { sum += x[i - sy]; cnt += 1.0; }
  if (c.y < n.y - 1) { if (!solidL(c + vec3<i32>(0, 1, 0))) { sum += x[i + sy]; cnt += 1.0; } } else { cnt += 1.0; }
  return vec2<f32>(sum, cnt);
}

@compute @workgroup_size(16, 4, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nL();
  let c = vec3<i32>(i32(2u * id.x + ((id.y + id.z + L.n.w) & 1u)), i32(id.y), i32(id.z));
  if (any(c >= n)) { return; }
  let i = at(c, n);
  if (solidL(c)) {
    x[i] = 0.0;
    return;
  }
  let s = gather(c, n);
  x[i] = mix(x[i], (s.x + g[i] * L.h.x) / max(s.y, 1.0), L.h.y);
}
`

export const MG_RESIDUAL = /* wgsl */ `
${MG_COMMON}
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;

// The 7-point operator's neighbour sum and count at a fluid cell: a neighbour inside the domain counts unless solid;
// past an open side or the top, p = 0 counts; below the floor nothing does.
fn gather(c: vec3<i32>, n: vec3<i32>) -> vec2<f32> {
  let i = at(c, n);
  let sy = n.x;
  let sz = n.x * n.y;
  var sum = 0.0;
  var cnt = 0.0;
  if (c.x > 0) { if (!solidL(c - vec3<i32>(1, 0, 0))) { sum += x[i - 1]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.x < n.x - 1) { if (!solidL(c + vec3<i32>(1, 0, 0))) { sum += x[i + 1]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.z > 0) { if (!solidL(c - vec3<i32>(0, 0, 1))) { sum += x[i - sz]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.z < n.z - 1) { if (!solidL(c + vec3<i32>(0, 0, 1))) { sum += x[i + sz]; cnt += 1.0; } } else { cnt += 1.0; }
  if (c.y > 0 && !solidL(c - vec3<i32>(0, 1, 0))) { sum += x[i - sy]; cnt += 1.0; }
  if (c.y < n.y - 1) { if (!solidL(c + vec3<i32>(0, 1, 0))) { sum += x[i + sy]; cnt += 1.0; } } else { cnt += 1.0; }
  return vec2<f32>(sum, cnt);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nL();
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let i = at(c, n);
  if (solidL(c)) {
    r[i] = 0.0;
    return;
  }
  let s = gather(c, n);
  r[i] = g[i] - (s.y * x[i] - s.x) / L.h.x;
}
`

// The coarse right-hand side is the average of the eight children's residuals; the coarse correction starts at 0.
export const MG_RESTRICT = /* wgsl */ `
${MG_COMMON}
@group(0) @binding(1) var<storage, read> r: array<f32>;
@group(0) @binding(2) var<storage, read_write> gc: array<f32>;
@group(0) @binding(3) var<storage, read_write> xc: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nL();
  let nf = vec3<i32>(L.nf.xyz);
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var s = 0.0;
  var k = 0.0;
  for (var o = 0; o < 8; o++) {
    let f = 2 * c + vec3<i32>(o & 1, (o >> 1) & 1, (o >> 2) & 1);
    if (all(f < nf)) {
      s += r[at(f, nf)];
      k += 1.0;
    }
  }
  gc[at(c, n)] = s / max(k, 1.0);
  xc[at(c, n)] = 0.0;
}
`

// The fine solution gains the coarse correction, interpolated trilinearly at each fine cell centre. Past an open side
// or the top the correction mirrors to its negative (0 on the boundary face); below the floor it mirrors as is.
export const MG_PROLONG = /* wgsl */ `
${MG_COMMON}
@group(0) @binding(1) var<storage, read> xc: array<f32>;
@group(0) @binding(2) var<storage, read_write> xf: array<f32>;

fn coarse(c0: vec3<i32>) -> f32 {
  let n = nL();
  var c = c0;
  if (c.y < 0) { c.y = 0; }
  let inside = c.x >= 0 && c.x < n.x && c.z >= 0 && c.z < n.z && c.y < n.y;
  let v = xc[at(clamp(c, vec3<i32>(0), n - 1), n)];
  return select(-v, v, inside);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nf = vec3<i32>(L.nf.xyz);
  let f = vec3<i32>(id);
  if (any(f >= nf)) { return; }
  let pc = (vec3<f32>(f) + 0.5) * 0.5 - 0.5;
  let b = vec3<i32>(floor(pc));
  let t = pc - vec3<f32>(b);
  var v = 0.0;
  for (var o = 0; o < 8; o++) {
    let d = vec3<i32>(o & 1, (o >> 1) & 1, (o >> 2) & 1);
    let w = select(1.0 - t.x, t.x, d.x == 1) * select(1.0 - t.y, t.y, d.y == 1) * select(1.0 - t.z, t.z, d.z == 1);
    v += w * coarse(b + d);
  }
  xf[at(f, nf)] += v;
}
`

// 6. Projection: subtract the pressure gradient from the face velocities.
export const PROJECT = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> p: array<f32>;
@group(0) @binding(3) var velOut: texture_storage_3d<rgba16float, write>;

fn pAt(c: vec3<i32>) -> f32 {
  if (any(c < vec3<i32>(0)) || any(c >= nCells())) { return 0.0; }
  return p[cellIndex(c)];
}
// A face between a solid cell and any other moves with the solid (the collision ball) or stands still (the set).
fn ballFaces(c: vec3<i32>, v0: vec3<f32>) -> vec3<f32> {
  var v = v0;
  if (anySolid()) {
    if (cellSolid(c) || cellSolid(c - X)) { v.x = wallVel(c, c - X).x; }
    if (cellSolid(c) || cellSolid(c - Y)) { v.y = wallVel(c, c - Y).y; }
    if (cellSolid(c) || cellSolid(c - Z)) { v.z = wallVel(c, c - Z).z; }
  }
  return v;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  let c = vec3<i32>(id);
  let k = P.grid.y / P.grid.x;
  var v = textureLoad(vel, c, 0).xyz;
  if (id.y < n.y && id.z < n.z) { v.x -= k * (pAt(c) - pAt(c - X)); }
  if (id.x < n.x && id.z < n.z) {
    if (id.y == 0u) { v.y = 0.0; } else { v.y -= k * (pAt(c) - pAt(c - Y)); }
  }
  if (id.x < n.x && id.y < n.y) { v.z -= k * (pAt(c) - pAt(c - Z)); }
  v = ballFaces(c, v);
  textureStore(velOut, c, vec4<f32>(v, 0.0));
}
`

// 6b. Advection-reflection's projection (Zehnder, Narain and Thomaszewski 2018): the projected velocity, and the field
// reflected through it, u_proj + (u_proj - u) = u - 2 dt grad p, which the step's second half advects. Projecting
// halfway through the step and advecting the reflection, rather than the projected field, loses far less of the
// flow's energy to the projection.
export const PROJECT_REFLECT = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> p: array<f32>;
@group(0) @binding(3) var velOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var velRefl: texture_storage_3d<rgba16float, write>;

fn pAt(c: vec3<i32>) -> f32 {
  if (any(c < vec3<i32>(0)) || any(c >= nCells())) { return 0.0; }
  return p[cellIndex(c)];
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  let c = vec3<i32>(id);
  let k = P.grid.y / P.grid.x;
  let v = textureLoad(vel, c, 0).xyz;
  var g = vec3<f32>(0.0);
  if (id.y < n.y && id.z < n.z) { g.x = k * (pAt(c) - pAt(c - X)); }
  if (id.x < n.x && id.z < n.z && id.y > 0u) { g.y = k * (pAt(c) - pAt(c - Y)); }
  if (id.x < n.x && id.y < n.y) { g.z = k * (pAt(c) - pAt(c - Z)); }
  var proj = v - g;
  var refl = v - 2.0 * g;
  if (id.y == 0u) {
    proj.y = 0.0;
    refl.y = 0.0;
  }
  if (anySolid()) {
    if (cellSolid(c) || cellSolid(c - X)) { proj.x = wallVel(c, c - X).x; refl.x = proj.x; }
    if (cellSolid(c) || cellSolid(c - Y)) { proj.y = wallVel(c, c - Y).y; refl.y = proj.y; }
    if (cellSolid(c) || cellSolid(c - Z)) { proj.z = wallVel(c, c - Z).z; refl.z = proj.z; }
  }
  textureStore(velOut, c, vec4<f32>(proj, 0.0));
  textureStore(velRefl, c, vec4<f32>(refl, 0.0));
}
`

// 7. Scalar advection, semi-Lagrangian: DIR 1 traces back (the forward step), -1 traces forward (MacCormack's
// reverse step).
export const ADVECT_SCAL = /* wgsl */ `
${SIM_COMMON}
override DIR: f32 = 1.0;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var src: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let g = backtrace(vel, samp, vec3<f32>(id) + 0.5, DIR);
  var o = vec4<f32>(0.0);
  if (inDomain(g)) { o = scalAt(src, samp, g); }
  textureStore(dst, c, o);
}
`

// Upres detail: two layers of texture coordinates ride with the flow (each cell stores where its smoke was when the
// layer was last reset, relative to the cell), and the renderer looks up a fine noise there, so wisps finer than the
// grid move with the smoke. A layer stretches as the flow shears it, so each is reset to the identity on its own
// schedule, half a period apart, and the renderer cross-fades them: each is invisible at the moment it is reset.
export const ADVECT_DETAIL = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var srcA: texture_3d<f32>;
@group(0) @binding(4) var srcB: texture_3d<f32>;
@group(0) @binding(5) var dstA: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var dstB: texture_storage_3d<rgba16float, write>;

// How far a layer has strayed from a rigid motion since its reset (after Neyret 2003): the map from here to the
// material's origin has the Jacobian J = I + the gradient of the stored offset, and J^T J - I measures its stretch,
// 0 for a translation or a rotation and about 1.6 for a twofold stretch.
fn distortion(src: texture_3d<f32>, c: vec3<i32>) -> f32 {
  let n = nCells() - 1;
  var J: array<vec3<f32>, 3>;
  for (var k = 0; k < 3; k++) {
    var e = vec3<i32>(0);
    e[k] = 1;
    let hi = min(c + e, n);
    let lo = max(c - e, vec3<i32>(0));
    let span = f32(hi[k] - lo[k]) * P.grid.x;
    var col = (textureLoad(src, hi, 0).xyz - textureLoad(src, lo, 0).xyz) / span;
    col[k] += 1.0;
    J[k] = col;
  }
  var s = 0.0;
  for (var i = 0; i < 3; i++) {
    for (var j = 0; j < 3; j++) {
      let m = dot(J[i], J[j]) - select(0.0, 1.0, i == j);
      s += m * m;
    }
  }
  return sqrt(s);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let g = vec3<f32>(id) + 0.5;
  let gs = backtrace(vel, samp, g, 1.0);
  // The material here came from gs: its origin, relative to here, is its origin relative to gs plus (gs - g).
  // The fourth channel keeps the layer's distortion (from the last step's layer: a step behind, which is close enough).
  let shift = (gs - g) * P.grid.x;
  let resetA = P.detail.x > 0.5;
  let resetB = P.detail.y > 0.5;
  let a = select(vec4<f32>(scalAt(srcA, samp, gs).xyz + shift, distortion(srcA, c)), vec4<f32>(0.0), resetA);
  let b = select(vec4<f32>(scalAt(srcB, samp, gs).xyz + shift, distortion(srcB, c)), vec4<f32>(0.0), resetB);
  textureStore(dstA, c, a);
  textureStore(dstB, c, b);
}
`

// The detail noise: tileable gradient-noise fbm (lattice periods 8, 16 and 32 across a 64-texel tile; gradient noise
// has none of value noise's blocky, axis-aligned look), made once.
export const NOISE_GEN = /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_3d<rgba8unorm, write>;

fn hashW(c0: vec3<i32>, period: i32, seed: u32) -> f32 {
  let c = ((c0 % period) + period) % period;
  var h = (bitcast<u32>(c.x) * 0x8da6b343u) ^ (bitcast<u32>(c.y) * 0xd8163841u) ^ (bitcast<u32>(c.z) * 0xcb1ab31fu) ^ ((u32(period) + 97u * seed) * 0x9e3779b9u);
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return f32(h) * (1.0 / 4294967296.0);
}
// A random unit gradient at a lattice point (uniform on the sphere).
fn gradW(c: vec3<i32>, period: i32, seed: u32) -> vec3<f32> {
  let z = 2.0 * hashW(c, period, seed) - 1.0;
  let a = 6.2831853 * hashW(c + vec3<i32>(173, 311, 467), period, seed);
  let r = sqrt(max(1.0 - z * z, 0.0));
  return vec3<f32>(r * cos(a), r * sin(a), z);
}
fn corner(c: vec3<i32>, o: vec3<i32>, f: vec3<f32>, period: i32, seed: u32) -> f32 {
  return dot(gradW(c + o, period, seed), f - vec3<f32>(o));
}
// Gradient (Perlin) noise, about -0.7 to 0.7, wrapping every period lattice cells.
fn tileNoise(q: vec3<f32>, period: i32, seed: u32) -> f32 {
  let i = floor(q);
  let f = q - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let c = vec3<i32>(i);
  let x00 = mix(corner(c, vec3<i32>(0, 0, 0), f, period, seed), corner(c, vec3<i32>(1, 0, 0), f, period, seed), u.x);
  let x10 = mix(corner(c, vec3<i32>(0, 1, 0), f, period, seed), corner(c, vec3<i32>(1, 1, 0), f, period, seed), u.x);
  let x01 = mix(corner(c, vec3<i32>(0, 0, 1), f, period, seed), corner(c, vec3<i32>(1, 0, 1), f, period, seed), u.x);
  let x11 = mix(corner(c, vec3<i32>(0, 1, 1), f, period, seed), corner(c, vec3<i32>(1, 1, 1), f, period, seed), u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm(q: vec3<f32>, seed: u32) -> f32 {
  let v = 0.55 * tileNoise(q * 8.0, 8, seed) + 0.3 * tileNoise(q * 16.0, 16, seed) + 0.15 * tileNoise(q * 32.0, 32, seed);
  return clamp(0.5 + 1.7 * v, 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = textureDimensions(dst);
  if (any(id >= n)) { return; }
  let q = vec3<f32>(id) / vec3<f32>(n);
  // Three independent fbms, one per channel.
  textureStore(dst, vec3<i32>(id), vec4<f32>(fbm(q, 0u), fbm(q, 1u), fbm(q, 2u), 1.0));
}
`

// 7a. BiMocq2 for a smoke source's density and temperature (Qu, Zhang, Gao, Jiang and Chen 2019). Semi-Lagrangian
// advection re-samples last step's field every step and blurs it a little each time; instead the scalars are rebuilt
// each step from snapshots taken at the last two re-initialisations, through maps of where the fluid has travelled
// since: a backward map X (where the gas in each cell was at the last re-initialisation, advected with the flow) and a
// forward map Y (where the gas that was in each cell then is now, integrated with RK3). What the step's sources and
// mixing change is gathered back to the snapshot's frame through Y and accumulated, so it rides the same maps. The
// reconstruction is the paper's two-level one (its Eq. 24): the mean of the estimate through both levels' maps and
// the one through the current level's alone. It is clamped to the values one step's backtrace would interpolate
// between, as MacCormack is. This lab's simplifications: the backward map's backtrace is the solver's RK2 (the paper's
// dual-mesh characteristics need CFL below one), it re-initialises every few steps rather than on the paper's
// distortion criterion, and without its error correction. Positions are in grid units, cell centres at i + 0.5.

// The maps: a step of each. X: semi-Lagrangian, the map read where the cell's backtrace lands. Y: each reference
// point moved on by the flow (RK3).
export const BIMOCQ_MAPS = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> xIn: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> xOut: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> yIn: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> yOut: array<vec4<f32>>;

// A map, trilinear between cell centres (clamped at the grid's edge); w is 0 if any corner has no past.
fn mapAt(g: vec3<f32>) -> vec4<f32> {
  let n = nCells();
  let p = clamp(g - 0.5, vec3<f32>(0.0), vec3<f32>(n - 1));
  let b = vec3<i32>(floor(p));
  let f = p - vec3<f32>(b);
  var v = vec3<f32>(0.0);
  var ok = 1.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * select(1.0 - f.z, f.z, o.z == 1);
    let m = xIn[cellIndex(min(b + o, n - 1))];
    v += w * m.xyz;
    ok = min(ok, m.w);
  }
  return vec4<f32>(v, ok);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let i = cellIndex(c);
  let g = backtrace(vel, samp, vec3<f32>(id) + 0.5, 1.0);
  // Gas that came from outside the domain has no past here: it maps outside (and reads as clean air).
  xOut[i] = select(vec4<f32>(0.0), mapAt(g), inDomain(g));
  let k = P.grid.y / P.grid.x;
  let y = yIn[i].xyz;
  let v1 = velAt(vel, samp, y);
  let v2 = velAt(vel, samp, y + 0.5 * k * v1);
  let v3 = velAt(vel, samp, y + 0.75 * k * v2);
  yOut[i] = vec4<f32>(y + k * (2.0 * v1 + 3.0 * v2 + 4.0 * v3) / 9.0, 1.0);
}
`

// The scalars rebuilt through the maps (Eq. 24), clamped; written for the sources to work on (and kept, so the
// accumulation can tell what they changed).
export const BIMOCQ_RECON = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> xCurr: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> xPrev: array<vec4<f32>>;
@group(0) @binding(5) var phiCurr: texture_3d<f32>; // the snapshot at the last re-initialisation
@group(0) @binding(6) var phiPrev: texture_3d<f32>; // and at the one before
@group(0) @binding(7) var dCurr: texture_3d<f32>; // the changes accumulated since each
@group(0) @binding(8) var dPrev: texture_3d<f32>;
@group(0) @binding(9) var phiNow: texture_3d<f32>; // this step's field before advection (for the clamp)
@group(0) @binding(10) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(11) var keep: texture_storage_3d<rgba16float, write>;
@group(0) @binding(12) var<uniform> B: vec4<f32>; // x: whether there is a previous level

// The previous level's backward map between cell centres (clamped at the grid's edge); w is 0 if any corner has no
// past.
fn xPrevAt(g: vec3<f32>) -> vec4<f32> {
  let n = nCells();
  let p = clamp(g - 0.5, vec3<f32>(0.0), vec3<f32>(n - 1));
  let b = vec3<i32>(floor(p));
  let f = p - vec3<f32>(b);
  var v = vec4<f32>(0.0);
  var ok = 1.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * select(1.0 - f.z, f.z, o.z == 1);
    let m = xPrev[cellIndex(min(b + o, n - 1))];
    v += w * m;
    ok = min(ok, m.w);
  }
  v.w = ok;
  return v;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  let n = nCells();
  if (any(c >= n)) { return; }
  let i = cellIndex(c);
  var phi = vec4<f32>(0.0);
  let x1 = xCurr[i];
  if (x1.w > 0.5 && inDomain(x1.xyz)) {
    let one = scalAt(phiCurr, samp, x1.xyz) + scalAt(dCurr, samp, x1.xyz);
    phi = one;
    if (B.x > 0.5) {
      let x2 = xPrevAt(x1.xyz);
      if (x2.w > 0.99 && inDomain(x2.xyz)) {
        phi = 0.5 * (scalAt(phiPrev, samp, x2.xyz) + scalAt(dPrev, samp, x2.xyz)) + 0.5 * scalAt(phiCurr, samp, x1.xyz) + scalAt(dCurr, samp, x1.xyz);
      }
    }
  }
  // Clamped to the values one step's backtrace would interpolate between.
  let g = backtrace(vel, samp, vec3<f32>(id) + 0.5, 1.0);
  if (inDomain(g)) {
    let b = vec3<i32>(floor(clamp(g - 0.5, vec3<f32>(0.0), vec3<f32>(n - 1))));
    var lo = vec4<f32>(1e9);
    var hi = vec4<f32>(-1e9);
    for (var k = 0; k < 8; k++) {
      let q = textureLoad(phiNow, min(b + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1), n - 1), 0);
      lo = min(lo, q);
      hi = max(hi, q);
    }
    phi = clamp(phi, lo, hi);
  } else {
    phi = vec4<f32>(0.0);
  }
  textureStore(dst, c, phi);
  textureStore(keep, c, phi);
}
`

// What this step's sources and mixing changed (the new field less the rebuilt one), gathered back to the snapshot's
// frame at each reference point's present position (Y) and added to the accumulator (Eq. 17).
export const BIMOCQ_ACCUM = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var phiNew: texture_3d<f32>;
@group(0) @binding(3) var phiStar: texture_3d<f32>;
@group(0) @binding(4) var<storage, read> y: array<vec4<f32>>;
@group(0) @binding(5) var accIn: texture_3d<f32>;
@group(0) @binding(6) var accOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  var a = textureLoad(accIn, c, 0);
  let p = y[cellIndex(c)].xyz;
  if (inDomain(p)) { a += scalAt(phiNew, samp, p) - scalAt(phiStar, samp, p); }
  textureStore(accOut, c, a);
}
`

// 7b. Keeping a fire's mass. Semi-Lagrangian advection carries each cell's value along the flow but not its mass.
// Where the flow diverges sharply, around gas that has just burned and is swelling to several times its volume in a
// step, the backtraces around it sample it less than the room it should fill, and the hot gas, with its fuel and
// products, loses mass (a quarter to a third of the fire's, measured). So first every cell's backtrace scatters its
// trilinear weights onto the cells it samples, and each cell learns over how much volume its gas was spread (in
// cells); then each cell gets the ratio of the room its gas should fill (1 + e dt, e its expansion) to that, and the
// advection scales what it takes from the cell by it. This follows the conservative semi-Lagrangian method of
// Lentine, Gretarsson and Fedkiw (2011), which measures the same per-cell sums but only scales over-drawn cells down
// (to 1, its gas incompressible in conserved form) and traces the undrawn rest of a cell forward; here both ways
// are a scale, toward a room that includes the expansion. (Correcting only the expanding cells is not enough: the
// cells around them, sampled more to fill the gap, then gain what the hot gas lost.) The fields are all per unit volume: masses, and the density deficit, which is the
// gas's heat (rho (T - T0) = rho0 T0 times the deficit), so the fire keeps its fuel, its products and its heat.
export const CONSERVE_SPREAD = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> spread: array<atomic<u32>>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nCells();
  if (any(vec3<i32>(id) >= n)) { return; }
  let g = backtrace(vel, samp, vec3<f32>(id) + 0.5, 1.0);
  if (!inDomain(g)) { return; }
  let p = clamp(g - 0.5, vec3<f32>(0.0), nF() - 1.0);
  let b = vec3<i32>(floor(p));
  let f = p - vec3<f32>(b);
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * select(1.0 - f.z, f.z, o.z == 1);
    atomicAdd(&spread[cellIndex(min(b + o, n - 1))], u32(w * SPREAD_ONE + 0.5));
  }
}
`

export const CONSERVE_RATIO = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var scal: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> fireSums: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> hrr: array<f32>;
@group(0) @binding(4) var<storage, read> spread: array<u32>;
@group(0) @binding(5) var<storage, read_write> ratio: array<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var vel: texture_3d<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let i = cellIndex(c);
  var r = 1.0;
  if (!cellSolid(c)) {
    // The room this cell's gas should fill: its own cell, grown by the expansion the pressure solve made room for.
    let grow = expansionOf(gasOf(textureLoad(scal, c, 0)), hrr[i], worldOf(vec3<f32>(id) + 0.5), fireSums[2].x) * P.grid.y;
    r = clamp((1.0 + grow) / max(f32(spread[i]) / SPREAD_ONE, 1e-3), 0.25, 4.0);
    // Gas on its way out of the domain is sampled less because it leaves; it is not made up.
    let n = nF();
    let to = backtrace(vel, samp, vec3<f32>(id) + 0.5, -1.0);
    if (to.x < 1.0 || to.x > n.x - 1.0 || to.z < 1.0 || to.z > n.z - 1.0 || to.y > n.y - 1.0) { r = min(r, 1.0); }
    // (An expanding cell's ratio is stored negative, which tells MacCormack's correction to stand aside there.)
    if (grow > 1e-3) { r = -r; }
  }
  ratio[i] = r;
}
`

// 7c. The wooden ball's skin, heated by the fire. Each texel is a patch of wood, a surface layer 1.5 mm thick over a
// layer 6 mm thick over the cool core, with its heat balance worked out every step:
//  - radiation from the gas outside it: four layers along the normal, out to 25 cm, each a layer of soot at its own
//    temperature (soot's Planck-mean absorption coefficient, 3.72 C0 T / C2 per unit volume fraction with C0 = 7,
//    about 1810 f_v T per metre), seen through the layers nearer, then the air beyond;
//  - radiation from the fire further off: each band of its height a point source of its share of the radiated heat
//    (the radiant fraction of the heat release), q cos / (4 pi r^2);
//  - convection from the gas just outside (10 W/m2K), and the surface's own radiation (emissivity 0.9);
//  - conduction into the wood (0.14 W/mK), the heat the pyrolysis takes (0.3 MJ per kg of volatiles; measured values
//    run 0.1 to 0.3), and, once the char glows, the char's own oxidation (up to 12 kW/m2, while there is oxygen at the
//    surface).
// The char front advances by charRate(), so the volatiles come off at pyrolysisOf(), which COMBINE adds to the gas as
// fuel: the ball's flames are the solver's own combustion. Ash builds on glowing char over a minute or two.
export const BALL_HEAT = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var scal: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> fireSums: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> skinOut: array<vec4<f32>>;
// The fuel flowing up along the surface past each texel (kg per metre of width per second), for the drawn flames.
@group(0) @binding(5) var<storage, read_write> sheetOut: array<f32>;
@group(0) @binding(6) var noiseTex: texture_3d<f32>;
@group(0) @binding(7) var noiseSamp: sampler;
const SIGMA = 5.670374e-8;
const EPS_WOOD = 0.9;
const H_CONV = 10.0;
const K_WOOD = 0.14;
const C1 = WOOD_RHO * 1700.0 * 0.0015; // J / (m2 K), the surface layer
const C2 = WOOD_RHO * 1700.0 * 0.006; // the layer below
const G12 = K_WOOD / 0.00375; // W / (m2 K), between their middles
const G2C = K_WOOD / 0.018; // from the lower layer to the core
const L_PYRO = 3.0e5; // J per kg of volatiles
const Q_OX = 12000.0; // W / m2, glowing char
fn gasAt(w: vec3<f32>) -> Gas { return gasOf(scalAt(scal, samp, (w - P.origin.xyz) / P.grid.x)); }
fn t4(t: f32) -> f32 { let t2 = t * t; return t2 * t2; }
// The wood is not uniform. Its late wood, in the growth rings the renderer draws (the same rings), is denser than its
// early wood, and the whole ball varies a little from patch to patch: where it is lighter its surface heats faster and
// catches sooner, so the burning front climbs raggedly, along the grain, rather than in a level line (Scots pine's late
// wood is some 1.41 times its rings' mean density and its early wood 0.77; Fernandes et al. 2017). Scaled so the ball as
// a whole heats as fast as wood of its mean density: light early wood heats faster by more than dense late wood heats
// slower, and unscaled the pattern heated the surface some 12% faster on average, enough that a ball held a minute in
// the fire burned on to nothing instead of starving under its char (1.118 is the mean of 1 / density over the rings).
fn woodDensity(n: vec3<f32>) -> f32 {
  let axis = normalize(vec3<f32>(0.35, 1.0, 0.2));
  let wave = textureSampleLevel(noiseTex, noiseSamp, n * 0.3 + 0.5, 0.0).x - 0.5;
  let ring = fract(length(n - dot(n, axis) * axis) * 16.0 + 0.6 * wave);
  let late = smoothstep(0.5, 0.8, ring) * (1.0 - smoothstep(0.85, 1.0, ring));
  return 1.118 * mix(0.77, 1.41, late) * (0.85 + 0.3 * fbm(n * 3.0 + 11.0));
}
fn burning(m: f32) -> f32 { return m * smoothstep(0.0015, 0.0025, m); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= SKIN_W || id.y >= SKIN_H) { return; }
  let i = id.y * SKIN_W + id.x;
  var s = skin[i];
  let n = skinDir((f32(id.x) + 0.5) / f32(SKIN_W), (f32(id.y) + 0.5) / f32(SKIN_H));
  let dx = P.grid.x;
  let surface = P.ball.xyz + n * P.ball.w;
  // The gas along the normal.
  var through = 1.0;
  var qGas = 0.0;
  let near = gasAt(surface + n * dx);
  for (var k = 0; k < 4; k++) {
    var g = near;
    if (k > 0) { g = gasAt(surface + n * (1.0 + 2.0 * f32(k)) * dx); }
    let t = T_AMB + g.rise;
    let fv = g.soot / SOOT_G * (RHO_T / t) / 1800.0; // soot's volume fraction (1800 kg/m3)
    let a = 1.0 - exp(-1810.0 * fv * t * 2.0 * dx * 1.5); // (1.5: the mean path across the layer over a hemisphere)
    qGas += through * a * SIGMA * t4(t);
    through *= 1.0 - a;
  }
  // The flames' emissivity is held to 0.25, a wood flame's some 40 cm deep (such flames absorb about 0.7 per metre):
  // the lab's soot yield is a look as much as a measurement, and its flames run hotter than measured (see
  // Validation), which T^4 would magnify.
  let flameEps = 1.0 - through;
  if (flameEps > 0.25) { qGas *= 0.25 / flameEps; through = 0.75; }
  qGas += through * SIGMA * t4(T_AMB);
  // The fire further off.
  let vol = dx * dx * dx;
  var lumAll = 0.0;
  for (var b = 0u; b < ${FIRE_BANDS}u; b++) { lumAll += fireSums[3u + 2u * b].w * vol; }
  var qFire = 0.0;
  if (lumAll > 0.0) {
    let radiated = P.fire.w * fireSums[2].y;
    for (var b = 0u; b < ${FIRE_BANDS}u; b++) {
      let lum = fireSums[3u + 2u * b].w * vol;
      if (lum <= 0.0) { continue; }
      let d = fireSums[4u + 2u * b].xyz - surface;
      let r = max(length(d), 1e-3);
      // (Bands within about half a metre are the flames the gas samples above already see.)
      qFire += radiated * lum / lumAll * max(dot(n, d) / r, 0.0) / (12.566371 * r * r) * smoothstep(0.35, 0.7, r);
    }
  }
  // The ball's own flame along its surface: a layer too thin for the grid, it heats the wood above a burning patch,
  // which is how fire climbs a surface. A wall flame's heat flux grows with the burning height beneath it, to about
  // 25 kW/m2 once some 9 cm of the surface below is alight (measured averages run 15 to 20, the usual assumption 25 to
  // 30): a small new patch cannot keep thick wood burning without
  // outside heat, a tall one can. Up the surface is up the map (v toward the top), within a texel either way around.
  var below = 0.0;
  for (var dj = 1u; dj <= 8u; dj++) {
    let j = id.y + dj;
    if (j >= SKIN_H) { break; }
    var row = 0.0;
    for (var di = -1; di <= 1; di++) {
      let m = pyrolysisOf(skin[j * SKIN_W + u32((i32(id.x) + di + i32(SKIN_W)) % i32(SKIN_W))]);
      row = max(row, smoothstep(0.0015, 0.0025, m));
    }
    below += row / 8.0;
  }
  // The flame flickers, a pattern rising over the surface at a couple of hertz, so the front advances in fingers.
  let flicker = 0.5 + fbm(n * 5.0 + vec3<f32>(0.0, -2.0 * P.grid.z, 0.0));
  let qWall = 25000.0 * below * flicker;
  let o2 = clamp((Y_O2_AIR * (1.0 - near.z) - S_O2 * (near.z - near.fuel)) / Y_O2_AIR, 0.0, 1.0);
  let charred = clamp(s.z / 0.5, 0.0, 1.0);
  let rate = charRate(s);
  let qOx = Q_OX * smoothstep(600.0, 750.0, s.x) * charred * o2 * (1.0 - 0.7 * s.w);
  let q1 = EPS_WOOD * (qGas + qFire + qWall - SIGMA * t4(s.x)) + H_CONV * (T_AMB + near.rise - s.x) + qOx
    - WOOD_RHO * (1.0 - CHAR_YIELD) * rate * L_PYRO - G12 * (s.x - s.y);
  let q2 = G12 * (s.x - s.y) - G2C * (s.y - T_AMB);
  let dt = P.grid.y;
  // (The grain sets how fast the wood heats until it chars; char is char.)
  let dens = mix(woodDensity(n), 1.0, charred);
  var next = s;
  next.x = clamp(s.x + dt * q1 / (C1 * dens), T_AMB - 30.0, 1500.0);
  next.y = clamp(s.y + dt * q2 / C2, T_AMB - 30.0, 1500.0);
  next.z = s.z + dt * 1000.0 * rate;
  next.w = min(s.w + dt * smoothstep(700.0, 850.0, s.x) * charred / 90.0, 1.0);
  skinOut[i] = next;
  // The volatiles burn as they flow up the surface, so the flame at a texel carries all the fuel that came off below it
  // (per metre of width: the circles of latitude narrow toward the poles), the column and its two neighbours.
  var sheet = 0.0;
  for (var j = id.y; j < SKIN_H; j++) {
    var f = 0.0;
    for (var di = -1; di <= 1; di++) { f += burning(pyrolysisOf(skin[j * SKIN_W + u32((i32(id.x) + di + i32(SKIN_W)) % i32(SKIN_W))])); }
    sheet += f / 3.0 * sin(3.1415927 * (f32(j) + 0.5) / f32(SKIN_H));
  }
  let ds = 3.1415927 * P.ball.w / f32(SKIN_H);
  sheetOut[i] = sheet * ds / max(sin(3.1415927 * (f32(id.y) + 0.5) / f32(SKIN_H)), 0.15);
}
`

// 8. Modified MacCormack (Selle et al. 2008): the forward result plus half the round-trip error, clamped to the
// values the forward step interpolated between; then dissipation, cooling, the source and, in a fire, combustion.
export const COMBINE = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var s0: texture_3d<f32>;
@group(0) @binding(4) var s1: texture_3d<f32>;
@group(0) @binding(5) var s2: texture_3d<f32>;
@group(0) @binding(6) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var<storage, read> fireSums: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> hrr: array<f32>; // heat release rate, W/m3
@group(0) @binding(9) var<storage, read> ratio: array<f32>; // a fire's mass-keeping ratios (7b)
@group(0) @binding(10) var flameOut: texture_storage_3d<rgba16float, write>; // the heat release rate (MW/m3), to render

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  let n = nCells();
  if (any(c >= n)) { return; }
  var phi = textureLoad(s1, c, 0);
  let mac = P.stirR.w > 0.5;
  let fire = P.fire.x > 0.5;
  if (mac || fire) {
    let g = backtrace(vel, samp, vec3<f32>(id) + 0.5, 1.0);
    if (inDomain(g)) {
      let p = clamp(g - 0.5, vec3<f32>(0.0), nF() - 1.0);
      let b = vec3<i32>(floor(p));
      let f = p - vec3<f32>(b);
      var lo = vec4<f32>(1e9);
      var hi = vec4<f32>(-1e9);
      var kept = vec4<f32>(0.0);
      var expanding = false;
      for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let qc = min(b + o, n - 1);
        let q = textureLoad(s0, qc, 0);
        lo = min(lo, q);
        hi = max(hi, q);
        if (fire) {
          let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1) * select(1.0 - f.z, f.z, o.z == 1);
          let r = ratio[cellIndex(qc)];
          kept += w * (abs(r) - 1.0) * q;
          expanding = expanding || r < 0.0;
        }
      }
      // MacCormack's round trip is not mass-neutral where the gas expands (its error estimate takes in the change of
      // volume), so there the semi-Lagrangian result stands.
      if (mac && !expanding) { phi = clamp(phi + 0.5 * (textureLoad(s0, c, 0) - textureLoad(s2, c, 0)), lo, hi); }
      // What the expanding cells' ratios add back (7b).
      phi += kept;
      // Where the gas is not expanding it only moves, so nothing here can exceed the most the gas it came from held.
      // MacCormack's correction already restores a peak that the trilinear sample smeared, and the mass keeping, added
      // on top where a cell was under-sampled, then counted that again: beside the collision ball, where the flow round
      // the solid leaves cells a little under-sampled, a speck of soot in still air grew by some 6% a step until it
      // filled the air with soot at a fifth of its mass.
      if (!expanding) { phi = min(phi, hi); }
    }
  }
  let dt = P.grid.y;
  // Near the open top the smoke disperses faster, so it thins out before the boundary instead of being cut there.
  let top = smoothstep(0.7, 1.0, (f32(id.y) + 0.5) / f32(n.y));
  phi.x = max(phi.x, 0.0) * exp(-(P.decay.y + 2.5 * top) * dt);
  phi.y = max(phi.y, 0.0) * exp(-P.decay.x * dt); // a temperature rise, or a fire's density deficit
  let m = sourceMask(worldOf(vec3<f32>(id) + 0.5));
  if (P.fire.x > 0.5) {
    // Fuel vapour from the bed mixes in by mass (and its volume pushes out through the pressure solve).
    var g = gasOf(phi);
    g.z = min(g.z, 1.0);
    g.fuel = min(g.fuel, g.z);
    let a = fuelAdded(g, worldOf(vec3<f32>(id) + 0.5), fireSums[2].x, dt);
    g.z = (g.z + a) / (1.0 + a);
    g.fuel = (g.fuel + a) / (1.0 + a);
    g.soot = g.soot / (1.0 + a);
    // Burning: the fuel burned heats the gas (less the share radiated away) and leaves soot at the fuel's yield.
    // (Gas at the cap burns no further until it cools: the expansion follows the heat released.)
    let burned = min(fuelBurned(g, dt, mixTime(strainRate(vel, c))), max(RISE_MAX - g.rise, 0.0) / heatRise(1.0));
    hrr[cellIndex(c)] = burned * HEAT * RHO_T / (T_AMB + g.rise) / dt;
    g.fuel -= burned;
    g.rise = min(g.rise + heatRise(burned), RISE_MAX);
    g.soot += SOOT_G * P.fire.z * burned;
    phi = storeGas(g);
  } else {
    phi.x = max(phi.x, P.srcProps.x * m);
    phi.y = max(phi.y, P.srcProps.y * m);
  }
  if (cellSolid(c)) {
    phi = vec4<f32>(0.0);
    if (P.fire.x > 0.5) { hrr[cellIndex(c)] = 0.0; }
  }
  textureStore(dst, c, phi);
  var heatMW = 0.0;
  if (P.fire.x > 0.5) { heatMW = hrr[cellIndex(c)] * 1e-6; }
  textureStore(flameOut, c, vec4<f32>(heatMW, 0.0, 0.0, 0.0));
}
`

// 8b. Subgrid mixing, as large-eddy simulations model it: motion finer than the grid mixes the scalars at the eddy
// diffusivity D = (C_s dx)^2 |S| / Sc_t, Smagorinsky's eddy viscosity (|S| the resolved strain rate) over a
// turbulent Schmidt number. The velocity, advected semi-Lagrangian, already loses more to its own scheme than an
// eddy viscosity would take; the scalars, advected with the sharper MacCormack, need the model, or a fire's fuel rises
// in an unmixed column far past the flame's height. One explicit step, well inside its stability limit.
export const DIFFUSE = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
const SC_T = ${wf(SCHMIDT_T)};
fn at(c: vec3<i32>) -> vec4<f32> { return textureLoad(src, clamp(c, vec3<i32>(0), nCells() - 1), 0); }

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let dx = P.grid.x;
  let d = (C_S * dx) * (C_S * dx) * strainRate(vel, c) / SC_T;
  let k = min(d * P.grid.y / (dx * dx), 0.15);
  let phi = at(c);
  let lap = at(c + X) + at(c - X) + at(c + Y) + at(c - Y) + at(c + Z) + at(c - Z) - 6.0 * phi;
  textureStore(dst, c, phi + k * lap);
}
`

// Colours are linear ACEScg throughout (the labs' working space, viewTransforms.ts): the blackbody table, the sun and
// the sky, the soot's extinction by channel, the albedos, the white balance.
export const RENDER_COMMON = /* wgsl */ `
${AP1_WGSL}
struct RParams {
  camPos: vec4<f32>,   // position (m), tan of half the vertical field of view
  camFwd: vec4<f32>,   // forward, aspect (width / height)
  camRight: vec4<f32>, // right, frame number
  camUp: vec4<f32>,    // up, exposure (1 / the luminance in cd/m2 that the camera maps to white)
  sunDir: vec4<f32>,   // toward the sun, phase anisotropy g
  sunCol: vec4<f32>,   // the sun's illuminance (lux) by colour, a smoke source's extinction per unit density (1/m)
  skyCol: vec4<f32>,   // the sky's illuminance on open ground (lux) by colour, single-scattering albedo
  floorCol: vec4<f32>, // floor albedo, radius (m) over which the floor fades to black
  size: vec4<f32>,     // render target width, height, simulation time (s), subgrid flame temperature (1 = the model)
  detail: vec4<f32>,   // weight of detail layer A, of layer B, amount, 1 / (8 x detail scale in m)
  fire: vec4<f32>,     // source is a fire (0 or 1), extinction in red and in blue relative to 550 nm, flame eddies' rise (m/s)
  wb0: vec4<f32>,      // the camera's white balance, rows of a 3x3 on linear ACEScg; wind (m/s) in the first row's w
  wb1: vec4<f32>,      // (w: sample the scalars tricubically, 0 or 1)
  wb2: vec4<f32>,      // (w: render the smoke from the upres grid, 0 or 1)
  scene: vec4<f32>,    // the stage: a studio's black (0), the sky and a floor to the horizon (1), the cyc (2); extinction in
                       // green relative to 550 nm; the view transform (0 ACES 2.0, 1 AgX, 2 PBR Neutral, 3 Standard);
                       // whether the ACES 2.0 LUT has loaded (0 or 1)
  env: vec4<f32>,      // an HDR environment (0 or 1); cos and sin of its turn into the map's frame; its scale to lux
  sh: array<vec4<f32>, 9>, // its light but the key, as spherical harmonics (rgb, cd/m2 x sr), the floor below
  rimDir: vec4<f32>,   // the softbox rig's rim: toward it, on (0 or 1)
  rimCol: vec4<f32>,   // its illuminance (lux) by colour
  camera: vec4<f32>,   // the shutter's open time (s, 0 for none); the glare's share of the light
}
@group(0) @binding(1) var<uniform> R: RParams;

const FIRE_BANDS = ${FIRE_BANDS}u;
const BAND_R2 = 0.02; // m2: softens each band of the fire's light over the flame's width
// How much of a round light gets past a round occluder of radius a, missing its centre by miss, where the light's radius
// as seen across the occluder is pen: the occluder hides at most its share of the light's disc, (a / pen)^2, and that
// fades out as the two discs part (an approximation of their overlap). For a light much smaller than the occluder, a
// sharp edge softened over pen.
fn discVis(a: f32, pen: f32, miss: f32) -> f32 {
  let cover = min(a * a / max(pen * pen, 1e-10), 1.0);
  return 1.0 - cover * (1.0 - smoothstep(abs(pen - a), pen + a, miss));
}
// The fire's escape table (FIRE_SUM): for each slice of its height, the share of its light that gets out of its own
// soot, over a bed's radius, toward each of ESCAPE_DIRS directions: rings at ESCAPE_RINGS degrees of elevation, each
// toward ESCAPE_AZ azimuths from +x round toward +z (ESCAPE_AZ ring + azimuth), at the quarters of the diamond angle in
// each quadrant (0, 18.4, 45 and 71.6 degrees from its first axis, where bandEscape reads them exactly), then straight
// up and straight down, and its mean over the sphere. ESCAPE_STRIDE entries of the sums per slice, after the slices'
// own: the rings, four entries each (a quadrant an entry), then (up, down, mean); the scattered light's table follows
// the slices' escape tables (ESCAPE_BASE2), laid out the same, and after the slices' spread, the escape by height
// (ESCAPE_BASE3: each cell's light that gets out times its height over the floor, over the light times the height, of
// the cells marched), for the floor (bandOutFloor). (Four azimuths missed a flame sheet lying across the diagonal: the
// bonfire's light fell a fifth short on one side of its cyc and a quarter over on the other. Eight, 45 degrees apart,
// read its middle slices' light toward the far left of its wall, between -x and the diagonal, as much as twice what
// gets out: the escape falls threefold within those 45 degrees, and the table read straight across them, so the wall
// there took a quarter more of the fire's light than every glowing cell's own, marched to it through the soot; sixteen
// bring the table within a twelfth. With rings at -30, 0 and 30 only, read linearly in the sine between them and the
// poles, the escape from a slice's soot came out short wherever it bends between rings: up toward the wall above a
// bonfire, where straight up looks along the whole flame column and pulls the reading down, three fifths of the cells'
// own at 50 to 60 degrees, and just below the level, where it climbs steeply, a fifth short at -20; the rings at -15
// and 60 bring both within a tenth. Above 60, read toward straight up, a ring at 75 and the caps by the angle bring
// every slice within two hundredths, where the upper slices were a quarter short; and between 30 and 60, where the
// lowest slice's escape peaks near 40, a ring at 45 brings it within a hundredth, where it was an eighth short; and
// just above the level toward a bonfire's right, where the share its bed's sheet lets out climbs by half again within
// 25 degrees (0.14 to 0.22), the rings at 0 and 30 read a line across it, a tenth short at 15 to 20, and a ring at 15
// brings it within 3% at the table's azimuths; above that, toward the cyc's wall, read linearly in the sine between 15,
// 30 and 45 where it bends, the lowest slice's escape came out 2% short of its own march, and the fire's light on the
// wall's middle a few percent short of every glowing cell's own, and rings at 22.5 and 37.5 bring it within 0.7%.) A
// step marches a twenty-fourth of the directions in turn (ESCAPE_PHASES), seven, and keeps the rest from the steps
// before: all 58 of the eight azimuths every step added some 1.6 ms (all of them every step, the rest of the pass done
// once, would add 1 to 3.5 ms at the Medium resolution). Running, the table is a blend of the last twenty-four steps;
// the share that gets out each way moves with the flames, a seventh from one step to the next about a bonfire, so each
// way reads about a quarter off the step's own whether a step marches seven or thirteen (over nine steps). Paused, the
// engine marches all of it for the frame (resumFireNow). (Cells with under ESCAPE_SKIP of their slice's light are left
// out. Dim and outer, they let more of it out than the rest: left out at a ten-thousandth, the table read a bonfire's
// lowest slice 5 to 10% low in every direction.)
const ESCAPE_DIRS = ${ESCAPE_NDIRS}u;
const ESCAPE_PHASES = ${ESCAPE_PHASES}u;
const ESCAPE_NRINGS = ${ESCAPE_RINGS.length}u;
const ESCAPE_AZ = ${ESCAPE_AZ}u;
const ESCAPE_POLES = ${ESCAPE_POLES}u; // (the entry with straight up, straight down and the mean)
const ESCAPE_RINGS = array<f32, ${ESCAPE_RINGS.length}>(${ESCAPE_RINGS.map(wf).join(', ')});
const ESCAPE_SINES = array<f32, ${ESCAPE_RINGS.length}>(${ESCAPE_SINES.map(wf).join(', ')});
const ESCAPE_BASE = 3u + 2u * FIRE_BANDS;
const ESCAPE_STRIDE = ${ESCAPE_ENTRIES}u;
const ESCAPE_BASE2 = 3u + ${2 + ESCAPE_ENTRIES}u * FIRE_BANDS; // (the scattered light's table)
const SPREAD_BASE = 3u + ${2 + 2 * ESCAPE_ENTRIES}u * FIRE_BANDS; // (each slice's spread, FIRE_SUM: its variance in x, y and z, and its xz covariance)
const ESCAPE_BASE3 = 3u + ${3 + 2 * ESCAPE_ENTRIES}u * FIRE_BANDS; // (the escape table by height, laid out as the others)
fn escapeDir(k: u32) -> vec3<f32> {
  if (k == ESCAPE_DIRS - 2u) { return vec3<f32>(0.0, 1.0, 0.0); }
  if (k == ESCAPE_DIRS - 1u) { return vec3<f32>(0.0, -1.0, 0.0); }
  let el = radians(ESCAPE_RINGS[k / ESCAPE_AZ]);
  let dia = f32(k % ESCAPE_AZ) * 4.0 / f32(ESCAPE_AZ);
  let f = fract(dia);
  var h = vec2<f32>(1.0 - f, f);
  for (var q = 0u; q < u32(dia); q++) { h = vec2<f32>(-h.y, h.x); }
  h = normalize(h) * cos(el);
  return vec3<f32>(h.x, sin(el), h.y);
}
// Where fuel is burning, soot radiates at the flame's temperature, not the cell's. The burning happens in sheets far
// thinner than a cell, at about 1400 K, and the soot forms in them, while the cell's mean temperature spreads the same
// heat over all of it (a thin flame on a burning surface would never glow at the mean). How much of a cell's soot sits
// in a sheet follows its heat release rate (MW/m3).
const FLAME_SHEET_K = 1400.0;
fn radiatingKelvin(meanK: f32, heatMW: f32) -> f32 {
  return max(meanK, mix(meanK, FLAME_SHEET_K, smoothstep(0.2, 2.0, heatMW)));
}
// The flame's own light (chemiluminescence, physics.ts), in proportion to the heat released: cd/m2 per metre of
// flame at 1 MW/m3, the light per watt over the sphere. Its colour, CH's blue with C2's green, at unit luminance.
const CHEMI_CD = ${wf((CHEMI_LM_PER_W * 1e6) / (4 * Math.PI))};
const CHEMI_COLOUR = vec3<f32>(${chemiluminescenceColour().map(wf).join(', ')});
// The share of a burning cell's soot formed by now. Soot forms SOOT_DELAY after its fuel burns, so the soot of the
// last SOOT_DELAY of burning is not there yet: of the heat released so far (the fuel burned, the mixture fraction
// less the fuel left), the part released that recently. Where the fuel has only begun to burn, at the flame's base,
// that is most of it, so the gas there is clear and the flame's blue shows through. (The grid's soot stays as the
// yield makes it, for the smoke; this is how much of it is soot yet, for the flame's glow and its light.)
const SOOT_DELAY = ${wf(SOOT_DELAY)}; // s
fn sootFormed(g: Gas, heatMW: f32) -> f32 {
  let released = max(g.z - g.fuel, 0.0) * HEAT * RHO_T / (T_AMB + g.rise); // J/m3
  return clamp(1.0 - SOOT_DELAY * heatMW * 1e6 / max(released, 1e-3), 0.0, 1.0);
}
const INV_PI = 0.31830989;
const INV_2PI = 0.15915494;
const INV_4PI = 0.07957747;
const SOOT_KE = ${wf(SOOT_EXTINCTION)}; // m2 per kg of soot, at 550 nm

// Extinction at the reference wavelength (550 nm), 1/m: a fire's soot by its mass per volume (its partial density), a
// smoke source's smoke by its density. Each channel scales by chroma().
fn sigmaOf(s: vec4<f32>) -> f32 {
  if (R.fire.x > 0.5) { return SOOT_KE * (RHO_T / T_AMB) * max(s.x, 0.0) / SOOT_G; }
  return R.sunCol.w * max(s.x, 0.0);
}
fn chroma() -> vec3<f32> { return vec3<f32>(R.fire.y, R.scene.y, R.fire.z); }

// Entry and exit distances of a ray through an axis-aligned box.
fn boxHit(o: vec3<f32>, d: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec2<f32> {
  let inv = 1.0 / select(d, vec3<f32>(1e-6), abs(d) < vec3<f32>(1e-6));
  let t0 = (lo - o) * inv;
  let t1 = (hi - o) * inv;
  let a = min(t0, t1);
  let b = max(t0, t1);
  return vec2<f32>(max(max(a.x, a.y), a.z), min(min(b.x, b.y), b.z));
}
fn domainHi() -> vec3<f32> { return P.origin.xyz + vec3<f32>(P.dims.xyz) * P.grid.x; }
// 1 inside, falling to 0 toward the open sides and top, so smoke dissolves as it nears them instead of meeting a
// wall: the outer 5% of the width (at the frame's edges on wide screens), the outer fifth of the depth, the top fifth.
fn edgeFade(uvw: vec3<f32>) -> f32 {
  let ex = min(uvw.x, 1.0 - uvw.x);
  let ez = min(uvw.z, 1.0 - uvw.z);
  return smoothstep(0.0, 0.05, ex) * smoothstep(0.0, 0.2, ez) * (1.0 - smoothstep(0.8, 1.0, uvw.y));
}
fn uvwOf(w: vec3<f32>) -> vec3<f32> { return (w - P.origin.xyz) / (vec3<f32>(P.dims.xyz) * P.grid.x); }
// Where a ray (unit direction) first meets the collision ball, or 1e9.
fn ballHit(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
  if (P.ball.w <= 0.0) { return 1e9; }
  let oc = ro - P.ball.xyz;
  let b = dot(oc, rd);
  let h = b * b - (dot(oc, oc) - P.ball.w * P.ball.w);
  if (h < 0.0) { return 1e9; }
  let t = -b - sqrt(h);
  return select(1e9, t, t > 0.0);
}
const BALL_ALBEDO = 0.45;
// How much sunlight the collision ball lets past to a point: 0 in its shadow, with a soft edge a few millimetres wide
// plus the sun's own width at that distance (it spans half a degree).
fn ballSun(w: vec3<f32>, sun: vec3<f32>) -> f32 {
  if (P.ball.w <= 0.0) { return 1.0; }
  let tb = P.ball.xyz - w;
  let along = dot(tb, sun);
  if (along <= 0.0) { return 1.0; }
  let soft = 0.004 + 0.0046 * along;
  return smoothstep(P.ball.w - soft, P.ball.w + soft, length(tb - sun * along));
}
`

// 9. Light: optical depth (at 550 nm) from each point of a half-resolution lattice toward the sun, and straight up
// (the sky), and the fire's light there, shadowed by the soot.
export const LIGHT = /* wgsl */ `
${SIM_COMMON}
${RENDER_COMMON}
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var scal: texture_3d<f32>;
@group(0) @binding(4) var lightOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var<storage, read> fireSums: array<vec4<f32>>;
// x: optical depth toward the rig's rim; y: the sky's (the dome's mean, as a depth); in the bottom layer, z and w: the
// sun's and the fire's light on the floor below (floorLit).
@group(0) @binding(6) var light2Out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var upresTex: texture_3d<f32>; // the upres grid's density over each coarse cell (UPRES_COARSE)

// The smoke as the image pass draws it: with the upres grid on, its density. (Shadowed by the coarse grid's instead,
// the densest smoke caught up to a third more of the sun than the path-traced reference gives it.)
fn density(w: vec3<f32>) -> f32 {
  if (R.wb2.w > 0.5) { return R.sunCol.w * max(textureSampleLevel(upresTex, samp, uvwOf(w), 0.0).x, 0.0); }
  return sigmaOf(textureSampleLevel(scal, samp, uvwOf(w), 0.0));
}

// The fire's light here, shadowed by the soot: each slice of the fire's height as a point light at its brightness-
// weighted centre, carrying its emitted intensity (I / r^2), through the optical depth on the way (12 samples). In
// kilolux of luminance (the colour is the fire's overall colour), so it fits half precision near the flames.
${bandOutWgsl('fireSums')}
fn fireIrradiance(w: vec3<f32>, n: vec3<f32>) -> f32 {
  if (P.fire.x < 0.5) { return 0.0; }
  let vol = P.grid.x * P.grid.x * P.grid.x;
  var e = 0.0;
  for (var b = 0u; b < FIRE_BANDS; b++) {
    let lum = fireSums[3u + 2u * b].w * vol;
    if (lum <= 0.0) { continue; }
    let d = fireSums[4u + 2u * b].xyz - w;
    let r2 = dot(d, d);
    // The collision ball's shadow, soft: the band is a slice of flame as wide as the bed, whose penumbra at the ball
    // is that width scaled by how far along the way the ball sits (and this lattice is too coarse for a hard edge); a ball
// smaller than the penumbra hides only its share of the slice (discVis).
    let dist = max(sqrt(r2), 1e-6);
    var vis = 1.0;
    if (P.ball.w > 0.0) {
      let along = dot(P.ball.xyz - w, d / dist);
      if (along > 0.0 && along < dist) {
        let miss = length(P.ball.xyz - w - d / dist * along);
        let pen = max(P.src.w, 0.1) * along / dist + P.grid.x;
        vis = discVis(P.ball.w, pen, miss);
      }
    }
    if (vis <= 0.0) { continue; }
    // The soot on the way: outside the slice (beyond the bed's radius from its centre) it dims the light as it does;
    // inside, it is the slice's own, mixed with its glow, and the share that gets out this way is the slice's escape
    // table's (bandEscape), its glowing cells' light marched through the soot about them. From a point on the line
    // the light misses that soot: the flames sit in it, and for a bonfire the line let out some 1.7 times the light,
    // for an oil fire six times (an escape worked out from the soot along the line, as a glowing sphere's, against
    // the slices' cells, one by one).
    let outside = max(dist - max(P.src.w, 0.1), 0.0);
    var tauOut = 0.0;
    for (var i = 0; i < 12; i++) {
      tauOut += density(w + d / dist * (f32(i) + 0.5) / 12.0 * outside);
    }
    tauOut *= outside / 12.0;
    let facing = select(1.0, max(dot(d, n), 0.0) / dist, dot(n, n) > 0.0);
    e += vis * facing * lum * exp(-tauOut) * bandOut(b, w) / (r2 + BAND_R2);
  }
  return 0.001 * e;
}
${bandSpreadWgsl('fireSums')}
// The fire's light on the floor at pf, for the light the floor sends up into the smoke, as the floor's own shading
// takes it (fireOnSurface): each slice spread as its flames are, by the floor's escape (by height), through the soot
// beyond the slice; in kilolux, as fireIrradiance.
fn fireOnFloor(pf: vec3<f32>) -> f32 {
  if (P.fire.x < 0.5) { return 0.0; }
  let vol = P.grid.x * P.grid.x * P.grid.x;
  let up = vec3<f32>(0.0, 1.0, 0.0);
  let rad = max(P.src.w, 0.1);
  var e = 0.0;
  for (var b = 0u; b < FIRE_BANDS; b++) {
    let lum = fireSums[3u + 2u * b].w * vol;
    if (lum <= 0.0) { continue; }
    let d = fireSums[4u + 2u * b].xyz - pf;
    let dist = max(length(d), 1e-6);
    let outside = max(dist - rad, 0.0);
    var tauOut = 0.0;
    for (var i = 0; i < 12; i++) { tauOut += density(pf + d / dist * (f32(i) + 0.5) / 12.0 * outside); }
    tauOut *= outside / 12.0;
    e += lum * bandOutFloor(b, pf, up) * bandSpread(b, pf, up) * exp(-tauOut);
  }
  return 0.001 * e;
}

fn depth(w: vec3<f32>, d: vec3<f32>, ds: f32, ball: bool) -> f32 {
  let te = boxHit(w, d, P.origin.xyz, domainHi()).y;
  // (The collision ball's own shadows the renderer works out exactly, rather than at this lattice's resolution.)
  if (ball && ballHit(w, d) < te) { return 30.0; }
  var tau = 0.0;
  var t = 0.5 * ds;
  for (var i = 0; i < 256; i++) {
    if (t >= te) { break; }
    tau += density(w + d * t);
    t += ds;
  }
  return tau * ds;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ld = textureDimensions(lightOut);
  if (any(id >= ld)) { return; }
  let ext = vec3<f32>(P.dims.xyz) * P.grid.x;
  let w = P.origin.xyz + (vec3<f32>(id) + 0.5) / vec3<f32>(ld) * ext;
  // Occupancy: the largest smoke or soot in this cell's block of the grid, plus its neighbours' (one cell each way),
  // so a march can stride through empty space without stepping over the edge of the smoke.
  let n = nCells();
  let b = vec3<i32>(id) * 2;
  var occ = 0.0;
  for (var z = -1; z <= 2; z++) {
    for (var y = -1; y <= 2; y++) {
      for (var x = -1; x <= 2; x++) {
        let q = textureLoad(scal, clamp(b + vec3<i32>(x, y, z), vec3<i32>(0), n - 1), 0);
        occ = max(occ, q.x);
        // (A fire's fuel too: a flame burns blue where there is no soot yet, or none at all.)
        if (P.fire.x > 0.5) { occ = max(occ, 0.1 * q.w); }
      }
    }
  }
  // Every cell gets its depths, smoke or not: the floor's shadow and sky lookups land in empty cells too.
  let tauSun = depth(w, R.sunDir.xyz, 1.5 * P.grid.x, false);
  let tauUp = depth(w, vec3<f32>(0.0, 1.0, 0.0), 2.0 * P.grid.x, false);
  textureStore(lightOut, vec3<i32>(id), vec4<f32>(tauSun, tauUp, occ, fireIrradiance(w, vec3<f32>(0.0))));
  var tauRim = 0.0;
  if (R.rimDir.w > 0.5) { tauRim = depth(w, R.rimDir.xyz, 1.5 * P.grid.x, false); }
  // The sky's light through the smoke: the dome's mean transmittance (straight up and four ways at 30 degrees), kept as
  // a depth. Halving the depth straight up stood in for it, which let too much sky into the middle of a plume and too
  // little under its cap, where the sky comes in from the sides (against the path-traced reference, 20% over and 35%
  // under).
  var tm = exp(-tauUp);
  for (var k = 0; k < 4; k++) {
    let a = 0.78539816 + 1.5707963 * f32(k);
    tm += exp(-depth(w, vec3<f32>(0.8660254 * cos(a), 0.5, 0.8660254 * sin(a)), 2.0 * P.grid.x, false));
  }
  let tauDome = -log(max(tm / 5.0, 1e-6));
  // The bottom layer also holds the floor's light under it, for the light the floor sends up into the smoke: the sun's
  // share on it (its cosine, through the smoke) and the fire's, as the floor's own shading has it (fireOnFloor). (The
  // sky's is the layer's dome.) (Taken as the slices' point lights by the cosine, the fire's light on the floor under a
  // bonfire's bed came to a sixth to two thirds of the floor's own shading, and a fifth short further out; with that
  // and the octaves, the floor's light in a studio bonfire's flames went from a third of the reference's to six
  // sevenths.)
  var floorLit = vec2<f32>(0.0);
  if (id.y == 0u) {
    let pf = vec3<f32>(w.x, 0.5 * P.grid.x, w.z);
    floorLit = vec2<f32>(max(R.sunDir.y, 0.0) * exp(-depth(pf, R.sunDir.xyz, 1.5 * P.grid.x, false)), fireOnFloor(vec3<f32>(w.x, 0.0, w.z)));
  }
  textureStore(light2Out, vec3<i32>(id), vec4<f32>(tauRim, tauDome, floorLit));
}
`

// Blackbody radiance from the table in blackbody.ts: Planck's law through the CIE colour matching functions, as linear
// Rec.709 in cd/m2. Below the table's range a blackbody is too dim to matter (at 500 K a billionth as bright as at
// 1000 K, itself a dull red glow).
function fireCommon(bbBinding: number) {
  return /* wgsl */ `
@group(0) @binding(${bbBinding}) var<storage, read> bb: array<vec4<f32>>;
const BB_MIN_K = ${BB_MIN_K}.0;
const BB_MAX_K = ${BB_MAX_K}.0;
fn planck(kelvin: f32) -> vec3<f32> {
  if (kelvin <= BB_MIN_K) { return vec3<f32>(0.0); }
  let n = arrayLength(&bb);
  let x = min((kelvin - BB_MIN_K) / (BB_MAX_K - BB_MIN_K), 1.0) * f32(n - 1u);
  let i = u32(floor(x));
  let j = min(i + 1u, n - 1u);
  return mix(bb[i].xyz, bb[j].xyz, x - f32(i));
}
`
}

export const FIRE_SUM_TOTAL = /* wgsl */ `
// x: the number of partial sums (one per workgroup of FIRE_SUM); y, z, w: its workgroups along x, y and z.
@group(0) @binding(0) var<uniform> C: vec4<u32>;
@group(0) @binding(1) var<storage, read> partial: array<vec4<f32>>;
// 0: the emitted intensity summed over cells, per unit of cell volume (rgb, and its luminance); 1: its centre; 2: the
// bed's area (m2) and the heat release rate (W); then for each of FIRE_BANDS slices of the grid's height, its emitted
// intensity (as 0) and its centre; then each slice's escape table, seven entries (RENDER_COMMON's ESCAPE_DIRS), and
// then each slice's scattered light's table, the same; each slice's spread; and each slice's escape table by height.
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;
const FIRE_BANDS = ${FIRE_BANDS}u;
var<workgroup> sE: array<vec4<f32>, 256>;
var<workgroup> sP: array<vec4<f32>, 256>;
var<workgroup> sB: array<vec4<f32>, 256>;

// Workgroup 0 sums everything; workgroup 1 + b sums band b. (The escape tables after the rest, through the centre's
// and the bed's memory: the sums the simulation reads, the bed's area and heat release, keep their order of addition,
// so its run is the same to the last bit.)
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let band = i32(wg.x) - 1;
  var e = vec4<f32>(0.0);
  var p = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  var x0 = vec4<f32>(0.0);
  var x1 = vec4<f32>(0.0);
  var x2 = vec4<f32>(0.0);
  var x3 = vec4<f32>(0.0);
  var x4 = vec4<f32>(0.0);
  var x5 = vec4<f32>(0.0);
  var mo = vec4<f32>(0.0);
  let phase = u32(partial[1].w); // (the step's quarter of the escape table, as FIRE_SUM marched it)
  for (var k = li; k < C.x; k += 256u) {
    if (band >= 0 && i32(((k / C.y) % C.z) * FIRE_BANDS / C.z) != band) { continue; }
    e += partial[10u * k];
    p += partial[10u * k + 1u];
    b += partial[10u * k + 2u];
    x0 += partial[10u * k + 3u];
    x1 += partial[10u * k + 4u];
    x2 += partial[10u * k + 5u];
    x3 += partial[10u * k + 6u];
    x4 += partial[10u * k + 7u];
    x5 += partial[10u * k + 8u];
    mo += partial[10u * k + 9u];
  }
  sE[li] = e;
  sP[li] = p;
  sB[li] = b;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (li < st) {
      sE[li] += sE[li + st];
      sP[li] += sP[li + st];
      sB[li] += sB[li + st];
    }
    workgroupBarrier();
  }
  var lumB = 0.0;
  var cen = vec3<f32>(0.0);
  if (li == 0u) {
    let centre = vec4<f32>(sP[0].xyz / max(sE[0].w, 1e-9), 0.0);
    lumB = sE[0].w;
    cen = centre.xyz;
    if (band < 0) {
      result[0] = sE[0];
      result[1] = centre;
      result[2] = sB[0];
    } else {
      result[3u + 2u * u32(band)] = sE[0];
      result[4u + 2u * u32(band)] = centre;
    }
  }
  if (band < 0) { return; }
  workgroupBarrier();
  sP[li] = x0;
  sB[li] = x1;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (li < st) {
      sP[li] += sP[li + st];
      sB[li] += sB[li + st];
    }
    workgroupBarrier();
  }
  var marched = 0.0;
  if (li == 0u) {
    // This step's directions (k = ESCAPE_PHASES j + phase), over the light of the cells marched (with none, as after a
    // restart, when the last step's sums are another fire's, all of it gets out); then the mean over the sphere from
    // the whole table: each ring's sixteen azimuths by the share of the turn each stands for (half the gaps either
    // side), each ring by the share of the sphere it stands for.
    marched = sB[0].w;
    writeTable(3u + 2u * FIRE_BANDS + ${ESCAPE_ENTRIES}u * u32(band), phase, sP[0], sB[0], marched);
  }
  // Then the scattered light's table the same way (after the escape table, the same layout), and the slice's spread
  // (its moments about the origin over its light, less its centre's).
  workgroupBarrier();
  sP[li] = x2;
  sB[li] = x3;
  sE[li] = mo;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (li < st) {
      sP[li] += sP[li + st];
      sB[li] += sB[li + st];
      sE[li] += sE[li + st];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    writeTable(3u + ${2 + ESCAPE_ENTRIES}u * FIRE_BANDS + ${ESCAPE_ENTRIES}u * u32(band), phase, sP[0], sB[0], marched);
    let m = sE[0] / max(lumB, 1e-30);
    let v = vec4<f32>(m.x - cen.x * cen.x, m.y - cen.y * cen.y, m.z - cen.z * cen.z, m.w - cen.x * cen.z);
    result[3u + ${2 + 2 * ESCAPE_ENTRIES}u * FIRE_BANDS + u32(band)] = select(vec4<f32>(0.0), vec4<f32>(max(v.xyz, vec3<f32>(0.0)), v.w), lumB > 0.0);
  }
  // Then the escape table by height, over the light times the height of the cells marched.
  workgroupBarrier();
  sP[li] = x4;
  sB[li] = x5;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (li < st) {
      sP[li] += sP[li + st];
      sB[li] += sB[li + st];
    }
    workgroupBarrier();
  }
  if (li == 0u) { writeTable(3u + ${3 + 2 * ESCAPE_ENTRIES}u * FIRE_BANDS + ${ESCAPE_ENTRIES}u * u32(band), phase, sP[0], sB[0], sB[0].w); }
}
fn writeTable(base: u32, phase: u32, lo: vec4<f32>, hi: vec4<f32>, marched: f32) {
  for (var j = 0u; j < 7u; j++) {
    let k = ${ESCAPE_PHASES}u * j + phase;
    if (k < ${ESCAPE_NDIRS}u) { // (RENDER_COMMON's ESCAPE_DIRS)
      let v = select(hi[j % 4u], lo[j % 4u], j < 4u);
      result[base + k / 4u][k % 4u] = select(1.0, v / marched, marched > 0.0);
    }
  }
  let w = array<f32, ${ESCAPE_RINGS.length + 2}>(${ESCAPE_WEIGHTS.map(wf).join(', ')});
  var mean = w[0] * result[base + ${ESCAPE_POLES}u].y + w[${ESCAPE_RINGS.length + 1}] * result[base + ${ESCAPE_POLES}u].x;
  var wa = array<f32, ${ESCAPE_AZ / 4}>(${AZ_SHARES.map(wf).join(', ')});
  for (var r = 0u; r < ${ESCAPE_RINGS.length}u; r++) {
    var ring = 0.0;
    for (var a = 0u; a < ${ESCAPE_AZ}u; a++) {
      let k = ${ESCAPE_AZ}u * r + a;
      ring += wa[a % ${ESCAPE_AZ / 4}u] * result[base + k / 4u][k % 4u];
    }
    mean += w[r + 1u] * ring;
  }
  result[base + ${ESCAPE_POLES}u].z = mean;
}
`

// 10. The image, in photometric units: march each camera ray through the volume (Hillaire's integration of each
// step, T (1 - exp(-sigma dt)) S), lit by the sun through the light volume with Wrenninge's multiple-scattering
// octaves (weighted by the smoke around and its albedo) and Henyey-Greenstein phase, by the sky through the smoke over
// it and the floor's light through the smoke under it (smokeAround), and by the fire; hot soot glows at its blackbody
// radiance; then a floor that catches the smoke's shadow, fading to black. Written exposed, at high dynamic range, for
// the accumulation pass.
// (Its library, which the path-traced reference shares: the bindings, the scene and the gas.)
const RENDER_LIB = /* wgsl */ `
${SIM_COMMON}
${RENDER_COMMON}
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var scal: texture_3d<f32>;
@group(0) @binding(4) var light: texture_3d<f32>;
@group(0) @binding(5) var detA: texture_3d<f32>;
@group(0) @binding(6) var detB: texture_3d<f32>;
@group(0) @binding(7) var noiseTex: texture_3d<f32>;
@group(0) @binding(8) var noiseSamp: sampler;
${fireCommon(9)}
@group(0) @binding(10) var<storage, read> fireLight: array<vec4<f32>>;
${bandOutWgsl('fireLight')}
// The geometry of slice b's light on a surface at p facing n, cos / r^2 per unit of its intensity, with the slice
// spread as its flames are (FIRE_SUM's light-weighted spread of its glowing cells): six points along the spread's
// principal axes, root 3 standard deviations out, which hold its spread exactly (the unscented transform; up and down
// no further than keeps both above the floor, which keeps its mean height), each softened by a sixth of BAND_R2.
// Far off, the same as one point at the centre. (One point at the centre left the floor at the edge of a campfire's
// bed with half its light, the flames leaning to that side. Two points an axis, weighted by the light on each side of
// the centre, put the lesser side's further out and a tenth too much light there.)
${bandSpreadWgsl('fireLight')}
@group(0) @binding(11) var blueTex: texture_2d_array<f32>; // spatiotemporal blue noise, a slice a frame
@group(0) @binding(12) var upresTex: texture_3d<f32>; // a smoke source's density on the twice-as-fine grid
@group(0) @binding(13) var flameTex: texture_3d<f32>; // a fire's heat release rate (MW/m3)
@group(0) @binding(32) var<storage, read> ballSheet: array<f32>; // the fuel flowing up the burning ball (BALL_HEAT)
@group(0) @binding(33) var envTex: texture_2d<f32>; // an HDR environment's backdrop, softened (environment.ts)
@group(0) @binding(34) var light2: texture_3d<f32>; // x: optical depth toward the softbox rig's rim (LIGHT)
@group(0) @binding(35) var velTex: texture_3d<f32>; // the velocity, for the motion blur
@group(0) @binding(39) var cycBounce: texture_2d<f32>; // the light the rest of the cyc sends to each point of it (CYC_BOUNCE), lux
@group(0) @binding(43) var cycDirectMap: texture_2d<f32>; // the light straight onto each point of it (CYC_DIRECT), lux
@group(0) @binding(44) var flameNear: texture_3d<f32>; // the fire's light about each point of the flames (FLAME_NEAR), k cd/m2
@group(0) @binding(45) var flameNearDir: texture_3d<f32>; // its luminance's first moment and mean (FLAME_NEAR), k cd/m2
fn rimOn() -> bool { return R.rimDir.w > 0.5; }
// The rim's shadow on a surface point outside the gas: the optical depth where the way to the rim enters the domain.
fn rimShadow(p: vec3<f32>, ch: vec3<f32>) -> vec3<f32> {
  let rimDir = R.rimDir.xyz;
  let sb = boxHit(p, rimDir, P.origin.xyz, domainHi());
  var s = vec3<f32>(1.0);
  if (sb.y > max(sb.x, 0.0)) {
    let pe = p + rimDir * max(sb.x, 0.0) + vec3<f32>(0.0, 0.5 * P.grid.x, 0.0);
    s = exp(-textureSampleLevel(light2, samp, uvwOf(pe), 0.0).x * ch);
  }
  return s * ballSun(p, rimDir);
}

// An HDR environment (environment.ts). A direction into the map's frame: turned about the vertical so the map's light
// faces the light azimuth.
fn envOn() -> bool { return R.env.x > 0.5; }
fn toEnv(d: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(d.x * R.env.y + d.z * R.env.z, d.y, d.z * R.env.y - d.x * R.env.z);
}
// Its light but the key, from the spherical harmonics, with each band weighted (environment.ts shBasis, y up).
fn shAt(d: vec3<f32>, w0: f32, w1: f32, w2: f32) -> vec3<f32> {
  let x = d.x;
  let y = d.y;
  let z = d.z;
  var c = R.sh[0].xyz * (0.282095 * w0);
  c += (R.sh[1].xyz * y + R.sh[2].xyz * z + R.sh[3].xyz * x) * (0.488603 * w1);
  c += (R.sh[4].xyz * (1.092548 * x * y) + R.sh[5].xyz * (1.092548 * y * z) + R.sh[6].xyz * (0.315392 * (3.0 * y * y - 1.0))
    + R.sh[7].xyz * (1.092548 * x * z) + R.sh[8].xyz * (0.546274 * (x * x - z * z))) * w2;
  return max(c, vec3<f32>(0.0));
}
// Scattered toward the camera along rd, per unit of scattering: the map convolved with the Henyey-Greenstein phase
// function, whose bands scale by g^l (Funk-Hecke).
fn envScatter(rd: vec3<f32>, g: f32) -> vec3<f32> { return shAt(toEnv(rd), 1.0, g, g * g); }
// Irradiance on a surface facing n, lux: the map convolved with the clamped cosine (bands pi, 2 pi / 3, pi / 4;
// Ramamoorthi and Hanrahan 2001).
fn envIrradiance(n: vec3<f32>) -> vec3<f32> { return shAt(toEnv(n), 3.14159265, 2.0943951, 0.78539816); }
// The cyc (the Lookdev lab's stage, sized for the fire): the floor runs to CYC_Z behind it, sweeps up a quarter circle
// of radius CYC_R into a wall, as wide as the view. The nearest hit along a ray and the surface's normal there (t is
// 1e9 for none).
const CYC_Z = ${wf(CYC_Z)};
const CYC_R = ${wf(CYC_R)};
const CYC_MAP_X0 = ${wf(CYC_MAP.x0)};
const CYC_MAP_FRONT = ${wf(CYC_MAP.front)};
const CYC_MAP_SIZE = vec2<f32>(${wf(CYC_MAP.w * CYC_MAP.dx)}, ${wf(CYC_MAP.h * CYC_MAP.dx)});
const CYC_FLOOR_LEN = ${wf(CYC_MAP.front - CYC_Z)};
const CYC_SWEEP_LEN = ${wf((Math.PI / 2) * CYC_R)};
// A point of the cyc and its normal from its place on the map (x, and s along it), and back.
fn cycPoint(x: f32, sc: f32) -> array<vec3<f32>, 2> {
  if (sc <= CYC_FLOOR_LEN) { return array<vec3<f32>, 2>(vec3<f32>(x, 0.0, CYC_MAP_FRONT - sc), vec3<f32>(0.0, 1.0, 0.0)); }
  if (sc <= CYC_FLOOR_LEN + CYC_SWEEP_LEN) {
    let a = (sc - CYC_FLOOR_LEN) / CYC_R;
    return array<vec3<f32>, 2>(vec3<f32>(x, CYC_R - CYC_R * cos(a), CYC_Z - CYC_R * sin(a)), vec3<f32>(0.0, cos(a), sin(a)));
  }
  return array<vec3<f32>, 2>(vec3<f32>(x, CYC_R + sc - CYC_FLOOR_LEN - CYC_SWEEP_LEN, CYC_Z - CYC_R), vec3<f32>(0.0, 0.0, 1.0));
}
fn cycUv(p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
  var sc = CYC_MAP_FRONT - p.z;
  if (n.y < 1e-4) { sc = CYC_FLOOR_LEN + CYC_SWEEP_LEN + p.y - CYC_R; }
  else if (n.z > 1e-4) { sc = CYC_FLOOR_LEN + CYC_R * atan2(n.z, n.y); }
  return vec2<f32>((p.x - CYC_MAP_X0) / CYC_MAP_SIZE.x, sc / CYC_MAP_SIZE.y);
}
fn cycHit(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
  var best = vec4<f32>(1e9, 0.0, 1.0, 0.0);
  if (rd.y < -1e-5) {
    let t = -ro.y / rd.y;
    if (ro.z + rd.z * t >= CYC_Z) { best = vec4<f32>(t, 0.0, 1.0, 0.0); }
  }
  // The sweep: a circle in (y, z) about (CYC_R, CYC_Z); its lower back quarter, facing the camera and up.
  let oy = ro.y - CYC_R;
  let oz = ro.z - CYC_Z;
  let a = rd.y * rd.y + rd.z * rd.z;
  let b = oy * rd.y + oz * rd.z;
  let disc = b * b - a * (oy * oy + oz * oz - CYC_R * CYC_R);
  if (disc > 0.0 && a > 1e-8) {
    let s = sqrt(disc);
    for (var k = 0; k < 2; k++) {
      let t = (-b + select(-s, s, k == 1)) / a;
      let py = ro.y + rd.y * t;
      let pz = ro.z + rd.z * t;
      if (t > 0.0 && t < best.x && pz <= CYC_Z && py <= CYC_R) {
        best = vec4<f32>(t, 0.0, (CYC_R - py) / CYC_R, (CYC_Z - pz) / CYC_R);
      }
    }
  }
  if (rd.z < -1e-5) {
    let t = (CYC_Z - CYC_R - ro.z) / rd.z;
    if (t > 0.0 && t < best.x && ro.y + rd.y * t >= CYC_R) { best = vec4<f32>(t, 0.0, 0.0, 1.0); }
  }
  return best;
}

// The map itself, softened, as the backdrop (the Lookdev lab's direction mapping).
fn envBackdrop(d: vec3<f32>) -> vec3<f32> {
  let e = toEnv(d);
  let uv = vec2<f32>(fract(atan2(e.x, -e.z) * INV_2PI + 0.5), acos(clamp(e.y, -1.0, 1.0)) * INV_PI);
  return textureSampleLevel(envTex, noiseSamp, uv, 0.0).xyz * R.env.w;
}

// The fire's light on the collision ball (lux, by colour). Flames touching it: the radiance of the gas within about
// 10 cm of its surface along the normal, as from a hemisphere of it, pi L. The fire further off: each band of its
// height as a light, I cos / r^2, through the soot on the way (the light volume's shadowed light just outside the
// surface over the same bands' unshadowed light gives the mean transmittance). Close to the fire a band is no point
// but a slice of flame as wide as the bed, which lights a surface turned partly away from its centre too: its cosine
// wraps round by how large the slice looms, r_bed^2 / (r^2 + r_bed^2). The reference set shadows each band (all its
// pieces but skip, the one the surface belongs to; -1 for none).
// With the ball's surface (ballMat): the fire's light on it as irradiance for the base (e, lux), its reflection in the
// specular layer (spec, cd/m2), and what the fuzz takes of it (fuzz: the lobe-weighted illuminance, or the radiance of
// flames all around). The flames touching it act as surroundings; each band of the fire as a light of its own.
struct BallFire {
  e: vec3<f32>,
  spec: vec3<f32>,
  fuzz: vec3<f32>,
}
fn ballFire(ph: vec3<f32>, nb: vec3<f32>, n: vec3<f32>, v: vec3<f32>, m: BallMat, shadowed: f32, fireColour: vec3<f32>, ch: vec3<f32>, skip: i32) -> BallFire {
  var out = BallFire(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  if (R.fire.x < 0.5) { return out; }
  let F = tangentFrame(n);
  let nv = max(dot(n, v), 1e-4);
  let near = ballFireNear(ph, nb, ch);
  out.spec = near * envBRDF(m.f0, m.rough, nv);
  out.fuzz = near;
  let cellVol = P.grid.x * P.grid.x * P.grid.x;
  let bed2 = max(P.src.w * P.src.w, 0.04);
  var far = 0.0;
  var seen = 0.0;
  var spec = 0.0;
  var fuzz = 0.0;
  for (var b = 0u; b < FIRE_BANDS; b++) {
    let lum = fireLight[3u + 2u * b].w * cellVol;
    if (lum <= 0.0) { continue; }
    let d = fireLight[4u + 2u * b].xyz - ph;
    let dist = length(d);
    let r2 = dist * dist + BAND_R2;
    let l = d / max(dist, 1e-4);
    let wrap = bed2 / (dist * dist + bed2);
    let c = (dot(n, l) + wrap) / (1.0 + wrap);
    // (Every band counts toward the unshadowed light, as the light volume sums them all; counting only those the
    // surface faces made the transmittance jump where a band's light leaves it, a seam across the surface.) Each band's
    // light as it leaves its own soot toward the ball (bandOut), so the light volume gives only the soot beyond.
    let lout = lum / r2 * bandOut(b, ph);
    seen += lout;
    if (c > 0.0) {
      let e = lout * smoothstep(0.05, 0.15, dist) * refShadowBand(ph, l, dist, b, skip);
      far += e * c;
      if (m.specW > 0.0) { spec += e * ggxSpec(n, v, l, m.rough, m.f0); }
      if (m.fuzzW > 0.0) { fuzz += e * fuzzLobe(F, v, l, m.fuzzRough); }
    }
  }
  let trans = clamp(1000.0 * shadowed / max(seen, 1e-6), 0.0, 1.0);
  out.e = fireColour * far * trans + 3.1415927 * near;
  out.spec += fireColour * spec * trans;
  out.fuzz += fireColour * fuzz * trans;
  return out;
}
// The flames touching the ball: the radiance of the gas just outside its surface, along the normal.
fn ballFireNear(ph: vec3<f32>, nb: vec3<f32>, ch: vec3<f32>) -> vec3<f32> {
  let albedo = R.skyCol.w;
  let step = 2.0 * P.grid.x;
  var near = vec3<f32>(0.0);
  var tn = vec3<f32>(1.0);
  for (var k = 0; k < 2; k++) {
    let uvw = uvwOf(ph + nb * (0.5 + f32(k)) * step);
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { break; }
    let s4 = textureSampleLevel(scal, samp, uvw, 0.0);
    let ts = exp(-sigmaOf(s4) * ch * step);
    let rise = riseOf(s4);
    let heat = textureSampleLevel(flameTex, samp, uvw, 0.0).x * offBallShell(ph + nb * (0.5 + f32(k)) * step);
    if (rise > 400.0 || heat > 0.2) { near += tn * (1.0 - ts) * (1.0 - albedo) * planck(radiatingKelvin(T_AMB + rise, heat)); }
    tn *= ts;
  }
  return near;
}

// The burning ball's own flames. Its volatiles burn in a layer along its surface far thinner than a cell, which the
// grid smears into a dim, blocky shell, so the renderer draws that layer itself (as production renderers draw the
// flames of burning objects), shaped by the physics' map of the fuel flowing up the surface. Along the surface the
// flame is a thin layer, clinging, dim where its flow begins at the lowest burning wood and thicker as more fuel joins
// it on the way up: 0.8 to 3 cm (a laminar wall flame stands some 6 mm off the surface near its leading edge, a
// turbulent one about a sixteenth of its height; Coutin et al. 2003, Orloff et al. via Mitler 1987). Over the surface
// it burns in narrow licks that come and go as they rise, with the char showing between them (a continuous flame film,
// even a faint one, would outshine char glowing at 800 K many times over and coat the ball in orange); at the ball's
// rim, where a line of sight runs along the layer through many licks, it reads as a bright edge. It clings over the
// upper half too, until the flows from every side meet over the top and leave it together in tongues, a plume that
// narrows as it climbs. The flame is present half the time some 2 to 12 cm above the top, and its tips flicker to twice
// that: the heights wall-fire correlations give for this ball's 10 to 30 kW per metre of its girth (Delichatsios 1984
// and Hasemi 1984, about 0.044 and, for the uppermost tips, 0.057 Q'^(2/3) m). Its soot absorbs 3 per metre, as small
// wood flames' does (2 to 3.5 measured; Pastor et al. 2002), and is at 1400 K at the root, cooling to 1050 K at the
// tips (wood fires average 1300 to 1370 K, with brief peaks near 1470 K; Bova and Dickinson 2008). Returns the flame's
// extinction (1/m) and its soot's temperature.
const BALL_FLAME_REACH = 0.45; // m above the surface
const SHEET_REF = 0.002; // kg of fuel per metre of width per second, some 35 kW/m: a ball well alight
// The fuel flow where the outward normal n meets the surface, bilinear like skinAt.
fn sheetAt(n: vec3<f32>) -> f32 {
  let x = fract(atan2(n.z, n.x) * 0.15915494 + 1.0) * f32(SKIN_W) - 0.5;
  let y = clamp(acos(clamp(n.y, -1.0, 1.0)) * 0.31830989 * f32(SKIN_H) - 0.5, 0.0, f32(SKIN_H) - 1.0);
  let x0 = floor(x);
  let y0 = floor(y);
  let i0 = u32((i32(x0) + i32(SKIN_W)) % i32(SKIN_W));
  let i1 = (i0 + 1u) % SKIN_W;
  let j0 = u32(y0);
  let j1 = min(j0 + 1u, SKIN_H - 1u);
  let a = mix(ballSheet[j0 * SKIN_W + i0], ballSheet[j0 * SKIN_W + i1], x - x0);
  let b = mix(ballSheet[j1 * SKIN_W + i0], ballSheet[j1 * SKIN_W + i1], x - x0);
  return mix(a, b, y - y0) / SHEET_REF;
}
// The layer's thickness (m) for a fuel flow s (in SHEET_REF), and how sooty, so bright, the flame is (the layer a third
// as sooty as the tongues it feeds).
fn flameLayer(s: f32) -> f32 { return 0.008 + 0.022 * smoothstep(0.0, 1.0, s); }
fn flameSoot(s: f32) -> f32 { return 3.0 * smoothstep(0.05, 0.5, s); }
fn ballFlame(w: vec3<f32>) -> vec2<f32> {
  if (!ballBurns()) { return vec2<f32>(0.0); }
  let r = P.ball.w;
  let rel = w - P.ball.xyz;
  let dist = length(rel);
  let d = dist - r;
  if (d < 0.0 || d > BALL_FLAME_REACH) { return vec2<f32>(0.0); }
  var out = vec2<f32>(0.0);
  // The layer along the surface: licks some 4 cm across and 20 cm long, rising over it at 2 m/s.
  let s1 = sheetAt(rel / dist);
  let thick = flameLayer(s1);
  if (s1 > 0.0 && d < thick) {
    let q = vec3<f32>(w.x * 25.0, w.y * 5.0 - R.size.z * 2.0, w.z * 25.0);
    let lick = smoothstep(0.5, 0.75, textureSampleLevel(noiseTex, noiseSamp, q, 0.0).x);
    out = vec2<f32>(flameSoot(s1) * lick * (1.0 - smoothstep(0.5, 1.0, d / thick)), 1400.0 - 100.0 * d / thick);
  }
  // The tongues: over the top the flows from every side meet and leave the surface together, rising as a plume that
  // narrows as it climbs (noise stretched upward and rising at 1.6 m/s, thinning toward the tips). Below the top they
  // stay in the layer above, clinging.
  let rho = length(rel.xz);
  let cap = 0.8 * r;
  if (rel.y > 0.0 && rho < cap) {
    let ys = sqrt(r * r - rho * rho);
    let z = rel.y - ys;
    let s2 = sheetAt(vec3<f32>(rel.x, ys, rel.z) / r);
    if (z > 0.0 && s2 > 0.0) {
      // h runs to the uppermost tips, twice the height where flame is present half the time (2 to 12 cm).
      let h = z / (2.1 * (0.02 + 0.10 * smoothstep(0.0, 1.0, s2)));
      let edge = cap * (1.0 - 0.5 * h);
      if (h < 1.0 && rho < edge) {
        let q = vec3<f32>(w.x * 7.0, w.y * 1.8 - R.size.z * 1.6, w.z * 7.0);
        let n = textureSampleLevel(noiseTex, noiseSamp, q, 0.0).x + 0.5 * (textureSampleLevel(noiseTex, noiseSamp, q * 2.3 + 0.37, 0.0).y - 0.5);
        let k = flameSoot(s2) * smoothstep(0.0, 0.15, n - 0.25 - 0.6 * h) * (1.0 - smoothstep(0.6, 1.0, rho / edge));
        if (k > out.x) { out = vec2<f32>(k, mix(1400.0, 1050.0, h)); }
      }
    }
  }
  return out;
}
// Against the collision ball, a weight that fades the gas out within a cell or two of its true surface.
fn offBallStairs(w: vec3<f32>) -> f32 {
  if (P.ball.w <= 0.0) { return 1.0; }
  return smoothstep(0.3, 1.8, (distance(w, P.ball.xyz) - P.ball.w) / P.grid.x);
}
// How near a burning ball's surface w is, as a weight that keeps the solver's smeared shell of flame from glowing
// there (the drawn flames above take its place).
fn offBallShell(w: vec3<f32>) -> f32 {
  if (!ballBurns()) { return 1.0; }
  return smoothstep(1.0, 3.0, (distance(w, P.ball.xyz) - P.ball.w) / P.grid.x);
}

// The wooden ball's surface. Its grain: a turned pine ball, growth rings around a tilted axis, wavy, the early wood
// pale and the late wood darker. Then, by the char's depth: scorched brown, black char, the char's cracks (cells of
// about 2.5 cm, as wood's char crazes), and grey ash in plates. (Colours authored in Rec.709, carried into ACEScg.)
fn hash33(p: vec3<f32>) -> vec3<f32> {
  let q = vec3<f32>(dot(p, vec3<f32>(127.1, 311.7, 74.7)), dot(p, vec3<f32>(269.5, 183.3, 246.1)), dot(p, vec3<f32>(113.5, 271.9, 124.6)));
  return fract(sin(q) * 43758.5453);
}
// How far p is from the nearest edge between Voronoi cells (0 on an edge).
fn cellEdge(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = p - i;
  var d1 = 8.0;
  var d2 = 8.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3<f32>(f32(x), f32(y), f32(z));
        let r = o + hash33(i + o) - f;
        let d = dot(r, r);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
      }
    }
  }
  return sqrt(d2) - sqrt(d1);
}
fn charCracks(lp: vec3<f32>, sk: vec4<f32>) -> f32 {
  return (1.0 - smoothstep(0.02, 0.09, cellEdge(lp * 9.0))) * smoothstep(0.8, 2.5, sk.z);
}
fn woodAlbedo(lp: vec3<f32>, sk: vec4<f32>, crack: f32) -> vec3<f32> {
  let axis = normalize(vec3<f32>(0.35, 1.0, 0.2));
  let wave = textureSampleLevel(noiseTex, noiseSamp, lp * 0.3 + 0.5, 0.0).x - 0.5;
  let ring = fract(length(lp - dot(lp, axis) * axis) * 16.0 + 0.6 * wave);
  let late = smoothstep(0.5, 0.8, ring) * (1.0 - smoothstep(0.85, 1.0, ring));
  var a = mix(vec3<f32>(0.62, 0.45, 0.27), vec3<f32>(0.53, 0.36, 0.2), late);
  a = mix(a, vec3<f32>(0.16, 0.07, 0.03), smoothstep(0.0, 0.1, sk.z));
  a = mix(a, vec3<f32>(0.035, 0.03, 0.028), smoothstep(0.1, 0.7, sk.z));
  a *= 1.0 - 0.6 * crack;
  let plates = smoothstep(0.4, 0.6, textureSampleLevel(noiseTex, noiseSamp, lp * 1.3 + 0.5, 0.0).y);
  return REC709_TO_AP1 * mix(a, vec3<f32>(0.42, 0.41, 0.39), sk.w * mix(0.55, 1.0, plates) * (1.0 - crack));
}
// The burning ball as an OpenPBR surface (the subset it needs, as the Lookdev lab's materials are): a dielectric
// specular layer (GGX) over the diffuse base, the ash as OpenPBR's fuzz on top (the Lookdev lab's own code, Zeltner,
// Burley and Chiang's sheen), and the glow as emission. Turned pine, unfinished: a broad sheen (roughness 0.6, IOR
// 1.5). Char: a faint silvery sheen on its blocks, tighter and a little stronger (roughness 0.4, IOR 1.9, as graphite-
// like carbon reflects more than wood), none down in the cracks. Ash: a pale fuzz, roughness 0.7.
const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;
fn sq(x: f32) -> f32 { return x * x; }
${FUZZ}
struct BallMat {
  base: vec3<f32>, // base colour (ACEScg)
  specW: f32, // specular weight
  rough: f32, // specular roughness (GGX alpha = rough^2)
  f0: f32, // reflectance at normal incidence, from the specular IOR
  fuzzW: f32,
  fuzzColor: vec3<f32>,
  fuzzRough: f32,
}
fn ballMat(lp: vec3<f32>, sk: vec4<f32>, crack: f32) -> BallMat {
  let charred = smoothstep(0.1, 0.7, sk.z);
  let ior = mix(1.5, 1.9, charred);
  let plates = smoothstep(0.4, 0.6, textureSampleLevel(noiseTex, noiseSamp, lp * 1.3 + 0.5, 0.0).y);
  let ash = sk.w * mix(0.55, 1.0, plates) * (1.0 - crack);
  return BallMat(woodAlbedo(lp, sk, crack), (1.0 - crack) * (1.0 - 0.7 * ash), mix(0.6, 0.4, charred), sq((ior - 1.0) / (ior + 1.0)),
    0.8 * ash, REC709_TO_AP1 * vec3<f32>(0.42, 0.41, 0.39), 0.7);
}
// Columns: two tangents and the normal.
fn tangentFrame(n: vec3<f32>) -> mat3x3<f32> {
  let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(cross(a, n));
  return mat3x3<f32>(t, cross(n, t), n);
}
// The cracks have depth: the char dips about 2 mm into each, so at a crack's edges the surface tilts into it (a bump
// from the crack pattern's slope, by finite differences along the surface).
fn crackNormal(lp: vec3<f32>, nb: vec3<f32>, sk: vec4<f32>, crack: f32) -> vec3<f32> {
  if (sk.z < 0.8) { return nb; }
  let F = tangentFrame(nb);
  let e = 0.004;
  let gx = (charCracks(lp + F[0] * e, sk) - crack) / e;
  let gy = (charCracks(lp + F[1] * e, sk) - crack) / e;
  let depth = 0.002 / max(P.ball.w, 0.01);
  return normalize(nb + depth * (gx * F[0] + gy * F[1]));
}
// GGX (Trowbridge-Reitz; Walter and colleagues) with the height-correlated Smith visibility and Schlick's Fresnel: the
// specular lobe times the cosine, for light arriving from l.
fn ggxSpec(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: f32) -> f32 {
  let nl = dot(n, l);
  if (nl <= 0.0) { return 0.0; }
  let nv = max(dot(n, v), 1e-4);
  let h = normalize(v + l);
  let nh = max(dot(n, h), 0.0);
  let vh = max(dot(v, h), 0.0);
  let a2 = sq(rough * rough);
  let d = a2 / (PI * sq(nh * nh * (a2 - 1.0) + 1.0));
  let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
  return d * vis * (f0 + (1.0 - f0) * pow(1.0 - vh, 5.0)) * nl;
}
// The specular lobe's reflectance against light from all around (Karis's analytic fit of the split sum).
fn envBRDF(f0: f32, rough: f32, nv: f32) -> f32 {
  let r = rough * vec4<f32>(-1.0, -0.0275, -0.572, 0.022) + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
  let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
  return f0 * ab.x + ab.y;
}
// The surroundings' radiance about a reflected direction, blurred by the lobe: the environment's harmonics with each
// band weighted as for a Gaussian of the lobe's width (alpha); or the flat sky above, the lit floor below.
fn glossyEnv(r: vec3<f32>, rough: f32) -> vec3<f32> {
  if (envOn()) {
    let s2 = sq(rough * rough);
    return shAt(toEnv(r), 1.0, exp(-s2), exp(-3.0 * s2));
  }
  let sky = R.skyCol.xyz * INV_PI * mix(1.25, 0.75, sqrt(max(r.y, 0.0)));
  return select(R.floorCol.xyz * R.skyCol.xyz * INV_PI, sky, r.y > 0.0);
}
// The fuzz's lobe toward light from l (normalized: it integrates to 1), in the surface's frame F.
fn fuzzLobe(F: mat3x3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32) -> f32 {
  let vl = transpose(F) * v;
  let ll = transpose(F) * l;
  if (ll.z <= 0.0) { return 0.0; }
  return fuzzD(vl, ll, rough);
}
// Kirchhoff again: the surface glows at the blackbody radiance of its temperature (emissivity 0.9). In the cracks the
// char is hotter, out of the cooling air; ash over it glows less.
fn charGlow(sk: vec4<f32>, crack: f32) -> vec3<f32> {
  let t = sk.x + 120.0 * crack * smoothstep(650.0, 800.0, sk.x);
  return 0.9 * (1.0 - 0.5 * sk.w * (1.0 - crack)) * planck(t);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

// Phase relative to isotropic (x 4 pi).
fn hg(c: f32, g: f32) -> f32 {
  return (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}
// The scalars between cells, tricubically: a cubic B-spline, smooth (its slope continuous) where trilinear
// interpolation creases at every cell face, which shows as faceting in smooth smoke. Eight trilinear fetches, placed
// so the hardware's blend gives the B-spline's weights (Sigg and Hadwiger 2005).
fn scalSmooth(uvw: vec3<f32>) -> vec4<f32> {
  if (R.wb1.w < 0.5) { return textureSampleLevel(scal, samp, uvw, 0.0); }
  let n = vec3<f32>(textureDimensions(scal));
  let c = uvw * n - 0.5;
  let i = floor(c);
  let f = c - i;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let a = (i - 0.5 + w1 / g0) / n;
  let b = (i + 1.5 + w3 / g1) / n;
  let s000 = textureSampleLevel(scal, samp, vec3<f32>(a.x, a.y, a.z), 0.0);
  let s100 = textureSampleLevel(scal, samp, vec3<f32>(b.x, a.y, a.z), 0.0);
  let s010 = textureSampleLevel(scal, samp, vec3<f32>(a.x, b.y, a.z), 0.0);
  let s110 = textureSampleLevel(scal, samp, vec3<f32>(b.x, b.y, a.z), 0.0);
  let s001 = textureSampleLevel(scal, samp, vec3<f32>(a.x, a.y, b.z), 0.0);
  let s101 = textureSampleLevel(scal, samp, vec3<f32>(b.x, a.y, b.z), 0.0);
  let s011 = textureSampleLevel(scal, samp, vec3<f32>(a.x, b.y, b.z), 0.0);
  let s111 = textureSampleLevel(scal, samp, vec3<f32>(b.x, b.y, b.z), 0.0);
  let x0 = mix(mix(s000, s100, g1.x), mix(s010, s110, g1.x), g1.y);
  let x1 = mix(mix(s001, s101, g1.x), mix(s011, s111, g1.x), g1.y);
  return mix(x0, x1, g1.z);
}

// Where in its first step each pixel's march starts: spatiotemporal blue noise (Wolfe et al. 2022), a slice a frame,
// so each frame's offsets are blue noise in space (fine-grained, no clumps) and each pixel's offsets over frames are
// blue noise in time too, which the temporal accumulation averages away faster than blue noise stepped by the golden
// ratio (what it began as, and what it still uses until the mask has loaded).
fn jitterAt(p: vec2<f32>, frame: f32) -> f32 {
  let layers = i32(textureNumLayers(blueTex));
  return textureLoad(blueTex, vec2<i32>(p) & vec2<i32>(63), i32(frame) % layers, 0).x;
}

// Upres detail at a point: the noise at each layer's advected coordinates, cross-faded, with the contrast that mixing
// two independent noises loses restored (divided by the blend's standard-deviation factor). A layer the flow has
// stretched would streak its noise along the stretch, so each counts less the more it has stretched (Neyret 2003),
// and where both have, the detail fades rather than streaks.
fn detail(w: vec3<f32>, uvw: vec3<f32>) -> f32 {
  let a = textureSampleLevel(detA, samp, uvw, 0.0);
  let b = textureSampleLevel(detB, samp, uvw, 0.0);
  // A fixed rotation, so the noise lattice lines up with neither the grid nor the camera.
  let rot = mat3x3<f32>(0.8, -0.6, 0.0, 0.36, 0.48, 0.8, -0.48, -0.64, 0.6);
  let na = textureSampleLevel(noiseTex, noiseSamp, rot * (w + a.xyz) * R.detail.w, 0.0).x - 0.5;
  let nb = textureSampleLevel(noiseTex, noiseSamp, rot * (w + b.xyz) * R.detail.w + vec3<f32>(0.37, 0.61, 0.13), 0.0).x - 0.5;
  let wa = R.detail.x / (1.0 + a.w * a.w);
  let wb = R.detail.y / (1.0 + b.w * b.w);
  return (na * wa + nb * wb) / max(sqrt(R.detail.x * R.detail.x + R.detail.y * R.detail.y), 1e-3);
}

// Subgrid temperature. A flame's temperature varies on scales far finer than the grid (thin hot sheets, cooler gaps
// between them), and a blackbody's brightness climbs so steeply with temperature that a flame looks nothing like its
// cell-averaged temperature: the mean of the emission is not the emission at the mean temperature (the
// turbulence-radiation interaction). The fluctuation's size comes from the resolved field, as in LES subgrid-variance
// models (Pierce and Moin 1998): its standard deviation is a constant times the cell size times the temperature
// gradient, so it is large at a flame's edges and nothing in uniform gas. Only its shape is synthesised: a noise
// taller than it is wide (the updraft stretches the flame's sheets), rising with the flames' eddies, drifting with
// the wind and churning as it rises, warped by a second noise so the sheets bend and split. It is not carried by the
// flow, which would stretch it into streaks within a fraction of a second.
const FLAME_TILE = 1.4; // m across the noise tile: the largest sheets about 18 cm wide, 40 cm tall
const SGS_C = 0.4; // the gradient model's constant (the square root of the variance model's, about 0.16)
fn riseAt(uvw: vec3<f32>) -> f32 { return riseOf(textureSampleLevel(scal, samp, uvw, 0.0)); }
fn subgridKelvin(w: vec3<f32>, uvw: vec3<f32>, rot: mat3x3<f32>) -> f32 {
  let e = 1.0 / vec3<f32>(P.dims.xyz);
  let gx = riseAt(uvw + vec3<f32>(e.x, 0.0, 0.0)) - riseAt(uvw - vec3<f32>(e.x, 0.0, 0.0));
  let gy = riseAt(uvw + vec3<f32>(0.0, e.y, 0.0)) - riseAt(uvw - vec3<f32>(0.0, e.y, 0.0));
  let gz = riseAt(uvw + vec3<f32>(0.0, 0.0, e.z)) - riseAt(uvw - vec3<f32>(0.0, 0.0, e.z));
  // dx |grad T| is half the central difference.
  let sd = R.size.w * SGS_C * 0.5 * length(vec3<f32>(gx, gy, gz));
  let t = R.size.z;
  let k = 1.0 / FLAME_TILE;
  var p = (w - vec3<f32>(R.wb0.w, R.fire.w, 0.0) * t) * vec3<f32>(k, 0.45 * k, k) + vec3<f32>(0.0, 0.0, 0.35 * t);
  let warp = textureSampleLevel(noiseTex, noiseSamp, rot * (0.5 * p) + vec3<f32>(0.11, 0.53, 0.29), 0.0).yz - 0.5;
  p.x += 0.9 * warp.x;
  p.z += 0.9 * warp.y;
  let n = textureSampleLevel(noiseTex, noiseSamp, rot * p, 0.0).x;
  // The fbm's values cluster about its middle (a standard deviation of about 0.2): scaled to about unit variance.
  return sd * clamp((n - 0.5) * 5.0, -2.5, 2.5);
}

// The on-set reference (the Shading tab): a colour chart, an 18% grey ball and a chrome ball beside the fire, each ball
// on a thin stand, as a lighting team shoots them on set to record the light where the effect will go: the Lookdev
// lab's own (its ColorChecker, its OpenPBR grey card and chromium). The engine places them on the fire's upwind side,
// clear of its flames. They stand outside the simulation: the gas does not flow around them.
const CHART_GAP = 0.18; // between the patches and around them, in patch widths (the Lookdev lab's)
const CHART_COLOURS = array<vec3<f32>, 25>(${COLOR_CHECKER.map(c => `vec3<f32>(${c.map(wf).join(', ')})`).join(', ')});
fn chartColour(i: i32) -> vec3<f32> {
  let c = CHART_COLOURS;
  return c[i];
}
const CHROME_F0 = vec3<f32>(${SCENE_MATERIALS.chrome.base_color.map(wf).join(', ')});
const CHROME_TINT = vec3<f32>(${SCENE_MATERIALS.chrome.specular_color.map(wf).join(', ')});
// Its reflectance, OpenPBR's metal Fresnel ("F82-tint", the Lookdev lab's fresnelF82Tint): Schlick's from the base
// colour, less a term that brings it, at cos 1/7 (about 82 degrees), to the specular colour's share of Schlick's.
fn schlick3(f0: vec3<f32>, mu: f32) -> vec3<f32> {
  let m = clamp(1.0 - mu, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (1.0 - f0) * (m2 * m2 * m);
}
fn chromeFresnel(mu: f32) -> vec3<f32> {
  let muBar = 1.0 / 7.0;
  let b1 = 1.0 - muBar;
  let b3 = b1 * b1 * b1;
  let a = schlick3(CHROME_F0, muBar) * (1.0 - CHROME_TINT) / (muBar * b3 * b3);
  let x = clamp(mu, 0.0, 1.0);
  let y3 = (1.0 - x) * (1.0 - x) * (1.0 - x);
  return max(schlick3(CHROME_F0, x) - a * x * (y3 * y3), vec3<f32>(0.0));
}
fn refOn() -> bool { return P.refGrey.w > 0.0; }
// What a ray meets of the set first: the grey ball (what 0), the chrome ball (1), the grey ball's stand (2), the chrome
// ball's (3), or the chart (4 + a patch's index, row by row from dark skin, or 4 + 24 its black card). t is 1e9 for
// nothing. (The set's pieces, for refShadow: 0 to 3, and 4 the chart.)
struct RefHit { t: f32, n: vec3<f32>, what: i32 }
fn sphereT(ro: vec3<f32>, rd: vec3<f32>, s: vec4<f32>) -> f32 {
  let oc = ro - s.xyz;
  let b = dot(oc, rd);
  let h = b * b - (dot(oc, oc) - s.w * s.w);
  if (h < 0.0) { return 1e9; }
  let t = -b - sqrt(h);
  return select(1e9, t, t > 1e-4);
}
// A ball's stand: an upright rod from the floor to the ball's underside.
fn standT(ro: vec3<f32>, rd: vec3<f32>, s: vec4<f32>) -> f32 {
  let d = rd.xz;
  let o = ro.xz - s.xz;
  let a = dot(d, d);
  if (a < 1e-10) { return 1e9; }
  let b = dot(o, d);
  let h = b * b - a * (dot(o, o) - P.refChartO.w * P.refChartO.w);
  if (h < 0.0) { return 1e9; }
  let t = (-b - sqrt(h)) / a;
  let y = ro.y + rd.y * t;
  return select(1e9, t, t > 1e-4 && y > 0.0 && y < s.y - s.w);
}
fn refHit(ro: vec3<f32>, rd: vec3<f32>) -> RefHit {
  var h = RefHit(1e9, vec3<f32>(0.0, 1.0, 0.0), -1);
  if (!refOn()) { return h; }
  for (var k = 0; k < 2; k++) {
    let s = select(P.refGrey, P.refChrome, k == 1);
    let tb = sphereT(ro, rd, s);
    if (tb < h.t) { h = RefHit(tb, normalize(ro + rd * tb - s.xyz), k); }
    let ts = standT(ro, rd, s);
    if (ts < h.t) {
      let q = ro + rd * ts - s.xyz;
      h = RefHit(ts, normalize(vec3<f32>(q.x, 0.0, q.z)), 2 + k);
    }
  }
  // The chart (the Lookdev lab's layout): six patches across, four down, on a black card; its back is the card.
  let U = P.refChartU.xyz;
  let V = P.refChartV.xyz;
  let N = normalize(cross(U, V));
  let dn = dot(rd, N);
  if (abs(dn) > 1e-8) {
    let t = dot(P.refChartO.xyz - ro, N) / dn;
    if (t > 1e-4 && t < h.t) {
      let q = ro + rd * t - P.refChartO.xyz;
      let ab = vec2<f32>(dot(q, U) / dot(U, U), dot(q, V) / dot(V, V));
      if (all(ab >= vec2<f32>(0.0)) && all(ab <= vec2<f32>(1.0))) {
        var cell = 24;
        if (dn < 0.0) {
          let gr = ab * vec2<f32>(6.0 + 7.0 * CHART_GAP, 4.0 + 5.0 * CHART_GAP) - CHART_GAP;
          let c = floor(gr / (1.0 + CHART_GAP));
          let f = gr - c * (1.0 + CHART_GAP);
          if (c.x >= 0.0 && c.y >= 0.0 && c.x < 6.0 && c.y < 4.0 && f.x < 1.0 && f.y < 1.0) { cell = i32(c.x) + 6 * (3 - i32(c.y)); }
        }
        h = RefHit(t, select(-N, N, dn < 0.0), 4 + cell);
      }
    }
  }
  return h;
}
// Their surfaces as OpenPBR shades them (BallMat): the grey card's (18%, matte, specular roughness 0.9), each patch
// in its colour as the grey card, the card black, the stands dark and satin.
fn refMat(what: i32) -> BallMat {
  var base = vec3<f32>(0.18);
  var rough = 0.9;
  if (what == 2 || what == 3) {
    base = vec3<f32>(0.03);
    rough = 0.45;
  } else if (what >= 4) {
    base = chartColour(min(what - 4, 24));
  }
  return BallMat(base, 1.0, rough, 0.04, 0.0, vec3<f32>(0.0), 0.5);
}
// How much of a light at distance dist along l reaches p past the set (its balls, their stands and the chart; all but
// the piece skip, the one p is on), soft over a penumbra that widens with the distance behind each piece (as for the
// collision ball): the light as seen across the piece, an ellipse with semi-axes pen0 + kA t and pen0 + kB t along the
// unit vectors uA and uB across the way, t along it. A piece smaller than that hides only its share of the light: a
// ball of radius a, a^2 / (A B) of it, fading out as the two part (discVis, in the ellipse's frame stretched to a
// circle of its area); a stand, a strip of the ellipse along the stand, 4 a rho / (pi A B) where rho is the ellipse's
// radius along it, fading out over its half-width across; the chart as a disc of its area. (Ramped to black whatever
// its size, the set's shadows of the fire on the floor came out two to three times too dark against the path-traced
// reference: a stand 1.6 cm across, in front of a flame a metre wide.)
fn sphereVis(p: vec3<f32>, l: vec3<f32>, dist: f32, s: vec4<f32>, pen0: f32, uA: vec3<f32>, kA: f32, uB: vec3<f32>, kB: f32) -> f32 {
  let tb = s.xyz - p;
  let along = dot(tb, l);
  if (along <= 0.0 || along >= dist) { return 1.0; }
  let A = pen0 + kA * along;
  let B = pen0 + kB * along;
  let pe = sqrt(A * B);
  let m = tb - l * along;
  return discVis(s.w, pe, length(vec2<f32>(dot(m, uA) * pe / A, dot(m, uB) * pe / B)));
}
// A stand: where the way to the light passes its upright line (seen from above), how far from it, if at a height the
// stand reaches.
fn standVis(p: vec3<f32>, l: vec3<f32>, dist: f32, s: vec4<f32>, pen0: f32, uA: vec3<f32>, kA: f32, uB: vec3<f32>, kB: f32) -> f32 {
  let a = dot(l.xz, l.xz);
  if (a < 1e-6) { return 1.0; }
  let t = dot(s.xz - p.xz, l.xz) / a;
  let y = p.y + l.y * t;
  if (t <= 0.0 || t >= dist || y < 0.0 || y > s.y - s.w) { return 1.0; }
  let A = pen0 + kA * t;
  let B = pen0 + kB * t;
  let along = normalize(vec3<f32>(0.0, 1.0, 0.0) - l * l.y);
  let across = vec3<f32>(l.z, 0.0, -l.x) / sqrt(a);
  let ra = dot(along, uA) / A;
  let rb = dot(along, uB) / B;
  let rho = inverseSqrt(ra * ra + rb * rb);
  let wide = length(vec2<f32>(dot(across, uA) * A, dot(across, uB) * B));
  let r = P.refChartO.w;
  let cover = min(4.0 * r * rho / (3.14159265 * A * B), 1.0);
  return 1.0 - cover * (1.0 - smoothstep(max(r - wide, 0.0), r + wide, length(p.xz + l.xz * t - s.xz)));
}
fn refShadowE(p: vec3<f32>, l: vec3<f32>, dist: f32, pen0: f32, uA: vec3<f32>, kA: f32, uB: vec3<f32>, kB: f32, skip: i32) -> f32 {
  if (!refOn()) { return 1.0; }
  var vis = 1.0;
  if (skip != 0) { vis *= sphereVis(p, l, dist, P.refGrey, pen0, uA, kA, uB, kB); }
  if (skip != 1) { vis *= sphereVis(p, l, dist, P.refChrome, pen0, uA, kA, uB, kB); }
  if (skip != 2) { vis *= standVis(p, l, dist, P.refGrey, pen0, uA, kA, uB, kB); }
  if (skip != 3) { vis *= standVis(p, l, dist, P.refChrome, pen0, uA, kA, uB, kB); }
  if (skip == 4) { return vis; }
  // The chart: where the way to the light crosses it, how far inside its edges, against the penumbra there.
  let U = P.refChartU.xyz;
  let V = P.refChartV.xyz;
  let N = normalize(cross(U, V));
  let dn = dot(l, N);
  if (abs(dn) > 1e-6) {
    let t = dot(P.refChartO.xyz - p, N) / dn;
    if (t > 0.0 && t < dist) {
      let q = p + l * t - P.refChartO.xyz;
      let lu = length(U);
      let lv = length(V);
      let a = dot(q, U) / lu;
      let b = dot(q, V) / lv;
      let inside = min(min(a, lu - a), min(b, lv - b));
      let A = pen0 + kA * t;
      let B = pen0 + kB * t;
      let pe = sqrt(A * B);
      // (Its edges for a sharp shadow; for a soft one, a disc of its area, about its centre, in the ellipse's frame.)
      let aeq = sqrt(lu * lv * 0.31830989);
      let m = (U * (0.5 - a / lu) + V * (0.5 - b / lv));
      let mp = m - l * dot(m, l);
      let soft = discVis(aeq, pe, length(vec2<f32>(dot(mp, uA) * pe / A, dot(mp, uB) * pe / B)));
      vis *= mix(smoothstep(-pe, pe, -inside), soft, smoothstep(0.5 * aeq, 2.0 * aeq, pe));
    }
  }
  return vis;
}
// A round light, its radius pen0 + penK t across a piece at t.
fn refShadow(p: vec3<f32>, l: vec3<f32>, dist: f32, pen0: f32, penK: f32, skip: i32) -> f32 {
  var u = vec3<f32>(l.z, 0.0, -l.x);
  u = select(vec3<f32>(1.0, 0.0, 0.0), normalize(u), dot(u, u) > 1e-8);
  return refShadowE(p, l, dist, pen0, u, penK, cross(l, u), penK, skip);
}
// The set's bounding sphere (its stands reach the floor).
fn refBound() -> vec4<f32> {
  let g = P.refGrey;
  let k = P.refChrome;
  let o = P.refChartO.xyz;
  let u = P.refChartU.xyz;
  let v = P.refChartV.xyz;
  var lo = min(min(g.xyz - g.w, k.xyz - k.w), min(min(o, o + u), min(o + v, o + u + v)));
  let hi = max(max(g.xyz + g.w, k.xyz + k.w), max(max(o, o + u), max(o + v, o + u + v)));
  lo.y = 0.0;
  return vec4<f32>(0.5 * (lo + hi), 0.5 * length(hi - lo));
}
// Slice b of the fire's light, dist along l from p, past the set: the slice as its light is spread (FIRE_SUM: its
// variance in x, y and z and its xz covariance, its height no more than keeps it above the floor, as bandSpread takes
// it), seen across the way to it, as the ellipse of a uniform disc of the same spread, twice the standard deviation
// along each axis. (Taken as round and as wide as the bed whatever the slice, the set's shadows of the fire came out
// far wider and shallower than every glowing cell's own way past it: on the floor in front of the set and to its right
// it hid 2.4% of the fire's light where the cells lose 0.8%, and on the cyc's wall beyond 0.7% where they lose 0.1%, up
// to 9% at the wall's far right. As the slices' own spread, 1.1% and 0.1%; behind the set, where the balls cut the
// middle flames' light in narrow streaks, the error over the floor falls by a third.)
fn refShadowBand(p: vec3<f32>, l: vec3<f32>, dist: f32, b: u32, skip: i32) -> f32 {
  if (!refOn()) { return 1.0; }
  let c = fireLight[4u + 2u * b].xyz;
  let m = fireLight[SPREAD_BASE + b];
  let hy = min(sqrt(max(m.y, 0.0) * 3.0), max(c.y - 0.5 * P.grid.x, 0.0));
  let vy = hy * hy / 3.0;
  // (Most ways to a slice pass nowhere near the set: none of it within the widest penumbra the slice could cast.)
  let bs = refBound();
  let tc = clamp(dot(bs.xyz - p, l), 0.0, dist);
  let k0 = 2.0 * sqrt(max(m.x + vy + m.z, 0.0)) / max(dist, 1e-4);
  if (length(bs.xyz - p - l * tc) > bs.w + P.grid.x + k0 * (tc + bs.w)) { return 1.0; }
  var u = vec3<f32>(l.z, 0.0, -l.x);
  u = select(vec3<f32>(1.0, 0.0, 0.0), normalize(u), dot(u, u) > 1e-8);
  let v = cross(l, u);
  let su = vec3<f32>(m.x * u.x + m.w * u.z, vy * u.y, m.w * u.x + m.z * u.z);
  let sv = vec3<f32>(m.x * v.x + m.w * v.z, vy * v.y, m.w * v.x + m.z * v.z);
  let cuu = max(dot(u, su), 0.0);
  let cvv = max(dot(v, sv), 0.0);
  let cuv = dot(u, sv);
  let mid = 0.5 * (cuu + cvv);
  let dev = sqrt(0.25 * (cuu - cvv) * (cuu - cvv) + cuv * cuv);
  let e1 = vec2<f32>(mid + dev - cvv, cuv);
  let e2 = vec2<f32>(cuv, mid + dev - cuu);
  var ev = select(e2, e1, dot(e1, e1) >= dot(e2, e2));
  ev = select(vec2<f32>(1.0, 0.0), normalize(ev), dot(ev, ev) > 1e-20);
  let uA = ev.x * u + ev.y * v;
  let uB = ev.x * v - ev.y * u;
  let k = 2.0 / max(dist, 1e-4);
  return refShadowE(p, l, dist, P.grid.x, uA, sqrt(max(mid + dev, 0.0)) * k, uB, sqrt(max(mid - dev, 0.0)) * k, skip);
}

// A surface's light toward v (cd/m2), with OpenPBR's layering (BallMat): lit by the sun and the rig's rim through the
// smoke (the light volume just outside it, along its geometric normal nb) and past the collision ball (unless it is
// the ball) and the reference set (but its piece skip), by the sky or the environment through the smoke above, and by
// the fire (ballFire). n is the shading normal. The collision ball's surface (ballSurface) and the set's.
fn shadeSurface(ph: vec3<f32>, nb: vec3<f32>, n: vec3<f32>, v: vec3<f32>, m: BallMat, ch: vec3<f32>, fireColour: vec3<f32>, isBall: bool, skip: i32) -> vec3<f32> {
  let sun = R.sunDir.xyz;
  let rimDir = R.rimDir.xyz;
  let lb = textureSampleLevel(light, samp, uvwOf(ph + nb * 1.5 * P.grid.x), 0.0);
  let F = tangentFrame(n);
  let nv = max(dot(n, v), 1e-4);
  // Each light's illuminance at normal incidence (shadowed): the base takes its cosine share, the specular layer
  // and the fuzz their lobes'.
  let sunE = R.sunCol.xyz * exp(-lb.x * ch) * select(ballSun(ph, sun), 1.0, isBall) * refShadow(ph, sun, 1e9, 0.004, 0.0046, skip);
  var E = sunE * max(dot(n, sun), 0.0);
  var spec = sunE * ggxSpec(n, v, sun, m.rough, m.f0);
  var fuzz = sunE * fuzzLobe(F, v, sun, m.fuzzRough);
  if (rimOn()) {
    let lb2 = textureSampleLevel(light2, samp, uvwOf(ph + nb * 1.5 * P.grid.x), 0.0).x;
    let rimE = R.rimCol.xyz * exp(-lb2 * ch) * select(ballSun(ph, rimDir), 1.0, isBall) * refShadow(ph, rimDir, 1e9, 0.01, 0.15, skip);
    E += rimE * max(dot(n, rimDir), 0.0);
    spec += rimE * ggxSpec(n, v, rimDir, m.rough, m.f0);
    fuzz += rimE * fuzzLobe(F, v, rimDir, m.fuzzRough);
  }
  // The sky or the environment, through the smoke above.
  let occ = exp(-0.5 * lb.y * ch);
  let skyE = select(R.skyCol.xyz * (0.5 + 0.5 * n.y), envIrradiance(n), envOn()) * occ;
  E += skyE;
  spec += glossyEnv(reflect(-v, n), m.rough) * envBRDF(m.f0, m.rough, nv) * occ;
  fuzz += skyE * INV_PI;
  let bf = ballFire(ph, nb, n, v, m, lb.w, fireColour, ch, skip);
  E += bf.e;
  spec += bf.spec;
  fuzz += bf.fuzz;
  // OpenPBR's layering: the fuzz on top passes 1 - its albedo; the specular layer reflects its share and passes
  // the rest to the base.
  let eFuzz = select(0.0, fuzzDirAlbedo(nv, m.fuzzRough), m.fuzzW > 0.0);
  let tFuzz = 1.0 - m.fuzzW * eFuzz;
  let base = m.base * INV_PI * E * (1.0 - m.specW * envBRDF(m.f0, m.rough, nv));
  return tFuzz * (base + m.specW * spec) + m.fuzzW * eFuzz * m.fuzzColor * fuzz;
}

// The collision ball's light toward the camera along rd, met at ph: lit by the sun and the sky through the smoke (the
// light volume just outside its surface) and by the fire; matte grey, or wood that scorches, chars, glows and turns to
// ash as its skin heats, shaded as an OpenPBR surface (ballMat).
fn ballSurface(ph: vec3<f32>, rd: vec3<f32>, ch: vec3<f32>, fireColour: vec3<f32>) -> vec3<f32> {
  let nb = normalize(ph - P.ball.xyz);
  var m = BallMat(vec3<f32>(BALL_ALBEDO), 0.0, 1.0, 0.04, 0.0, vec3<f32>(0.0), 0.5);
  var n = nb;
  var glow = vec3<f32>(0.0);
  if (ballBurns()) {
    let lp = (ph - P.ball.xyz) / P.ball.w;
    let sk = skinAt(nb);
    let crack = charCracks(lp, sk);
    m = ballMat(lp, sk, crack);
    n = crackNormal(lp, nb, sk, crack);
    glow = charGlow(sk, crack);
  }
  return shadeSurface(ph, nb, n, -rd, m, ch, fireColour, true, -1) + glow;
}

// The floor's (or the cyc's) light toward the camera, cd/m2: at pf, facing nf, met tFloor along rd.
// The cyc (the gray and charcoal backgrounds): its wall stands as tall and wide as the view behind the fire, so every
// point of it sees the sky only in front, the quarter of directions up and toward the camera (a cosine-weighted
// (n.y + n.z) / 2 of it: half for the floor and for the wall, where the open floor would see all of it), and the rest
// of what it sees is the cyc itself. Against the path-traced reference the floor in front took the whole sky and came
// out up to a quarter too bright, and the wall, without the floor's light, a tenth too dark.
fn cycOn() -> bool { return R.scene.x > 1.5 && !envOn(); }
// The light the rest of the cyc sends to a point p of it facing n (CYC_BOUNCE), lux.
fn cycBounceAt(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(cycBounce, samp, cycUv(p, n), 0.0).xyz;
}

// The fire's light on the floor or the cyc at pf facing nf (lux, of luminance; floorLight says how).
fn fireOnSurface(pf: vec3<f32>, nf: vec3<f32>) -> f32 {
  var fireLux = 0.0;
  var seen = 0.0;
  let cellVol = P.grid.x * P.grid.x * P.grid.x;
  let lo = P.origin.xyz;
  let hi = domainHi();
  var q = pf + nf * P.grid.x;
  if (any(q < lo) || any(q > hi)) {
    let toFire = normalize(fireLight[1].xyz - q);
    let bh = boxHit(q, toFire, lo, hi);
    q = select(clamp(q, lo, hi), q + toFire * (max(bh.x, 0.0) + 0.5 * P.grid.x), bh.y > max(bh.x, 0.0));
  }
  for (var b = 0u; b < FIRE_BANDS; b++) {
    let lum = fireLight[3u + 2u * b].w * cellVol;
    if (lum <= 0.0) { continue; }
    let d = fireLight[4u + 2u * b].xyz - pf;
    let dl = length(d);
    fireLux += lum * bandOutFloor(b, pf, nf) * bandSpread(b, pf, nf) * refShadowBand(pf, d / max(dl, 1e-4), dl, b, -1);
    let dq = fireLight[4u + 2u * b].xyz - q;
    seen += lum * bandOut(b, q) / (dot(dq, dq) + BAND_R2);
  }
  if (seen > 0.0) { fireLux *= clamp(1000.0 * textureSampleLevel(light, samp, uvwOf(q), 0.0).w / seen, 0.0, 1.0); }
  return fireLux;
}
// The light straight onto the floor or the cyc at pf facing nf (lux, rgb): the sun, the sky, the fire and the rig's
// rim, each through the smoke (floorLight adds the rest of the cyc's, CYC_BOUNCE).
fn surfaceDirect(pf: vec3<f32>, nf: vec3<f32>, ch: vec3<f32>, fireColour: vec3<f32>) -> vec3<f32> {
  let sun = R.sunDir.xyz;
  let rimDir = R.rimDir.xyz;
  // The smoke's shadow: the light volume's depth at the point where the sun ray from the floor enters the domain.
  let sb = boxHit(pf, sun, P.origin.xyz, domainHi());
  var shadow = vec3<f32>(1.0);
  if (sb.y > max(sb.x, 0.0)) {
    let pe = pf + sun * max(sb.x, 0.0) + vec3<f32>(0.0, 0.5 * P.grid.x, 0.0);
    shadow = exp(-textureSampleLevel(light, samp, uvwOf(pe), 0.0).x * ch);
  }
  // And the collision ball's and the reference set's; the ball also hides its share of the sky, its view factor from
  // the floor, (r / d)^2 cos.
  shadow *= ballSun(pf, sun) * refShadow(pf, sun, 1e9, 0.004, 0.0046, -1);
  var ballSky = 1.0;
  if (P.ball.w > 0.0) {
    let tb = P.ball.xyz - pf;
    let d2 = dot(tb, tb);
    ballSky = 1.0 - P.ball.w * P.ball.w / d2 * max(dot(tb, nf), 0.0) / sqrt(d2);
  }
  // (The sky through the smoke: the dome's mean transmittance here, light2.y.)
  var skyVis = vec3<f32>(1.0);
  let uv = uvwOf(pf + vec3<f32>(0.0, 0.5 * P.grid.x, 0.0));
  if (all(uv.xz >= vec2<f32>(0.0)) && all(uv.xz <= vec2<f32>(1.0))) {
    skyVis = exp(-textureSampleLevel(light2, samp, uv, 0.0).y * ch);
  }
  // Illuminance on the floor (lux): the sun, the sky and the fire, seen as Lambertian radiance E / pi. The fire's
  // light from each band of its height, I cos / r^2 from the band spread as its flames are (bandSpread), as it leaves
  // the band's own soot toward the point (bandOut), shadowed by the soot beyond: the light volume's shadowed light
  // just above the floor (or, beyond the gas, where the line toward the fire's centre enters it) over the same bands'
  // light there gives the mean transmittance (as for the collision ball); and by the reference set. (Taken at the
  // nearest point of the gas instead, the wall behind a fire read its shadow straight up through the plume above the
  // flames, a far steeper line than the light it takes, and caught under half of it, as little as a ninth. With the
  // bands' own soot in that mean too, a bonfire's floor took the bottom band's: the band it hardly sees, edge on, and
  // the one whose light gets out the least that way, which left the floor in front of it 40% dark. The bands' light
  // there is taken toward q itself, as the light volume takes it: for the bottom band, a few centimetres up, the floor
  // and the point a cell above it lie either side of level, where its light gets out fastest-changing, and the floor
  // lost a tenth.)
  let fireE = fireColour * fireOnSurface(pf, nf);
  // The sky on a surface facing nf: the flat sky's open half (all of it on the floor; on the cyc, its open quarter, the
  // rest being the cyc), or the environment.
  var skyShare = 0.5 + 0.5 * nf.y;
  if (cycOn()) { skyShare = 0.5 * (max(nf.y, 0.0) + max(nf.z, 0.0)); }
  let skyE = select(R.skyCol.xyz * skyShare, envIrradiance(nf), envOn());
  var E = R.sunCol.xyz * max(dot(nf, sun), 0.0) * shadow + skyE * skyVis * ballSky + fireE;
  if (rimOn()) { E += R.rimCol.xyz * max(dot(nf, rimDir), 0.0) * rimShadow(pf, ch) * refShadow(pf, rimDir, 1e9, 0.01, 0.15, -1); }
  return E;
}
fn floorLight(pf: vec3<f32>, nf: vec3<f32>, rd: vec3<f32>, tFloor: f32, ch: vec3<f32>, fireColour: vec3<f32>) -> vec3<f32> {
  let skyStage = R.scene.x > 0.5 && R.scene.x < 1.5;
  // In the studio the floor fades to black away from the fire; under the sky it runs to the horizon; the cyc is lit
  // as it stands, and by the rest of itself (CYC_BOUNCE).
  let fade = select(exp(-dot(pf.xz, pf.xz) / (R.floorCol.w * R.floorCol.w)), 1.0, R.scene.x > 0.5);
  var E = surfaceDirect(pf, nf, ch, fireColour);
  if (cycOn()) { E += cycBounceAt(pf, nf); }
  var floorL = R.floorCol.xyz * INV_PI * E * fade;
  // Under the sky, the far floor fades into the horizon's haze (aerial perspective, over some 40 m).
  let haze = select(R.skyCol.xyz * INV_PI * 1.25, envBackdrop(normalize(vec3<f32>(rd.x, 0.0, rd.z))), envOn());
  if (skyStage) { floorL = mix(floorL, haze, 1.0 - exp(-tFloor / 40.0)); }
  return floorL;
}

// What the chrome ball reflects along rd from ro, cd/m2: the gas, then the collision ball, the rest of the set, the
// floor or the cyc, or the sky. Marched as the camera's rays are but more lightly (twice the step, trilinear, without
// the motion blur, the detail and the subgrid flame temperature: the ball is small in the frame).
fn mirrorView(ro: vec3<f32>, rd: vec3<f32>, jitter: f32, ch: vec3<f32>, fireColour: vec3<f32>) -> vec3<f32> {
  var tFloor = select(1e9, -ro.y / rd.y, rd.y < -1e-5);
  var nFloor = vec3<f32>(0.0, 1.0, 0.0);
  if (R.scene.x > 1.5) {
    let cy = cycHit(ro, rd);
    tFloor = cy.x;
    nFloor = cy.yzw;
  }
  let box = boxHit(ro, rd, P.origin.xyz, domainHi());
  let tBall = ballHit(ro, rd);
  let rh = refHit(ro, rd);
  let tEnd = min(min(min(box.y, tFloor), tBall), rh.t);
  let sun = R.sunDir.xyz;
  let g = R.sunDir.w;
  let cosT = dot(rd, sun);
  let ph0 = hg(cosT, g);
  let ph1 = hg(cosT, 0.5 * g);
  let ph2 = hg(cosT, 0.25 * g);
  let albedo = R.skyCol.w;
  let skyScatter = select(R.skyCol.xyz * INV_PI * skyShareOf(rd, g), envScatter(rd, g), envOn());
  let rimDir = R.rimDir.xyz;
  let pr0 = hg(dot(rd, rimDir), g);
  var col = vec3<f32>(0.0);
  var T = vec3<f32>(1.0);
  if (box.y > max(box.x, 0.0) && box.x < tFloor) {
    let ds = 1.3 * P.grid.x;
    var t = max(box.x, 0.0) + jitter * ds;
    for (var i = 0; i < 1024; i++) {
      if (t > tEnd) { break; }
      let w = ro + rd * t;
      let uvw = uvwOf(w);
      if (textureSampleLevel(light, samp, uvw, 0.0).z < 1e-4) {
        t += 2.0 * ds;
        continue;
      }
      t += ds;
      let s4 = textureSampleLevel(scal, samp, uvw, 0.0);
      let fade = edgeFade(uvw) * offBallStairs(w);
      var sig = sigmaOf(s4) * fade;
      var heat = 0.0;
      if (R.fire.x > 0.5) {
        heat = textureSampleLevel(flameTex, samp, uvw, 0.0).x * offBallShell(w);
        if (heat > 0.0) {
          sig *= sootFormed(gasOf(s4), heat);
          col += T * CHEMI_COLOUR * (CHEMI_CD * heat * fade * ds);
        }
      }
      if (sig > 1e-3) {
        let L = textureSampleLevel(light, samp, uvw, 0.0);
        let near = smokeAround(uvw, w, L, rd, g, albedo, ch, fireColour, skyScatter);
        let lsun = octaves(L.x * ch, ph0, ph1, ph2, near.f, albedo, g) * ballSun(w, sun);
        var S = albedo * (R.sunCol.xyz * INV_4PI * lsun + near.amb + fireColour * 1000.0 * L.w * INV_4PI);
        if (rimOn()) { S += albedo * R.rimCol.xyz * INV_4PI * exp(-near.tauRim * ch) * pr0; }
        var B = vec3<f32>(0.0);
        let rise = riseOf(s4);
        if (R.fire.x > 0.5 && (rise > 400.0 || heat > 0.2)) {
          let glow = planck(radiatingKelvin(T_AMB + rise, heat));
          B = (1.0 - albedo) * glow;
          S += albedo * flameSelfLit(uvw, fireColour * 1000.0 * L.w, rd, g);
        }
        let Ts = exp(-sig * ch * ds);
        col += T * (1.0 - Ts) * (S + B);
        T *= Ts;
        if (max(T.x, max(T.y, T.z)) < 0.004) { break; }
      }
    }
  }
  if (max(T.x, max(T.y, T.z)) <= 0.004) { return col; }
  if (rh.t < min(tFloor, tBall) && rh.t < 1e8) {
    if (rh.what != 1) { col += T * shadeSurface(ro + rd * rh.t, rh.n, rh.n, -rd, refMat(rh.what), ch, fireColour, false, min(rh.what, 4)); }
  } else if (tBall < tFloor && tBall < 1e8) {
    col += T * ballSurface(ro + rd * tBall, rd, ch, fireColour);
  } else if (tFloor < 1e8) {
    col += T * floorLight(ro + rd * tFloor, nFloor, rd, tFloor, ch, fireColour);
  } else if (R.scene.x > 0.5 && R.scene.x < 1.5) {
    col += T * select(R.skyCol.xyz * INV_PI * mix(1.25, 0.75, sqrt(max(rd.y, 0.0))), envBackdrop(rd), envOn());
  }
  return col;
}

// The gas at w as the image pass marches it, step for step: extinction at 550 nm, the soot's glow per unit absorbed, the
// flame's own light per metre; at moment tau of the shutter (0: the step's own). The path-traced reference marches the
// same, and the fire's light is summed from it (FIRE_SUM).
struct Gas3 { sig: f32, glow: vec3<f32>, chem: vec3<f32> }
const ROT = mat3x3<f32>(0.8, -0.6, 0.0, 0.36, 0.48, 0.8, -0.48, -0.64, 0.6);
fn gasAt(w: vec3<f32>, tau: f32, withLight: bool) -> Gas3 {
  var wm = w;
  if (R.camera.x > 0.0 && tau != 0.0) {
    let u = textureSampleLevel(velTex, samp, ((w - P.origin.xyz) / P.grid.x + 0.25) / (nF() + 1.0), 0.0).xyz;
    wm = w - u * tau;
  }
  let uvm = uvwOf(wm);
  let s4 = scalSmooth(uvm);
  let fade = edgeFade(uvm) * offBallStairs(wm);
  var sig = sigmaOf(s4) * fade;
  let upres = R.wb2.w > 0.5;
  if (upres) { sig = R.sunCol.w * max(textureSampleLevel(upresTex, samp, uvm, 0.0).x, 0.0) * fade; }
  var out = Gas3(0.0, vec3<f32>(0.0), vec3<f32>(0.0));
  var heat = 0.0;
  if (R.fire.x > 0.5) {
    heat = textureSampleLevel(flameTex, samp, uvm, 0.0).x * offBallShell(wm);
    if (heat > 0.0) {
      sig *= sootFormed(gasOf(s4), heat);
      if (withLight) { out.chem = CHEMI_COLOUR * (CHEMI_CD * heat * fade); }
    }
  }
  if (sig > 1e-3) {
    if (R.detail.z > 0.0 && !upres) {
      let thin = 1.0 - smoothstep(0.2, 4.0, sig);
      let faint = smoothstep(0.0, 1.0, sig);
      sig *= clamp(1.0 + 2.0 * R.detail.z * detail(wm, uvm) * (0.35 + 0.65 * thin) * faint, 0.0, 2.0);
    }
    let rise = riseOf(s4);
    if (withLight && R.fire.x > 0.5 && (rise > 400.0 || heat > 0.2)) {
      var kelvin = radiatingKelvin(T_AMB + rise, heat);
      if (R.size.w > 0.0) { kelvin = max(T_AMB, kelvin + subgridKelvin(wm, uvm, ROT)); }
      out.glow = (1.0 - R.skyCol.w) * planck(kelvin);
    }
  }
  out.sig = sig;
  return out;
}

// Light that reaches a point of the smoke from a light outside it: scattered there once, through tauS (the optical depth
// toward the light) by the phase p0; and, as Wrenninge's octaves, light that has scattered before, arriving by broader
// phases (p1, p2) and dimmed less. Each further scattering needs the light to meet the smoke again and be scattered
// rather than absorbed, so the first octave is weighted by a = c x albedo x f, with f the chance of meeting the smoke
// again (smokeAround) and c = 1.35 - albedo / 2, and the second by a squared. Scattered light dims with depth as
// diffusion theory has it, at sqrt(3 (1 - albedo) (1 - albedo g)) of the light's own rate (at least half: Wrenninge's
// octaves halve it, which suits clouds, albedo near 1; smoke of albedo 0.6 dims its scattered light about as fast as the
// light itself). The weights and their depths were fitted against the path-traced reference, the sun's light alone, on
// plumes of albedo 0.2, 0.6 and 0.95. (With fixed weights, a half and a quarter, the octaves added three quarters again
// to every point's single scattering, and sunlit smoke came out 30 to 60% brighter than the reference.)
fn octaves(tauS: vec3<f32>, p0: f32, p1: f32, p2: f32, f: f32, albedo: f32, g: f32) -> vec3<f32> {
  let b = clamp(sqrt(3.0 * (1.0 - albedo) * (1.0 - albedo * g)), 0.5, 1.0);
  let a = (1.35 - 0.5 * albedo) * albedo * f;
  return exp(-tauS) * p0 + a * exp(-b * tauS) * p1 + a * a * exp(-b * b * tauS) * p2;
}

// Kirchhoff in the flames: soot that glows sits in the glow about it, which it scatters as it does any light. Where the
// soot is thick the light about it nears the blackbody radiance of the gas within a mean free path or so (in a thick
// flame of even temperature, the blackbody's, whatever the albedo). So a glowing point scatters the fire's light as it
// arrives there (FLAME_NEAR, per unit of albedo, cd/m2): the glowing gas's within a bed's radius, marched, and the rest
// of the fire's as the slices' point lights have it, through the soot on the way, in place of the slices' alone.
// (Taken as the point's own glow in place of a share of the slices', the light about it ran 2.7 times what arrives in
// the bonfire's sooty core, where the hottest gas is the sootiest, and its flames scattered a quarter too much.)
fn flameSelfLit(uvw: vec3<f32>, fireE: vec3<f32>, rd: vec3<f32>, g: f32) -> vec3<f32> {
  // (By the phase to first order in its anisotropy, from the light's first moment: soot scatters forward, and the side
  // of the flames toward the camera scatters the light from the core behind it most.)
  let m = textureSampleLevel(flameNearDir, samp, uvw, 0.0);
  let lobe = clamp(1.0 + 3.0 * g * dot(rd, m.xyz) / max(m.w, 1e-9), 0.0, 2.5);
  return textureSampleLevel(flameNear, samp, uvw, 0.0).xyz * 1000.0 * lobe - fireE * INV_4PI;
}

// The share of a Henyey-Greenstein lobe (anisotropy g) about a direction of elevation sine mu that lies in the upper
// hemisphere: the lobe's Legendre series against the hemisphere's, 1/2 + 3/4 g mu - 7/16 g^3 P3(mu). Smoke that scatters
// forward, seen from below against the sky, takes more than half the sky's light toward the camera (a plume's top, a
// tenth more), and seen from above, more of the floor's.
fn domeShare(mu: f32, g: f32) -> f32 {
  let p3 = 0.5 * mu * (5.0 * mu * mu - 3.0);
  return clamp(0.5 + 0.75 * g * mu - 0.4375 * g * g * g * p3, 0.0, 1.0);
}
// On the cyc the wall, as tall and wide as the view, hides every way back (toward -z) from the smoke in front of it, so
// the smoke sees the sky only over the upper hemisphere's front quarter: its share of the phase, to first order in the
// anisotropy, 1/4 + 3/8 g (y + z) of the view's direction (the quarter's first moment, (0, pi/2, pi/2)). Smoke seen
// against the wall scatters toward the camera mostly the light from behind it, the wall's; taking the whole sky, the
// bonfire's smoke over its flames took 1.6 times the path-traced reference's sky light (with the wall, 1.15).
fn cycQuarterShare(rd: vec3<f32>, g: f32) -> f32 {
  return clamp(0.25 + 0.375 * g * (rd.y + rd.z), 0.0, domeShare(rd.y, g));
}
fn skyShareOf(rd: vec3<f32>, g: f32) -> f32 { return select(domeShare(rd.y, g), cycQuarterShare(rd, g), cycOn()); }

// The light around a point of the smoke (uvw; at wm, the light volume there L) besides the sun, the rim and the fire,
// toward the camera along rd, per unit of albedo (cd/m2): the sky through the smoke over it (the dome's mean
// transmittance) with its own octaves, and the floor's light through the smoke under it. The floor is the one straight
// below (lit by the sun, the sky and the fire, through the smoke over it), its studio fade averaged over the floor the
// point sees, by the lower hemisphere's share of the phase. (A map's spherical harmonics hold the light from below as
// well; with a map, only the map.) Against the path-traced reference the sky's light scattered more than once was 5 to
// 10% of a plume's, and the floor's 1 to 9%, the most low down. (The floor straight below gives a bonfire's smoke over
// its flames half again the fire's light off the floor that the reference does, and the gas low beside its bed a third.
// Gathered over the whole lower hemisphere instead, from a map of the floor's light through the smoke, 24 to 128
// directions, its transmittance within 3% of the reference gas's marched a cell a step with its detail, and with the
// octaves' weights and the similarity relation, the low gas came to four fifths, but the flames and the smoke beside
// them fell to half and the frame came no closer: the light scattered on through the flames' soot, which neither has,
// is likely what is left.) Also f, the chance that light scattered here meets the smoke again (the mean of
// 1 - exp(-depth) toward the sun, up and down), and the depths down and toward the rim.
struct Around { amb: vec3<f32>, f: f32, tauDown: f32, tauRim: f32 }
fn smokeAround(uvw: vec3<f32>, wm: vec3<f32>, L: vec4<f32>, rd: vec3<f32>, g: f32, albedo: f32, ch: vec3<f32>, fireColour: vec3<f32>, skyScatter: vec3<f32>) -> Around {
  let L2 = textureSampleLevel(light2, samp, uvw, 0.0);
  let uvF = vec3<f32>(uvw.x, 0.5 / f32(textureDimensions(light2).y), uvw.z);
  let tauDown = max(textureSampleLevel(light, samp, uvF, 0.0).y - L.y, 0.0);
  let f = 1.0 - (exp(-L.x) + exp(-L.y) + exp(-tauDown)) / 3.0;
  var amb = skyScatter * exp(-L2.y * ch);
  if (!envOn()) {
    let b = clamp(sqrt(3.0 * (1.0 - albedo) * (1.0 - albedo * g)), 0.5, 1.0);
    let a = (1.35 - 0.5 * albedo) * albedo * f;
    // (On the cyc, the wall behind and above: its light as the map has it, at the point straight behind, by its share
    // of the phase; and the light scattered more than once, half the sky's, half the wall's.)
    var skyOct = R.skyCol.xyz * INV_2PI;
    if (cycOn()) {
      let wp = vec3<f32>(wm.x, max(wm.y, CYC_R), CYC_Z - CYC_R);
      let wuv = cycUv(wp, vec3<f32>(0.0, 0.0, 1.0));
      let wallL = R.floorCol.xyz * INV_PI * (textureSampleLevel(cycDirectMap, samp, wuv, 0.0).xyz + textureSampleLevel(cycBounce, samp, wuv, 0.0).xyz);
      amb += wallL * max(domeShare(rd.y, g) - cycQuarterShare(rd, g), 0.0) * exp(-L2.y * ch);
      skyOct = R.skyCol.xyz * INV_PI * 0.25 + wallL * 0.25;
    }
    amb += skyOct * (a * exp(-b * L2.y * ch) + a * a * exp(-b * b * L2.y * ch));
    let F2 = textureSampleLevel(light2, samp, uvF, 0.0);
    let r2 = R.floorCol.w * R.floorCol.w;
    let fade = select(exp(-dot(wm.xz, wm.xz) / r2) / (1.0 + wm.y * wm.y / r2), 1.0, R.scene.x > 0.5);
    var floorE = R.sunCol.xyz * F2.z + R.skyCol.xyz * exp(-F2.y * ch) + fireColour * 1000.0 * F2.w;
    // (On the cyc the floor sees half the sky, and the rest of the cyc's light, CYC_BOUNCE.)
    if (cycOn()) { floorE += R.skyCol.xyz * exp(-F2.y * ch) * -0.5 + cycBounceAt(vec3<f32>(wm.x, 0.0, wm.z), vec3<f32>(0.0, 1.0, 0.0)); }
    let floorL = R.floorCol.xyz * INV_PI * fade * floorE;
    // (And its light scattered more than once, as the sky's octaves take it, over the lower half of the sphere: split
    // by order, more than half of the floor's light in a bonfire's flames and the smoke about them comes so, and with
    // the light scattered once alone the flames had a third of the path-traced reference's floor light.)
    amb += floorL * ((1.0 - domeShare(rd.y, g)) * exp(-tauDown * ch) + 0.5 * (a * exp(-b * tauDown * ch) + a * a * exp(-b * b * tauDown * ch)));
  }
  return Around(amb, f, tauDown, L2.x);
}
`

// 9c. The cyc's light: a map of it (CYC_MAP), across the view and along its floor, sweep and wall. First the light
// straight onto each point (CYC_DIRECT, as floorLight's surfaceDirect has it); then the light the rest of the cyc sends
// it (CYC_BOUNCE): 32 rays over the point's sky, cosine-weighted in rings, to where they meet the cyc, its direct light
// and its bounce there (the map's last, so each update adds an order of the bounce), times its albedo, through the
// smoke on the way. Twice an update. (In its place, the wall's light straight behind the fire at the flames' height, its
// brightest, gave the far floor of a bonfire five times the fire's light back that the path-traced reference does, and
// a patch grid of the fire-lit floor about the bed gave the wall a third to a fifth of the light it takes from it.
// Where a bonfire's floor near the fire shows, the reference brings two thirds of the bounced fire light by way of the
// smoke, the low gas lit by the floor and what it sends back down, of which smokeAround's floor light gives a fifth.)
export const CYC_DIRECT = /* wgsl */ `
${RENDER_LIB}
@group(0) @binding(40) var directOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = textureDimensions(light);
  _ = textureDimensions(light2);
  _ = fireLight[0];
  let dims = textureDimensions(directOut);
  if (any(id.xy >= dims)) { return; }
  var e = vec3<f32>(0.0);
  if (cycOn()) {
    let pn = cycPoint(CYC_MAP_X0 + (f32(id.x) + 0.5) * CYC_MAP_SIZE.x / f32(dims.x), (f32(id.y) + 0.5) * CYC_MAP_SIZE.y / f32(dims.y));
    let fireColour = select(vec3<f32>(0.0), fireLight[0].xyz / max(fireLight[0].w, 1e-12), fireLight[0].w > 0.0);
    e = surfaceDirect(pn[0], pn[1], chroma(), fireColour);
  }
  textureStore(directOut, vec2<i32>(id.xy), vec4<f32>(e, 0.0));
}
`

export const CYC_BOUNCE = /* wgsl */ `
${RENDER_LIB}
@group(0) @binding(40) var bounceOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(41) var cycDirect: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  _ = textureDimensions(scal);
  let dims = textureDimensions(bounceOut);
  if (any(id.xy >= dims)) { return; }
  var e = vec3<f32>(0.0);
  if (cycOn()) {
    let pn = cycPoint(CYC_MAP_X0 + (f32(id.x) + 0.5) * CYC_MAP_SIZE.x / f32(dims.x), (f32(id.y) + 0.5) * CYC_MAP_SIZE.y / f32(dims.y));
    let p = pn[0];
    let n = pn[1];
    let t1 = vec3<f32>(1.0, 0.0, 0.0);
    let t2 = vec3<f32>(0.0, n.z, -n.y);
    let ch = chroma();
    let lo = P.origin.xyz;
    let hi = domainHi();
    for (var k = 0u; k < 32u; k++) {
      let u = (f32(k / 8u) + 0.5) / 4.0;
      let phi = 6.2831853 * (f32(k % 8u) + 0.5 * f32((k / 8u) & 1u) + 0.25) / 8.0;
      let d = (t1 * cos(phi) + t2 * sin(phi)) * sqrt(u) + n * sqrt(1.0 - u);
      let h = cycHit(p + n * 1e-3, d);
      if (h.x >= 1e8) { continue; }
      let hp = p + d * h.x;
      let uv = cycUv(hp, vec3<f32>(0.0, h.z, h.w));
      var src = textureSampleLevel(cycDirect, samp, uv, 0.0).xyz + textureSampleLevel(cycBounce, samp, uv, 0.0).xyz;
      // (Through the smoke on the way, where the way crosses the gas.)
      let bh = boxHit(p, d, lo, hi);
      let t0 = max(bh.x, 0.0);
      let t1b = min(bh.y, h.x);
      if (t1b > t0) {
        var od = 0.0;
        for (var i = 0; i < 12; i++) { od += sigmaOf(textureSampleLevel(scal, samp, uvwOf(p + d * (t0 + (f32(i) + 0.5) * (t1b - t0) / 12.0)), 0.0)); }
        src *= exp(-od * (t1b - t0) / 12.0 * ch);
      }
      e += src;
    }
    e *= R.floorCol.xyz / 32.0;
  }
  textureStore(bounceOut, vec2<i32>(id.xy), vec4<f32>(e, 0.0));
}
`

// 9e. The fire's light about the flames (FLAME_NEAR), at each point of the light lattice within reach of the flames
// (each slice's light-weighted centre, three standard deviations of its spread and a bed's radius out) where the image
// will read it (a cell glows within three of the point's own two: the image takes this light in glowing gas alone, from
// the lattice's eight points about it); elsewhere nothing is marched (about 1 ms a frame for the bonfire). The glowing
// gas's light within a bed's radius, 64 ways spread evenly over the sphere, a cell and a half a step, from each cell's
// glow and extinction as the fire sums drew them (eight points a cell, its subgrid temperature among them: one read a
// step), dimmed on the way by absorption alone (in a thick flame of even temperature that tends to its blackbody
// radiance, as Kirchhoff's law has it, scattering and all); and the rest of the fire's light, the slices' as point
// lights (with their escape, and the smoke's shadow as the light volume has it), each less the share of its light
// within the bed's radius of the point (already marched: the share of a Gaussian of the slice's spread within that
// ball), through the near soot the way it comes. With them, the light's first moment, for the phase (flameSelfLit).
// (Counting the slices' whole light again on top of the near light lit the low flames at the bonfire's bed a third to
// two thirds too bright; marched along the axes and the cube's diagonals, four ways lay in the bed's thin hot sheet,
// and its edges came out two fifths too bright. Sixteen ways, each a sixteenth of the sphere, saw a thin bright layer
// edgewise by one near-level way only: above the bed it caught the sheet's long grazing path or missed it, by the
// height, and the flames just above came out two fifths too bright, which in a frame's sum hid the upper flames' tips
// being too dark.)
export const FLAME_NEAR = /* wgsl */ `
${RENDER_LIB}
@group(0) @binding(40) var nearOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(41) var emitTex: texture_3d<f32>;
@group(0) @binding(42) var nearDirOut: texture_storage_3d<rgba16float, write>;

// The standard normal's distribution function (a tanh fit, within 2e-4).
fn normCdf(x: f32) -> f32 { return 0.5 * (1.0 + tanh(0.7978846 * (x + 0.044715 * x * x * x))); }
// The share of an isotropic Gaussian (standard deviation s, its centre d away) within l of a point: the non-central chi
// distribution's, three degrees of freedom.
fn ballShare(d: f32, s: f32, l: f32) -> f32 {
  let dd = max(d, 1e-3);
  let a = (l - dd) / s;
  let b = (l + dd) / s;
  return clamp(normCdf(a) - normCdf(-b) - s / (dd * 2.5066283) * (exp(-0.5 * a * a) - exp(-0.5 * b * b)), 0.0, 1.0);
}
const NEAR_N = 64u;
const NEAR_STEP = 1.5; // (cells)
fn nearDir(k: u32) -> vec3<f32> {
  let mu = 1.0 - (f32(k) + 0.5) * 2.0 / f32(NEAR_N);
  let sn = sqrt(max(1.0 - mu * mu, 0.0));
  let phi = 2.39996323 * f32(k);
  return vec3<f32>(sn * cos(phi), mu, sn * sin(phi));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ld = textureDimensions(nearOut);
  if (any(id >= ld)) { return; }
  let uvw = (vec3<f32>(id) + 0.5) / vec3<f32>(ld);
  let w = P.origin.xyz + uvw * vec3<f32>(P.dims.xyz) * P.grid.x;
  let rad = max(P.src.w, 0.1);
  var out = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  var mom = vec4<f32>(0.0);
  // (Within reach of the flames only.)
  var near = false;
  if (R.fire.x > 0.5) {
    for (var b = 0u; b < FIRE_BANDS; b++) {
      if (fireLight[3u + 2u * b].w <= 0.0) { continue; }
      let sd = sqrt(max(fireLight[SPREAD_BASE + b].xyz, vec3<f32>(0.0)));
      if (all(abs(w - fireLight[4u + 2u * b].xyz) < 3.0 * sd + rad + P.grid.x)) { near = true; }
    }
  }
  // (And only where the image will read it: some cell glows within three of the point's own two, the lattice being the
  // grid's at half resolution.)
  if (near) {
    let nc = vec3<i32>(textureDimensions(emitTex));
    let c0 = 2 * vec3<i32>(id) - 3;
    var lit = false;
    for (var z = 0; z < 7 && !lit; z++) {
      for (var y = 0; y < 7 && !lit; y++) {
        for (var x = 0; x < 7; x++) {
          let c = c0 + vec3<i32>(x, y, z);
          if (any(c < vec3<i32>(0)) || any(c >= nc)) { continue; }
          let q = textureLoad(emitTex, c, 0);
          if (max(q.x, max(q.y, q.z)) > 0.0) { lit = true; break; }
        }
      }
    }
    near = lit;
  }
  if (near) {
    let ch = chroma() * (1.0 - R.skyCol.w);
    let ds = NEAR_STEP * P.grid.x;
    let lo = P.origin.xyz;
    let hi = domainHi();
    var l = vec3<f32>(0.0);
    var dip = vec3<f32>(0.0);
    var tk: array<f32, NEAR_N>; // (each set as its way is marched)
    for (var k = 0u; k < NEAR_N; k++) {
      let d = nearDir(k);
      let reach = min(rad, max(boxHit(w, d, lo, hi).y, 0.0));
      var tr = vec3<f32>(1.0);
      var lk = vec3<f32>(0.0);
      for (var i = 0; i < 64; i++) {
        let t = (f32(i) + 0.5) * ds;
        if (t > reach) { break; }
        let q = textureSampleLevel(emitTex, samp, uvwOf(w + d * t), 0.0);
        let a = exp(-q.w * ch * ds);
        lk += tr * q.xyz * ds * select((vec3<f32>(1.0) - a) / max(-log(a), vec3<f32>(1e-6)), vec3<f32>(1.0), a > vec3<f32>(0.9999));
        tr *= a;
      }
      l += lk;
      dip += d * ap1Lum(lk);
      tk[k] = dot(tr, vec3<f32>(0.2676, 0.6744, 0.058));
    }
    // The rest of the fire's light: each slice's, less its share within the bed's radius, through the near soot toward
    // it; the smoke's shadow beyond as the light volume has it (its light over the slices' unshadowed).
    let vol = P.grid.x * P.grid.x * P.grid.x;
    var far = 0.0;
    var dfar = vec3<f32>(0.0);
    var seen = 0.0;
    for (var b = 0u; b < FIRE_BANDS; b++) {
      let lum = fireLight[3u + 2u * b].w * vol;
      if (lum <= 0.0) { continue; }
      let dv = fireLight[4u + 2u * b].xyz - w;
      let dist = length(dv);
      let e = lum * bandOut(b, w) / (dist * dist + BAND_R2);
      seen += e;
      let sd = sqrt(max(dot(fireLight[SPREAD_BASE + b].xyz, vec3<f32>(1.0 / 3.0)), P.grid.x * P.grid.x));
      var best = 0u;
      var bd = -2.0;
      for (var k = 0u; k < NEAR_N; k++) { let c = dot(nearDir(k), dv / max(dist, 1e-6)); if (c > bd) { bd = c; best = k; } }
      let fb = e * (1.0 - ballShare(dist, sd, rad)) * tk[best];
      far += fb;
      dfar += dv / max(dist, 1e-6) * fb;
    }
    let shade = select(1.0, clamp(1000.0 * textureSampleLevel(light, samp, uvw, 0.0).w / max(seen, 1e-20), 0.0, 1.0), seen > 0.0);
    far *= shade;
    let fireColour = select(vec3<f32>(0.0), fireLight[0].xyz / max(fireLight[0].w, 1e-12), fireLight[0].w > 0.0);
    out = vec4<f32>(l / f32(NEAR_N) + fireColour * far * INV_4PI * 1e-3, 1.0);
    mom = vec4<f32>(dip / f32(NEAR_N) + dfar * shade * ap1Lum(fireColour) * INV_4PI * 1e-3, ap1Lum(out.xyz));
  }
  textureStore(nearOut, vec3<i32>(id), out);
  textureStore(nearDirOut, vec3<i32>(id), mom);
}
`

export const RENDER = /* wgsl */ `
${RENDER_LIB}
@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  // Each frame's rays pass through a different point of their pixel (the R2 sequence), which the accumulation turns
  // into antialiasing (of the ball's outline, the floor's shadows and the flames' edges).
  let sub = fract(vec2<f32>(0.7548777, 0.5698403) * R.camRight.w + 0.5) - 0.5;
  let ndc = vec2<f32>((fc.x + sub.x) / R.size.x * 2.0 - 1.0, 1.0 - (fc.y + sub.y) / R.size.y * 2.0);
  let tanY = R.camPos.w;
  let rd = normalize(R.camFwd.xyz + R.camRight.xyz * ndc.x * tanY * R.camFwd.w + R.camUp.xyz * ndc.y * tanY);
  let ro = R.camPos.xyz;
  let frame = R.camRight.w;
  let jitter = jitterAt(fc.xy, frame);
  // The ray's moment within the shutter (the motion blur): blue noise of its own, apart from the march's offset, so
  // where the flow along the ray matches the step the two do not land every pixel on the same points.
  let shutterTau = (jitterAt(fc.xy + vec2<f32>(32.0, 32.0), frame) - 0.5) * R.camera.x;

  // The stage under and behind the fire: the floor's plane, or the cyc (and the normal where the ray meets it).
  var tFloor = select(1e9, -ro.y / rd.y, rd.y < -1e-5);
  var nFloor = vec3<f32>(0.0, 1.0, 0.0);
  if (R.scene.x > 1.5) {
    let cy = cycHit(ro, rd);
    tFloor = cy.x;
    nFloor = cy.yzw;
  }
  let skyStage = R.scene.x > 0.5 && R.scene.x < 1.5;
  let box = boxHit(ro, rd, P.origin.xyz, domainHi());
  let tBall = ballHit(ro, rd);
  let rh = refHit(ro, rd);
  let tEnd = min(min(min(box.y, tFloor), tBall), rh.t);
  let sun = R.sunDir.xyz;
  let g = R.sunDir.w;
  let cosT = dot(rd, sun);
  let ph0 = hg(cosT, g);
  let ph1 = hg(cosT, 0.5 * g);
  let ph2 = hg(cosT, 0.25 * g);
  let albedo = R.skyCol.w;
  let ch = chroma();
  let rot = mat3x3<f32>(0.8, -0.6, 0.0, 0.36, 0.48, 0.8, -0.48, -0.64, 0.6);
  // The fire's colour at unit luminance (its light's strength comes from the light volume).
  let fireColour = fireLight[0].xyz / max(fireLight[0].w, 1e-12);
  // The sky's light scattered toward the camera, before the smoke dims it: a uniform dome over open ground (radiance
  // E / pi, by its share of the phase, domeShare), or an HDR environment through the phase function.
  let skyScatter = select(R.skyCol.xyz * INV_PI * skyShareOf(rd, g), envScatter(rd, g), envOn());
  // The softbox rig's rim, through the same phase octaves as the sun.
  let rimDir = R.rimDir.xyz;
  let cosR = dot(rd, rimDir);
  let pr0 = hg(cosR, g);
  let pr1 = hg(cosR, 0.5 * g);
  let pr2 = hg(cosR, 0.25 * g);

  // Radiance in cd/m2, per colour channel (soot dims blue more than red, so transmittance is a colour too).
  var col = vec3<f32>(0.0);
  var T = vec3<f32>(1.0);
  if (box.y > max(box.x, 0.0) && box.x < tFloor) {
    let t0 = max(box.x, 0.0);
    let ds = 0.65 * P.grid.x; // the temporal accumulation averages away the error of the longer step
    // Empty space (no smoke in a light cell's neighbourhood) is crossed in strides of a light cell.
    var t = t0 + jitter * ds;
    for (var i = 0; i < 2048; i++) {
      if (t > tEnd) { break; }
      let w = ro + rd * t;
      let uvw = uvwOf(w);
      // (Near the ball the light lattice's cells straddle it and read as empty, and near a burning ball are its flames,
      // so there the march takes every step.)
      let toBall = distance(w, P.ball.xyz) - P.ball.w;
      let nearFlames = ballBurns() && toBall < BALL_FLAME_REACH;
      let nearBall = P.ball.w > 0.0 && toBall < 0.15;
      if (textureSampleLevel(light, samp, uvw, 0.0).z < 1e-4 && !nearFlames && !nearBall) {
        t += 3.0 * ds;
        continue;
      }
      t += ds;
      if (nearFlames) {
        let bf = ballFlame(w);
        if (bf.x > 0.0) {
          let tf = exp(-bf.x * ds);
          col += T * (1.0 - tf) * (1.0 - albedo) * planck(bf.y);
          T *= tf;
        }
      }
      // Motion blur: the ray's moment within the shutter, one for all its samples (a different one each pixel and
      // frame, which the accumulation averages), and the gas that was here then, carried back along the flow. A sheet
      // of soot that moves across the ray covers it for part of the shutter; with a moment for each sample, as this
      // took them, it dimmed the light behind it at every sample instead, and the moving flames came out a tenth to two
      // fifths off the path-traced reference's (a moment a path) where they move most. (The velocity in one lookup:
      // each component lives on its own faces, a quarter cell either way of where this reads it, near enough for a
      // blur.)
      var wm = w;
      if (R.camera.x > 0.0) {
        let u = textureSampleLevel(velTex, samp, ((w - P.origin.xyz) / P.grid.x + 0.25) / (nF() + 1.0), 0.0).xyz;
        wm = w - u * shutterTau;
      }
      let uvm = uvwOf(wm);
      let s4 = scalSmooth(uvm);
      // (Against the collision ball the grid holds it as a staircase of solid cells, which the gas within a cell or
      // two of its true surface takes the shape of; there the gas fades out and the sphere itself takes over.)
      let fade = edgeFade(uvm) * offBallStairs(wm);
      var sig = sigmaOf(s4) * fade;
      let upres = R.wb2.w > 0.5;
      if (upres) { sig = R.sunCol.w * max(textureSampleLevel(upresTex, samp, uvm, 0.0).x, 0.0) * fade; }
      // Where a fire burns: the flame's own blue light, and only the soot formed by now (sootFormed). The blue is there
      // all through the flame, hidden wherever soot glows in front of it, and shows where the gas is clear.
      var heat = 0.0;
      if (R.fire.x > 0.5) {
        heat = textureSampleLevel(flameTex, samp, uvm, 0.0).x * offBallShell(wm);
        if (heat > 0.0) {
          sig *= sootFormed(gasOf(s4), heat);
          col += T * CHEMI_COLOUR * (CHEMI_CD * heat * fade * ds);
        }
      }
      if (sig > 1e-3) {
        // Upres detail (unless the wavelet-turbulence grid supplies it): strongest in thin smoke (the edges and wisps),
        // a third as strong in the dense core, and fading out in the faintest haze, where fine noise reads as grain.
        if (R.detail.z > 0.0 && !upres) {
          let thin = 1.0 - smoothstep(0.2, 4.0, sig);
          let faint = smoothstep(0.0, 1.0, sig);
          sig *= clamp(1.0 + 2.0 * R.detail.z * detail(wm, uvm) * (0.35 + 0.65 * thin) * faint, 0.0, 2.0);
        }
        let L = textureSampleLevel(light, samp, uvm, 0.0);
        let near = smokeAround(uvm, wm, L, rd, g, albedo, ch, fireColour, skyScatter);
        let lsun = octaves(L.x * ch, ph0, ph1, ph2, near.f, albedo, g) * ballSun(wm, sun);
        // The fire's light, shadowed by the soot on the way (from the light volume, in kilolux).
        let fireE = fireColour * 1000.0 * L.w;
        // Light scattered toward the camera per unit of extinction, cd/m2: the phase function is hg / (4 pi).
        var S = albedo * (R.sunCol.xyz * INV_4PI * lsun + near.amb + fireE * INV_4PI);
        if (rimOn()) {
          let fRim = 1.0 - (exp(-near.tauRim) + exp(-L.y) + exp(-near.tauDown)) / 3.0;
          let lrim = octaves(near.tauRim * ch, pr0, pr1, pr2, fRim, albedo, g) * ballSun(wm, rimDir);
          S += albedo * R.rimCol.xyz * INV_4PI * lrim;
        }
        // Emission (Kirchhoff's law): what the soot absorbs, it emits at the blackbody radiance of its temperature.
        var B = vec3<f32>(0.0);
        let rise = riseOf(s4);
        if (R.fire.x > 0.5 && (rise > 400.0 || heat > 0.2)) {
          var kelvin = radiatingKelvin(T_AMB + rise, heat);
          if (R.size.w > 0.0) { kelvin = max(T_AMB, kelvin + subgridKelvin(wm, uvm, rot)); }
          let glow = planck(kelvin);
          B = (1.0 - albedo) * glow;
          S += albedo * flameSelfLit(uvm, fireE, rd, g);
        }
        // Integrated exactly over the step for constant properties (Hillaire 2015).
        let Ts = exp(-sig * ch * ds);
        col += T * (1.0 - Ts) * (S + B);
        T *= Ts;
        if (max(T.x, max(T.y, T.z)) < 0.004) { break; }
      }
    }
  }

  if (rh.t < min(tFloor, tBall) && rh.t < 1e8 && max(T.x, max(T.y, T.z)) > 0.004) {
    // The reference set.
    let ph = ro + rd * rh.t;
    let v = -rd;
    if (rh.what == 1) {
      // The chrome ball: the Lookdev lab's chromium, as a mirror (its roughness, 0.02, left out), its reflectance by
      // chromeFresnel. The lights a mirror shows as points it shows as highlights (the sun, unless an HDR map's
      // backdrop shows it already; the rig's softboxes, broader); everything else as its reflected ray finds it
      // (mirrorView), the fire included.
      let F = chromeFresnel(max(dot(rh.n, v), 0.0));
      let lb = textureSampleLevel(light, samp, uvwOf(ph + rh.n * 1.5 * P.grid.x), 0.0);
      let rig = envOn() && R.env.w <= 0.0;
      var hl = vec3<f32>(0.0);
      if (!envOn() || rig) {
        hl += R.sunCol.xyz * exp(-lb.x * ch) * ballSun(ph, sun) * refShadow(ph, sun, 1e9, 0.004, 0.0046, 1) * ggxSpec(rh.n, v, sun, select(0.07, 0.3, rig), 1.0);
      }
      if (rimOn()) {
        let lb2 = textureSampleLevel(light2, samp, uvwOf(ph + rh.n * 1.5 * P.grid.x), 0.0).x;
        hl += R.rimCol.xyz * exp(-lb2 * ch) * ballSun(ph, rimDir) * refShadow(ph, rimDir, 1e9, 0.01, 0.15, 1) * ggxSpec(rh.n, v, rimDir, 0.3, 1.0);
      }
      col += T * F * (hl + mirrorView(ph + rh.n * 1e-3, reflect(rd, rh.n), jitter, ch, fireColour));
    } else {
      col += T * shadeSurface(ph, rh.n, rh.n, v, refMat(rh.what), ch, fireColour, false, min(rh.what, 4));
    }
  } else if (tBall < tFloor && tBall < 1e8 && max(T.x, max(T.y, T.z)) > 0.004) {
    col += T * ballSurface(ro + rd * tBall, rd, ch, fireColour);
  } else if (tFloor < 1e8 && max(T.x, max(T.y, T.z)) > 0.004) {
    col += T * floorLight(ro + rd * tFloor, nFloor, rd, tFloor, ch, fireColour);
  } else if (skyStage && tBall >= 1e8 && rh.t >= 1e8) {
    // The sky: a dome of the sky's radiance (its illuminance on the ground over pi), a little brighter toward the
    // horizon (at night too dim to see); or the HDR environment.
    col += T * select(R.skyCol.xyz * INV_PI * mix(1.25, 0.75, sqrt(max(rd.y, 0.0))), envBackdrop(rd), envOn());
  }

  // Exposed (as a fraction of the luminance the camera maps to white), for the accumulation pass.
  return vec4<f32>(col * R.camUp.w, 1.0);
}
`

// 9b. Sums over the grid, each step, per workgroup and then across workgroups: the fire's light, for the smoke and
// the floor (its total emitted intensity, from soot too thin to hide itself, and its brightness-weighted centre, lit
// as a point light from there); the fire bed's area (its mask's integral, which spreads the fuel); and the heat
// release rate.
export const FIRE_SUM = /* wgsl */ `
${RENDER_LIB}
@group(0) @binding(40) var<storage, read_write> partial: array<vec4<f32>>;
@group(0) @binding(41) var<storage, read> hrr: array<f32>;
@group(0) @binding(42) var emitOut: texture_storage_3d<rgba16float, write>; // each cell's glow and flame light per metre as drawn (k cd/m2 per m), and its extinction (1/m): FLAME_NEAR
var<workgroup> sumE: array<vec4<f32>, 64>;
var<workgroup> sumP: array<vec4<f32>, 64>;
var<workgroup> sumB: array<vec4<f32>, 64>;
var<workgroup> sumM: array<vec4<f32>, 64>; // the slices' spread: x^2, y^2, z^2 and xz, light-weighted
var<workgroup> sumX: array<vec4<f32>, 384>; // this step's quarter of the three escape tables, six per cell

// Each cell's glow as the image pass draws it (gasAt): the mean of eight points, one in each eighth of the cell, of its
// interpolated soot and temperature, its detail and its subgrid flame temperature, and the flame's own light. A
// blackbody's glow climbs so steeply with temperature that the cells' own values give several times more: the hottest
// cells, at the bed, are only a sliver of the interpolated field (for the campfire about four times more light than the
// flames show; the path-traced reference found it). Cells with no soot, or too cool to glow and not burning (under
// 600 K, below 0.05 MW/m3), are skipped: most of the smoke. The camera's meter reads the same sums.
const OCTANTS = array<vec3<f32>, 8>(
  vec3<f32>(-0.25, -0.25, -0.25), vec3<f32>(0.25, -0.25, -0.25), vec3<f32>(-0.25, 0.25, -0.25), vec3<f32>(0.25, 0.25, -0.25),
  vec3<f32>(-0.25, -0.25, 0.25), vec3<f32>(0.25, -0.25, 0.25), vec3<f32>(-0.25, 0.25, 0.25), vec3<f32>(0.25, 0.25, 0.25));
// The escape table: each glowing cell's light marched a bed's radius toward each of the table's directions (a cell a
// step), through the soot as the image draws it (formed by now), weighted by the cell's light. (Trilinear rather than
// the image's tricubic, at an eighth of the cost.) A cell with under ESCAPE_SKIP of its slice's light (at the last
// step) is left out: the glow climbs so steeply with temperature that most glowing cells carry almost none of it.
const ESCAPE_SKIP = 1e-5; // (of the slice's light)
fn sootSig(w: vec3<f32>) -> f32 {
  let uvw = uvwOf(w);
  let s4 = textureSampleLevel(scal, samp, uvw, 0.0);
  let heat = textureSampleLevel(flameTex, samp, uvw, 0.0).x;
  return sigmaOf(s4) * select(1.0, sootFormed(gasOf(s4), heat), heat > 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  // (The bindings the glow reads, all of them, so the layout is the engine's.)
  _ = textureDimensions(upresTex);
  _ = textureDimensions(velTex);
  let c = vec3<i32>(id);
  let vol = P.grid.x * P.grid.x * P.grid.x;
  var e = vec3<f32>(0.0);
  var sg = 0.0; // (the cell's extinction, for FLAME_NEAR)
  var b = vec4<f32>(0.0);
  if (all(c < nCells()) && P.fire.x > 0.5) {
    let s = textureLoad(scal, c, 0);
    sg = sigmaOf(s);
    let g = gasOf(s);
    let heatMW = hrr[cellIndex(c)] * 1e-6;
    if ((s.x > 0.0 || heatMW > 0.0) && (g.rise > 300.0 || heatMW > 0.05)) {
      let w = worldOf(vec3<f32>(id) + 0.5);
      let ch = chroma();
      for (var k = 0; k < 8; k++) {
        let m = gasAt(w + OCTANTS[k] * P.grid.x, 0.0, true);
        e += m.sig * ch * m.glow + m.chem;
        if (k == 0) { sg = 0.0; }
        sg += 0.125 * m.sig;
      }
      e *= 0.125;
    }
    let bed = bedMask(worldOf(vec3<f32>(id) + 0.5)) * P.grid.x * P.grid.x;
    b = vec4<f32>(bed, hrr[cellIndex(c)] * vol, 0.0, 0.0);
  }
  let lum = ap1Lum(e);
  if (all(c < nCells())) { textureStore(emitOut, c, vec4<f32>(e * 1e-3, sg)); }
  // This step's share of the table, k = ESCAPE_PHASES j + phase: j 0..3, then 4..6 and the light of the cells marched.
  let phase = P.dims.w % ESCAPE_PHASES;
  var x0 = vec4<f32>(0.0);
  var x1 = vec4<f32>(0.0);
  var x2 = vec4<f32>(0.0);
  var x3 = vec4<f32>(0.0);
  var x4 = vec4<f32>(0.0);
  var x5 = vec4<f32>(0.0);
  let slice = fireLight[3u + 2u * (wid.y * FIRE_BANDS / nw.y)].w;
  if (lum > 0.0 && lum >= ESCAPE_SKIP * slice) {
    let w = worldOf(vec3<f32>(id) + 0.5);
    let rad = max(P.src.w, 0.1);
    let ds = P.grid.x;
    // (Less the share the soot scatters onward, by the similarity relation: its albedo times the phase's anisotropy,
    // R.sunDir.w; bandOut has the rest.)
    let ch = chroma() * (1.0 - R.skyCol.w * R.sunDir.w);
    for (var j = 0u; j < 7u; j++) {
      let k = ESCAPE_PHASES * j + phase;
      if (k >= ESCAPE_DIRS) { break; }
      let dir = escapeDir(k);
      let reach = min(rad, max(boxHit(w, dir, P.origin.xyz, domainHi()).y, 0.0));
      var od = 0.0;
      for (var i = 0; i < 64; i++) {
        let t = (f32(i) + 0.5) * ds;
        if (t > reach) { break; }
        od += sootSig(w + dir * t);
      }
      // (And the share of what the soot stops on the way that gets out from where it stopped, onward along the same
      // line: tau e^-tau / (1 - e^-tau).)
      // (Colour by colour, as the soot reddens it, then its luminance: a fire's light is mostly red, which gets out
      // more easily than green.)
      let tau = od * ds * ch;
      let tr = exp(-tau);
      let v = ap1Lum(e * tr);
      let vs = ap1Lum(e * select(tau * tr / max(vec3<f32>(1.0) - tr, vec3<f32>(1e-6)), vec3<f32>(1.0) - 0.5 * tau, tau < vec3<f32>(1e-3)));
      // (And by its height over the floor, for the floor: bandOutFloor.)
      let h = v * max(w.y, 0.0);
      if (j < 4u) { x0[j] = v; x2[j] = vs; x4[j] = h; } else { x1[j - 4u] = v; x3[j - 4u] = vs; x5[j - 4u] = h; }
    }
    x1.w = lum;
    x5.w = lum * max(w.y, 0.0);
  }
  sumE[li] = vec4<f32>(e, lum);
  let wc = worldOf(vec3<f32>(id) + 0.5);
  sumP[li] = vec4<f32>(wc * lum, 0.0);
  sumB[li] = b;
  sumM[li] = vec4<f32>(wc.x * wc.x, wc.y * wc.y, wc.z * wc.z, wc.x * wc.z) * lum;
  sumX[6u * li] = x0;
  sumX[6u * li + 1u] = x1;
  sumX[6u * li + 2u] = x2;
  sumX[6u * li + 3u] = x3;
  sumX[6u * li + 4u] = x4;
  sumX[6u * li + 5u] = x5;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st = st >> 1u) {
    if (li < st) {
      sumE[li] += sumE[li + st];
      sumP[li] += sumP[li + st];
      sumB[li] += sumB[li + st];
      sumM[li] += sumM[li + st];
      for (var j = 0u; j < 6u; j++) { sumX[6u * li + j] += sumX[6u * (li + st) + j]; }
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let k = wid.x + nw.x * (wid.y + nw.y * wid.z);
    partial[10u * k] = sumE[0];
    partial[10u * k + 1u] = vec4<f32>(sumP[0].xyz, f32(phase)); // (the phase, for the totals)
    partial[10u * k + 2u] = sumB[0];
    for (var j = 0u; j < 6u; j++) { partial[10u * k + 3u + j] = sumX[j]; }
    partial[10u * k + 9u] = sumM[0];
  }
}
`

// 12. The path-traced reference (paused, the Path trace button): the same scene and the same gas, its light
// carried exactly, one path per pixel each frame, averaged. It checks the image pass's shortcuts: multiple scattering
// as octaves, the fire's light as eight point lights through a half-resolution light volume, the sky's through the
// dome's mean transmittance from five directions, the floor's light into the smoke as the floor's straight below, the
// environment as nine spherical harmonics. Here the flames themselves are the fire's light, reached by the paths that
// find them.
//
//   the gas       the image pass's own: its tricubic density, detail and subgrid flame temperature, the soot formed by
//                 now, the flame's own blue light, the burning ball's flames, at one moment of the shutter per path
//                 (the motion blur); each march step's gas held constant over the step, as the image pass holds it
//   free flight   step by step (half a cell): in each, the soot's glow, then a collision at the step's chance of
//                 one (by the most opaque colour; the others weighted), a scattering by the albedo; so the medium is
//                 the image pass's piecewise-constant one, with no bias against it
//   scattering    Henyey-Greenstein at the lab's anisotropy, sampled exactly
//   lights        the sun's disc (0.53 degrees), an HDR map's key in the cone it was taken from, the rig's softboxes
//                 as cones of their solid angles: each sampled at every scattering and surface (next-event estimation,
//                 shadowed through the gas by its optical depth, and by the balls, the set and the cyc), and seen as
//                 such by camera rays and the chrome ball
//   the flames    sampled too, at every scattering and surface: a point in one of the eight slices of the fire's
//                 height, the slice chosen by its light (the image pass's sums), mostly in an upright cylinder about
//                 its brightness-weighted centre and otherwise anywhere across the slice (the hot plume spreads wider);
//                 its glow and the flame's blue toward the point sampled from, through the gas between, over the
//                 point's probability. A path that leaves such a point then counts only the glow that sampling cannot
//                 reach (within two cells of the point, where a sampled point would bring light without limit), so
//                 nothing is counted twice and nothing missed
//   sky           what a path that leaves finds: under the open sky the sky as drawn, in the studio or on the cyc the
//                 flat dome the image pass lights by (unseen by the camera), with a map the map without its key
//   surfaces      the floor and cyc Lambertian; the balls, the chart and the stands as the image pass's OpenPBR (a
//                 Lambertian base under a GGX layer, ash as part of the base), the chrome ball a mirror
//   paths         Russian roulette after the fourth event, capped at maxBounce
export const PATH_TRACE = /* wgsl */ `
${RENDER_LIB}
struct PtParams {
  frame: u32,     // seeds this frame's paths
  n: u32,         // paths averaged so far
  maxBounce: u32,
  parts: u32,     // (development: light left out, to split the reference's by where it comes from; engine.ts ptParts)
  l0d: vec4<f32>, // three cone lights: toward each, the cosine of its half-angle (0: off)
  l0e: vec4<f32>, // and its illuminance at normal incidence (lux, by colour)
  l1d: vec4<f32>,
  l1e: vec4<f32>,
  l2d: vec4<f32>,
  l2e: vec4<f32>,
}
@group(0) @binding(36) var ptPrev: texture_2d<f32>;
@group(0) @binding(37) var<uniform> Q: PtParams;
@group(0) @binding(38) var envRes: texture_2d<f32>; // the HDR map without its key (environment.ts)

var<private> rng: u32;
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand() -> f32 {
  rng = pcg(rng);
  return f32(rng >> 8u) * (1.0 / 16777216.0);
}
fn around(axis: vec3<f32>, c: f32) -> vec3<f32> {
  let s = sqrt(max(1.0 - c * c, 0.0));
  let phi = TWO_PI * rand();
  let F = tangentFrame(axis);
  return normalize(F[0] * (s * cos(phi)) + F[1] * (s * sin(phi)) + axis * c);
}
fn cosineDir(n: vec3<f32>) -> vec3<f32> { return around(n, sqrt(rand())); }
// The direction scattered from travel along d (forward for g > 0).
fn hgSample(d: vec3<f32>, g: f32) -> vec3<f32> {
  let u = rand();
  var c = 1.0 - 2.0 * u;
  if (abs(g) > 1e-3) {
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
    c = clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
  }
  return around(d, c);
}

struct Cone { d: vec3<f32>, c: f32, e: vec3<f32> }
fn coneOf(i: i32) -> Cone {
  if (i == 0) { return Cone(Q.l0d.xyz, Q.l0d.w, Q.l0e.xyz); }
  if (i == 1) { return Cone(Q.l1d.xyz, Q.l1d.w, Q.l1e.xyz); }
  return Cone(Q.l2d.xyz, Q.l2d.w, Q.l2e.xyz);
}

// How much of the light from dist along l reaches p: none past a surface (the floor, the cyc, the balls, the set);
// through the gas, its transmittance (the optical depth marched a cell at a step). shadowT: from a light far off.
fn transTo(p: vec3<f32>, l: vec3<f32>, dist: f32, tau: f32) -> vec3<f32> {
  if (l.y < 0.0 && p.y > 0.0 && -p.y / l.y < dist) { return vec3<f32>(0.0); }
  if (ballHit(p, l) < dist || refHit(p, l).t < dist) { return vec3<f32>(0.0); }
  if (R.scene.x > 1.5 && cycHit(p, l).x < dist) { return vec3<f32>(0.0); }
  let box = boxHit(p, l, P.origin.xyz, domainHi());
  let tEnd = min(box.y, dist);
  if (tEnd <= max(box.x, 0.0)) { return vec3<f32>(1.0); }
  let ds = P.grid.x;
  var od = 0.0;
  var t = max(box.x, 0.0) + rand() * ds;
  for (var i = 0; i < 1024; i++) {
    if (t > tEnd || od > 40.0) { break; }
    let w = p + l * t;
    if (textureSampleLevel(light, samp, uvwOf(w), 0.0).z < 1e-4) {
      t += 2.0 * ds;
      continue;
    }
    od += gasAt(w, tau, false).sig * ds;
    t += ds;
  }
  return exp(-od * chroma());
}
fn shadowT(p: vec3<f32>, l: vec3<f32>, tau: f32) -> vec3<f32> { return transTo(p, l, 1e9, tau); }

// The flames as a light (the header): the slices' light, the cylinders, a point and its probability (per m3).
var<private> fireTot: f32;
const FIRE_WIDE = 0.15; // the share of the points drawn anywhere across the slice
fn fireSlab() -> f32 { return f32(P.dims.y) * P.grid.x / f32(FIRE_BANDS); }
fn fireRadius() -> f32 { return max(1.5 * P.src.w, 0.3); }
fn firePdf(x: vec3<f32>) -> f32 {
  if (fireTot <= 0.0) { return 0.0; }
  let h = fireSlab();
  let b = floor(x.y / h);
  let lo = P.origin.xyz;
  let hi = domainHi();
  if (b < 0.0 || b >= f32(FIRE_BANDS) || any(x.xz < lo.xz) || any(x.xz > hi.xz)) { return 0.0; }
  let bi = u32(b);
  let lum = fireLight[3u + 2u * bi].w;
  // (A slice with no light has no centre.)
  if (!(lum > 0.0)) { return 0.0; }
  let r = fireRadius();
  var pdf = FIRE_WIDE / ((hi.x - lo.x) * (hi.z - lo.z) * h);
  if (distance(x.xz, fireLight[4u + 2u * bi].xz) <= r) { pdf += (1.0 - FIRE_WIDE) / (PI * r * r * h); }
  return lum / fireTot * pdf;
}
fn fireSample() -> vec3<f32> {
  var u = rand() * fireTot;
  var bi = 0u;
  for (var b = 0u; b < FIRE_BANDS; b++) {
    let lum = max(fireLight[3u + 2u * b].w, 0.0);
    if (lum <= 0.0) { continue; }
    bi = b;
    if (u < lum) { break; }
    u -= lum;
  }
  let y = (f32(bi) + rand()) * fireSlab();
  if (rand() < FIRE_WIDE) {
    let lo = P.origin.xyz;
    let hi = domainHi();
    return vec3<f32>(mix(lo.x, hi.x, rand()), y, mix(lo.z, hi.z, rand()));
  }
  let rr = fireRadius() * sqrt(rand());
  let a = TWO_PI * rand();
  let c = fireLight[4u + 2u * bi].xyz;
  return vec3<f32>(c.x + rr * cos(a), y, c.z + rr * sin(a));
}
// From p: the direction to a sampled point of the flames, and the light it brings, over its probability (the glow per
// metre toward p, through the gas between, over the distance squared). Most of a cylinder holds little glow and its
// sheets a great deal, so the point is chosen from FIRE_CANDIDATES drawn as above, in proportion to the light each
// would bring unshadowed, as the scattering (the phase toward rd, g) or the surface (its cosine about n) takes it
// (resampled importance sampling; Talbot et al. 2005): unbiased, with the weight the average of the candidates' over
// the chosen one's, which keeps the bright sheets from making speckles. (The caller applies the phase or the surface's
// reflectance; the chosen point's share of it is divided out here.) Glow within FIRE_NEAR cells
// of the point sampled from is left to the path's next segment instead (its march counts it, smoothly): sampled, a
// point that near would bring light without limit, as one over its distance squared.
const FIRE_CANDIDATES = 8u; // candidates drawn each time the flames are sampled
const FIRE_NEAR = 2.0; // cells about the point sampled from
struct FireLight { l: vec3<f32>, L: vec3<f32> }
fn fireToward(p: vec3<f32>, tau: f32, ch: vec3<f32>, surface: bool, dir: vec3<f32>, g: f32) -> FireLight {
  var f = FireLight(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0));
  if (fireTot <= 0.0) { return f; }
  var sum = 0.0;
  var x = vec3<f32>(0.0);
  var S = vec3<f32>(0.0);
  var chosen = 0.0;
  for (var i = 0u; i < FIRE_CANDIDATES; i++) {
    let xi = fireSample();
    let pdf = firePdf(xi);
    if (pdf <= 0.0) { continue; }
    let m = gasAt(xi, tau, true);
    let Si = m.sig * ch * m.glow + m.chem;
    let d = xi - p;
    let dd = dot(d, d);
    if (dd < sq(FIRE_NEAR * P.grid.x)) { continue; }
    let li = d / sqrt(dd);
    let ti = ap1Lum(Si) / dd * select(hg(dot(dir, li), g), max(dot(dir, li), 0.0), surface);
    let wi = ti / pdf;
    if (wi <= 0.0) { continue; }
    sum += wi;
    if (rand() * sum < wi) {
      x = xi;
      S = Si;
      chosen = ti;
    }
  }
  if (sum <= 0.0 || chosen <= 0.0) { return f; }
  let d = x - p;
  let d2 = dot(d, d);
  let dist = sqrt(d2);
  f.l = d / dist;
  f.L = S / d2 / chosen * (sum / f32(FIRE_CANDIDATES)) * transTo(p, f.l, dist, tau);
  return f;
}

// What a path that leaves the scene finds. seen: a camera ray, or one off the chrome ball.
fn leaving(d: vec3<f32>, seen: bool) -> vec3<f32> {
  if (!seen && (Q.parts & 4u) != 0u) { return vec3<f32>(0.0); } // (the sky left out)
  var L = vec3<f32>(0.0);
  if (envOn()) {
    // A map (the rig's studio has none): seen whole, lit by without its key (a light of its own).
    if (R.env.w > 0.0) {
      if (seen) {
        L = envBackdrop(d);
      } else {
        let e = toEnv(d);
        let uv = vec2<f32>(fract(atan2(e.x, -e.z) * INV_2PI + 0.5), acos(clamp(e.y, -1.0, 1.0)) * INV_PI);
        L = textureSampleLevel(envRes, noiseSamp, uv, 0.0).xyz * R.env.w;
      }
    }
  } else if (R.scene.x > 0.5 && R.scene.x < 1.5) {
    L = R.skyCol.xyz * INV_PI * mix(1.25, 0.75, sqrt(max(d.y, 0.0)));
  } else if (!seen && d.y > 0.0) {
    L = R.skyCol.xyz * INV_PI;
  }
  // The lights, where seen: their radiance, illuminance over solid angle. (Left out with the rest of the lights in the
  // parts' split, 8: seen in the chrome ball, the sun's disc stayed in the fire's light alone, one path in a hundred
  // finding it, a single pixel worth two thirds of the fire's light about the ball.)
  if (seen && (Q.parts & 8u) == 0u) {
    for (var i = 0; i < 3; i++) {
      let c = coneOf(i);
      if (c.c > 0.0 && dot(d, c.d) > c.c) { L += c.e / (TWO_PI * (1.0 - c.c)); }
    }
  }
  return L;
}

// A surface met: the floor or the cyc, the collision ball, the set (its chrome ball a mirror).
struct Surf { n: vec3<f32>, ng: vec3<f32>, m: BallMat, glow: vec3<f32>, mirror: bool }
fn surfaceAt(p: vec3<f32>, kind: i32, nGeo: vec3<f32>, what: i32) -> Surf {
  var s = Surf(nGeo, nGeo, BallMat(vec3<f32>(0.0), 0.0, 1.0, 0.04, 0.0, vec3<f32>(0.0), 0.5), vec3<f32>(0.0), false);
  if (kind == 0) {
    let fade = select(exp(-dot(p.xz, p.xz) / (R.floorCol.w * R.floorCol.w)), 1.0, R.scene.x > 0.5);
    s.m.base = R.floorCol.xyz * fade;
  } else if (kind == 1) {
    s.m = BallMat(vec3<f32>(BALL_ALBEDO), 0.0, 1.0, 0.04, 0.0, vec3<f32>(0.0), 0.5);
    if (ballBurns()) {
      let lp = (p - P.ball.xyz) / P.ball.w;
      let sk = skinAt(nGeo);
      let crack = charCracks(lp, sk);
      s.m = ballMat(lp, sk, crack);
      s.n = crackNormal(lp, nGeo, sk, crack);
      s.glow = charGlow(sk, crack);
    }
  } else {
    s.mirror = what == 1;
    s.m = refMat(what);
  }
  return s;
}
// The surface's reflectance toward v of light from l, times the cosine: the base (what the specular layer passes,
// the fuzz folded into it) over pi, and the specular lobe.
fn surfF(s: Surf, v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
  let nl = dot(s.n, l);
  if (nl <= 0.0 || dot(s.ng, l) <= 0.0) { return vec3<f32>(0.0); }
  let nv = max(dot(s.n, v), 1e-4);
  let eFuzz = select(0.0, fuzzDirAlbedo(nv, s.m.fuzzRough), s.m.fuzzW > 0.0);
  let base = (s.m.base * (1.0 - s.m.fuzzW * eFuzz) + s.m.fuzzW * eFuzz * s.m.fuzzColor) * (1.0 - s.m.specW * envBRDF(s.m.f0, s.m.rough, nv));
  return base * INV_PI * nl + s.m.specW * ggxSpec(s.n, v, l, s.m.rough, s.m.f0);
}

@fragment
fn pt(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let px = vec2<i32>(fc.xy);
  rng = pcg(u32(px.x) + u32(px.y) * 8192u + pcg(Q.frame));
  // (The image pass's bindings, in full, so the two share their resources.)
  _ = fireLight[0];
  _ = textureNumLayers(blueTex);
  _ = textureDimensions(light2);
  // A ray through a random point of the pixel, at a random moment of the shutter.
  let ndc = vec2<f32>((floor(fc.x) + rand()) / R.size.x * 2.0 - 1.0, 1.0 - (floor(fc.y) + rand()) / R.size.y * 2.0);
  let tanY = R.camPos.w;
  var rd = normalize(R.camFwd.xyz + R.camRight.xyz * ndc.x * tanY * R.camFwd.w + R.camUp.xyz * ndc.y * tanY);
  var ro = R.camPos.xyz;
  let tau = (rand() - 0.5) * R.camera.x;
  let ch = chroma();
  let g = R.sunDir.w;
  let albedo = R.skyCol.w;
  var L = vec3<f32>(0.0);
  var beta = vec3<f32>(1.0);
  var seen = true;
  var events = 0u;
  // The fire's light in the slices (0 for a smoke source), and whether this segment leaves a vertex where the flames
  // were sampled (its glow inside the cylinders then already counted).
  fireTot = 0.0;
  if (R.fire.x > 0.5) {
    for (var b = 0u; b < FIRE_BANDS; b++) { fireTot += max(fireLight[3u + 2u * b].w, 0.0); }
  }
  var sampled = false;
  var lastLeg = false; // (single scattering only: the path goes on only to find the sky)
  var pv = vec3<f32>(0.0); // the vertex the segment leaves
  for (var bounce = 0u; bounce < 64u; bounce++) {
    // The surface ahead: the floor (or the cyc), the collision ball, the set.
    var tS = select(1e9, -ro.y / rd.y, rd.y < -1e-5 && ro.y > 0.0);
    var nS = vec3<f32>(0.0, 1.0, 0.0);
    if (R.scene.x > 1.5) {
      let cy = cycHit(ro, rd);
      tS = cy.x;
      nS = cy.yzw;
    }
    var kind = 0;
    var what = -1;
    let tB = ballHit(ro, rd);
    if (tB < tS) {
      tS = tB;
      nS = normalize(ro + rd * tB - P.ball.xyz);
      kind = 1;
    }
    let rh = refHit(ro, rd);
    if (rh.t < tS) {
      tS = rh.t;
      nS = rh.n;
      kind = 2;
      what = rh.what;
    }
    // The gas up to it: glow, and a collision or not, step by step.
    var scattered = false;
    var dead = false;
    let box = boxHit(ro, rd, P.origin.xyz, domainHi());
    if (box.y > max(box.x, 0.0) && box.x < tS) {
      let ds = 0.5 * P.grid.x;
      let tEnd = min(box.y, tS);
      var t = max(box.x, 0.0) + rand() * ds;
      for (var i = 0; i < 4096; i++) {
        if (t > tEnd) { break; }
        let w = ro + rd * t;
        let toBall = distance(w, P.ball.xyz) - P.ball.w;
        let nearFlames = ballBurns() && toBall < BALL_FLAME_REACH;
        let nearBall = P.ball.w > 0.0 && toBall < 0.15;
        if (textureSampleLevel(light, samp, uvwOf(w), 0.0).z < 1e-4 && !nearFlames && !nearBall) {
          t += 3.0 * ds;
          continue;
        }
        if (nearFlames) {
          let bf = ballFlame(w);
          if (bf.x > 0.0) {
            let tf = exp(-bf.x * ds);
            L += beta * (1.0 - tf) * (1.0 - albedo) * planck(bf.y);
            beta *= tf;
          }
        }
        let m = gasAt(w, tau, true);
        let counted = sampled && firePdf(w) > 0.0 && distance(w, pv) >= FIRE_NEAR * P.grid.x;
        if (!counted) { L += beta * m.chem * ds; }
        if (m.sig > 1e-3) {
          let sc = m.sig * ch;
          let Ts = exp(-sc * ds);
          if (!counted) { L += beta * (1.0 - Ts) * m.glow; }
          let sm = max(sc.x, max(sc.y, sc.z));
          let q = 1.0 - exp(-sm * ds);
          if (rand() < q) {
            if (lastLeg || (Q.parts & 16u) != 0u) { dead = true; break; }
            // A collision, somewhere in the step: scattered (the albedo's share; the rest absorbed, in the weight).
            let tc = t + min(-log(max(1.0 - rand() * q, 1e-7)) / sm, ds);
            beta *= albedo * (1.0 - Ts) / q;
            let p = ro + rd * tc;
            for (var k = 0; k < 3; k++) {
              let c = coneOf(k);
              if (c.c <= 0.0 || (Q.parts & 8u) != 0u) { continue; }
              let l = around(c.d, 1.0 - rand() * (1.0 - c.c));
              L += beta * c.e * shadowT(p, l, tau) * hg(dot(rd, l), g) * INV_4PI;
            }
            let fl = fireToward(p, tau, ch, false, rd, g);
            L += beta * fl.L * hg(dot(rd, fl.l), g) * INV_4PI;
            sampled = fireTot > 0.0;
            pv = p;
            ro = p;
            rd = hgSample(rd, g);
            scattered = true;
            break;
          }
          beta *= Ts / (1.0 - q);
        }
        t += ds;
      }
    }
    // (Development, Q.parts: 1 no light off the floor but the camera's own sight of it, 2 single scattering in the gas,
    // 4 no sky, 8 no lights, 16 no scattering in the gas: its glow, dimmed.)
    if (dead) { break; }
    if (scattered && (Q.parts & 2u) != 0u) { lastLeg = true; }
    if (!scattered) {
      if (kind == 0 && tS < 1e8 && !seen && (Q.parts & 1u) != 0u) { break; }
      if (lastLeg && tS < 1e8) { break; }
      if (tS >= 1e8) {
        L += beta * leaving(rd, seen);
        break;
      }
      let p = ro + rd * tS;
      // Under the open sky the far floor fades into the horizon's haze, as the image pass draws it.
      if (kind == 0 && R.scene.x > 0.5 && R.scene.x < 1.5 && rand() > exp(-tS / 40.0)) {
        L += beta * select(R.skyCol.xyz * INV_PI * 1.25, envBackdrop(normalize(vec3<f32>(rd.x, 0.0, rd.z))), envOn());
        break;
      }
      let s = surfaceAt(p, kind, nS, what);
      let v = -rd;
      if (s.mirror) {
        beta *= chromeFresnel(max(dot(s.n, v), 0.0));
        ro = p + s.ng * 1e-3;
        rd = reflect(rd, s.n);
        sampled = false;
        continue;
      }
      L += beta * s.glow;
      let po = p + s.ng * 1e-3;
      for (var k = 0; k < 3; k++) {
        let c = coneOf(k);
        if (c.c <= 0.0 || (Q.parts & 8u) != 0u) { continue; }
        let l = around(c.d, 1.0 - rand() * (1.0 - c.c));
        L += beta * c.e * shadowT(po, l, tau) * surfF(s, v, l);
      }
      let fl = fireToward(po, tau, ch, true, s.ng, 0.0);
      L += beta * fl.L * surfF(s, v, fl.l);
      sampled = fireTot > 0.0;
      pv = po;
      let l = cosineDir(s.n);
      let nl = dot(s.n, l);
      if (nl <= 0.0 || dot(s.ng, l) <= 0.0) { break; }
      beta *= surfF(s, v, l) * PI / nl;
      ro = po;
      rd = l;
    }
    seen = false;
    events++;
    if (events >= Q.maxBounce) { break; }
    if (events > 3u) {
      let keep = clamp(max(beta.x, max(beta.y, beta.z)), 0.05, 0.95);
      if (rand() > keep) { break; }
      beta /= keep;
    }
  }
  // The running mean, exposed as the image pass writes its frames. (A path that went numerically wrong, which none
  // should, is dropped rather than left to stick in the mean.)
  var sample = L * R.camUp.w;
  if (!all(sample == sample) || any(abs(sample) > vec3<f32>(1e30))) { sample = vec3<f32>(0.0); }
  let prev = select(vec3<f32>(0.0), textureLoad(ptPrev, px, 0).xyz, Q.n > 0u);
  return vec4<f32>(prev + (sample - prev) / f32(Q.n + 1u), 1.0);
}

// (Development check, engine.ts fireFieldCheck: at the centres of the grid's cells in its middle plane along z, one a
// pixel, the fire's light as the image pass has it (its light volume, lux of luminance), as the reference samples the
// flames (64 points, as above, the light from within two cells left out), and the soot's extinction there.)
@fragment
fn fluence(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  rng = pcg(u32(fc.x) + u32(fc.y) * 8192u + 977u);
  _ = fireLight[0];
  _ = textureNumLayers(blueTex);
  _ = textureDimensions(light2);
  _ = textureDimensions(ptPrev);
  _ = textureDimensions(envRes);
  _ = Q.n;
  _ = skin[0];
  _ = ballSheet[0];
  _ = textureDimensions(envTex);
  _ = textureDimensions(velTex);
  let n = vec3<f32>(P.dims.xyz);
  let w = P.origin.xyz + vec3<f32>(floor(fc.x) + 0.5, n.y - floor(fc.y) - 0.5, 0.5 * n.z) * P.grid.x;
  fireTot = 0.0;
  if (R.fire.x > 0.5) {
    for (var b = 0u; b < FIRE_BANDS; b++) { fireTot += max(fireLight[3u + 2u * b].w, 0.0); }
  }
  var acc = 0.0;
  for (var i = 0; i < 64; i++) { acc += ap1Lum(fireToward(w, 0.0, chroma(), false, vec3<f32>(0.0, 1.0, 0.0), 0.0).L); }
  // (And the flames' whole light as the sampling sees it, cd: its glow over the points' probability, averaged.)
  var total = 0.0;
  for (var i = 0; i < 64; i++) {
    let x = fireSample();
    let pdf = firePdf(x);
    if (pdf > 0.0) {
      let m = gasAt(x, 0.0, true);
      total += ap1Lum(m.sig * chroma() * m.glow + m.chem) / pdf;
    }
  }
  return vec4<f32>(acc / 64.0, 1000.0 * textureSampleLevel(light, samp, uvwOf(w), 0.0).w, total / 64.0, 1.0);
}
`

// The wipe: the live frame left of a column, the traced right of it, into the texture the glare and display read.
export const PT_COMPOSITE = /* wgsl */ `
@group(0) @binding(0) var live: texture_2d<f32>;
@group(0) @binding(1) var traced: texture_2d<f32>;
@group(0) @binding(2) var<uniform> W: vec4<f32>; // the wipe's pixel column

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let p = vec2<i32>(fc.xy);
  if (fc.x < W.x) { return textureLoad(live, p, 0); }
  return vec4<f32>(textureLoad(traced, p, 0).xyz, 1.0);
}
`

// 11. Temporal accumulation, then the camera. Each frame's march is blended into a running history, which averages
// away the sampling noise (and, with the motion blur, the shutter's moments); the history is first clamped to the range
// of the current frame's 3 x 3 neighbourhood (found in a luminance-compressed space, so a bright flame does not set it),
// so what moves or flickers is not smeared behind it, and the blend is in linear light. (Blended compressed, a
// flickering flame averaged a sixth darker than its frames do.) Then the lens's glare (GLARE), and DISPLAY: the white
// balance, the view transform (the Lookdev lab's own, viewTransforms.ts: ACES 2.0 by default) and a dither against
// banding.
export const RESOLVE = /* wgsl */ `
${SIM_COMMON}
${RENDER_COMMON}
@group(0) @binding(2) var current: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var<uniform> T: vec4<f32>; // history valid (0 or 1), weight of the current frame

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn squash(c: vec3<f32>) -> vec3<f32> { return c / (1.0 + ap1Lum(c)); }
fn unsquash(c: vec3<f32>) -> vec3<f32> { return c / max(1.0 - ap1Lum(c), 1e-4); }

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let p = vec2<i32>(fc.xy);
  let last = vec2<i32>(textureDimensions(current)) - 1;
  let cur = squash(max(textureLoad(current, p, 0).xyz, vec3<f32>(0.0)));
  var lo = cur;
  var hi = cur;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = squash(max(textureLoad(current, clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), last), 0).xyz, vec3<f32>(0.0)));
      lo = min(lo, q);
      hi = max(hi, q);
    }
  }
  if (T.x > 0.5) {
    let h = clamp(squash(textureLoad(history, p, 0).xyz), lo, hi);
    return vec4<f32>(mix(unsquash(h), unsquash(cur), T.y), 1.0);
  }
  return vec4<f32>(unsquash(cur), 1.0);
}
`

// The lens's glare: a real lens scatters a few percent of the light it passes into a wide, soft halo (veiling glare),
// unseen around most of a scene but plain around anything far brighter than what surrounds it, as flames are at
// dusk. Without a threshold, so it keeps the light's energy: the image is blurred into a chain of GLARE_LEVELS mips,
// each half the size of the last, by the 13-tap downsampling and 3 x 3 tent upsampling of Jimenez (2014), the levels
// summed, and the display mixes that sum (over the levels' count) in at the Glare control's share. The first
// downsample weights its five groups by 1 / (1 + luminance) (Karis's average), so a single blazing pixel cannot flare
// into a blotch.
export const GLARE_LEVELS = 6
export const GLARE = /* wgsl */ `
${AP1_WGSL}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn tap(uv: vec2<f32>) -> vec3<f32> { return textureSampleLevel(src, smp, uv, 0.0).xyz; }
fn karis(c: vec3<f32>) -> f32 { return 1.0 / (1.0 + ap1Lum(c)); }
fn down(fc: vec4<f32>, firstPass: bool) -> vec4<f32> {
  let ts = 1.0 / vec2<f32>(textureDimensions(src));
  let uv = fc.xy * 2.0 * ts;
  let a = tap(uv + ts * vec2<f32>(-2.0, -2.0));
  let b = tap(uv + ts * vec2<f32>(0.0, -2.0));
  let c = tap(uv + ts * vec2<f32>(2.0, -2.0));
  let d = tap(uv + ts * vec2<f32>(-2.0, 0.0));
  let e = tap(uv);
  let f = tap(uv + ts * vec2<f32>(2.0, 0.0));
  let g = tap(uv + ts * vec2<f32>(-2.0, 2.0));
  let h = tap(uv + ts * vec2<f32>(0.0, 2.0));
  let i = tap(uv + ts * vec2<f32>(2.0, 2.0));
  let j = tap(uv + ts * vec2<f32>(-1.0, -1.0));
  let k = tap(uv + ts * vec2<f32>(1.0, -1.0));
  let l = tap(uv + ts * vec2<f32>(-1.0, 1.0));
  let m = tap(uv + ts * vec2<f32>(1.0, 1.0));
  let g0 = (j + k + l + m) * 0.25;
  let g1 = (a + b + d + e) * 0.25;
  let g2 = (b + c + e + f) * 0.25;
  let g3 = (d + e + g + h) * 0.25;
  let g4 = (e + f + h + i) * 0.25;
  if (!firstPass) { return vec4<f32>(g0 * 0.5 + (g1 + g2 + g3 + g4) * 0.125, 1.0); }
  let w0 = 0.5 * karis(g0);
  let w1 = 0.125 * karis(g1);
  let w2 = 0.125 * karis(g2);
  let w3 = 0.125 * karis(g3);
  let w4 = 0.125 * karis(g4);
  return vec4<f32>((g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4), 1.0);
}
@fragment fn fsDownFirst(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> { return down(fc, true); }
@fragment fn fsDown(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> { return down(fc, false); }
// Into a level twice this one's size, added to what is there (the pipeline blends by adding).
@fragment fn fsUp(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let ts = 1.0 / vec2<f32>(textureDimensions(src));
  let uv = fc.xy * 0.5 * ts;
  var c = tap(uv) * 4.0;
  c += (tap(uv + ts * vec2<f32>(1.0, 0.0)) + tap(uv - ts * vec2<f32>(1.0, 0.0)) + tap(uv + ts * vec2<f32>(0.0, 1.0)) + tap(uv - ts * vec2<f32>(0.0, 1.0))) * 2.0;
  c += tap(uv + ts) + tap(uv - ts) + tap(uv + ts * vec2<f32>(1.0, -1.0)) + tap(uv + ts * vec2<f32>(-1.0, 1.0));
  return vec4<f32>(c / 16.0, 1.0);
}
`

// The camera's last step: the glare mixed in, the white balance, the view transform, a dither against banding.
export const DISPLAY = /* wgsl */ `
${SIM_COMMON}
${RENDER_COMMON}
${VIEW_TRANSFORMS_WGSL}
@group(0) @binding(2) var image: texture_2d<f32>; // the accumulated history
@group(0) @binding(3) var glareTex: texture_2d<f32>; // the glare's levels summed (GLARE), at half size
@group(0) @binding(4) var smp: sampler;
@group(0) @binding(5) var lut: texture_2d<f32>; // the ACES 2.0 LUT: 65 slices of 65 x 65 stacked (blue by slice)
@group(0) @binding(6) var lutSampler: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn ign(p: vec2<f32>) -> f32 { return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715)))); }
// As the Lookdev lab shows its image: ACES 2.0 from the LUT (AgX until it has loaded), or AgX, PBR Neutral or the
// plain sRGB curve on the image carried into Rec.709.
fn view(c: vec3<f32>) -> vec3<f32> {
  let v = u32(R.scene.z + 0.5);
  if (v == 0u && R.scene.w > 0.5) { return aces2(c); }
  var r = max(AP1_TO_REC709 * c, vec3<f32>(0.0));
  if (v == 2u) { r = PBRNeutralToneMapping(r); }
  else if (v != 3u) { r = AgXToneMapping(r); }
  return srgbOETF(r);
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  var c = textureLoad(image, vec2<i32>(fc.xy), 0).xyz;
  let k = R.camera.y;
  if (k > 0.0) {
    let glare = textureSampleLevel(glareTex, smp, fc.xy / vec2<f32>(textureDimensions(image)), 0.0).xyz / max(R.camera.z, 1.0);
    c = c * (1.0 - k) + glare * k;
  }
  let balanced = vec3<f32>(dot(R.wb0.xyz, c), dot(R.wb1.xyz, c), dot(R.wb2.xyz, c));
  return vec4<f32>(view(max(balanced, vec3<f32>(0.0))) + (ign(fc.xy + 7.13) - 0.5) / 255.0, 1.0);
}
`

// 12. Wavelet-turbulence upres (after Kim, Thuerey, James and Gross 2008), for a smoke source: the smoke's density again
// on a grid twice as fine, carried by the solver's velocity plus turbulence too fine for the coarse grid. That
// turbulence is the curl of a noise (divergence free), read at the texture coordinates the detail layers carry with the
// flow, so its eddies travel with the smoke and fade where the flow has stretched them, and is as strong as the local
// flow, as eddies a band finer carry energy in proportion to the resolved flow's. Semi-Lagrangian: at half the cell
// size it smears half as much. The renderer reads this density; the lighting still comes from the coarse grid.
//
// Most of the fine grid holds no smoke (a plume fills about a third of it, smoke in wind two fifths), so each block of
// UPRES_BLOCK fine cells a side is marked when it holds any, and a cell whose backtrace, wherever the turbulence can
// take it, reads only blocks that held none last step takes no density from them: it skips the turbulence and the
// lookups, with the same result. The turbulence is read from CURL_GEN's texture, whose largest magnitude bounds how
// far it can move the backtrace.
export const UPRES_BLOCK = 8
const NOISE_ROT = 'mat3x3<f32>(0.8, -0.6, 0.0, 0.36, 0.48, 0.8, -0.48, -0.64, 0.6)'

// The upres pass's turbulence, made once with the detail noise: the curl of the noise's three channels as a vector
// potential, by central differences half a texel apart, scaled to about unit size (the fbm's gradients run about 20 per
// tile), at each texel of a texture the noise's size, in the noise's rotated frame. Read back with one lookup, it is
// the field the upres pass once differenced from six (between texels, interpolated rather than recomputed). And its
// largest magnitude anywhere, for the upres pass's bound on how far the turbulence can move a backtrace.
export const CURL_GEN = /* wgsl */ `
@group(0) @binding(0) var noiseTex: texture_3d<f32>;
@group(0) @binding(1) var noiseSamp: sampler;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var<storage, read_write> curlMax: array<atomic<u32>>;
const ROT = ${NOISE_ROT};

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = textureDimensions(dst);
  if (any(id >= n)) { return; }
  let q = (vec3<f32>(id) + 0.5) / vec3<f32>(n);
  let h = 0.5 / 64.0;
  let ex = ROT * vec3<f32>(h, 0.0, 0.0);
  let ey = ROT * vec3<f32>(0.0, h, 0.0);
  let ez = ROT * vec3<f32>(0.0, 0.0, h);
  let dx = textureSampleLevel(noiseTex, noiseSamp, q + ex, 0.0).xyz - textureSampleLevel(noiseTex, noiseSamp, q - ex, 0.0).xyz;
  let dy = textureSampleLevel(noiseTex, noiseSamp, q + ey, 0.0).xyz - textureSampleLevel(noiseTex, noiseSamp, q - ey, 0.0).xyz;
  let dz = textureSampleLevel(noiseTex, noiseSamp, q + ez, 0.0).xyz - textureSampleLevel(noiseTex, noiseSamp, q - ez, 0.0).xyz;
  let c = vec3<f32>(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) / (2.0 * h * 20.0);
  textureStore(dst, id, vec4<f32>(c, 0.0));
  // (A positive float's bits order as the float does.)
  atomicMax(&curlMax[0], bitcast<u32>(length(c)));
}
`

export const UPRES = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var src: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var detA: texture_3d<f32>;
@group(0) @binding(6) var detB: texture_3d<f32>;
@group(0) @binding(7) var curlTex: texture_3d<f32>;
@group(0) @binding(8) var noiseSamp: sampler;
@group(0) @binding(9) var scal: texture_3d<f32>;
@group(0) @binding(11) var<storage, read> curlMax: array<u32>;
// The blocks that held smoke last step (read), and this step's (marked).
@group(0) @binding(12) var<storage, read> heldPrev: array<u32>;
@group(0) @binding(13) var<storage, read_write> heldNext: array<atomic<u32>>;
var<workgroup> wgHolds: atomic<u32>;
struct Upres {
  strength: f32, // the turbulence's speed per unit of the local flow's
  k: f32, // 1/m: noise tiles per metre
  wa: f32, // the detail layers' cross-fade weights
  wb: f32,
  init: f32, // 1: fill from the coarse density instead (when the upres starts)
  p0: f32,
  p1: f32,
  p2: f32,
}
@group(0) @binding(10) var<uniform> U: Upres;

const ROT = ${NOISE_ROT};
const BLOCK = ${UPRES_BLOCK};
fn curlNoise(p: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(curlTex, noiseSamp, ROT * p, 0.0).xyz;
}
fn blocksOf(nf: vec3<i32>) -> vec3<i32> { return (nf + BLOCK - 1) / BLOCK; }
// Whether a sample anywhere from lo to hi (fine cells) can read a texel of a block that held smoke last step: the
// texels a trilinear lookup there touches, clamped as the lookup clamps. A reach over more than three blocks a side
// (a fast flow) is taken as yes rather than searched.
fn mayHold(lo: vec3<f32>, hi: vec3<f32>, nf: vec3<i32>) -> bool {
  let t0 = clamp(vec3<i32>(floor(lo - 0.5)), vec3<i32>(0), nf - 1);
  let t1 = clamp(vec3<i32>(floor(hi - 0.5)) + 1, vec3<i32>(0), nf - 1);
  let b0 = t0 / BLOCK;
  let b1 = t1 / BLOCK;
  if (any(b1 - b0 > vec3<i32>(2))) { return true; }
  let nb = blocksOf(nf);
  for (var z = b0.z; z <= b1.z; z++) {
    for (var y = b0.y; y <= b1.y; y++) {
      for (var x = b0.x; x <= b1.x; x++) {
        if (heldPrev[(z * nb.y + y) * nb.x + x] != 0u) { return true; }
      }
    }
  }
  return false;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let nf = vec3<i32>(textureDimensions(dst));
  let c = vec3<i32>(id);
  if (all(c < nf)) {
    let gF = vec3<f32>(id) + 0.5;
    let g = 0.5 * gF; // coarse grid coordinates
    let w = worldOf(g);
    let n = vec3<f32>(nf);
    var d = 0.0;
    if (U.init > 0.5) {
      d = max(scalAt(scal, samp, g).x, 0.0);
    } else {
      let u = velAt(vel, samp, g);
      // Back along the flow, in fine cells (twice as many per metre): without the turbulence, and how far the
      // turbulence can move it at most (its magnitude is at most curlMax, the two layers' blend at most root 2 times
      // that; 1% over for the texture's half floats).
      let back = 2.0 * P.grid.y / P.grid.x;
      let gb0 = gF - back * u;
      let reach = back * U.strength * length(u) * bitcast<f32>(curlMax[0]) * 1.4142136 * 1.01;
      if (mayHold(gb0 - reach, gb0 + reach, nf)) {
        let uvw = g / nF();
        let a = textureSampleLevel(detA, samp, uvw, 0.0);
        let b = textureSampleLevel(detB, samp, uvw, 0.0);
        let wa = U.wa / (1.0 + a.w * a.w);
        let wb = U.wb / (1.0 + b.w * b.w);
        let turb = (curlNoise((w + a.xyz) * U.k) * wa + curlNoise((w + b.xyz) * U.k + vec3<f32>(0.37, 0.61, 0.13)) * wb) / max(sqrt(U.wa * U.wa + U.wb * U.wb), 1e-3);
        let gb = gb0 - back * U.strength * length(u) * turb;
        if (gb.x >= 0.0 && gb.x <= n.x && gb.z >= 0.0 && gb.z <= n.z && gb.y <= n.y) {
          d = textureSampleLevel(src, samp, (clamp(gb - 0.5, vec3<f32>(0.0), n - 1.0) + 0.5) / n, 0.0).x;
        }
      }
      let top = smoothstep(0.7, 1.0, gF.y / n.y);
      d = max(d, 0.0) * exp(-(P.decay.y + 2.5 * top) * P.grid.y);
      d = max(d, P.srcProps.x * sourceMask(w));
      if (solidAt(P.origin.xyz + gF * 0.5 * P.grid.x)) { d = 0.0; }
    }
    textureStore(dst, c, vec4<f32>(d, 0.0, 0.0, 0.0));
    if (d > 0.0) { atomicStore(&wgHolds, 1u); }
  }
  // A workgroup (4 cells a side, at a multiple of 4) lies in one block: its first cell marks the block once.
  workgroupBarrier();
  if (li == 0u && atomicLoad(&wgHolds) != 0u) {
    let nb = blocksOf(nf);
    let bk = c / BLOCK;
    atomicStore(&heldNext[(bk.z * nb.y + bk.y) * nb.x + bk.x], 1u);
  }
}
`

// The upres grid's density averaged over each coarse cell (its eight fine cells), for the light pass, whose lattice is
// coarser still. (Read in every direction from every point of the lattice, the fine grid's texture, eight times the
// size, took the light pass from 1 ms to 16 on a laptop.)
export const UPRES_COARSE = /* wgsl */ `
@group(0) @binding(0) var fine: texture_3d<f32>;
@group(0) @binding(1) var coarse: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = textureDimensions(coarse);
  if (any(id >= n)) { return; }
  var s = 0.0;
  for (var k = 0u; k < 8u; k++) {
    s += max(textureLoad(fine, 2u * id + vec3<u32>(k & 1u, (k >> 1u) & 1u, k >> 2u), 0).x, 0.0);
  }
  textureStore(coarse, id, vec4<f32>(0.125 * s, 0.0, 0.0, 0.0));
}
`
