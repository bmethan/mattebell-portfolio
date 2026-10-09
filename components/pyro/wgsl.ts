import { BB_MAX_K, BB_MIN_K } from './blackbody'
import { FIRE_BANDS } from './physics'
import { CP, FUEL, RHO_T, SCHMIDT_T, SMAGORINSKY, SOOT_EXTINCTION, T_AMBIENT, Y_O2_AIR, wf } from './physics'

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

export const SIM_COMMON = /* wgsl */ `
struct Params {
  dims: vec4<u32>,      // nx, ny, nz, frame
  grid: vec4<f32>,      // dx (m), dt (s), time (s), unused
  src: vec4<f32>,       // source centre (m), radius (m)
  srcProps: vec4<f32>,  // smoke density, smoke temperature rise (K), upward speed (m/s), noise (0..1)
  forces: vec4<f32>,    // buoyancy (m/s2 per K of temperature rise), unused, vorticity, turbulence (m/s2)
  forces2: vec4<f32>,   // turbulence scale (m), wind x (m/s), wind z (m/s), drag (1/s)
  decay: vec4<f32>,     // cooling (1/s), dissipation (1/s), SOR relaxation, speed limit (m/s)
  stirO: vec4<f32>,     // stir centre (m), radius (m)
  stirD: vec4<f32>,     // stir velocity change (m/s), on (0 or 1)
  stirR: vec4<f32>,     // unused xyz, MacCormack on (0 or 1)
  origin: vec4<f32>,    // domain min corner (m)
  detail: vec4<f32>,    // reset detail layer A this step (0 or 1), layer B
  fire: vec4<f32>,      // source is a fire bed (0 or 1), fuel it gives off (kg/s), soot yield (kg/kg), radiant fraction
  fire2: vec4<f32>,     // expansion (1 = physical), mixing time (s), the bed's nominal area (m2), unused
  ball: vec4<f32>,      // the collision ball's centre (m), radius (m; 0 = none)
  ballVel: vec4<f32>,   // its velocity (m/s), unused
}
@group(0) @binding(0) var<uniform> P: Params;

// The collision ball: a solid sphere. A cell (or a face) is solid when its centre is inside it.
fn solidAt(w: vec3<f32>) -> bool { return P.ball.w > 0.0 && distance(w, P.ball.xyz) < P.ball.w; }
fn cellSolid(c: vec3<i32>) -> bool { return solidAt(P.origin.xyz + (vec3<f32>(c) + 0.5) * P.grid.x); }

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
// The fuel the bed gives off into a cell in dt, per unit of the cell's mass: the rate that releases the chosen heat,
// spread over the bed's surface in proportion to its mask (the mask's area as measured last step; before that, its
// nominal area), into the bottom layer of cells.
fn fuelAdded(g: Gas, w: vec3<f32>, measuredArea: f32, dt: f32) -> f32 {
  let m = bedMask(w);
  if (m <= 0.0) { return 0.0; }
  let area = select(P.fire2.z, measuredArea, measuredArea > 1e-6);
  return P.fire.y * m / area * dt / (RHO_T / (T_AMB + g.rise) * P.grid.x);
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

// 2. Vorticity at cell centres (from cell-centred velocity, central differences), and its magnitude.
export const CURL = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var curlOut: texture_storage_3d<rgba16float, write>;

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
  let dt = P.grid.y;
  let vmax = P.decay.w;
  let keep = exp(-P.forces2.w * dt);
  let windT = 1.0 - exp(-WIND_DRAG * dt);
  var v = textureLoad(velIn, c, 0).xyz;

  if (id.y < n.y && id.z < n.z) {
    let w = worldOf(vec3<f32>(f.x, f.y + 0.5, f.z + 0.5));
    var a = 0.5 * (confine(c - X).x + confine(c).x);
    a += P.forces.w * turbMask(cellScal(c - X), cellScal(c)) * curlX(noiseSpace(w));
    var u = v.x + a * dt;
    u += (P.forces2.y - u) * windT;
    u = (u + stir(w).x) * keep;
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
      vy = (vy + stir(w).y) * keep;
      v.y = clamp(vy, -vmax, vmax);
    }
  }
  if (id.x < n.x && id.y < n.y) {
    let w = worldOf(vec3<f32>(f.x + 0.5, f.y + 0.5, f.z));
    var a = 0.5 * (confine(c - Z).z + confine(c).z);
    a += P.forces.w * turbMask(cellScal(c - Z), cellScal(c)) * curlZ(noiseSpace(w));
    var wz = v.z + a * dt;
    wz += (P.forces2.z - wz) * windT;
    wz = (wz + stir(w).z) * keep;
    v.z = clamp(wz, -vmax, vmax);
  }
  // Faces inside the collision ball move with it.
  if (P.ball.w > 0.0) {
    if (solidAt(worldOf(vec3<f32>(f.x, f.y + 0.5, f.z + 0.5)))) { v.x = P.ballVel.x; }
    if (solidAt(worldOf(vec3<f32>(f.x + 0.5, f.y, f.z + 0.5)))) { v.y = P.ballVel.y; }
    if (solidAt(worldOf(vec3<f32>(f.x + 0.5, f.y + 0.5, f.z)))) { v.z = P.ballVel.z; }
  }
  textureStore(velOut, c, vec4<f32>(v, 0.0));
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
  // A face against the collision ball moves with it.
  if (P.ball.w > 0.0) {
    if (cellSolid(c - X)) { x0 = P.ballVel.x; }
    if (cellSolid(c + X)) { x1 = P.ballVel.x; }
    if (cellSolid(c - Y)) { y0 = P.ballVel.y; }
    if (cellSolid(c + Y)) { y1 = P.ballVel.y; }
    if (cellSolid(c - Z)) { z0 = P.ballVel.z; }
    if (cellSolid(c + Z)) { z1 = P.ballVel.z; }
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
// The collision ball (centre, radius; radius 0 = none) and the grid's origin and finest cell size: a cell of any level
// is solid when its centre is inside the ball, so the coarse levels see it too.
struct Ball { s: vec4<f32>, o: vec4<f32> }
@group(0) @binding(5) var<uniform> B: Ball;
fn solidL(c: vec3<i32>) -> bool {
  if (B.s.w <= 0.0) { return false; }
  let w = B.o.xyz + (vec3<f32>(c) + 0.5) * sqrt(L.h.x);
  return distance(w, B.s.xyz) < B.s.w;
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
// A face between a solid cell (the collision ball) and any other moves with the ball.
fn ballFaces(c: vec3<i32>, v0: vec3<f32>) -> vec3<f32> {
  var v = v0;
  if (P.ball.w > 0.0) {
    if (cellSolid(c) || cellSolid(c - X)) { v.x = P.ballVel.x; }
    if (cellSolid(c) || cellSolid(c - Y)) { v.y = P.ballVel.y; }
    if (cellSolid(c) || cellSolid(c - Z)) { v.z = P.ballVel.z; }
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
  if (P.ball.w > 0.0) {
    if (cellSolid(c) || cellSolid(c - X)) { proj.x = P.ballVel.x; refl.x = P.ballVel.x; }
    if (cellSolid(c) || cellSolid(c - Y)) { proj.y = P.ballVel.y; refl.y = P.ballVel.y; }
    if (cellSolid(c) || cellSolid(c - Z)) { proj.z = P.ballVel.z; refl.z = P.ballVel.z; }
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

export const RENDER_COMMON = /* wgsl */ `
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
  fire: vec4<f32>,     // source is a fire (0 or 1), extinction in red and in blue relative to green, flame eddies' rise (m/s)
  wb0: vec4<f32>,      // the camera's white balance, rows of a 3x3 on linear Rec.709; wind (m/s) in the first row's w
  wb1: vec4<f32>,      // (w: sample the scalars tricubically, 0 or 1)
  wb2: vec4<f32>,      // (w: render the smoke from the upres grid, 0 or 1)
  scene: vec4<f32>,    // background: the sky and a floor to the horizon (1) or a studio's black (0); unused
}
@group(0) @binding(1) var<uniform> R: RParams;

const FIRE_BANDS = ${FIRE_BANDS}u;
const BAND_R2 = 0.02; // m2: softens each band of the fire's light over the flame's width
const INV_PI = 0.31830989;
const INV_2PI = 0.15915494;
const INV_4PI = 0.07957747;
const SOOT_KE = ${wf(SOOT_EXTINCTION)}; // m2 per kg of soot, at 550 nm

// Extinction at the reference (green) wavelength, 1/m: a fire's soot by its mass per volume (its partial density), a
// smoke source's smoke by its density. Red and blue scale by chroma().
fn sigmaOf(s: vec4<f32>) -> f32 {
  if (R.fire.x > 0.5) { return SOOT_KE * (RHO_T / T_AMB) * max(s.x, 0.0) / SOOT_G; }
  return R.sunCol.w * max(s.x, 0.0);
}
fn chroma() -> vec3<f32> { return vec3<f32>(R.fire.y, 1.0, R.fire.z); }

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

fn density(w: vec3<f32>) -> f32 { return sigmaOf(textureSampleLevel(scal, samp, uvwOf(w), 0.0)); }

// The fire's light here, shadowed by the soot: each slice of the fire's height as a point light at its brightness-
// weighted centre, carrying its emitted intensity (I / r^2), through the optical depth on the way (12 samples). In
// kilolux of luminance (the colour is the fire's overall colour), so it fits half precision near the flames.
fn fireIrradiance(w: vec3<f32>) -> f32 {
  if (P.fire.x < 0.5) { return 0.0; }
  let vol = P.grid.x * P.grid.x * P.grid.x;
  var e = 0.0;
  for (var b = 0u; b < FIRE_BANDS; b++) {
    let lum = fireSums[3u + 2u * b].w * vol;
    if (lum <= 0.0) { continue; }
    let d = fireSums[4u + 2u * b].xyz - w;
    let r2 = dot(d, d);
    if (ballHit(w, d / max(sqrt(r2), 1e-6)) < sqrt(r2)) { continue; }
    var tau = 0.0;
    for (var i = 0; i < 12; i++) { tau += density(w + d * ((f32(i) + 0.5) / 12.0)); }
    e += lum * exp(-tau * sqrt(r2) / 12.0) / (r2 + BAND_R2);
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
      }
    }
  }
  // Every cell gets its depths, smoke or not: the floor's shadow and sky lookups land in empty cells too.
  let tauSun = depth(w, R.sunDir.xyz, 1.5 * P.grid.x, false);
  let tauUp = depth(w, vec3<f32>(0.0, 1.0, 0.0), 2.0 * P.grid.x, false);
  textureStore(lightOut, vec3<i32>(id), vec4<f32>(tauSun, tauUp, occ, fireIrradiance(w)));
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

// 9b. Sums over the grid, each step, per workgroup and then across workgroups: the fire's light, for the smoke and
// the floor (its total emitted intensity, from soot too thin to hide itself, and its brightness-weighted centre, lit
// as a point light from there); the fire bed's area (its mask's integral, which spreads the fuel); and the heat
// release rate.
export const FIRE_SUM = /* wgsl */ `
${SIM_COMMON}
${RENDER_COMMON}
${fireCommon(3)}
@group(0) @binding(2) var scal: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> partial: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> hrr: array<f32>;
var<workgroup> sumE: array<vec4<f32>, 64>;
var<workgroup> sumP: array<vec4<f32>, 64>;
var<workgroup> sumB: array<vec4<f32>, 64>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let c = vec3<i32>(id);
  let vol = P.grid.x * P.grid.x * P.grid.x;
  var e = vec3<f32>(0.0);
  var b = vec4<f32>(0.0);
  if (all(c < nCells()) && P.fire.x > 0.5) {
    let s = textureLoad(scal, c, 0);
    // Emission per unit volume (cd/m3): what the soot absorbs times the blackbody radiance at its temperature.
    let g = gasOf(s);
    e = (1.0 - R.skyCol.w) * sigmaOf(s) * chroma() * planck(T_AMB + g.rise);
    let bed = bedMask(worldOf(vec3<f32>(id) + 0.5)) * P.grid.x * P.grid.x;
    b = vec4<f32>(bed, hrr[cellIndex(c)] * vol, 0.0, 0.0);
  }
  let lum = dot(e, vec3<f32>(0.2126, 0.7152, 0.0722));
  sumE[li] = vec4<f32>(e, lum);
  sumP[li] = vec4<f32>(worldOf(vec3<f32>(id) + 0.5) * lum, 0.0);
  sumB[li] = b;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st = st >> 1u) {
    if (li < st) {
      sumE[li] += sumE[li + st];
      sumP[li] += sumP[li + st];
      sumB[li] += sumB[li + st];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let k = wid.x + nw.x * (wid.y + nw.y * wid.z);
    partial[3u * k] = sumE[0];
    partial[3u * k + 1u] = sumP[0];
    partial[3u * k + 2u] = sumB[0];
  }
}
`

export const FIRE_SUM_TOTAL = /* wgsl */ `
// x: the number of partial sums (one per workgroup of FIRE_SUM); y, z, w: its workgroups along x, y and z.
@group(0) @binding(0) var<uniform> C: vec4<u32>;
@group(0) @binding(1) var<storage, read> partial: array<vec4<f32>>;
// 0: the emitted intensity summed over cells, per unit of cell volume (rgb, and its luminance); 1: its centre; 2: the
// bed's area (m2) and the heat release rate (W); then for each of FIRE_BANDS slices of the grid's height, its emitted
// intensity (as 0) and its centre.
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;
const FIRE_BANDS = ${FIRE_BANDS}u;
var<workgroup> sE: array<vec4<f32>, 256>;
var<workgroup> sP: array<vec4<f32>, 256>;
var<workgroup> sB: array<vec4<f32>, 256>;

// Workgroup 0 sums everything; workgroup 1 + b sums band b.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let band = i32(wg.x) - 1;
  var e = vec4<f32>(0.0);
  var p = vec4<f32>(0.0);
  var b = vec4<f32>(0.0);
  for (var k = li; k < C.x; k += 256u) {
    if (band >= 0 && i32(((k / C.y) % C.z) * FIRE_BANDS / C.z) != band) { continue; }
    e += partial[3u * k];
    p += partial[3u * k + 1u];
    b += partial[3u * k + 2u];
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
  if (li == 0u) {
    let centre = vec4<f32>(sP[0].xyz / max(sE[0].w, 1e-9), 0.0);
    if (band < 0) {
      result[0] = sE[0];
      result[1] = centre;
      result[2] = sB[0];
    } else {
      result[3u + 2u * u32(band)] = sE[0];
      result[4u + 2u * u32(band)] = centre;
    }
  }
}
`

// 10. The image, in photometric units: march each camera ray through the volume (Hillaire's integration of each
// step, T (1 - exp(-sigma dt)) S), lit by the sun through the light volume with Wrenninge's multiple-scattering
// octaves (a = b = c = 1/2, which conserves energy) and Henyey-Greenstein phase, by the sky through the optical depth
// above, and by the fire; hot soot glows at its blackbody radiance; then a floor that catches the smoke's shadow,
// fading to black. Written exposed, at high dynamic range, for the accumulation pass.
export const RENDER = /* wgsl */ `
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
@group(0) @binding(11) var blueTex: texture_2d<f32>;
@group(0) @binding(12) var upresTex: texture_3d<f32>; // a smoke source's density on the twice-as-fine grid

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

// Where in its first step each pixel's march starts: blue noise in space (fine-grained, no clumps), stepped by the
// golden ratio each frame, so the temporal accumulation averages the sampling error away within a few frames.
fn jitterAt(p: vec2<f32>, frame: f32) -> f32 {
  let b = textureLoad(blueTex, vec2<i32>(p) & vec2<i32>(63), 0).x;
  return fract(b + frame * 0.61803399);
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

  let tFloor = select(1e9, -ro.y / rd.y, rd.y < -1e-5);
  let box = boxHit(ro, rd, P.origin.xyz, domainHi());
  let tBall = ballHit(ro, rd);
  let tEnd = min(min(box.y, tFloor), tBall);
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
      if (textureSampleLevel(light, samp, uvw, 0.0).z < 1e-4) {
        t += 3.0 * ds;
        continue;
      }
      t += ds;
      let fade = edgeFade(uvw);
      let s4 = scalSmooth(uvw);
      var sig = sigmaOf(s4) * fade;
      let upres = R.wb2.w > 0.5;
      if (upres) { sig = R.sunCol.w * max(textureSampleLevel(upresTex, samp, uvw, 0.0).x, 0.0) * fade; }
      if (sig > 1e-3) {
        // Upres detail (unless the wavelet-turbulence grid supplies it): strongest in thin smoke (the edges and wisps),
        // a third as strong in the dense core, and fading out in the faintest haze, where fine noise reads as grain.
        if (R.detail.z > 0.0 && !upres) {
          let thin = 1.0 - smoothstep(0.2, 4.0, sig);
          let faint = smoothstep(0.0, 1.0, sig);
          sig *= clamp(1.0 + 2.0 * R.detail.z * detail(w, uvw) * (0.35 + 0.65 * thin) * faint, 0.0, 2.0);
        }
        let L = textureSampleLevel(light, samp, uvw, 0.0);
        let tauS = L.x * ch;
        let lsun = (exp(-tauS) * ph0 + 0.5 * exp(-0.5 * tauS) * ph1 + 0.25 * exp(-0.25 * tauS) * ph2) * ballSun(w, sun);
        // The sky as a uniform dome over open ground (radiance E / pi), half of it seen by an isotropic scatterer.
        let amb = R.skyCol.xyz * INV_2PI * exp(-0.5 * L.y * ch);
        // The fire's light, shadowed by the soot on the way (from the light volume, in kilolux).
        let fireE = fireColour * 1000.0 * L.w;
        // Light scattered toward the camera per unit of extinction, cd/m2: the phase function is hg / (4 pi).
        let S = albedo * (R.sunCol.xyz * INV_4PI * lsun + amb + fireE * INV_4PI);
        // Emission (Kirchhoff's law): what the soot absorbs, it emits at the blackbody radiance of its temperature.
        var B = vec3<f32>(0.0);
        let rise = riseOf(s4);
        if (R.fire.x > 0.5 && rise > 400.0) {
          var kelvin = T_AMB + rise;
          if (R.size.w > 0.0) { kelvin = max(T_AMB, kelvin + subgridKelvin(w, uvw, rot)); }
          B = (1.0 - albedo) * planck(kelvin);
        }
        // Integrated exactly over the step for constant properties (Hillaire 2015).
        let Ts = exp(-sig * ch * ds);
        col += T * (1.0 - Ts) * (S + B);
        T *= Ts;
        if (max(T.x, max(T.y, T.z)) < 0.004) { break; }
      }
    }
  }

  if (tBall < tFloor && tBall < 1e8 && max(T.x, max(T.y, T.z)) > 0.004) {
    // The collision ball: matte grey, lit by the sun and the sky through the smoke (the light volume just outside its
    // surface) and by the fire's shadowed light, turned onto the surface toward the fire's centre.
    let ph = ro + rd * tBall;
    let nb = normalize(ph - P.ball.xyz);
    let lb = textureSampleLevel(light, samp, uvwOf(ph + nb * 1.5 * P.grid.x), 0.0);
    let toFire = fireLight[1].xyz - ph;
    let fireE = fireColour * 1000.0 * lb.w * max(dot(nb, toFire), 0.0) / max(length(toFire), 1e-3);
    let E = R.sunCol.xyz * max(dot(nb, sun), 0.0) * exp(-lb.x * ch) + R.skyCol.xyz * (0.5 + 0.5 * nb.y) * exp(-0.5 * lb.y * ch) + fireE;
    col += T * BALL_ALBEDO * INV_PI * E;
  } else if (tFloor < 1e8 && max(T.x, max(T.y, T.z)) > 0.004) {
    let pf = ro + rd * tFloor;
    // In the studio the floor fades to black away from the fire; under the sky it runs to the horizon.
    let fade = select(exp(-dot(pf.xz, pf.xz) / (R.floorCol.w * R.floorCol.w)), 1.0, R.scene.x > 0.5);
    // The smoke's shadow: the light volume's depth at the point where the sun ray from the floor enters the domain.
    let sb = boxHit(pf, sun, P.origin.xyz, domainHi());
    var shadow = vec3<f32>(1.0);
    if (sb.y > max(sb.x, 0.0)) {
      let pe = pf + sun * max(sb.x, 0.0) + vec3<f32>(0.0, 0.5 * P.grid.x, 0.0);
      shadow = exp(-textureSampleLevel(light, samp, uvwOf(pe), 0.0).x * ch);
    }
    // And the collision ball's; it also hides its share of the sky, its view factor from the floor, (r / d)^2 cos.
    shadow *= ballSun(pf, sun);
    var ballSky = 1.0;
    if (P.ball.w > 0.0) {
      let tb = P.ball.xyz - pf;
      let d2 = dot(tb, tb);
      ballSky = 1.0 - P.ball.w * P.ball.w / d2 * max(tb.y, 0.0) / sqrt(d2);
    }
    var skyVis = vec3<f32>(1.0);
    let uv = uvwOf(pf + vec3<f32>(0.0, 0.5 * P.grid.x, 0.0));
    if (all(uv.xz >= vec2<f32>(0.0)) && all(uv.xz <= vec2<f32>(1.0))) {
      skyVis = exp(-0.5 * textureSampleLevel(light, samp, uv, 0.0).y * ch);
    }
    // Illuminance on the floor (lux): the sun, the sky and the fire, seen as Lambertian radiance E / pi. The fire's
    // light from each band of its height, I cos / r^2 (the floor reaches beyond the light volume, and smoke rises away
    // from the light between the flames and the floor, so it goes unshadowed here).
    var fireLux = 0.0;
    let cellVol = P.grid.x * P.grid.x * P.grid.x;
    for (var b = 0u; b < FIRE_BANDS; b++) {
      let lum = fireLight[3u + 2u * b].w * cellVol;
      if (lum <= 0.0) { continue; }
      let d = fireLight[4u + 2u * b].xyz - pf;
      let r2 = dot(d, d) + BAND_R2;
      fireLux += lum * max(d.y, 0.0) / (sqrt(r2) * r2);
    }
    let fireE = fireColour * fireLux;
    let E = R.sunCol.xyz * max(sun.y, 0.0) * shadow + R.skyCol.xyz * skyVis * ballSky + fireE;
    var floorL = R.floorCol.xyz * INV_PI * E * fade;
    // Under the sky, the far floor fades into the horizon's haze (aerial perspective, over some 40 m).
    if (R.scene.x > 0.5) { floorL = mix(floorL, R.skyCol.xyz * INV_PI * 1.25, 1.0 - exp(-tFloor / 40.0)); }
    col += T * floorL;
  } else if (R.scene.x > 0.5 && tBall >= 1e8) {
    // The sky: a dome of the sky's radiance (its illuminance on the ground over pi), a little brighter toward the
    // horizon. At night it is too dim to see.
    col += T * R.skyCol.xyz * INV_PI * mix(1.25, 0.75, sqrt(max(rd.y, 0.0)));
  }

  // Exposed (as a fraction of the luminance the camera maps to white), for the accumulation pass.
  return vec4<f32>(col * R.camUp.w, 1.0);
}
`

// 11. Temporal accumulation, then the camera. Each frame's march is blended into a running history, which averages
// away the sampling noise; the history is first clamped to the range of the current frame's 3 x 3 neighbourhood (in a
// luminance-compressed space, so a bright flame does not dominate), so what moves or flickers is not smeared behind
// it. Then the white balance and ACES tone mapping, and a dither against banding.
export const RESOLVE = /* wgsl */ `
${SIM_COMMON}
${RENDER_COMMON}
@group(0) @binding(2) var current: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var<uniform> T: vec4<f32>; // history valid (0 or 1), weight of the current frame

struct Out {
  @location(0) color: vec4<f32>,
  @location(1) history: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn ign(p: vec2<f32>) -> f32 { return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715)))); }
fn lum(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }
fn squash(c: vec3<f32>) -> vec3<f32> { return c / (1.0 + lum(c)); }
fn unsquash(c: vec3<f32>) -> vec3<f32> { return c / max(1.0 - lum(c), 1e-4); }
// Stephen Hill's fit of the ACES RRT and sRGB ODT (rows of the input and output matrices).
fn aces(c: vec3<f32>) -> vec3<f32> {
  let a = vec3<f32>(dot(vec3<f32>(0.59719, 0.35458, 0.04823), c), dot(vec3<f32>(0.07600, 0.90834, 0.01566), c), dot(vec3<f32>(0.02840, 0.13383, 0.83777), c));
  let b = (a * (a + 0.0245786) - 0.000090537) / (a * (0.983729 * a + 0.4329510) + 0.238081);
  let o = vec3<f32>(dot(vec3<f32>(1.60475, -0.53108, -0.07367), b), dot(vec3<f32>(-0.10208, 1.10813, -0.00605), b), dot(vec3<f32>(-0.00327, -0.07276, 1.07602), b));
  return clamp(o, vec3<f32>(0.0), vec3<f32>(1.0));
}
fn srgb(c: vec3<f32>) -> vec3<f32> {
  return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> Out {
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
  var blended = cur;
  if (T.x > 0.5) {
    let h = clamp(squash(textureLoad(history, p, 0).xyz), lo, hi);
    blended = mix(h, cur, T.y);
  }
  let c = unsquash(blended);
  let balanced = vec3<f32>(dot(R.wb0.xyz, c), dot(R.wb1.xyz, c), dot(R.wb2.xyz, c));
  let mapped = srgb(aces(max(balanced, vec3<f32>(0.0))));
  return Out(vec4<f32>(mapped + (ign(fc.xy + 7.13) - 0.5) / 255.0, 1.0), vec4<f32>(c, 1.0));
}
`

// 12. Wavelet-turbulence upres (after Kim, Thuerey, James and Gross 2008), for a smoke source: the smoke's density again
// on a grid twice as fine, carried by the solver's velocity plus turbulence too fine for the coarse grid. That
// turbulence is the curl of a noise (divergence free), read at the texture coordinates the detail layers carry with the
// flow, so its eddies travel with the smoke and fade where the flow has stretched them, and is as strong as the local
// flow, as eddies a band finer carry energy in proportion to the resolved flow's. Semi-Lagrangian: at half the cell
// size it smears half as much. The renderer reads this density; the lighting still comes from the coarse grid.
export const UPRES = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var src: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var detA: texture_3d<f32>;
@group(0) @binding(6) var detB: texture_3d<f32>;
@group(0) @binding(7) var noiseTex: texture_3d<f32>;
@group(0) @binding(8) var noiseSamp: sampler;
@group(0) @binding(9) var scal: texture_3d<f32>;
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

const ROT = mat3x3<f32>(0.8, -0.6, 0.0, 0.36, 0.48, 0.8, -0.48, -0.64, 0.6);
// The curl of the noise's three channels as a vector potential, by central differences half a texel apart, scaled to
// about unit size (the fbm's gradients run about 20 per tile).
fn curlNoise(p: vec3<f32>) -> vec3<f32> {
  let h = 0.5 / 64.0;
  let q = ROT * p;
  let ex = ROT * vec3<f32>(h, 0.0, 0.0);
  let ey = ROT * vec3<f32>(0.0, h, 0.0);
  let ez = ROT * vec3<f32>(0.0, 0.0, h);
  let dx = textureSampleLevel(noiseTex, noiseSamp, q + ex, 0.0).xyz - textureSampleLevel(noiseTex, noiseSamp, q - ex, 0.0).xyz;
  let dy = textureSampleLevel(noiseTex, noiseSamp, q + ey, 0.0).xyz - textureSampleLevel(noiseTex, noiseSamp, q - ey, 0.0).xyz;
  let dz = textureSampleLevel(noiseTex, noiseSamp, q + ez, 0.0).xyz - textureSampleLevel(noiseTex, noiseSamp, q - ez, 0.0).xyz;
  return vec3<f32>(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) / (2.0 * h * 20.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let nf = vec3<i32>(textureDimensions(dst));
  let c = vec3<i32>(id);
  if (any(c >= nf)) { return; }
  let gF = vec3<f32>(id) + 0.5;
  let g = 0.5 * gF; // coarse grid coordinates
  let w = worldOf(g);
  if (U.init > 0.5) {
    textureStore(dst, c, vec4<f32>(max(scalAt(scal, samp, g).x, 0.0), 0.0, 0.0, 0.0));
    return;
  }
  let u = velAt(vel, samp, g);
  let uvw = g / nF();
  let a = textureSampleLevel(detA, samp, uvw, 0.0);
  let b = textureSampleLevel(detB, samp, uvw, 0.0);
  let wa = U.wa / (1.0 + a.w * a.w);
  let wb = U.wb / (1.0 + b.w * b.w);
  let turb = (curlNoise((w + a.xyz) * U.k) * wa + curlNoise((w + b.xyz) * U.k + vec3<f32>(0.37, 0.61, 0.13)) * wb) / max(sqrt(U.wa * U.wa + U.wb * U.wb), 1e-3);
  let v = u + U.strength * length(u) * turb;
  // Back along the flow, in fine cells (twice as many per metre).
  let gb = gF - 2.0 * v * P.grid.y / P.grid.x;
  let n = vec3<f32>(nf);
  var d = 0.0;
  if (gb.x >= 0.0 && gb.x <= n.x && gb.z >= 0.0 && gb.z <= n.z && gb.y <= n.y) {
    d = textureSampleLevel(src, samp, (clamp(gb - 0.5, vec3<f32>(0.0), n - 1.0) + 0.5) / n, 0.0).x;
  }
  let top = smoothstep(0.7, 1.0, gF.y / n.y);
  d = max(d, 0.0) * exp(-(P.decay.y + 2.5 * top) * P.grid.y);
  d = max(d, P.srcProps.x * sourceMask(w));
  if (solidAt(P.origin.xyz + gF * 0.5 * P.grid.x)) { d = 0.0; }
  textureStore(dst, c, vec4<f32>(d, 0.0, 0.0, 0.0));
}
`
