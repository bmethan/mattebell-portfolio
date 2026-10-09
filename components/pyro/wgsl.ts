import { BB_MAX_K, BB_MIN_K } from './blackbody'

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
  grid: vec4<f32>,      // dx (m), dt (s), time (s), ambient temperature
  src: vec4<f32>,       // source centre (m), radius (m)
  srcProps: vec4<f32>,  // density, temperature, upward speed (m/s), noise (0..1)
  forces: vec4<f32>,    // buoyancy (m/s2 per unit temperature), weight (m/s2 per unit density), vorticity, turbulence (m/s2)
  forces2: vec4<f32>,   // turbulence scale (m), wind x (m/s), wind z (m/s), drag (1/s)
  decay: vec4<f32>,     // cooling (1/s), dissipation (1/s), SOR relaxation, speed limit (m/s)
  stirO: vec4<f32>,     // stir centre (m), radius (m)
  stirD: vec4<f32>,     // stir velocity change (m/s), on (0 or 1)
  stirR: vec4<f32>,     // unused xyz, MacCormack on (0 or 1)
  origin: vec4<f32>,    // domain min corner (m)
  detail: vec4<f32>,    // reset detail layer A this step (0 or 1), layer B
  fire: vec4<f32>,      // source is fuel (0 or 1), source fuel, burn rate (1/s), heat released per unit of fuel
  fire2: vec4<f32>,     // expansion (1/s at full flame), soot per unit of fuel, flame decay (1/s), ignition temperature
}
@group(0) @binding(0) var<uniform> P: Params;

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

// The source, its edge one cell soft, broken into puffs by fast-drifting noise: a sphere resting on the floor for smoke,
// for fire a squat dome, 1.6 times wider than tall (a fire bed).
fn sourceMask(w: vec3<f32>) -> f32 {
  let r = P.src.w;
  let q = w - P.src.xyz;
  let d = length(vec3<f32>(q.x, q.y * select(1.0, 1.6, P.fire.x > 0.5), q.z));
  let m = 1.0 - smoothstep(r - 2.0 * P.grid.x, r, d);
  if (m <= 0.0) { return 0.0; }
  // Coarse noise drifting in every direction: a pattern held still in x and z would emit the same thin columns
  // step after step, which the updraft stretches into streaks.
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

// 1. Advect the velocity by itself (semi-Lagrangian, RK2 backtrace).
export const ADVECT_VEL = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var velSrc: texture_3d<f32>;
@group(0) @binding(3) var velDst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = P.dims.xyz;
  if (id.x > n.x || id.y > n.y || id.z > n.z) { return; }
  let f = vec3<f32>(id);
  var o = vec3<f32>(0.0);
  if (id.y < n.y && id.z < n.z) {
    o.x = uAt(velSrc, samp, backtrace(velSrc, samp, vec3<f32>(f.x, f.y + 0.5, f.z + 0.5), 1.0));
  }
  if (id.x < n.x && id.z < n.z && id.y > 0u) {
    o.y = vAt(velSrc, samp, backtrace(velSrc, samp, vec3<f32>(f.x + 0.5, f.y, f.z + 0.5), 1.0));
  }
  if (id.x < n.x && id.y < n.y) {
    o.z = wAt(velSrc, samp, backtrace(velSrc, samp, vec3<f32>(f.x + 0.5, f.y + 0.5, f.z), 1.0));
  }
  textureStore(velDst, vec3<i32>(id), vec4<f32>(o, 0.0));
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

// 3. Forces on each face: vorticity confinement (Fedkiw, Stam and Jensen 2001), buoyancy from temperature and
// weight from density, curl-noise turbulence where there is smoke, wind, the source's upward push, the visitor's
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

// Turbulence: the curl of a noise vector potential (divergence free by construction), one component at a time.
fn psi(q: vec3<f32>, k: f32) -> f32 { return vnoise(q + vec3<f32>(k * 37.1, k * 11.7, k * 23.3)) - 0.5; }
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
  return w / max(P.forces2.x, 0.01) + vec3<f32>(0.0, -0.5 * P.grid.z, 0.13 * P.grid.z);
}
fn turbMask(d: f32) -> f32 { return clamp(d * 3.0, 0.0, 1.0); }

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
    let d = 0.5 * (cellScal(c - X).x + cellScal(c).x);
    var a = 0.5 * (confine(c - X).x + confine(c).x);
    a += P.forces.w * turbMask(d) * curlX(noiseSpace(w));
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
      let d = 0.5 * (s0.x + s1.x);
      let temp = 0.5 * (s0.y + s1.y);
      var a = 0.5 * (confine(c - Y).y + confine(c).y) + P.forces.x * (temp - P.grid.w) - P.forces.y * d;
      a += P.forces.w * turbMask(d) * curlY(noiseSpace(w));
      var vy = v.y + a * dt;
      vy -= vy * windT;
      vy = mix(vy, P.srcProps.z, sourceMask(w));
      vy = (vy + stir(w).y) * keep;
      v.y = clamp(vy, -vmax, vmax);
    }
  }
  if (id.x < n.x && id.y < n.y) {
    let w = worldOf(vec3<f32>(f.x + 0.5, f.y + 0.5, f.z));
    let d = 0.5 * (cellScal(c - Z).x + cellScal(c).x);
    var a = 0.5 * (confine(c - Z).z + confine(c).z);
    a += P.forces.w * turbMask(d) * curlZ(noiseSpace(w));
    var wz = v.z + a * dt;
    wz += (P.forces2.z - wz) * windT;
    wz = (wz + stir(w).z) * keep;
    v.z = clamp(wz, -vmax, vmax);
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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let a = textureLoad(vel, c, 0);
  let v0 = select(a.y, 0.0, c.y == 0);
  let div = ((textureLoad(vel, c + X, 0).x - a.x) + (textureLoad(vel, c + Y, 0).y - v0) + (textureLoad(vel, c + Z, 0).z - a.z)) / P.grid.x;
  // Burning gas expands: the projection makes the divergence equal this source instead of 0 where there is flame.
  let expansion = P.fire2.x * textureLoad(scal, c, 0).w;
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
  var sum = 0.0;
  var cnt = 4.0; // x and z neighbours: inside, or p = 0 just outside
  if (c.x > 0) { sum += p[i - 1]; }
  if (c.x < n.x - 1) { sum += p[i + 1]; }
  if (c.z > 0) { sum += p[i - sz]; }
  if (c.z < n.z - 1) { sum += p[i + sz]; }
  if (c.y > 0) { sum += p[i - sy]; cnt += 1.0; }
  cnt += 1.0;
  if (c.y < n.y - 1) { sum += p[i + sy]; }
  let pn = (sum + rhs[i] * P.grid.x * P.grid.x) / cnt;
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
`

export const MG_SMOOTH = /* wgsl */ `
${MG_COMMON}
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(16, 4, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nL();
  let c = vec3<i32>(i32(2u * id.x + ((id.y + id.z + L.n.w) & 1u)), i32(id.y), i32(id.z));
  if (any(c >= n)) { return; }
  let i = at(c, n);
  let sy = n.x;
  let sz = n.x * n.y;
  var sum = 0.0;
  var cnt = 5.0; // the x and z sides and the top, inside or open
  if (c.x > 0) { sum += x[i - 1]; }
  if (c.x < n.x - 1) { sum += x[i + 1]; }
  if (c.z > 0) { sum += x[i - sz]; }
  if (c.z < n.z - 1) { sum += x[i + sz]; }
  if (c.y < n.y - 1) { sum += x[i + sy]; }
  if (c.y > 0) { sum += x[i - sy]; cnt += 1.0; }
  x[i] = mix(x[i], (sum + g[i] * L.h.x) / cnt, L.h.y);
}
`

export const MG_RESIDUAL = /* wgsl */ `
${MG_COMMON}
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = nL();
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let i = at(c, n);
  let sy = n.x;
  let sz = n.x * n.y;
  var sum = 0.0;
  var cnt = 5.0;
  if (c.x > 0) { sum += x[i - 1]; }
  if (c.x < n.x - 1) { sum += x[i + 1]; }
  if (c.z > 0) { sum += x[i - sz]; }
  if (c.z < n.z - 1) { sum += x[i + sz]; }
  if (c.y < n.y - 1) { sum += x[i + sy]; }
  if (c.y > 0) { sum += x[i - sy]; cnt += 1.0; }
  r[i] = g[i] - (cnt * x[i] - sum) / L.h.x;
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
  textureStore(velOut, c, vec4<f32>(v, 0.0));
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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let g = vec3<f32>(id) + 0.5;
  let gs = backtrace(vel, samp, g, 1.0);
  // The material here came from gs: its origin, relative to here, is its origin relative to gs plus (gs - g).
  let shift = (gs - g) * P.grid.x;
  let a = select(scalAt(srcA, samp, gs).xyz + shift, vec3<f32>(0.0), P.detail.x > 0.5);
  let b = select(scalAt(srcB, samp, gs).xyz + shift, vec3<f32>(0.0), P.detail.y > 0.5);
  textureStore(dstA, c, vec4<f32>(a, 0.0));
  textureStore(dstB, c, vec4<f32>(b, 0.0));
}
`

// The detail noise: tileable gradient-noise fbm (lattice periods 8, 16 and 32 across a 64-texel tile; gradient noise
// has none of value noise's blocky, axis-aligned look), made once.
export const NOISE_GEN = /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_3d<rgba8unorm, write>;

fn hashW(c0: vec3<i32>, period: i32) -> f32 {
  let c = ((c0 % period) + period) % period;
  var h = (bitcast<u32>(c.x) * 0x8da6b343u) ^ (bitcast<u32>(c.y) * 0xd8163841u) ^ (bitcast<u32>(c.z) * 0xcb1ab31fu) ^ (u32(period) * 0x9e3779b9u);
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return f32(h) * (1.0 / 4294967296.0);
}
// A random unit gradient at a lattice point (uniform on the sphere).
fn gradW(c: vec3<i32>, period: i32) -> vec3<f32> {
  let z = 2.0 * hashW(c, period) - 1.0;
  let a = 6.2831853 * hashW(c + vec3<i32>(173, 311, 467), period);
  let r = sqrt(max(1.0 - z * z, 0.0));
  return vec3<f32>(r * cos(a), r * sin(a), z);
}
fn corner(c: vec3<i32>, o: vec3<i32>, f: vec3<f32>, period: i32) -> f32 {
  return dot(gradW(c + o, period), f - vec3<f32>(o));
}
// Gradient (Perlin) noise, about -0.7 to 0.7, wrapping every period lattice cells.
fn tileNoise(q: vec3<f32>, period: i32) -> f32 {
  let i = floor(q);
  let f = q - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let c = vec3<i32>(i);
  let x00 = mix(corner(c, vec3<i32>(0, 0, 0), f, period), corner(c, vec3<i32>(1, 0, 0), f, period), u.x);
  let x10 = mix(corner(c, vec3<i32>(0, 1, 0), f, period), corner(c, vec3<i32>(1, 1, 0), f, period), u.x);
  let x01 = mix(corner(c, vec3<i32>(0, 0, 1), f, period), corner(c, vec3<i32>(1, 0, 1), f, period), u.x);
  let x11 = mix(corner(c, vec3<i32>(0, 1, 1), f, period), corner(c, vec3<i32>(1, 1, 1), f, period), u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = textureDimensions(dst);
  if (any(id >= n)) { return; }
  let q = vec3<f32>(id) / vec3<f32>(n);
  let v = 0.55 * tileNoise(q * 8.0, 8) + 0.3 * tileNoise(q * 16.0, 16) + 0.15 * tileNoise(q * 32.0, 32);
  let s = clamp(0.5 + 1.7 * v, 0.0, 1.0);
  textureStore(dst, vec3<i32>(id), vec4<f32>(s, s, s, 1.0));
}
`

// 8. Modified MacCormack (Selle et al. 2008): the forward result plus half the round-trip error, clamped to the
// values the forward step interpolated between; then dissipation, cooling and the source.
export const COMBINE = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var s0: texture_3d<f32>;
@group(0) @binding(4) var s1: texture_3d<f32>;
@group(0) @binding(5) var s2: texture_3d<f32>;
@group(0) @binding(6) var dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  let n = nCells();
  if (any(c >= n)) { return; }
  var phi = textureLoad(s1, c, 0);
  if (P.stirR.w > 0.5) {
    let g = backtrace(vel, samp, vec3<f32>(id) + 0.5, 1.0);
    if (inDomain(g)) {
      let corrected = phi + 0.5 * (textureLoad(s0, c, 0) - textureLoad(s2, c, 0));
      let b = vec3<i32>(floor(clamp(g - 0.5, vec3<f32>(0.0), vec3<f32>(n - 1))));
      var lo = vec4<f32>(1e9);
      var hi = vec4<f32>(-1e9);
      for (var k = 0; k < 8; k++) {
        let q = textureLoad(s0, min(b + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1), n - 1), 0);
        lo = min(lo, q);
        hi = max(hi, q);
      }
      phi = clamp(corrected, lo, hi);
    }
  }
  let dt = P.grid.y;
  // Near the open top the smoke disperses faster, so it thins out before the boundary instead of being cut there.
  let top = smoothstep(0.7, 1.0, (f32(id.y) + 0.5) / f32(n.y));
  phi.x = max(phi.x, 0.0) * exp(-(P.decay.y + 2.5 * top) * dt);
  phi.y = P.grid.w + (max(phi.y, P.grid.w) - P.grid.w) * exp(-P.decay.x * dt);
  phi.z = max(phi.z, 0.0);
  // Combustion, with a reaction coordinate (after Nguyen et al. 2002): the fuel field is the fraction of the reaction
  // left, 1 as gas leaves the source, falling at the burn rate (per second) once it is hotter than the ignition
  // temperature; what burns releases heat (the buoyancy). Soot survives as smoke where the flame burns out, so it is
  // released as the reaction runs down (weighted toward its end), not where the flame is bright. The flame field is
  // the reaction left where it burns: it sets the radiating temperature and the expansion, and fades once the
  // reaction has run out.
  let m = sourceMask(worldOf(vec3<f32>(id) + 0.5));
  phi.w = max(phi.w, 0.0) * exp(-P.fire2.z * dt);
  if (phi.z > 0.0 && phi.y > P.fire2.w) {
    let burned = min(phi.z, P.fire.z * dt);
    phi.z -= burned;
    phi.y += P.fire.w * burned;
    let spent = 1.0 - phi.z;
    phi.x += P.fire2.y * burned * 3.0 * spent * spent;
    phi.w = max(phi.w, phi.z);
  }
  if (P.fire.x > 0.5) {
    phi.z = max(phi.z, P.fire.y * m);
  } else {
    phi.x = max(phi.x, P.srcProps.x * m);
  }
  phi.y = max(phi.y, P.srcProps.y * m);
  textureStore(dst, c, phi);
}
`

export const RENDER_COMMON = /* wgsl */ `
struct RParams {
  camPos: vec4<f32>,   // position (m), tan of half the vertical field of view
  camFwd: vec4<f32>,   // forward, aspect (width / height)
  camRight: vec4<f32>, // right, frame number
  camUp: vec4<f32>,    // up, exposure (stops)
  sunDir: vec4<f32>,   // toward the sun, phase anisotropy g
  sunCol: vec4<f32>,   // sun colour x intensity, extinction per unit density (1/m)
  skyCol: vec4<f32>,   // sky colour x intensity, single-scattering albedo
  floorCol: vec4<f32>, // floor albedo, radius (m) over which the floor fades to black
  size: vec4<f32>,     // render target width, height
  detail: vec4<f32>,   // weight of detail layer A, of layer B, amount, 1 / (8 x detail scale in m)
  fire: vec4<f32>,     // flame intensity, kelvins per unit of temperature (peak flame K - 300), flame contrast (1 = physical), unused
}
@group(0) @binding(1) var<uniform> R: RParams;

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
`

// 9. Light: optical depth from each point of a half-resolution lattice toward the sun, and straight up (the sky).
export const LIGHT = /* wgsl */ `
${SIM_COMMON}
${RENDER_COMMON}
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var scal: texture_3d<f32>;
@group(0) @binding(4) var lightOut: texture_storage_3d<rgba16float, write>;

fn density(w: vec3<f32>) -> f32 { return textureSampleLevel(scal, samp, uvwOf(w), 0.0).x; }

fn depth(w: vec3<f32>, d: vec3<f32>, ds: f32) -> f32 {
  let te = boxHit(w, d, P.origin.xyz, domainHi()).y;
  var tau = 0.0;
  var t = 0.5 * ds;
  for (var i = 0; i < 256; i++) {
    if (t >= te) { break; }
    tau += density(w + d * t);
    t += ds;
  }
  return tau * ds * R.sunCol.w;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ld = textureDimensions(lightOut);
  if (any(id >= ld)) { return; }
  let ext = vec3<f32>(P.dims.xyz) * P.grid.x;
  let w = P.origin.xyz + (vec3<f32>(id) + 0.5) / vec3<f32>(ld) * ext;
  // Occupancy: the largest density or flame in this cell's block of the grid, plus its neighbours' (one cell each way),
  // so a march can stride through empty space without stepping over the edge of the smoke.
  let n = nCells();
  let b = vec3<i32>(id) * 2;
  var occ = 0.0;
  for (var z = -1; z <= 2; z++) {
    for (var y = -1; y <= 2; y++) {
      for (var x = -1; x <= 2; x++) {
        let q = textureLoad(scal, clamp(b + vec3<i32>(x, y, z), vec3<i32>(0), n - 1), 0);
        occ = max(occ, max(q.x, q.w));
      }
    }
  }
  // Every cell gets its depths, smoke or not: the floor's shadow and sky lookups land in empty cells too.
  let tauSun = depth(w, R.sunDir.xyz, 1.5 * P.grid.x);
  let tauUp = depth(w, vec3<f32>(0.0, 1.0, 0.0), 2.0 * P.grid.x);
  textureStore(lightOut, vec3<i32>(id), vec4<f32>(tauSun, tauUp, occ, 0.0));
}
`

// Flame emission: blackbody colour and brightness (from the table in blackbody.ts) at the flame's radiating
// temperature, which follows the reaction coordinate (the flame field): the peak flame temperature where the reaction
// is fresh, at the base, cooling toward the tips as it runs out (a temperature profile along the reaction, after
// Nguyen et al. 2002). The advected temperature drives buoyancy. The table holds true relative brightness; the
// contrast exponent compresses it (1 = physical).
function fireCommon(bbBinding: number) {
  return /* wgsl */ `
@group(0) @binding(${bbBinding}) var<storage, read> bb: array<vec4<f32>>;
const BB_MIN_K = ${BB_MIN_K}.0;
const BB_MAX_K = ${BB_MAX_K}.0;
fn blackbody(kelvin: f32) -> vec3<f32> {
  let n = arrayLength(&bb);
  let x = clamp((kelvin - BB_MIN_K) / (BB_MAX_K - BB_MIN_K), 0.0, 1.0) * f32(n - 1u);
  let i = u32(floor(x));
  let j = min(i + 1u, n - 1u);
  let c = mix(bb[i], bb[j], x - f32(i));
  // Contrast: keep the colour, raise the luminance to the exponent.
  let lum = max(c.w, 1e-12);
  return c.xyz / lum * pow(lum, R.fire.z);
}
fn emission(s: vec4<f32>) -> vec3<f32> {
  if (s.w <= 1e-3 || R.fire.x <= 0.0) { return vec3<f32>(0.0); }
  let kelvin = 300.0 + pow(clamp(s.w, 0.0, 1.0), 0.3) * R.fire.y;
  return R.fire.x * blackbody(kelvin) * smoothstep(0.0, 0.08, s.w);
}
`
}

// 9b. The fire's light, for the smoke and the floor: its total emitted intensity and its brightness-weighted centre,
// summed on the GPU each step (per workgroup, then across workgroups) and lit as a point light from there.
export const FIRE_SUM = /* wgsl */ `
${SIM_COMMON}
${RENDER_COMMON}
${fireCommon(3)}
@group(0) @binding(2) var scal: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> partial: array<vec4<f32>>;
var<workgroup> sumE: array<vec4<f32>, 64>;
var<workgroup> sumP: array<vec4<f32>, 64>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let c = vec3<i32>(id);
  var e = vec3<f32>(0.0);
  if (all(c < nCells())) { e = emission(textureLoad(scal, c, 0)); }
  let lum = dot(e, vec3<f32>(0.2126, 0.7152, 0.0722));
  sumE[li] = vec4<f32>(e, lum);
  sumP[li] = vec4<f32>(worldOf(vec3<f32>(id) + 0.5) * lum, 0.0);
  workgroupBarrier();
  for (var st = 32u; st > 0u; st = st >> 1u) {
    if (li < st) {
      sumE[li] += sumE[li + st];
      sumP[li] += sumP[li + st];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let k = wid.x + nw.x * (wid.y + nw.y * wid.z);
    partial[2u * k] = sumE[0];
    partial[2u * k + 1u] = sumP[0];
  }
}
`

export const FIRE_SUM_TOTAL = /* wgsl */ `
@group(0) @binding(0) var<uniform> C: vec4<u32>; // x: number of partial sums
@group(0) @binding(1) var<storage, read> partial: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;
var<workgroup> sE: array<vec4<f32>, 256>;
var<workgroup> sP: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var e = vec4<f32>(0.0);
  var p = vec4<f32>(0.0);
  for (var k = li; k < C.x; k += 256u) {
    e += partial[2u * k];
    p += partial[2u * k + 1u];
  }
  sE[li] = e;
  sP[li] = p;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (li < st) {
      sE[li] += sE[li + st];
      sP[li] += sP[li + st];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    result[0] = sE[0];
    result[1] = vec4<f32>(sP[0].xyz / max(sE[0].w, 1e-9), 0.0);
  }
}
`

// 10. The image: march each camera ray through the volume (Hillaire's integration of each step,
// T (1 - exp(-sigma dt)) S), lit by the sun through the light volume with Wrenninge's multiple-scattering
// octaves (a = b = c = 1/2, which conserves energy) and Henyey-Greenstein phase, and by the sky through the
// optical depth above; then a floor that catches the smoke's shadow, fading to black; ACES tone mapping.
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

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

// Phase relative to isotropic (x 4 pi).
fn hg(c: f32, g: f32) -> f32 {
  return (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}
fn ign(p: vec2<f32>) -> f32 { return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715)))); }
// Per-pixel, per-frame white noise for the march's start: interleaved gradient noise has a diagonal structure that
// shows through bright flames.
fn jitterAt(p: vec2<f32>, frame: f32) -> f32 {
  var h = (bitcast<u32>(i32(p.x)) * 0x8da6b343u) ^ (bitcast<u32>(i32(p.y)) * 0xd8163841u) ^ (u32(frame) * 0xcb1ab31fu);
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return f32(h) * (1.0 / 4294967296.0);
}
const FIRE_R2 = 0.06; // m2: softens the fire's point light over the flames' size

// Upres detail at a point: the noise at each layer's advected coordinates, cross-faded, with the contrast that mixing
// two independent noises loses restored (divided by the blend's standard-deviation factor).
fn detail(w: vec3<f32>, uvw: vec3<f32>) -> f32 {
  let a = textureSampleLevel(detA, samp, uvw, 0.0).xyz;
  let b = textureSampleLevel(detB, samp, uvw, 0.0).xyz;
  // A fixed rotation, so the noise lattice lines up with neither the grid nor the camera.
  let rot = mat3x3<f32>(0.8, -0.6, 0.0, 0.36, 0.48, 0.8, -0.48, -0.64, 0.6);
  let na = textureSampleLevel(noiseTex, noiseSamp, rot * (w + a) * R.detail.w, 0.0).x - 0.5;
  let nb = textureSampleLevel(noiseTex, noiseSamp, rot * (w + b) * R.detail.w + vec3<f32>(0.37, 0.61, 0.13), 0.0).x - 0.5;
  let wa = R.detail.x;
  let wb = R.detail.y;
  return (na * wa + nb * wb) / max(sqrt(wa * wa + wb * wb), 1e-3);
}

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
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let ndc = vec2<f32>(fc.x / R.size.x * 2.0 - 1.0, 1.0 - fc.y / R.size.y * 2.0);
  let tanY = R.camPos.w;
  let rd = normalize(R.camFwd.xyz + R.camRight.xyz * ndc.x * tanY * R.camFwd.w + R.camUp.xyz * ndc.y * tanY);
  let ro = R.camPos.xyz;
  let frame = R.camRight.w;
  let jitter = jitterAt(fc.xy, frame);

  let tFloor = select(1e9, -ro.y / rd.y, rd.y < -1e-5);
  let box = boxHit(ro, rd, P.origin.xyz, domainHi());
  let tEnd = min(box.y, tFloor);
  let sun = R.sunDir.xyz;
  let g = R.sunDir.w;
  let cosT = dot(rd, sun);
  let ph0 = hg(cosT, g);
  let ph1 = hg(cosT, 0.5 * g);
  let ph2 = hg(cosT, 0.25 * g);
  let albedo = R.skyCol.w;
  let sigScale = R.sunCol.w;
  let fireVolume = P.grid.x * P.grid.x * P.grid.x; // each cell's emission counts over its volume

  var col = vec3<f32>(0.0);
  var T = 1.0;
  if (box.y > max(box.x, 0.0) && box.x < tFloor) {
    let t0 = max(box.x, 0.0);
    let ds = 0.45 * P.grid.x;
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
      let s4 = textureSampleLevel(scal, samp, uvw, 0.0);
      var d = s4.x * fade;
      let flame = s4.w * fade;
      if (d > 1e-4 || flame > 1e-3) {
        // Upres detail: strongest in thin smoke (the edges and wisps), a third as strong in the dense core; it shapes
        // the flames too.
        var f = 1.0;
        if (R.detail.z > 0.0) {
          let thin = 1.0 - smoothstep(0.02, 0.4, d);
          f = clamp(1.0 + 2.0 * R.detail.z * detail(w, uvw) * (0.35 + 0.65 * thin), 0.0, 2.0);
        }
        d *= f;
        let L = textureSampleLevel(light, samp, uvw, 0.0);
        let tau = L.x;
        let lsun = exp(-tau) * ph0 + 0.5 * exp(-0.5 * tau) * ph1 + 0.25 * exp(-0.25 * tau) * ph2;
        let amb = R.skyCol.xyz * exp(-0.5 * L.y);
        // The fire as a point light at its brightness-weighted centre (softened over its size): I / (4 pi r^2).
        let toFire = fireLight[1].xyz - w;
        let fireE = fireLight[0].xyz * fireVolume / (12.566371 * (dot(toFire, toFire) + FIRE_R2));
        let S = albedo * (R.sunCol.xyz * lsun + amb + fireE);
        let Ts = exp(-d * sigScale * ds);
        col += T * ((1.0 - Ts) * S + emission(vec4<f32>(s4.xyz, flame)) * f * ds);
        T *= Ts;
        if (T < 0.004) { break; }
      }
    }
  }

  if (tFloor < 1e8 && T > 0.004) {
    let pf = ro + rd * tFloor;
    let fade = exp(-dot(pf.xz, pf.xz) / (R.floorCol.w * R.floorCol.w));
    // The smoke's shadow: the light volume's depth at the point where the sun ray from the floor enters the domain.
    let sb = boxHit(pf, sun, P.origin.xyz, domainHi());
    var shadow = 1.0;
    if (sb.y > max(sb.x, 0.0)) {
      let pe = pf + sun * max(sb.x, 0.0) + vec3<f32>(0.0, 0.5 * P.grid.x, 0.0);
      shadow = exp(-textureSampleLevel(light, samp, uvwOf(pe), 0.0).x);
    }
    var skyVis = 1.0;
    let uv = uvwOf(pf + vec3<f32>(0.0, 0.5 * P.grid.x, 0.0));
    if (all(uv.xz >= vec2<f32>(0.0)) && all(uv.xz <= vec2<f32>(1.0))) {
      skyVis = exp(-0.5 * textureSampleLevel(light, samp, uv, 0.0).y);
    }
    // The fire's light on the floor: Lambertian, I cos / r^2 / pi.
    let toFire = fireLight[1].xyz - pf;
    let r2 = dot(toFire, toFire) + FIRE_R2;
    let fireE = fireLight[0].xyz * fireVolume * max(toFire.y, 0.0) / (sqrt(r2) * r2 * 3.14159265);
    let lf = R.floorCol.xyz * (R.sunCol.xyz * max(sun.y, 0.0) * shadow + R.skyCol.xyz * skyVis + fireE);
    col += T * lf * fade;
  }

  let mapped = srgb(aces(col * exp2(R.camUp.w)));
  return vec4<f32>(mapped + (ign(fc.xy + 7.13) - 0.5) / 255.0, 1.0);
}
`
