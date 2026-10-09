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

// The source: a sphere resting on the floor, its edge one cell soft, broken into puffs by fast-drifting noise.
fn sourceMask(w: vec3<f32>) -> f32 {
  let r = P.src.w;
  let d = length(w - P.src.xyz);
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

// 4. Divergence of the face velocities at each cell (the floor face is solid).
export const DIVERGENCE = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> div: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= nCells())) { return; }
  let a = textureLoad(vel, c, 0);
  let v0 = select(a.y, 0.0, c.y == 0);
  div[cellIndex(c)] = ((textureLoad(vel, c + X, 0).x - a.x) + (textureLoad(vel, c + Y, 0).y - v0) + (textureLoad(vel, c + Z, 0).z - a.z)) / P.grid.x;
}
`

// 5. Pressure: red-black Gauss-Seidel with over-relaxation on the 7-point Laplacian, lap(p) = div / dt (unit
// density). Open sides and top: p = 0 outside. Solid floor: no flow through it, so the floor neighbour drops out.
export const SOR = /* wgsl */ `
${SIM_COMMON}
@group(0) @binding(1) var<uniform> S: vec4<u32>; // x: the colour updated in this pass
@group(0) @binding(2) var<storage, read> div: array<f32>;
@group(0) @binding(3) var<storage, read_write> p: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  let n = nCells();
  if (any(c >= n)) { return; }
  if (((id.x + id.y + id.z) & 1u) != S.x) { return; }
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
  let pn = (sum - div[i] * P.grid.x * P.grid.x / P.grid.y) / cnt;
  p[i] = mix(p[i], pn, P.decay.z);
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
  let m = sourceMask(worldOf(vec3<f32>(id) + 0.5));
  phi.x = max(phi.x, P.srcProps.x * m);
  phi.y = max(phi.y, P.srcProps.y * m);
  textureStore(dst, c, vec4<f32>(phi.xy, 0.0, 0.0));
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
// 1 inside, falling to 0 over the last few cells at the open sides and top (the box's edges never show).
fn edgeFade(uvw: vec3<f32>) -> f32 {
  let e = min(min(uvw.x, 1.0 - uvw.x), min(uvw.z, 1.0 - uvw.z));
  return smoothstep(0.0, 0.08, e) * (1.0 - smoothstep(0.85, 1.0, uvw.y));
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
  let tauSun = depth(w, R.sunDir.xyz, 1.5 * P.grid.x);
  let tauUp = depth(w, vec3<f32>(0.0, 1.0, 0.0), 2.0 * P.grid.x);
  textureStore(lightOut, vec3<i32>(id), vec4<f32>(tauSun, tauUp, 0.0, 0.0));
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
  let jitter = ign(fc.xy + 5.588238 * frame);

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

  var col = vec3<f32>(0.0);
  var T = 1.0;
  if (box.y > max(box.x, 0.0) && box.x < tFloor) {
    let t0 = max(box.x, 0.0);
    let ds = 0.6 * P.grid.x;
    let steps = min(i32(ceil((tEnd - t0) / ds)), 512);
    for (var i = 0; i < steps; i++) {
      let t = t0 + (f32(i) + jitter) * ds;
      if (t > tEnd) { break; }
      let w = ro + rd * t;
      let uvw = uvwOf(w);
      let d = textureSampleLevel(scal, samp, uvw, 0.0).x * edgeFade(uvw);
      if (d > 1e-4) {
        let L = textureSampleLevel(light, samp, uvw, 0.0);
        let tau = L.x;
        let lsun = exp(-tau) * ph0 + 0.5 * exp(-0.5 * tau) * ph1 + 0.25 * exp(-0.25 * tau) * ph2;
        let amb = R.skyCol.xyz * exp(-0.5 * L.y);
        let S = albedo * (R.sunCol.xyz * lsun + amb);
        let Ts = exp(-d * sigScale * ds);
        col += T * (1.0 - Ts) * S;
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
    let lf = R.floorCol.xyz * (R.sunCol.xyz * max(sun.y, 0.0) * shadow + R.skyCol.xyz * skyVis);
    col += T * lf * fade;
  }

  let mapped = srgb(aces(col * exp2(R.camUp.w)));
  return vec4<f32>(mapped + (ign(fc.xy + 7.13) - 0.5) / 255.0, 1.0);
}
`
