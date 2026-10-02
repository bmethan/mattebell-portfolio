// WGSL ports of the shared math and sampling of shaders.ts (GLSL), line for line where WGSL allows: the same
// constants, hashes and Sobol' sequence, so a pixel draws the same samples on either renderer.

export const COMMON = /* wgsl */ `
const PI: f32 = 3.14159265358979;
const INV_PI: f32 = 0.31830988618379;
const TWO_PI: f32 = 6.28318530717959;
const ONE_MINUS_EPS: f32 = 0.99999994;

fn sq(x: f32) -> f32 { return x * x; }
fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sat3(x: vec3f) -> vec3f { return clamp(x, vec3f(0.0), vec3f(1.0)); }
fn maxc(c: vec3f) -> f32 { return max(c.r, max(c.g, c.b)); }

// Working space ACEScg (AP1). Matrices from OpenColorIO 2.5 built-in cg-config-v4.0.0_aces-v2.0_ocio-v2.5
// (Bradford D60 <-> D65). Column-major, as in GLSL.
const AP1_TO_REC709 = mat3x3f(
   1.7050510, -0.1302564, -0.0240034,
  -0.6217921,  1.1408048, -0.1289690,
  -0.0832589, -0.0105483,  1.1529723);
const REC709_TO_AP1 = mat3x3f(
   0.6130974,  0.0701937,  0.0206156,
   0.3395231,  0.9163539,  0.1095698,
   0.0473795,  0.0134524,  0.8698146);
// Luminance of an ACEScg color: the Rec.709 luminance row pushed through AP1_TO_REC709.
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.2676014, 0.6743990, 0.0579996)); }

// PCG hashes: Jarzynski and Olano 2020, "Hash Functions for GPU Rendering", JCGT 9(3).
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn pcg3d(vIn: vec3u) -> vec3u {
  var v = vIn * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn u01(x: u32) -> f32 { return f32(x >> 8u) * (1.0 / 16777216.0); }

// Orthonormal basis: Duff et al. 2017, "Building an Orthonormal Basis, Revisited", JCGT 6(1), Listing 3.
struct Basis { b1: vec3f, b2: vec3f }
fn onb(n: vec3f) -> Basis {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return Basis(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y));
}
`

// Owen-scrambled, shuffled Sobol': Burley 2020, "Practical Hash-based Owen Scrambling", JCGT 9(4). The direction
// numbers (dimensions 1-3) come from the scene uniform (sobol), as in the GLSL tracer.
export const SAMPLING = /* wgsl */ `
fn laineKarrasPermutation(xIn: u32, seed: u32) -> u32 {
  var x = xIn + seed;
  x ^= x * 0x6c50b47cu;
  x ^= x * 0xb82f1e52u;
  x ^= x * 0xc7afe638u;
  x ^= x * 0x8d22f6e6u;
  return x;
}
fn nestedUniformScramble(x: u32, seed: u32) -> u32 { return reverseBits(laineKarrasPermutation(reverseBits(x), seed)); }
fn hashCombine(seed: u32, v: u32) -> u32 { return seed ^ (v + (seed << 6u) + (seed >> 2u)); }
// Walks only the set bits of the index.
fn sobol4d(index: u32) -> vec4u {
  var X = vec4u(reverseBits(index), 0u, 0u, 0u);
  var bits = index;
  var bit = 0u;
  loop {
    if (bits == 0u) { break; }
    if ((bits & 1u) != 0u) {
      let d = scene.sobol[bit];
      X = vec4u(X.x, X.y ^ d.x, X.z ^ d.y, X.w ^ d.z);
    }
    bits >>= 1u;
    bit++;
  }
  return X;
}
fn shuffledScrambledSobol4d(indexIn: u32, seed: u32) -> vec4f {
  let index = nestedUniformScramble(indexIn, seed);
  let X = sobol4d(index);
  return vec4f(
    u01(nestedUniformScramble(X.x, hashCombine(seed, 0u))),
    u01(nestedUniformScramble(X.y, hashCombine(seed, 1u))),
    u01(nestedUniformScramble(X.z, hashCombine(seed, 2u))),
    u01(nestedUniformScramble(X.w, hashCombine(seed, 3u))));
}
// One 4D point per (bounce, group). Burley 2020 ("Use in a Path Tracer"): hash the seed as it advances.
fn sample4(sampleIndex: u32, pixSeed: u32, bounce: i32, group: i32) -> vec4f {
  return shuffledScrambledSobol4d(sampleIndex, pcg(hashCombine(pixSeed, u32(bounce * 8 + group))));
}
`
