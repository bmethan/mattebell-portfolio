// GLSL ES 3.00 (WebGL2) sources for the lookdev lab: a progressive path tracer that renders a subset of
// OpenPBR Surface v1.1.1 in the ACEScg working space.
//
// Every model below was researched from primary sources and independently re-verified (constants checked
// against the papers and reference code, compile-tested in WebGL2, furnace-tested numerically). Where this
// implementation deliberately deviates from a reference, the comment says so.

const HEADER = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;
`

export const VERT = /* glsl */ `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`

// ------------------------------------------------------------------------------------------------------------
// Shared math, color space, and sampling sequences.
// ------------------------------------------------------------------------------------------------------------
const COMMON = /* glsl */ `
const float PI = 3.14159265358979;
const float INV_PI = 0.31830988618379;
const float TWO_PI = 6.28318530717959;
const float ONE_MINUS_EPS = 0.99999994;

float sq(float x) { return x * x; }
float sat(float x) { return clamp(x, 0.0, 1.0); }
vec3 sat3(vec3 x) { return clamp(x, 0.0, 1.0); }
float maxc(vec3 c) { return max(c.r, max(c.g, c.b)); }

// Working space ACEScg (AP1). Matrices from OpenColorIO 2.5 built-in cg-config-v4.0.0_aces-v2.0_ocio-v2.5
// (Bradford D60 <-> D65). GLSL mat3 is column-major.
const mat3 AP1_TO_REC709 = mat3(
   1.7050510, -0.1302564, -0.0240034,
  -0.6217921,  1.1408048, -0.1289690,
  -0.0832589, -0.0105483,  1.1529723);
const mat3 REC709_TO_AP1 = mat3(
   0.6130974,  0.0701937,  0.0206156,
   0.3395231,  0.9163539,  0.1095698,
   0.0473795,  0.0134524,  0.8698146);
// Luminance of an ACEScg color: the Rec.709 luminance row pushed through AP1_TO_REC709.
float lum(vec3 c) { return dot(c, vec3(0.2676014, 0.6743990, 0.0579996)); }

// PCG hashes: Jarzynski and Olano 2020, "Hash Functions for GPU Rendering", JCGT 9(3).
uint pcg(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
float u01(uint x) { return float(x >> 8u) * (1.0 / 16777216.0); }

// Owen-scrambled, shuffled Sobol': Burley 2020, "Practical Hash-based Owen Scrambling", JCGT 9(4).
uint reverseBits32(uint x) {
  x = ((x & 0xaaaaaaaau) >> 1) | ((x & 0x55555555u) << 1);
  x = ((x & 0xccccccccu) >> 2) | ((x & 0x33333333u) << 2);
  x = ((x & 0xf0f0f0f0u) >> 4) | ((x & 0x0f0f0f0fu) << 4);
  x = ((x & 0xff00ff00u) >> 8) | ((x & 0x00ff00ffu) << 8);
  return (x >> 16) | (x << 16);
}
uint laineKarrasPermutation(uint x, uint seed) {
  x += seed;
  x ^= x * 0x6c50b47cu;
  x ^= x * 0xb82f1e52u;
  x ^= x * 0xc7afe638u;
  x ^= x * 0x8d22f6e6u;
  return x;
}
uint nestedUniformScramble(uint x, uint seed) { return reverseBits32(laineKarrasPermutation(reverseBits32(x), seed)); }
uint hashCombine(uint seed, uint v) { return seed ^ (v + (seed << 6) + (seed >> 2)); }
// Sobol' direction numbers for dimensions 1-3 (Burley 2020 supplement, sobol.cpp), one uvec4 per bit
// (x, y, z = dims 1, 2, 3), uploaded by the host as a uniform array. Dynamic indexing into a uniform buffer is
// cheap; the same table as a const array is copied into scratch registers at every inlined call site by
// Direct3D's shader compiler, which was a large share of the compile time.
uniform uvec4 uSobol[32];
// Walks only the set bits of the index; a data-dependent loop is never unrolled at compile time.
uvec4 sobol4d(uint index) {
  uvec4 X = uvec4(reverseBits32(index), 0u, 0u, 0u);
  uint bits = index;
  int bit = 0;
  while (bits != 0u) {
    if ((bits & 1u) != 0u) X.yzw ^= uSobol[bit].xyz;
    bits >>= 1u;
    bit++;
  }
  return X;
}
vec4 shuffledScrambledSobol4d(uint index, uint seed) {
  index = nestedUniformScramble(index, seed);
  uvec4 X = sobol4d(index);
  X.x = nestedUniformScramble(X.x, hashCombine(seed, 0u));
  X.y = nestedUniformScramble(X.y, hashCombine(seed, 1u));
  X.z = nestedUniformScramble(X.z, hashCombine(seed, 2u));
  X.w = nestedUniformScramble(X.w, hashCombine(seed, 3u));
  return vec4(u01(X.x), u01(X.y), u01(X.z), u01(X.w));
}
// One 4D point per (bounce, group). Burley 2020 ("Use in a Path Tracer"): hash the seed as it advances.
vec4 sample4(uint sampleIndex, uint pixSeed, int bounce, int group) {
  return shuffledScrambledSobol4d(sampleIndex, pcg(hashCombine(pixSeed, uint(bounce * 8 + group))));
}

// Orthonormal basis: Duff et al. 2017, "Building an Orthonormal Basis, Revisited", JCGT 6(1), Listing 3.
void onb(vec3 n, out vec3 b1, out vec3 b2) {
  float s = n.z >= 0.0 ? 1.0 : -1.0;
  float a = -1.0 / (s + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
  b2 = vec3(b, s + n.y * n.y * a, -n.y);
}
`

// ------------------------------------------------------------------------------------------------------------
// GGX microfacet specular (local frame, n = +z; i = view direction, o = light direction).
// ------------------------------------------------------------------------------------------------------------
const MICROFACET = /* glsl */ `
// NDF, Heitz 2014 (JCGT 3(2)); isotropic case is Walter et al. 2007 eq. 33.
float D_GGX(vec3 m, vec2 alpha) {
  if (m.z <= 0.0) return 0.0;
  vec2 he = m.xy / alpha;
  float d = dot(he, he) + m.z * m.z;
  return 1.0 / (PI * alpha.x * alpha.y * d * d);
}
// Smith masking in the stable form T(w) = sqrt(ax^2 wx^2 + ay^2 wy^2 + wz^2) (Heitz 2014).
float T_GGX(vec3 w, vec2 alpha) { vec2 aw = alpha * w.xy; return sqrt(dot(aw, aw) + w.z * w.z); }
float G1_GGX(vec3 w, vec2 alpha) { float z = abs(w.z); return 2.0 * z / (z + T_GGX(w, alpha)); }
// Height-correlated G2 = 1 / (1 + Lambda(i) + Lambda(o)).
float G2_GGX(vec3 i, vec3 o, vec2 alpha) {
  float zi = abs(i.z), zo = abs(o.z);
  return 2.0 * zi * zo / (T_GGX(i, alpha) * zo + T_GGX(o, alpha) * zi);
}

// Visible-normal sampling with spherical caps: Dupuy and Benyoub 2023, CGF 42(8) (HPG 2023), Listings 1 and 3.
// Used for the albedo table precompute.
vec3 sampleVNDF_SphericalCap(vec2 u, vec3 wi, vec2 alpha) {
  vec3 wiStd = normalize(vec3(wi.xy * alpha, wi.z));
  float phi = TWO_PI * u.x;
  float z = (1.0 - u.y) * (1.0 + wiStd.z) - wiStd.z;
  float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  vec3 h = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z) + wiStd;
  return normalize(vec3(h.xy * alpha, h.z));
}

// Bounded VNDF sampling, reflection only: Tokuyoshi and Eto 2024, "Bounded VNDF Sampling for the Smith-GGX
// BRDF", PACMCGIT 7(1) (I3D 2024), Listings 1-2, Eq. 6-7. Returns the reflected direction o.
float boundedVNDF_k(vec3 i, vec2 alpha) {
  float a = clamp(min(alpha.x, alpha.y), 0.0, 1.0);
  float s = 1.0 + length(i.xy);
  float a2 = a * a, s2 = s * s;
  return (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
}
vec3 sampleGGXReflection_Bounded(vec2 rand, vec3 i, vec2 alpha) {
  vec3 iStd = normalize(vec3(i.xy * alpha, i.z));
  float phi = TWO_PI * rand.x;
  float k = boundedVNDF_k(i, alpha);
  float lowerBound = i.z > 0.0 ? -k * iStd.z : -iStd.z;
  float z = lowerBound * rand.y + (1.0 - rand.y);
  float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  vec3 mStd = iStd + vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  vec3 m = normalize(vec3(mStd.xy * alpha, mStd.z));
  return 2.0 * dot(i, m) * m - i;
}
float pdfGGXReflection_Bounded(vec3 i, vec3 o, vec2 alpha) {
  vec3 m = normalize(i + o);
  float ndf = D_GGX(m, alpha);
  vec2 ai = alpha * i.xy;
  float len2 = dot(ai, ai);
  float t = sqrt(len2 + i.z * i.z);
  if (i.z >= 0.0) return ndf / (2.0 * (boundedVNDF_k(i, alpha) * i.z + t));
  return ndf * (t - i.z) / (2.0 * len2);
}

// Exact unpolarized dielectric Fresnel, Walter et al. 2007 eq. 22; eta = eta_t / eta_i.
float fresnelDielectric(float cosThetaI, float eta) {
  float c = abs(cosThetaI);
  float g2 = eta * eta - 1.0 + c * c;
  if (g2 < 0.0) return 1.0;
  float g = sqrt(g2);
  float A = (g - c) / (g + c);
  float B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
  return 0.5 * A * A * (1.0 + B * B);
}
float f0FromEta(float eta) { return sq((eta - 1.0) / (eta + 1.0)); }

// OpenPBR metal Fresnel, "F82-tint" (Kutz, Hasan, Edmondson 2021, extending Hoffman 2019).
vec3 fresnelSchlick(vec3 F0, float mu) {
  float m = clamp(1.0 - mu, 0.0, 1.0);
  float m2 = m * m;
  return F0 + (1.0 - F0) * (m2 * m2 * m);
}
vec3 fresnelF82Tint(float mu, vec3 F0, vec3 tint) {
  const float MU_BAR = 1.0 / 7.0;
  float b1 = 1.0 - MU_BAR;
  float b3 = b1 * b1 * b1;
  vec3 a = fresnelSchlick(F0, MU_BAR) * (1.0 - tint) / (MU_BAR * b3 * b3);
  float x = clamp(mu, 0.0, 1.0);
  float y = 1.0 - x;
  float y3 = y * y * y;
  return max(fresnelSchlick(F0, x) - a * x * (y3 * y3), vec3(0.0)); // max(): implementation guard
}
// Hemispherical average of F82-tint: Schlick part 1/21 (exact), dip term -a/126 (exact Beta integral).
vec3 favgF82Tint(vec3 F0, vec3 tint) {
  const float MU_BAR = 1.0 / 7.0;
  float b1 = 1.0 - MU_BAR;
  float b3 = b1 * b1 * b1;
  vec3 a = fresnelSchlick(F0, MU_BAR) * (1.0 - tint) / (MU_BAR * b3 * b3);
  return F0 + (1.0 - F0) * (1.0 / 21.0) - a * (1.0 / 126.0);
}
`

// Albedo table layout: 3D texture over (mu_o, roughness, x), with x = sqrt(F0) = (eta - 1)/(eta + 1) in [0, X_MAX].
// Samples sit exactly on the grid endpoints (mu = 0 and 1, r = 0 and 1) so grazing angles are interpolated,
// not clamped to the first texel center.
const TABLE_CONSTS = /* glsl */ `
const float E_X_MAX = 0.5;   // eta up to 3
const vec3 E_DIMS = vec3(32.0, 32.0, 16.0);
float etaCoord(float eta) { return clamp(((eta - 1.0) / (eta + 1.0)) / E_X_MAX, 0.0, 1.0); }
vec3 eTableCoord(float mu, float r, float eta) {
  return (vec3(clamp(mu, 0.0, 1.0), clamp(r, 0.0, 1.0), etaCoord(eta)) * (E_DIMS - 1.0) + 0.5) / E_DIMS;
}
`

// Kulla and Conty 2017 / Turquin 2019 directional albedo tables, computed once at startup.
// R: E_ss with F = 1. G: E_ss with exact dielectric Fresnel at this IOR. B: average Fresnel F_avg(eta).
export const E_TABLE_FRAG = /* glsl */ `${HEADER}
${COMMON}
${MICROFACET}
${TABLE_CONSTS}
uniform vec2 uSize;
uniform float uLayer;   // normalized x coordinate of this slice, k / (layers - 1)
uniform int uSamples;   // 2048 VNDF samples; from a uniform so the loop is not unrolled at compile time
uniform int uFresnelSamples; // 256
out vec4 outColor;
void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5) / (uSize - 1.0); // texel i holds mu = i / (N - 1)
  float mu = max(uv.x, 1e-4);
  float r = uv.y;
  vec2 alpha = vec2(max(r * r, 1e-4));
  float x = uLayer * E_X_MAX;
  float eta = (1.0 + x) / (1.0 - x);
  vec3 wo = vec3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
  int N = uSamples;
  float e1 = 0.0, ed = 0.0;
  for (int i = 0; i < N; i++) {
    vec2 u = vec2((float(i) + 0.5) / float(N), u01(reverseBits32(uint(i))));
    vec3 m = sampleVNDF_SphericalCap(u, wo, alpha);
    vec3 wi = reflect(-wo, m);
    if (wi.z <= 0.0) continue;
    float w = G2_GGX(wo, wi, alpha) / G1_GGX(wo, alpha); // f cos / pdf for F = 1
    e1 += w;
    ed += w * fresnelDielectric(dot(wo, m), eta);
  }
  float favg = 0.0;
  int M = uFresnelSamples;
  for (int j = 0; j < M; j++) {
    float c = (float(j) + 0.5) / float(M);
    favg += fresnelDielectric(c, eta) * c;
  }
  favg *= 2.0 / float(M);
  outColor = vec4(e1 / float(N), ed / float(N), favg, 1.0);
}
`

// ------------------------------------------------------------------------------------------------------------
// EON diffuse: Portsmouth, Kutz, Hill, "EON: A Practical Energy-Preserving Rough Diffuse BRDF", JCGT 14(1),
// 2025 (rev. 2026-02-04), Listings 1-4; reference main.glsl (jamportz/EON-diffuse). Uses the paper's signed
// dot(wi, wo). (MaterialX clamps it, which gains up to ~12% energy at grazing view in a furnace test.)
// ------------------------------------------------------------------------------------------------------------
const EON = /* glsl */ `
const float CONSTANT1_FON = 0.5 - 2.0 / (3.0 * PI);
const float CONSTANT2_FON = 2.0 / 3.0 - 28.0 / (15.0 * PI);
const float EON_EPS = 1.0e-7;

float E_FON_exact(float mu, float r) {
  float AF = 1.0 / (1.0 + CONSTANT1_FON * r);
  float BF = r * AF;
  mu = clamp(mu, -1.0, 1.0);
  float Si = sqrt(max(0.0, 1.0 - mu * mu));
  float G = Si * (acos(mu) - Si * mu) + (2.0 / 3.0) * (Si * mu * (1.0 + Si + Si * Si) / (1.0 + Si) - Si);
  return AF + (BF * INV_PI) * G;
}
float E_FON_avg(float r) { return (1.0 + CONSTANT2_FON * r) / (1.0 + CONSTANT1_FON * r); }

vec3 f_EON(vec3 rho, float r, vec3 wi, vec3 wo) {
  float mu_i = wi.z, mu_o = wo.z;
  if (mu_i < EON_EPS || mu_o < EON_EPS) return vec3(0.0);
  float s = dot(wi, wo) - mu_i * mu_o;
  float sovertF = s > 0.0 ? s / max(mu_i, mu_o) : s;
  float AF = 1.0 / (1.0 + CONSTANT1_FON * r);
  vec3 f_ss = (rho * INV_PI) * AF * (1.0 + r * sovertF);
  float EFo = E_FON_exact(mu_o, r);
  float EFi = E_FON_exact(mu_i, r);
  float avgEF = AF * (1.0 + CONSTANT2_FON * r);
  vec3 rho_ms = (rho * rho) * avgEF / (vec3(1.0) - rho * (1.0 - avgEF));
  vec3 f_ms = (rho_ms * INV_PI) * max(EON_EPS, 1.0 - EFo) * max(EON_EPS, 1.0 - EFi) / max(EON_EPS, 1.0 - avgEF);
  return f_ss + f_ms;
}
vec3 E_EON(vec3 rho, float r, float mu) {
  float EF = E_FON_exact(mu, r);
  float avgEF = E_FON_avg(r);
  vec3 rho_ms = (rho * rho) * avgEF / (vec3(1.0) - rho * (1.0 - avgEF));
  return rho * EF + rho_ms * (1.0 - EF);
}

mat3 orthonormal_basis_ltc(vec3 w) {
  float lenSqr = dot(w.xy, w.xy);
  vec3 X = lenSqr > 0.0 ? vec3(w.x, w.y, 0.0) * inversesqrt(lenSqr) : vec3(1.0, 0.0, 0.0);
  vec3 Y = vec3(-X.y, X.x, 0.0);
  return mat3(X, Y, vec3(0.0, 0.0, 1.0));
}
void ltc_coeffs(float mu, float r, out float a, out float b, out float c, out float d) {
  mu = clamp(mu, 0.0, 1.0);
  a = 1.0 + r * (0.303392 + (-0.518982 + 0.111709 * mu) * mu + (-0.276266 + 0.335918 * mu) * r);
  b = r * (-1.16407 + 1.15859 * mu + (0.150815 - 0.150105 * mu) * r) / (mu * mu * mu - 1.43545);
  c = 1.0 + r * (0.20013 + (-0.506373 + 0.261777 * mu) * mu);
  d = r * (0.540852 + (-1.01625 + 0.475392 * mu) * mu) / (-1.0743 + (0.0725628 + mu) * mu);
}
vec4 cltc_sample(vec3 wo, float r, float u1, float u2) {
  float a, b, c, d;
  ltc_coeffs(wo.z, r, a, b, c, d);
  float R = sqrt(u1);
  float phi = TWO_PI * u2;
  float x = R * cos(phi);
  float y = R * sin(phi);
  float vz = 1.0 / sqrt(d * d + 1.0);
  float s = 0.5 * (1.0 + vz);
  x = -mix(sqrt(max(0.0, 1.0 - y * y)), x, s);
  vec3 wh = vec3(x, y, sqrt(max(1.0 - (x * x + y * y), 0.0)));
  float pdf_wh = wh.z / (PI * s);
  vec3 wi = vec3(a * wh.x + b * wh.z, c * wh.y, d * wh.x + wh.z);
  float len = length(wi);
  float detM = c * (a - b * d);
  float pdf_wi = pdf_wh * len * len * len / detM;
  wi = normalize(orthonormal_basis_ltc(wo) * wi);
  return vec4(wi, pdf_wi);
}
float cltc_pdf(vec3 wo, vec3 wi_local, float r) {
  vec3 wi = transpose(orthonormal_basis_ltc(wo)) * wi_local;
  float a, b, c, d;
  ltc_coeffs(wo.z, r, a, b, c, d);
  float detM = c * (a - b * d);
  vec3 wh = vec3(c * (wi.x - b * wi.z), (a - b * d) * wi.y, -c * (d * wi.x - a * wi.z));
  float lenSqr = dot(wh, wh);
  float vz = 1.0 / sqrt(d * d + 1.0);
  float s = 0.5 * (1.0 + vz);
  return detM * detM / (lenSqr * lenSqr) * max(wh.z, 0.0) / (PI * s);
}
vec3 uniform_lobe_sample(float u1, float u2) {
  float sinTheta = sqrt(max(0.0, 1.0 - u1 * u1));
  float phi = TWO_PI * u2;
  return vec3(sinTheta * cos(phi), sinTheta * sin(phi), u1);
}
float eon_uniform_prob(float mu, float r) {
  float rw = r > 0.0 ? pow(r, 0.1) : 0.0;
  return rw * (0.162925 + (-0.372058 + (0.538233 - 0.290822 * mu) * mu) * mu);
}
vec3 sample_EON(vec3 wo, float r, float u1, float u2) {
  float P_u = eon_uniform_prob(wo.z, r);
  if (P_u > 0.0 && u1 < P_u) return uniform_lobe_sample(u1 / P_u, u2);
  return cltc_sample(wo, r, (u1 - P_u) / (1.0 - P_u), u2).xyz;
}
float pdf_EON(vec3 wo, vec3 wi, float r) {
  if (wo.z < EON_EPS || wi.z < EON_EPS) return 0.0;
  float P_u = eon_uniform_prob(wo.z, r);
  return P_u * (1.0 / TWO_PI) + (1.0 - P_u) * cltc_pdf(wo, wi, r);
}
`

// ------------------------------------------------------------------------------------------------------------
// Fuzz: Zeltner, Burley, Chiang 2022, "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines"
// (SIGGRAPH Talks), via the MaterialX analytic fits (S. Hill, MaterialX PR #1825). Local frame, n = +z.
// ------------------------------------------------------------------------------------------------------------
const FUZZ = /* glsl */ `
float fuzzDirAlbedo(float x, float y) {
  float s = y * (0.0206607 + 1.58491 * y) / (0.0379424 + y * (1.32227 + y));
  float m = y * (-0.193854 + y * (-1.14885 + y * (1.7932 - 0.95943 * y * y))) / (0.046391 + y);
  float o = y * (0.000654023 + (-0.0207818 + 0.119681 * y) * y) / (1.26264 + y * (-1.92021 + y));
  return max(0.0, exp(-0.5 * sq((x - m) / s)) / (s * sqrt(2.0 * PI)) + o); // max(): fit dips to -2e-5
}
float fuzzLtcAInv(float x, float y) {
  return (2.58126 * x + 0.813703 * y) * y / (1.0 + 0.310327 * x * x + 2.60994 * x * y);
}
float fuzzLtcBInv(float x, float y) {
  return sqrt(1.0 - x) * (y - 1.0) * y * y * y / (0.0000254053 + 1.71228 * x - 1.71506 * x * y + 1.34174 * y * y);
}
mat3 fuzzLtcBasis(vec3 V) {
  vec3 X = vec3(V.xy, 0.0);
  float l2 = dot(X, X);
  if (l2 > 0.0) {
    X *= inversesqrt(l2);
    return mat3(X, vec3(-X.y, X.x, 0.0), vec3(0.0, 0.0, 1.0));
  }
  return mat3(1.0);
}
// D(L | V): a normalized PDF over solid angle, so f_fuzz cos = fuzz_color * E_fuzz * D.
float fuzzD(vec3 V, vec3 L, float r) {
  float NdotV = clamp(V.z, 1e-8, 1.0);
  r = clamp(r, 0.01, 1.0);
  vec3 w = transpose(fuzzLtcBasis(V)) * L;
  float aInv = fuzzLtcAInv(NdotV, r);
  float bInv = fuzzLtcBInv(NdotV, r);
  vec3 wo = vec3(aInv * w.x + bInv * w.z, aInv * w.y, w.z);
  float l2 = dot(wo, wo);
  return max(wo.z, 0.0) * INV_PI * sq(aInv / l2);
}
vec3 sampleFuzz(vec3 V, float r, vec2 u) {
  float NdotV = clamp(V.z, 1e-8, 1.0);
  r = clamp(r, 0.01, 1.0);
  float phi = TWO_PI * u.x;
  float ct = sqrt(u.y);
  float st = sqrt(1.0 - u.y);
  vec3 c = vec3(cos(phi) * st, sin(phi) * st, ct);
  float aInv = fuzzLtcAInv(NdotV, r);
  float bInv = fuzzLtcBInv(NdotV, r);
  vec3 w = normalize(vec3(c.x / aInv - c.z * bInv / aInv, c.y / aInv, c.z));
  return fuzzLtcBasis(V) * w;
}
`

// ------------------------------------------------------------------------------------------------------------
// Thin-film iridescence: Belcour and Barla 2017, ACM TOG 36(4), Article 65, as implemented for
// KHR_materials_iridescence in the Khronos glTF Sample Renderer (iridescence.glsl). Two changes, both labeled:
// the interference colors are white balanced (the Gaussian fit integrates to equal-energy white; the Khronos
// code sends it through a D65 matrix, tinting it ~20% warm), and the result is converted to ACEScg.
// ------------------------------------------------------------------------------------------------------------
const THIN_FILM = /* glsl */ `
const mat3 XYZ_TO_REC709 = mat3(
   3.2404542, -0.9692660,  0.0556434,
  -1.5371385,  1.8760108, -0.2040259,
  -0.4985314,  0.0415560,  1.0572252);
const vec3 E_WHITE_IN_REC709 = vec3(1.2047843, 0.9483008, 0.9088427); // row sums of XYZ_TO_REC709

float F_SchlickTF(float f0, float VdotH) { float x = clamp(1.0 - VdotH, 0.0, 1.0); float x2 = x * x; return f0 + (1.0 - f0) * (x * x2 * x2); }
vec3 F_SchlickTF(vec3 f0, float VdotH) { float x = clamp(1.0 - VdotH, 0.0, 1.0); float x2 = x * x; return f0 + (1.0 - f0) * (x * x2 * x2); }
vec3 Fresnel0ToIor(vec3 f0) { vec3 s = sqrt(f0); return (vec3(1.0) + s) / (vec3(1.0) - s); }
vec3 IorToFresnel0(vec3 t, float i) { vec3 q = (t - vec3(i)) / (t + vec3(i)); return q * q; }
float IorToFresnel0(float t, float i) { return sq((t - i) / (t + i)); }

vec3 evalSensitivity(float OPD, vec3 shift) {
  float phase = 2.0 * PI * OPD * 1.0e-9;
  vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
  vec3 pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
  vec3 var = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);
  vec3 xyz = val * sqrt(2.0 * PI * var) * cos(pos * phase + shift) * exp(-sq(phase) * var);
  xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * sq(phase));
  xyz /= 1.0685e-7;
  return REC709_TO_AP1 * ((XYZ_TO_REC709 * xyz) / E_WHITE_IN_REC709);
}

vec3 evalIridescence(float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0) {
  float iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
  float sinTheta2Sq = sq(outsideIOR / iridescenceIor) * (1.0 - sq(cosTheta1));
  float cosTheta2Sq = 1.0 - sinTheta2Sq;
  if (cosTheta2Sq < 0.0) return vec3(1.0);
  float cosTheta2 = sqrt(cosTheta2Sq);
  float R0 = IorToFresnel0(iridescenceIor, outsideIOR);
  float R12 = F_SchlickTF(R0, cosTheta1);
  float T121 = 1.0 - R12;
  float phi12 = iridescenceIor < outsideIOR ? PI : 0.0;
  float phi21 = PI - phi12;
  vec3 baseIOR = Fresnel0ToIor(clamp(baseF0, 0.0, 0.9999));
  vec3 R1 = IorToFresnel0(baseIOR, iridescenceIor);
  vec3 R23 = F_SchlickTF(R1, cosTheta2);
  vec3 phi23 = vec3(baseIOR.x < iridescenceIor ? PI : 0.0, baseIOR.y < iridescenceIor ? PI : 0.0, baseIOR.z < iridescenceIor ? PI : 0.0);
  float OPD = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
  vec3 phi = vec3(phi21) + phi23;
  vec3 R123 = clamp(R12 * R23, 1e-5, 0.9999);
  vec3 r123 = sqrt(R123);
  vec3 Rs = sq(T121) * R23 / (vec3(1.0) - R123);
  vec3 I = R12 + Rs;
  vec3 Cm = Rs - T121;
  for (int m = 1; m <= 2; ++m) {
    Cm *= r123;
    I += Cm * 2.0 * evalSensitivity(float(m) * OPD, float(m) * phi);
  }
  return max(I, vec3(0.0));
}
// OpenPBR adapter: thickness in micrometers, film IOR clamped to >= 1 (as MaterialX), evaluated at dot(V, H)
// per sampled microfacet (Belcour and Barla, Sec. 3 Eq. 1).
vec3 thinFilmF(float cosTheta, float filmIor, float thicknessUm, vec3 baseF0) {
  return evalIridescence(1.0, max(filmIor, 1.0), clamp(cosTheta, 0.0, 1.0), thicknessUm * 1000.0, baseF0);
}
`

// ------------------------------------------------------------------------------------------------------------
// OpenPBR Surface v1.1.1 layering (ASWF, 2026-04-17), following the spec's "reduction to a mixture of lobes"
// as realized by the reference graph (open_pbr_surface.mtlx):
//   f_diel    = f_specR + (1 - E_specR) f_diffuse
//   f_base    = mix(f_diel, specular_weight * f_metal(F82), metalness)
//   f_coated  = C f_coat + (1 - C E_coat) * darkening * mix(1, coat_color, C) * f_base
//   f_surface = F f_fuzz + (1 - F E_fuzz) f_coated
// Specular lobes use Turquin 2019 energy compensation (eq. 14, F_ms = F_avg).
// ------------------------------------------------------------------------------------------------------------
const OPENPBR = /* glsl */ `
struct Mat {
  float base_weight; vec3 base_color; float base_metalness; float base_diffuse_roughness;
  float specular_weight; vec3 specular_color; float specular_roughness; float specular_ior;
  float coat_weight; vec3 coat_color; float coat_roughness; float coat_ior; float coat_darkening;
  float fuzz_weight; vec3 fuzz_color; float fuzz_roughness;
  float thin_film_weight; float thin_film_thickness; float thin_film_ior;
};

struct Surf {
  vec3 baseColor; float baseWeight; float metal; float diffRough;
  float specW; vec3 specColor; float specRough; float eta;
  float coatW; float coatRough; float coatIor;
  float fuzzW; vec3 fuzzColor; float fuzzRough;
  float tfW; float tfThick; float tfIor;
  vec2 aS; vec2 aC;
  float compD; vec3 compM; float compC;
  float EspecR; float Ecoat; float Efuzz;
  vec3 EspecR3; // per-channel specular albedo that attenuates the diffuse (colored when a thin film is present)
  vec3 tBase;   // everything multiplying f_base below fuzz and coat
  vec4 p;       // lobe selection weights: diffuse, base specular, coat, fuzz
};

uniform sampler3D uETable;
uniform int uMultiScatter;

// Coat roughens the base specular (spec, Coat > Roughening). Products instead of pow() for pow(0, y) safety.
float openpbrCoatedSpecRoughness(float rB, float rC, float C) {
  float rB2 = rB * rB, rC2 = rC * rC;
  float t = min(1.0, rB2 * rB2 + 2.0 * rC2 * rC2);
  return mix(rB, sqrt(sqrt(t)), C);
}
// Base dielectric IOR ratio with coat, TIR fix and specular_weight modulation (spec; reference node modulated_eta_s).
float openpbrSpecularEta(float specIor, float coatIor, float C, float specWeight) {
  float r = specIor / coatIor;
  float tirFix = r > 1.0 ? r : coatIor / specIor;
  float etaS = mix(specIor, tirFix, C);
  float Fs = sq((etaS - 1.0) / (etaS + 1.0));
  float e = sign(etaS - 1.0) * sqrt(clamp(specWeight * Fs, 0.0, 0.99999));
  return (1.0 + e) / (1.0 - e);
}
// Darkening times absorption under the coat, as in the reference graph (K_r with E_F ~ F0).
vec3 openpbrCoatBaseFactor(vec3 baseColor, float specWeight, float M, float C, vec3 coatColor, float coatIor, float coatDarkening) {
  float F0c = sq((coatIor - 1.0) / (coatIor + 1.0));
  float K = 1.0 - (1.0 - F0c) / (coatIor * coatIor);
  vec3 Ebase = mix(baseColor, baseColor * specWeight, M);
  vec3 Delta = vec3(1.0 - K) / (vec3(1.0) - Ebase * K);
  return mix(vec3(1.0), Delta, C * coatDarkening) * mix(vec3(1.0), coatColor, C);
}

Surf setupSurf(Mat m, vec3 wo) {
  Surf s;
  s.baseColor = max(m.base_color, vec3(0.0));
  s.baseWeight = max(m.base_weight, 0.0);
  s.metal = sat(m.base_metalness);
  s.diffRough = sat(m.base_diffuse_roughness);
  s.specW = max(m.specular_weight, 0.0);
  s.specColor = max(m.specular_color, vec3(0.0));
  s.coatW = sat(m.coat_weight);
  s.coatRough = sat(m.coat_roughness);
  s.coatIor = max(m.coat_ior, 1.0);
  s.fuzzW = sat(m.fuzz_weight);
  s.fuzzColor = max(m.fuzz_color, vec3(0.0));
  s.fuzzRough = clamp(m.fuzz_roughness, 0.01, 1.0);
  s.tfW = m.thin_film_thickness > 0.0 ? sat(m.thin_film_weight) : 0.0;
  s.tfThick = m.thin_film_thickness;
  s.tfIor = m.thin_film_ior;
  s.specRough = openpbrCoatedSpecRoughness(sat(m.specular_roughness), s.coatRough, s.coatW);
  s.aS = vec2(max(s.specRough * s.specRough, 1e-4));
  s.aC = vec2(max(s.coatRough * s.coatRough, 1e-4));
  s.eta = openpbrSpecularEta(max(m.specular_ior, 1.0), s.coatIor, s.coatW, s.specW);

  float mu = clamp(wo.z, 1e-4, 1.0);
  bool ms = uMultiScatter == 1;
  vec3 F0m = s.baseWeight * s.baseColor;

  // textureLod: implicit-derivative fetches inside loops force Direct3D to unroll them.
  vec3 tS = textureLod(uETable, eTableCoord(mu, s.specRough, s.eta), 0.0).rgb;
  float Ess = max(tS.r, 1e-4);
  s.compD = ms ? 1.0 + tS.b * (1.0 - Ess) / Ess : 1.0;
  s.compM = ms ? 1.0 + favgF82Tint(F0m, s.specColor) * (1.0 - Ess) / Ess : vec3(1.0);
  s.EspecR = sat(tS.g * s.compD);
  // A thin film changes how much the specular reflects, per wavelength; interference conserves energy, so the
  // base receives the complement. Scale the tabulated albedo by the film-to-plain Fresnel ratio at the view angle.
  s.EspecR3 = vec3(s.EspecR);
  if (s.tfW > 0.0) {
    vec3 ratio = thinFilmF(mu, s.tfIor, s.tfThick, vec3(f0FromEta(s.eta))) / max(fresnelDielectric(mu, s.eta), 1e-4);
    s.EspecR3 = sat3(s.EspecR * mix(vec3(1.0), ratio, s.tfW));
  }

  vec3 tC = textureLod(uETable, eTableCoord(mu, s.coatRough, s.coatIor), 0.0).rgb;
  float EssC = max(tC.r, 1e-4);
  s.compC = ms ? 1.0 + tC.b * (1.0 - EssC) / EssC : 1.0;
  s.Ecoat = sat(tC.g * s.compC);

  s.Efuzz = s.fuzzW > 0.0 ? fuzzDirAlbedo(mu, s.fuzzRough) : 0.0;
  float tFuzz = 1.0 - s.fuzzW * s.Efuzz;
  vec3 coatBase = openpbrCoatBaseFactor(s.baseColor, s.specW, s.metal, s.coatW, m.coat_color, s.coatIor, m.coat_darkening);
  s.tBase = tFuzz * (1.0 - s.coatW * s.Ecoat) * coatBase;

  float tb = lum(s.tBase);
  vec3 diffAlb = s.baseWeight * E_EON(sat3(s.baseColor), s.diffRough, mu);
  s.p = vec4(
    tb * (1.0 - s.metal) * max(lum((1.0 - s.EspecR3) * diffAlb), 0.0),
    tb * ((1.0 - s.metal) * lum(s.EspecR3) * max(lum(s.specColor), 0.05) + s.metal * s.specW * max(lum(F0m), 0.05)),
    tFuzz * s.coatW * s.Ecoat,
    s.fuzzW * s.Efuzz * max(lum(s.fuzzColor), 0.05));
  return s;
}

// filt: 0 all lobes, 1 diffuse only, 2 everything but diffuse (first-hit light path expressions for AOVs).
vec4 lobeProbs(Surf s, int filt) {
  vec4 p = s.p;
  if (filt == 1) p = vec4(p.x, 0.0, 0.0, 0.0);
  if (filt == 2) p.x = 0.0;
  float sum = p.x + p.y + p.z + p.w;
  return sum > 0.0 ? p / sum : vec4(0.0);
}

// Returns the mixture pdf; fD and fS receive f * cos(theta_i), split into diffuse and everything else.
float evalSurf(Surf s, vec3 wo, vec3 wi, int filt, out vec3 fD, out vec3 fS) {
  fD = vec3(0.0);
  fS = vec3(0.0);
  if (wi.z <= 0.0 || wo.z <= 0.0) return 0.0;
  vec4 p = lobeProbs(s, filt);
  vec3 h = normalize(wo + wi);
  float voh = sat(dot(wo, h));
  float pdf = 0.0;

  if (filt != 2 && s.metal < 1.0) {
    vec3 d = s.baseWeight * f_EON(sat3(s.baseColor), s.diffRough, wi, wo) * wi.z;
    fD = s.tBase * (1.0 - s.metal) * (1.0 - s.EspecR3) * d;
    pdf += p.x * pdf_EON(wo, wi, s.diffRough);
  }
  if (filt != 1) {
    float g = G2_GGX(wo, wi, s.aS) * D_GGX(h, s.aS) / (4.0 * wo.z);
    vec3 Fd = s.specColor * fresnelDielectric(voh, s.eta) * s.compD;
    vec3 Fm = s.specW * fresnelF82Tint(voh, s.baseWeight * s.baseColor, s.specColor) * s.compM;
    if (s.tfW > 0.0) {
      // One interference evaluation, with the base reflectance blended by metalness (exact for the pure
      // dielectric and pure metal cases, and it keeps a single inlined copy of the film code).
      vec3 Ftf = thinFilmF(voh, s.tfIor, s.tfThick, mix(vec3(f0FromEta(s.eta)), s.baseWeight * s.baseColor, s.metal));
      Fd = mix(Fd, s.specColor * Ftf * s.compD, s.tfW);
      Fm = mix(Fm, s.specW * Ftf * s.compM, s.tfW);
    }
    fS += s.tBase * mix(Fd, Fm, s.metal) * g;
    pdf += p.y * pdfGGXReflection_Bounded(wo, wi, s.aS);

    if (s.coatW > 0.0) {
      float gc = G2_GGX(wo, wi, s.aC) * D_GGX(h, s.aC) / (4.0 * wo.z);
      fS += vec3((1.0 - s.fuzzW * s.Efuzz) * s.coatW * fresnelDielectric(voh, s.coatIor) * s.compC * gc);
      pdf += p.z * pdfGGXReflection_Bounded(wo, wi, s.aC);
    }
    if (s.fuzzW > 0.0) {
      float D = fuzzD(wo, wi, s.fuzzRough);
      fS += s.fuzzW * s.fuzzColor * s.Efuzz * D;
      pdf += p.w * D;
    }
  }
  return pdf;
}

bool sampleSurf(Surf s, vec3 wo, int filt, vec3 u, out vec3 wi) {
  vec4 p = lobeProbs(s, filt);
  if (u.z < p.x) wi = sample_EON(wo, s.diffRough, u.x, u.y);
  else if (u.z < p.x + p.y) wi = sampleGGXReflection_Bounded(u.xy, wo, s.aS);
  else if (u.z < p.x + p.y + p.z) wi = sampleGGXReflection_Bounded(u.xy, wo, s.aC);
  else if (p.w > 0.0) wi = sampleFuzz(wo, s.fuzzRough, u.xy);
  else return false;
  return wi.z > 0.0;
}

vec3 albedoAOV(Mat m) {
  vec3 a = mix(m.base_color * m.base_weight, m.base_color, m.base_metalness);
  a *= mix(vec3(1.0), m.coat_color, m.coat_weight);
  return mix(a, m.fuzz_color, m.fuzz_weight);
}
`

// ------------------------------------------------------------------------------------------------------------
// Rectangular softboxes: solid-angle sampling of spherical rectangles, Urena, Fajardo, King 2013, CGF 32(4)
// (EGSR 2013), Appendix B, hardened for float32. The paper's pdf uses the angle-sum solid angle, which carries an
// absolute error of a few ulp(pi); for a small, distant light that error dwarfs the true solid angle and the
// light comes out far too bright. The pdf therefore uses the van Oosterom-Strackee (1983) triangle solid angle,
// accurate to ~1e-7 relative in float32. The angle sum still drives the sampling map, as in the paper.
// ------------------------------------------------------------------------------------------------------------
const LIGHTS = /* glsl */ `
uniform vec3 uLightCorner[3];
uniform vec3 uLightU[3];
uniform vec3 uLightV[3];
uniform vec3 uLightRadiance[3];

bool lightOn(int k) { return maxc(uLightRadiance[k]) > 0.0; }
bool facesLight(int k, vec3 o) { return dot(o - uLightCorner[k], cross(uLightU[k], uLightV[k])) > 0.0; }

struct SphQuad {
  vec3 o, x, y, z;
  float z0, z0sq, x0, y0, y0sq, x1, y1, y1sq;
  float b0, b1, b0sq, k, S, Spdf;
};
float safeAsin(float x) { return asin(clamp(x, -1.0, 1.0)); }
float angleBetween(vec3 v1, vec3 v2) {
  return dot(v1, v2) < 0.0 ? PI - 2.0 * safeAsin(length(v1 + v2) * 0.5) : 2.0 * safeAsin(length(v2 - v1) * 0.5);
}
bool sphQuadInit(out SphQuad q, vec3 s, vec3 ex, vec3 ey, vec3 o) {
  q.o = o;
  float exl = length(ex), eyl = length(ey);
  q.x = ex / exl;
  q.y = ey / eyl;
  q.z = cross(q.x, q.y);
  vec3 d = s - o;
  q.z0 = dot(d, q.z);
  if (q.z0 > 0.0) { q.z = -q.z; q.z0 = -q.z0; }
  q.z0sq = q.z0 * q.z0;
  q.x0 = dot(d, q.x);
  q.y0 = dot(d, q.y);
  q.x1 = q.x0 + exl;
  q.y1 = q.y0 + eyl;
  q.y0sq = q.y0 * q.y0;
  q.y1sq = q.y1 * q.y1;
  q.S = 0.0; q.Spdf = 0.0; q.b0 = 0.0; q.b1 = 0.0; q.b0sq = 0.0; q.k = 0.0;
  if (-q.z0 <= 1e-6 * max(exl, eyl)) return false;
  vec3 v00 = vec3(q.x0, q.y0, q.z0), v01 = vec3(q.x0, q.y1, q.z0);
  vec3 v10 = vec3(q.x1, q.y0, q.z0), v11 = vec3(q.x1, q.y1, q.z0);
  vec3 n0 = normalize(vec3(0.0, q.z0, -q.y0));
  vec3 n1 = normalize(vec3(-q.z0, 0.0, q.x1));
  vec3 n2 = normalize(vec3(0.0, -q.z0, q.y1));
  vec3 n3 = normalize(vec3(q.z0, 0.0, -q.x0));
  float g0 = angleBetween(-n0, n1);
  float g1 = angleBetween(-n1, n2);
  float g2 = angleBetween(-n2, n3);
  float g3 = angleBetween(-n3, n0);
  q.b0 = n0.z;
  q.b1 = n2.z;
  q.b0sq = q.b0 * q.b0;
  q.k = TWO_PI - g2 - g3;
  q.S = g0 + g1 - q.k;
  float N = exl * eyl * (-q.z0);
  float l00 = length(v00), l10 = length(v10), l11 = length(v11), l01 = length(v01);
  float D1 = l00 * l10 * l11 + dot(v00, v10) * l11 + dot(v00, v11) * l10 + dot(v10, v11) * l00;
  float D2 = l00 * l11 * l01 + dot(v00, v11) * l01 + dot(v00, v01) * l11 + dot(v11, v01) * l00;
  q.Spdf = 2.0 * atan(N, D1) + 2.0 * atan(N, D2);
  return q.S > 0.0 && q.Spdf > 0.0;
}
vec3 sphQuadSample(SphQuad q, float u, float v) {
  float au = u * q.S + q.k;
  float sa = sin(au);
  sa = abs(sa) < 1e-7 ? (sa < 0.0 ? -1e-7 : 1e-7) : sa;
  float fu = (cos(au) * q.b0 - q.b1) / sa;
  float cu = (fu > 0.0 ? 1.0 : -1.0) * inversesqrt(fu * fu + q.b0sq);
  cu = clamp(cu, -ONE_MINUS_EPS, ONE_MINUS_EPS);
  float xu = -(cu * q.z0) / sqrt(max(0.0, 1.0 - cu * cu));
  xu = clamp(xu, q.x0, q.x1);
  float dd = sqrt(xu * xu + q.z0sq);
  float h0 = q.y0 / sqrt(dd * dd + q.y0sq);
  float h1 = q.y1 / sqrt(dd * dd + q.y1sq);
  float hv = h0 + v * (h1 - h0), hv2 = hv * hv;
  float yv = hv2 < 1.0 - 1e-6 ? (hv * dd) / sqrt(1.0 - hv2) : q.y1;
  return q.o + xu * q.x + yv * q.y + q.z0 * q.z;
}
// Below this solid angle, fall back to area sampling (pbrt-v4 MinSphericalSampleArea).
const float MIN_SPHERICAL_SAMPLE_AREA = 1e-4;
bool sampleLight(int k, vec3 o, vec2 u, out vec3 wi, out float dist, out float pdf) {
  pdf = 0.0; wi = vec3(0.0, 0.0, 1.0); dist = 0.0;
  if (!facesLight(k, o)) return false;
  SphQuad q;
  bool ok = sphQuadInit(q, uLightCorner[k], uLightU[k], uLightV[k], o);
  vec3 p;
  if (ok && q.Spdf > MIN_SPHERICAL_SAMPLE_AREA) {
    p = sphQuadSample(q, u.x, u.y);
    pdf = 1.0 / q.Spdf;
  } else {
    p = uLightCorner[k] + u.x * uLightU[k] + u.y * uLightV[k];
    vec3 nL = cross(uLightU[k], uLightV[k]);
    float area = length(nL);
    vec3 w = p - o;
    float d2 = dot(w, w);
    float cosL = abs(dot(nL / area, w)) * inversesqrt(max(d2, 1e-30));
    pdf = cosL > 0.0 ? d2 / (area * cosL) : 0.0;
  }
  vec3 w = p - o;
  dist = length(w);
  if (dist <= 0.0 || pdf <= 0.0) { pdf = 0.0; return false; }
  wi = w / dist;
  return true;
}
// Solid angle of light k seen from o: the same van Oosterom-Strackee two-triangle sum as SphQuad.Spdf, computed
// directly from the corners (rotation invariant, so it equals the local-frame value), without the sampler setup.
float rectSolidAngle(int k, vec3 o) {
  vec3 ex = uLightU[k], ey = uLightV[k];
  vec3 a = uLightCorner[k] - o, b = a + ex, c = b + ey, d = a + ey;
  float N = abs(dot(a, cross(ex, ey)));
  float la = length(a), lb = length(b), lc = length(c), ld = length(d);
  float D1 = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
  float D2 = la * lc * ld + dot(a, c) * ld + dot(a, d) * lc + dot(c, d) * la;
  return 2.0 * atan(N, D1) + 2.0 * atan(N, D2);
}
float lightPdf(int k, vec3 o, vec3 wi, float dist) {
  if (!facesLight(k, o)) return 0.0;
  float S = rectSolidAngle(k, o);
  if (S > MIN_SPHERICAL_SAMPLE_AREA) return 1.0 / S;
  vec3 nL = cross(uLightU[k], uLightV[k]);
  float area = length(nL);
  float cosL = abs(dot(nL / area, wi));
  return cosL > 0.0 ? dist * dist / (area * cosL) : 0.0;
}
float intersectRect(vec3 ro, vec3 rd, int k, float tMax) {
  vec3 s = uLightCorner[k], ex = uLightU[k], ey = uLightV[k];
  vec3 n = cross(ex, ey);
  float denom = dot(n, rd);
  if (denom >= 0.0) return -1.0; // one-sided: visible only from the emitting side
  float t = dot(s - ro, n) / denom;
  if (!(t > 1e-4 && t < tMax)) return -1.0;
  vec3 d = (ro - s) + t * rd;
  vec2 uv = vec2(dot(d, ex) / dot(ex, ex), dot(d, ey) / dot(ey, ey));
  return (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) ? t : -1.0;
}
// Power heuristic, beta = 2 (Veach and Guibas 1995).
float powerHeuristic(float f, float g) {
  float f2 = f * f, g2 = g * g;
  if (f2 > 1e30) return 1.0;
  return f2 + g2 > 0.0 ? f2 / (f2 + g2) : 0.0;
}
`

// Triangle meshes for the hero models: a bounding volume hierarchy and triangles in float textures, traversed with
// a short stack in object space so the model can turn without rebuilding anything. Only the MESH variant of the
// tracer includes it; the reference-ball scene compiles without it.
const MESH_GLSL = /* glsl */ `
precision highp usampler2D;
uniform sampler2D uBvh;      // 4 texels per inner node: each child's bmin.xyz + ref, bmax.xyz (see the builder)
uniform sampler2D uTriPos;   // 3 texels per triangle: v0, v1 - v0, v2 - v0
uniform usampler2D uTriNrm;  // 1 texel per triangle: three octahedral vertex normals (snorm16 x2), material slot
uniform mat3 uModelRot;      // object to world rotation
uniform vec3 uModelPos;
uniform float uModelScale;
uniform int uMaxNodeVisits;  // from a uniform so the traversal loop is never unrolled

const int TEX_W_SHIFT = 11;  // textures are 2048 texels wide
ivec2 texAt(int i) { return ivec2(i & 2047, i >> TEX_W_SHIFT); }

// Octahedral normal decoding: Cigolle et al. 2014, "A Survey of Efficient Representations for Independent Unit
// Vectors", JCGT 3(2).
vec3 octDecode(uint u) {
  vec2 f = unpackSnorm2x16(u);
  vec3 n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
  float t = max(-n.z, 0.0);
  n.xy += vec2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
  return normalize(n);
}

// Entry distance of a box, or 1e30 when the ray misses it or it lies beyond tBest.
float boxEnter(vec3 bmin, vec3 bmax, vec3 o, vec3 invD, float tBest) {
  vec3 t0 = (bmin - o) * invD, t1 = (bmax - o) * invD;
  vec3 tn = min(t0, t1), tf = max(t0, t1);
  float tEnter = max(max(tn.x, tn.y), max(tn.z, 0.0));
  float tExit = min(min(tf.x, tf.y), tf.z);
  return tEnter <= tExit && tEnter < tBest ? tEnter : 1e30;
}

const int BVH_STACK = 32;
// Closest hit before tMax (or, with anyHit, the first hit found). Ray/triangle: Moller and Trumbore 1997.
// Returns the distance (the object-space ray is scaled so it equals the world distance) or -1.
// cur packs (index << 5) | count: a leaf of count triangles, or with count 0 the inner node at index. Each inner
// node holds both children's boxes, so only entered children are visited, nearer first; the farther one waits on
// the stack with its entry distance and is skipped if a closer hit has been found by then. One loop body handles
// leaves and inner nodes, so the triangle test is compiled once.
float traceMesh(vec3 roW, vec3 rdW, float tMax, bool anyHit, out int triHit, out vec2 bary) {
  triHit = -1;
  bary = vec2(0.0);
  mat3 toObj = transpose(uModelRot);
  vec3 o = toObj * (roW - uModelPos) / uModelScale;
  vec3 d = toObj * rdW / uModelScale;
  vec3 dSafe = vec3(abs(d.x) < 1e-20 ? 1e-20 : d.x, abs(d.y) < 1e-20 ? 1e-20 : d.y, abs(d.z) < 1e-20 ? 1e-20 : d.z);
  vec3 invD = 1.0 / dSafe;
  float tBest = tMax;
  int stackRef[BVH_STACK];
  float stackT[BVH_STACK];
  int sp = 0;
  int cur = 0; // the root, an inner node
  for (int visit = 0; visit < uMaxNodeVisits; visit++) {
    int count = cur & 31;
    int index = cur >> 5;
    if (count == 0) {
      vec4 la = texelFetch(uBvh, texAt(4 * index), 0);
      vec4 lb = texelFetch(uBvh, texAt(4 * index + 1), 0);
      vec4 ra = texelFetch(uBvh, texAt(4 * index + 2), 0);
      vec4 rb = texelFetch(uBvh, texAt(4 * index + 3), 0);
      float tl = boxEnter(la.xyz, lb.xyz, o, invD, tBest);
      float tr = boxEnter(ra.xyz, rb.xyz, o, invD, tBest);
      bool hitL = tl < 1e30, hitR = tr < 1e30;
      if (hitL && hitR) {
        bool leftFirst = tl <= tr;
        if (sp < BVH_STACK) {
          stackRef[sp] = int(leftFirst ? ra.w : la.w);
          stackT[sp] = leftFirst ? tr : tl;
          sp++;
        }
        cur = int(leftFirst ? la.w : ra.w);
        continue;
      }
      if (hitL || hitR) {
        cur = int(hitL ? la.w : ra.w);
        continue;
      }
    } else {
      for (int i = 0; i < count; i++) {
        int tri = index + i;
        vec3 v0 = texelFetch(uTriPos, texAt(3 * tri), 0).xyz;
        vec3 e1 = texelFetch(uTriPos, texAt(3 * tri + 1), 0).xyz;
        vec3 e2 = texelFetch(uTriPos, texAt(3 * tri + 2), 0).xyz;
        vec3 pv = cross(d, e2);
        float det = dot(e1, pv);
        if (det == 0.0) continue;
        float inv = 1.0 / det;
        vec3 tv = o - v0;
        float u = dot(tv, pv) * inv;
        if (u < 0.0 || u > 1.0) continue;
        vec3 qv = cross(tv, e1);
        float v = dot(d, qv) * inv;
        if (v < 0.0 || u + v > 1.0) continue;
        float t = dot(e2, qv) * inv;
        if (t > 0.0 && t < tBest) {
          tBest = t;
          triHit = tri;
          bary = vec2(u, v);
        }
      }
      if (anyHit && triHit >= 0) return tBest;
    }
    // Next: the nearest deferred child still in front of the closest hit.
    bool found = false;
    while (sp > 0) {
      sp--;
      if (stackT[sp] < tBest) {
        cur = stackRef[sp];
        found = true;
        break;
      }
    }
    if (!found) break;
  }
  return triHit >= 0 ? tBest : -1.0;
}
`

function traceFrag(mesh: boolean) {
  return /* glsl */ `${HEADER}${mesh ? '#define MESH 1\n' : ''}
${COMMON}
${MICROFACET}
${TABLE_CONSTS}
${EON}
${FUZZ}
${THIN_FILM}
${OPENPBR}
${LIGHTS}

uniform sampler2D uPrev;
uniform vec2 uResolution;
uniform int uSppDone;
uniform int uSppNew;
uniform vec3 uCamPos;
uniform vec3 uCamFwd;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec2 uTanHalf;
uniform vec3 uEnv;
uniform int uFurnace;
uniform int uPass;
uniform float uIndirectClamp;
#ifdef MESH
// Material slots: 0 gray ball, 1 chrome, 2 hero, 3 floor, 4-9 the model's other parts.
uniform vec4 uBall[3]; // center, radius (0 = absent)
uniform Mat uMat[10];
#else
uniform vec3 uBallX;
uniform Mat uMat[4];
#endif

out vec4 outColor;

// Loop bounds come from uniforms so the Direct3D compiler cannot unroll the path loop (compile time).
uniform int uMaxBounces;
uniform int uNumLights;

#ifndef MESH
float ballX(int i) { return i == 0 ? uBallX.x : (i == 1 ? uBallX.y : uBallX.z); }
#endif

// Ray/sphere with the precision improvements of Haines, Guenther, Akenine-Moller, Ray Tracing Gems ch. 7.
float intersectSphere(vec3 ro, vec3 rd, vec3 center, float tMax) {
  vec3 f = ro - center;
  float bp = -dot(f, rd);
  vec3 l = f + bp * rd;
  float disc = 1.0 - dot(l, l);
  if (disc < 0.0) return -1.0;
  float c = dot(f, f) - 1.0;
  float q = bp + (bp > 0.0 ? 1.0 : -1.0) * sqrt(disc);
  if (q == 0.0) return -1.0;
  float t0 = c / q, t1 = q;
  float tn = min(t0, t1), tf = max(t0, t1);
  if (tn > 0.0 && tn < tMax) return tn;
  if (tf > 0.0 && tf < tMax) return tf;
  return -1.0;
}

// Self-intersection offset: Waechter and Binder, Ray Tracing Gems ch. 6, Listing 6-1.
vec3 offsetRay(vec3 p, vec3 n) {
  const float ORIGIN = 1.0 / 32.0;
  const float FLOAT_SCALE = 1.0 / 65536.0;
  const float INT_SCALE = 256.0;
  ivec3 of_i = ivec3(INT_SCALE * n);
  vec3 p_i = intBitsToFloat(floatBitsToInt(p) + ivec3(
    p.x < 0.0 ? -of_i.x : of_i.x,
    p.y < 0.0 ? -of_i.y : of_i.y,
    p.z < 0.0 ? -of_i.z : of_i.z));
  return vec3(
    abs(p.x) < ORIGIN ? p.x + FLOAT_SCALE * n.x : p_i.x,
    abs(p.y) < ORIGIN ? p.y + FLOAT_SCALE * n.y : p_i.y,
    abs(p.z) < ORIGIN ? p.z + FLOAT_SCALE * n.z : p_i.z);
}

// n: shading normal; ng: geometric normal on the side the ray arrived from (they differ only on meshes).
struct Hit { float t; vec3 n; vec3 ng; int mat; int light; };

#ifdef MESH
${MESH_GLSL}
float intersectBall(vec3 ro, vec3 rd, vec4 s, float tMax) {
  if (s.w <= 0.0) return -1.0;
  float t = intersectSphere((ro - s.xyz) / s.w, rd, vec3(0.0), tMax / s.w);
  return t > 0.0 ? t * s.w : -1.0;
}
#endif

Hit intersect(vec3 ro, vec3 rd, bool withLights) {
  Hit h;
  h.t = 1e30; h.mat = -1; h.light = -1; h.n = vec3(0.0, 1.0, 0.0); h.ng = h.n;
#ifdef MESH
  for (int i = 0; i < 3; i++) {
    vec4 s = uBall[i];
    float t = intersectBall(ro, rd, s, h.t);
    if (t > 0.0) { h.t = t; h.n = normalize(ro + rd * t - s.xyz); h.ng = h.n; h.mat = i; }
  }
  int tri;
  vec2 bc;
  float tm = traceMesh(ro, rd, h.t, false, tri, bc);
  if (tm > 0.0) {
    h.t = tm;
    uvec4 nn = texelFetch(uTriNrm, texAt(tri), 0);
    vec3 e1 = texelFetch(uTriPos, texAt(3 * tri + 1), 0).xyz;
    vec3 e2 = texelFetch(uTriPos, texAt(3 * tri + 2), 0).xyz;
    vec3 ng = normalize(uModelRot * cross(e1, e2));
    vec3 ns = normalize(uModelRot * ((1.0 - bc.x - bc.y) * octDecode(nn.x) + bc.x * octDecode(nn.y) + bc.y * octDecode(nn.z)));
    // Two-sided: both normals face the ray. An interpolated normal that would put the viewer below the surface
    // falls back to the facet's own.
    if (dot(ng, rd) > 0.0) ng = -ng;
    if (dot(ns, ng) < 0.0) ns = -ns;
    if (dot(ns, -rd) <= 1e-4) ns = ng;
    h.n = ns;
    h.ng = ng;
    h.mat = int(nn.w);
  }
#else
  for (int i = 0; i < 3; i++) {
    vec3 c = vec3(ballX(i), 1.0, 0.0);
    float t = intersectSphere(ro, rd, c, h.t);
    if (t > 0.0) { h.t = t; h.n = normalize(ro + rd * t - c); h.ng = h.n; h.mat = i; }
  }
#endif
  if (uFurnace == 0 && rd.y < 0.0) {
    float t = -ro.y / rd.y;
    if (t > 0.0 && t < h.t) { h.t = t; h.n = vec3(0.0, 1.0, 0.0); h.ng = h.n; h.mat = 3; }
  }
  if (withLights) {
    for (int k = 0; k < uNumLights; k++) {
      if (!lightOn(k)) continue;
      float t = intersectRect(ro, rd, k, h.t);
      if (t > 0.0) { h.t = t; h.light = k; h.mat = -1; }
    }
  }
  return h;
}

bool occluded(vec3 ro, vec3 rd, float tMax) {
#ifdef MESH
  for (int i = 0; i < 3; i++) {
    if (intersectBall(ro, rd, uBall[i], tMax) > 0.0) return true;
  }
  int tri;
  vec2 bc;
  if (traceMesh(ro, rd, tMax, true, tri, bc) > 0.0) return true;
#else
  for (int i = 0; i < 3; i++) {
    if (intersectSphere(ro, rd, vec3(ballX(i), 1.0, 0.0), tMax) > 0.0) return true;
  }
#endif
  if (uFurnace == 0 && rd.y < 0.0) {
    float t = -ro.y / rd.y;
    if (t > 0.0 && t < tMax) return true;
  }
  return false;
}

Mat getMat(int i) {
  Mat m;
  if (i == 0) m = uMat[0];
  else if (i == 1) m = uMat[1];
  else if (i == 2) m = uMat[2];
#ifdef MESH
  else if (i == 3) m = uMat[3];
  else if (i == 4) m = uMat[4];
  else if (i == 5) m = uMat[5];
  else if (i == 6) m = uMat[6];
  else if (i == 7) m = uMat[7];
  else if (i == 8) m = uMat[8];
  else m = uMat[9];
#else
  else m = uMat[3];
#endif
  if (uFurnace == 1) {
    // White furnace: every albedo at 1. An energy-conserving material must vanish.
    m.base_weight = 1.0;
    m.base_color = vec3(1.0);
    m.specular_weight = 1.0;
    m.specular_color = vec3(1.0);
    m.coat_color = vec3(1.0);
    m.fuzz_color = vec3(1.0);
  }
  return m;
}

// Firefly control on indirect light only (Cycles-style split), hue preserving. Direct light stays unbiased.
vec3 clampIndirect(vec3 c) {
  float m = maxc(c);
  return (uIndirectClamp > 0.0 && m > uIndirectClamp) ? c * (uIndirectClamp / m) : c;
}
bool nonFinite(float x) { return (floatBitsToUint(x) & 0x7f800000u) == 0x7f800000u; }

void main() {
  ivec2 pix = ivec2(gl_FragCoord.xy);
  uint pixSeed = pcg3d(uvec3(uvec2(pix), 0x9e37u)).x;
  vec3 acc = vec3(0.0);
  float accMask = 0.0;

  for (int sIdx = 0; sIdx < uSppNew; sIdx++) {
    uint sampleIndex = uint(uSppDone + sIdx);
    vec2 jitter = sample4(sampleIndex, pixSeed, 0, 7).xy;
    vec2 ndc = ((gl_FragCoord.xy - 0.5 + jitter) / uResolution) * 2.0 - 1.0;
    vec3 ro = uCamPos;
    vec3 rd = normalize(uCamFwd + ndc.x * uTanHalf.x * uCamRight + ndc.y * uTanHalf.y * uCamUp);

    vec3 L = vec3(0.0);
    vec3 beta = vec3(1.0);
    float mask = 0.0;
    float prevPdf = 0.0;
    vec3 prevP = ro;

    for (int depth = 0; depth < uMaxBounces; depth++) {
      Hit h = intersect(ro, rd, depth > 0);

      if (h.light >= 0) {
        // Softbox reached by BSDF sampling (lights are invisible to camera rays). MIS vs light sampling.
        float pl = lightPdf(h.light, prevP, rd, h.t);
        vec3 c = beta * uLightRadiance[h.light] * powerHeuristic(prevPdf, pl);
        L += depth >= 2 ? clampIndirect(c) : c;
        break;
      }
      if (h.mat < 0) {
        if (!(depth == 0 && uPass >= 3)) {
          vec3 c = beta * uEnv;
          L += depth >= 2 ? clampIndirect(c) : c;
        }
        break;
      }

      vec3 p = ro + rd * h.t;
      vec3 n = h.n;
      Mat m = getMat(h.mat);

      if (depth == 0) {
        mask = h.mat != 3 ? 1.0 : 0.0; // everything but the floor
        if (uPass == 3) { L = albedoAOV(m); break; }
        if (uPass == 4) { L = n * 0.5 + 0.5; break; }
      }
      int filt = depth == 0 ? (uPass == 1 ? 1 : (uPass == 2 ? 2 : 0)) : 0;

      vec3 t1, t2;
      onb(n, t1, t2);
      vec3 wo = vec3(dot(-rd, t1), dot(-rd, t2), dot(-rd, n));
      if (wo.z <= 1e-6) break;
      Surf s = setupSurf(m, wo);
      vec3 po = offsetRay(p, h.ng); // off the true surface, on the side the ray came from

      // One pass over the vertex's sampled directions: k < uNumLights is next-event estimation toward softbox k
      // (solid-angle sampling); k == uNumLights is the BSDF sample that continues the path. Sharing the loop
      // keeps a single inlined copy of evalSurf, which is most of the shader's compile cost on Direct3D.
      bool last = depth == uMaxBounces - 1;
      bool continued = false;
      for (int k = 0; k <= uNumLights; k++) {
        bool isLight = k < uNumLights;
        if (!isLight && last) break; // the continuation ray from the last vertex is never traced
        if (isLight && !lightOn(k)) continue;
        vec4 u = sample4(sampleIndex, pixSeed, depth, k);
        vec3 dirW, wi;
        float ldist = 0.0, lpdf = 0.0;
        if (isLight) {
          if (!sampleLight(k, po, u.xy, dirW, ldist, lpdf)) continue;
          wi = vec3(dot(dirW, t1), dot(dirW, t2), dot(dirW, n));
          if (wi.z <= 0.0) continue;
        } else {
          if (!sampleSurf(s, wo, filt, u.xyz, wi)) break;
          dirW = normalize(t1 * wi.x + t2 * wi.y + n * wi.z);
        }
        vec3 fD, fS;
        float bpdf = evalSurf(s, wo, wi, filt, fD, fS);
        vec3 f = fD + fS;
        if (isLight) {
          if (maxc(f) <= 0.0) continue;
          if (occluded(po, dirW, ldist * (1.0 - 1e-4))) continue;
          // At the last vertex the BSDF-sampled ray is never traced, so light sampling takes the full weight.
          float wl = last ? 1.0 : powerHeuristic(lpdf, bpdf);
          vec3 c = beta * f * uLightRadiance[k] * (wl / lpdf);
          L += depth >= 1 ? clampIndirect(c) : c;
        } else {
          if (bpdf <= 0.0) break;
          beta *= f / bpdf;
          if (maxc(beta) <= 0.0) break;
          if (depth >= 3) {
            float q = min(maxc(beta), 0.95);
            if (u.w >= q) break;
            beta /= q;
          }
          prevPdf = bpdf;
          prevP = po;
          rd = dirW;
          ro = po;
          continued = true;
        }
      }
      if (!continued) break;
    }
    if (nonFinite(L.r) || nonFinite(L.g) || nonFinite(L.b)) L = vec3(0.0);
    acc += L;
    accMask += mask;
  }

  float nNew = float(uSppNew);
  vec4 cur = min(vec4(acc / nNew, accMask / nNew), vec4(65504.0)); // stay finite on RGBA16F targets
  if (uSppDone == 0) {
    outColor = cur;
  } else {
    vec4 prev = texelFetch(uPrev, pix, 0);
    outColor = mix(prev, cur, nNew / (float(uSppDone) + nNew));
  }
}
`
}

export const TRACE_FRAG = traceFrag(false)
// Compiled only when a visitor first picks a model, so the reference-ball scene never pays for it.
export const TRACE_FRAG_MESH = traceFrag(true)

// ------------------------------------------------------------------------------------------------------------
// Display: exposure in EV, then a view transform. ACES 2.0 is the real Output Transform baked from OpenColorIO
// 2.5 (cg-config-v4.0.0_aces-v2.0_ocio-v2.5, 'sRGB - Display' / 'ACES 2.0 - SDR 100 nits (Rec.709)') into a
// 65^3 LUT behind a log2 shaper. AgX and Khronos PBR Neutral take linear Rec.709.
// ------------------------------------------------------------------------------------------------------------
export const DISPLAY_FRAG = /* glsl */ `${HEADER}
${COMMON}
uniform sampler2D uAccum;
uniform int uAccumDiv;  // 1 for the accumulation; k for a 1/k-resolution preview (nearest, like an IPR proxy)
uniform sampler3D uAcesLut;
uniform int uAcesReady;
uniform float uExposure;
uniform int uView;   // 0 ACES 2.0, 1 AgX, 2 PBR Neutral, 3 Standard
uniform int uPass;
out vec4 outColor;

// IEC 61966-2-1 sRGB encoding.
vec3 srgbOETF(vec3 L) {
  L = clamp(L, 0.0, 1.0);
  return mix(1.055 * pow(L, vec3(1.0 / 2.4)) - 0.055, 12.92 * L, lessThanEqual(L, vec3(0.0031308)));
}

// Khronos PBR Neutral (2024), verbatim from KhronosGroup/ToneMapping PBR_Neutral/pbrNeutral.glsl.
vec3 PBRNeutralToneMapping(vec3 color) {
  const float startCompression = 0.8 - 0.04;
  const float desaturation = 0.15;
  float x = min(color.r, min(color.g, color.b));
  float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  color -= offset;
  float peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) return color;
  const float d = 1.0 - startCompression;
  float newPeak = 1.0 - d * d / (peak + d - startCompression);
  color *= newPeak / peak;
  float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(color, newPeak * vec3(1.0), g);
}

// AgX (Blender 4.x-style approximation) as in three.js AgXToneMapping / Filament; sigmoid by B. Wrensch.
// Rec.2020 -> Rec.709 from ITU-R BT.2407-0; Rec.709 -> Rec.2020 from ITU-R BT.2087-0, which prints 0.0114 for
// the G/B entry (three.js carries a rounding error, 0.0113).
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
  vec3( 1.6605, -0.1246, -0.0182),
  vec3(-0.5876,  1.1329, -0.1006),
  vec3(-0.0728, -0.0083,  1.1187));
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
  vec3(0.6274, 0.0691, 0.0164),
  vec3(0.3293, 0.9195, 0.0880),
  vec3(0.0433, 0.0114, 0.8956));
vec3 agxDefaultContrastApprox(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 AgXToneMapping(vec3 color) {
  const mat3 AgXInsetMatrix = mat3(
    vec3(0.856627153315983, 0.137318972929847, 0.11189821299995),
    vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
    vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
  const mat3 AgXOutsetMatrix = mat3(
    vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
    vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
    vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
  const float AgxMinEv = -12.47393;
  const float AgxMaxEv = 4.026069;
  color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
  color = AgXInsetMatrix * color;
  color = max(color, 1e-10);
  color = log2(color);
  color = (color - AgxMinEv) / (AgxMaxEv - AgxMinEv);
  color = clamp(color, 0.0, 1.0);
  color = agxDefaultContrastApprox(color);
  color = AgXOutsetMatrix * color;
  color = pow(max(vec3(0.0), color), vec3(2.2));
  color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
  return clamp(color, 0.0, 1.0);
}

vec3 aces2(vec3 ap1) {
  const float MIN_EV = -12.0;
  const float MAX_EV = 10.0;
  const float N = 65.0;
  vec3 s = clamp((log2(max(ap1, vec3(1e-10)) / 0.18) - MIN_EV) / (MAX_EV - MIN_EV), 0.0, 1.0);
  return texture(uAcesLut, (s * (N - 1.0) + 0.5) / N).rgb; // already display encoded
}

void main() {
  vec3 c = texelFetch(uAccum, ivec2(gl_FragCoord.xy) / uAccumDiv, 0).rgb;
  if (uPass == 4) { outColor = vec4(c, 1.0); return; }                                   // normals: raw data
  if (uPass == 3) { outColor = vec4(srgbOETF(max(AP1_TO_REC709 * c, 0.0)), 1.0); return; } // albedo
  c = max(c, 0.0) * exp2(uExposure);
  if (uView == 0 && uAcesReady == 1) { outColor = vec4(aces2(c), 1.0); return; }
  vec3 r = max(AP1_TO_REC709 * c, 0.0);
  if (uView == 2) r = PBRNeutralToneMapping(r);
  else if (uView != 3) r = AgXToneMapping(r); // AgX, and the fallback while the ACES LUT loads
  outColor = vec4(srgbOETF(r), 1.0);
}
`
