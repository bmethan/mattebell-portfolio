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
uniform int uMode;      // 0: the reflection tables above; 1: the dielectric tables below
out vec4 outColor;

// Total single-scattering albedo of a rough dielectric interface, reflection plus refraction, with the IOR ratio
// eta (transmitted over incident): by visible-normal sampling the weight is F G2/G1 for the reflected direction
// and (1 - F) G2/G1 for the refracted one (energy, i.e. without radiance's 1/eta^2). Its shortfall from 1 is
// what multiple scattering between microfacets would return; Turquin 2019's compensation idea, applied to the
// whole dielectric lobe.
float dielectricAlbedo(vec3 wo, vec3 m, vec2 alpha, float eta) {
  float F = fresnelDielectric(dot(wo, m), eta);
  float e = 0.0;
  vec3 wr = reflect(-wo, m);
  if (wr.z > 0.0) e += F * G2_GGX(wo, wr, alpha) / G1_GGX(wo, alpha);
  vec3 wt = refract(-wo, m, 1.0 / eta);
  if (dot(wt, wt) > 0.0 && wt.z < 0.0) e += (1.0 - F) * G2_GGX(wo, wt, alpha) / G1_GGX(wo, alpha);
  return e;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5) / (uSize - 1.0); // texel i holds mu = i / (N - 1)
  float mu = max(uv.x, 1e-4);
  float r = uv.y;
  vec2 alpha = vec2(max(r * r, 1e-4));
  float x = uLayer * E_X_MAX;
  float eta = (1.0 + x) / (1.0 - x);
  vec3 wo = vec3(sqrt(max(0.0, 1.0 - mu * mu)), 0.0, mu);
  int N = uSamples;
  if (uMode == 1) {
    // R: entering the denser medium (eta); G: leaving it (1 / eta, with total internal reflection).
    float eIn = 0.0, eOut = 0.0;
    for (int i = 0; i < N; i++) {
      vec2 u = vec2((float(i) + 0.5) / float(N), u01(reverseBits32(uint(i))));
      vec3 m = sampleVNDF_SphericalCap(u, wo, alpha);
      eIn += dielectricAlbedo(wo, m, alpha, eta);
      eOut += dielectricAlbedo(wo, m, alpha, 1.0 / eta);
    }
    outColor = vec4(eIn / float(N), eOut / float(N), 0.0, 1.0);
    return;
  }
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
  float specular_weight; vec3 specular_color; float specular_roughness; float specular_roughness_anisotropy;
  float specular_ior;
  float coat_weight; vec3 coat_color; float coat_roughness; float coat_ior; float coat_darkening;
  float fuzz_weight; vec3 fuzz_color; float fuzz_roughness;
  float thin_film_weight; float thin_film_thickness; float thin_film_ior;
  float transmission_weight; vec3 transmission_color; float transmission_depth;
  float transmission_dispersion_scale; float transmission_dispersion_abbe_number;
  float subsurface_weight; vec3 subsurface_color; float subsurface_radius; vec3 subsurface_radius_scale;
  float subsurface_scatter_anisotropy;
  float geometry_thin_walled; // 0 or 1
  float lab_flake_coverage; float lab_flake_size; float lab_flake_tilt; // lab extension (see materials.ts)
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
  vec3 nf;      // flake normal in the shading frame, (0, 0, 1) off the flakes; tilts the base specular only
  // Transmission (spec, Transmission): the dielectric base's substrate is transmissive by transW instead of
  // diffuse. etaT is the IOR ratio the refraction uses (the path's channel when the glass disperses).
  float transW; float etaT; bool thin; vec3 transTint;
  float pT;     // lobe selection weight of the transmission lobe
  float compT;  // multiple-scattering compensation of the transmissive dielectric lobe (reflection + refraction)
};

uniform sampler3D uETable;
uniform sampler3D uETableT; // dielectric albedo: R entering, G leaving (see E_TABLE_FRAG)
uniform int uMultiScatter;
// The compensation as this pixel renders it: uMultiScatter, except in the split compare of it (see main).
bool gMultiScatter;

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

// Dispersion (spec, Transmission > Dispersion): Cauchy's n(lambda) = A + B / lambda^2 through n_d at the
// Fraunhofer d line, with B set by the Abbe number V_d = abbe / scale (F and C lines). Each path that refracts
// through dispersive glass is traced at one wavelength, drawn uniformly from 380-780 nm, and weighted by that
// wavelength's ACEScg response (below), so the spectrum is sampled continuously.
float cauchyIor(float nd, float vd, float nm) {
  const float LF = 486.13, LC = 656.27, LD = 587.56;
  float B = (nd - 1.0) / (vd * (1.0 / (LF * LF) - 1.0 / (LC * LC)));
  float A = nd - B / (LD * LD);
  return A + B / (nm * nm);
}
bool disperses(Mat m) {
  return m.transmission_weight > 0.0 && m.transmission_dispersion_scale > 0.0 && m.transmission_dispersion_abbe_number > 0.0;
}

// CIE 1931 color matching functions: the multi-lobe Gaussian fit of Wyman, Sloan, Shirley 2013, "Simple Analytic
// Approximations to the CIE XYZ Color Matching Functions", JCGT 2(2), Listing 1.
float cmfLobe(float w, float mu, float a, float b) { float t = (w - mu) * (w < mu ? a : b); return exp(-0.5 * t * t); }
vec3 cieXYZ(float w) {
  return vec3(
    0.362 * cmfLobe(w, 442.0, 0.0624, 0.0374) + 1.056 * cmfLobe(w, 599.8, 0.0264, 0.0323) - 0.065 * cmfLobe(w, 501.1, 0.0490, 0.0382),
    0.821 * cmfLobe(w, 568.8, 0.0213, 0.0247) + 0.286 * cmfLobe(w, 530.9, 0.0613, 0.0322),
    1.217 * cmfLobe(w, 437.0, 0.0845, 0.0278) + 0.681 * cmfLobe(w, 459.0, 0.0385, 0.0725));
}
// Path weight for a wavelength drawn uniformly over 380-780 nm (pdf 1/400): its ACEScg response over the
// response's integral, per channel, so the weights average to exactly (1, 1, 1) and white stays white. (Some
// wavelengths fall outside the gamut; their negative components still average out.)
const vec3 SPECTRAL_NORM = vec3(117.6829, 103.4330, 98.1822); // integrals over 380-780 nm, computed offline
vec3 spectralWeight(float nm) {
  return REC709_TO_AP1 * (XYZ_TO_REC709 * cieXYZ(nm)) * 400.0 / SPECTRAL_NORM;
}

// back: the ray reached the surface from inside the object. lambda: the path's wavelength for dispersion (nm);
// colored: the path has already refracted through dispersive glass, so it now carries that wavelength alone.
Surf setupSurf(Mat m, vec3 wo, bool back, float lambda, bool colored) {
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
  // Anisotropy (spec, Specular > Roughness anisotropy): alpha_t = r^2 sqrt(2 / (1 + (1 - a)^2)), alpha_b =
  // (1 - a) alpha_t, along the tangent (x) and bitangent (y) of the shading frame. At a = 0 both are r^2.
  float aniso = sat(m.specular_roughness_anisotropy);
  float alphaT = s.specRough * s.specRough * sqrt(2.0 / (1.0 + sq(1.0 - aniso)));
  s.aS = max(vec2(alphaT, (1.0 - aniso) * alphaT), vec2(1e-4));
  // The albedo tables are isotropic: an anisotropic lobe looks them up at the roughness of its mean alpha.
  float rTable = aniso > 0.0 ? sqrt(sqrt(s.aS.x * s.aS.y)) : s.specRough;
  s.nf = vec3(0.0, 0.0, 1.0);
  s.aC = vec2(max(s.coatRough * s.coatRough, 1e-4));
  float nd = max(m.specular_ior, 1.0);
#ifndef GLASS
  // Opaque variant: no transmission. Thin variant: thin-walled transmission only (no medium, no dispersion).
  s.eta = openpbrSpecularEta(nd, s.coatIor, s.coatW, s.specW);
  s.etaT = s.eta;
#ifdef THIN
  s.thin = m.geometry_thin_walled > 0.5;
  s.transW = s.thin ? sat(m.transmission_weight) : 0.0;
  s.transTint = max(m.transmission_color, vec3(0.0));
#else
  s.transW = 0.0;
  s.thin = false;
  s.transTint = vec3(1.0);
#endif
#else
#ifdef NO_DISPERSION
  float nChan = nd;
#else
  float nChan = disperses(m) ? cauchyIor(nd, m.transmission_dispersion_abbe_number / m.transmission_dispersion_scale, lambda) : nd;
#endif
  s.eta = openpbrSpecularEta(colored ? nChan : nd, s.coatIor, s.coatW, s.specW);
  s.etaT = openpbrSpecularEta(nChan, s.coatIor, s.coatW, s.specW);
  // Subsurface (spec, Subsurface): the opaque base mixes diffuse and subsurface by subsurface_weight, and the
  // subsurface part refracts through the same rough dielectric interface into a scattering medium (the random
  // walk in main). Under transmission it is the translucent base's complement. (The lab's presets use one or the
  // other; a path that refracts into a material with any subsurface walks.)
  float sssW = sat(m.subsurface_weight) * (1.0 - sat(m.transmission_weight));
  s.transW = sat(m.transmission_weight) + sssW;
  // An IOR of 1 does not refract, and the refraction half vector is undefined there: pass straight through, as
  // thin-walled glass does (the same light, with no bending).
  s.thin = m.geometry_thin_walled > 0.5 || abs(s.etaT - 1.0) < 1e-3;
  // With no depth there is no medium, and the color tints the refraction instead (spec).
  s.transTint = m.transmission_depth > 0.0 || sssW > 0.0 ? vec3(1.0) : max(m.transmission_color, vec3(0.0));
  if (back && !s.thin) {
    // Leaving a solid: the IOR ratio inverts (total internal reflection follows from the Fresnel terms).
    s.eta = 1.0 / s.eta;
    s.etaT = 1.0 / s.etaT;
  }
#endif
  // The albedo tables cover eta >= 1 (from outside); inside a solid they are looked up at the reciprocal, an
  // approximation that only matters for rough interior reflections.
  float etaTable = s.eta >= 1.0 ? s.eta : 1.0 / s.eta;

  float mu = clamp(wo.z, 1e-4, 1.0);
  bool ms = gMultiScatter;
  vec3 F0m = s.baseWeight * s.baseColor;

  // textureLod: implicit-derivative fetches inside loops force Direct3D to unroll them.
  vec3 tS = textureLod(uETable, eTableCoord(mu, rTable, etaTable), 0.0).rgb;
  float Ess = max(tS.r, 1e-4);
  s.compD = ms ? 1.0 + tS.b * (1.0 - Ess) / Ess : 1.0;
  s.compM = ms ? 1.0 + favgF82Tint(F0m, s.specColor) * (1.0 - Ess) / Ess : vec3(1.0);
  s.EspecR = sat(tS.g * s.compD);
  // A thin film changes how much the specular reflects, per wavelength; interference conserves energy, so the
  // base receives the complement. Scale the tabulated albedo by the film-to-plain Fresnel ratio at the view angle.
  s.EspecR3 = vec3(s.EspecR);
  if (s.tfW > 0.0) {
    // Where the bare surface barely reflects (an IOR near 1, like a soap film's), the ratio is undefined and the
    // film's reflectance is used with the lobe's F = 1 albedo instead; otherwise this lobe would never be sampled.
    vec3 film = thinFilmF(mu, s.tfIor, s.tfThick, vec3(f0FromEta(s.eta)));
    float Fp = fresnelDielectric(mu, s.eta);
    vec3 withFilm = Fp > 1e-3 ? s.EspecR * film / Fp : Ess * s.compD * film;
    s.EspecR3 = sat3(mix(vec3(s.EspecR), withFilm, s.tfW));
  }

  // Transmissive glass scatters between microfacets too: scale its whole dielectric lobe (reflection and
  // refraction) by 1 / its single-scattering albedo, read for entering or leaving (total internal reflection
  // included). Thin-walled glass mirrors the reflection lobe, so its albedo is the F = 1 one, Ess.
  s.compT = 1.0;
#ifdef THIN
  if (ms && s.transW > 0.0) {
    if (s.thin) {
      s.compT = 1.0 / Ess;
    }
#ifdef GLASS
    else {
      vec2 tT = textureLod(uETableT, eTableCoord(mu, rTable, s.etaT >= 1.0 ? s.etaT : 1.0 / s.etaT), 0.0).rg;
      s.compT = 1.0 / max(s.etaT >= 1.0 ? tT.r : tT.g, 1e-3);
    }
#endif
  }
#endif

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
    tb * (1.0 - s.metal) * (1.0 - s.transW) * max(lum((1.0 - s.EspecR3) * diffAlb), 0.0),
    tb * ((1.0 - s.metal) * lum(s.EspecR3) * max(lum(s.specColor), 0.05) + s.metal * s.specW * max(lum(F0m), 0.05)),
    tFuzz * s.coatW * s.Ecoat,
    s.fuzzW * s.Efuzz * max(lum(s.fuzzColor), 0.05));
  s.pT = tb * (1.0 - s.metal) * s.transW * max(lum((1.0 - s.EspecR3) * s.transTint), 0.05);
#ifdef THIN
  if (s.transW > 0.0) {
    // Glass: split reflection and refraction by the actual Fresnel at the view angle, total internal reflection
    // included (the albedo tables only know the view from outside). Inside past the critical angle nearly every
    // sample must reflect; picking refraction there only fails. Floors keep both lobes reachable (rough
    // microfacets can still refract near the critical angle, or reflect where F is tiny).
    float Fmu = fresnelDielectric(mu, s.eta);
    float dielW = tb * (1.0 - s.metal) * s.transW;
    s.p.y = mix(s.p.y, dielW * max(Fmu, 0.05), s.transW);
    s.pT = dielW * max((1.0 - Fmu) * lum(s.transTint), 0.05);
  }
#endif
  return s;
}

// filt: 0 all lobes, 1 diffuse only, 2 everything but diffuse (first-hit light path expressions for AOVs).
// Returns the selection probabilities of diffuse, base specular, coat and fuzz; pt receives transmission's.
vec4 lobeProbs(Surf s, int filt, out float pt) {
  vec4 p = s.p;
  float t = s.pT;
  if (filt == 1) { p = vec4(p.x, 0.0, 0.0, 0.0); t = 0.0; }
  if (filt == 2) p.x = 0.0;
  float sum = p.x + p.y + p.z + p.w + t;
  pt = sum > 0.0 ? t / sum : 0.0;
  return sum > 0.0 ? p / sum : vec4(0.0);
}

#ifdef THIN
// Glass is sampled as one strategy, as in pbrt-v4's DielectricBxDF: a visible microfacet (spherical caps), then
// reflect or refract by that microfacet's own Fresnel, weighted by how much of the substrate transmits. So no
// sample is spent on a refraction that total internal reflection forbids, or on a reflection the Fresnel makes
// negligible. vndfPdf: the visible normal density; glassReflectProb: the chance of reflecting at cosine cosM.
float vndfPdf(vec3 wo, vec3 m, vec2 alpha) { return G1_GGX(wo, alpha) * max(dot(wo, m), 0.0) * D_GGX(m, alpha) / wo.z; }
float glassReflectProb(Surf s, float cosM) {
  float F = fresnelDielectric(cosM, s.etaT);
  // With a film, much of the reflection is the film's (a soap film's bare surface has IOR 1, F = 0): blend in its
  // albedo at the view angle. The clamp keeps both choices reachable wherever either could carry light.
  if (s.tfW > 0.0) F = mix(F, sat(lum(s.EspecR3)), s.tfW);
  return clamp(F / max(F + s.transW * (1.0 - F), 1e-6), 0.02, 0.98);
}
#endif

// Returns the mixture pdf; fD and fS receive f * cos(theta_i), split into diffuse and everything else.
float evalSurf(Surf s, vec3 wo, vec3 wi, int filt, out vec3 fD, out vec3 fS) {
  fD = vec3(0.0);
  fS = vec3(0.0);
  if (wo.z <= 0.0) return 0.0;
  float pt;
  vec4 p = lobeProbs(s, filt, pt);

#ifndef THIN
  if (wi.z <= 0.0) return 0.0;
#else
  if (wi.z <= 0.0) {
    // Transmission, the only lobe below the surface. Its Fresnel complements the reflection lobe's at the same
    // microfacet, and like it ignores multiple scattering between microfacets: smooth glass is exact, rough
    // glass loses some energy (the furnace test shows how much).
    if (pt <= 0.0) return 0.0;
    float ft, cosM, pdfT; // pdfT: density of the sampled microfacet mapped to wi, before the reflect/refract choice
    if (s.thin) {
      // Thin-walled: the reflection lobe mirrored through the surface, so light passes straight through.
      vec3 wr = vec3(wi.xy, -wi.z);
      vec3 hr = normalize(wo + wr);
      cosM = sat(dot(wo, hr));
      ft = G2_GGX(wo, wr, s.aS) * D_GGX(hr, s.aS) / (4.0 * wo.z);
      pdfT = cosM > 0.0 ? vndfPdf(wo, hr, s.aS) / (4.0 * cosM) : 0.0;
    } else {
#ifndef GLASS
      return 0.0;
#else
      // Rough dielectric refraction: Walter, Marschner, Li, Torrance 2007 (eq. 21), radiance transport (1/eta^2),
      // as in pbrt-v4's DielectricBxDF; pdf through the generalized half vector's Jacobian.
      float eta = s.etaT;
      vec3 wm = normalize(wo + wi * eta);
      if (wm.z < 0.0) wm = -wm;
      float dI = dot(wi, wm), dO = dot(wo, wm);
      if (dI >= 0.0 || dO <= 0.0) return 0.0;
      float denom = dI + dO / eta;
      float D = D_GGX(wm, s.aS);
      ft = D * G2_GGX(wo, wi, s.aS) * abs(dI * dO / (wo.z * denom * denom)) / (eta * eta);
      cosM = dO;
      pdfT = G1_GGX(wo, s.aS) * dO * D / wo.z * abs(dI) / (denom * denom);
#endif
    }
    vec3 T3 = vec3(1.0 - fresnelDielectric(cosM, s.etaT));
    if (s.tfW > 0.0) T3 = mix(T3, vec3(1.0) - thinFilmF(cosM, s.tfIor, s.tfThick, vec3(f0FromEta(s.etaT))), s.tfW);
    fS = s.tBase * (1.0 - s.metal) * s.transW * s.transTint * T3 * ft * s.compT;
    return (p.y + pt) * pdfT * (1.0 - glassReflectProb(s, cosM));
  }
#endif

  vec3 h = normalize(wo + wi);
  float voh = sat(dot(wo, h));
  float pdf = 0.0;

  if (filt != 2 && s.metal < 1.0) {
    vec3 d = s.baseWeight * f_EON(sat3(s.baseColor), s.diffRough, wi, wo) * wi.z;
    fD = s.tBase * (1.0 - s.metal) * (1.0 - s.transW) * (1.0 - s.EspecR3) * d;
    pdf += p.x * pdf_EON(wo, wi, s.diffRough);
  }
  if (filt != 1) {
    // On a flake, the base specular lobe sits on the flake's tilted normal: evaluate it in the flake's frame (the
    // half vector's angle to the view, and so the Fresnel, does not change), with the macro cosine for the
    // rendering equation. Off the flakes, the frames coincide.
    vec3 woS = wo, wiS = wi, hS = h;
    bool flaked = s.nf.z < 0.99999;
    if (flaked) {
      vec3 b1, b2;
      onb(s.nf, b1, b2);
      woS = vec3(dot(wo, b1), dot(wo, b2), dot(wo, s.nf));
      wiS = vec3(dot(wi, b1), dot(wi, b2), dot(wi, s.nf));
      hS = normalize(woS + wiS);
    }
    if (woS.z > 0.0 && wiS.z > 0.0) {
      float g = G2_GGX(woS, wiS, s.aS) * D_GGX(hS, s.aS) / (4.0 * woS.z);
      if (flaked) g *= wi.z / wiS.z;
      // Over the transmissive substrate, the dielectric reflection takes the glass compensation (compT) instead.
      float compR = mix(s.compD, s.compT, s.transW);
      vec3 Fd = s.specColor * fresnelDielectric(voh, s.eta) * compR;
      vec3 Fm = s.specW * fresnelF82Tint(voh, s.baseWeight * s.baseColor, s.specColor) * s.compM;
      if (s.tfW > 0.0) {
        // The film over each substrate it can sit on: the dielectric, the metal, or both when metalness is
        // partial (then exact too, where one evaluation over a metalness-blended substrate lost energy: 0.89 in
        // the furnace for a 30% metal pearl). A loop keeps a single inlined copy of the film code.
        vec3 FtfD = vec3(0.0), FtfM = vec3(0.0);
        int subs = s.metal > 0.0 && s.metal < 1.0 ? 2 : 1;
        for (int k = 0; k < subs; k++) {
          bool overMetal = subs == 2 ? k == 1 : s.metal >= 1.0;
          vec3 f = thinFilmF(voh, s.tfIor, s.tfThick, overMetal ? s.baseWeight * s.baseColor : vec3(f0FromEta(s.eta)));
          if (overMetal) FtfM = f;
          else FtfD = f;
        }
        Fd = mix(Fd, s.specColor * FtfD * compR, s.tfW);
        Fm = mix(Fm, s.specW * FtfM * s.compM, s.tfW);
      }
      fS += s.tBase * mix(Fd, Fm, s.metal) * g;
#ifdef THIN
      if (s.transW > 0.0) pdf += (p.y + pt) * vndfPdf(wo, h, s.aS) / (4.0 * max(voh, 1e-6)) * glassReflectProb(s, voh);
      else pdf += p.y * pdfGGXReflection_Bounded(woS, wiS, s.aS);
#else
      pdf += p.y * pdfGGXReflection_Bounded(woS, wiS, s.aS);
#endif
    }

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

// Lobes in [0, 1): diffuse, base specular, coat, fuzz, transmission. Returns false when no direction results.
bool sampleSurf(Surf s, vec3 wo, int filt, vec3 u, out vec3 wi) {
  float pt;
  vec4 p = lobeProbs(s, filt, pt);
  float cCoat = p.x + p.y + p.z, cFuzz = cCoat + p.w;
  if (u.z < p.x) {
    wi = sample_EON(wo, s.diffRough, u.x, u.y);
    return wi.z > 0.0;
  }
#ifdef THIN
  bool specPick = u.z < p.x + p.y, transPick = u.z >= cFuzz && pt > 0.0;
  if (s.transW > 0.0 && (specPick || transPick)) {
    // Glass (see glassReflectProb): the base specular and transmission intervals form one strategy. Where u.z
    // falls inside their union picks reflect or refract against the sampled microfacet's probability.
    float t = (specPick ? u.z - p.x : p.y + u.z - cFuzz) / max(p.y + pt, 1e-9);
    vec3 m = sampleVNDF_SphericalCap(u.xy, wo, s.aS);
    float cosM = dot(wo, m);
    if (cosM <= 0.0) return false;
    vec3 wr = reflect(-wo, m);
    if (t < glassReflectProb(s, cosM)) {
      wi = wr;
      return wi.z > 0.0;
    }
    if (s.thin) {
      wi = vec3(wr.xy, -wr.z); // straight through
      return wi.z < 0.0;
    }
#ifdef GLASS
    wi = refract(-wo, m, 1.0 / s.etaT);
    return dot(wi, wi) > 0.0 && wi.z < 0.0;
#else
    return false;
#endif
  }
#endif
  if (u.z < cCoat) {
    // Base specular or coat, through one call site (Direct3D inlines every call). The base specular on a flake is
    // sampled in the flake's frame (see evalSurf) and brought back; otherwise the frame is the identity.
    bool coat = u.z >= p.x + p.y;
    vec3 b1 = vec3(1.0, 0.0, 0.0), b2 = vec3(0.0, 1.0, 0.0), b3 = vec3(0.0, 0.0, 1.0);
    if (!coat && s.nf.z < 0.99999) {
      onb(s.nf, b1, b2);
      b3 = s.nf;
    }
    vec3 woS = vec3(dot(wo, b1), dot(wo, b2), dot(wo, b3));
    if (woS.z <= 0.0) return false;
    vec3 wiS = sampleGGXReflection_Bounded(u.xy, woS, coat ? s.aC : s.aS);
    wi = wiS.x * b1 + wiS.y * b2 + wiS.z * b3;
    return wi.z > 0.0;
  }
  if (u.z < cFuzz && p.w > 0.0) {
    wi = sampleFuzz(wo, s.fuzzRough, u.xy);
    return wi.z > 0.0;
  }
  return false;
}

vec3 albedoAOV(Mat m) {
  vec3 a = mix(m.base_color * m.base_weight, m.base_color, m.base_metalness);
  a = mix(a, m.subsurface_color, m.subsurface_weight * (1.0 - m.base_metalness));
  a = mix(a, m.transmission_color, m.transmission_weight * (1.0 - m.base_metalness));
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

bool lightOn(int k) { return k >= 3 || maxc(uLightRadiance[k]) > 0.0; } // light 3 is the environment, when on
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
// Image-based lighting (uEnvOn 1): the environment is light 3, after the three softboxes, with its own image.
// uEnvMap holds its ACEScg radiance (mipmapped, so the camera can see it as a softened backdrop); uEnvAlias, per
// pixel, the alias table's probability and alias and the pixel's pdf constant (see environments.ts). Lookups are
// nearest-texel, so the radiance is constant over each pixel, as the sampling density is.
uniform int uEnvOn;
uniform sampler2D uEnvMap;
uniform highp sampler2D uEnvAlias;
uniform ivec2 uEnvSize;
uniform float uEnvRot;  // turn about the vertical axis, in turns
uniform float uEnvBlur; // mip level of the backdrop
#define ENV_LIGHT 3
vec2 envUV(vec3 d) {
  return vec2(fract(atan(d.x, -d.z) * (0.5 / PI) + 0.5 + uEnvRot), acos(clamp(d.y, -1.0, 1.0)) / PI);
}
ivec2 envTexel(vec2 uv) { return min(ivec2(uv * vec2(uEnvSize)), uEnvSize - 1); }
vec3 envLe(vec3 d) { return texelFetch(uEnvMap, envTexel(envUV(d)), 0).rgb; }
vec3 envBackdrop(vec3 d) { return textureLod(uEnvMap, envUV(d), uEnvBlur).rgb; }
// Solid-angle density of sampleEnv: the pixel's constant over sin(theta).
float envPdf(vec3 d) {
  float s = sqrt(max(0.0, 1.0 - d.y * d.y));
  return s > 1e-6 ? texelFetch(uEnvAlias, envTexel(envUV(d)), 0).z / s : 0.0;
}
// A direction toward the environment: a pixel from the alias table (u.x picks a slot, u.y keeps it or takes its
// alias), then a point uniform in the pixel's longitude and colatitude (u.zw).
bool sampleEnv(vec4 u, out vec3 wi, out float pdf) {
  int n = uEnvSize.x * uEnvSize.y;
  int i = min(int(u.x * float(n)), n - 1);
  vec4 a = texelFetch(uEnvAlias, ivec2(i % uEnvSize.x, i / uEnvSize.x), 0);
  if (u.y >= a.x) i = int(a.y);
  ivec2 px = ivec2(i % uEnvSize.x, i / uEnvSize.x);
  float theta = (float(px.y) + u.w) / float(uEnvSize.y) * PI;
  float phi = ((float(px.x) + u.z) / float(uEnvSize.x) - 0.5 - uEnvRot) * TWO_PI;
  float s = sin(theta);
  wi = vec3(s * sin(phi), cos(theta), -s * cos(phi));
  pdf = s > 1e-6 ? texelFetch(uEnvAlias, px, 0).z / s : 0.0;
  return pdf > 0.0;
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
uniform usampler2D uBvh;     // 4 texels per four-wide node, child boxes compressed (see collapseBvh4 in models.ts)
uniform sampler2D uTriPos;   // 3 texels per triangle: v0, v1 - v0, v2 - v0
uniform usampler2D uTriNrm;  // 1 texel per triangle: three octahedral vertex normals (snorm16 x2), material slot
uniform mat3 uModelRot;      // object to world rotation
uniform vec3 uModelPos;
uniform float uModelScale;
uniform int uMaxNodeVisits;  // from a uniform so the traversal loop is never unrolled
uniform int uThinSlot;       // the model's smooth thin-glass material slot, or -1

// Transmittance of the model's thin glass (slot uThinSlot) at a crossing with cosine c: no film on it, so this
// stays a few lines (shadow rays evaluate it inside the traversal).
vec3 thinGlassT(float c) {
  Mat g = uMat[4];
  float T = 1.0 - fresnelDielectric(c, max(g.specular_ior, 1.0));
  vec3 tint = g.transmission_depth > 0.0 ? vec3(1.0) : max(g.transmission_color, vec3(0.0));
  return T * tint * sat(g.transmission_weight) * (1.0 - sat(g.base_metalness));
}

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

const int BVH_STACK = 48; // BVH4_STACK in models.ts: three deferred children per level at most
// Closest hit before tMax (or, with anyHit, the first hit found). Ray/triangle: Moller and Trumbore 1997.
// Returns the distance (the object-space ray is scaled so it equals the world distance) or -1.
// cur packs (index << 5) | count: a leaf of count triangles, or with count 0 the four-wide node at index. A node
// holds its four children's boxes, so only entered children are visited, nearest first; the others wait on the
// stack with their entry distances and are skipped if a closer hit has been found by then. One loop body
// handles leaves and nodes, so the triangle test is compiled once.
// With anyHit (shadow rays), triangles in the thin-glass slot (uThinSlot) do not stop the ray: their
// transmittance multiplies Tthin instead, in any order (it does not depend on the order of crossings).
float traceMesh(vec3 roW, vec3 rdW, float tMax, bool anyHit, out int triHit, out vec2 bary, inout vec3 Tthin) {
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
  int cur = 0; // the root, a four-wide node
  for (int visit = 0; visit < uMaxNodeVisits; visit++) {
    int count = cur & 31;
    int index = cur >> 5;
    if (count == 0) {
      uvec4 h = texelFetch(uBvh, texAt(4 * index), 0);
      uvec4 q0 = texelFetch(uBvh, texAt(4 * index + 1), 0);
      uvec4 q1 = texelFetch(uBvh, texAt(4 * index + 2), 0);
      uvec4 refs = texelFetch(uBvh, texAt(4 * index + 3), 0);
      vec3 corner = uintBitsToFloat(h.xyz);
      // The step per axis is a power of two: its biased exponent goes straight into a float's exponent field.
      vec3 stepSize = uintBitsToFloat(uvec3(h.w & 255u, (h.w >> 8) & 255u, (h.w >> 16) & 255u) << 23);
      ivec4 rc = ivec4(refs);
      vec4 tc;
      for (int i = 0; i < 4; i++) {
        uint sh = uint(8 * i);
        vec3 lo = corner + vec3((q0.xyz >> sh) & 255u) * stepSize;
        vec3 hi = corner + vec3(uvec3(q0.w >> sh, q1.x >> sh, q1.y >> sh) & 255u) * stepSize;
        tc[i] = refs[i] == 0xffffffffu ? 1e30 : boxEnter(lo, hi, o, invD, tBest); // an empty slot
      }
      // Nearest first: sort the four (distance, ref) pairs, misses (1e30) last.
      if (tc.y < tc.x) { tc.xy = tc.yx; rc.xy = rc.yx; }
      if (tc.w < tc.z) { tc.zw = tc.wz; rc.zw = rc.wz; }
      if (tc.z < tc.x) { tc.xz = tc.zx; rc.xz = rc.zx; }
      if (tc.w < tc.y) { tc.yw = tc.wy; rc.yw = rc.wy; }
      if (tc.z < tc.y) { tc.yz = tc.zy; rc.yz = rc.zy; }
      if (tc.x < 1e30) {
        // Defer the farther hits, farthest deepest.
        if (tc.w < 1e30 && sp < BVH_STACK) { stackRef[sp] = rc.w; stackT[sp] = tc.w; sp++; }
        if (tc.z < 1e30 && sp < BVH_STACK) { stackRef[sp] = rc.z; stackT[sp] = tc.z; sp++; }
        if (tc.y < 1e30 && sp < BVH_STACK) { stackRef[sp] = rc.y; stackT[sp] = tc.y; sp++; }
        cur = rc.x;
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
#ifdef THIN
          if (anyHit && uThinSlot >= 0 && int(texelFetch(uTriNrm, texAt(tri), 0).w) == uThinSlot) {
            Tthin *= thinGlassT(abs(dot(normalize(cross(e1, e2)), normalize(d))));
            if (maxc(Tthin) <= 0.0) return t;
            continue;
          }
#endif
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

// Variants: MESH adds triangle meshes; GLASS adds transmission (refraction, absorption, dispersion, thin-walled
// glass and shadows through it). Each costs seconds of compile time on Direct3D, so the lab compiles the plain
// variant first and the others only when a scene needs them.
// THIN is the cheaper subset for thin-walled glass only (car windows, a soap film): straight-through transmission
// and shadows through it, without solid refraction, absorption or dispersion. GLASS includes it.
// SSS is the solid variant for subsurface media: refraction and the random walk, without dispersion (skin does
// not disperse), so neither glass nor skin compiles the other's code.
// AUX adds the denoiser's guide images and noise statistics (two more render targets), on GPUs that can draw
// five at once.
export function traceFrag(mesh: boolean, glass: 'none' | 'thin' | 'full' | 'sss' = 'none', aux = false) {
  const solid = glass === 'full' || glass === 'sss'
  const defs = `${mesh ? '#define MESH 1\n' : ''}${solid ? '#define GLASS 1\n' : ''}${glass === 'sss' ? '#define SSS 1\n#define NO_DISPERSION 1\n' : ''}${glass !== 'none' ? '#define THIN 1\n' : ''}${aux ? '#define AUX 1\n' : ''}`
  return /* glsl */ `${HEADER}${defs}
${COMMON}
${MICROFACET}
${TABLE_CONSTS}
${EON}
${FUZZ}
${THIN_FILM}
${OPENPBR}
${LIGHTS}

// The running means of the three per-light images (see main): key (with the dim environment), fill, rim.
uniform sampler2D uPrev0;
uniform sampler2D uPrev1;
uniform sampler2D uPrev2;
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
// Cyc: the floor runs back to z = uCycZ, sweeps up a quarter cylinder of radius uCycR (axis along x), and
// becomes a wall at z = uCycZ - uCycR. Off (uCyc 0), the floor is an endless plane in a black void.
uniform int uCyc;
uniform float uCycZ;
uniform float uCycR;
// The scene has smooth thin-walled glass (a soap bubble, car windows): shadow rays pass through it, dimmed.
uniform int uThinGlass;
// The color chart (uChart 1): a card from corner uChartO along its full width uChartU and height uChartV, the front
// facing cross(U, V). Its 24 patches (row by row from the top left), then its frame, in uChartColor.
uniform int uChart;
uniform vec3 uChartO;
uniform vec3 uChartU;
uniform vec3 uChartV;
uniform vec3 uChartColor[25];
#define CHART_MAT 16 // material index of the chart's first patch; the frame is CHART_MAT + 24
#ifdef MESH
// Material slots: 0 gray ball, 1 chrome, 2 hero, 3 floor, 4-9 the model's other parts.
uniform vec4 uBall[3]; // center, radius (0 = absent)
uniform Mat uMat[10];
#else
uniform vec3 uBallX;
uniform Mat uMat[4];
#endif

// Light mixer: every light's light lands in its own image (the softboxes traced in white), so the display can
// recolor, rescale or switch off each light exactly without tracing again (light transport is linear in each
// light's emission). The faint constant environment of the softbox rig rides with the key; an HDR environment has
// the fourth image. Alpha of the key image is the coverage mask.
layout(location = 0) out vec4 outKey;
layout(location = 1) out vec4 outFill;
layout(location = 2) out vec4 outRim;
layout(location = 3) out vec4 outEnv;
uniform sampler2D uPrev3;
#ifdef AUX
// For the denoiser, as running means like the light images: fill.a, the mean of Y^2, where Y is the luminance
// of the pixel's light (all lights, in white); outAux0, the first-hit albedo with a = mean of Y; outAux1, the
// first-hit normal.
layout(location = 4) out vec4 outAux0;
layout(location = 5) out vec4 outAux1;
uniform sampler2D uPrev4;
uniform sampler2D uPrev5;
#endif
void addLight(int k, vec3 c, inout vec3 L0, inout vec3 L1, inout vec3 L2, inout vec3 L3) {
  if (k == 0) L0 += c;
  else if (k == 1) L1 += c;
  else if (k == 2) L2 += c;
  else L3 += c;
}

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
// back: the ray arrived from inside the object (only a transmissive material lets a path get there).
// light, lightT: the nearest softbox in front of the hit (path rays only), which does not block it (see main).
struct Hit { float t; vec3 n; vec3 ng; int mat; int light; float lightT; bool back; };

// The cyc's back wall and sweep (the floor is the plane test in trace()): nearest hit before tMax, with the
// normal facing into the room, or -1. Also a shadow occluder: a key light dragged low and behind can end up
// partly behind the wall.
float cycHit(vec3 ro, vec3 rd, float tMax, out vec3 n) {
  n = vec3(0.0, 0.0, 1.0);
  float best = -1.0;
  // Back wall, above the sweep.
  if (rd.z < 0.0) {
    float t = (uCycZ - uCycR - ro.z) / rd.z;
    if (t > 1e-4 && t < tMax && ro.y + rd.y * t >= uCycR) { best = t; tMax = t; }
  }
  // The sweep: the quarter of the cylinder below its axis and behind the floor's end, seen from inside.
  vec2 o2 = vec2(ro.y - uCycR, ro.z - uCycZ);
  vec2 d2 = rd.yz;
  float a = dot(d2, d2), b = dot(o2, d2), c = dot(o2, o2) - uCycR * uCycR;
  float disc = b * b - a * c;
  if (a > 0.0 && disc > 0.0) {
    float sq = sqrt(disc);
    for (int i = 0; i < 2; i++) {
      float t = (-b + (i == 0 ? -sq : sq)) / a;
      vec2 q = o2 + d2 * t;
      if (t > 1e-4 && t < tMax && q.x <= 0.0 && q.y <= 0.0) {
        best = t;
        n = vec3(0.0, -q.x, -q.y) / uCycR;
        break;
      }
    }
  }
  return best;
}

// The color chart: hit distance before tMax or -1, the normal facing the ray, and the patch hit (24: the frame or
// the back). The patches are squares in a 6 x 4 grid, CHART_GAP patch widths apart and from the edges (the host
// sizes the card to match).
#define CHART_GAP 0.18
float chartHit(vec3 ro, vec3 rd, float tMax, out vec3 nOut, out int cellId) {
  vec3 N = normalize(cross(uChartU, uChartV));
  float dn = dot(rd, N);
  nOut = dn < 0.0 ? N : -N;
  cellId = 24;
  if (abs(dn) < 1e-8) return -1.0;
  float t = dot(uChartO - ro, N) / dn;
  if (t <= 1e-4 || t >= tMax) return -1.0;
  vec3 q = ro + rd * t - uChartO;
  vec2 ab = vec2(dot(q, uChartU) / dot(uChartU, uChartU), dot(q, uChartV) / dot(uChartV, uChartV));
  if (any(lessThan(ab, vec2(0.0))) || any(greaterThan(ab, vec2(1.0)))) return -1.0;
  if (dn < 0.0) {
    vec2 g = ab * vec2(6.0 + 7.0 * CHART_GAP, 4.0 + 5.0 * CHART_GAP) - CHART_GAP; // from the first patch's corner
    vec2 cell = floor(g / (1.0 + CHART_GAP));
    vec2 f = g - cell * (1.0 + CHART_GAP);
    if (cell.x >= 0.0 && cell.y >= 0.0 && cell.x < 6.0 && cell.y < 4.0 && f.x < 1.0 && f.y < 1.0)
      cellId = int(cell.x) + 6 * (3 - int(cell.y));
  }
  return t;
}

#ifdef MESH
${MESH_GLSL}
float intersectBall(vec3 ro, vec3 rd, vec4 s, float tMax) {
  if (s.w <= 0.0) return -1.0;
  float t = intersectSphere((ro - s.xyz) / s.w, rd, vec3(0.0), tMax / s.w);
  return t > 0.0 ? t * s.w : -1.0;
}
#endif

// Smooth thin-walled transmission does not change a ray's direction, so light can be sampled straight through
// it exactly: a shadow ray crossing such a surface is dimmed by its transmittance instead of blocked. (Paths that
// cross one by BSDF sampling keep the previous vertex's MIS state; see the main loop.)
bool smoothThin(Mat m) {
  return m.geometry_thin_walled > 0.5 && m.transmission_weight > 0.0 && m.specular_roughness <= 0.01 && m.coat_weight <= 0.0;
}
#if !defined(MESH) && defined(THIN)
// The hero ball as thin glass (the soap bubble): its transmittance at a crossing with cosine c, film included.
vec3 thinPassT(Mat m, float c) {
  float nd = max(m.specular_ior, 1.0);
  vec3 T = vec3(1.0 - fresnelDielectric(c, nd));
  if (m.thin_film_weight > 0.0 && m.thin_film_thickness > 0.0)
    T = mix(T, vec3(1.0) - thinFilmF(c, m.thin_film_ior, m.thin_film_thickness, vec3(f0FromEta(nd))), sat(m.thin_film_weight));
  vec3 tint = m.transmission_depth > 0.0 ? vec3(1.0) : max(m.transmission_color, vec3(0.0));
  return T * tint * sat(m.transmission_weight) * (1.0 - sat(m.base_metalness));
}
#endif

// The one scene query, for both kinds of ray (it is called from a single place: see main).
// A path ray (shadow false) finds the closest hit before tMax, softboxes included when withLights.
// A shadow ray returns at the first blocker (h.mat >= 0) and otherwise reports the light that gets through:
// smooth thin glass (uThinGlass: the hero ball, or the model's glass slot) dims T at each crossing instead.
Hit trace(vec3 ro, vec3 rd, float tMax, bool shadow, bool withLights, inout vec3 T) {
  Hit h;
  h.t = tMax; h.mat = -1; h.light = -1; h.lightT = tMax; h.n = vec3(0.0, 1.0, 0.0); h.ng = h.n; h.back = false;
#ifdef MESH
  for (int i = 0; i < 3; i++) {
    vec4 s = uBall[i];
    float t = intersectBall(ro, rd, s, h.t);
    if (t > 0.0) {
      vec3 nOut = normalize(ro + rd * t - s.xyz);
      h.t = t; h.back = dot(rd, nOut) > 0.0; h.n = h.back ? -nOut : nOut; h.ng = h.n; h.mat = i;
      if (shadow) return h;
    }
  }
  int tri;
  vec2 bc;
  float tm = traceMesh(ro, rd, h.t, shadow, tri, bc, T);
  if (tm > 0.0) {
    h.t = tm;
    if (shadow) { h.mat = 4; return h; } // blocked (the model's part does not matter)
    uvec4 nn = texelFetch(uTriNrm, texAt(tri), 0);
    h.mat = int(nn.w);
    vec3 e1 = texelFetch(uTriPos, texAt(3 * tri + 1), 0).xyz;
    vec3 e2 = texelFetch(uTriPos, texAt(3 * tri + 2), 0).xyz;
    vec3 ng = normalize(uModelRot * cross(e1, e2));
    vec3 ns = normalize(uModelRot * ((1.0 - bc.x - bc.y) * octDecode(nn.x) + bc.x * octDecode(nn.y) + bc.y * octDecode(nn.z)));
    // Outside is where the authored vertex normals point (winding may vary between parts).
    h.back = dot(rd, dot(ng, ns) < 0.0 ? -ng : ng) > 0.0;
    // Two-sided: both normals face the ray. An interpolated normal that would put the viewer below the surface
    // falls back to the facet's own.
    if (dot(ng, rd) > 0.0) ng = -ng;
    if (dot(ns, ng) < 0.0) ns = -ns;
    if (dot(ns, -rd) <= 1e-4) ns = ng;
    h.n = ns;
    h.ng = ng;
  }
#else
  for (int i = 0; i < 3; i++) {
    vec3 c = vec3(ballX(i), 1.0, 0.0);
#ifdef THIN
    if (shadow && i == 2 && uThinGlass == 1) {
      // Both crossings of the bubble within the segment.
      vec3 f = ro - c;
      float b = dot(f, rd), disc = b * b - (dot(f, f) - 1.0);
      if (disc > 0.0) {
        float sq = sqrt(disc);
        for (int j = 0; j < 2; j++) {
          float t = -b + (j == 0 ? -sq : sq);
          if (t > 1e-4 && t < tMax) T *= thinPassT(uMat[2], abs(dot(rd, normalize(ro + rd * t - c))));
        }
      }
      continue;
    }
#endif
    float t = intersectSphere(ro, rd, c, h.t);
    if (t > 0.0) {
      vec3 nOut = normalize(ro + rd * t - c);
      h.t = t; h.back = dot(rd, nOut) > 0.0; h.n = h.back ? -nOut : nOut; h.ng = h.n; h.mat = i;
      if (shadow) return h;
    }
  }
#endif
  if (uChart == 1 && uFurnace == 0) {
    vec3 nc;
    int cid;
    float t = chartHit(ro, rd, h.t, nc, cid);
    if (t > 0.0) {
      h.t = t; h.n = nc; h.ng = nc; h.mat = CHART_MAT + cid; h.back = false;
      if (shadow) return h;
    }
  }
  if (uFurnace == 0 && rd.y < 0.0) {
    float t = -ro.y / rd.y;
    if (t > 0.0 && t < h.t && (uCyc == 0 || ro.z + rd.z * t >= uCycZ)) {
      h.t = t; h.n = vec3(0.0, 1.0, 0.0); h.ng = h.n; h.mat = 3; h.back = false;
      if (shadow) return h;
    }
  }
  if (uFurnace == 0 && uCyc == 1) {
    vec3 nc;
    float t = cycHit(ro, rd, h.t, nc);
    if (t > 0.0) { h.t = t; h.n = nc; h.ng = nc; h.mat = 3; h.back = false; }
  }
  if (withLights) {
    for (int k = 0; k < uNumLights; k++) {
      if (!lightOn(k)) continue;
      float t = intersectRect(ro, rd, k, min(h.lightT, h.t)); // in front of the surface hit, if any
      if (t > 0.0) { h.lightT = t; h.light = k; }
    }
  }
  return h;
}

Mat getMat(int i) {
  Mat m;
  if (i >= CHART_MAT) {
    // The chart: the gray card's matte surface in the patch's color.
    m = uMat[0];
    m.base_color = uChartColor[i - CHART_MAT];
  }
  else if (i == 0) m = uMat[0];
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
    m.transmission_color = vec3(1.0);
    m.subsurface_color = vec3(1.0);
  }
  return m;
}

// Lab extension: the flake under a point, in the shading frame. P is the point in the hero's own space, so the
// flakes stay put on the surface as a model turns. Cells of the flake size; each holds a flake with probability
// lab_flake_coverage, its normal tilted in a random direction by up to lab_flake_tilt (tangent of the angle).
vec3 flakeNormal(Mat m, vec3 P) {
  if (m.lab_flake_coverage <= 0.0) return vec3(0.0, 0.0, 1.0);
  uvec3 h = pcg3d(uvec3(ivec3(floor(P / m.lab_flake_size)) + 1048576));
  if (u01(h.x) >= m.lab_flake_coverage) return vec3(0.0, 0.0, 1.0);
  float r = m.lab_flake_tilt * sqrt(u01(h.y));
  float phi = TWO_PI * u01(h.z);
  return normalize(vec3(r * cos(phi), r * sin(phi), 1.0));
}

// Firefly control on indirect light only (Cycles-style split), hue preserving. Direct light stays unbiased.
vec3 clampIndirect(vec3 c) {
  float m = maxc(c);
  return (uIndirectClamp > 0.0 && m > uIndirectClamp) ? c * (uIndirectClamp / m) : c;
}
bool nonFinite(float x) { return (floatBitsToUint(x) & 0x7f800000u) == 0x7f800000u; }

uniform int uMaxScatter; // a path's scattering events inside subsurface media
// Cosine-weighted direction about +z (the Lambertian exit of a walk).
vec3 sampleCosine(vec2 u) {
  float r = sqrt(u.x), phi = TWO_PI * u.y;
  return vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}
#ifdef SSS
// Random numbers for the subsurface walk (its dimensions are unbounded, past the Sobol' sequence's).
float walkRand(inout uint seed) {
  seed = pcg(seed);
  return u01(seed);
}
// Henyey-Greenstein phase function, sampled about the direction of travel d (g > 0 scatters forward).
vec3 sampleHG(vec3 d, float g, float u1, float u2) {
  float c = 1.0 - 2.0 * u1;
  if (abs(g) > 1e-3) {
    float q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
    c = clamp((1.0 + g * g - q * q) / (2.0 * g), -1.0, 1.0);
  }
  float sn = sqrt(max(0.0, 1.0 - c * c)), phi = TWO_PI * u2;
  vec3 b1, b2;
  onb(d, b1, b2);
  return normalize(b1 * (sn * cos(phi)) + b2 * (sn * sin(phi)) + d * c);
}
#endif

// Split compare of the multiple-scattering compensation: off left of uSplitX (a fraction of the width), on right.
uniform int uCompareMS;
uniform float uSplitX;

void main() {
  ivec2 pix = ivec2(gl_FragCoord.xy);
  uint pixSeed = pcg3d(uvec3(uvec2(pix), 0x9e37u)).x;
  gMultiScatter = uCompareMS == 1 ? gl_FragCoord.x >= uSplitX * uResolution.x : uMultiScatter == 1;
  vec3 acc0 = vec3(0.0), acc1 = vec3(0.0), acc2 = vec3(0.0), acc3 = vec3(0.0);
  int nL = uNumLights + uEnvOn; // lights sampled at each vertex: the softboxes, then the environment
  float accMask = 0.0;
  vec3 accAlbedo = vec3(0.0), accN = vec3(0.0);
  float accY = 0.0, accY2 = 0.0;

  for (int sIdx = 0; sIdx < uSppNew; sIdx++) {
    uint sampleIndex = uint(uSppDone + sIdx);
    vec2 jitter = sample4(sampleIndex, pixSeed, 0, 7).xy;
    vec2 ndc = ((gl_FragCoord.xy - 0.5 + jitter) / uResolution) * 2.0 - 1.0;
    vec3 ro = uCamPos;
    vec3 rd = normalize(uCamFwd + ndc.x * uTanHalf.x * uCamRight + ndc.y * uTanHalf.y * uCamUp);

    vec3 L = vec3(0.0), L1 = vec3(0.0), L2 = vec3(0.0), L3 = vec3(0.0); // per light: key, fill, rim, environment
    vec3 firstAlbedo = vec3(0.0), firstN = vec3(0.0); // at the camera ray's hit (zero on a miss)
    vec3 beta = vec3(1.0);
    float mask = 0.0;
    float prevPdf = 0.0;
    vec3 prevP = ro;
    // Next-event estimation only samples lights on the incident side, so a light reached through a refraction had
    // no light-sampling counterpart: it takes the full weight.
    bool prevTrans = false;
    // Dispersion: each path carries one wavelength through dispersive glass, drawn here and committed (weighted by
    // its ACEScg response) at its first refraction through such glass. Drawn from the pixel's own scrambled
    // Sobol' sequence (a dimension nothing else uses), so a pixel's samples sweep the spectrum evenly instead of
    // landing at random: far less color noise for the same sample count.
    float lambda = 380.0 + 400.0 * sample4(sampleIndex, pixSeed, 0, 6).x;
    bool colored = false;

    // The path runs as a sequence of steps that each cast exactly one ray: the camera ray, then at every vertex
    // one shadow ray per softbox (next-event estimation, solid-angle sampling) and the BSDF sample that continues
    // the path. So the scene is traced, and evalSurf evaluated, from one place each: the Direct3D compiler inlines
    // every call, and these two are most of the shader's compile time.
    int depth = 0;
    bool atVertex = false; // a surface vertex is set up and its directions are being sampled
    int k = 0;             // at a vertex, the next direction: k < uNumLights is softbox k, k == uNumLights the BSDF
    // The current vertex.
    vec3 p = ro, po = ro, ng = vec3(0.0, 1.0, 0.0), t1 = vec3(1.0, 0.0, 0.0), t2 = vec3(0.0, 0.0, 1.0);
    vec3 n = ng, wo = vec3(0.0, 0.0, 1.0);
    Mat m = uMat[0];
    Surf s;
    int filt = 0;
    bool solidInside = false, last = false;
    vec3 pending = vec3(0.0); // what the shadow ray in flight delivers if it gets through
    int maxSteps = uMaxBounces * (nL + 1) + 1;
    // Subsurface random walk (spec, Subsurface). The path refracts into the medium through the material's own
    // rough dielectric surface (so the specular reflection and the Fresnel split are OpenPBR's), then walks:
    // free flights with extinction 1 / mean free path per channel, scattering with the single-scattering albedo
    // the spec derives from subsurface_color (its inversion of van de Hulst's relation), in Henyey-Greenstein
    // directions. The walk leaves at the first surface it reaches, through a Lambertian exit, as in Cycles' random
    // walk: light that has scattered many times beneath a surface leaves it close to diffusely, and an exact
    // rough-dielectric exit (tried first) made every exit's light depend on the angle the walk happened to arrive
    // at, sixteen times plastic's noise at the same sample count. Colors: each walk samples its flights by one
    // hero channel's extinction, picked on entry, and tracks the walk's probability under every channel relative to
    // the hero's (walkR); light leaving is weighted by the walk's balance heuristic over the three channels, as in
    // pbrt-v4's chromatic media (weights per flight multiply up to 3 each: unbiased but heavy-tailed, the furnace
    // read 0.7-1.1). Walk events spend uMaxScatter, not the bounces.
    bool vBack = false;   // the current vertex was reached from inside its object
    bool sssExit = false; // the current vertex is where a walk left its medium (shaded as Lambertian)
    int sssCross = 0;     // media entered: the light that comes out is direct light, exempt from the indirect clamp
#ifdef SSS
    bool inSSS = false;
    vec3 sigT = vec3(1.0), ssAlb = vec3(0.0), walkR = vec3(1.0);
    vec3 heroMask = vec3(1.0, 0.0, 0.0); // the hero channel, one-hot (a dynamic vector index costs Direct3D dearly)
    float hgG = 0.0, betaIn = 1.0;
    int scatters = 0;
    uint walkSeed = pcg3d(uvec3(uvec2(pix), sampleIndex)).x;
    maxSteps += uMaxScatter;
#endif

    for (int step = 0; step < maxSteps; step++) {
      vec3 tro = ro, trd = rd;
      float tMax = 1e30;
      bool shadow = false;
      if (atVertex) {
        bool isLight = k < nL;
        if (!isLight && last) break; // the continuation ray from the last vertex is never traced
        if (isLight && (!lightOn(k) || solidInside)) { k++; continue; } // from inside a solid, every light is behind its wall
        vec4 u = sample4(sampleIndex, pixSeed, depth, k);
        vec3 dirW, wi;
        float ldist = 0.0, lpdf = 0.0;
        if (isLight) {
          bool got;
          if (k == ENV_LIGHT) {
            got = sampleEnv(u, dirW, lpdf);
            ldist = 1e30;
          } else {
            got = sampleLight(k, po, u.xy, dirW, ldist, lpdf);
          }
          if (!got) { k++; continue; }
          wi = vec3(dot(dirW, t1), dot(dirW, t2), dot(dirW, n));
          if (wi.z <= 0.0) { k++; continue; }
        } else {
#ifdef SSS
          if (sssExit) wi = sampleCosine(u.xy);
          else
#endif
          if (!sampleSurf(s, wo, filt, u.xyz, wi)) break;
          dirW = normalize(t1 * wi.x + t2 * wi.y + n * wi.z);
        }
        vec3 fD, fS;
        float bpdf;
#ifdef SSS
        if (sssExit) {
          fD = vec3(wi.z / PI); // Lambertian exit, white: the walk's weight carries the color
          fS = vec3(0.0);
          bpdf = wi.z / PI;
        } else
#endif
        {
          bpdf = evalSurf(s, wo, wi, filt, fD, fS);
        }
        vec3 f = fD + fS;
        if (isLight) {
          if (maxc(f) <= 0.0) { k++; continue; }
          // At the last vertex the BSDF-sampled ray is never traced, so light sampling takes the full weight.
          float wl = last ? 1.0 : powerHeuristic(lpdf, bpdf);
          vec3 Le;
          if (k == ENV_LIGHT) Le = envLe(dirW);
          else Le = uLightRadiance[k];
          pending = beta * f * Le * (wl / lpdf);
          tro = po;
          trd = dirW;
          tMax = ldist * (1.0 - 1e-4);
          shadow = true;
        } else {
          if (bpdf <= 0.0) break;
          beta *= f / bpdf;
          if (maxc(beta) <= 0.0) break;
          if (depth >= 3) {
            float q = min(maxc(beta), 0.95);
            if (u.w >= q) break;
            beta /= q;
          }
          // A refracted path leaves from the far side of the surface.
          bool through = wi.z < 0.0;
#if defined(GLASS) && !defined(NO_DISPERSION)
          if (through && !colored && disperses(m)) {
            colored = true;
            beta *= spectralWeight(lambda);
          }
#endif
          ro = through ? offsetRay(p, -ng) : po;
#ifdef THIN
          // Straight through smooth thin glass, the path keeps the previous vertex's MIS state: shadow rays from
          // that vertex already sample lights through the glass (see trace).
          bool passed = through && s.thin && uThinGlass == 1 && smoothThin(m);
#else
          const bool passed = false;
#endif
#ifdef SSS
          if (through && !sssExit && !s.thin && !vBack && m.subsurface_weight > 0.0) {
            // Into the medium. The refraction carried radiance's 1 / eta^2; the Lambertian exit does not give it
            // back, so undo it here (the walk carries the light that entered).
            inSSS = true;
            sssCross++;
            beta *= sq(s.etaT);
            betaIn = max(maxc(beta), 1e-20);
            float uh = walkRand(walkSeed) * 3.0;
            heroMask = vec3(float(uh < 1.0), float(uh >= 1.0 && uh < 2.0), float(uh >= 2.0));
            walkR = vec3(1.0);
            vec3 r = max(m.subsurface_radius * m.subsurface_radius_scale, vec3(1e-6));
            sigT = 1.0 / r;
            hgG = clamp(m.subsurface_scatter_anisotropy, -0.95, 0.95);
            vec3 C = sat3(m.subsurface_color);
            vec3 sv = 4.09712 + 4.20863 * C - sqrt(9.59217 + 41.6808 * C + 17.7126 * C * C);
            ssAlb = sat3((1.0 - sv * sv) / (1.0 - hgG * sv * sv));
          }
#endif
          if (!passed) {
            prevPdf = bpdf;
            prevTrans = through;
            prevP = ro;
          }
          rd = dirW;
          tro = ro;
          trd = rd;
          atVertex = false;
          depth++;
        }
      }

      bool walkExit = false; // this step's walk reached the surface
#ifdef SSS
      float tFlight = 1e30;
      if (!shadow && inSSS) {
        tFlight = -log(max(1.0 - walkRand(walkSeed), 1e-30)) / dot(sigT, heroMask);
        tMax = tFlight;
      }
#endif
      vec3 T = vec3(1.0);
      Hit h = trace(tro, trd, tMax, shadow, !shadow && depth > 0, T);

      if (shadow) {
        if (h.mat < 0 && maxc(T) > 0.0) {
          vec3 c = pending * T;
          addLight(k, depth - sssCross >= 1 ? clampIndirect(c) : c, L, L1, L2, L3);
        }
        k++;
        continue;
      }
#ifdef SSS
      if (inSSS) {
        bool scattered = h.mat < 0; // no surface before the flight ends
        float tt = scattered ? tFlight : h.t;
        vec3 Tr = exp(-sigT * tt);
        // Each channel's flight density (scattered) or probability of flying this far (surface), over the hero's,
        // which sampled the distance (so its value never underflows).
        vec3 pc = scattered ? sigT * Tr : Tr;
        float ph = max(dot(pc, heroMask), 1e-30);
        beta *= (scattered ? ssAlb * sigT * Tr : Tr) / ph; // sigma_s Tr when it scatters
        walkR *= pc / ph;
        // Only beta / mean(walkR) matters until the walk ends: keep mean(walkR) at 1, so neither drifts out of
        // range on a long walk.
        float wn = dot(walkR, vec3(1.0 / 3.0));
        beta /= wn;
        walkR /= wn;
        if (scattered) {
          if (++scatters > uMaxScatter || maxc(beta) <= 0.0) break;
          // Russian roulette only once the walk has lost weight (absorption): a floor on the kill rate (say 1% per
          // event) leaves a long lossless walk alive at odds like 1 in 25,000 with a matching weight, so few that
          // the mean reads low (the furnace on Winged Victory read 0.90-0.96 that way).
          float q = min(maxc(beta) / betaIn, 1.0);
          if (q < 1.0) {
            if (walkRand(walkSeed) >= q) break;
            beta /= q;
          }
          ro += rd * tt;
          rd = sampleHG(rd, hgG, walkRand(walkSeed), walkRand(walkSeed));
          continue;
        }
        // Reached the surface: the walk leaves here.
        inSSS = false;
        walkExit = true;
      }
#endif

      if (h.light >= 0) {
        // Softbox reached by BSDF sampling (lights are invisible to camera rays). MIS vs light sampling. It emits
        // but does not block what lies behind it, as shadow rays never see it: so an environment light sampled
        // through a softbox's place and one reached past it agree (and a switched-off softbox leaves no hole).
        float pl = lightPdf(h.light, prevP, rd, distance(prevP, ro + rd * h.lightT));
        vec3 c = beta * uLightRadiance[h.light] * (prevTrans ? 1.0 : powerHeuristic(prevPdf, pl));
        addLight(h.light, depth - sssCross >= 2 ? clampIndirect(c) : c, L, L1, L2, L3);
      }
      if (h.mat < 0) {
        if (!(depth == 0 && uPass >= 3)) {
          if (uEnvOn == 1) {
            // The camera sees the softened backdrop; a bounce sees the environment as a light, weighted against
            // sampling it from the vertex before (none was made through a refraction).
            vec3 c = beta * (depth == 0 ? envBackdrop(rd) : envLe(rd)) * (depth == 0 || prevTrans ? 1.0 : powerHeuristic(prevPdf, envPdf(rd)));
            L3 += depth - sssCross >= 2 ? clampIndirect(c) : c;
          } else {
            vec3 c = beta * uEnv;
            L += depth - sssCross >= 2 ? clampIndirect(c) : c;
          }
        }
        break;
      }

      p = tro + trd * h.t;
      // A walk's exit is a white Lambertian surface facing out of the medium (the trace's normals face the ray);
      // it shares the vertex setup below but takes no material.
      sssExit = walkExit;
      bool flip = walkExit && h.back;
      n = flip ? -h.n : h.n;
      ng = flip ? -h.ng : h.ng;
      m = getMat(h.mat);
      vBack = h.back && !walkExit;
#ifdef GLASS
      solidInside = vBack && (m.transmission_weight > 0.0 || m.subsurface_weight > 0.0) && m.geometry_thin_walled < 0.5;
      if (solidInside && m.transmission_depth > 0.0) {
        // The segment just traced ran through the medium: Beer-Lambert with mu_t = -ln(color) / depth (spec).
        beta *= exp(log(clamp(m.transmission_color, 1e-4, 1.0)) / m.transmission_depth * h.t);
      }
#endif

      if (depth == 0) {
        mask = h.mat != 3 ? 1.0 : 0.0; // everything but the floor
        firstAlbedo = albedoAOV(m);
        firstN = n;
        if (uPass == 3) { L = firstAlbedo; break; }
        if (uPass == 4) { L = n * 0.5 + 0.5; break; }
      }
      filt = depth == 0 ? (uPass == 1 ? 1 : (uPass == 2 ? 2 : 0)) : 0;

      onb(n, t1, t2);
      if (m.specular_roughness_anisotropy > 0.0 && !walkExit) {
        // Anisotropic lobes need a tangent field: the surface is brushed around the vertical axis, as on a lathe,
        // so the tangent (the rougher direction) runs pole to pole and highlights stretch that way.
        vec3 Tg = vec3(0.0, 1.0, 0.0) - n * n.y;
        float lT = length(Tg);
        if (lT > 1e-4) {
          t1 = Tg / lT;
          t2 = cross(n, t1);
        }
      }
      wo = vec3(dot(-rd, t1), dot(-rd, t2), dot(-rd, n));
      if (wo.z <= 1e-6 && !walkExit) break;
      if (!walkExit) s = setupSurf(m, wo, h.back, lambda, colored);
#ifdef MESH
      s.nf = flakeNormal(m, transpose(uModelRot) * (p - uModelPos));
#else
      s.nf = flakeNormal(m, p - vec3(ballX(2), 1.0, 0.0));
#endif
      po = offsetRay(p, ng); // off the true surface, on the side the ray came from
      last = depth == uMaxBounces - 1;
      atVertex = true;
      k = 0;
    }
    // A sample with any non-finite light is dropped whole, so the light images stay consistent.
    vec3 all = L + L1 + L2 + L3;
    if (!(nonFinite(all.r) || nonFinite(all.g) || nonFinite(all.b))) {
      acc0 += L;
      acc1 += L1;
      acc2 += L2;
      acc3 += L3;
      float Y = lum(all);
      accY += Y;
      accY2 += Y * Y;
    }
    accMask += mask;
    accAlbedo += firstAlbedo;
    accN += firstN;
  }

  // The running means: each pass's new samples blend in with weight new / (done + new).
  float nNew = float(uSppNew);
  float w = uSppDone == 0 ? 1.0 : nNew / (float(uSppDone) + nNew);
  // min(): stay finite on RGBA16F targets.
  vec4 cur0 = min(vec4(acc0 / nNew, accMask / nNew), vec4(65504.0));
  vec4 cur1 = min(vec4(acc1 / nNew, accY2 / nNew), vec4(65504.0));
  vec4 cur2 = min(vec4(acc2 / nNew, 1.0), vec4(65504.0));
  vec4 cur3 = min(vec4(acc3 / nNew, 1.0), vec4(65504.0));
  if (uSppDone == 0) {
    outKey = cur0;
    outFill = cur1;
    outRim = cur2;
    outEnv = cur3;
  } else {
    outKey = mix(texelFetch(uPrev0, pix, 0), cur0, w);
    outFill = mix(texelFetch(uPrev1, pix, 0), cur1, w);
    outRim = mix(texelFetch(uPrev2, pix, 0), cur2, w);
    outEnv = mix(texelFetch(uPrev3, pix, 0), cur3, w);
  }
#ifdef AUX
  vec4 cur4 = min(vec4(accAlbedo / nNew, accY / nNew), vec4(65504.0));
  vec4 cur5 = vec4(accN / nNew, 1.0);
  outAux0 = uSppDone == 0 ? cur4 : mix(texelFetch(uPrev4, pix, 0), cur4, w);
  outAux1 = uSppDone == 0 ? cur5 : mix(texelFetch(uPrev5, pix, 0), cur5, w);
#endif
}
`
}


// ------------------------------------------------------------------------------------------------------------
// Display: exposure in EV, then a view transform. ACES 2.0 is the real Output Transform baked from OpenColorIO
// 2.5 (cg-config-v4.0.0_aces-v2.0_ocio-v2.5, 'sRGB - Display' / 'ACES 2.0 - SDR 100 nits (Rec.709)') into a
// 65^3 LUT behind a log2 shaper. AgX and Khronos PBR Neutral take linear Rec.709.
// ------------------------------------------------------------------------------------------------------------
export const DISPLAY_FRAG = /* glsl */ `${HEADER}
${COMMON}
uniform sampler2D uAccum0; // key (and the rig's faint environment); the AOV passes use this one alone
uniform sampler2D uAccum1; // fill
uniform sampler2D uAccum2; // rim
uniform sampler2D uAccum3; // HDR environment
uniform vec3 uMix[4];      // light mixer: each light's color times its intensity (traced in white)
uniform int uAccumDiv;  // 1 for the accumulation; k for a 1/k-resolution preview (nearest, like an IPR proxy)
uniform sampler3D uAcesLut;
uniform int uAcesReady;
uniform float uExposure;
uniform int uView;   // 0 ACES 2.0, 1 AgX, 2 PBR Neutral, 3 Standard
uniform int uPass;
uniform sampler2D uDenoised; // the denoiser's output (mixed radiance)
uniform float uDenoiseMix;   // its share of the displayed image: 1 while noisy, falling to 0 as the render converges
// Split compare (uCompare): 1, raw left of uSplitPx and fully denoised right; 2, ACES 2.0 left and AgX right.
uniform int uCompare;
uniform float uSplitPx;
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
  ivec2 px = ivec2(gl_FragCoord.xy) / uAccumDiv;
  vec3 c = texelFetch(uAccum0, px, 0).rgb;
  if (uPass == 4) { outColor = vec4(c, 1.0); return; }                                   // normals: raw data
  if (uPass == 3) { outColor = vec4(srgbOETF(max(AP1_TO_REC709 * c, 0.0)), 1.0); return; } // albedo
  c = c * uMix[0] + texelFetch(uAccum1, px, 0).rgb * uMix[1] + texelFetch(uAccum2, px, 0).rgb * uMix[2] +
    texelFetch(uAccum3, px, 0).rgb * uMix[3];
  bool right = gl_FragCoord.x >= uSplitPx;
  float dm = uCompare == 1 ? (right ? 1.0 : 0.0) : uDenoiseMix;
  if (dm > 0.0) c = mix(c, texelFetch(uDenoised, px, 0).rgb, dm);
  int view = uCompare == 2 ? (right ? 1 : 0) : uView;
  c = max(c, 0.0) * exp2(uExposure);
  if (view == 0 && uAcesReady == 1) { outColor = vec4(aces2(c), 1.0); return; }
  vec3 r = max(AP1_TO_REC709 * c, 0.0);
  if (view == 2) r = PBRNeutralToneMapping(r);
  else if (view != 3) r = AgXToneMapping(r); // AgX, and the fallback while the ACES LUT loads
  outColor = vec4(srgbOETF(r), 1.0);
}
`

// ------------------------------------------------------------------------------------------------------------
// Denoiser (optional, labelled in the lab): an edge-avoiding a-trous wavelet filter, Dammertz et al. 2010,
// "Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering" (HPG), with the luminance
// edge-stopping scaled by each pixel's variance as in Schied et al. 2017, "Spatiotemporal Variance-Guided
// Filtering" (SVGF, HPG). It filters the mixed image divided by the first-hit albedo, so texture and material
// edges are restored afterward, and it only ever changes what is displayed: the accumulated images, the furnace
// test and the readouts stay raw. As samples accumulate, each pixel's variance falls and the filter backs off,
// and the display hands over to the raw render (uDenoiseMix), so the finished image is unfiltered.
// ------------------------------------------------------------------------------------------------------------
const DENOISE_COMMON = /* glsl */ `
uniform ivec2 uSize; // the image being filtered (the accumulation, or a preview's part of its target)
ivec2 clampPx(ivec2 q) { return clamp(q, ivec2(0), uSize - 1); }
// Albedo to divide out: misses (no albedo) and near-black surfaces keep their radiance as is.
vec3 demodBase(vec3 a) { return lum(a) < 1e-3 ? vec3(1.0) : max(a, vec3(0.02)); }
`

// Mixed radiance over albedo with the variance of its luminance (of the pixel's mean), and the guide the
// filter compares pixels by: unit normal and albedo luminance, in one texel. The variance comes from the
// pixel's own samples (their second moment); with fewer than four (a preview, the first passes), from its
// 3 x 3 neighborhood instead, as SVGF does before a pixel has history.
export const DENOISE_PREP_FRAG = /* glsl */ `${HEADER}
${COMMON}
${DENOISE_COMMON}
uniform sampler2D uAccum0;
uniform sampler2D uAccum1;
uniform sampler2D uAccum2;
uniform sampler2D uAccum3;
uniform sampler2D uAux0;
uniform sampler2D uAux1;
uniform vec3 uMix[4];
uniform float uSpp; // samples per pixel in the image
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outGuide;

vec3 demod(ivec2 q) {
  vec3 c = texelFetch(uAccum0, q, 0).rgb * uMix[0] + texelFetch(uAccum1, q, 0).rgb * uMix[1] + texelFetch(uAccum2, q, 0).rgb * uMix[2] +
    texelFetch(uAccum3, q, 0).rgb * uMix[3];
  return max(c, 0.0) / demodBase(texelFetch(uAux0, q, 0).rgb);
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 d = demod(p);
  vec4 aux0 = texelFetch(uAux0, p, 0);
  float n = uSpp;
  float var;
  if (n >= 4.0) {
    // Variance of the mean of the white luminance Y, scaled to this pixel's mixed and demodulated value
    // (exact when the lights keep their ratio from sample to sample).
    float m1 = aux0.a;
    float m2 = texelFetch(uAccum1, p, 0).a;
    float s = lum(d) / max(m1, 1e-6);
    var = max(m2 - m1 * m1, 0.0) / (n - 1.0) * s * s;
  } else {
    float sum = 0.0, sum2 = 0.0;
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) {
        float l = lum(demod(clampPx(p + ivec2(dx, dy))));
        sum += l;
        sum2 += l * l;
      }
    var = max(sum2 / 9.0 - sq(sum / 9.0), 0.0) / max(n, 1.0);
  }
  vec3 nrm = texelFetch(uAux1, p, 0).rgb;
  float len = length(nrm);
  outColor = vec4(d, var);
  outGuide = vec4(len < 0.1 ? vec3(0.0) : nrm / len, lum(aux0.rgb)); // zero normal: no surface (background)
}
`

// One a-trous level: a 3 x 3 kernel (1/4, 1/2, 1/4) with holes, taps uStep pixels apart, weighted by normal,
// albedo and luminance similarity. The variance is filtered alongside with squared weights. The first level
// steers by a 3 x 3 blur of the variance (a single pixel's estimate is noisy); later ones by the filtered
// variance. The last level multiplies the albedo back in.
export const ATROUS_FRAG = /* glsl */ `${HEADER}
${COMMON}
${DENOISE_COMMON}
uniform sampler2D uIn;    // demodulated color, a = variance
uniform sampler2D uGuide; // unit normal (zero for background), albedo luminance
uniform sampler2D uAux0;  // albedo, for the last level
uniform int uStep;
uniform int uLast;
out vec4 outColor;

const float SIGMA_L = 4.0;   // luminance edge-stopping, in standard deviations (SVGF)
const float SIGMA_N = 128.0; // normal edge-stopping exponent (SVGF)
const float SIGMA_A = 0.05;  // albedo luminance difference that cuts a tap's weight by e

float h(int i) { return i == 0 ? 0.5 : 0.25; }

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 cP = texelFetch(uIn, p, 0);
  vec4 gP = texelFetch(uGuide, p, 0);
  bool bgP = dot(gP.xyz, gP.xyz) < 0.01;
  float lP = lum(cP.rgb);
  float v = cP.a;
  if (uStep == 1) {
    v = 0.0;
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) v += h(dx) * h(dy) * texelFetch(uIn, clampPx(p + ivec2(dx, dy)), 0).a;
  }
  float sigL = SIGMA_L * sqrt(max(v, 0.0)) + 1e-5;

  vec3 sum = vec3(0.0);
  float wSum = 0.0, vSum = 0.0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = p + ivec2(dx, dy) * uStep;
      if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, uSize))) continue;
      vec4 cQ = texelFetch(uIn, q, 0);
      vec4 gQ = texelFetch(uGuide, q, 0);
      bool bgQ = dot(gQ.xyz, gQ.xyz) < 0.01;
      // Background (no surface) only mixes with background.
      float wN = (bgP || bgQ) ? float(bgP && bgQ) : pow(max(dot(gP.xyz, gQ.xyz), 0.0), SIGMA_N);
      float wA = exp(-abs(gP.w - gQ.w) / SIGMA_A);
      float wL = exp(-abs(lP - lum(cQ.rgb)) / sigL);
      float w = h(dx) * h(dy) * wN * wA * wL;
      sum += w * cQ.rgb;
      wSum += w;
      vSum += w * w * cQ.a;
    }
  vec3 c = sum / wSum; // the center tap always has weight h(0)^2
  float var = vSum / (wSum * wSum);
  if (uLast == 1) c *= demodBase(texelFetch(uAux0, p, 0).rgb);
  outColor = vec4(c, var);
}
`
