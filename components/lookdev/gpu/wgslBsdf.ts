// WGSL ports of shaders.ts's GGX microfacets, albedo tables, EON diffuse, fuzz, thin film and the OpenPBR layering.
// Every comment on the models' sources and deviations lives with the GLSL original; these are its translation.

export const MICROFACET = /* wgsl */ `
// NDF, Heitz 2014 (JCGT 3(2)); isotropic case is Walter et al. 2007 eq. 33.
fn D_GGX(m: vec3f, alpha: vec2f) -> f32 {
  if (m.z <= 0.0) { return 0.0; }
  let he = m.xy / alpha;
  let d = dot(he, he) + m.z * m.z;
  return 1.0 / (PI * alpha.x * alpha.y * d * d);
}
fn T_GGX(w: vec3f, alpha: vec2f) -> f32 { let aw = alpha * w.xy; return sqrt(dot(aw, aw) + w.z * w.z); }
fn G1_GGX(w: vec3f, alpha: vec2f) -> f32 { let z = abs(w.z); return 2.0 * z / (z + T_GGX(w, alpha)); }
fn G2_GGX(i: vec3f, o: vec3f, alpha: vec2f) -> f32 {
  let zi = abs(i.z);
  let zo = abs(o.z);
  return 2.0 * zi * zo / (T_GGX(i, alpha) * zo + T_GGX(o, alpha) * zi);
}
// Visible-normal sampling with spherical caps: Dupuy and Benyoub 2023.
fn sampleVNDF_SphericalCap(u: vec2f, wi: vec3f, alpha: vec2f) -> vec3f {
  let wiStd = normalize(vec3f(wi.xy * alpha, wi.z));
  let phi = TWO_PI * u.x;
  let z = (1.0 - u.y) * (1.0 + wiStd.z) - wiStd.z;
  let sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  let h = vec3f(sinTheta * cos(phi), sinTheta * sin(phi), z) + wiStd;
  return normalize(vec3f(h.xy * alpha, h.z));
}
// Bounded VNDF sampling, reflection only: Tokuyoshi and Eto 2024.
fn boundedVNDF_k(i: vec3f, alpha: vec2f) -> f32 {
  let a = clamp(min(alpha.x, alpha.y), 0.0, 1.0);
  let s = 1.0 + length(i.xy);
  let a2 = a * a;
  let s2 = s * s;
  return (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
}
fn sampleGGXReflection_Bounded(rand: vec2f, i: vec3f, alpha: vec2f) -> vec3f {
  let iStd = normalize(vec3f(i.xy * alpha, i.z));
  let phi = TWO_PI * rand.x;
  let k = boundedVNDF_k(i, alpha);
  let lowerBound = select(-iStd.z, -k * iStd.z, i.z > 0.0);
  let z = lowerBound * rand.y + (1.0 - rand.y);
  let sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  let mStd = iStd + vec3f(sinTheta * cos(phi), sinTheta * sin(phi), z);
  let m = normalize(vec3f(mStd.xy * alpha, mStd.z));
  return 2.0 * dot(i, m) * m - i;
}
fn pdfGGXReflection_Bounded(i: vec3f, o: vec3f, alpha: vec2f) -> f32 {
  let m = normalize(i + o);
  let ndf = D_GGX(m, alpha);
  let ai = alpha * i.xy;
  let len2 = dot(ai, ai);
  let t = sqrt(len2 + i.z * i.z);
  if (i.z >= 0.0) { return ndf / (2.0 * (boundedVNDF_k(i, alpha) * i.z + t)); }
  return ndf * (t - i.z) / (2.0 * len2);
}
// Exact unpolarized dielectric Fresnel, Walter et al. 2007 eq. 22; eta = eta_t / eta_i.
fn fresnelDielectric(cosThetaI: f32, eta: f32) -> f32 {
  let c = abs(cosThetaI);
  let g2 = eta * eta - 1.0 + c * c;
  if (g2 < 0.0) { return 1.0; }
  let g = sqrt(g2);
  let A = (g - c) / (g + c);
  let B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
  return 0.5 * A * A * (1.0 + B * B);
}
fn f0FromEta(eta: f32) -> f32 { return sq((eta - 1.0) / (eta + 1.0)); }
// OpenPBR metal Fresnel, "F82-tint".
fn fresnelSchlick(F0: vec3f, mu: f32) -> vec3f {
  let m = clamp(1.0 - mu, 0.0, 1.0);
  let m2 = m * m;
  return F0 + (1.0 - F0) * (m2 * m2 * m);
}
const MU_BAR: f32 = 1.0 / 7.0;
fn fresnelF82Tint(mu: f32, F0: vec3f, tint: vec3f) -> vec3f {
  let b1 = 1.0 - MU_BAR;
  let b3 = b1 * b1 * b1;
  let a = fresnelSchlick(F0, MU_BAR) * (1.0 - tint) / (MU_BAR * b3 * b3);
  let x = clamp(mu, 0.0, 1.0);
  let y = 1.0 - x;
  let y3 = y * y * y;
  return max(fresnelSchlick(F0, x) - a * x * (y3 * y3), vec3f(0.0));
}
fn favgF82Tint(F0: vec3f, tint: vec3f) -> vec3f {
  let b1 = 1.0 - MU_BAR;
  let b3 = b1 * b1 * b1;
  let a = fresnelSchlick(F0, MU_BAR) * (1.0 - tint) / (MU_BAR * b3 * b3);
  return F0 + (1.0 - F0) * (1.0 / 21.0) - a * (1.0 / 126.0);
}
`

export const TABLE_CONSTS = /* wgsl */ `
const E_X_MAX: f32 = 0.5;   // eta up to 3
const E_DIMS = vec3f(32.0, 32.0, 16.0);
fn etaCoord(eta: f32) -> f32 { return clamp(((eta - 1.0) / (eta + 1.0)) / E_X_MAX, 0.0, 1.0); }
fn eTableCoord(mu: f32, r: f32, eta: f32) -> vec3f {
  return (vec3f(clamp(mu, 0.0, 1.0), clamp(r, 0.0, 1.0), etaCoord(eta)) * (E_DIMS - 1.0) + 0.5) / E_DIMS;
}
`

export const EON = /* wgsl */ `
const CONSTANT1_FON: f32 = 0.5 - 2.0 / (3.0 * PI);
const CONSTANT2_FON: f32 = 2.0 / 3.0 - 28.0 / (15.0 * PI);
const EON_EPS: f32 = 1.0e-7;
fn E_FON_exact(muIn: f32, r: f32) -> f32 {
  let AF = 1.0 / (1.0 + CONSTANT1_FON * r);
  let BF = r * AF;
  let mu = clamp(muIn, -1.0, 1.0);
  let Si = sqrt(max(0.0, 1.0 - mu * mu));
  let G = Si * (acos(mu) - Si * mu) + (2.0 / 3.0) * (Si * mu * (1.0 + Si + Si * Si) / (1.0 + Si) - Si);
  return AF + (BF * INV_PI) * G;
}
fn E_FON_avg(r: f32) -> f32 { return (1.0 + CONSTANT2_FON * r) / (1.0 + CONSTANT1_FON * r); }
fn f_EON(rho: vec3f, r: f32, wi: vec3f, wo: vec3f) -> vec3f {
  let mu_i = wi.z;
  let mu_o = wo.z;
  if (mu_i < EON_EPS || mu_o < EON_EPS) { return vec3f(0.0); }
  let s = dot(wi, wo) - mu_i * mu_o;
  let sovertF = select(s, s / max(mu_i, mu_o), s > 0.0);
  let AF = 1.0 / (1.0 + CONSTANT1_FON * r);
  let f_ss = (rho * INV_PI) * AF * (1.0 + r * sovertF);
  let EFo = E_FON_exact(mu_o, r);
  let EFi = E_FON_exact(mu_i, r);
  let avgEF = AF * (1.0 + CONSTANT2_FON * r);
  let rho_ms = (rho * rho) * avgEF / (vec3f(1.0) - rho * (1.0 - avgEF));
  let f_ms = (rho_ms * INV_PI) * max(EON_EPS, 1.0 - EFo) * max(EON_EPS, 1.0 - EFi) / max(EON_EPS, 1.0 - avgEF);
  return f_ss + f_ms;
}
fn E_EON(rho: vec3f, r: f32, mu: f32) -> vec3f {
  let EF = E_FON_exact(mu, r);
  let avgEF = E_FON_avg(r);
  let rho_ms = (rho * rho) * avgEF / (vec3f(1.0) - rho * (1.0 - avgEF));
  return rho * EF + rho_ms * (1.0 - EF);
}
fn orthonormal_basis_ltc(w: vec3f) -> mat3x3f {
  let lenSqr = dot(w.xy, w.xy);
  let X = select(vec3f(1.0, 0.0, 0.0), vec3f(w.x, w.y, 0.0) * inverseSqrt(max(lenSqr, 1e-30)), lenSqr > 0.0);
  let Y = vec3f(-X.y, X.x, 0.0);
  return mat3x3f(X, Y, vec3f(0.0, 0.0, 1.0));
}
// a, b, c, d of the LTC fit.
fn ltc_coeffs(muIn: f32, r: f32) -> vec4f {
  let mu = clamp(muIn, 0.0, 1.0);
  let a = 1.0 + r * (0.303392 + (-0.518982 + 0.111709 * mu) * mu + (-0.276266 + 0.335918 * mu) * r);
  let b = r * (-1.16407 + 1.15859 * mu + (0.150815 - 0.150105 * mu) * r) / (mu * mu * mu - 1.43545);
  let c = 1.0 + r * (0.20013 + (-0.506373 + 0.261777 * mu) * mu);
  let d = r * (0.540852 + (-1.01625 + 0.475392 * mu) * mu) / (-1.0743 + (0.0725628 + mu) * mu);
  return vec4f(a, b, c, d);
}
fn cltc_sample(wo: vec3f, r: f32, u1: f32, u2: f32) -> vec4f {
  let k = ltc_coeffs(wo.z, r);
  let R = sqrt(u1);
  let phi = TWO_PI * u2;
  var x = R * cos(phi);
  let y = R * sin(phi);
  let vz = 1.0 / sqrt(k.w * k.w + 1.0);
  let s = 0.5 * (1.0 + vz);
  x = -mix(sqrt(max(0.0, 1.0 - y * y)), x, s);
  let wh = vec3f(x, y, sqrt(max(1.0 - (x * x + y * y), 0.0)));
  let pdf_wh = wh.z / (PI * s);
  var wi = vec3f(k.x * wh.x + k.y * wh.z, k.z * wh.y, k.w * wh.x + wh.z);
  let len = length(wi);
  let detM = k.z * (k.x - k.y * k.w);
  let pdf_wi = pdf_wh * len * len * len / detM;
  wi = normalize(orthonormal_basis_ltc(wo) * wi);
  return vec4f(wi, pdf_wi);
}
fn cltc_pdf(wo: vec3f, wi_local: vec3f, r: f32) -> f32 {
  let wi = transpose(orthonormal_basis_ltc(wo)) * wi_local;
  let k = ltc_coeffs(wo.z, r);
  let detM = k.z * (k.x - k.y * k.w);
  let wh = vec3f(k.z * (wi.x - k.y * wi.z), (k.x - k.y * k.w) * wi.y, -k.z * (k.w * wi.x - k.x * wi.z));
  let lenSqr = dot(wh, wh);
  let vz = 1.0 / sqrt(k.w * k.w + 1.0);
  let s = 0.5 * (1.0 + vz);
  return detM * detM / (lenSqr * lenSqr) * max(wh.z, 0.0) / (PI * s);
}
fn uniform_lobe_sample(u1: f32, u2: f32) -> vec3f {
  let sinTheta = sqrt(max(0.0, 1.0 - u1 * u1));
  let phi = TWO_PI * u2;
  return vec3f(sinTheta * cos(phi), sinTheta * sin(phi), u1);
}
fn eon_uniform_prob(mu: f32, r: f32) -> f32 {
  let rw = select(0.0, pow(r, 0.1), r > 0.0);
  return rw * (0.162925 + (-0.372058 + (0.538233 - 0.290822 * mu) * mu) * mu);
}
fn sample_EON(wo: vec3f, r: f32, u1: f32, u2: f32) -> vec3f {
  let P_u = eon_uniform_prob(wo.z, r);
  if (P_u > 0.0 && u1 < P_u) { return uniform_lobe_sample(u1 / P_u, u2); }
  return cltc_sample(wo, r, (u1 - P_u) / (1.0 - P_u), u2).xyz;
}
fn pdf_EON(wo: vec3f, wi: vec3f, r: f32) -> f32 {
  if (wo.z < EON_EPS || wi.z < EON_EPS) { return 0.0; }
  let P_u = eon_uniform_prob(wo.z, r);
  return P_u * (1.0 / TWO_PI) + (1.0 - P_u) * cltc_pdf(wo, wi, r);
}
`

export const FUZZ = /* wgsl */ `
fn fuzzDirAlbedo(x: f32, y: f32) -> f32 {
  let s = y * (0.0206607 + 1.58491 * y) / (0.0379424 + y * (1.32227 + y));
  let m = y * (-0.193854 + y * (-1.14885 + y * (1.7932 - 0.95943 * y * y))) / (0.046391 + y);
  let o = y * (0.000654023 + (-0.0207818 + 0.119681 * y) * y) / (1.26264 + y * (-1.92021 + y));
  return max(0.0, exp(-0.5 * sq((x - m) / s)) / (s * sqrt(2.0 * PI)) + o);
}
fn fuzzLtcAInv(x: f32, y: f32) -> f32 {
  return (2.58126 * x + 0.813703 * y) * y / (1.0 + 0.310327 * x * x + 2.60994 * x * y);
}
fn fuzzLtcBInv(x: f32, y: f32) -> f32 {
  return sqrt(1.0 - x) * (y - 1.0) * y * y * y / (0.0000254053 + 1.71228 * x - 1.71506 * x * y + 1.34174 * y * y);
}
fn fuzzLtcBasis(V: vec3f) -> mat3x3f {
  let X0 = vec3f(V.xy, 0.0);
  let l2 = dot(X0, X0);
  if (l2 > 0.0) {
    let X = X0 * inverseSqrt(l2);
    return mat3x3f(X, vec3f(-X.y, X.x, 0.0), vec3f(0.0, 0.0, 1.0));
  }
  return mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
}
fn fuzzD(V: vec3f, L: vec3f, rIn: f32) -> f32 {
  let NdotV = clamp(V.z, 1e-8, 1.0);
  let r = clamp(rIn, 0.01, 1.0);
  let w = transpose(fuzzLtcBasis(V)) * L;
  let aInv = fuzzLtcAInv(NdotV, r);
  let bInv = fuzzLtcBInv(NdotV, r);
  let wo = vec3f(aInv * w.x + bInv * w.z, aInv * w.y, w.z);
  let l2 = dot(wo, wo);
  return max(wo.z, 0.0) * INV_PI * sq(aInv / l2);
}
fn sampleFuzz(V: vec3f, rIn: f32, u: vec2f) -> vec3f {
  let NdotV = clamp(V.z, 1e-8, 1.0);
  let r = clamp(rIn, 0.01, 1.0);
  let phi = TWO_PI * u.x;
  let ct = sqrt(u.y);
  let st = sqrt(1.0 - u.y);
  let c = vec3f(cos(phi) * st, sin(phi) * st, ct);
  let aInv = fuzzLtcAInv(NdotV, r);
  let bInv = fuzzLtcBInv(NdotV, r);
  let w = normalize(vec3f(c.x / aInv - c.z * bInv / aInv, c.y / aInv, c.z));
  return fuzzLtcBasis(V) * w;
}
`

export const THIN_FILM = /* wgsl */ `
const XYZ_TO_REC709 = mat3x3f(
   3.2404542, -0.9692660,  0.0556434,
  -1.5371385,  1.8760108, -0.2040259,
  -0.4985314,  0.0415560,  1.0572252);
const E_WHITE_IN_REC709 = vec3f(1.2047843, 0.9483008, 0.9088427);
fn F_SchlickTF1(f0: f32, VdotH: f32) -> f32 { let x = clamp(1.0 - VdotH, 0.0, 1.0); let x2 = x * x; return f0 + (1.0 - f0) * (x * x2 * x2); }
fn F_SchlickTF3(f0: vec3f, VdotH: f32) -> vec3f { let x = clamp(1.0 - VdotH, 0.0, 1.0); let x2 = x * x; return f0 + (1.0 - f0) * (x * x2 * x2); }
fn Fresnel0ToIor(f0: vec3f) -> vec3f { let s = sqrt(f0); return (vec3f(1.0) + s) / (vec3f(1.0) - s); }
fn IorToFresnel03(t: vec3f, i: f32) -> vec3f { let q = (t - vec3f(i)) / (t + vec3f(i)); return q * q; }
fn IorToFresnel01(t: f32, i: f32) -> f32 { return sq((t - i) / (t + i)); }
fn evalSensitivity(OPD: f32, shift: vec3f) -> vec3f {
  let phase = 2.0 * PI * OPD * 1.0e-9;
  let val = vec3f(5.4856e-13, 4.4201e-13, 5.2481e-13);
  let pos = vec3f(1.6810e+06, 1.7953e+06, 2.2084e+06);
  let vr = vec3f(4.3278e+09, 9.3046e+09, 6.6121e+09);
  var xyz = val * sqrt(2.0 * PI * vr) * cos(pos * phase + shift) * exp(-sq(phase) * vr);
  xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * sq(phase));
  xyz /= 1.0685e-7;
  return REC709_TO_AP1 * ((XYZ_TO_REC709 * xyz) / E_WHITE_IN_REC709);
}
fn evalIridescence(outsideIOR: f32, eta2: f32, cosTheta1: f32, thinFilmThickness: f32, baseF0: vec3f) -> vec3f {
  let iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
  let sinTheta2Sq = sq(outsideIOR / iridescenceIor) * (1.0 - sq(cosTheta1));
  let cosTheta2Sq = 1.0 - sinTheta2Sq;
  if (cosTheta2Sq < 0.0) { return vec3f(1.0); }
  let cosTheta2 = sqrt(cosTheta2Sq);
  let R0 = IorToFresnel01(iridescenceIor, outsideIOR);
  let R12 = F_SchlickTF1(R0, cosTheta1);
  let T121 = 1.0 - R12;
  let phi12 = select(0.0, PI, iridescenceIor < outsideIOR);
  let phi21 = PI - phi12;
  let baseIOR = Fresnel0ToIor(clamp(baseF0, vec3f(0.0), vec3f(0.9999)));
  let R1 = IorToFresnel03(baseIOR, iridescenceIor);
  let R23 = F_SchlickTF3(R1, cosTheta2);
  let phi23 = select(vec3f(0.0), vec3f(PI), baseIOR < vec3f(iridescenceIor));
  let OPD = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
  let phi = vec3f(phi21) + phi23;
  let R123 = clamp(R12 * R23, vec3f(1e-5), vec3f(0.9999));
  let r123 = sqrt(R123);
  let Rs = sq(T121) * R23 / (vec3f(1.0) - R123);
  var I = R12 + Rs;
  var Cm = Rs - T121;
  for (var m = 1; m <= 2; m++) {
    Cm *= r123;
    I += Cm * 2.0 * evalSensitivity(f32(m) * OPD, f32(m) * phi);
  }
  return max(I, vec3f(0.0));
}
fn thinFilmF(cosTheta: f32, filmIor: f32, thicknessUm: f32, baseF0: vec3f) -> vec3f {
  return evalIridescence(1.0, max(filmIor, 1.0), clamp(cosTheta, 0.0, 1.0), thicknessUm * 1000.0, baseF0);
}
`

// OpenPBR Surface v1.1.1 layering, as in the GLSL tracer.
export const OPENPBR = /* wgsl */ `
struct Mat {
  base_weight: f32, base_color: vec3f, base_metalness: f32, base_diffuse_roughness: f32,
  specular_weight: f32, specular_color: vec3f, specular_roughness: f32, specular_roughness_anisotropy: f32,
  specular_ior: f32,
  coat_weight: f32, coat_color: vec3f, coat_roughness: f32, coat_ior: f32, coat_darkening: f32,
  fuzz_weight: f32, fuzz_color: vec3f, fuzz_roughness: f32,
  thin_film_weight: f32, thin_film_thickness: f32, thin_film_ior: f32,
  transmission_weight: f32, transmission_color: vec3f, transmission_depth: f32,
  transmission_dispersion_scale: f32, transmission_dispersion_abbe_number: f32,
  subsurface_weight: f32, subsurface_color: vec3f, subsurface_radius: f32, subsurface_radius_scale: vec3f,
  subsurface_scatter_anisotropy: f32,
  geometry_thin_walled: f32,
  lab_flake_coverage: f32, lab_flake_size: f32, lab_flake_tilt: f32,
  lab_fluor_weight: f32, lab_fluor_color: vec3f, lab_fluor_absorb: vec3f, lab_fluor_uv: f32, lab_uv_ratio: f32,
}

struct Surf {
  baseColor: vec3f, baseWeight: f32, metal: f32, diffRough: f32,
  specW: f32, specColor: vec3f, specRough: f32, eta: f32,
  coatW: f32, coatRough: f32, coatIor: f32,
  fuzzW: f32, fuzzColor: vec3f, fuzzRough: f32,
  tfW: f32, tfThick: f32, tfIor: f32,
  aS: vec2f, aC: vec2f,
  compD: f32, compM: vec3f, compC: f32,
  EspecR: f32, Ecoat: f32, Efuzz: f32,
  EspecR3: vec3f,
  tBase: vec3f,
  p: vec4f,
  nf: vec3f,
  transW: f32, etaT: f32, thin: bool, transTint: vec3f,
  pT: f32,
  compT: f32,
}

// The compensation as this pixel renders it (see the split compare in main).
var<private> gMultiScatter: bool;

fn openpbrCoatedSpecRoughness(rB: f32, rC: f32, C: f32) -> f32 {
  let rB2 = rB * rB;
  let rC2 = rC * rC;
  let t = min(1.0, rB2 * rB2 + 2.0 * rC2 * rC2);
  return mix(rB, sqrt(sqrt(t)), C);
}
fn openpbrSpecularEta(specIor: f32, coatIor: f32, C: f32, specWeight: f32) -> f32 {
  let r = specIor / coatIor;
  let tirFix = select(coatIor / specIor, r, r > 1.0);
  let etaS = mix(specIor, tirFix, C);
  let Fs = sq((etaS - 1.0) / (etaS + 1.0));
  let e = sign(etaS - 1.0) * sqrt(clamp(specWeight * Fs, 0.0, 0.99999));
  return (1.0 + e) / (1.0 - e);
}
fn openpbrCoatBaseFactor(baseColor: vec3f, specWeight: f32, M: f32, C: f32, coatColor: vec3f, coatIor: f32, coatDarkening: f32) -> vec3f {
  let F0c = sq((coatIor - 1.0) / (coatIor + 1.0));
  let K = 1.0 - (1.0 - F0c) / (coatIor * coatIor);
  let Ebase = mix(baseColor, baseColor * specWeight, M);
  let Delta = vec3f(1.0 - K) / (vec3f(1.0) - Ebase * K);
  return mix(vec3f(1.0), Delta, C * coatDarkening) * mix(vec3f(1.0), coatColor, C);
}

fn cauchyIor(nd: f32, vd: f32, nm: f32) -> f32 {
  let LF = 486.13;
  let LC = 656.27;
  let LD = 587.56;
  let B = (nd - 1.0) / (vd * (1.0 / (LF * LF) - 1.0 / (LC * LC)));
  let A = nd - B / (LD * LD);
  return A + B / (nm * nm);
}
fn disperses(m: Mat) -> bool {
  return m.transmission_weight > 0.0 && m.transmission_dispersion_scale > 0.0 && m.transmission_dispersion_abbe_number > 0.0;
}
fn cmfLobe(w: f32, mu: f32, a: f32, b: f32) -> f32 { let t = (w - mu) * select(b, a, w < mu); return exp(-0.5 * t * t); }
fn cieXYZ(w: f32) -> vec3f {
  return vec3f(
    0.362 * cmfLobe(w, 442.0, 0.0624, 0.0374) + 1.056 * cmfLobe(w, 599.8, 0.0264, 0.0323) - 0.065 * cmfLobe(w, 501.1, 0.0490, 0.0382),
    0.821 * cmfLobe(w, 568.8, 0.0213, 0.0247) + 0.286 * cmfLobe(w, 530.9, 0.0613, 0.0322),
    1.217 * cmfLobe(w, 437.0, 0.0845, 0.0278) + 0.681 * cmfLobe(w, 459.0, 0.0385, 0.0725));
}
const SPECTRAL_NORM = vec3f(117.6829, 103.4330, 98.1822);
fn spectralWeight(nm: f32) -> vec3f {
  return REC709_TO_AP1 * (XYZ_TO_REC709 * cieXYZ(nm)) * 400.0 / SPECTRAL_NORM;
}

fn setupSurf(m: Mat, wo: vec3f, back: bool, lambda: f32, colored: bool) -> Surf {
  var s: Surf;
  s.baseColor = max(m.base_color, vec3f(0.0));
  s.baseWeight = max(m.base_weight, 0.0);
  s.metal = sat(m.base_metalness);
  s.diffRough = sat(m.base_diffuse_roughness);
  s.specW = max(m.specular_weight, 0.0);
  s.specColor = max(m.specular_color, vec3f(0.0));
  s.coatW = sat(m.coat_weight);
  s.coatRough = sat(m.coat_roughness);
  s.coatIor = max(m.coat_ior, 1.0);
  s.fuzzW = sat(m.fuzz_weight);
  s.fuzzColor = max(m.fuzz_color, vec3f(0.0));
  s.fuzzRough = clamp(m.fuzz_roughness, 0.01, 1.0);
  s.tfW = select(0.0, sat(m.thin_film_weight), m.thin_film_thickness > 0.0);
  s.tfThick = m.thin_film_thickness;
  s.tfIor = m.thin_film_ior;
  s.specRough = openpbrCoatedSpecRoughness(sat(m.specular_roughness), s.coatRough, s.coatW);
  let aniso = sat(m.specular_roughness_anisotropy);
  let alphaT = s.specRough * s.specRough * sqrt(2.0 / (1.0 + sq(1.0 - aniso)));
  s.aS = max(vec2f(alphaT, (1.0 - aniso) * alphaT), vec2f(1e-4));
  let rTable = select(s.specRough, sqrt(sqrt(s.aS.x * s.aS.y)), aniso > 0.0);
  s.nf = vec3f(0.0, 0.0, 1.0);
  s.aC = vec2f(max(s.coatRough * s.coatRough, 1e-4));
  let nd = max(m.specular_ior, 1.0);
#ifndef GLASS
  s.eta = openpbrSpecularEta(nd, s.coatIor, s.coatW, s.specW);
  s.etaT = s.eta;
#ifdef THIN
  s.thin = m.geometry_thin_walled > 0.5;
  s.transW = select(0.0, sat(m.transmission_weight), s.thin);
  s.transTint = max(m.transmission_color, vec3f(0.0));
#else
  s.transW = 0.0;
  s.thin = false;
  s.transTint = vec3f(1.0);
#endif
#else
#ifdef NO_DISPERSION
  let nChan = nd;
#else
  let nChan = select(nd, cauchyIor(nd, m.transmission_dispersion_abbe_number / max(m.transmission_dispersion_scale, 1e-9), lambda), disperses(m));
#endif
  s.eta = openpbrSpecularEta(select(nd, nChan, colored), s.coatIor, s.coatW, s.specW);
  s.etaT = openpbrSpecularEta(nChan, s.coatIor, s.coatW, s.specW);
  let sssW = sat(m.subsurface_weight) * (1.0 - sat(m.transmission_weight));
  s.transW = sat(m.transmission_weight) + sssW;
  s.thin = m.geometry_thin_walled > 0.5 || abs(s.etaT - 1.0) < 1e-3;
  s.transTint = select(max(m.transmission_color, vec3f(0.0)), vec3f(1.0), m.transmission_depth > 0.0 || sssW > 0.0);
  if (back && !s.thin) {
    s.eta = 1.0 / s.eta;
    s.etaT = 1.0 / s.etaT;
  }
#endif
  let etaTable = select(1.0 / s.eta, s.eta, s.eta >= 1.0);

  let mu = clamp(wo.z, 1e-4, 1.0);
  let ms = gMultiScatter;
  let F0m = s.baseWeight * s.baseColor;

  let tS = textureSampleLevel(eTable, eSampler, eTableCoord(mu, rTable, etaTable), 0.0).rgb;
  let Ess = max(tS.r, 1e-4);
  s.compD = select(1.0, 1.0 + tS.b * (1.0 - Ess) / Ess, ms);
  s.compM = select(vec3f(1.0), 1.0 + favgF82Tint(F0m, s.specColor) * (1.0 - Ess) / Ess, ms);
  s.EspecR = sat(tS.g * s.compD);
  s.EspecR3 = vec3f(s.EspecR);
  if (s.tfW > 0.0) {
    let film = thinFilmF(mu, s.tfIor, s.tfThick, vec3f(f0FromEta(s.eta)));
    let Fp = fresnelDielectric(mu, s.eta);
    let withFilm = select(Ess * s.compD * film, s.EspecR * film / Fp, Fp > 1e-3);
    s.EspecR3 = sat3(mix(vec3f(s.EspecR), withFilm, s.tfW));
  }

  s.compT = 1.0;
#ifdef THIN
  if (ms && s.transW > 0.0) {
    if (s.thin) {
      s.compT = 1.0 / Ess;
    }
#ifdef GLASS
    else {
      let tT = textureSampleLevel(eTableT, eSampler, eTableCoord(mu, rTable, select(1.0 / s.etaT, s.etaT, s.etaT >= 1.0)), 0.0).rg;
      s.compT = 1.0 / max(select(tT.g, tT.r, s.etaT >= 1.0), 1e-3);
    }
#endif
  }
#endif

  let tC = textureSampleLevel(eTable, eSampler, eTableCoord(mu, s.coatRough, s.coatIor), 0.0).rgb;
  let EssC = max(tC.r, 1e-4);
  s.compC = select(1.0, 1.0 + tC.b * (1.0 - EssC) / EssC, ms);
  s.Ecoat = sat(tC.g * s.compC);

  s.Efuzz = select(0.0, fuzzDirAlbedo(mu, s.fuzzRough), s.fuzzW > 0.0);
  let tFuzz = 1.0 - s.fuzzW * s.Efuzz;
  let coatBase = openpbrCoatBaseFactor(s.baseColor, s.specW, s.metal, s.coatW, m.coat_color, s.coatIor, m.coat_darkening);
  s.tBase = tFuzz * (1.0 - s.coatW * s.Ecoat) * coatBase;

  let tb = lum(s.tBase);
  let diffAlb = s.baseWeight * E_EON(sat3(s.baseColor), s.diffRough, mu);
  s.p = vec4f(
    tb * (1.0 - s.metal) * (1.0 - s.transW) * max(lum((1.0 - s.EspecR3) * diffAlb), 0.0),
    tb * ((1.0 - s.metal) * lum(s.EspecR3) * max(lum(s.specColor), 0.05) + s.metal * s.specW * max(lum(F0m), 0.05)),
    tFuzz * s.coatW * s.Ecoat,
    s.fuzzW * s.Efuzz * max(lum(s.fuzzColor), 0.05));
  s.pT = tb * (1.0 - s.metal) * s.transW * max(lum((1.0 - s.EspecR3) * s.transTint), 0.05);
#ifdef THIN
  if (s.transW > 0.0) {
    let Fmu = fresnelDielectric(mu, s.eta);
    let dielW = tb * (1.0 - s.metal) * s.transW;
    s.p.y = mix(s.p.y, dielW * max(Fmu, 0.05), s.transW);
    s.pT = dielW * max((1.0 - Fmu) * lum(s.transTint), 0.05);
  }
#endif
  return s;
}

struct LobeProbs { p: vec4f, pt: f32 }
fn lobeProbs(s: Surf, filt: i32) -> LobeProbs {
  var p = s.p;
  var t = s.pT;
  if (filt == 1) { p = vec4f(p.x, 0.0, 0.0, 0.0); t = 0.0; }
  if (filt == 2) { p.x = 0.0; }
  let sum = p.x + p.y + p.z + p.w + t;
  if (sum <= 0.0) { return LobeProbs(vec4f(0.0), 0.0); }
  return LobeProbs(p / sum, t / sum);
}

#ifdef THIN
fn vndfPdf(wo: vec3f, m: vec3f, alpha: vec2f) -> f32 { return G1_GGX(wo, alpha) * max(dot(wo, m), 0.0) * D_GGX(m, alpha) / wo.z; }
fn glassReflectProb(s: Surf, cosM: f32) -> f32 {
  var F = fresnelDielectric(cosM, s.etaT);
  if (s.tfW > 0.0) { F = mix(F, sat(lum(s.EspecR3)), s.tfW); }
  return clamp(F / max(F + s.transW * (1.0 - F), 1e-6), 0.02, 0.98);
}
#endif

// The mixture pdf, and f * cos(theta_i) split into diffuse (fD) and everything else (fS).
struct SurfEval { pdf: f32, fD: vec3f, fS: vec3f }
fn evalSurf(s: Surf, wo: vec3f, wi: vec3f, filt: i32) -> SurfEval {
  var r: SurfEval;
  r.pdf = 0.0;
  r.fD = vec3f(0.0);
  r.fS = vec3f(0.0);
  if (wo.z <= 0.0) { return r; }
  let lp = lobeProbs(s, filt);
  let p = lp.p;
  let pt = lp.pt;

#ifndef THIN
  if (wi.z <= 0.0) { return r; }
#else
  if (wi.z <= 0.0) {
    if (pt <= 0.0) { return r; }
    var ft = 0.0;
    var cosM = 0.0;
    var pdfT = 0.0;
    if (s.thin) {
      let wr = vec3f(wi.xy, -wi.z);
      let hr = normalize(wo + wr);
      cosM = sat(dot(wo, hr));
      ft = G2_GGX(wo, wr, s.aS) * D_GGX(hr, s.aS) / (4.0 * wo.z);
      pdfT = select(0.0, vndfPdf(wo, hr, s.aS) / (4.0 * cosM), cosM > 0.0);
    } else {
#ifndef GLASS
      return r;
#else
      let eta = s.etaT;
      var wm = normalize(wo + wi * eta);
      if (wm.z < 0.0) { wm = -wm; }
      let dI = dot(wi, wm);
      let dO = dot(wo, wm);
      if (dI >= 0.0 || dO <= 0.0) { return r; }
      let denom = dI + dO / eta;
      let D = D_GGX(wm, s.aS);
      ft = D * G2_GGX(wo, wi, s.aS) * abs(dI * dO / (wo.z * denom * denom)) / (eta * eta);
      cosM = dO;
      pdfT = G1_GGX(wo, s.aS) * dO * D / wo.z * abs(dI) / (denom * denom);
#endif
    }
    var T3 = vec3f(1.0 - fresnelDielectric(cosM, s.etaT));
    if (s.tfW > 0.0) { T3 = mix(T3, vec3f(1.0) - thinFilmF(cosM, s.tfIor, s.tfThick, vec3f(f0FromEta(s.etaT))), s.tfW); }
    r.fS = s.tBase * (1.0 - s.metal) * s.transW * s.transTint * T3 * ft * s.compT;
    r.pdf = (p.y + pt) * pdfT * (1.0 - glassReflectProb(s, cosM));
    return r;
  }
#endif

  let h = normalize(wo + wi);
  let voh = sat(dot(wo, h));
  var pdf = 0.0;

  if (filt != 2 && s.metal < 1.0) {
    let d = s.baseWeight * f_EON(sat3(s.baseColor), s.diffRough, wi, wo) * wi.z;
    r.fD = s.tBase * (1.0 - s.metal) * (1.0 - s.transW) * (1.0 - s.EspecR3) * d;
    pdf += p.x * pdf_EON(wo, wi, s.diffRough);
  }
  if (filt != 1) {
    var woS = wo;
    var wiS = wi;
    var hS = h;
    let flaked = s.nf.z < 0.99999;
    if (flaked) {
      let B = onb(s.nf);
      woS = vec3f(dot(wo, B.b1), dot(wo, B.b2), dot(wo, s.nf));
      wiS = vec3f(dot(wi, B.b1), dot(wi, B.b2), dot(wi, s.nf));
      hS = normalize(woS + wiS);
    }
    if (woS.z > 0.0 && wiS.z > 0.0) {
      var g = G2_GGX(woS, wiS, s.aS) * D_GGX(hS, s.aS) / (4.0 * woS.z);
      if (flaked) { g *= wi.z / wiS.z; }
      let compR = mix(s.compD, s.compT, s.transW);
      var Fd = s.specColor * fresnelDielectric(voh, s.eta) * compR;
      var Fm = s.specW * fresnelF82Tint(voh, s.baseWeight * s.baseColor, s.specColor) * s.compM;
      if (s.tfW > 0.0) {
        var FtfD = vec3f(0.0);
        var FtfM = vec3f(0.0);
        let subs = select(1, 2, s.metal > 0.0 && s.metal < 1.0);
        for (var k = 0; k < subs; k++) {
          let overMetal = select(s.metal >= 1.0, k == 1, subs == 2);
          let f = thinFilmF(voh, s.tfIor, s.tfThick, select(vec3f(f0FromEta(s.eta)), s.baseWeight * s.baseColor, overMetal));
          if (overMetal) { FtfM = f; } else { FtfD = f; }
        }
        Fd = mix(Fd, s.specColor * FtfD * compR, s.tfW);
        Fm = mix(Fm, s.specW * FtfM * s.compM, s.tfW);
      }
      r.fS += s.tBase * mix(Fd, Fm, s.metal) * g;
#ifdef THIN
      if (s.transW > 0.0) {
        pdf += (p.y + pt) * vndfPdf(wo, h, s.aS) / (4.0 * max(voh, 1e-6)) * glassReflectProb(s, voh);
      } else {
        pdf += p.y * pdfGGXReflection_Bounded(woS, wiS, s.aS);
      }
#else
      pdf += p.y * pdfGGXReflection_Bounded(woS, wiS, s.aS);
#endif
    }
    if (s.coatW > 0.0) {
      let gc = G2_GGX(wo, wi, s.aC) * D_GGX(h, s.aC) / (4.0 * wo.z);
      r.fS += vec3f((1.0 - s.fuzzW * s.Efuzz) * s.coatW * fresnelDielectric(voh, s.coatIor) * s.compC * gc);
      pdf += p.z * pdfGGXReflection_Bounded(wo, wi, s.aC);
    }
    if (s.fuzzW > 0.0) {
      let D = fuzzD(wo, wi, s.fuzzRough);
      r.fS += s.fuzzW * s.fuzzColor * s.Efuzz * D;
      pdf += p.w * D;
    }
  }
  r.pdf = pdf;
  return r;
}

struct SurfSample { ok: bool, wi: vec3f }
fn sampleSurf(s: Surf, wo: vec3f, filt: i32, u: vec3f) -> SurfSample {
  let lp = lobeProbs(s, filt);
  let p = lp.p;
  let pt = lp.pt;
  let cCoat = p.x + p.y + p.z;
  let cFuzz = cCoat + p.w;
  if (u.z < p.x) {
    let wi = sample_EON(wo, s.diffRough, u.x, u.y);
    return SurfSample(wi.z > 0.0, wi);
  }
#ifdef THIN
  let specPick = u.z < p.x + p.y;
  let transPick = u.z >= cFuzz && pt > 0.0;
  if (s.transW > 0.0 && (specPick || transPick)) {
    let t = select(p.y + u.z - cFuzz, u.z - p.x, specPick) / max(p.y + pt, 1e-9);
    let m = sampleVNDF_SphericalCap(u.xy, wo, s.aS);
    let cosM = dot(wo, m);
    if (cosM <= 0.0) { return SurfSample(false, vec3f(0.0)); }
    let wr = reflect(-wo, m);
    if (t < glassReflectProb(s, cosM)) { return SurfSample(wr.z > 0.0, wr); }
    if (s.thin) {
      let wt = vec3f(wr.xy, -wr.z);
      return SurfSample(wt.z < 0.0, wt);
    }
#ifdef GLASS
    let wf = refract(-wo, m, 1.0 / s.etaT);
    return SurfSample(dot(wf, wf) > 0.0 && wf.z < 0.0, wf);
#else
    return SurfSample(false, vec3f(0.0));
#endif
  }
#endif
  if (u.z < cCoat) {
    let coat = u.z >= p.x + p.y;
    var b1 = vec3f(1.0, 0.0, 0.0);
    var b2 = vec3f(0.0, 1.0, 0.0);
    var b3 = vec3f(0.0, 0.0, 1.0);
    if (!coat && s.nf.z < 0.99999) {
      let B = onb(s.nf);
      b1 = B.b1;
      b2 = B.b2;
      b3 = s.nf;
    }
    let woS = vec3f(dot(wo, b1), dot(wo, b2), dot(wo, b3));
    if (woS.z <= 0.0) { return SurfSample(false, vec3f(0.0)); }
    let wiS = sampleGGXReflection_Bounded(u.xy, woS, select(s.aS, s.aC, coat));
    let wi = wiS.x * b1 + wiS.y * b2 + wiS.z * b3;
    return SurfSample(wi.z > 0.0, wi);
  }
  if (u.z < cFuzz && p.w > 0.0) {
    let wi = sampleFuzz(wo, s.fuzzRough, u.xy);
    return SurfSample(wi.z > 0.0, wi);
  }
  return SurfSample(false, vec3f(0.0));
}

fn albedoAOV(m: Mat) -> vec3f {
  var a = mix(m.base_color * m.base_weight, m.base_color, m.base_metalness);
  a = mix(a, m.subsurface_color, m.subsurface_weight * (1.0 - m.base_metalness));
  a = mix(a, m.transmission_color, m.transmission_weight * (1.0 - m.base_metalness));
  a *= mix(vec3f(1.0), m.coat_color, m.coat_weight);
  return mix(a, m.fuzz_color, m.fuzz_weight);
}
`
