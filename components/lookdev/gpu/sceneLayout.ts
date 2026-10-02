// The tracer's uniforms on WebGPU: the scene (set when it changes) and each dispatch's own (sample counts, the
// slice of rows). Both are lists of vec4s (see layout.ts). Materials pack into MAT_VEC4S vec4s each (matWords).
import { UniformLayout } from './layout'
import type { OpenPBR } from '../materials'

export const MAX_MATS = 10
export const MAT_VEC4S = 15

export const SCENE = new UniformLayout('Scene', [
  { name: 'camPos', kind: 'vec4f' },
  { name: 'camFwd', kind: 'vec4f' },
  { name: 'camRight', kind: 'vec4f' },
  { name: 'camUp', kind: 'vec4f' },
  { name: 'tanHalf', kind: 'vec4f' }, // xy
  { name: 'ambient', kind: 'vec4f' }, // the rig's faint environment (rgb); w: the backdrop's mip level
  { name: 'lightCorner', kind: 'vec4f', count: 3 },
  { name: 'lightU', kind: 'vec4f', count: 3 },
  { name: 'lightV', kind: 'vec4f', count: 3 },
  { name: 'lightRad', kind: 'vec4f', count: 3 }, // rgb radiance, w: ultraviolet
  { name: 'ballX', kind: 'vec4f' }, // the reference balls' x (no model)
  { name: 'balls', kind: 'vec4f', count: 3 }, // with a model: center, radius (0 = absent)
  { name: 'chartO', kind: 'vec4f' },
  { name: 'chartU', kind: 'vec4f' },
  { name: 'chartV', kind: 'vec4f' },
  { name: 'fparams', kind: 'vec4f' }, // cyc z, cyc radius, indirect clamp, environment turn
  { name: 'fparams2', kind: 'vec4f' }, // split x (fraction)
  { name: 'iparams0', kind: 'vec4i' }, // furnace, pass, multiple scattering, multiple-scattering compare
  { name: 'iparams1', kind: 'vec4i' }, // cyc, chart, thin glass, softboxes
  { name: 'iparams2', kind: 'vec4i' }, // environment on, max scatter events, thin glass slot, max BVH node visits
  { name: 'iparams3', kind: 'vec4i' }, // environment width, height
  { name: 'modelRot', kind: 'vec4f', count: 3 }, // columns
  { name: 'modelPos', kind: 'vec4f' }, // xyz, w: scale
  { name: 'sobol', kind: 'vec4u', count: 32 },
  { name: 'chartColor', kind: 'vec4f', count: 25 },
  { name: 'mats', kind: 'vec4f', count: MAX_MATS * MAT_VEC4S },
])

export const PASS = new UniformLayout('Pass', [
  { name: 'dims', kind: 'vec4u' }, // width, height, samples done, new samples
  { name: 'slice', kind: 'vec4u' }, // slice, slice count, rows per block
  { name: 'iparams', kind: 'vec4i' }, // max bounces
  { name: 'res', kind: 'vec4f' }, // resolution the camera rays are spread over (fractional for a preview)
])
// Dispatches of one frame take their pass uniforms at dynamic offsets of one buffer.
export const PASS_STRIDE = 256

// One material, as getMat in the tracer reads it back.
export function matWords(m: OpenPBR): Float32Array {
  const w = new Float32Array(MAT_VEC4S * 4)
  const put = (v: number, x: number, y = 0, z = 0, a = 0) => w.set([x, y, z, a], v * 4)
  put(0, ...m.base_color, m.base_weight)
  put(1, ...m.specular_color, m.specular_weight)
  put(2, ...m.coat_color, m.coat_weight)
  put(3, ...m.fuzz_color, m.fuzz_weight)
  put(4, ...m.transmission_color, m.transmission_weight)
  put(5, ...m.subsurface_color, m.subsurface_weight)
  put(6, ...m.subsurface_radius_scale, m.subsurface_radius)
  put(7, ...m.lab_fluor_color, m.lab_fluor_weight)
  put(8, ...m.lab_fluor_absorb, m.lab_fluor_uv)
  put(9, m.base_metalness, m.base_diffuse_roughness, m.specular_roughness, m.specular_roughness_anisotropy)
  put(10, m.specular_ior, m.coat_roughness, m.coat_ior, m.coat_darkening)
  put(11, m.fuzz_roughness, m.thin_film_weight, m.thin_film_thickness, m.thin_film_ior)
  put(12, m.transmission_depth, m.transmission_dispersion_scale, m.transmission_dispersion_abbe_number, m.subsurface_scatter_anisotropy)
  put(13, m.geometry_thin_walled, m.lab_flake_coverage, m.lab_flake_size, m.lab_flake_tilt)
  put(14, m.lab_uv_ratio)
  return w
}

// getMat's unpacking, mirroring matWords.
export const MAT_UNPACK = /* wgsl */ `
const MAT_VEC4S: u32 = ${MAT_VEC4S}u;
fn loadMat(i: u32) -> Mat {
  let b = i * MAT_VEC4S;
  var m: Mat;
  let v0 = scene.mats[b]; let v1 = scene.mats[b + 1u]; let v2 = scene.mats[b + 2u]; let v3 = scene.mats[b + 3u];
  let v4 = scene.mats[b + 4u]; let v5 = scene.mats[b + 5u]; let v6 = scene.mats[b + 6u]; let v7 = scene.mats[b + 7u];
  let v8 = scene.mats[b + 8u]; let v9 = scene.mats[b + 9u]; let v10 = scene.mats[b + 10u]; let v11 = scene.mats[b + 11u];
  let v12 = scene.mats[b + 12u]; let v13 = scene.mats[b + 13u]; let v14 = scene.mats[b + 14u];
  m.base_color = v0.xyz; m.base_weight = v0.w;
  m.specular_color = v1.xyz; m.specular_weight = v1.w;
  m.coat_color = v2.xyz; m.coat_weight = v2.w;
  m.fuzz_color = v3.xyz; m.fuzz_weight = v3.w;
  m.transmission_color = v4.xyz; m.transmission_weight = v4.w;
  m.subsurface_color = v5.xyz; m.subsurface_weight = v5.w;
  m.subsurface_radius_scale = v6.xyz; m.subsurface_radius = v6.w;
  m.lab_fluor_color = v7.xyz; m.lab_fluor_weight = v7.w;
  m.lab_fluor_absorb = v8.xyz; m.lab_fluor_uv = v8.w;
  m.base_metalness = v9.x; m.base_diffuse_roughness = v9.y; m.specular_roughness = v9.z; m.specular_roughness_anisotropy = v9.w;
  m.specular_ior = v10.x; m.coat_roughness = v10.y; m.coat_ior = v10.z; m.coat_darkening = v10.w;
  m.fuzz_roughness = v11.x; m.thin_film_weight = v11.y; m.thin_film_thickness = v11.z; m.thin_film_ior = v11.w;
  m.transmission_depth = v12.x; m.transmission_dispersion_scale = v12.y; m.transmission_dispersion_abbe_number = v12.z; m.subsurface_scatter_anisotropy = v12.w;
  m.geometry_thin_walled = v13.x; m.lab_flake_coverage = v13.y; m.lab_flake_size = v13.z; m.lab_flake_tilt = v13.w;
  m.lab_uv_ratio = v14.x;
  return m;
}
`
