// OpenPBR Surface v1.1.1 parameters (Academy Software Foundation). Colors are ACEScg, the renderer's working
// space. Presets marked "official" are copied verbatim from the OpenPBR repository's examples/*.mtlx files
// (all authored in ACEScg); the others are authored for this lab.
export type RGB = [number, number, number]

export interface OpenPBR {
  base_weight: number
  base_color: RGB
  base_metalness: number
  base_diffuse_roughness: number
  specular_weight: number
  specular_color: RGB
  specular_roughness: number
  specular_ior: number
  coat_weight: number
  coat_color: RGB
  coat_roughness: number
  coat_ior: number
  coat_darkening: number
  fuzz_weight: number
  fuzz_color: RGB
  fuzz_roughness: number
  thin_film_weight: number
  thin_film_thickness: number // micrometers
  thin_film_ior: number
}

export const MATERIAL_FIELDS = [
  'base_weight',
  'base_color',
  'base_metalness',
  'base_diffuse_roughness',
  'specular_weight',
  'specular_color',
  'specular_roughness',
  'specular_ior',
  'coat_weight',
  'coat_color',
  'coat_roughness',
  'coat_ior',
  'coat_darkening',
  'fuzz_weight',
  'fuzz_color',
  'fuzz_roughness',
  'thin_film_weight',
  'thin_film_thickness',
  'thin_film_ior',
] as const satisfies readonly (keyof OpenPBR)[]

// Reference nodedef defaults (open_pbr_surface.mtlx, version 1.1.1).
export const OPENPBR_DEFAULTS: OpenPBR = {
  base_weight: 1,
  base_color: [0.8, 0.8, 0.8],
  base_metalness: 0,
  base_diffuse_roughness: 0,
  specular_weight: 1,
  specular_color: [1, 1, 1],
  specular_roughness: 0.3,
  specular_ior: 1.5,
  coat_weight: 0,
  coat_color: [1, 1, 1],
  coat_roughness: 0,
  coat_ior: 1.6,
  coat_darkening: 1,
  fuzz_weight: 0,
  fuzz_color: [1, 1, 1],
  fuzz_roughness: 0.5,
  thin_film_weight: 0,
  thin_film_thickness: 0.5,
  thin_film_ior: 1.4,
}

const mat = (p: Partial<OpenPBR>): OpenPBR => ({ ...OPENPBR_DEFAULTS, ...p })

export const SCENE_MATERIALS = {
  // Official: examples/open_pbr_gray_card.mtlx
  gray: mat({ base_color: [0.18, 0.18, 0.18], specular_roughness: 0.9 }),
  // Official: examples/open_pbr_chromium.mtlx
  chrome: mat({
    base_color: [0.666, 0.682, 0.698],
    base_metalness: 1,
    specular_color: [0.706, 0.726, 0.788],
    specular_roughness: 0.02,
  }),
  // Authored: a dark studio floor with a soft sheen so the balls get a grounded reflection.
  floor: mat({ base_color: [0.05, 0.05, 0.05], base_diffuse_roughness: 0.5, specular_weight: 0.7, specular_roughness: 0.42 }),
}

export type Hero = 'carpaint' | 'gold' | 'velvet' | 'thinfilm' | 'plastic'
type RoughnessParam = 'specular_roughness' | 'fuzz_roughness' | 'coat_roughness'

export const HERO_ORDER: Hero[] = ['carpaint', 'gold', 'velvet', 'thinfilm', 'plastic']

export interface HeroPreset {
  label: string
  official: boolean
  params: OpenPBR
  roughnessParam: RoughnessParam
}

export const HERO_PRESETS: Record<Hero, HeroPreset> = {
  // Official: examples/open_pbr_carpaint.mtlx
  carpaint: {
    label: 'Car paint',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.1, 0.6, 0.9],
      specular_ior: 1.6,
      specular_roughness: 0.3,
      coat_weight: 1,
      coat_roughness: 0.02,
      coat_ior: 1.6,
    }),
  },
  // Official: examples/open_pbr_gold.mtlx
  gold: {
    label: 'Gold',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.929, 0.788, 0.374],
      base_metalness: 1,
      specular_color: [0.987, 1.013, 0.997],
      specular_roughness: 0.02,
    }),
  },
  // Official: examples/open_pbr_velvet.mtlx
  velvet: {
    label: 'Velvet',
    official: true,
    roughnessParam: 'fuzz_roughness',
    params: mat({
      base_color: [0.062, 0.01, 0.269],
      base_diffuse_roughness: 1,
      specular_roughness: 1,
      fuzz_weight: 0.5,
      fuzz_color: [0.315, 0.237, 0.465],
      fuzz_roughness: 0.5,
    }),
  },
  // Authored: a thin film over a near-black dielectric. Film values match the official soap bubble example
  // (thin_film_thickness 0.5 um, thin_film_ior 1.4); there is no official dark-base film example.
  thinfilm: {
    label: 'Thin film',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.01, 0.01, 0.01],
      specular_roughness: 0.05,
      thin_film_weight: 1,
      thin_film_thickness: 0.5,
      thin_film_ior: 1.4,
    }),
  },
  // Authored: the site's teal accent (#5DCAA5) as a glossy plastic, converted to ACEScg.
  plastic: {
    label: 'Plastic',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({ base_color: [0.2855, 0.554, 0.3942], specular_roughness: 0.25 }),
  },
}
