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
  specular_roughness_anisotropy: number
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
  // Lab extension, not part of OpenPBR 1.1: metallic flakes in the base layer, under the coat. A flake is a cell
  // of a 3D grid (lab_flake_size world units) present with probability lab_flake_coverage, whose normal is tilted
  // by up to lab_flake_tilt (the tangent of the cone angle); it tilts the base specular lobe only.
  lab_flake_coverage: number
  lab_flake_size: number
  lab_flake_tilt: number
}

export const MATERIAL_FIELDS = [
  'base_weight',
  'base_color',
  'base_metalness',
  'base_diffuse_roughness',
  'specular_weight',
  'specular_color',
  'specular_roughness',
  'specular_roughness_anisotropy',
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
  'lab_flake_coverage',
  'lab_flake_size',
  'lab_flake_tilt',
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
  specular_roughness_anisotropy: 0,
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
  lab_flake_coverage: 0,
  lab_flake_size: 0.012,
  lab_flake_tilt: 0.3,
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

export type Hero = 'carpaint' | 'gold' | 'velvet' | 'thinfilm' | 'plastic' | 'brushed' | 'titanium'
type RoughnessParam = 'specular_roughness' | 'fuzz_roughness' | 'coat_roughness'

export const HERO_ORDER: Hero[] = ['carpaint', 'gold', 'brushed', 'titanium', 'velvet', 'thinfilm', 'plastic']

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
  // Official: examples/open_pbr_aluminum_brushed.mtlx. The lab brushes around the vertical axis, as on a lathe,
  // so highlights stretch from pole to pole.
  brushed: {
    label: 'Brushed aluminum',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.918, 0.922, 0.923],
      base_metalness: 1,
      specular_color: [0.921, 0.934, 0.955],
      specular_roughness: 0.2,
      specular_roughness_anisotropy: 0.9,
    }),
  },
  // Authored on official examples/open_pbr_titanium.mtlx: brushed, with a thin oxide film (titanium dioxide,
  // IOR about 2.4). The film's thickness sets the tint, as on real heat-tinted titanium: in this renderer 0.04 um
  // reads straw, 0.06 blue-violet, 0.13 gold, 0.16 magenta. 0.06 is the blue of heat-tinted exhausts.
  titanium: {
    label: 'Heat-tinted titanium',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.424, 0.403, 0.367],
      base_metalness: 1,
      specular_color: [0.882, 0.903, 0.94],
      specular_roughness: 0.18,
      specular_roughness_anisotropy: 0.8,
      thin_film_weight: 1,
      thin_film_thickness: 0.06,
      thin_film_ior: 2.4,
    }),
  },
}

// Car paint finishes. Solid is the official example; the others are authored for the lab on the same layering
// (base, then the thin film that OpenPBR places between base and coat, then the clear coat).
export type PaintFinish = 'solid' | 'metallic' | 'pearl' | 'iridescent'
export const PAINT_ORDER: PaintFinish[] = ['solid', 'metallic', 'pearl', 'iridescent']
export const PAINT_FINISHES: Record<PaintFinish, { label: string; params: OpenPBR }> = {
  solid: { label: 'Solid', params: HERO_PRESETS.carpaint.params },
  // Metallic: pigment and metal together under the coat (metalness 0.6), so it keeps its color off the highlight
  // and brightens toward it, the flop of real metallic paint.
  metallic: {
    label: 'Metallic',
    params: mat({
      base_color: [0.06, 0.36, 0.82],
      base_metalness: 0.6,
      specular_roughness: 0.35,
      coat_weight: 1,
      coat_roughness: 0.02,
      coat_ior: 1.6,
    }),
  },
  // Pearl: a white base with an interference film, the soft pink-to-green shift of mica pearl paint.
  pearl: {
    label: 'Pearl',
    params: mat({
      base_color: [0.82, 0.82, 0.8],
      base_metalness: 0.45,
      specular_roughness: 0.3,
      thin_film_weight: 1,
      thin_film_thickness: 0.36,
      thin_film_ior: 1.6,
      coat_weight: 1,
      coat_roughness: 0.02,
      coat_ior: 1.6,
    }),
  },
  // Iridescent (color shift): a strong film over a mid-gray metal, the angle-dependent hues of chameleon paint
  // (over a near-black metal the hues only show in the small highlights of a dark studio).
  iridescent: {
    label: 'Iridescent',
    params: mat({
      base_color: [0.3, 0.3, 0.32],
      base_metalness: 1,
      specular_roughness: 0.32,
      thin_film_weight: 1,
      thin_film_thickness: 0.48,
      thin_film_ior: 1.8,
      coat_weight: 1,
      coat_roughness: 0.02,
      coat_ior: 1.6,
    }),
  },
}

// Flakes are a carrier for metallic, pearl and iridescent pigments; solid paint has none. With flakes, the flakes
// do most of the spreading, so the base lobe itself is smoother. Tilting a normal under a smooth macro surface
// loses energy (part of the tilted lobe points below the horizon); the furnace shows how much, on the hero
// alone: 5.1% at a tilt of 0.32, 2.7% at 0.22, 1.4% at 0.15. A tilt of 0.18 (about 10 degrees, within the
// 10-15 degree spread of real metallic flakes) keeps it near 2%.
// Size: about a pixel at the lab's framing, so flakes sparkle as fine grain rather than visible chips.
const FLAKES: Partial<OpenPBR> = { lab_flake_coverage: 0.55, lab_flake_size: 0.006, lab_flake_tilt: 0.18 }
export const paintHasFlakes = (finish: PaintFinish, flakes: boolean) => flakes && finish !== 'solid'

// The hero's parameters before any slider edits: car paint takes its finish (and flakes); the others their preset.
export function heroParams(hero: Hero, finish: PaintFinish, flakes: boolean): OpenPBR {
  if (hero !== 'carpaint') return HERO_PRESETS[hero].params
  const p = PAINT_FINISHES[finish].params
  if (!paintHasFlakes(finish, flakes)) return p
  return { ...p, ...FLAKES, specular_roughness: Math.min(p.specular_roughness, 0.18) }
}
