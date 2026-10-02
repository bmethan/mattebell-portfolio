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
  transmission_weight: number
  transmission_color: RGB
  transmission_depth: number // centimeters in the presets, as subsurface_radius; 0 = no medium, the color tints the refraction
  transmission_dispersion_scale: number
  transmission_dispersion_abbe_number: number
  subsurface_weight: number
  subsurface_color: RGB // the color the medium shows when deep (its multiple-scattering albedo)
  subsurface_radius: number // mean free path length scale: centimeters in the presets (see subsurfaceScale)
  subsurface_radius_scale: RGB // per-channel multiplier of the radius
  subsurface_scatter_anisotropy: number // Henyey-Greenstein g
  geometry_thin_walled: number // 0 or 1 (a boolean in OpenPBR)
  // Lab extension, not part of OpenPBR 1.1: metallic flakes in the base layer, under the coat. A flake is a cell
  // of a 3D grid (lab_flake_size world units) present with probability lab_flake_coverage, whose normal is tilted
  // by up to lab_flake_tilt (the tangent of the cone angle); it tilts the base specular lobe only.
  lab_flake_coverage: number
  lab_flake_size: number
  lab_flake_tilt: number
  // Lab extension, not part of OpenPBR 1.1: fluorescence in the base. The base absorbs lab_fluor_absorb of the
  // red, green and blue light reaching it and lab_fluor_uv of the ultraviolet, and re-emits lab_fluor_weight of that
  // energy (quantum yield times the Stokes loss) diffusely, spread over red, green and blue as lab_fluor_color
  // (summing to 1). With base_color + lab_fluor_absorb at most 1 per channel, no energy is created. lab_uv_ratio:
  // the base's diffuse reflectance of ultraviolet, relative to its mean visible reflectance.
  // In a transmissive solid the fluorophore fills the medium instead: lab_fluor_absorb and lab_fluor_uv are then
  // its absorption per centimeter (transmission_color and depth hold the medium's total visible absorption, the
  // fluorophore's included).
  lab_fluor_weight: number
  lab_fluor_color: RGB
  lab_fluor_absorb: RGB
  lab_fluor_uv: number
  lab_uv_ratio: number
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
  'transmission_weight',
  'transmission_color',
  'transmission_depth',
  'transmission_dispersion_scale',
  'transmission_dispersion_abbe_number',
  'subsurface_weight',
  'subsurface_color',
  'subsurface_radius',
  'subsurface_radius_scale',
  'subsurface_scatter_anisotropy',
  'geometry_thin_walled',
  'lab_flake_coverage',
  'lab_flake_size',
  'lab_flake_tilt',
  'lab_fluor_weight',
  'lab_fluor_color',
  'lab_fluor_absorb',
  'lab_fluor_uv',
  'lab_uv_ratio',
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
  transmission_weight: 0,
  transmission_color: [1, 1, 1],
  transmission_depth: 0,
  transmission_dispersion_scale: 0,
  transmission_dispersion_abbe_number: 20,
  subsurface_weight: 0,
  subsurface_color: [0.8, 0.8, 0.8],
  subsurface_radius: 1,
  subsurface_radius_scale: [1, 0.5, 0.25],
  subsurface_scatter_anisotropy: 0,
  geometry_thin_walled: 0,
  lab_flake_coverage: 0,
  lab_flake_size: 0.012,
  lab_flake_tilt: 0.3,
  lab_fluor_weight: 0,
  lab_fluor_color: [1 / 3, 1 / 3, 1 / 3],
  lab_fluor_absorb: [0, 0, 0],
  lab_fluor_uv: 0,
  lab_uv_ratio: 1,
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
  // Authored: a neutral gray cyc, painted like a studio sweep (matte, a faint sheen).
  grayCyc: mat({ base_color: [0.18, 0.18, 0.18], base_diffuse_roughness: 0.6, specular_weight: 0.5, specular_roughness: 0.55 }),
}

// The stage: the black void the still was rendered in, or a cyc (floor sweeping up into a back wall) in the
// floor's own charcoal or in neutral gray.
export type Stage = 'void' | 'charcoal' | 'gray'
export const STAGE_ORDER: Stage[] = ['void', 'charcoal', 'gray']
export const STAGES: Record<Stage, { label: string; cyc: boolean; material: OpenPBR }> = {
  void: { label: 'Void', cyc: false, material: SCENE_MATERIALS.floor },
  charcoal: { label: 'Charcoal cyc', cyc: true, material: SCENE_MATERIALS.floor },
  gray: { label: 'Gray cyc', cyc: true, material: SCENE_MATERIALS.grayCyc },
}

// The ColorChecker Classic's 24 patches, row by row from dark skin to black, as ACEScg reflectances: X-Rite's
// published CIE L*a*b* values for charts made after November 2014 (D50), to XYZ, Bradford-adapted to the ACES
// white point, to ACEScg. They shade as the gray card does (its matte surface in each patch's color). Last, the
// chart's black frame (authored).
export const COLOR_CHECKER: RGB[] = [
  [0.1358, 0.0851, 0.0582], [0.4474, 0.2963, 0.2254], [0.1436, 0.1848, 0.3089], [0.1184, 0.1462, 0.0629],
  [0.2318, 0.2163, 0.3989], [0.2625, 0.4786, 0.4156], [0.5274, 0.2379, 0.0635], [0.0887, 0.1021, 0.3495],
  [0.3761, 0.1142, 0.12], [0.0877, 0.048, 0.127], [0.3749, 0.4797, 0.0984], [0.5953, 0.3823, 0.0729],
  [0.0424, 0.0489, 0.2528], [0.1303, 0.2717, 0.0864], [0.2879, 0.0652, 0.0484], [0.7113, 0.5854, 0.0843],
  [0.3604, 0.1118, 0.2709], [0.0703, 0.2161, 0.3528], [0.8793, 0.8839, 0.8407], [0.5871, 0.5915, 0.5853],
  [0.3613, 0.3664, 0.3653], [0.1904, 0.1908, 0.1899], [0.0871, 0.0885, 0.0896], [0.0315, 0.0315, 0.0322],
  [0.02, 0.02, 0.02],
]

export type Hero =
  | 'carpaint' | 'gold' | 'velvet' | 'thinfilm' | 'plastic' | 'brushed' | 'titanium' | 'glass' | 'diamond' | 'soapbubble'
  | 'skin' | 'marble' | 'ceramic' | 'honey' | 'copper' | 'silver' | 'highlighter' | 'dayglo' | 'paper' | 'uranium' | 'tonic'
type RoughnessParam = 'specular_roughness' | 'fuzz_roughness' | 'coat_roughness'

// The hero materials by family: the Hero row picks a family, a second row its material (a family of one has no
// second row). Car paint and skin have their own rows below (finish, tone).
export type HeroFamily = 'metal' | 'dielectric' | 'layered' | 'transparent' | 'translucent' | 'fluorescent' | 'fabric'
export const HERO_FAMILIES: { id: HeroFamily; label: string; heroes: Hero[] }[] = [
  { id: 'layered', label: 'Layered', heroes: ['carpaint', 'thinfilm'] },
  { id: 'metal', label: 'Metal', heroes: ['gold', 'copper', 'silver', 'brushed', 'titanium'] },
  { id: 'dielectric', label: 'Dielectric', heroes: ['plastic', 'ceramic'] },
  { id: 'transparent', label: 'Transparent', heroes: ['glass', 'diamond', 'honey', 'soapbubble'] },
  { id: 'translucent', label: 'Translucent', heroes: ['skin', 'marble'] },
  { id: 'fluorescent', label: 'Fluorescent', heroes: ['highlighter', 'dayglo', 'paper', 'uranium', 'tonic'] },
  { id: 'fabric', label: 'Fabric', heroes: ['velvet'] },
]
export const HERO_ORDER: Hero[] = HERO_FAMILIES.flatMap(f => f.heroes)
export const familyOf = (h: Hero) => HERO_FAMILIES.find(f => f.heroes.includes(h))!

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
  // Authored, lab extension (fluorescence): highlighter ink, a yellow whose dye (pyranine-like) absorbs blue and
  // near-ultraviolet light and re-emits it green-yellow: brighter than its own reflectance in daylight, and the one
  // thing still glowing under a black light.
  highlighter: {
    label: 'Highlighter',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.78, 0.84, 0.06],
      specular_roughness: 0.5,
      lab_fluor_weight: 0.7,
      lab_fluor_color: [0.3, 0.67, 0.03],
      lab_fluor_absorb: [0, 0.05, 0.85],
      lab_fluor_uv: 0.9,
      lab_uv_ratio: 0.05,
    }),
  },
  // Authored, lab extension: daylight-fluorescent orange paint (Day-Glo type): the pigment absorbs green, blue and
  // ultraviolet and re-emits orange-red, so it reads hotter than any ordinary orange.
  dayglo: {
    label: 'Fluorescent orange',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.9, 0.28, 0.04],
      specular_roughness: 0.45,
      lab_fluor_weight: 0.7,
      lab_fluor_color: [0.75, 0.25, 0],
      lab_fluor_absorb: [0, 0.6, 0.9],
      lab_fluor_uv: 0.85,
      lab_uv_ratio: 0.05,
    }),
  },
  // Authored, lab extension: white paper with optical brighteners, which absorb near-ultraviolet (around 350 nm)
  // and re-emit blue (around 430 nm). Studio light has no ultraviolet, so it looks plain off-white; under a black
  // light it glows blue, as white shirts do.
  paper: {
    label: 'Brightened paper',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.8, 0.79, 0.74],
      specular_weight: 0.5,
      specular_roughness: 0.7,
      lab_fluor_weight: 0.5,
      lab_fluor_color: [0.2, 0.25, 0.55],
      lab_fluor_uv: 0.9,
      lab_uv_ratio: 0.1,
    }),
  },
  // Authored, lab extension: uranium ("Vaseline") glass. The uranyl ion in it absorbs ultraviolet and violet-blue
  // light and re-emits green: pale yellow-green in daylight, glowing green right through under a black light.
  uranium: {
    label: 'Uranium glass',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({
      specular_roughness: 0,
      specular_ior: 1.52,
      transmission_weight: 1,
      transmission_color: [0.85, 0.92, 0.55],
      transmission_depth: 5,
      lab_fluor_weight: 0.6,
      lab_fluor_color: [0.25, 0.72, 0.03],
      lab_fluor_absorb: [0, 0, 0.1],
      lab_fluor_uv: 1,
    }),
  },
  // Authored, lab extension: tonic water, clear, with quinine (about 80 mg per liter) that absorbs near-ultraviolet
  // within a centimeter or so and re-emits blue: under a black light it glows where the light enters.
  tonic: {
    label: 'Tonic water',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({
      specular_roughness: 0,
      specular_ior: 1.34,
      transmission_weight: 1,
      transmission_color: [0.98, 0.98, 0.97],
      transmission_depth: 10,
      lab_fluor_weight: 0.45,
      lab_fluor_color: [0.15, 0.3, 0.55],
      lab_fluor_uv: 3.4,
    }),
  },
  // Official: examples/open_pbr_copper.mtlx
  copper: {
    label: 'Copper',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.811, 0.643, 0.542],
      base_metalness: 1,
      specular_color: [0.97, 0.95, 0.946],
      specular_roughness: 0.02,
    }),
  },
  // Official: examples/open_pbr_silver.mtlx
  silver: {
    label: 'Silver',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      base_color: [0.988, 0.985, 0.975],
      base_metalness: 1,
      specular_color: [0.995, 0.995, 0.998],
      specular_roughness: 0.02,
    }),
  },
  // Authored: glazed porcelain, a white body under a clear, glossy glaze (OpenPBR's coat). There is no official
  // ceramic example; the values are chosen for the look, not measured.
  ceramic: {
    label: 'Glazed ceramic',
    official: false,
    roughnessParam: 'coat_roughness',
    params: mat({
      base_color: [0.82, 0.81, 0.78],
      specular_roughness: 0.35,
      coat_weight: 1,
      coat_roughness: 0.03,
      coat_ior: 1.5,
    }),
  },
  // Authored on official examples/open_pbr_honey_liquid.mtlx: its color, given a depth of 12 cm (centimeters as
  // read for the subsurface presets, see subsurfaceScale). The official example has no depth, so its color tints
  // the refraction at the surface; with a depth it is the color light takes after that distance in the honey
  // (spec, Transmission: Beer-Lambert), so thin edges stay pale and the thick middle runs deep amber.
  honey: {
    label: 'Honey',
    official: false,
    roughnessParam: 'specular_roughness',
    params: mat({
      specular_roughness: 0,
      specular_ior: 1.5,
      transmission_weight: 1,
      transmission_color: [0.705, 0.582, 0.112],
      transmission_depth: 12,
    }),
  },
  // Official: examples/open_pbr_marble.mtlx (subsurface, the default radius of 1 read as centimeters).
  marble: {
    label: 'Marble',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      specular_roughness: 0,
      subsurface_weight: 1,
      subsurface_color: [0.813, 0.793, 0.759],
      subsurface_radius_scale: [0.851, 0.557, 0.395],
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
  // Official: examples/open_pbr_glass.mtlx (crown glass: IOR 1.52, Abbe number 64). Raise the roughness slider
  // for frosted glass.
  glass: {
    label: 'Glass',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      specular_roughness: 0,
      specular_ior: 1.52,
      transmission_weight: 1,
      transmission_dispersion_scale: 1,
      transmission_dispersion_abbe_number: 64,
    }),
  },
  // Official: examples/open_pbr_diamond.mtlx (IOR 2.42, Abbe number 55.3): its fire is the dispersion.
  diamond: {
    label: 'Diamond',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      specular_roughness: 0,
      specular_ior: 2.42,
      transmission_weight: 1,
      transmission_dispersion_scale: 1,
      transmission_dispersion_abbe_number: 55.3,
    }),
  },
  // Official: examples/open_pbr_skin_*.mtlx (the tone row picks which; see SKIN_TONES).
  skin: {
    label: 'Skin',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({}), // replaced by the tone's preset in heroParams
  },
  // Official: examples/open_pbr_soapbubble.mtlx. Thin-walled, IOR 1, so every color comes from the film.
  soapbubble: {
    label: 'Soap bubble',
    official: true,
    roughnessParam: 'specular_roughness',
    params: mat({
      specular_roughness: 0,
      specular_ior: 1,
      transmission_weight: 1,
      thin_film_weight: 1,
      thin_film_thickness: 0.5,
      thin_film_ior: 1.4,
      geometry_thin_walled: 1,
    }),
  },
}

// Skin: the six official examples, examples/open_pbr_skin_i.mtlx to _vi.mtlx (lightest to darkest), verbatim.
// Each is a subsurface medium under a rough dielectric (IOR 1.40, roughness 0.5) with the default radius of 1;
// the radius scale gives the per-channel mean free path. The examples do not state a length unit; the lab reads
// them as centimeters (red light's mean free path near 5 mm for tones I to III), see subsurfaceScale.
export type SkinTone = 'i' | 'ii' | 'iii' | 'iv' | 'v' | 'vi'
export const SKIN_ORDER: SkinTone[] = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
const skin = (color: RGB, radiusScale: RGB): OpenPBR =>
  mat({
    specular_roughness: 0.5,
    specular_ior: 1.4,
    subsurface_weight: 1,
    subsurface_color: color,
    subsurface_radius_scale: radiusScale,
  })
const SKIN_LIGHT: RGB = [0.482, 0.169, 0.109]
const SKIN_DARK: RGB = [0.367, 0.137, 0.068]
export const SKIN_TONES: Record<SkinTone, { label: string; params: OpenPBR }> = {
  i: { label: 'I', params: skin([0.762, 0.652, 0.568], SKIN_LIGHT) },
  ii: { label: 'II', params: skin([0.671, 0.505, 0.371], SKIN_LIGHT) },
  iii: { label: 'III', params: skin([0.545, 0.445, 0.359], SKIN_LIGHT) },
  iv: { label: 'IV', params: skin([0.351, 0.24, 0.148], SKIN_DARK) },
  v: { label: 'V', params: skin([0.227, 0.157, 0.091], SKIN_DARK) },
  vi: { label: 'VI', params: skin([0.073, 0.052, 0.025], SKIN_DARK) },
}

// Subsurface radii are lengths; the scene's units are not centimeters, so each scene converts: scene units per
// centimeter = 0.01 / meters per scene unit (see the models' metersPerUnit, and BALLS_METERS_PER_UNIT).
export const BALLS_METERS_PER_UNIT = 0.09 // the reference balls taken as 18 cm across (radius 1 unit)
export const subsurfaceScale = (metersPerUnit: number) => 0.01 / metersPerUnit

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

// The hero's parameters before any slider edits: car paint takes its finish (and flakes), skin its tone; the
// others their preset.
export function heroParams(hero: Hero, finish: PaintFinish, flakes: boolean, tone: SkinTone = 'iii'): OpenPBR {
  if (hero === 'skin') return SKIN_TONES[tone].params
  if (hero !== 'carpaint') return HERO_PRESETS[hero].params
  const p = PAINT_FINISHES[finish].params
  if (!paintHasFlakes(finish, flakes)) return p
  return { ...p, ...FLAKES, specular_roughness: Math.min(p.specular_roughness, 0.18) }
}
