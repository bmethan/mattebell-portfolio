// The pyro lab's controls: names follow the ones FX artists know from production pyro solvers, in physical units
// where there are some (physics.ts holds the constants).
import { FUEL } from './physics'

export type PyroResolution = 'low' | 'medium' | 'high'
export type Advection = 'sl' | 'mac'
export type VelocityScheme = 'ar' | 'mac' | 'sl'
export type PressureSolver = 'mg' | 'sor'
export type SourceType = 'smoke' | 'fire'

// Cells along x, y, z. The domain is DOMAIN_WIDTH metres wide (so the cell size follows from x) and shaped like the
// wide frame, 2.5 : 1 : 0.5, so a plume pushed sideways leaves through the frame's edges before it meets a side; the
// depth, seen end on, gets the fewest cells (about a metre, ample for the plume).
export const RESOLUTIONS: Record<PyroResolution, { dims: [number, number, number]; label: string }> = {
  low: { dims: [104, 42, 20], label: 'Low' },
  medium: { dims: [160, 64, 32], label: 'Medium' },
  high: { dims: [200, 80, 40], label: 'High' },
}
export const RESOLUTION_ORDER: PyroResolution[] = ['low', 'medium', 'high']
export const DOMAIN_WIDTH = 4.8

export interface PyroState {
  resolution: PyroResolution
  advection: Advection // the scalars
  velocity: VelocityScheme // advection-reflection, Modified MacCormack or semi-Lagrangian
  pressure: PressureSolver
  vcycles: number // multigrid V-cycles per step
  pressureIters: number // red-black SOR sweeps per step
  // Source
  sourceType: SourceType
  srcRadius: number // m
  srcDensity: number // smoke emitted (smoke source)
  srcTemp: number // K above ambient (smoke source)
  srcSpeed: number // m/s, upward (smoke source)
  srcNoise: number // 0..1
  srcX: number // m along the floor from the middle
  heatRelease: number // kW (fire source): the bed gives off the fuel that releases this much heat burning
  // Forces
  buoyancy: number // 1 = physical, g (T - T_ambient) / T_ambient
  vorticity: number // confinement strength (dimensionless)
  turbulence: number // m/s2 for a smoke source; for a fire, times the calibrated turbulence model (1 = calibrated)
  turbScale: number // m
  wind: number // m/s along x
  cooling: number // 1/s, beyond mixing and radiation (0 = physical)
  dissipation: number // 1/s (0 = physical)
  // Combustion
  sootYield: number // kg of soot per kg of fuel burned
  radiantFraction: number // share of the heat released that is radiated away
  expansion: number // 1 = physical
  flameDetail: number // subgrid temperature fluctuation, 1 = the model
  // Shading and camera
  densityScale: number // smoke source: extinction per unit density, 1/m
  albedo: number // single-scattering albedo of the smoke or soot
  anisotropy: number // Henyey-Greenstein g
  lightAz: number // degrees, 0 = behind the camera
  lightEl: number // degrees
  sunEV: number // the sun's illuminance as an incident-meter reading (lux = 2.5 x 2^EV)
  skyEV: number // the sky's, on open ground
  exposure: number // the camera's EV100 (manual)
  exposureMode: 'auto' | 'manual'
  exposureComp: number // EV of compensation on the metered exposure: positive brightens, as on a camera's dial
  whiteBalance: number // K: light from a blackbody at this temperature comes out neutral
  background: 'studio' | 'sky' // black around the floor's pool of light, or the sky and a floor to the horizon
  tricubic: boolean // sample the scalars with a cubic B-spline rather than trilinearly
  upres: boolean // a smoke source's density also on a grid twice as fine (wavelet turbulence)
  upresStrength: number // its turbulence, 1 = Kolmogorov's next band
  // Interaction
  ball: boolean // a solid ball in the flow, dragged by the visitor
  ballRadius: number // m
  // Upres detail
  detail: number // 0 = off
  detailScale: number // m, the largest noise feature
}

export const PYRO_DEFAULTS: PyroState = {
  resolution: 'medium',
  advection: 'mac',
  velocity: 'mac',
  pressure: 'mg',
  vcycles: 2,
  pressureIters: 40,
  sourceType: 'fire',
  srcRadius: 0.4,
  srcDensity: 0.55,
  srcTemp: 70,
  srcSpeed: 0.45,
  srcNoise: 0.8,
  srcX: 0,
  heatRelease: 180,
  buoyancy: 1,
  vorticity: 0.35,
  turbulence: 1,
  turbScale: 0.2,
  wind: 0,
  cooling: 0,
  dissipation: 0,
  sootYield: 0.04,
  radiantFraction: FUEL.radiantFraction,
  expansion: 1,
  flameDetail: 1,
  densityScale: 9,
  albedo: 0.45,
  anisotropy: 0.3,
  lightAz: -60,
  lightEl: 12,
  sunEV: 8,
  skyEV: 7,
  exposure: 7,
  exposureMode: 'auto',
  exposureComp: 0,
  whiteBalance: 5000,
  background: 'studio',
  tricubic: true,
  upres: true,
  upresStrength: 1,
  ball: false,
  ballRadius: 0.22,
  detail: 1,
  detailScale: 0.05,
}

// Choosing a source sets the light and camera it reads best with (each control can still be changed): a fire is
// shot at dusk, exposed for the flames as a photographer would, and its smoke is soot, darker than a smoke pot's.
export const SOURCE_LOOKS: Record<SourceType, Partial<PyroState>> = {
  smoke: { sunEV: 13, skyEV: 11, exposure: 11, whiteBalance: 6500, albedo: 0.6, turbulence: 2, vorticity: 0.25 },
  fire: { sunEV: 8, skyEV: 7, exposure: 7, whiteBalance: 5000, albedo: 0.45, turbulence: 1, vorticity: 0.35 },
}

// Starting points: each sets the source, the light and the wind, and the lab restarts from it. A fire's smoke is
// soot (dark, albedo about 0.25) when the fuel burns sooty; wood smoke also carries tarry droplets, which make it paler.
const FIRE_START: Partial<PyroState> = { sourceType: 'fire', ...SOURCE_LOOKS.fire, srcNoise: 0.8, radiantFraction: FUEL.radiantFraction, srcX: 0, wind: 0, exposureMode: 'auto', exposureComp: 0 }
const SMOKE_START: Partial<PyroState> = { sourceType: 'smoke', ...SOURCE_LOOKS.smoke, srcNoise: 0.8, srcRadius: 0.3, srcX: 0, wind: 0, exposureMode: 'auto', exposureComp: 0 }
export const BONFIRE_AT_DUSK: Partial<PyroState> = { ...FIRE_START, heatRelease: 180, srcRadius: 0.4, sootYield: 0.04, albedo: 0.45, sunEV: 8, skyEV: 7, lightAz: -60, lightEl: 12, whiteBalance: 5000 }
export const PRESETS: { id: string; label: string; state: Partial<PyroState> }[] = [
  { id: 'bonfire', label: 'Bonfire at dusk', state: { ...BONFIRE_AT_DUSK, background: 'studio' } },
  { id: 'campfire', label: 'Campfire at night', state: { ...FIRE_START, background: 'studio', heatRelease: 80, srcRadius: 0.3, sootYield: FUEL.sootYield, albedo: 0.35, sunEV: -1, skyEV: -2, lightAz: -45, lightEl: 35, whiteBalance: 5500 } },
  { id: 'oil', label: 'Oil fire in daylight', state: { ...FIRE_START, background: 'sky', heatRelease: 250, srcRadius: 0.4, sootYield: 0.12, albedo: 0.25, sunEV: 14, skyEV: 11, lightAz: -40, lightEl: 40, whiteBalance: 5500 } },
  { id: 'fire-wind', label: 'Fire in wind', state: { ...FIRE_START, background: 'sky', heatRelease: 120, srcRadius: 0.3, sootYield: 0.05, albedo: 0.4, sunEV: 12, skyEV: 10, lightAz: -45, lightEl: 30, whiteBalance: 6000, srcX: -1, wind: 1.8 } },
  { id: 'plume', label: 'Smoke plume', state: { ...SMOKE_START, background: 'studio' } },
  { id: 'smoke-wind', label: 'Smoke in wind', state: { ...SMOKE_START, background: 'sky', srcX: -1.4, wind: 1.2, skyEV: 10, lightAz: 30, lightEl: 40 } },
  { id: 'steam', label: 'Steam', state: { ...SMOKE_START, background: 'sky', albedo: 0.95, densityScale: 5, srcTemp: 60, srcSpeed: 0.7, srcRadius: 0.2, turbulence: 1.5, sunEV: 13, skyEV: 10, lightAz: 30, lightEl: 40 } },
]
