// The pyro lab's controls: names follow the ones FX artists know from production pyro solvers.

export type PyroResolution = 'low' | 'medium' | 'high'
export type Advection = 'sl' | 'mac'
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
  advection: Advection
  pressure: PressureSolver
  vcycles: number // multigrid V-cycles per step
  pressureIters: number // red-black SOR sweeps per step
  // Source
  sourceType: SourceType
  srcRadius: number // m
  srcDensity: number // smoke emitted (smoke source)
  srcFuel: number // fuel emitted (fire source)
  srcTemp: number
  srcSpeed: number // m/s, upward
  srcNoise: number // 0..1
  // Forces
  buoyancy: number // m/s2 per unit temperature
  vorticity: number // confinement strength (dimensionless)
  turbulence: number // m/s2
  turbScale: number // m
  wind: number // m/s along x
  cooling: number // 1/s
  dissipation: number // 1/s
  // Shading
  densityScale: number // extinction per unit density, 1/m
  albedo: number
  anisotropy: number // Henyey-Greenstein g
  lightAz: number // degrees, 0 = behind the camera
  lightEl: number // degrees
  sunIntensity: number
  skyIntensity: number
  exposure: number // stops
  // Combustion and flames
  burnRate: number // reaction burned per second (a flame lasts fuel / burn rate seconds)
  heat: number // temperature released per unit of fuel burned
  expansion: number // 1/s at full flame
  soot: number // smoke per unit of fuel burned
  flameIntensity: number
  flameKelvin: number // the flame's peak temperature: solver temperature 1 maps to this, 0 to ambient (300 K)
  flameContrast: number // exponent on the blackbody brightness, 1 = physical
  // Upres detail
  detail: number // 0 = off
  detailScale: number // m, the largest noise feature
}

export const PYRO_DEFAULTS: PyroState = {
  resolution: 'medium',
  advection: 'mac',
  pressure: 'mg',
  vcycles: 2,
  pressureIters: 40,
  sourceType: 'fire',
  srcRadius: 0.3,
  srcDensity: 0.55,
  srcFuel: 1,
  srcTemp: 1,
  srcSpeed: 0.8,
  srcNoise: 0.8,
  buoyancy: 3,
  vorticity: 0.35,
  turbulence: 1.8,
  turbScale: 0.2,
  wind: 0,
  cooling: 1.5,
  dissipation: 0.08,
  densityScale: 12,
  albedo: 0.45,
  anisotropy: 0.3,
  lightAz: -45,
  lightEl: 35,
  sunIntensity: 0.5,
  skyIntensity: 0.12,
  exposure: 0,
  detail: 1,
  detailScale: 0.05,
  burnRate: 1,
  heat: 1.2,
  expansion: 0.8,
  soot: 0.3,
  flameIntensity: 1,
  flameKelvin: 1900,
  flameContrast: 0.5,
}

// Choosing a source sets the lighting and shading it reads best with (each control can still be changed): fire
// lights its own scene, so the sun and sky come down, and its smoke is soot, darker than plain smoke.
export const SOURCE_LOOKS: Record<SourceType, Partial<PyroState>> = {
  smoke: { sunIntensity: 2.6, skyIntensity: 0.35, albedo: 0.6, densityScale: 9, cooling: 0.5, srcTemp: 1, srcSpeed: 0.45, srcRadius: 0.22, turbulence: 1.2, vorticity: 0.25, buoyancy: 2.2 },
  fire: { sunIntensity: 0.5, skyIntensity: 0.12, albedo: 0.45, densityScale: 12, cooling: 1.5, srcTemp: 1, srcSpeed: 0.8, srcRadius: 0.3, turbulence: 1.8, vorticity: 0.35, buoyancy: 3 },
}
