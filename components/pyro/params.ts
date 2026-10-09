// The pyro lab's controls: names follow the ones FX artists know from production pyro solvers.

export type PyroResolution = 'low' | 'medium' | 'high'
export type Advection = 'sl' | 'mac'
export type PressureSolver = 'mg' | 'sor'

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
  srcRadius: number // m
  srcDensity: number
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
}

export const PYRO_DEFAULTS: PyroState = {
  resolution: 'medium',
  advection: 'mac',
  pressure: 'mg',
  vcycles: 2,
  pressureIters: 40,
  srcRadius: 0.22,
  srcDensity: 0.55,
  srcTemp: 1,
  srcSpeed: 0.45,
  srcNoise: 0.8,
  buoyancy: 2.2,
  vorticity: 0.25,
  turbulence: 1.2,
  turbScale: 0.2,
  wind: 0,
  cooling: 0.5,
  dissipation: 0.08,
  densityScale: 9,
  albedo: 0.6,
  anisotropy: 0.3,
  lightAz: -45,
  lightEl: 35,
  sunIntensity: 2.6,
  skyIntensity: 0.35,
  exposure: 0,
}
