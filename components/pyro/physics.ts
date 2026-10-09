// Physical constants for the pyro lab's fire, in SI units. The combustion follows the "fast chemistry" model of
// NIST's Fire Dynamics Simulator (FDS): fuel and oxygen react as fast as turbulence mixes them (the eddy dissipation
// concept), the heat released raises the gas temperature, a measured fraction of it is radiated away, and soot forms
// as a measured fraction (the yield) of the fuel burned. The light comes from that same soot.

export const G = 9.81 // m/s2
export const T_AMBIENT = 293 // K
// Ideal-gas density of air at one atmosphere: rho = p M / (R T) = 353 / T kg/m3 (M = 0.02897 kg/mol).
export const RHO_T = 353
export const Y_O2_AIR = 0.233 // mass fraction of oxygen in air
// Specific heat of hot combustion products (it rises from about 1.0 at room temperature to about 1.3 kJ/(kg K) in a
// flame); one value for the whole range.
export const CP = 1200 // J/(kg K)

// Wood, as FDS models it: the gas a burning wood fire gives off, lumped as C3.4H6.2O2.5. Burning a kilogram takes
// 1.36 kg of oxygen (3.7 mol of O2 per 87 g), and releases 13.1 MJ per kilogram of oxygen consumed (Huggett 1980),
// so 17.8 MJ per kilogram of fuel.
export const FUEL = {
  oxygenPerFuel: 1.36, // kg O2 / kg fuel
  heatPerFuel: 1.36 * 13.1e6, // J / kg fuel
  sootYield: 0.015, // kg soot / kg fuel burned (wood)
  radiantFraction: 0.3, // share of the heat released that is radiated away
}
// The stoichiometric mixture fraction: the share of fuel-stream gas in a mixture with exactly enough air to burn it.
export const Z_STOICH = Y_O2_AIR / (FUEL.oxygenPerFuel + Y_O2_AIR)

// Subgrid mixing (large-eddy simulation): Smagorinsky's constant and the turbulent Schmidt number, FDS's values.
export const SMAGORINSKY = 0.2
export const SCHMIDT_T = 0.5

// The fire's turbulence model. At the lab's resolution the grid cannot resolve the eddies that mix air into a fire's
// plume (without them the plume stays three to four times too hot), so synthetic turbulence stands in for them, at
// the strength of eddies of the fire's own velocity: TURB_C u^2 / l, with u = 1.9 Q^(1/5) m/s (McCaffrey's velocity
// scale, Q in kW) and l the noise's scale. TURB_C was calibrated once, on an 80 kW fire, against McCaffrey's measured
// centreline temperatures (balancing the intermittent flame against the plume above), with FDS's mixing time in the
// combustion; other fire sizes then follow from the scaling.
export const TURB_C = 0.02
export const fireTurbulence = (kW: number, scale: number) => (TURB_C * (1.9 * kW ** 0.2) ** 2) / scale

// Soot optics. Mass-specific extinction of flame-generated smoke, 8.7 m2/g at 633 nm (Mulholland and Croarkin
// 2000), scaled to the reference wavelength (green, 550 nm) by the soot's absorption spectrum, lambda^-alpha with
// alpha about 1. The single-scattering albedo of soot is low; the lab's albedo control sets it.
export const SOOT_ALPHA = 1
export const SOOT_EXTINCTION_633 = 8700 // m2/kg
export const REF_NM = 550
export const SOOT_EXTINCTION = SOOT_EXTINCTION_633 * (633 / REF_NM) ** SOOT_ALPHA // m2/kg at 550 nm
// Extinction relative to green at the effective wavelengths of the display's red and blue primaries: soot dims blue
// light more than red (which is why the sun turns orange through smoke).
export const SOOT_CHROMA: [number, number] = [(REF_NM / 612) ** SOOT_ALPHA, (REF_NM / 465) ** SOOT_ALPHA]

// Light levels as photographers meter them: an incident meter reading of EV (ISO 100) is 2.5 x 2^EV lux; a camera at
// EV100 saturates at 1.2 x 2^EV100 cd/m2 (Lagarde and de Rousiers 2014).
export const luxOfEV = (ev: number) => 2.5 * 2 ** ev
export const exposureOfEV = (ev: number) => 1 / (1.2 * 2 ** ev)
// Auto exposure. The ambient light is metered as an incident meter reads it, less 2.3 EV (smoke is pale and sits in
// the open, and the camera's saturation-based exposure runs darker than a meter's grey); a fire sets the least
// exposure (the highest EV) the frame may take: EV 7 for an 80 kW campfire (about 130 cd), and half a stop more for
// each doubling of the fire's intensity (less than in proportion: photographers let flames burn out a little).
export const METER_AMBIENT = 0.2
export const METER_FIRE_EV = 7
export const METER_FIRE_CD = 130
export const METER_FIRE_SLOPE = 0.35

// The fire's light is gathered from this many slices of the grid's height, each a point light at its own centre.
export const FIRE_BANDS = 8

// A WGSL float literal.
export const wf = (x: number) => (Number.isInteger(x) ? `${x}.0` : x.toPrecision(7))
