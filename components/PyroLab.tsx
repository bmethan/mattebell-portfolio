'use client'
// The lab's pyro mode: a 3D smoke solver and volume renderer on the GPU (see pyro/engine.ts), stirred by dragging
// across the image, with the solver's controls below. It mounts when the Lab switch first selects it, and starts
// at once (choosing it is the visitor's go-ahead).
import { useEffect, useId, useRef, useState, useSyncExternalStore } from 'react'
import { PyroEngine, type BallStatus, type PyroStatus } from './pyro/engine'
import { ZipStore } from './pyro/vdb'
import { PRESETS, PYRO_DEFAULTS, RESOLUTIONS, RESOLUTION_ORDER, SOURCE_LOOKS, type PyroState, type SourceType } from './pyro/params'
import { PYRO_CITATIONS, PYRO_CITATION_GROUPS } from './pyro/citations'
import { citeAuthors } from './lookdev/citations'
import { Group, Pill, Range, Tabs } from './lab/ui'
import DeviceCheck from './pyro/DeviceCheck'
import { VIEW_TRANSFORMS } from './lab/viewTransforms'
import { ENVIRONMENTS, ENV_ORDER, type Env } from './lookdev/environments'

// Each HDR environment's illuminance on open ground when it is picked (the Environment EV; lux = 2.5 x 2^EV): a clear
// midday about 100,000 lux, an overcast day about 15,000, the sun just set about 600, a photo studio about 1,200.
const ENV_EV: Record<Exclude<Env, 'none'>, number> = { midday: 15.5, overcast: 12.5, sunset: 8, studio: 9 }
// The Environment row: none (the sun and a flat sky), the Lookdev lab's softbox rig, then its four HDR maps.
const LIGHTINGS: PyroState['env'][] = ['none', 'rig', ...ENV_ORDER.filter((e): e is Exclude<Env, 'none'> => e !== 'none')]

const TOUCH_SLOP = 8 // px a finger must travel sideways before it stirs (a vertical swipe scrolls the page)
const noSubscribe = () => () => {}

type PyroTab = 'source' | 'forces' | 'fire' | 'shading' | 'solver'
const PYRO_TABS = [
  { id: 'source', label: 'Source' },
  { id: 'forces', label: 'Forces' },
  { id: 'fire', label: 'Fire' },
  { id: 'shading', label: 'Shading' },
  { id: 'solver', label: 'Solver' },
] as const

const TECH_NOTES: [string, string][] = [
  [
    'Solver',
    'Air, smoke and fire on a staggered grid, the layout production solvers use: velocity on the cell faces, everything else at the cell centres. Each step, 30 per simulated second, advects the velocity, adds the forces, then projects the flow with a pressure solve: incompressible, except where burning gas expands.',
  ],
  [
    'Pressure',
    'Geometric multigrid, as production solvers use: each V-cycle smooths with red-black Gauss-Seidel on the full grid, hands the remaining error down through coarser grids (each half the resolution of the last), where the broad, slow-to-fix part of it is cheap to remove, and brings the corrections back up. Two V-cycles a step, started from the previous step’s pressure, leave about 1% of the flow’s divergence, the floor set by storing velocity at half precision (measured on this solver). The Solver tab also offers plain red-black SOR for comparison: 40 sweeps reach the same floor at four to eight times the GPU time, and a shorter solve shows as smoke that swells and loses some of its swirl.',
  ],
  [
    'Advection',
    'Velocity and scalars move with Modified MacCormack: a forward and a backward semi-Lagrangian step, half the round-trip error added back, then clamped to the values the forward step read, so detail survives without new extremes. The Solver tab also offers semi-Lagrangian alone, the softer, more diffusive choice, and for the velocity advection-reflection (Zehnder, Narain and Thomaszewski 2018): it advects half a step, projects, reflects the velocity through the projection and advects the reflection the second half, which loses far less of the flow’s energy to the projection. It needs two full pressure solves a step (with one V-cycle each, the reflection amplifies the solver’s error until it blows up). Measured against fire experiments it mixes the plume no differently from MacCormack at this resolution, where the grid itself, not the scheme, limits the turbulence, so MacCormack, the cheaper of the two, is the default. A smoke source’s density and temperature go further, with BiMocq² (Qu, Zhang, Gao, Jiang and Chen 2019): rather than re-sampling last step’s field each step, which blurs it a little every time, they are rebuilt from snapshots taken at the last two re-initialisations (every 16 steps), through maps of where the gas has travelled since, with what the sources and mixing change gathered back along the same maps. Measured on the smoke plume, its edges stay 4 to 13% sharper than with MacCormack, and 16 to 24% with IVOCK as well (see Vorticity). A fire’s fields keep MacCormack: their combustion needs the mass keeping described under Mixing.',
  ],
  [
    'Vorticity',
    'Advecting the velocity loses some of the flow’s swirl every step, most of all in its smallest eddies. For a smoke source, IVOCK (Zhang, Bridson and Greif 2015) restores it: the vorticity before the step is stretched by the flow and carried along by the same scheme as the velocity, and what the carried velocity’s own vorticity falls short of it is put back through a vector potential. It keeps 15 to 37% more swirl (enstrophy) in the smoke plume. With the numerical loss gone the flow needs a real sink for its smallest eddies, so Smagorinsky’s eddy viscosity, which FDS uses on the momentum, comes with it. This lab’s version differs from the paper: cell-centred where the paper staggers it on cell edges, its potential solved in a screened form over a few cells, and it restores only lost strength along the vorticity’s own direction (unrestricted, it fed on itself). On a fire it mixed the plume far too fast, its centreline down to a tenth of McCaffrey’s temperatures, most likely the fire’s strong expansion meeting this cell-centred form, so fires keep the validated solver. The pair costs a smoke source about a third of the solver’s speed on an integrated GPU, and the Solver tab turns each off.',
  ],
  [
    'Flow maps',
    'The Solver tab also offers leapfrog flow maps for a smoke source’s velocity (Sun, Li, Wang, Wang, Li, van Bloemen Waanders and Zhu 2025). Rather than re-sampling the velocity every step, which blurs it a little each time, the solver follows where the air has travelled over four steps, through half-step velocities that leapfrog over one another, and rebuilds the velocity from where it started, carried along those paths as impulse, with the forces gathered along the way and the error of a round trip through the paths taken off. The paper’s examples are closed boxes. This lab’s box is open on four sides and the top, where the pressure is held at zero, which the rebuilt impulse does not respect, and the first version ran away within seconds. Four additions hold it: the impulse’s gauge (the gradient that separates it from the velocity) followed along the same paths and taken off; the velocity itself carried near the open sides and wherever a path has stretched too far within its four steps; the source’s push applied again at each rebuild; and Smagorinsky’s eddy viscosity, as with IVOCK, as a sink for the smallest eddies. Stable, it matches IVOCK rather than beating it: about the same swirl, edges 5 to 8% softer, and structures that hold together longer as they rise, in larger, smoother folds. It costs about a tenth of the smoke’s speed on an integrated GPU, with a heavier step every fourth, and IVOCK stands aside while it runs.',
  ],
  [
    'Forces',
    'Buoyancy at its physical strength: gas hotter than the air around it rises at g (T − T₀) / T₀. Vorticity confinement, which restores the small swirls a coarse grid smooths away. Divergence-free turbulence from the curl of a noise potential, rising with the plume and churning as it rises (held still, the updraft would draw its pushes out into steady vertical streaks): for a smoke source its strength is the Turbulence control, in m/s²; for a fire it is a turbulence model, calibrated against measurements (see Validation). Wind, and a smoke source’s push. Cooling and dissipation default to 0: the plume cools by mixing with the air it draws in.',
  ],
  [
    'Combustion',
    'The fire follows the fast-chemistry model of NIST’s Fire Dynamics Simulator (FDS), the standard tool of fire-safety engineering. The bed is a disc on the floor that gives off wood gas (Douglas-fir volatiles, C3.4H6.2O2.5) through its surface at the rate that releases the chosen heat, as FDS models a burning surface, the vapour pushing out at its own small blowing speed. Two fields track it: the mixture fraction, the share of the gas that came from the fuel, and the fuel still unburned. In each cell, fuel burns with the oxygen mixed in with it as fast as turbulence mixes them (the eddy dissipation concept, with FDS’s mixing time: the turnover time of the eddies finer than the grid, from the same Smagorinsky model the Mixing note describes, and never slower than the buoyant time √(2Δx / g)), so the flame sits where fuel meets air, a sheet around a fuel-rich core, and its height comes out of the simulation rather than a control. Burning a kilogram of the fuel takes 1.36 kg of oxygen and releases 17.8 MJ (13.1 MJ per kilogram of oxygen, Huggett’s constant); 30% of the heat is radiated away and the rest heats the gas, and 1.5% of the fuel’s mass becomes soot, wood’s measured yield. Heated gas expands at (dT/dt) / T, which the pressure solve turns into the flame’s outward push.',
  ],
  [
    'Mixing',
    'Three corrections make a coarse grid mix the way gas does. A fire’s fields are kept per unit volume (the density deficit and partial densities) rather than as a temperature and mass fractions, because the solver mixes by volume and gas mixes by mass: equal volumes of 1300 K gas and 293 K air make 478 K, not the 796 K an average of temperatures gives. Without this the plume stayed three times too hot. And mixing finer than the grid, as large-eddy simulations model it: Smagorinsky’s eddy diffusivity over a turbulent Schmidt number (0.2 and 0.5, FDS’s values), on the scalars. And mass: semi-Lagrangian advection carries values along the flow, not masses, and where burning gas swells to several times its volume in a step it made too little room for it, so a quarter to a third of the fire’s products and heat, and a tenth of its fuel, simply vanished. A pass before the advection measures how widely each cell’s gas was spread and scales it to the room it should fill, after Lentine, Grétarsson and Fedkiw’s conservative semi-Lagrangian method (which scales over-drawn cells down and traces the undrawn rest of a cell forward; here both are a scale, toward a room that includes the expansion); MacCormack’s sharpening, which is not mass-neutral where gas expands, stands aside in those cells. Where the gas is not expanding, what the scaling adds back is held to the most the gas it came from held, as advection keeps it anyway: without that bound, MacCormack’s sharpening and the scaling both restored the same peak, and beside the collision ball a speck of soot grew until it filled the air. The fire now releases 98 to 102% of the heat asked for.',
  ],
  [
    'Flames',
    'Most of a fire’s light is its soot’s glow. Hot soot emits what it absorbs times the blackbody radiance of its temperature (Kirchhoff’s law), so a flame’s colour and brightness follow from the simulated temperature and soot alone: Planck’s law through the CIE colour matching functions, in absolute units (a blackbody at 2042 K comes out at 600,000 cd/m², the old definition of the candle). The colours are worked in ACEScg, whose primaries hold a blackbody’s colour at every temperature a flame reaches, where Rec.709’s lose its blue below about 1900 K. The soot’s extinction is the measured 8.7 m²/g of flame smoke and its albedo 0.25, and it absorbs blue more than red (as 1/λ), so it reddens what shines through it. Below about 1000 K its glow is too faint to see, which is where the visible flame ends. A cell’s temperature is an average, and a blackbody’s glow climbs so steeply with temperature that the average glow is far from the glow at the average temperature (the turbulence-radiation interaction), so the renderer adds the temperature fluctuation finer than the grid: its size from the resolved temperature gradient, as large-eddy simulations model a subgrid variance (Pierce and Moin), only its shape from a noise that rises with the flames and churns (Flame detail scales it; 0 renders the cell averages). For the same reason, where fuel is burning the soot radiates at the temperature of the flame sheet it forms in, about 1400 K, rather than its cell’s mean: the burning happens in sheets far thinner than a cell, and the cell spreads their heat over all of it (a thin flame along a burning surface would otherwise never glow). The burning gives off light of its own too: molecules formed excited in the reactions give it up as CH’s blue at 431 nm, C2’s green Swan bands and a faint CO2 continuum (chemiluminescence), in proportion to the heat released. It is the blue of a gas flame, about a ten-thousandth of a lumen per watt of heat (an estimate: no measurement of it turned up), far fainter than glowing soot, so it shows only where there is no soot yet. And soot takes some 20 ms in the hot gas to form (Kumfer, Skeen and Axelbaum), so where burning has only begun, at the flame’s base, the gas is still clear. A wood fire’s soot hides nearly all of the blue at the grid’s scale; a fuel that makes no soot burns all blue (the Alcohol fire), too dim to see in daylight, as alcohol fires are. The fire lights its own smoke, the floor and everything near it: its emitted light is summed on the GPU every step in eight slices of its height, each a point light at its brightness-weighted centre, and each point of the smoke gathers them through the soot in the way, so dense smoke shadows the fire’s light as it does the sun’s; the floor and the balls take the same shadow. The light is summed as the flames are drawn, from the interpolated gas at eight points in each cell rather than each cell’s own temperature: the glow climbs so steeply with temperature that the hottest cells, at the bed, gave several times the light the flames show (the path-traced reference found it). Within a slice, its soot is mixed with its own glow, and how much of the glow gets out depends on the way it goes: every glowing cell’s light is marched a bed’s width through the soot about it toward 26 directions over the sphere (a quarter of them each step), and the shares, weighted by each cell’s light, make a table the smoke, the floor and the set each read in their own direction. A bonfire’s brightest flames sit in a sheet of soot at its bed, and let out two to three times as much of their light a little below the level as along it. Of what the soot stops it scatters a share, mostly onward (the albedo and the anisotropy); the rest, scattered every way from where it was stopped, starts nearer the soot’s edge than the flames that made it, and gets out more easily and more evenly, by a second table marched with the first. Glowing soot also scatters its own glow: where it is thick the light about it nears its blackbody radiance, so a thick flame glows at the full radiance whatever its albedo (Kirchhoff’s law), where it glowed only the share its soot does not scatter (the bonfire’s flames, a fifth dark against the path-traced reference).',
  ],
  [
    'Detail',
    'Wisps finer than the grid, in the spirit of production upres: two layers of texture coordinates ride with the flow, and the renderer reads a fine noise at them, so the detail moves with the smoke rather than sliding through it. The flow stretches each layer as it shears it, so each resets on its own schedule, half a period after the other, and the two cross-fade: a layer is invisible at the moment it resets. A stretched layer would streak its noise along the stretch, so each layer also tracks how far the flow has stretched it (after Neyret 2003) and counts for less the more it has; where both are stretched, as in the fast updraft over a fire, the detail fades rather than streaks. The detail works hardest in thin smoke, at the edges and in the wisps, and leaves the faintest haze smooth, where fine noise would read as grain. It shapes what the camera sees; the light still travels through the simulated density.',
  ],
  [
    'Upres',
    'A smoke source’s density also runs on a grid twice as fine, in the manner of wavelet turbulence (Kim, Thürey, James and Gross): carried by the solver’s velocity plus turbulence too fine for the coarse grid, the curl of a noise, so it stays divergence free, read at the texture coordinates the detail layers carry with the flow (its eddies travel with the smoke and fade where the flow has stretched them), and as strong as the local flow times 2^(−5/6), the share of the next band of eddies in Kolmogorov’s spectrum. The camera sees this finer smoke; the lighting still comes from the coarse grid. It takes the place of the rendered detail for smoke, and the Solver tab turns it off. A fire keeps the rendered detail: its soot and temperature come from the combustion on the coarse grid.',
  ],
  ['Boundaries', 'A solid floor, and open sides and top: smoke leaves the box, and the air that flows in carries none. And the collision ball, when there is one (see Interaction).'],
  [
    'Interaction',
    'Drag to stir the gas, move the source along the floor, or move a solid ball through the flow; the arrow keys do the same. The ball is a solid in the solver: the pressure solve treats its cells as walls at every multigrid level (a cell is solid when its centre is inside it, so the coarse levels see it too), and the faces around it move with its velocity, so a moving ball pushes the gas and the gas flows around it. It casts soft shadows from the sun on the floor and the smoke, and from the fire on the smoke (a penumbra as wide as the bed’s), hides its share of the sky (its view factor from the floor, (r/d)² cos θ), and is lit by the fire: by the flames touching it, and by each slice of the fire as a light as wide as the bed, through the soot in the way. The grid holds it as a staircase of cells, which the gas within a cell or two of its surface takes the shape of, so there the renderer fades the gas out and draws the true sphere. Each starting point sets a source, its look and the wind, and starts again.',
  ],
  [
    'Burning',
    'The ball is wood, a pine of 500 kg/m³, and it burns as wood does. A map of its surface, 128 by 64 patches, each a 1.5 mm surface layer over a 6 mm layer over the cool core, takes in the fire’s heat every step: radiation from the soot just outside it (soot’s Planck-mean absorption coefficient, with the flames’ emissivity held to 0.25, as for wood flames some 40 cm deep), the radiant share of the fire’s heat from further off, and convection. It loses heat by its own radiation and by conduction inward. Above about 550 K the wood chars and gives off its volatiles, the char front advancing at an Arrhenius rate in the temperature at the front (about 2 mm a minute at 650 K); pine leaves 16% of its mass as char, and the char insulates, so the front cools as it deepens. Once the volatiles come off faster than about 2 g/m² a second, wood’s critical mass flux for flaming (Bamford, Crank and Malan measured 2.5), they reach the gas as fuel and burn in the solver’s own combustion, with the bed fuel’s heat, soot and smoke. The flame along a burning surface is thinner than a cell, so two of its effects are modelled rather than resolved: its heat on the wood above a burning patch, which grows with the burning height beneath to about 25 kW/m² (measured averages run 15 to 20, the usual assumption 25 to 30; it is how fire climbs a surface, and why a small new patch goes out without outside heat while a tall one keeps going), and its look, which the renderer draws from the model’s map of the fuel flowing up the surface: a layer 1 to 3 cm thick that clings to it and burns in narrow licks with the char showing between them, then gathers over the top and rises as a plume that narrows as it climbs, present half the time 2 to 12 cm above the ball with tips flickering to twice that, as wall-fire correlations give for its strength, its soot at 1400 K at the root and 1050 K at the tips, as wood flames measure. The wood is not uniform: its late-wood rings are denser than its early wood (1.41 and 0.77 times the mean, in Scots pine), so until it chars it heats unevenly, and with the flicker of the flame’s heat the front climbs raggedly along the grain. Held in a 180 kW bonfire’s flames the ball browns within seconds and catches in 6 to 8 s. Pulled out soon after, its flames die; after a minute in the flames it keeps burning and the fire climbs it, until its char grows thick enough to starve it, three or four minutes on. Glowing char oxidises while there is oxygen at its surface, crazes into cracks that glow hotter, and ash builds on char that glows long enough. Its surface is shaded as an OpenPBR material, as the Lookdev lab’s are: the pine with a broad, faint sheen; the char with a tighter, silvery one on its blocks that catches the fire’s reflection, and none down in its cracks, which have depth; the ash as a pale fuzz (the Lookdev lab’s own, after Zeltner and colleagues), the glow as emission. Fresh ball starts again; Burning ball off makes it inert grey.',
  ],
  [
    'Rendering',
    'Each pixel marches through the volume and integrates each step in closed form, in photometric units (cd/m²), each colour channel on its own. Sunlight reaches every point through the optical depth toward the sun, recomputed each frame on a half-resolution grid from the smoke the camera sees (with the smoke upres on, its finer grid, averaged over the coarse grid’s cells); multiple-scattering octaves stand in for light bouncing inside the smoke, each weighted by the chance that light scattered once meets more smoke (toward the sun, up and down) and is scattered again rather than absorbed (its albedo), so pale steam takes far more of it than soot. The sky lights the smoke through the smoke around it (its mean transmittance in five directions), by the share of the scattering the sky fills (smoke that scatters forward, seen from below, takes more than half), and bounces inside it too; the floor lights it from below, as the floor straight under it is lit. The floor takes the smoke’s shadow and the fire’s light. Between cells the smoke and soot are sampled with a cubic B-spline (eight trilinear fetches placed so the hardware’s blend gives the spline’s weights, after Sigg and Hadwiger), smooth where trilinear interpolation creases at every cell face; the Shading tab switches back to trilinear, which renders in about two thirds of the time. Each frame starts its march at an offset from spatiotemporal blue noise (Wolfe, Morrical, Akenine-Möller and Ramamoorthi 2022), made for the lab by void and cluster: each frame’s offsets are blue noise in space, and each pixel’s offsets from frame to frame are blue noise in time, which cuts the noise of the first one to four accumulated frames by 9 to 16% against blue noise stepped by the golden ratio. Frames accumulate: each blends into a running history, first clamped to the range of the new frame’s neighbourhood so what moves is not smeared behind it, which averages the sampling noise away. The blend is in linear light (blended in a compressed one, as is common to keep bright pixels from flickering, a flame averaged a sixth darker than its own frames). A paused frame keeps accumulating until it settles.',
  ],
  [
    'Camera',
    'Exposed like a real camera. Light is set as a light meter reads it, the sun and sky as incident readings (2.5 × 2^EV lux), and the exposure is the camera’s EV at ISO 100, saturating at 1.2 × 2^EV cd/m² (Frostbite’s physically based camera). The white balance is a Bradford chromatic adaptation from a blackbody of the chosen temperature to the display’s white, the step Nguyen and colleagues took for fire; then the Lookdev lab’s own view transforms: ACES 2.0 (baked from OpenColorIO 2.5) by default, or AgX, Khronos PBR Neutral or a plain sRGB curve. The shutter is a film camera’s at 24 frames a second, 180 degrees by default (open 1/48 s): each sample along a ray takes its own moment within it and reads the gas where the flow had it then, so fast flames streak as they do on film. The lens scatters a few percent of the light it passes (4% by default) into a wide, soft glare, by a chain of blurs that keeps the light’s energy (Jimenez’s, from Call of Duty), which shows only around what is far brighter than its surroundings, as the flames’ cores are. Auto exposure meters the scene as a camera in program mode would: the sun and sky as an incident meter reads them, but never brighter than the fire allows (read from its flames as they are drawn: EV 7.5 to 8.5 for an 80 kW campfire as it puffs, a third of a stop more for each doubling of its light), eased over a second so the flames’ puffing does not pump it; its brightest flames saturate to yellow-white, as they do in photographs. The background is a studio’s black, or the sky (its radiance the sky’s illuminance over π, a little brighter toward the horizon) over a floor that runs into the horizon’s haze, or the Lookdev lab’s cyc in its charcoal or gray, the floor sweeping up into a wall behind the fire, lit by the fire too. The cyc’s wall hides the half of the sky behind it from the cyc itself, which sees the rest of the cyc there: the floor lights the wall and the wall the floor, and the fire-lit floor lights the wall behind the fire.',
  ],
  [
    'Environments',
    'The Lookdev lab’s four HDR environments from Poly Haven can light the scene. A volume renderer cannot sample a map at every scattering event at this frame rate, so each map’s brightest light, where it has one, is pulled out as a key that casts the smoke’s shadows (the midday sun, 68% of that map’s light within a degree of it; the photo studio’s main softbox, 43% within 15 degrees), and the rest is kept as three bands of spherical harmonics: convolved with the smoke’s phase function for its scattering, and with the cosine for the floor and the ball (Ramamoorthi and Hanrahan). The overcast sky and the sunset’s glow have no key, so they light the smoke softly from all around. Below the horizon the map gives way to the floor, lit by the whole environment. The Turn slider turns the map; the Environment EV is its illuminance on open ground, key included, which auto exposure meters. The softbox rig is the Lookdev lab’s three softboxes as lights, at that lab’s ratios: the key where the light sliders put it (4300 K), a fill on the far side (8000 K, about a sixth of the key) through the same spherical harmonics, and a rim strip behind (4300 K, about as strong as the key) with a light volume of its own, so the smoke shadows it and only its edges and wisps catch it.',
  ],
  [
    'On-set reference',
    'The Lookdev lab’s reference set stands beside the fire, as a lighting team shoots one on set to record the light where an effect will go: its ColorChecker (X-Rite’s published values), an 18% grey ball and a chrome ball, the balls 16 cm across on thin stands at about the middle of the flames’ height, the chart on the floor in front of them, turned to the camera and a little toward the fire. It stands in the bonfire’s opening scene, and the Shading tab turns it on or off for any other. It stands on the fire’s upwind side (the right, in still air), clear of the bed and inside the frame; where the frame leaves no room upwind, it stands downwind, as far from the fire as the frame allows. The balls and the chart are solids in the simulation, as the collision ball is, so the gas flows around them (the stands, thinner than a cell, are left out). The grey ball and the chart are lit as the burning ball is, by the sun, the sky and the fire through the smoke, so the chart shows the firelight’s colour against the camera’s white balance. The chrome ball is a mirror with OpenPBR’s metal Fresnel (its F82 tint): each ray that meets it is reflected and marched again through the gas, more lightly than the camera’s rays (twice the step, without the motion blur or the subgrid detail), so it shows the fire, its smoke, the stage, the collision ball and the rest of the set as they stand, and the sun and the rig’s softboxes as highlights. The set’s pieces shadow each other, the collision ball and the floor, from the sun, the rig’s rim and each slice of the fire, each hiding no more than its share of the light: a stand 1.6 cm across in front of a flame a metre wide dims the floor behind it a little, not to black. (Shaded to black whatever their size, the set’s shadows of the fire on the floor came out two to three times too dark against the path-traced reference.)',
  ],
  [
    'Path-traced reference',
    'Paused, the Path trace button traces the same frame with its light carried exactly, one path a pixel each frame, averaged, behind a wipe against the live render (drag it). The gas is the live render’s own, step for step: its density, detail and subgrid flame temperature, the soot formed by now and the flame’s blue light, each path at its own moment of the shutter. What the reference carries exactly is what the live render takes shortcuts with: light scattered any number of times through the smoke (the live render folds multiple scattering into octaves of the sun’s light); the fire’s light from the flames themselves, shadowed by the smoke on the way (the live render’s is eight point lights through a light volume at half resolution); the sky through the smoke in every direction (not only straight up); an HDR map in full (not nine spherical harmonics). At every scattering and surface a path samples the sun’s disc, a map’s key or the rig’s softboxes, and the flames: a point drawn in one of eight slices of the fire’s height by its light, chosen from eight candidates in proportion to what each would bring (resampled importance sampling, after Talbot, Cline and Egbert). Paths end by Russian roulette, and the reference stops at 4,096 paths a pixel. The surfaces are the live render’s materials, simplified (the chrome ball a mirror, the rest a matte base under a glossy layer). Checked with it, the live render’s fire light is now within about 45% of the reference’s across the presets (an oil fire in daylight within 15%; a campfire’s smoke and floor up to 45% darker), where it was up to seven times too bright. The smoke lit by the sun was 30 to 60% too bright, dark smoke the most, from the octaves, which added the same three quarters again to thin smoke as to thick and to soot as to steam. Split by where the light comes from (the reference can leave out the floor, the sky or the sun, or keep only light scattered once), the smoke’s light showed four more gaps. Its shadows came from the coarse grid while the camera saw the finer upres smoke, so the densest smoke caught a third too much sun. The sky came through the depth straight up only, so the middle of a plume took a fifth too much and the smoke under its cap a third too little. And the sky’s light scattered more than once (5 to 10% of a plume’s light) and the floor’s (1 to 9%) were missing. With those fixed and the octaves fitted to plumes of albedo 0.2, 0.6 and 0.95, a plume’s thin edges are within 8% of the reference, where they were 10% short, and the error over the smoke presets falls by half to 70%. On the bonfire, with scattering in the gas left out of both, the live flames’ glow matched the reference’s; what they lacked was the light their soot scatters of its own glow, and the cyc’s floor took the whole sky though the wall hides half of it, and no light from the wall. With those in, the bonfire’s flames are within 2% of the reference and its cyc within 5%, where they were 20% and 10% short, and its far floor, a fifth too bright, is within 5%. The wall behind the fire was a sixth dark from a shadow taken along the wrong line, straight up through the plume above the flames; with that fixed, every part of the bonfire’s frame is within 8% of the reference. Taken apart, though, its fire’s direct light ran half again too strong, a sooty flame letting out less of its light than the uniformly glowing sphere the live render took it for, while the glow its soot scatters out was missing, and on its wall and floor the two offset. Checked slice by slice against every glowing cell’s light marched through the soot to each point, the sphere let out 1.7 times a bonfire’s light and six times an oil fire’s; the table of directions is within a tenth. Against the reference, the bonfire’s direct light on its surfaces is now some 15% short where it was half again over, and with scattering in the gas left out of both, every part of its frame is within 9%. Split by where it comes from, nine tenths of the light the gas scatters onto the floor beside the fire is the fire’s own, scattered by the flames’ soot rather than the smoke above; with that light leaving the soot as scattered light does, every part of the bonfire’s frame is within 6% (within 8% while the two errors offset). A campfire’s surfaces, a quarter dark in the whole frame, are within 15%: what was left short on one side is the floor right beside its bed, where its flames sit to one side of each slice’s centre, which a point light per slice cannot follow. A fire in wind’s error falls by more than two thirds.',
  ],
  [
    'Export',
    'The Solver tab writes the simulation as OpenVDB, the volume format of film effects work, so a frame or a two-second sequence can be rendered in Houdini, Blender or any other program that reads it. The writer is the lab’s own, a few hundred lines: each grid a standard OpenVDB tree (a root, internal nodes of 32³ and 16³ children, leaves of 8³ voxels), with only the leaves that hold smoke, heat or fuel written, and values as 32-bit floats with OpenVDB’s active-mask compression. Every grid reads back bit for bit in OpenVDB 13, the version Blender 5.2 bundles, and the frames render in Blender’s Cycles.',
  ],
  [
    'Validation',
    'The fire was checked against fire-science measurements, on a uniform 0.6 m bed (the renderer plays no part). At this grid the solver resolves too little of the turbulence that mixes air into a plume (with none, the plume stays three to four times too hot), so the fire’s turbulence is a model: eddies of the fire’s own velocity, u² / l with u = 1.9 Q^(1/5) m/s (McCaffrey’s velocity scale), times one constant, calibrated on an 80 kW fire against McCaffrey’s measured centreline temperatures, as FDS calibrates its mixing constant against flame heights. With that one constant, the time-averaged centreline temperature of the 80 kW fire comes within about 20% of McCaffrey’s measurements over the intermittent flame and within about 30% over the plume above it. The size scaling is weaker than measured: the 40 kW fire runs 12 to 47% cooler than McCaffrey’s values, and the 150 kW fire up to 60% hotter in its intermittent flame (its flames reach the top of the 1.9 m domain). The flames come out taller than Heskestad’s correlation: the height below which 99% of the heat is released is 1.4 to 1.7 times his flame height, and the height where the flame (gas 500 K above the air) is present half the time about twice it. In tests, a constant strong enough to shorten them much left the plume far too cool, so the constant is set by the temperatures. The fire’s heat release puffs at 2.4 to 3.0 Hz, faster than the 1.9 Hz of Cetegen and Ahmed’s correlation for this bed. It releases 98 to 102% of the heat asked for (see Mixing).',
  ],
  [
    'Limits',
    'A real-time study rather than a production simulation: the grid is coarse (at most 200 × 80 × 40 cells over a box 4.8 m wide), velocity is stored at half precision, the pressure solve treats the gas as one density (FDS solves for a variable one), and a GPU that cannot keep up runs the simulation in slow motion. The chemistry is infinitely fast (no ignition, extinction or smouldering), soot forms at a fixed yield rather than forming and burning out within the flame, and the fire’s light reaches the floor unshadowed. Production pyro adds sparse and adaptive grids, far larger solves and collisions.',
  ],
]

// The solver's own frame of the default settings, one per frame shape (the frame is 4:3 on narrow screens).
function PyroStill({ alt }: { alt: string }) {
  return (
    <picture>
      <source media="(max-width: 768px)" srcSet="/pyro/poster-4x3.jpg" />
      <img className="lab-still-img" src="/pyro/poster.jpg" alt={alt} decoding="async" />
    </picture>
  )
}

// A labelled slider with its value.
function Slider({
  label,
  min,
  max,
  step,
  value,
  digits = 2,
  unit = '',
  onChange,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  digits?: number
  unit?: string
  onChange: (v: number) => void
}) {
  return (
    <Group label={label}>
      <Range min={min} max={max} step={step} value={value} label={unit ? `${label}, ${unit.trim()}` : label} onChange={onChange} />
      <span className="lab-readout">
        {value.toFixed(digits)}
        {unit}
      </span>
    </Group>
  )
}

// The wooden ball on the HUD: its hottest surface, and whether it is alight or how much of it has charred.
function ballLine(b: BallStatus) {
  const c = Math.round(b.surfaceK - 273.15)
  if (b.burningKW >= 0.5) return `\nBall ${c} °C  burning ${b.burningKW.toFixed(0)} kW`
  if (b.charred >= 0.01) return `\nBall ${c} °C  ${Math.round(b.charred * 100)}% charred`
  return `\nBall ${c} °C`
}

export default function PyroLab() {
  const frameRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<PyroEngine | null>(null)
  const [pyro, setPyro] = useState<PyroState>(PYRO_DEFAULTS)
  const pyroRef = useRef(pyro)
  const [status, setStatus] = useState<PyroStatus | null>(null)
  // Mounted only on the client (the Lab switch shows it), so WebGPU can be checked here.
  const [mode, setMode] = useState<'starting' | 'live' | 'fallback'>(() => (PyroEngine.detect() ? 'starting' : 'fallback'))
  const [paused, setPaused] = useState(false)
  const [tab, setTab] = useState<PyroTab>('source')
  // VDB export: frames written so far while a sequence records (null when idle).
  const [exporting, setExporting] = useState<{ done: number; of: number } | null>(null)
  const exportCancel = useRef(false)
  // What a drag on the image does: stir the gas, move the source along the floor, or move the collision ball.
  const [tool, setTool] = useState<'stir' | 'source' | 'ball'>('stir')
  // The callout inviting visitors to burn the wooden ball, until they try it or wave it away (remembered here).
  const [calloutGone, setCalloutGone] = useState(() => {
    try {
      return localStorage.getItem('pyro-wood-ball-callout') === 'done'
    } catch {
      return false
    }
  })
  const dismissCallout = () => {
    setCalloutGone(true)
    try {
      localStorage.setItem('pyro-wood-ball-callout', 'done')
    } catch {}
  }
  const tryBall = () => {
    dismissCallout()
    update({ ball: true, ballBurns: true })
    setTool('ball')
    engineRef.current?.ballBeside()
  }
  const [renderer, setRenderer] = useState('')
  const tabsId = useId()
  const debugOn = useSyncExternalStore(noSubscribe, () => new URLSearchParams(location.search).has('debug'), () => false)
  // The device check (components/pyro/DeviceCheck.tsx), for testing the lab on other devices: only with ?check.
  const checkOn = useSyncExternalStore(noSubscribe, () => new URLSearchParams(location.search).has('check'), () => false)
  const statusRef = useRef<PyroStatus | null>(null)
  const shown = mode === 'live' && !!status && status.simTime > 0

  useEffect(() => {
    pyroRef.current = pyro
  }, [pyro])
  // The engine's reports; when it steps down from the default resolution on a slow GPU, the Solver tab follows.
  const onStatus = (s: PyroStatus) => {
    statusRef.current = s
    setStatus(s)
    if (s.autoStepped) setPyro(p => (p.resolution === s.resolution ? p : { ...p, resolution: s.resolution }))
  }

  useEffect(() => {
    const canvas = canvasRef.current
    const frame = frameRef.current
    if (!canvas || !frame || !PyroEngine.detect()) return
    let engine: PyroEngine | null = null
    let unmounted = false
    let visible = false
    let losses = 0
    const fit = () => {
      const r = frame.getBoundingClientRect()
      engine?.resize(r.width, r.height, window.devicePixelRatio || 1)
    }
    const start = async () => {
      const e = await PyroEngine.create(canvas, onStatus, pyroRef.current, onLost)
      if (unmounted) {
        e?.dispose()
        return
      }
      if (!e) {
        setMode('fallback')
        return
      }
      engine = e
      engineRef.current = e
      if (process.env.NODE_ENV !== 'production') {
        ;(window as unknown as { __pyro?: PyroEngine }).__pyro = e
      }
      setRenderer(`WebGPU, ${e.adapterName}`)
      fit()
      e.setVisible(visible)
      const ok = await e.whenReady.catch(err => {
        console.error(err)
        return false
      })
      if (unmounted || engine !== e) return
      if (!ok) {
        e.dispose()
        engine = null
        engineRef.current = null
        setMode('fallback')
        return
      }
      setMode('live')
    }
    // A lost device is replaced, twice at most.
    const onLost = () => {
      engine?.dispose()
      engine = null
      engineRef.current = null
      if (++losses > 2) {
        setMode('fallback')
        return
      }
      setTimeout(() => !unmounted && start(), 1000)
    }
    const io = new IntersectionObserver(entries => {
      visible = entries.some(en => en.isIntersecting)
      engine?.setVisible(visible)
    })
    io.observe(frame)
    const ro = new ResizeObserver(fit)
    ro.observe(frame)
    start()
    return () => {
      unmounted = true
      io.disconnect()
      ro.disconnect()
      engine?.dispose()
      engineRef.current = null
    }
  }, [])

  // A new source brings the look it reads best with (see SOURCE_LOOKS).
  const saveFile = (blob: Blob, name: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 10000)
  }
  const exportVdb = async (frames: number) => {
    const engine = engineRef.current
    if (!engine || exporting) return
    const d = new Date()
    const two = (v: number) => String(v).padStart(2, '0')
    const stamp = `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`
    const name = `pyro-${pyro.sourceType}-${stamp}`
    setExporting({ done: 0, of: frames })
    exportCancel.current = false
    try {
      if (frames === 1) {
        const file = await engine.vdbFrame()
        if (file) saveFile(new Blob([file], { type: 'application/octet-stream' }), `${name}.vdb`)
      } else {
        const zip = new ZipStore()
        const made = await engine.vdbSequence(
          frames,
          (file, i) => {
            zip.add(`${name}/${name}.${String(i + 1).padStart(4, '0')}.vdb`, file)
            setExporting({ done: i + 1, of: frames })
          },
          () => exportCancel.current,
        )
        if (made) saveFile(zip.finish(), `${name}.zip`)
      }
    } finally {
      setExporting(null)
    }
  }
  const applyPreset = (state: Partial<PyroState>) => {
    update(state)
    engineRef.current?.restart()
  }
  const pickTool = (t: 'stir' | 'source' | 'ball') => {
    setTool(t)
    if (t === 'ball' && !pyro.ball) update({ ball: true })
  }
  const pickSource = (t: SourceType) => {
    if (t !== pyro.sourceType) update({ sourceType: t, ...SOURCE_LOOKS[t] })
  }
  // A map brings its typical illuminance; the rig is a studio's, so it takes the gray cyc in place of the sky.
  const pickLighting = (en: PyroState['env']) => {
    if (en === pyro.env) return
    if (en === 'none') update({ env: en })
    else if (en === 'rig') update(pyro.background === 'sky' ? { env: en, background: 'gray' } : { env: en })
    else update({ env: en, skyEV: ENV_EV[en] })
  }
  const update = (patch: Partial<PyroState>) => {
    setPyro(p => ({ ...p, ...patch }))
    if (patch.resolution && engineRef.current) engineRef.current.autoResolution = false
    engineRef.current?.setState(patch)
  }
  const togglePause = () => {
    setPaused(!paused)
    engineRef.current?.setPaused(!paused)
  }
  // The path-traced reference (paused): a wipe across the frame, the live render left of it, the traced right.
  const tracing = paused && !!status?.pathTrace.on
  const [wipe, setWipeAt] = useState(0.5)
  const moveWipe = (share: number) => {
    const w = Math.min(Math.max(share, 0), 1)
    setWipeAt(w)
    engineRef.current?.setWipe(w)
  }
  const toggleTrace = () => {
    engineRef.current?.setWipe(wipe)
    engineRef.current?.setPathTrace(!tracing)
  }
  const wipeDrag = useRef<number | null>(null)
  const wipeAt = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    moveWipe((e.clientX - r.left) / r.width)
  }

  // Stirring: a mouse or pen stirs at once; a finger must first travel sideways (a vertical swipe scrolls).
  const drag = useRef<{ id: number; x: number; y: number; t: number; live: boolean } | null>(null)
  // Moving the source or the ball: to the point under the pointer.
  const moveTo = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - r.left) / r.width) * 2 - 1
    const y = 1 - ((e.clientY - r.top) / r.height) * 2
    const engine = engineRef.current
    if (!engine) return
    if (tool === 'ball') engine.ballTo(x, y)
    else {
      const sx = engine.sourceTo(x, y)
      if (sx !== null) setPyro(p => ({ ...p, srcX: sx }))
    }
  }
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return
    const moving = tool !== 'stir'
    const live = moving || e.pointerType !== 'touch'
    drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY, t: e.timeStamp, live }
    if (live) e.currentTarget.setPointerCapture(e.pointerId)
    if (moving) moveTo(e)
  }
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current
    if (!d || d.id !== e.pointerId) return
    if (!d.live) {
      const mx = Math.abs(e.clientX - d.x)
      const my = Math.abs(e.clientY - d.y)
      if (mx < TOUCH_SLOP && my < TOUCH_SLOP) return
      if (my > mx) {
        drag.current = null
        return
      }
      d.live = true
      e.currentTarget.setPointerCapture(e.pointerId)
    }
    if (tool !== 'stir') {
      moveTo(e)
      return
    }
    const r = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - r.left) / r.width) * 2 - 1
    const y = 1 - ((e.clientY - r.top) / r.height) * 2
    engineRef.current?.stirAt(x, y, ((e.clientX - d.x) / r.width) * 2, (-(e.clientY - d.y) / r.height) * 2, e.timeStamp - d.t)
    d.x = e.clientX
    d.y = e.clientY
    d.t = e.timeStamp
  }
  const endDrag = () => {
    drag.current = null
  }
  // The arrow keys push the smoke sideways or up from the middle of the frame.
  const onKeyDown = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    const push: Record<string, [number, number]> = { ArrowLeft: [-0.05, 0], ArrowRight: [0.05, 0], ArrowUp: [0, 0.05], ArrowDown: [0, -0.05] }
    const p = push[e.key]
    if (!p) return
    e.preventDefault()
    if (tool === 'ball') engineRef.current?.nudgeBall(p[0] * 2, p[1] * 2)
    else if (tool === 'source') {
      const sx = Math.max(-2, Math.min(2, pyro.srcX + p[0] * 2))
      update({ srcX: sx })
    } else engineRef.current?.stirAt(0, 0, p[0], p[1], 33)
  }

  const dims = status?.dims ?? RESOLUTIONS[pyro.resolution].dims
  const inputOn = mode === 'live'

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em', marginBottom: 12 }}>Fire and smoke, simulated live</h2>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, maxWidth: 640 }}>
          A 3D fire and smoke solver running on your GPU, built the way production FX tools are: combustion, buoyancy,
          vorticity, turbulence and a pressure solve every step, lit with physically based volume rendering.
        </p>
      </div>

      <p className="lab-howto">
        {tool === 'stir' ? (
          <>
            <span className="lab-howto-pointer">Drag across the image</span>
            <span className="lab-howto-touch">Swipe sideways on the image</span> to stir the {pyro.sourceType === 'fire' ? 'fire' : 'smoke'}
          </>
        ) : tool === 'source' ? (
          <>Drag along the floor to move the {pyro.sourceType === 'fire' ? 'fire' : 'smoke'}</>
        ) : pyro.ballBurns && pyro.sourceType === 'fire' ? (
          <>Drag the ball into the flames and hold it there</>
        ) : (
          <>Drag the ball through the {pyro.sourceType === 'fire' ? 'flames' : 'smoke'}</>
        )}
        <span aria-hidden="true"> · </span>
        <span className="sr-only">. </span>Every control below is live
      </p>

      <div ref={frameRef} className="lab-frame">
        {mode === 'fallback' ? (
          <PyroStill alt="A fire burning on the floor beside a colour chart, a grey ball and a chrome ball, its flames yellow at the base and red at the tips, its smoke rising, rendered by the pyro lab's solver" />
        ) : (
          <canvas
            ref={canvasRef}
            tabIndex={0}
            role="application"
            aria-roledescription="fire and smoke simulation"
            aria-label={
              tool === 'stir'
                ? 'Live fire and smoke simulation. Drag, or use the arrow keys, to stir it.'
                : tool === 'source'
                  ? 'Live fire and smoke simulation. Drag, or use the left and right arrow keys, to move the source.'
                  : 'Live fire and smoke simulation. Drag, or use the arrow keys, to move the ball.'
            }
            className={`lab-canvas${tool === 'stir' ? '' : ' lab-canvas-move'}`}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onKeyDown={onKeyDown}
          />
        )}
        {mode !== 'fallback' && (
          <div className={`lab-still${shown ? ' gone' : ''}`}>
            <PyroStill alt="" />
          </div>
        )}
        <div className="lab-hud lab-hud-left">
          {mode === 'fallback'
            ? 'Still frame. The live solver needs WebGPU, which this browser does not offer.'
            : mode === 'starting'
              ? 'Starting the solver'
              : `${pyro.sourceType === 'fire' ? 'Fire' : 'Smoke'}  ${dims.join(' × ')}`}
          {mode === 'live' && status?.ball && ballLine(status.ball)}
        </div>
        {mode === 'live' && status && (
          <div className="lab-hud lab-hud-right">
            {paused ? `Paused  ${status.simTime.toFixed(1)} s` : `${status.simTime.toFixed(1)} s  ${status.stepsPerSec} steps/s`}
          </div>
        )}
        {mode === 'live' && shown && !calloutGone && !tracing && pyro.sourceType === 'fire' && pyro.ballBurns && (
          <div className="lab-callout" role="note">
            <p>
              <b>New</b>The ball is wood. Hold it in the flames and it scorches, chars and catches fire.
            </p>
            <div className="lab-callout-actions">
              <Pill on={false} onClick={tryBall}>
                Try it
              </Pill>
              <Pill on={false} onClick={dismissCallout}>
                Not now
              </Pill>
            </div>
          </div>
        )}
        {mode === 'live' && tracing && status && (
          <div
            className="lab-wipe"
            onPointerDown={e => {
              wipeDrag.current = e.pointerId
              e.currentTarget.setPointerCapture(e.pointerId)
              wipeAt(e)
            }}
            onPointerMove={e => {
              if (wipeDrag.current === e.pointerId) wipeAt(e)
            }}
            onPointerUp={() => (wipeDrag.current = null)}
            onPointerCancel={() => (wipeDrag.current = null)}
          >
            <div
              className="lab-wipe-line"
              style={{ left: `${wipe * 100}%` }}
              role="slider"
              tabIndex={0}
              aria-label="Wipe between the live render and the path-traced reference"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(wipe * 100)}
              onKeyDown={e => {
                const step = e.key === 'ArrowLeft' ? -0.05 : e.key === 'ArrowRight' ? 0.05 : 0
                if (!step) return
                e.preventDefault()
                moveWipe(wipe + step)
              }}
            />
            <div className="lab-hud lab-wipe-tag lab-wipe-live">Live</div>
            <div className="lab-hud lab-wipe-tag lab-wipe-traced">
              {status.pathTrace.compiling ? 'Path traced, compiling' : `Path traced, ${status.pathTrace.samples} ${status.pathTrace.samples === 1 ? 'path' : 'paths'} a pixel`}
            </div>
          </div>
        )}
        {debugOn && (
          <div className="lab-hud lab-hud-bottom">
            {`Renderer: ${renderer || (mode === 'fallback' ? 'none (no WebGPU)' : 'starting')}`}
            {status && `\nRender scale ${status.renderScale.toFixed(2)}`}
          </div>
        )}
      </div>

      {checkOn && mode !== 'starting' && (
        <DeviceCheck
          engineRef={engineRef}
          statusRef={statusRef}
          frameRef={frameRef}
          applyPreset={applyPreset}
          ensureRunning={() => {
            setPaused(false)
            engineRef.current?.setPaused(false)
          }}
          fallback={mode === 'fallback'}
        />
      )}

      {mode !== 'fallback' && (
        <fieldset className="lab-controls" disabled={!inputOn}>
          <div className="lab-row">
            <Pill on={paused} onClick={togglePause}>
              {paused ? 'Resume' : 'Pause'}
            </Pill>
            <Pill on={false} onClick={() => engineRef.current?.restart()}>
              Restart
            </Pill>
            <Pill
              on={false}
              onClick={() => {
                setPyro(PYRO_DEFAULTS)
                engineRef.current?.setState(PYRO_DEFAULTS)
                setTool('stir')
              }}
            >
              Reset controls
            </Pill>
            {paused && (
              <Pill on={tracing} onClick={toggleTrace}>
                Path trace
              </Pill>
            )}
          </div>
          <div className="lab-row">
            <Group label="Drag to">
              <Pill on={tool === 'stir'} onClick={() => pickTool('stir')}>
                Stir
              </Pill>
              <Pill on={tool === 'source'} onClick={() => pickTool('source')}>
                {pyro.sourceType === 'fire' ? 'Move the fire' : 'Move the smoke'}
              </Pill>
              <Pill on={tool === 'ball'} onClick={() => pickTool('ball')}>
                Move the ball
              </Pill>
            </Group>
            {pyro.ball && (
              <Pill
                on={false}
                onClick={() => {
                  update({ ball: false })
                  if (tool === 'ball') setTool('stir')
                }}
              >
                Remove the ball
              </Pill>
            )}
          </div>
          <Tabs id={tabsId} label="Pyro controls" tabs={PYRO_TABS} tab={tab} onTab={setTab} />
          <div className="lab-panel" role="tabpanel" id={`${tabsId}-panel`} aria-labelledby={`${tabsId}-${tab}`}>
            {tab === 'source' && (
              <>
                <div className="lab-row">
                  <Group label="Start from">
                    {PRESETS.map(p => (
                      <Pill key={p.id} on={false} onClick={() => applyPreset(p.state)}>
                        {p.label}
                      </Pill>
                    ))}
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Type">
                    <Pill on={pyro.sourceType === 'smoke'} onClick={() => pickSource('smoke')}>
                      Smoke
                    </Pill>
                    <Pill on={pyro.sourceType === 'fire'} onClick={() => pickSource('fire')}>
                      Fire
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Slider label="Radius" min={0.08} max={0.4} step={0.01} value={pyro.srcRadius} unit=" m" onChange={v => update({ srcRadius: v })} />
                  {pyro.sourceType === 'fire' ? (
                    <Slider label="Heat release" min={10} max={400} step={5} value={pyro.heatRelease} digits={0} unit=" kW" onChange={v => update({ heatRelease: v })} />
                  ) : (
                    <Slider label="Density" min={0.2} max={2} step={0.05} value={pyro.srcDensity} onChange={v => update({ srcDensity: v })} />
                  )}
                </div>
                {pyro.sourceType === 'smoke' && (
                  <div className="lab-row">
                    <Slider label="Temperature" min={10} max={300} step={5} value={pyro.srcTemp} digits={0} unit=" K" onChange={v => update({ srcTemp: v })} />
                    <Slider label="Speed" min={0} max={2} step={0.05} value={pyro.srcSpeed} unit=" m/s" onChange={v => update({ srcSpeed: v })} />
                  </div>
                )}
                <div className="lab-row">
                  <Slider label="Noise" min={0} max={1} step={0.05} value={pyro.srcNoise} onChange={v => update({ srcNoise: v })} />
                  <Slider label="Position" min={-2} max={2} step={0.05} value={pyro.srcX} unit=" m" onChange={v => update({ srcX: v })} />
                </div>
                {pyro.ball && (
                  <div className="lab-row">
                    <Slider label="Ball radius" min={0.1} max={0.4} step={0.01} value={pyro.ballRadius} unit=" m" onChange={v => update({ ballRadius: v })} />
                    <Group label="Burning ball">
                      <Pill on={pyro.ballBurns} onClick={() => update({ ballBurns: true })}>
                        On
                      </Pill>
                      <Pill on={!pyro.ballBurns} onClick={() => update({ ballBurns: false })}>
                        Off
                      </Pill>
                    </Group>
                    {pyro.ballBurns && (
                      <Pill on={false} onClick={() => engineRef.current?.newBall()}>
                        Fresh ball
                      </Pill>
                    )}
                  </div>
                )}
              </>
            )}
            {tab === 'forces' && (
              <>
                <div className="lab-row">
                  <Slider label="Buoyancy" min={0} max={2} step={0.05} value={pyro.buoyancy} onChange={v => update({ buoyancy: v })} />
                  <Slider label="Vorticity" min={0} max={1} step={0.01} value={pyro.vorticity} onChange={v => update({ vorticity: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Turbulence" min={0} max={3} step={0.05} value={pyro.turbulence} onChange={v => update({ turbulence: v })} />
                  <Slider label="Turbulence scale" min={0.1} max={1} step={0.01} value={pyro.turbScale} unit=" m" onChange={v => update({ turbScale: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Wind" min={-2} max={2} step={0.05} value={pyro.wind} unit=" m/s" onChange={v => update({ wind: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Cooling" min={0} max={2} step={0.05} value={pyro.cooling} unit=" /s" onChange={v => update({ cooling: v })} />
                  <Slider label="Dissipation" min={0} max={0.5} step={0.01} value={pyro.dissipation} unit=" /s" onChange={v => update({ dissipation: v })} />
                </div>
                <p className="lab-note">
                  {pyro.sourceType === 'fire'
                    ? 'Buoyancy 1 is physical, Turbulence 1 the calibrated turbulence model, Cooling and Dissipation 0 physical.'
                    : 'Turbulence in m/s². Buoyancy 1 is physical.'}
                </p>
              </>
            )}
            {tab === 'fire' && (
              <>
                <div className="lab-row">
                  <Slider label="Soot yield" min={0} max={0.2} step={0.001} value={pyro.sootYield} digits={3} unit=" kg/kg" onChange={v => update({ sootYield: v })} />
                  <Slider label="Radiant fraction" min={0.1} max={0.5} step={0.01} value={pyro.radiantFraction} onChange={v => update({ radiantFraction: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Expansion" min={0} max={1.5} step={0.05} value={pyro.expansion} onChange={v => update({ expansion: v })} />
                  <Slider label="Flame detail" min={0} max={2} step={0.05} value={pyro.flameDetail} onChange={v => update({ flameDetail: v })} />
                </div>
                <p className="lab-note">
                  {pyro.sourceType === 'fire'
                    ? 'Wood burns with a soot yield of 0.015 and radiates 0.3 of its heat. Expansion and buoyancy 1 are physical. Flame detail 0 shows the flame at the grid\u2019s resolution.'
                    : 'These act on a fire source: choose Fire under Source.'}
                </p>
              </>
            )}
            {tab === 'shading' && (
              <>
                <div className="lab-row">
                  {pyro.sourceType === 'smoke' && (
                    <Slider label="Smoke density" min={2} max={60} step={1} value={pyro.densityScale} digits={0} unit=" /m" onChange={v => update({ densityScale: v })} />
                  )}
                  <Slider label="Albedo" min={0.1} max={1} step={0.01} value={pyro.albedo} onChange={v => update({ albedo: v })} />
                  <Slider label="Anisotropy" min={-0.5} max={0.9} step={0.01} value={pyro.anisotropy} onChange={v => update({ anisotropy: v })} />
                </div>
                <div className="lab-row">
                  <Group label="Environment">
                    {LIGHTINGS.map(en => (
                      <Pill key={en} on={pyro.env === en} onClick={() => pickLighting(en)}>
                        {en === 'none' ? 'None' : en === 'rig' ? 'Softbox rig' : ENVIRONMENTS[en].label}
                      </Pill>
                    ))}
                  </Group>
                </div>
                {pyro.env === 'rig' ? (
                  <>
                    <div className="lab-row">
                      <Group label="Lights">
                        <Pill on={pyro.rigKey} onClick={() => update({ rigKey: !pyro.rigKey })}>
                          Key
                        </Pill>
                        <Pill on={pyro.rigFill} onClick={() => update({ rigFill: !pyro.rigFill })}>
                          Fill
                        </Pill>
                        <Pill on={pyro.rigRim} onClick={() => update({ rigRim: !pyro.rigRim })}>
                          Rim
                        </Pill>
                      </Group>
                    </div>
                    <div className="lab-row">
                      <Slider label="Light azimuth" min={-180} max={180} step={1} value={pyro.lightAz} digits={0} unit={'°'} onChange={v => update({ lightAz: v })} />
                      <Slider label="Light elevation" min={5} max={85} step={1} value={pyro.lightEl} digits={0} unit={'°'} onChange={v => update({ lightEl: v })} />
                      <Slider label="Key" min={-4} max={16} step={0.5} value={pyro.sunEV} digits={1} unit=" EV" onChange={v => update({ sunEV: v })} />
                    </div>
                  </>
                ) : pyro.env === 'none' ? (
                  <>
                    <div className="lab-row">
                      <Slider label="Light azimuth" min={-180} max={180} step={1} value={pyro.lightAz} digits={0} unit={'°'} onChange={v => update({ lightAz: v })} />
                      <Slider label="Light elevation" min={5} max={85} step={1} value={pyro.lightEl} digits={0} unit={'°'} onChange={v => update({ lightEl: v })} />
                    </div>
                    <div className="lab-row">
                      <Slider label="Sun" min={-4} max={16} step={0.5} value={pyro.sunEV} digits={1} unit=" EV" onChange={v => update({ sunEV: v })} />
                      <Slider label="Sky" min={-4} max={14} step={0.5} value={pyro.skyEV} digits={1} unit=" EV" onChange={v => update({ skyEV: v })} />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="lab-row">
                      <Slider label="Turn" min={-180} max={180} step={1} value={pyro.lightAz} digits={0} unit={'°'} onChange={v => update({ lightAz: v })} />
                      <Slider label="Environment" min={-4} max={17} step={0.5} value={pyro.skyEV} digits={1} unit=" EV" onChange={v => update({ skyEV: v })} />
                    </div>
                    <p className="lab-credit">
                      <span className="lab-label">Environment</span>
                      <span>
                        {ENVIRONMENTS[pyro.env].title} by {ENVIRONMENTS[pyro.env].authors}, Poly Haven.{' '}
                        <a href={ENVIRONMENTS[pyro.env].url} target="_blank" rel="noopener noreferrer">
                          Source
                        </a>
                        {' · '}
                        <a href="https://creativecommons.org/publicdomain/zero/1.0/" target="_blank" rel="noopener noreferrer">
                          CC0
                        </a>
                      </span>
                    </p>
                  </>
                )}
                <div className="lab-row">
                  <Group label="Exposure">
                    <Pill on={pyro.exposureMode === 'auto'} onClick={() => update({ exposureMode: 'auto' })}>
                      Auto
                    </Pill>
                    <Pill on={pyro.exposureMode === 'manual'} onClick={() => update({ exposureMode: 'manual', exposure: Math.round((status?.exposureEV ?? pyro.exposure) * 4) / 4 })}>
                      Manual
                    </Pill>
                  </Group>
                  {pyro.exposureMode === 'auto' ? (
                    <Slider label="Compensation" min={-3} max={3} step={0.25} value={pyro.exposureComp} digits={2} unit=" EV" onChange={v => update({ exposureComp: v })} />
                  ) : (
                    <Slider label="Exposure" min={-2} max={16} step={0.25} value={pyro.exposure} digits={2} unit=" EV" onChange={v => update({ exposure: v })} />
                  )}
                  <Slider label="White balance" min={2000} max={10000} step={100} value={pyro.whiteBalance} digits={0} unit=" K" onChange={v => update({ whiteBalance: v })} />
                </div>
                <div className="lab-row">
                  <Slider label="Shutter" min={0} max={360} step={15} value={pyro.shutter} digits={0} unit={'°'} onChange={v => update({ shutter: v })} />
                  <Slider label="Glare" min={0} max={10} step={0.5} value={pyro.glare * 100} digits={1} unit="%" onChange={v => update({ glare: v / 100 })} />
                </div>
                <div className="lab-row">
                  <Group label="View">
                    {VIEW_TRANSFORMS.map(v => (
                      <Pill key={v.id} on={pyro.view === v.id} onClick={() => update({ view: v.id })}>
                        {v.label}
                      </Pill>
                    ))}
                  </Group>
                  <Group label="On-set reference">
                    <Pill on={!pyro.reference} onClick={() => update({ reference: false })}>
                      Off
                    </Pill>
                    <Pill on={pyro.reference} onClick={() => update({ reference: true })}>
                      On
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Background">
                    {(pyro.env === 'none' || pyro.env === 'rig') && (
                      <Pill on={pyro.background === 'studio'} onClick={() => update({ background: 'studio' })}>
                        Studio
                      </Pill>
                    )}
                    {pyro.env !== 'rig' && (
                      <Pill on={pyro.background === 'sky' || (pyro.env !== 'none' && pyro.background === 'studio')} onClick={() => update({ background: 'sky' })}>
                        Sky
                      </Pill>
                    )}
                    <Pill on={pyro.background === 'charcoal'} onClick={() => update({ background: 'charcoal' })}>
                      Charcoal cyc
                    </Pill>
                    <Pill on={pyro.background === 'gray'} onClick={() => update({ background: 'gray' })}>
                      Gray cyc
                    </Pill>
                  </Group>
                  <Group label="Sampling">
                    <Pill on={!pyro.tricubic} onClick={() => update({ tricubic: false })}>
                      Trilinear
                    </Pill>
                    <Pill on={pyro.tricubic} onClick={() => update({ tricubic: true })}>
                      Tricubic
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Slider label="Detail" min={0} max={1.5} step={0.05} value={pyro.detail} onChange={v => update({ detail: v })} />
                  <Slider label="Detail scale" min={0.02} max={0.2} step={0.005} value={pyro.detailScale} digits={3} unit=" m" onChange={v => update({ detailScale: v })} />
                </div>
                <p className="lab-note">
                  Light as a meter reads it: the sun and sky are incident readings (2.5 × 2^EV lux), the exposure is the camera’s EV at ISO 100.
                  Pause the simulation to relight a frozen frame: shading changes redraw at once.
                </p>
              </>
            )}
            {tab === 'solver' && (
              <>
                <div className="lab-row">
                  <Group label="Resolution">
                    {RESOLUTION_ORDER.map(r => (
                      <Pill key={r} on={pyro.resolution === r} onClick={() => update({ resolution: r })}>
                        {`${RESOLUTIONS[r].label} ${RESOLUTIONS[r].dims.join('×')}`}
                      </Pill>
                    ))}
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Velocity">
                    <Pill on={pyro.velocity === 'mac'} onClick={() => update({ velocity: 'mac' })}>
                      Modified MacCormack
                    </Pill>
                    <Pill on={pyro.velocity === 'ar'} onClick={() => update({ velocity: 'ar' })}>
                      Advection-reflection
                    </Pill>
                    <Pill on={pyro.velocity === 'sl'} onClick={() => update({ velocity: 'sl' })}>
                      Semi-Lagrangian
                    </Pill>
                    <Pill on={pyro.velocity === 'lfm'} onClick={() => update({ velocity: 'lfm' })}>
                      Leapfrog flow maps, for smoke
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Lost vorticity">
                    <Pill on={pyro.ivock && pyro.velocity !== 'lfm'} onClick={() => update({ ivock: true })}>
                      IVOCK, for smoke
                    </Pill>
                    <Pill on={!pyro.ivock && pyro.velocity !== 'lfm'} onClick={() => update({ ivock: false })}>
                      Off
                    </Pill>
                  </Group>
                </div>
                {pyro.velocity === 'lfm' && <p className="lab-note">Flow maps keep the swirl themselves, so IVOCK stands aside while they run.</p>}
                <div className="lab-row">
                  <Group label="Scalars">
                    <Pill on={pyro.advection === 'bimocq'} onClick={() => update({ advection: 'bimocq' })}>
                      BiMocq², for smoke
                    </Pill>
                    <Pill on={pyro.advection === 'mac'} onClick={() => update({ advection: 'mac' })}>
                      Modified MacCormack
                    </Pill>
                    <Pill on={pyro.advection === 'sl'} onClick={() => update({ advection: 'sl' })}>
                      Semi-Lagrangian
                    </Pill>
                  </Group>
                </div>
                <div className="lab-row">
                  <Group label="Smoke upres">
                    <Pill on={!pyro.upres} onClick={() => update({ upres: false })}>
                      Off
                    </Pill>
                    <Pill on={pyro.upres} onClick={() => update({ upres: true })}>
                      Wavelet turbulence, 2×
                    </Pill>
                  </Group>
                  {pyro.upres && (
                    <Slider label="Upres turbulence" min={0} max={2} step={0.05} value={pyro.upresStrength} onChange={v => update({ upresStrength: v })} />
                  )}
                </div>
                <div className="lab-row">
                  <Group label="Pressure">
                    <Pill on={pyro.pressure === 'mg'} onClick={() => update({ pressure: 'mg' })}>
                      Multigrid
                    </Pill>
                    <Pill on={pyro.pressure === 'sor'} onClick={() => update({ pressure: 'sor' })}>
                      Red-black SOR
                    </Pill>
                  </Group>
                  {pyro.pressure === 'mg' ? (
                    <Slider label="V-cycles" min={1} max={4} step={1} value={pyro.vcycles} digits={0} onChange={v => update({ vcycles: v })} />
                  ) : (
                    <Slider label="Iterations" min={5} max={120} step={5} value={pyro.pressureIters} digits={0} onChange={v => update({ pressureIters: v })} />
                  )}
                </div>
                <p className="lab-note">A new resolution restarts the simulation.</p>
                <div className="lab-row">
                  <Group label="Export to VDB">
                    <Pill on={false} onClick={() => void exportVdb(1)}>
                      This frame
                    </Pill>
                    <Pill on={false} onClick={() => void exportVdb(60)}>
                      Next 2 seconds
                    </Pill>
                    {exporting && exporting.of > 1 && (
                      <Pill on={false} onClick={() => (exportCancel.current = true)}>
                        {`Stop (${exporting.done} of ${exporting.of})`}
                      </Pill>
                    )}
                  </Group>
                </div>
                <p className="lab-note">
                  OpenVDB, y up, in metres: density (extinction, 1/m), temperature (K) and vel (m/s), and for a fire its unburned fuel and heat release. Two seconds is 60 frames in a zip, 2 to 5 MB each. In Blender, turn the volume 90° about x, and set Principled Volume’s Density and Temperature to 1 so the grids read in their own units.
                </p>
              </>
            )}
          </div>
        </fieldset>
      )}

      <details className="lab-tech">
        <summary>Technical notes and references</summary>
        <div className="lab-tech-body">
          <dl className="lab-tech-notes">
            {TECH_NOTES.map(([term, text]) => (
              <div key={term}>
                <dt>{term}</dt>
                <dd>{text}</dd>
              </div>
            ))}
          </dl>
          <div className="lab-refs">
            {PYRO_CITATION_GROUPS.map(g => (
              <div key={g} className="lab-refs-group">
                <h3>{g}</h3>
                <ol>
                  {PYRO_CITATIONS.filter(c => c.group === g).map(c => (
                    <li key={c.id}>
                      {citeAuthors(c)} ({c.year}).{' '}
                      <a href={c.url} target="_blank" rel="noopener noreferrer">
                        {c.title}
                      </a>
                      . <span className="lab-refs-venue">{c.venue}.</span>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
        </div>
      </details>
    </div>
  )
}
