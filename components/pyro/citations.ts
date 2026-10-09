// References for the pyro lab's technical notes, grouped by topic. Every entry was checked against its primary
// source (the publisher's DOI record, the authors' or organisation's own pages, or the official repository) before
// it went on the page.
export const PYRO_CITATION_GROUPS = ['Solver', 'Rendering', 'Display'] as const
export type PyroCitationGroup = (typeof PYRO_CITATION_GROUPS)[number]

export interface PyroCitation {
  id: string
  group: PyroCitationGroup
  authors: string
  // Short author form for the list (else derived: one or two surnames, or the first and "et al.").
  by?: string
  title: string
  venue: string
  year: string
  url: string
}

export const PYRO_CITATIONS: PyroCitation[] = [
  {
    id: 'mac',
    group: 'Solver',
    authors: 'Francis H. Harlow, J. Eddie Welch',
    title: 'Numerical Calculation of Time-Dependent Viscous Incompressible Flow of Fluid with Free Surface',
    venue: 'The Physics of Fluids 8(12), the marker-and-cell method and its staggered grid',
    year: '1965',
    url: 'https://doi.org/10.1063/1.1761178',
  },
  {
    id: 'stable-fluids',
    group: 'Solver',
    authors: 'Jos Stam',
    title: 'Stable Fluids',
    venue: 'Proceedings of SIGGRAPH 99, semi-Lagrangian advection, pressure projection, and texture coordinates advected for detail in smoke',
    year: '1999',
    url: 'https://doi.org/10.1145/311535.311548',
  },
  {
    id: 'smoke',
    group: 'Solver',
    authors: 'Ronald Fedkiw, Jos Stam, Henrik Wann Jensen',
    title: 'Visual Simulation of Smoke',
    venue: 'Proceedings of SIGGRAPH 2001, vorticity confinement',
    year: '2001',
    url: 'https://doi.org/10.1145/383259.383260',
  },
  {
    id: 'maccormack',
    group: 'Solver',
    authors: 'Andrew Selle, Ronald Fedkiw, ByungMoon Kim, Yingjie Liu, Jarek Rossignac',
    title: 'An Unconditionally Stable MacCormack Method',
    venue: 'Journal of Scientific Computing 35(2-3)',
    year: '2008',
    url: 'https://doi.org/10.1007/s10915-007-9166-4',
  },
  {
    id: 'curl-noise',
    group: 'Solver',
    authors: 'Robert Bridson, Jim Hourihan, Marcus Nordenstam',
    title: 'Curl-Noise for Procedural Fluid Flow',
    venue: 'ACM Transactions on Graphics 26(3) (SIGGRAPH 2007)',
    year: '2007',
    url: 'https://doi.org/10.1145/1276377.1276435',
  },
  {
    id: 'sor',
    group: 'Solver',
    authors: 'David Young',
    title: 'Iterative Methods for Solving Partial Difference Equations of Elliptic Type',
    venue: 'Transactions of the American Mathematical Society 76(1), successive over-relaxation',
    year: '1954',
    url: 'https://doi.org/10.1090/S0002-9947-1954-0059635-7',
  },
  {
    id: 'multigrid-tutorial',
    group: 'Solver',
    authors: 'William L. Briggs, Van Emden Henson, Steve F. McCormick',
    title: 'A Multigrid Tutorial, second edition',
    venue: 'SIAM, smoothing, restriction, interpolation and V-cycles',
    year: '2000',
    url: 'https://doi.org/10.1137/1.9780898719505',
  },
  {
    id: 'mgpcg',
    group: 'Solver',
    authors: 'Aleka McAdams, Eftychios Sifakis, Joseph Teran',
    title: 'A Parallel Multigrid Poisson Solver for Fluids Simulation on Large Grids',
    venue: 'Symposium on Computer Animation 2010, multigrid for the pressure equation in fluid simulation',
    year: '2010',
    url: 'https://doi.org/10.2312/SCA/SCA10/065-073',
  },
  {
    id: 'bridson-book',
    group: 'Solver',
    authors: 'Robert Bridson',
    title: 'Fluid Simulation for Computer Graphics, second edition',
    venue: 'A K Peters / CRC Press',
    year: '2015',
    url: 'https://doi.org/10.1201/9781315266008',
  },
  {
    id: 'hg',
    group: 'Rendering',
    authors: 'L. G. Henyey, J. L. Greenstein',
    title: 'Diffuse Radiation in the Galaxy',
    venue: 'The Astrophysical Journal 93, the Henyey-Greenstein phase function',
    year: '1941',
    url: 'https://doi.org/10.1086/144246',
  },
  {
    id: 'oz',
    group: 'Rendering',
    authors: 'Magnus Wrenninge, Chris Kulla, Viktor Lundqvist',
    title: 'Oz: The Great and Volumetric',
    venue: 'ACM SIGGRAPH 2013 Talks, the multiple-scattering octaves',
    year: '2013',
    url: 'https://doi.org/10.1145/2504459.2504518',
  },
  {
    id: 'moving-textures',
    group: 'Rendering',
    authors: 'Nelson Max, Barry Becker',
    title: 'Flow Visualization Using Moving Textures',
    venue: 'ICASE/LaRC Symposium on Visualizing Time-Varying Data (1995), NASA Conference Publication 3321: texture coordinates advected and restarted in two cross-faded layers',
    year: '1996',
    url: 'https://ntrs.nasa.gov/citations/19960015577',
  },
  {
    id: 'advected-textures',
    group: 'Rendering',
    authors: 'Fabrice Neyret',
    title: 'Advected Textures',
    venue: 'Symposium on Computer Animation 2003, regenerated layers cross-faded to limit distortion',
    year: '2003',
    url: 'https://doi.org/10.2312/SCA03/147-153',
  },
  {
    id: 'frostbite',
    group: 'Rendering',
    authors: 'Sébastien Hillaire',
    title: 'Physically Based and Unified Volumetric Rendering in Frostbite',
    venue: 'SIGGRAPH 2015 course, Advances in Real-Time Rendering in Games: the integration of each step',
    year: '2015',
    url: 'https://advances.realtimerendering.com/s2015/index.html',
  },
  {
    id: 'aces-fit',
    group: 'Display',
    authors: 'Stephen Hill',
    title: 'ACES filmic tone mapping fit (ACES.hlsl)',
    venue: 'in MJP and David Neubelt’s BakingLab, MIT license',
    year: '2016',
    url: 'https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl',
  },
]
