// One source for how the site names Matthew, so the page title, search description, share card and structured
// data can never drift apart. Both practices are named everywhere, VFX first, joined by "and".
export const SITE = 'https://mattebell.xyz'
export const NAME = 'Matthew Bell'
export const TITLE = 'Matthew Bell | VFX Artist & Creative Technologist'

export const description = (years: number) =>
  `VFX artist and creative technologist: lighting and look development for film, TV and real-time production, and generative AI and real-time R&D. ${years} years in 3D and VFX, based in San Antonio.`

export const SAME_AS = [
  'https://www.imdb.com/name/nm2998873/',
  'https://www.linkedin.com/in/mattebell',
  'https://vimeo.com/user6348780',
]

// Shared Open Graph fields. No url here: each page sets its own.
export const OG_SHARED = {
  siteName: NAME,
  locale: 'en_US',
  type: 'profile' as const,
  firstName: 'Matthew',
  lastName: 'Bell',
}
