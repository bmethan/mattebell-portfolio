// One source for how the site names Matthew, so the page title, search description, share card and structured
// data can never drift apart. Both practices are named everywhere, VFX first, joined by "and".
export const SITE = 'https://mattebell.xyz'
export const NAME = 'Matthew Bell'
export const TITLE = 'Matthew Bell | VFX Artist & Creative Technologist'

// His strongest fact leads (it is what VFX producers search), and both practices stay in the first sentence.
export const description = (years: number) =>
  `VFX artist and creative technologist: lighting and lookdev lead on Ready Player One, Iron Man 3, Real Steel and Star Trek, and generative AI and real-time R&D. ${years} years in 3D, based in San Antonio.`

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
