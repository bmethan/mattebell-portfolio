import type { Metadata } from 'next'
import Nav from '@/components/Nav'
import Hero from '@/components/Hero'
import Reel from '@/components/Reel'
import Services from '@/components/Services'
import Work from '@/components/Work'
import Skills from '@/components/Skills'
import About from '@/components/About'
import Testimonials from '@/components/Testimonials'
import Contact from '@/components/Contact'
import Footer from '@/components/Footer'
import { getWorkCards, getTestimonials, getSettings } from '@/sanity/lib/client'
import { getYearsExperience } from '@/lib/yearsExperience'
import { SITE, NAME, TITLE, description, SAME_AS, OG_SHARED } from '@/lib/site'

export const revalidate = 60

// Regenerated with the page, so the year count in the description rolls over each June 8 without a deploy.
export async function generateMetadata(): Promise<Metadata> {
  const desc = description(getYearsExperience())
  return {
    description: desc,
    alternates: { canonical: '/' },
    openGraph: { ...OG_SHARED, title: TITLE, description: desc, url: '/' },
  }
}

// Structured data: who the site is about, and the profiles that are the same person (IMDb credits him as
// Matthew E. Bell). It states nothing that is not visible on the page, and deliberately no awards.
function jsonLd(years: number) {
  return {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'WebSite',
        '@id': `${SITE}/#website`,
        url: `${SITE}/`,
        name: NAME,
        inLanguage: 'en-US',
        publisher: { '@id': `${SITE}/#person` },
      },
      {
        '@type': 'ProfilePage',
        '@id': `${SITE}/#webpage`,
        url: `${SITE}/`,
        name: TITLE,
        isPartOf: { '@id': `${SITE}/#website` },
        mainEntity: { '@id': `${SITE}/#person` },
      },
      {
        '@type': 'Person',
        '@id': `${SITE}/#person`,
        name: NAME,
        alternateName: ['Matt Bell', 'Matthew E. Bell'],
        url: `${SITE}/`,
        image: `${SITE}/images/matthew-bell.webp`,
        jobTitle: ['VFX Artist, Lighting and Look Development', 'Creative Technologist'],
        description: description(years),
        knowsAbout: [
          'Lighting',
          'Look development',
          'Houdini',
          'Nuke',
          'Compositing',
          'Unreal Engine',
          'Virtual production',
          'Real-time rendering',
          'Generative AI',
          'ComfyUI',
        ],
        address: { '@type': 'PostalAddress', addressLocality: 'San Antonio', addressRegion: 'TX', addressCountry: 'US' },
        sameAs: SAME_AS,
      },
    ],
  }
}

export default async function Home() {
  const [workCards, testimonials, settings] = await Promise.all([
    getWorkCards(),
    getTestimonials(),
    getSettings(),
  ])

  const years = getYearsExperience()

  return (
    <main>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd(years)).replace(/</g, '\\u003c') }}
      />
      <Nav available={settings?.available ?? true} />
      <Hero years={years} />
      <Reel reelUrl={settings?.reelUrl} />
      <Services />
      <Work cards={workCards} />
      <Skills />
      <About settings={settings} years={years} />
      <Testimonials items={testimonials} />
      <Contact />
      <Footer />
    </main>
  )
}
