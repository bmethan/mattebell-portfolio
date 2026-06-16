import Nav from '@/components/Nav'
import Hero from '@/components/Hero'
import Reel from '@/components/Reel'
import Work from '@/components/Work'
import Skills from '@/components/Skills'
import About from '@/components/About'
import Testimonials from '@/components/Testimonials'
import Contact from '@/components/Contact'
import Footer from '@/components/Footer'
import { getWorkCards, getTestimonials, getSettings } from '@/sanity/lib/client'

export const revalidate = 60

export default async function Home() {
  const [workCards, testimonials, settings] = await Promise.all([
    getWorkCards(),
    getTestimonials(),
    getSettings(),
  ])

  return (
    <main>
      <Nav available={settings?.available ?? true} />
      <Hero />
      <Reel reelUrl={settings?.reelUrl} />
      <Work cards={workCards} />
      <Skills />
      <About settings={settings} />
      <Testimonials items={testimonials} />
      <Contact />
      <Footer />
    </main>
  )
}
