'use client'
import { useEffect, useState } from 'react'

// The award stats follow the Oscar stat's pattern: the VES and the Sports Emmy went to productions he worked on
// (Walmart "Famous Visitors", The Mill; NFL Network "Run", Motion Theory, as lighter), not to him personally.
// The four Oscar productions are Best Visual Effects nominees he was lighting and lookdev lead on: Star Trek
// (82nd), Real Steel (84th), Iron Man 3 (86th) and Ready Player One (91st).
const stats = (years: number) => [
  { num: `${years}+`, label: 'Years experience' },
  { num: '20+', label: 'Feature film credits' },
  { num: '4×', label: 'Oscar-nominated productions' },
  { num: '1×', label: 'VES Award-winning production' },
  { num: '1×', label: 'Sports Emmy-winning production' },
]

// His cycling headline, in his words. The server render shows the first pair, so search engines read a
// complete heading; reduced-motion visitors keep that pair.
const phrases = (years: number) => [
  { line1: 'Lighting. Look Development.', line2: 'Full-pipeline expertise.' },
  { line1: 'Real-time & Unreal Engine.', line2: 'Virtual production.' },
  { line1: `${years} years. Feature film.`, line2: 'Television. Commercial.' },
  { line1: 'LookDev & Lighting Lead.', line2: 'Oscar-nominated productions.' },
  { line1: 'Creative Technologist.', line2: 'Generative AI / R&D.' },
]

function CyclingLines({ years }: { years: number }) {
  const list = phrases(years)
  const [cur, setCur] = useState(0)
  const [phase, setPhase] = useState<'idle' | 'exit' | 'enter'>('idle')

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const timers: number[] = []
    const interval = window.setInterval(() => {
      setPhase('exit')
      timers.push(window.setTimeout(() => {
        setCur(c => (c + 1) % list.length)
        setPhase('enter')
        timers.push(window.setTimeout(() => setPhase('idle'), 600))
      }, 380))
    }, 4000)
    return () => { window.clearInterval(interval); timers.forEach(t => window.clearTimeout(t)) }
  }, [list.length])

  const style: React.CSSProperties =
    phase === 'exit' ? { opacity: 0, transform: 'translateY(-10px)', transition: 'opacity 0.35s ease, transform 0.35s ease' }
    : phase === 'enter' ? { opacity: 1, transform: 'translateY(0)', transition: 'opacity 0.55s ease, transform 0.55s ease' }
    : { opacity: 1, transform: 'translateY(0)' }

  return (
    <span style={{ display: 'block', minHeight: '2.1em' }}>
      <span style={{ display: 'block', color: 'var(--text-bright)', ...style }}>{list[cur].line1}</span>
      <span style={{ display: 'block', color: 'var(--accent)', ...style }}>{list[cur].line2}</span>
    </span>
  )
}

export default function Hero({ years }: { years: number }) {
  return (
    <section id="hero" className="hero-pad" style={{
      borderBottom: '0.5px solid var(--border)',
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', inset: 0, opacity: 0.04, pointerEvents: 'none',
        backgroundImage: 'linear-gradient(var(--grid-line) 1px, transparent 1px), linear-gradient(90deg, var(--grid-line) 1px, transparent 1px)',
        backgroundSize: '48px 48px',
      }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 24 }}>
        <span style={{ display: 'inline-block', width: 24, height: 1, background: 'var(--accent)' }} />
        VFX Artist&nbsp;·&nbsp;Creative Technologist
        <span style={{ display: 'inline-block', width: 24, height: 1, background: 'var(--accent)' }} />
      </div>

      {/* His name is the page's main heading; both practices are on the first screen. */}
      <h1 style={{ fontSize: 'clamp(32px, 5vw, 52px)', fontWeight: 500, lineHeight: 1.05, letterSpacing: '-0.02em', marginBottom: 16 }}>
        <span style={{ display: 'block', fontSize: 'clamp(20px, 3vw, 28px)', color: 'var(--text-soft)', letterSpacing: '-0.01em', marginBottom: 12 }}>
          Matthew Bell
        </span>
        <CyclingLines years={years} />
      </h1>

      <p style={{ fontSize: 15, color: 'var(--text-muted)', maxWidth: 580, lineHeight: 1.8, marginBottom: 40 }}>
        {years} years in 3D and visual effects, with lead roles on four films nominated for the Best Visual Effects Oscar.{' '}
        <strong style={{ color: 'var(--text-soft)', fontWeight: 500 }}>A specialist in lighting and look development</strong>
        {' for film, episodic TV, commercials and real-time production, and a creative technologist building generative AI and real-time tools. Based in San Antonio, working remote or on-location.'}
      </p>

      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <a href="#reel" style={{
          background: 'var(--accent)', color: 'var(--on-accent)', fontSize: 12,
          letterSpacing: '0.1em', textTransform: 'uppercase', padding: '12px 28px',
          fontWeight: 500, textDecoration: 'none', display: 'inline-block',
        }}>View showreel</a>
        <a href="#contact" style={{
          fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase',
          color: 'var(--text-muted)', textDecoration: 'none',
        }}>→ Get in touch</a>
      </div>

      <div style={{
        display: 'flex', gap: 32, marginTop: 56, paddingTop: 40,
        borderTop: '0.5px solid var(--border)', flexWrap: 'wrap',
      }}>
        {stats(years).map(s => (
          <div key={s.label}>
            <div style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.02em' }}>{s.num}</div>
            <div style={{ fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-dim)', marginTop: 4 }}>{s.label}</div>
          </div>
        ))}
      </div>
    </section>
  )
}
