'use client'
import { useEffect, useRef, useState } from 'react'

const stats = (years: number) => [
  { num: `${years}+`, label: 'Years experience' },
  { num: '20+', label: 'Feature film credits' },
  { num: '3×', label: 'Oscar-nominated productions' },
  { num: '1×', label: 'VES Award win' },
  { num: '1×', label: 'Emmy Award win' },
]

const phrases = (years: number) => [
  { line1: 'Lighting. Look Development.', line2: 'Full-pipeline expertise.' },
  { line1: 'Real-time & Unreal Engine.', line2: 'Virtual production.' },
  { line1: `${years} years. Feature film.`, line2: 'Television. Commercial.' },
  { line1: 'LookDev & Lighting Lead.', line2: 'Oscar-nominated productions.' },
  { line1: 'Creative Technologist.', line2: 'Generative AI / R&D.' },
]

function CyclingHero({ years }: { years: number }) {
  const list = phrases(years)
  const [cur, setCur] = useState(0)
  const [prev, setPrev] = useState<number | null>(null)
  const [phase, setPhase] = useState<'idle' | 'exit' | 'enter'>('idle')
  const nextRef = useRef(0)

  useEffect(() => {
    const interval = setInterval(() => {
      const next = (nextRef.current + 1) % list.length
      nextRef.current = next
      setPrev(nextRef.current === next ? cur : cur)
      setPhase('exit')
      setTimeout(() => {
        setCur(next)
        setPrev(null)
        setPhase('enter')
        setTimeout(() => setPhase('idle'), 600)
      }, 380)
    }, 4000)
    return () => clearInterval(interval)
  }, [])

  const exitStyle: React.CSSProperties = { opacity: 0, transform: 'translateY(-10px)', transition: 'opacity 0.35s ease, transform 0.35s ease' }
  const enterStyle: React.CSSProperties = { opacity: 1, transform: 'translateY(0)', transition: 'opacity 0.55s ease, transform 0.55s ease' }
  const idleStyle: React.CSSProperties = { opacity: 1, transform: 'translateY(0)' }

  const currentStyle = phase === 'exit' ? exitStyle : phase === 'enter' ? enterStyle : idleStyle

  return (
    <h1 style={{ fontSize: 'clamp(32px, 5vw, 52px)', fontWeight: 500, lineHeight: 1.05, letterSpacing: '-0.02em', marginBottom: 16, minHeight: '2.15em' }}>
      <span style={{ display: 'block', color: 'var(--text-bright)', ...currentStyle }}>
        {list[cur].line1}
      </span>
      <span style={{ display: 'block', color: 'var(--accent)', ...currentStyle }}>
        {list[cur].line2}
      </span>
    </h1>
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
        backgroundImage: 'linear-gradient(#888 1px, transparent 1px), linear-gradient(90deg, #888 1px, transparent 1px)',
        backgroundSize: '48px 48px',
      }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 24 }}>
        <span style={{ display: 'inline-block', width: 24, height: 1, background: 'var(--accent)' }} />
        Creative Technologist&nbsp;·&nbsp;VFX Artist
      </div>

      <CyclingHero years={years} />

      <p style={{ fontSize: 15, color: 'var(--text-muted)', maxWidth: 580, lineHeight: 1.8, marginBottom: 40 }}>
        Crafting worlds for film and television since 1998.{' '}
        <strong style={{ color: '#b4b2a9', fontWeight: 500 }}>Specialist in lighting and look development</strong>
        {', and a trusted generalist across the full pipeline - from environment builds and FX to final comp. Bringing vision to screen across '}
        {years} years of feature film and episodic television.
      </p>

      <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
        <a href="#reel" style={{
          background: 'var(--accent)', color: '#04342C', fontSize: 12,
          letterSpacing: '0.1em', textTransform: 'uppercase', padding: '12px 28px',
          fontWeight: 500, textDecoration: 'none', display: 'inline-block',
        }}>View showreel</a>
        <a href="#work" style={{
          fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase',
          color: 'var(--text-muted)', textDecoration: 'none',
        }}>→ See selected work</a>
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
