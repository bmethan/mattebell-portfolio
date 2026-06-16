const stats = [
  { num: '27+', label: 'Years experience' },
  { num: '20+', label: 'Feature film credits' },
  { num: '3×', label: 'Oscar-nominated productions' },
  { num: '1×', label: 'VES Award win' },
  { num: '1×', label: 'Emmy Award win' },
]

export default function Hero() {
  return (
    <section id="hero" style={{
      padding: '80px 48px 64px', borderBottom: '0.5px solid var(--border)',
      position: 'relative', overflow: 'hidden',
    }}>
      {/* grid bg */}
      <div style={{
        position: 'absolute', inset: 0, opacity: 0.04, pointerEvents: 'none',
        backgroundImage: 'linear-gradient(#888 1px, transparent 1px), linear-gradient(90deg, #888 1px, transparent 1px)',
        backgroundSize: '48px 48px',
      }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 24 }}>
        <span style={{ display: 'inline-block', width: 24, height: 1, background: 'var(--accent)' }} />
        Creative Technologist&nbsp;·&nbsp;VFX Artist
      </div>

      <h1 style={{ fontSize: 'clamp(32px, 5vw, 52px)', fontWeight: 500, lineHeight: 1.05, letterSpacing: '-0.02em', color: 'var(--text-bright)', marginBottom: 16 }}>
        Lighting. Look Development.<br />
        <span style={{ color: 'var(--accent)' }}>Full-pipeline expertise.</span>
      </h1>

      <p style={{ fontSize: 15, color: 'var(--text-muted)', maxWidth: 580, lineHeight: 1.8, marginBottom: 40 }}>
        Crafting worlds for film and television since 1998.{' '}
        <strong style={{ color: '#b4b2a9', fontWeight: 500 }}>Specialist in lighting and look development</strong>
        {', and a trusted generalist across the full pipeline - from environment builds and FX to final comp. Bringing vision to screen across 27 years of feature film and episodic television.'}
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
        {stats.map(s => (
          <div key={s.label}>
            <div style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.02em' }}>{s.num}</div>
            <div style={{ fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-dim)', marginTop: 4 }}>{s.label}</div>
          </div>
        ))}
      </div>
    </section>
  )
}
