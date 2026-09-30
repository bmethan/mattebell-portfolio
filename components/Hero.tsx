// Award stats name the award and his part in it, the way producers will check them: the VES and the Emmy went
// to the productions (Walmart "Famous Visitors", The Mill; NFL Network "Run", Motion Theory), and he was on the
// crew, as lighter on "Run".
const stats = (years: number) => [
  { num: `${years}+`, label: 'Years experience' },
  { num: '20+', label: 'Feature film credits' },
  { num: '3×', label: 'Oscar-nominated productions' },
  { num: 'VES', label: 'Award-winning spot (crew)' },
  { num: 'Emmy', label: 'Award-winning spot (lighter)' },
]

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
        <span style={{ display: 'block', color: 'var(--text-bright)' }}>Film-grade lighting and lookdev.</span>
        <span style={{ display: 'block', color: 'var(--accent)' }}>Real-time and generative AI R&amp;D.</span>
      </h1>

      <p style={{ fontSize: 15, color: 'var(--text-muted)', maxWidth: 580, lineHeight: 1.8, marginBottom: 40 }}>
        {years} years in 3D and visual effects, with credits on three Oscar-nominated productions.{' '}
        <strong style={{ color: 'var(--text-soft)', fontWeight: 500 }}>A specialist in lighting and look development</strong>
        {' for film, episodic TV and real-time production, and a creative technologist building generative AI and real-time tools. Based in San Antonio, working remote or on-location.'}
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
