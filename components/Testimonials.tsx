const DEFAULT_TESTIMONIALS = [
  { quote: 'One of the most technically rigorous and creatively driven VFX artists I\'ve had the pleasure of working with. The work speaks for itself.', name: 'Your collaborator\'s name', title: 'Director / VFX Supervisor', initials: '—' },
  { quote: 'Brings an exceptional command of both the technical and artistic sides of production. An invaluable collaborator on any effects-heavy project.', name: 'Your collaborator\'s name', title: 'VFX Producer / Supervisor', initials: '—' },
]

type Testimonial = { quote: string; name: string; title: string; initials: string }

export default function Testimonials({ items }: { items: Testimonial[] }) {
  const display = items?.length > 0 ? items : DEFAULT_TESTIMONIALS

  return (
    <section id="testimonials" style={{ padding: '72px 48px', borderBottom: '0.5px solid var(--border)', background: 'var(--bg-alt)' }}>
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
          Testimonials
        </div>
        <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em' }}>What collaborators say</h2>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'var(--border)' }}>
        {display.map((t, i) => (
          <div key={i} style={{ background: 'var(--bg)', padding: 32 }}>
            <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, marginBottom: 24, fontStyle: 'italic' }}>
              &ldquo;{t.quote}&rdquo;
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{
                width: 32, height: 32, background: 'var(--border-subtle)', border: '0.5px solid var(--border)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, color: 'var(--text-dim)', fontWeight: 500,
              }}>{t.initials}</div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 500, color: '#b4b2a9' }}>{t.name}</div>
                <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>{t.title}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
