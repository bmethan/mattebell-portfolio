const DEFAULT_TESTIMONIALS = [
  {
    quote: 'Matt showed excellent Look Development and Lighting skills on Real Steel working in Maya/V-Ray. He was able to quickly hit the ground running and lookdev a large number of photo real robots under an extremely tight deadline. He was given a sequence of 50 shots to setup master lighting and compositing templates, and also managed a small team underneath him. His attitude, pride and attention to detail showed me just how passionate he is about Visual Effects. It\'s been a great pleasure working with Matt — I would highly recommend him to anyone looking for a Senior level Lighting / Look Development Artist.',
    name: 'Isaac Irvin',
    title: 'VFX / CG Supervisor & AI Workflow Specialist',
    initials: 'II',
  },
  {
    quote: 'Matt Bell and I studied at Savannah College of Art and Design and went on to be lighters on various projects at Digital Domain. After 3 features and 1 commercial together I can safely say that Matt knows his stuff. He practically did an entire character sequence on Star Trek by himself: Modeling, Texture, Look-Dev, Lighting... He can cross over 3D packages and has the skills and experience to know which is best suited for the task at hand. Very professional and a good generalist as well. This is a guy who is in it for the love of the art!',
    name: 'Kevin Sears',
    title: 'CG Supervisor | Framestore Montreal',
    initials: 'KS',
  },
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
