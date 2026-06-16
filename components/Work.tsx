const DEFAULT_CARDS = [
  { title: 'Ready Player One', year: '2018', type: 'Feature film', badge: 'Spielberg', role: 'LookDev & Lighting Lead', studio: 'Digital Domain', icon: '🎬', image: '/images/rpocard.jpg' },
  { title: 'Iron Man 3', year: '2013', type: 'Feature film', badge: 'Oscar-nominated production', role: 'LookDev & Lighting Lead', studio: 'Digital Domain', icon: '🎬', image: '/images/im3card.jpg' },
  { title: 'Real Steel', year: '2011', type: 'Feature film', badge: 'Oscar-nominated production', role: 'Senior LookDev & Lighting Artist', studio: 'Digital Domain', icon: '🎬', image: '/images/realsteelcard.jpg' },
  { title: 'Star Trek', year: '2009', type: 'Feature film', badge: 'Oscar-nominated production', role: 'Senior LookDev & Lighting Artist', studio: 'Digital Domain', icon: '🎬', image: '/images/startrekcard.jpg' },
  { title: 'The Flash', year: '2021–23', type: 'Television series', badge: '', role: 'Sequence Lighting', studio: 'Zoic Studios', icon: '📺', image: '' },
  { title: 'Walmart — Famous Visitors', year: '2021', type: 'Commercial', badge: 'VES Award — Outstanding VFX in a Commercial', role: 'Senior VFX Artist', studio: 'The Mill LA', icon: '📢', image: '/images/walmart-famous-visitors.jpg' },
]

type Card = {
  title: string
  year: string
  type: string
  badge?: string
  role: string
  studio: string
  image?: string
}

function WorkCard({ card }: { card: Card }) {
  return (
    <div className="work-card" style={{
      background: 'var(--bg)', padding: 28, position: 'relative', cursor: 'pointer',
    }}>
      <div style={{
        width: '100%', aspectRatio: '16/9', background: 'var(--bg-card)',
        marginBottom: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: '0.5px solid var(--border-subtle)', fontSize: 28, color: 'var(--border)',
        overflow: 'hidden', position: 'relative',
      }}>
        {card.image
          ? <img src={card.image} alt={card.title} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
          : <span>◻</span>
        }
      </div>
      <div style={{ position: 'absolute', top: 28, right: 28, fontSize: 11, color: 'var(--text-ghost)', letterSpacing: '0.06em' }}>
        {card.year}
      </div>
      <div style={{ fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 6 }}>
        {card.type}
      </div>
      {card.badge && (
        <div style={{
          display: 'inline-block', fontSize: 9, letterSpacing: '0.08em', textTransform: 'uppercase',
          background: 'var(--bg-teal)', color: 'var(--accent)', border: '0.5px solid var(--border-teal)',
          padding: '3px 8px', marginBottom: 6,
        }}>
          {card.badge}
        </div>
      )}
      <div style={{ fontSize: 16, fontWeight: 500, color: 'var(--text)', marginBottom: 4 }}>{card.title}</div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>{card.role}&nbsp;·&nbsp;{card.studio}</div>
    </div>
  )
}

export default function Work({ cards }: { cards: Card[] }) {
  const display = cards?.length > 0 ? cards : DEFAULT_CARDS

  return (
    <section id="work" style={{ padding: '72px 48px', borderBottom: '0.5px solid var(--border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 40 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
            <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
            Selected work
          </div>
          <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em' }}>Featured projects</h2>
        </div>
        <a href="https://www.imdb.com/name/nm2998873/" target="_blank" rel="noopener noreferrer" style={{ fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-dim)', textDecoration: 'none' }}>
          All credits →
        </a>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'var(--border)' }}>
        {display.map((card, i) => <WorkCard key={i} card={card} />)}
      </div>
    </section>
  )
}
