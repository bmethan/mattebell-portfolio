'use client'
import { useState, useEffect } from 'react'

type Clip = { label: string; url: string }
type Modal = { blurb: string; clips: Clip[]; imdb?: string }

type Card = {
  title: string
  year: string
  type: string
  badge?: string
  role: string
  studio: string
  summary?: string // one line of what he did, as text search engines can read
  icon?: string
  image?: string
  modal?: Modal
}

const VIMEO = (id: string) =>
  `https://player.vimeo.com/video/${id}?title=0&byline=0&portrait=0&like=0&watchlater=0&share=0`

const DEFAULT_CARDS: Card[] = [
  // Card years are release or air years. Badges describe the production, not a personal award.
  { title: 'Ready Player One', year: '2018', type: 'Feature film · Steven Spielberg', badge: 'Oscar-nominated: Best Visual Effects', role: 'LookDev & Lighting Lead', studio: 'Digital Domain', icon: '🎬', image: '/images/rpocard.jpg' },
  { title: 'Iron Man 3', year: '2013', type: 'Feature film', badge: 'Oscar-nominated: Best Visual Effects', role: 'LookDev & Lighting Lead', studio: 'Digital Domain', icon: '🎬', image: '/images/im3card.jpg' },
  { title: 'Real Steel', year: '2011', type: 'Feature film', badge: 'Oscar-nominated: Best Visual Effects', role: 'LookDev & Lighting Lead', studio: 'Digital Domain', icon: '🎬', image: '/images/realsteelcard.jpg' },
  { title: 'Star Trek', year: '2009', type: 'Feature film', badge: 'Oscar-nominated: Best Visual Effects', role: 'LookDev & Lighting Lead', studio: 'Digital Domain', icon: '🎬', image: '/images/startrekcard.jpg' },
  { title: "DC's Legends of Tomorrow", year: '2021–22', type: 'Television series', badge: '', role: 'Sequence Lighting', studio: 'Zoic Studios', icon: '📺', image: '/images/lotcard.jpg' },
  {
    title: 'Walmart: Famous Visitors', year: '2020', type: 'Super Bowl commercial',
    badge: 'VES Award-winning production',
    role: 'Senior VFX Artist', studio: 'The Mill LA', icon: '📢',
    summary: 'Look development of the LEGO assets and the Blade Runner Spinner; shot lighting for the Spinner and all volumetrics.',
    image: '/images/walmart-famous-visitors.jpg',
    modal: {
      blurb: 'This commercial was an absolute blast to work on. I was involved with the LEGO and Blade Runner portions of the spot, and as a lifelong LEGO fan, getting to work on the LEGO characters and spaceship really brought out my inner kid. I was responsible for the look development of the LEGO assets and the Spinner from Blade Runner, as well as the shot lighting for the Spinner and all volumetrics.',
      clips: [
        { label: 'Commercial Extended Cut', url: VIMEO('1202709086') },
        { label: 'Behind the Scenes', url: VIMEO('1202709087') },
      ],
    },
  },
]

function WorkModal({ card, onClose }: { card: Card; onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', handler)
      document.body.style.overflow = ''
    }
  }, [onClose])

  const m = card.modal!
  const twoClips = m.clips.length === 2

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: 'var(--overlay)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '24px',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: 'var(--bg-card)', border: '0.5px solid var(--border)',
          width: '100%', maxWidth: 960, maxHeight: '90vh',
          overflowY: 'auto', position: 'relative',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          padding: '24px 28px 20px', borderBottom: '0.5px solid var(--border)',
        }}>
          <div>
            <div style={{ fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 4 }}>
              {card.type}
            </div>
            {card.badge && (
              <div style={{
                display: 'inline-block', fontSize: 9, letterSpacing: '0.08em', textTransform: 'uppercase',
                background: 'var(--bg-teal)', color: 'var(--accent)', border: '0.5px solid var(--border-teal)',
                padding: '3px 8px', marginBottom: 8,
              }}>{card.badge}</div>
            )}
            <h2 style={{ fontSize: 22, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em', marginBottom: 4 }}>
              {card.title}
            </h2>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
              {card.role}&nbsp;·&nbsp;{card.studio}&nbsp;·&nbsp;{card.year}
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            style={{
              background: 'none', border: '0.5px solid var(--border)', cursor: 'pointer',
              color: 'var(--text-muted)', fontSize: 18, lineHeight: 1,
              padding: '6px 10px', flexShrink: 0,
            }}
          >✕</button>
        </div>

        {/* Video clips */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: twoClips ? '1fr 1fr' : '1fr',
          gap: 1, background: 'var(--border)',
          borderBottom: '0.5px solid var(--border)',
        }}>
          {m.clips.map(clip => (
            <div key={clip.label} style={{ background: 'var(--bg)' }}>
              <div style={{ aspectRatio: '16/9', position: 'relative' }}>
                <iframe
                  src={clip.url}
                  title={`${card.title}: ${clip.label}`}
                  style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', border: 'none' }}
                  allow="autoplay; fullscreen"
                  allowFullScreen
                />
              </div>
              <div style={{
                padding: '10px 16px', fontSize: 10, letterSpacing: '0.1em',
                textTransform: 'uppercase', color: 'var(--text-dim)',
                borderTop: '0.5px solid var(--border)',
              }}>
                {clip.label}
              </div>
            </div>
          ))}
        </div>

        {/* Blurb */}
        <div style={{ padding: '24px 28px' }}>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, margin: 0 }}>
            {m.blurb}
          </p>
          {m.imdb && (
            <a href={m.imdb} target="_blank" rel="noopener noreferrer" style={{
              display: 'inline-block', marginTop: 16, fontSize: 11, letterSpacing: '0.1em',
              textTransform: 'uppercase', color: 'var(--accent)', textDecoration: 'none',
            }}>
              View on IMDb →
            </a>
          )}
        </div>
      </div>
    </div>
  )
}

function WorkCard({ card, onClick }: { card: Card; onClick?: () => void }) {
  const clickable = !!card.modal
  return (
    <div
      className="work-card"
      onClick={clickable ? onClick : undefined}
      style={{ background: 'var(--bg)', padding: 28, position: 'relative', cursor: clickable ? 'pointer' : 'default' }}
    >
      <div style={{
        width: '100%', aspectRatio: '16/9', background: 'var(--bg-card)',
        marginBottom: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: '0.5px solid var(--border-subtle)', fontSize: 28, color: 'var(--border)',
        overflow: 'hidden', position: 'relative',
      }}>
        {card.image
          ? <img src={card.image} alt={`Still from ${card.title}`} loading="lazy" decoding="async" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
          : <span>◻</span>
        }
        {/* The chip is the keyboard and screen reader way in (its click bubbles to the card); the card's text
            stays plain content, so the badge and summary are still read out. */}
        {clickable && (
          <button type="button" aria-label={`Learn more about ${card.title}`} style={{
            position: 'absolute', bottom: 10, right: 10, font: 'inherit', cursor: 'pointer',
            fontSize: 9, letterSpacing: '0.1em', textTransform: 'uppercase',
            background: 'var(--overlay-panel)', color: 'var(--accent)',
            border: '0.5px solid var(--border-teal)', padding: '4px 8px',
          }}>
            Learn more
          </button>
        )}
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
      {card.summary && <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.6, margin: '10px 0 0' }}>{card.summary}</p>}
    </div>
  )
}

export default function Work({ cards }: { cards?: Card[] }) {
  const display = cards && cards.length > 0 ? cards : DEFAULT_CARDS
  const [active, setActive] = useState<Card | null>(null)

  return (
    <section id="work" className="section-pad" style={{ borderBottom: '0.5px solid var(--border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 40 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
            <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
            Selected work
            <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
          </div>
          <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em' }}>Featured projects</h2>
        </div>
        <a href="https://www.imdb.com/name/nm2998873/" target="_blank" rel="noopener noreferrer" style={{ fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-dim)', textDecoration: 'none' }}>
          All credits →
        </a>
      </div>
      <div className="grid-2col">
        {display.map((card, i) => (
          <WorkCard key={i} card={card} onClick={() => setActive(card)} />
        ))}
      </div>
      {active?.modal && <WorkModal card={active} onClose={() => setActive(null)} />}
    </section>
  )
}
