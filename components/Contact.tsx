'use client'
import { useEffect, useState } from 'react'
import WorkIndicator from './WorkIndicator'

const ENDPOINT = 'https://formspree.io/f/mwvjzyqj'
const EMAIL = 'bmethan@gmail.com'
const SOURCE_KEY = 'mb-source'

// Two options per practice, so neither market reads as secondary. The choice becomes the email subject, which
// is also how inquiries are counted per market.
const TOPICS = [
  'Lighting and lookdev (film, episodic, commercials)',
  'Unreal Engine and virtual production',
  'Generative AI R&D and pipelines',
  'Real-time prototypes and custom tools',
  'Consultation',
  'A talk or panel',
  'Something else',
]

// Where the visitor came from (a UTM-tagged link, else the referring site), remembered for the session.
function rememberSource() {
  try {
    if (sessionStorage.getItem(SOURCE_KEY)) return
    const q = new URLSearchParams(location.search)
    let source = [q.get('utm_source'), q.get('utm_medium'), q.get('utm_campaign')].filter(Boolean).join(' / ')
    if (!source && document.referrer) {
      const host = new URL(document.referrer).hostname
      if (host !== location.hostname) source = host
    }
    sessionStorage.setItem(SOURCE_KEY, source || 'direct')
  } catch {
    // Storage blocked: the inquiry just goes without a source.
  }
}

function readSource() {
  try {
    return sessionStorage.getItem(SOURCE_KEY) ?? ''
  } catch {
    return ''
  }
}

const subjectOf = (data: FormData) => `${data.get('topic') ?? ''} inquiry from ${String(data.get('name') ?? '').trim()}`

// The same inquiry as an email, for when the send fails: the visitor's own words, nothing retyped. Line breaks
// are CRLF, as mailto bodies require (RFC 6068).
function buildMailto(data: FormData) {
  const dates = String(data.get('dates') ?? '').trim()
  const body = [String(data.get('message') ?? ''), dates && `Dates: ${dates}`].filter(Boolean).join('\n\n')
  return `mailto:${EMAIL}?subject=${encodeURIComponent(subjectOf(data))}&body=${encodeURIComponent(body.replace(/\r?\n/g, '\r\n'))}`
}

const fieldStyle: React.CSSProperties = {
  background: 'var(--bg)', border: '0.5px solid var(--border)', padding: '10px 14px',
  fontSize: 13, color: 'var(--text)', fontFamily: 'inherit',
}
const labelStyle: React.CSSProperties = { fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-muted)' }
const THANKS = `Thanks. Your message is in my inbox, and I'll reply from ${EMAIL}.`

export default function Contact() {
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const [fallback, setFallback] = useState('')

  useEffect(rememberSource, [])

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (status === 'sending') return
    const form = e.currentTarget
    const data = new FormData(form)
    data.set('subject', subjectOf(data))
    data.set('source', readSource())
    setFallback(buildMailto(data))
    setStatus('sending')
    try {
      const res = await fetch(ENDPOINT, {
        method: 'POST',
        body: data,
        headers: { Accept: 'application/json' },
        signal: AbortSignal.timeout(15000),
      })
      setStatus(res.ok ? 'sent' : 'error')
    } catch {
      setStatus('error')
    }
  }

  return (
    <section id="contact" className="section-pad" style={{ borderBottom: '0.5px solid var(--border)', background: 'var(--bg-card)' }}>
      <div className="grid-2col-contact">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
            <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
            Get in touch
            <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
          </div>
          <h2 style={{ fontSize: 22, fontWeight: 500, color: 'var(--text-bright)', marginBottom: 16, letterSpacing: '-0.01em' }}>
            Let&apos;s make something worth the render time.
          </h2>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, marginBottom: 12 }}>
            Two kinds of work, remote or on-location from San Antonio (US Central time):
          </p>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, marginBottom: 8 }}>
            <strong style={{ color: 'var(--text-soft)', fontWeight: 500 }}>Film, episodic and commercials.</strong>{' '}
            Lighting and look development leads, senior generalist roles, Unreal Engine and virtual production, VFX consultation.
          </p>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.8, marginBottom: 32 }}>
            <strong style={{ color: 'var(--text-soft)', fontWeight: 500 }}>Creative technology.</strong>{' '}
            Generative AI and real-time R&amp;D, prototypes, and custom tools and pipelines for studios, agencies, brands and in-house innovation teams.
          </p>
          {[
            { icon: '✉', text: EMAIL, href: `mailto:${EMAIL}`, external: false },
            { icon: '📍', text: 'San Antonio, Texas. Remote or on-location, worldwide.', href: null, external: false },
            { icon: 'in', text: 'linkedin.com/in/mattebell', href: 'https://linkedin.com/in/mattebell', external: true },
            { icon: 'IMDb', text: 'Matthew E. Bell on IMDb', href: 'https://www.imdb.com/name/nm2998873/', external: true },
            { icon: '▶', text: 'vimeo.com/user6348780', href: 'https://vimeo.com/user6348780', external: true },
          ].map(item => (
            <div key={item.text} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, fontSize: 13, color: 'var(--text-muted)' }}>
              <span aria-hidden="true" style={{
                fontSize: item.icon === 'IMDb' ? 8 : item.icon === 'in' ? 10 : 14,
                color: item.icon === 'IMDb' ? 'var(--on-accent)' : 'var(--accent)',
                background: item.icon === 'IMDb' ? 'var(--accent)' : 'transparent',
                padding: item.icon === 'IMDb' ? '1px 3px' : 0,
                fontWeight: 500,
                minWidth: 16,
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              }}>{item.icon}</span>
              {item.href
                ? <a href={item.href} {...(item.external ? { target: '_blank', rel: 'noopener noreferrer' } : {})} style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>{item.text}</a>
                : <span>{item.text}</span>
              }
            </div>
          ))}
        </div>

        <div style={{ minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: status === 'sent' ? 'center' : 'flex-start' }}>
          {/* Always mounted, so screen readers announce the confirmation when its text arrives. */}
          <p role="status" style={{ fontSize: 15, color: 'var(--text-soft)', lineHeight: 1.8, margin: 0 }}>
            {status === 'sent' ? THANKS : ''}
          </p>
          {status !== 'sent' && (
          // action and method let the form post straight to Formspree before the page's script has loaded.
          <form action={ENDPOINT} method="POST" onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* Formspree fills this template server side, so the topic reaches the email subject even without
                JavaScript; with it, handleSubmit sets the same text directly. */}
            <input type="hidden" name="subject" value="{{ topic }} inquiry from {{ name }}" />
            <div className="contact-name-email" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <label htmlFor="contact-name" style={labelStyle}>Name</label>
                <input id="contact-name" className="contact-field" name="name" type="text" placeholder="Your name" autoComplete="name" maxLength={120} required style={fieldStyle} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <label htmlFor="contact-email" style={labelStyle}>Email</label>
                <input id="contact-email" className="contact-field" name="email" type="email" placeholder="your@email.com" autoComplete="email" maxLength={200} required style={fieldStyle} />
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <label htmlFor="contact-topic" style={labelStyle}>What&apos;s this about?</label>
              <select id="contact-topic" className="contact-field" name="topic" required defaultValue="" style={{ ...fieldStyle, width: '100%', appearance: 'none', cursor: 'pointer' }}>
                <option value="" disabled>Choose one</option>
                {TOPICS.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <label htmlFor="contact-dates" style={labelStyle}>Dates or deadline (optional)</label>
              <input id="contact-dates" className="contact-field" name="dates" type="text" placeholder="e.g. 6 weeks from mid-November" maxLength={200} style={fieldStyle} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <label htmlFor="contact-message" style={labelStyle}>Message</label>
              <textarea id="contact-message" className="contact-field" name="message" placeholder="Tell me about your project..." required rows={5} maxLength={5000} style={{ ...fieldStyle, resize: 'none' }} />
            </div>
            {/* Spam trap: hidden from people, filled in by bots, and dropped by Formspree. */}
            <input type="text" name="_gotcha" tabIndex={-1} autoComplete="off" aria-hidden="true" style={{ display: 'none' }} />
            <button type="submit" disabled={status === 'sending'} style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              background: 'var(--accent)', color: 'var(--on-accent)', fontSize: 12, letterSpacing: '0.1em',
              textTransform: 'uppercase', padding: 12, fontWeight: 500, border: 'none', cursor: 'pointer',
              fontFamily: 'inherit', marginTop: 4,
            }}>
              {status === 'sending' ? <><WorkIndicator />Sending</> : 'Send message'}
            </button>
            {status === 'error' && (
              <p role="alert" style={{ fontSize: 12, color: 'var(--danger)', lineHeight: 1.6, margin: 0 }}>
                That didn&apos;t go through, and your message is still here. Try again, or{' '}
                <a
                  href={fallback}
                  // Rebuilt at click time, so edits made after the failed send go into the email too.
                  onClick={e => {
                    const form = e.currentTarget.closest('form')
                    if (form) e.currentTarget.href = buildMailto(new FormData(form))
                  }}
                  style={{ color: 'var(--danger)' }}
                >
                  send it from your email app
                </a>
                .
              </p>
            )}
            <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
              Sent to my inbox via Formspree, with the site that referred you. No mailing lists.
            </p>
          </form>
          )}
        </div>
      </div>
    </section>
  )
}
