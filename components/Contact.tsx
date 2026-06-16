'use client'
import { useState } from 'react'

export default function Contact() {
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setStatus('sending')
    const form = e.currentTarget
    const data = new FormData(form)
    const res = await fetch('https://formspree.io/f/mwvjzyqj', {
      method: 'POST',
      body: data,
      headers: { Accept: 'application/json' },
    })
    if (res.ok) {
      form.reset()
      setStatus('sent')
      setTimeout(() => setStatus('idle'), 5000)
    } else {
      setStatus('error')
    }
  }

  return (
    <section id="contact" style={{ padding: '72px 48px', borderBottom: '0.5px solid var(--border)', background: 'var(--bg-card)' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
            <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
            Get in touch
          </div>
          <h3 style={{ fontSize: 22, fontWeight: 500, color: 'var(--text-bright)', marginBottom: 16, letterSpacing: '-0.01em' }}>
            Let&apos;s make something worth the render time.
          </h3>
          <p style={{ fontSize: 14, color: 'var(--text-dim)', lineHeight: 1.8, marginBottom: 32 }}>
            Open to lighting & lookdev leads, senior generalist roles, real-time production, and VFX consultation. If it involves making things look incredible on screen, I&apos;m listening.
          </p>
          {[
            { icon: '✉', text: 'bmethan@gmail.com', href: 'mailto:bmethan@gmail.com' },
            { icon: '📍', text: 'San Antonio, TX — open to remote & on-location', href: null },
            { icon: 'in', text: 'linkedin.com/in/mattebell', href: 'https://linkedin.com/in/mattebell' },
            { icon: 'IMDb', text: 'Matthew Bell on IMDb', href: 'https://www.imdb.com/name/nm2998873/' },
            { icon: '▶', text: 'vimeo.com/user6348780', href: 'https://vimeo.com/user6348780' },
          ].map(item => (
            <div key={item.text} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, fontSize: 13, color: 'var(--text-muted)' }}>
              <span style={{
                fontSize: item.icon === 'IMDb' ? 8 : item.icon === 'in' ? 10 : 14,
                color: item.icon === 'IMDb' ? '#04342C' : 'var(--accent)',
                background: item.icon === 'IMDb' ? 'var(--accent)' : 'transparent',
                padding: item.icon === 'IMDb' ? '1px 3px' : 0,
                fontWeight: 500,
                minWidth: 16,
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              }}>{item.icon}</span>
              {item.href
                ? <a href={item.href} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>{item.text}</a>
                : <span>{item.text}</span>
              }
            </div>
          ))}
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {[{ name: 'name', label: 'Name', type: 'text', placeholder: 'Your name' }, { name: 'email', label: 'Email', type: 'email', placeholder: 'your@email.com' }].map(f => (
              <div key={f.name} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <label style={{ fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-dim)' }}>{f.label}</label>
                <input name={f.name} type={f.type} placeholder={f.placeholder} required style={{ background: 'var(--bg)', border: '0.5px solid var(--border)', padding: '10px 14px', fontSize: 13, color: 'var(--text)', outline: 'none', fontFamily: 'inherit' }} />
              </div>
            ))}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <label style={{ fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-dim)' }}>Subject</label>
            <input name="subject" type="text" placeholder="Project type / enquiry" style={{ background: 'var(--bg)', border: '0.5px solid var(--border)', padding: '10px 14px', fontSize: 13, color: 'var(--text)', outline: 'none', fontFamily: 'inherit' }} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <label style={{ fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-dim)' }}>Message</label>
            <textarea name="message" placeholder="Tell me about your project..." required rows={5} style={{ background: 'var(--bg)', border: '0.5px solid var(--border)', padding: '10px 14px', fontSize: 13, color: 'var(--text)', outline: 'none', resize: 'none', fontFamily: 'inherit' }} />
          </div>
          <button type="submit" disabled={status === 'sending' || status === 'sent'} style={{
            background: 'var(--accent)', color: '#04342C', fontSize: 12, letterSpacing: '0.1em',
            textTransform: 'uppercase', padding: 12, fontWeight: 500, border: 'none', cursor: 'pointer',
            fontFamily: 'inherit', marginTop: 4, opacity: status === 'sending' ? 0.7 : 1,
          }}>
            {status === 'sent' ? 'Message sent!' : status === 'sending' ? 'Sending...' : 'Send message'}
          </button>
          {status === 'error' && <p style={{ fontSize: 12, color: '#e05c5c' }}>Something went wrong — please email directly at bmethan@gmail.com</p>}
        </form>
      </div>
    </section>
  )
}
