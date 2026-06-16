'use client'
import Link from 'next/link'

export default function Nav({ available }: { available: boolean }) {
  return (
    <nav style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '20px 48px', borderBottom: '0.5px solid var(--border)',
      position: 'sticky', top: 0, background: 'var(--bg)', zIndex: 50,
    }}>
      <div style={{ fontSize: 13, fontWeight: 500, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
        Matthew Bell&nbsp;/&nbsp;VFX
      </div>
      <div style={{ display: 'flex', gap: 32 }}>
        {['Work', 'Skills', 'About', 'Contact'].map(item => (
          <a key={item} href={`#${item.toLowerCase()}`} style={{
            fontSize: 12, letterSpacing: '0.08em', color: 'var(--text-muted)',
            textTransform: 'uppercase', textDecoration: 'none',
          }}>{item}</a>
        ))}
      </div>
      <a href="#contact" style={{
        fontSize: 12, letterSpacing: '0.08em', textTransform: 'uppercase',
        border: '0.5px solid var(--text-ghost)', padding: '8px 20px',
        color: available ? 'var(--accent)' : 'var(--text-muted)',
        textDecoration: 'none', cursor: 'pointer',
      }}>
        {available ? 'Available for hire' : 'Currently unavailable'}
      </a>
    </nav>
  )
}
