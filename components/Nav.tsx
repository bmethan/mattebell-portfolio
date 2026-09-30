'use client'
import { useState } from 'react'

const LINKS = ['Services', 'Work', 'Skills', 'About', 'Contact']

export default function Nav({ available }: { available: boolean }) {
  const [open, setOpen] = useState(false)

  return (
    <nav className="nav-pad" style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      borderBottom: '0.5px solid var(--border)',
      position: 'sticky', top: 0, background: 'var(--bg)', zIndex: 50,
    }}>
      <div style={{ fontSize: 13, fontWeight: 500, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
        Matthew Bell
      </div>

      {/* Desktop links */}
      <div className="nav-links" style={{ gap: 32 }}>
        {LINKS.map(item => (
          <a key={item} href={`#${item.toLowerCase()}`} style={{
            fontSize: 12, letterSpacing: '0.08em', color: 'var(--text-muted)',
            textTransform: 'uppercase', textDecoration: 'none',
          }}>{item}</a>
        ))}
      </div>

      {/* Desktop hire button */}
      <a href="#contact" className="nav-links" style={{
        fontSize: 12, letterSpacing: '0.08em', textTransform: 'uppercase',
        border: '0.5px solid var(--text-ghost)', padding: '8px 20px',
        color: available ? 'var(--accent)' : 'var(--text-muted)',
        textDecoration: 'none', cursor: 'pointer',
      }}>
        {available ? 'Available for hire' : 'Currently unavailable'}
      </a>

      {/* Phones: a contact link stays visible next to the menu button. */}
      <div className="nav-mobile-actions">
        <a href="#contact" className="nav-cta-mobile" onClick={() => setOpen(false)}>Contact</a>
        <button
          className="nav-hamburger"
          onClick={() => setOpen(o => !o)}
          aria-label="Menu"
          aria-expanded={open}
          aria-controls="nav-mobile-menu"
        >
          <svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            {open
              ? <><line x1="4" y1="4" x2="18" y2="18"/><line x1="18" y1="4" x2="4" y2="18"/></>
              : <><line x1="3" y1="7" x2="19" y2="7"/><line x1="3" y1="11" x2="19" y2="11"/><line x1="3" y1="15" x2="19" y2="15"/></>
            }
          </svg>
        </button>
      </div>

      {/* Mobile dropdown */}
      <div id="nav-mobile-menu" className={`nav-mobile-menu${open ? ' open' : ''}`}>
        {LINKS.map(item => (
          <a key={item} href={`#${item.toLowerCase()}`} onClick={() => setOpen(false)}>
            {item}
          </a>
        ))}
        <a href="#contact" onClick={() => setOpen(false)} style={{
          color: available ? 'var(--accent)' : 'var(--text-muted)',
          borderBottom: 'none',
        }}>
          {available ? 'Available for hire' : 'Currently unavailable'}
        </a>
      </div>
    </nav>
  )
}
