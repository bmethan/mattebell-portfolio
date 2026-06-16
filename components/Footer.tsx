export default function Footer() {
  return (
    <footer style={{
      padding: '24px 48px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      borderTop: '0.5px solid var(--border)',
    }}>
      <div style={{ fontSize: 11, color: 'var(--text-ghost)', letterSpacing: '0.06em' }}>
        &copy; {new Date().getFullYear()} Matthew Bell. All rights reserved.
      </div>
      <div style={{ display: 'flex', gap: 24 }}>
        {[
          { label: 'IMDb', href: 'https://www.imdb.com/name/nm2998873/' },
          { label: 'LinkedIn', href: 'https://linkedin.com/in/mattebell' },
          { label: 'Vimeo', href: 'https://vimeo.com/user6348780' },
        ].map(l => (
          <a key={l.label} href={l.href} target="_blank" rel="noopener noreferrer" style={{
            fontSize: 11, color: 'var(--text-ghost)', letterSpacing: '0.08em',
            textTransform: 'uppercase', textDecoration: 'none',
          }}>{l.label}</a>
        ))}
      </div>
    </footer>
  )
}
