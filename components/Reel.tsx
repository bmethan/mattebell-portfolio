const DEFAULT_REEL = 'https://player.vimeo.com/video/357635744?h=896815367e&title=0&byline=0&portrait=0&like=0&watchlater=0&share=0'

export default function Reel({ reelUrl }: { reelUrl?: string }) {
  reelUrl = reelUrl || DEFAULT_REEL
  return (
    <section id="reel" style={{ background: 'var(--bg-card)', borderBottom: '0.5px solid var(--border)' }}>
      <div style={{
        width: '100%', aspectRatio: '16/7', background: 'var(--bg-alt)',
        borderTop: '0.5px solid var(--border)', borderBottom: '0.5px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative',
        overflow: 'hidden',
      }}>
        {reelUrl ? (
          <iframe
            src={reelUrl}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', border: 'none' }}
            allow="autoplay; fullscreen"
            allowFullScreen
          />
        ) : (
          <div style={{
            width: 64, height: 64, border: '0.5px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
          }}>
            <span style={{ fontSize: 22, color: 'var(--text)', marginLeft: 4 }}>▶</span>
          </div>
        )}
      </div>
      <div className="reel-bar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 48px' }}>
        <span style={{ fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-dim)' }}>2024 Showreel</span>
        <a href="https://vimeo.com/user6348780" target="_blank" rel="noopener noreferrer" style={{ fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-dim)', textDecoration: 'none' }}>
          ↓ Download reel
        </a>
      </div>
    </section>
  )
}
