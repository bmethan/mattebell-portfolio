import { ImageResponse } from 'next/og'

// The link preview for LinkedIn, Slack, iMessage and the like: typographic, in the site's colors, and with no
// film stills (so no image rights questions). Both practices are named, VFX first.
export const alt = 'Matthew Bell, VFX artist and creative technologist'
export const size = { width: 1200, height: 630 }
export const contentType = 'image/png'

const BG = '#0a0a0b'
const ACCENT = '#5DCAA5'
const BRIGHT = '#f0ede8'
const MUTED = '#888780'
const BORDER = '#2a2a2e'

export default function OpengraphImage() {
  const rule = { width: 40, height: 2, background: ACCENT }
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '72px 80px',
          background: BG,
          border: `2px solid ${BORDER}`,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, color: ACCENT, fontSize: 24, letterSpacing: 5 }}>
          <div style={rule} />
          VFX ARTIST · CREATIVE TECHNOLOGIST
          <div style={rule} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: 112, color: BRIGHT, letterSpacing: -3, lineHeight: 1 }}>Matthew Bell</div>
          <div style={{ fontSize: 40, color: BRIGHT, marginTop: 36 }}>Film-grade lighting and lookdev.</div>
          <div style={{ fontSize: 40, color: ACCENT, marginTop: 8 }}>Real-time and generative AI R&amp;D.</div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 24, color: MUTED, letterSpacing: 2 }}>
          <div>FILM · EPISODIC · COMMERCIALS · REAL-TIME</div>
          <div style={{ color: ACCENT }}>mattebell.xyz</div>
        </div>
      </div>
    ),
    size,
  )
}
