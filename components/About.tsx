const timeline = [
  { year: '2020–', role: 'Freelance Senior LookDev & Lighting / Visualization Consultant', studio: 'Zero VFX · Zoic Studios · Reactor VFX · Artjail · and others', note: 'Challengers · The Instigators · Good Burger 2 · The Flash · Mrs. Davis · Hello Tomorrow!' },
  { year: '2019–20', role: 'Senior VFX Artist', studio: 'The Mill LA', note: "Walmart 'Famous Visitors' (VES Award — Outstanding VFX in a Commercial)" },
  { year: '2016–17', role: 'Lighting Lead', studio: 'Digital Domain', note: 'Ready Player One · X-Men: Apocalypse' },
  { year: '2012–13', role: 'LookDev & Lighting Lead', studio: 'Digital Domain', note: 'Iron Man 3 (Oscar-nominated production) · G.I. Joe: Retaliation' },
  { year: '2007–12', role: 'Senior LookDev & Lighting Artist', studio: 'Digital Domain · Method Studios · Dr. D Studios', note: 'Star Trek · Real Steel (Oscar-nominated productions) · Wrath of the Titans · Happy Feet 2' },
  { year: '1998–07', role: 'LightWave3D Expert / Demo Artist · Graphic Designer', studio: 'NewTek Inc. · NewTek Europe · News 9 San Antonio', note: '' },
]

const pills = [
  { label: 'Lighting & LookDev', accent: true },
  { label: 'Unreal Engine', accent: true },
  { label: 'Creative Technologist', accent: true },
  { label: 'Full-pipeline generalist', accent: false },
  { label: 'VFX consultation', accent: false },
  { label: 'Virtual production', accent: false },
  { label: 'Fur & hair groom', accent: false },
  { label: 'Generative AI / R&D', accent: false },
]

type Settings = {
  bio?: string
  bio2?: string
  bio3?: string
  bio4?: string
  resumeUrl?: string
}

const DEFAULT_BIO = (years: number) => [
  `<strong>Matthew Bell.</strong> Creative Technologist and Senior VFX Artist with ${years} years across feature film, episodic television, commercials, and real-time production. Core specialist in lighting and look development — trusted generalist across the full pipeline.`,
  'Lighting and look development lead on three Oscar-nominated productions — Iron Man 3, Real Steel, and Star Trek — with credits spanning Digital Domain, MPC, The Mill, Method Studios, Zero VFX, Zoic Studios, and more.',
  "Senior artist on the VES Award-winning Walmart 'Famous Visitors' — Outstanding Visual Effects in a Commercial, 2021. Also available for VFX consultation across visualization, real-time pipeline, and UI/UX strategy. Open to remote and on-location engagements worldwide.",
  'Actively exploring generative AI workflows — including ComfyUI and Stable Diffusion pipelines — for pitch development and personal R&D, with an eye toward production integration as the toolset matures.',
]

export default function About({ settings, years }: { settings: Settings; years: number }) {
  const defaultBio = DEFAULT_BIO(years)
  const bio = [
    settings?.bio || defaultBio[0],
    settings?.bio2 || defaultBio[1],
    settings?.bio3 || defaultBio[2],
    settings?.bio4 || defaultBio[3],
  ]

  return (
    <section id="about" className="section-pad" style={{ borderBottom: '0.5px solid var(--border)' }}>
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
          Background
        </div>
        <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em' }}>About & experience</h2>
      </div>

      <div className="grid-2col-gap">
        <div>
          {bio.map((p, i) => (
            <p key={i} style={{ fontSize: 15, color: 'var(--text-muted)', lineHeight: 1.8, marginBottom: 16 }}
              dangerouslySetInnerHTML={{ __html: p }} />
          ))}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 24 }}>
            {pills.map(p => (
              <span key={p.label} style={{
                fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase',
                border: `0.5px solid ${p.accent ? 'var(--border-teal)' : 'var(--border)'}`,
                padding: '6px 12px',
                color: p.accent ? 'var(--accent)' : 'var(--text-dim)',
              }}>{p.label}</span>
            ))}
          </div>
        </div>

        <div>
          {timeline.map((t, i) => (
            <div key={i} style={{
              display: 'flex', gap: 20, marginBottom: 28, paddingBottom: 28,
              borderBottom: i < timeline.length - 1 ? '0.5px solid var(--border-subtle)' : 'none',
            }}>
              <div style={{ fontSize: 11, letterSpacing: '0.1em', color: 'var(--accent)', minWidth: 52, paddingTop: 2 }}>{t.year}</div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)', marginBottom: 2 }}>{t.role}</div>
                <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>{t.studio}</div>
                {t.note && <div style={{ fontSize: 11, color: 'var(--text-ghost)', marginTop: 3, fontStyle: 'italic' }}>{t.note}</div>}
              </div>
            </div>
          ))}

          <a
            href={settings?.resumeUrl || '/matthew_bell_resume_2026.pdf'}
            download
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 8, marginTop: 28,
              padding: '10px 20px', border: '0.5px solid var(--border)',
              fontSize: 12, letterSpacing: '0.1em', textTransform: 'uppercase',
              color: 'var(--text-muted)', textDecoration: 'none',
            }}
          >
            ↓ Download full resume
          </a>
        </div>
      </div>
    </section>
  )
}
