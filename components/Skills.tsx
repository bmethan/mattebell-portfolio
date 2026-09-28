const primary = [
  { label: 'Lighting', note: 'Primary specialty' },
  { label: 'Look Development', note: 'Primary specialty' },
  { label: 'Real-time / Unreal', note: 'Creative Technologist' },
  { label: 'VFX Consultation', note: 'Strategy & pipeline' },
]

const secondary = [
  'Modeling', 'Texturing', 'Fur / Hair Groom', 'Rigging / Animation',
  'FX / Simulations', 'Compositing', 'Environments', 'Generative AI / R&D',
]

const skillGroups = [
  {
    cat: 'Lighting & LookDev', desc: 'Core specialty',
    skills: [
      { name: 'Houdini / Karma', pct: 100 },
      { name: 'Arnold / RenderMan', pct: 100 },
      { name: 'V-Ray / Redshift', pct: 100 },
      { name: 'XGen / Yeti', pct: 90 },
      { name: 'Mari / Substance', pct: 95 },
    ],
    rnd: false,
  },
  {
    cat: 'Real-time & 3D', desc: 'Creative Technologist',
    skills: [
      { name: 'Unreal Engine', pct: 95 },
      { name: 'Maya', pct: 95 },
      { name: 'Blender / Cinema 4D / Modo', pct: 85 },
      { name: 'Unity', pct: 80 },
    ],
    rnd: false,
  },
  {
    cat: 'Compositing & FX', desc: 'Full-pipeline generalist',
    skills: [
      { name: 'Nuke', pct: 95 },
      { name: 'After Effects', pct: 90 },
      { name: 'DaVinci Resolve', pct: 85 },
    ],
    rnd: false,
  },
  {
    cat: 'Generative AI', desc: 'Creative Technologist, Active R&D',
    skills: [
      { name: 'Image', tools: 'FLUX.2 · Qwen-Image · Z-Image', pct: 100 },
      { name: 'Video', tools: 'Seedance 2 · Kling 3.0 · Veo 3.1 · LTX-2.5 · Wan', pct: 95 },
      { name: '3D', tools: 'TRELLIS 2 · Hunyuan 3D · Gaussian splats', pct: 90 },
      { name: 'Orchestration', tools: 'ComfyUI · LoRAs & control adapters · agents via MCP', pct: 100 },
    ],
    rnd: false,
    note: 'Strong node-based workflow fluency, carried directly from Houdini and Nuke. Used in pitch development and look exploration, with production integration actively developing.',
  },
]

export default function Skills() {
  return (
    <section id="skills" className="section-pad" style={{ borderBottom: '0.5px solid var(--border)', background: 'var(--bg-alt)' }}>
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
          Expertise
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
        </div>
        <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em' }}>Skills & tools</h2>
      </div>

      {/* Primary disciplines */}
      <div className="grid-4col" style={{ marginBottom: 1 }}>
        {primary.map(d => (
          <div key={d.label} style={{ background: 'var(--bg-teal)', padding: '16px 20px' }}>
            <div style={{ fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 4 }}>{d.label}</div>
            <div style={{ fontSize: 11, color: 'var(--text-ghost)' }}>{d.note}</div>
          </div>
        ))}
      </div>

      {/* Secondary disciplines */}
      <div className="grid-4col" style={{ marginBottom: 32 }}>
        {secondary.map(d => (
          <div key={d} style={{ background: 'var(--bg)', padding: '16px 20px' }}>
            <div style={{ fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-dim)' }}>{d}</div>
          </div>
        ))}
      </div>

      {/* Skill bars */}
      <div className="grid-2col">
        {skillGroups.map(group => (
          <div key={group.cat} style={{ background: 'var(--bg)', padding: '24px 28px' }}>
            <div style={{ fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--text-dim)', marginBottom: 4 }}>{group.cat}</div>
            <div style={{ fontSize: 11, color: 'var(--text-ghost)', marginBottom: 16, fontStyle: 'italic' }}>{group.desc}</div>
            {group.skills.map(s => (
              <div key={s.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, marginBottom: 'tools' in s ? 14 : 10 }}>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 13, color: 'var(--text-soft)' }}>{s.name}</div>
                  {'tools' in s && (
                    <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 3, lineHeight: 1.5 }}>{s.tools}</div>
                  )}
                </div>
                <div style={{ width: 80, height: 2, background: 'var(--border)', flexShrink: 0 }}>
                  <div style={{ width: `${s.pct}%`, height: 2, background: group.rnd ? 'var(--bar-dim)' : 'var(--accent)' }} />
                </div>
              </div>
            ))}
            {group.note && (
              <div style={{ fontSize: 11, color: 'var(--text-ghost)', marginTop: 16, fontStyle: 'italic', lineHeight: 1.6, paddingTop: 16, borderTop: '0.5px solid var(--border-subtle)' }}>
                {group.note}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}
