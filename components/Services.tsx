// Two practices, each in the vocabulary its buyers use, with its own anchor (#vfx, #creative-technology) so
// either audience can be sent straight to its lane. Every line is backed by the Skills and About copy.
const LANES = [
  {
    id: 'vfx',
    eyebrow: 'VFX',
    title: 'Lighting and look development',
    items: [
      'Asset lookdev, sequence and shot lighting, and lighting TD work for feature film, episodic and commercials',
      'Unreal Engine lighting and lookdev for real-time and virtual production',
      'Senior generalist across the pipeline, from environments and FX to final comp',
      'VFX consultation',
      'Texas-based, for productions hiring Texas crew under the state incentive',
    ],
  },
  {
    id: 'creative-technology',
    eyebrow: 'Creative technology',
    title: 'Generative AI and real-time R&D',
    items: [
      'Generative AI pipelines: ComfyUI workflows, LoRAs and control adapters, and image, video and 3D models, finished to VFX standard',
      'Look exploration and pitch development for studios, agencies and brands',
      'Real-time prototypes in Unreal Engine and Unity',
      'Pipeline tools and DCC automation, including agents via MCP',
      'Visualization, real-time pipeline and UI/UX consulting',
    ],
  },
]

export default function Services() {
  return (
    <section id="services" className="section-pad" style={{ borderBottom: '0.5px solid var(--border)' }}>
      <div style={{ marginBottom: 40 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
          Services
          <span style={{ display: 'inline-block', width: 16, height: 1, background: 'var(--accent)' }} />
        </div>
        <h2 style={{ fontSize: 26, fontWeight: 500, color: 'var(--text-bright)', letterSpacing: '-0.01em' }}>
          Two practices, one standard of finish
        </h2>
      </div>

      <div className="grid-2col">
        {LANES.map(lane => (
          <div key={lane.id} id={lane.id} className="anchor-target" style={{ background: 'var(--bg)', padding: '28px 28px 32px' }}>
            <div style={{ fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 6 }}>
              {lane.eyebrow}
            </div>
            <h3 style={{ fontSize: 18, fontWeight: 500, color: 'var(--text)', marginBottom: 20, letterSpacing: '-0.01em' }}>
              {lane.title}
            </h3>
            <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
              {lane.items.map(item => (
                <li key={item} style={{ display: 'flex', gap: 12, fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.6 }}>
                  <span aria-hidden="true" style={{ flexShrink: 0, width: 12, height: 1, background: 'var(--border-teal)', marginTop: 11 }} />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  )
}
