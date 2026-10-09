'use client'
// The Lab section: a switch on its eyebrow line between the lookdev lab (the path tracer) and the pyro lab (the
// smoke solver). Each keeps its state while the other is shown; the hidden one stops drawing (its frame is off
// screen to its own visibility watch). The pyro lab mounts the first time it is chosen, or at once for a link
// with ?lab=pyro.
import { Fragment, useRef, useState, useSyncExternalStore } from 'react'
import LookdevLab from './LookdevLab'
import PyroLab from './PyroLab'

type LabMode = 'lookdev' | 'pyro'
const MODES: { id: LabMode; label: string }[] = [
  { id: 'lookdev', label: 'Lookdev lab' },
  { id: 'pyro', label: 'Pyro lab' },
]
const noSubscribe = () => () => {}

export default function Lab() {
  const linked = useSyncExternalStore(noSubscribe, () => new URLSearchParams(location.search).get('lab') === 'pyro', () => false)
  const [chosen, setChosen] = useState<LabMode | null>(null)
  const [pyroSeen, setPyroSeen] = useState(false)
  const mode: LabMode = chosen ?? (linked ? 'pyro' : 'lookdev')
  const pyroMounted = pyroSeen || mode === 'pyro'
  const refs = useRef<(HTMLButtonElement | null)[]>([])

  const pick = (m: LabMode) => {
    if (m === 'pyro' || mode === 'pyro') setPyroSeen(true)
    setChosen(m)
  }
  const onKeyDown = (e: React.KeyboardEvent, i: number) => {
    const n = MODES.length
    const j = e.key === 'ArrowRight' ? (i + 1) % n : e.key === 'ArrowLeft' ? (i + n - 1) % n : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : -1
    if (j < 0) return
    e.preventDefault()
    pick(MODES[j].id)
    refs.current[j]?.focus()
  }

  return (
    <section id="lab" className="section-pad" style={{ borderBottom: '0.5px solid var(--border)' }}>
      <div className="lab-switch" role="tablist" aria-label="Lab">
        <span className="lab-switch-dash" aria-hidden="true" />
        {MODES.map((m, i) => (
          <Fragment key={m.id}>
            {i > 0 && (
              <span className="lab-switch-sep" aria-hidden="true">
                ·
              </span>
            )}
            <button
              ref={el => {
                refs.current[i] = el
              }}
              type="button"
              role="tab"
              id={`lab-tab-${m.id}`}
              aria-selected={mode === m.id}
              aria-controls={`lab-panel-${m.id}`}
              tabIndex={mode === m.id ? 0 : -1}
              className={`lab-switch-tab${mode === m.id ? ' on' : ''}`}
              onClick={() => pick(m.id)}
              onKeyDown={e => onKeyDown(e, i)}
            >
              {m.label}
            </button>
          </Fragment>
        ))}
        <span className="lab-switch-dash" aria-hidden="true" />
      </div>
      <div id="lab-panel-lookdev" role="tabpanel" aria-labelledby="lab-tab-lookdev" style={mode === 'lookdev' ? undefined : { display: 'none' }}>
        <LookdevLab />
      </div>
      {pyroMounted && (
        <div id="lab-panel-pyro" role="tabpanel" aria-labelledby="lab-tab-pyro" style={mode === 'pyro' ? undefined : { display: 'none' }}>
          <PyroLab />
        </div>
      )}
    </section>
  )
}
