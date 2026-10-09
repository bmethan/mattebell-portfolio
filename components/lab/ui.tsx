'use client'
// Controls shared by the lab's modes (lookdev and pyro): sliders on the skill meters' track, pills, labelled
// groups, and a row of WAI-ARIA tabs.
import { useId, useRef } from 'react'

// A slider on the skill meters' track: --pct drives the teal fill (see .lab-range).
export function Range({
  min,
  max,
  step,
  value,
  label,
  onChange,
  disabled,
}: {
  min: number
  max: number
  step: number
  value: number
  label: string
  onChange: (v: number) => void
  disabled?: boolean
}) {
  return (
    <input
      type="range"
      className="lab-range"
      disabled={disabled}
      min={min}
      max={max}
      step={step}
      value={value}
      aria-label={label}
      style={{ '--pct': `${((value - min) / (max - min)) * 100}%` } as React.CSSProperties}
      onChange={e => onChange(+e.target.value)}
    />
  )
}

export function Pill({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" className={`lab-pill${on ? ' on' : ''}`} aria-pressed={on} onClick={onClick}>
      {children}
    </button>
  )
}

export function Group({ label, children }: { label: string; children: React.ReactNode }) {
  const id = useId()
  return (
    <div className="lab-group" role="group" aria-labelledby={id}>
      <span id={id} className="lab-label">
        {label}
      </span>
      {children}
    </div>
  )
}

// Tabs: the arrow keys, Home and End move between them. The panel they control has id `${id}-panel`.
export function Tabs<T extends string>({
  id,
  label,
  tabs,
  tab,
  onTab,
}: {
  id: string
  label: string
  tabs: readonly { id: T; label: string }[]
  tab: T
  onTab: (t: T) => void
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const onKeyDown = (e: React.KeyboardEvent, i: number) => {
    const n = tabs.length
    const j = e.key === 'ArrowRight' ? (i + 1) % n : e.key === 'ArrowLeft' ? (i + n - 1) % n : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : -1
    if (j < 0) return
    e.preventDefault()
    onTab(tabs[j].id)
    refs.current[j]?.focus()
  }
  return (
    <div className="lab-tabs" role="tablist" aria-label={label}>
      {tabs.map((t, i) => (
        <button
          key={t.id}
          ref={el => {
            refs.current[i] = el
          }}
          type="button"
          role="tab"
          id={`${id}-${t.id}`}
          aria-selected={tab === t.id}
          aria-controls={`${id}-panel`}
          tabIndex={tab === t.id ? 0 : -1}
          className={`lab-tab${tab === t.id ? ' on' : ''}`}
          onClick={() => onTab(t.id)}
          onKeyDown={e => onKeyDown(e, i)}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}
