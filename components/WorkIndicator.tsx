'use client'
import { useEffect, useRef } from 'react'

// Working indicator: a wireframe cube seen straight on, so at rest it reads as a square. Each beat it turns a
// quarter about X, then Y, then Z (a quarter turn maps the cube onto itself, so every beat starts from rest),
// with nearer edges drawn brighter. It draws in the element's text color, and holds still for visitors who
// ask for reduced motion. Shown only while something is actually working.
type Vec3 = [number, number, number]

const TURN = 0.5 // seconds per quarter turn
const HOLD = 0.26 // seconds at rest between turns
const AXES: Vec3[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]
const CUBE: Vec3[] = []
for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) CUBE.push([x, y, z])
const EDGES: [number, number][] = []
for (let a = 0; a < 8; a++)
  for (let b = a + 1; b < 8; b++)
    if (CUBE[a].filter((v, i) => v !== CUBE[b][i]).length === 1) EDGES.push([a, b])

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

// Rodrigues' rotation of p about the unit axis k.
function rotate(p: Vec3, k: Vec3, a: number): Vec3 {
  const c = Math.cos(a)
  const s = Math.sin(a)
  const d = (p[0] * k[0] + p[1] * k[1] + p[2] * k[2]) * (1 - c)
  return [
    p[0] * c + (k[1] * p[2] - k[2] * p[1]) * s + k[0] * d,
    p[1] * c + (k[2] * p[0] - k[0] * p[2]) * s + k[1] * d,
    p[2] * c + (k[0] * p[1] - k[1] * p[0]) * s + k[2] * d,
  ]
}

export default function WorkIndicator({ size = 12, className }: { size?: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    const g = canvas?.getContext('2d')
    if (!canvas || !g) return
    const dpr = Math.min(3, window.devicePixelRatio || 1)
    canvas.width = canvas.height = Math.round(size * dpr)
    const ink = getComputedStyle(canvas).color
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    // Half-size ending in .5, so at rest the square's 1px edges sit on pixel centers and stay crisp.
    const half = Math.floor(size * 0.3) + 0.5
    const center = size / 2
    let raf = 0
    let start = -1

    const draw = (now: number) => {
      if (start < 0) start = now
      const t = reduced ? 0 : Math.max(0, now - start) / 1000
      const beat = Math.floor(t / (TURN + HOLD))
      const f = ease(Math.min(1, (t - beat * (TURN + HOLD)) / TURN))
      const pts = CUBE.map(p => rotate(p, AXES[beat % 3], (Math.PI / 2) * f))

      g.setTransform(dpr, 0, 0, dpr, 0, 0)
      g.clearRect(0, 0, size, size)
      g.lineWidth = Math.max(1, size / 90)
      g.lineCap = 'round'
      g.strokeStyle = ink
      // Back edges first, so nearer (brighter) edges draw over them.
      const order = EDGES.map(([a, b]) => ({ a, b, z: (pts[a][2] + pts[b][2]) / 2 })).sort((p, q) => p.z - q.z)
      for (const { a, b, z } of order) {
        // Mid-turn depths reach sqrt 2; canvas ignores (rather than clamps) an alpha outside 0..1.
        g.globalAlpha = 0.35 + 0.65 * Math.min(1, Math.max(0, (z / Math.SQRT2 + 1) / 2))
        g.beginPath()
        g.moveTo(center + pts[a][0] * half, center - pts[a][1] * half)
        g.lineTo(center + pts[b][0] * half, center - pts[b][1] * half)
        g.stroke()
      }
      g.globalAlpha = 1
      if (!reduced) raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [size])

  return (
    <canvas
      ref={ref}
      className={className}
      aria-hidden="true"
      style={{ width: size, height: size, display: 'inline-block', verticalAlign: 'middle', flexShrink: 0 }}
    />
  )
}
