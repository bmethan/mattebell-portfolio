export default function StudioPage() {
  return (
    <div style={{ padding: 48, fontFamily: 'sans-serif', color: '#e8e6e0', background: '#0a0a0b', minHeight: '100vh' }}>
      <h1 style={{ fontSize: 24, marginBottom: 16 }}>Manage content at Sanity Studio</h1>
      <p style={{ color: '#888780', marginBottom: 24 }}>Run your local Sanity studio to manage content:</p>
      <code style={{ background: '#111', padding: '12px 16px', display: 'block', color: '#5DCAA5' }}>
        cd C:\Users\leahp\matthew-bell-studio && npx sanity dev
      </code>
    </div>
  )
}
