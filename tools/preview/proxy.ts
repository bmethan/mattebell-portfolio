import { NextResponse, type NextRequest } from 'next/server'

// The password gate for the private preview deployment (tools/preview/deploy.sh copies this file to the root
// of the upload; the site itself, and its production deploy from master, has no gate and never sees it).
//
// One shared password (SITE_PASSWORD in the preview project's environment variables), the same pattern as the
// other private previews: a small form sets a cookie, rather than Basic Auth, whose challenge loops in Chrome
// over Vercel's edge network. The cookie holds a hash of the password, not the password. Fails closed: with
// no password set, every request gets the gate with a note saying so. GATE_OPEN=1 opens it for a while.
const COOKIE_NAME = 'mb_preview'
const GATE_PATH = '/__gate'

async function token(password: string): Promise<string> {
  const bytes = new TextEncoder().encode(`mattebell-preview|${password}`)
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')
}

function sameText(a: string, b: string): boolean {
  if (a.length !== b.length) return false
  let diff = 0
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i)
  return diff === 0
}

// The site's own colors and the nav's lettering (globals.css, Nav.tsx), in the site's system font.
function gatePage(error?: string): Response {
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta name="robots" content="noindex, nofollow" />
<meta name="theme-color" content="#0a0a0b" />
<title>Matthew Bell, private preview</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-sans-serif, system-ui, sans-serif; -webkit-font-smoothing: antialiased; background: #0a0a0b; color: #e8e6e0; display: flex; align-items: center; justify-content: center; min-height: 100vh; min-height: 100dvh; margin: 0; padding: 16px; box-sizing: border-box; }
  form { background: #111114; padding: 32px 28px 28px; width: 100%; max-width: 340px; box-sizing: border-box; border: 0.5px solid #2a2a2e; }
  .brand { font-size: 13px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: #f0ede8; margin: 0 0 6px; }
  .brand span { color: #888780; }
  p { margin: 0 0 22px; font-size: 12px; letter-spacing: 0.06em; color: #888780; }
  input { width: 100%; box-sizing: border-box; padding: 11px 12px; border: 0.5px solid #2a2a2e; background: #0a0a0b; color: #f0ede8; margin-bottom: 14px; font: inherit; font-size: 16px; }
  input:focus { outline: none; border-color: #5DCAA5; }
  button { width: 100%; padding: 12px; border: 0; background: #5DCAA5; color: #04342C; font: inherit; font-size: 12px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; cursor: pointer; }
  .error { color: #e07a6e; font-size: 12px; margin: -8px 0 14px; }
</style>
</head>
<body>
  <form method="POST" action="${GATE_PATH}">
    <div class="brand">Matthew Bell<span>&nbsp;/&nbsp;VFX &amp; Creative Tech</span></div>
    <p>A private preview.</p>
    ${error ? `<div class="error">${error}</div>` : ''}
    <input type="password" name="password" placeholder="Password" autocomplete="current-password" autofocus />
    <button type="submit">Enter</button>
  </form>
</body>
</html>`
  return new Response(html, {
    status: 401,
    headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store', 'x-robots-tag': 'noindex, nofollow' },
  })
}

export async function proxy(request: NextRequest): Promise<Response> {
  // Temporary open switch: GATE_OPEN=1 in the preview project's environment bypasses the gate (a demo
  // window). Unset it and deploy again to lock it.
  if (process.env.GATE_OPEN === '1') return NextResponse.next()

  const password = process.env.SITE_PASSWORD
  if (!password) return gatePage('Not open yet: no password has been set for this preview.')
  const expected = await token(password)

  if (request.nextUrl.pathname === GATE_PATH && request.method === 'POST') {
    const form = await request.formData()
    const supplied = form.get('password')
    if (typeof supplied === 'string' && sameText(await token(supplied), expected)) {
      return new Response(null, {
        status: 303,
        headers: {
          Location: '/',
          'Set-Cookie': `${COOKIE_NAME}=${expected}; Path=/; Max-Age=2592000; HttpOnly; Secure; SameSite=Lax`,
        },
      })
    }
    return gatePage('That password is not right.')
  }

  if (sameText(request.cookies.get(COOKIE_NAME)?.value ?? '', expected)) return NextResponse.next()
  return gatePage()
}

// Everything, the site's static files included.
export const config = { matcher: '/:path*' }
