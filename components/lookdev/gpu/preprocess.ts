// WGSL has no preprocessor. The tracer's variants (meshes, glass, subsurface, fluorescence, the denoiser's
// images) switch code in and out exactly as the GLSL tracer does, with lines of the form
//   #ifdef NAME   #ifndef NAME   #if defined(A) && !defined(B) || defined(C)   #else   #endif
// resolved here against a set of defined names before the module is compiled.

function evalCondition(expr: string, defs: ReadonlySet<string>): boolean {
  // Only defined(NAME), !, && and || appear (no parentheses beyond defined()).
  return expr.split('||').some(term =>
    term.split('&&').every(factor => {
      const f = factor.trim()
      const neg = f.startsWith('!')
      const m = /defined\((\w+)\)/.exec(f)
      if (!m) throw new Error(`preprocess: cannot read "${f}"`)
      return defs.has(m[1]) !== neg
    }),
  )
}

export function preprocess(src: string, defines: Iterable<string>): string {
  const defs = new Set(defines)
  const out: string[] = []
  // One frame per open conditional: whether its enclosing code is live, and whether this branch is.
  const stack: { parentLive: boolean; live: boolean; taken: boolean }[] = []
  const live = () => (stack.length ? stack[stack.length - 1].live : true)
  for (const line of src.split('\n')) {
    const t = line.trim()
    let m: RegExpExecArray | null
    if ((m = /^#ifdef\s+(\w+)/.exec(t)) || (m = /^#ifndef\s+(\w+)/.exec(t))) {
      const want = t.startsWith('#ifdef') ? defs.has(m[1]) : !defs.has(m[1])
      const parentLive = live()
      stack.push({ parentLive, live: parentLive && want, taken: want })
    } else if ((m = /^#if\s+(.+)$/.exec(t))) {
      const want = evalCondition(m[1], defs)
      const parentLive = live()
      stack.push({ parentLive, live: parentLive && want, taken: want })
    } else if (t.startsWith('#else')) {
      const top = stack[stack.length - 1]
      if (!top) throw new Error('preprocess: #else without #if')
      top.live = top.parentLive && !top.taken
      top.taken = true
    } else if (t.startsWith('#endif')) {
      if (!stack.pop()) throw new Error('preprocess: #endif without #if')
    } else if (live()) {
      out.push(line)
    }
  }
  if (stack.length) throw new Error('preprocess: unclosed #if')
  return out.join('\n')
}
