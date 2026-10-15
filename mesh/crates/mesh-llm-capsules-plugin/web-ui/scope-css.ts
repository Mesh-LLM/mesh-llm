// Scopes the Evidence page's injected stylesheet to the page's own elements.
//
// The sheet is injected into the console's document, where a utility such as
// `.hidden { display: none }` would also match the console's own elements and,
// being last in the cascade, win over theirs (`<nav class="hidden md:flex">`).
// Every style rule's selector therefore also requires the element to carry the
// scope attribute or sit inside an element that does:
//
//   .hidden          ->  .hidden:where([data-capsules-ui], [data-capsules-ui] *)
//   .file\:x::file-selector-button
//                    ->  .file\:x:where(...)::file-selector-button
//
// The condition goes on the subject (the last compound), before any
// pseudo-element. `:where()` adds no specificity, so inside the page the cascade
// is unchanged. `@keyframes` and `@property` have no selectors and are left as
// they are. Nothing else in the sheet changes: only selector text is rewritten,
// in place.
import type { PluginOption } from 'vite'
import { SCOPE_ATTRIBUTE } from './src/lib/scope.ts'

export const IN_SCOPE = `:where([${SCOPE_ATTRIBUTE}], [${SCOPE_ATTRIBUTE}] *)`

/** Legacy single-colon spellings of pseudo-elements. */
const LEGACY_PSEUDO_ELEMENT = /^:(before|after|first-line|first-letter)(?![\w-])/i

/** The index of the last character of the escape starting at `text[i]` (a
 *  backslash): one character, or up to six hex digits and one optional
 *  whitespace character after them (`\32 xl`). */
function escapeEnd(text: string, i: number): number {
  const hex = /^[0-9a-fA-F]{1,6}\s?/.exec(text.slice(i + 1, i + 8))
  return hex ? i + hex[0].length : i + 1
}

/**
 * Walks `text` and calls `visit(i, c)` for each character outside strings,
 * comments and escapes, with the bracket depth (`(` and `[`) at that point.
 * `visit` may return an index to continue after.
 */
function scan(text: string, visit: (i: number, c: string, depth: number) => number | void): void {
  let depth = 0
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!
    if (c === '\\') {
      i = escapeEnd(text, i)
    } else if (c === '"' || c === "'") {
      let j = i + 1
      while (j < text.length && text[j] !== c) j = text[j] === '\\' ? escapeEnd(text, j) + 1 : j + 1
      i = j
    } else if (c === '/' && text[i + 1] === '*') {
      const end = text.indexOf('*/', i + 2)
      i = end < 0 ? text.length : end + 1
    } else {
      if (c === ')' || c === ']') depth--
      const next = visit(i, c, depth)
      if (c === '(' || c === '[') depth++
      if (typeof next === 'number') i = next
    }
  }
}

/** `selector` with the scope condition added to its subject. */
export function scopeSelector(selector: string): string {
  const leading = /^\s*/.exec(selector)![0]
  const trailing = /\s*$/.exec(selector)![0]
  const body = selector.slice(leading.length, selector.length - trailing.length)
  if (!body) throw new Error('scope-css: an empty selector')
  // Where the subject compound starts (after the last top-level combinator),
  // and its first top-level pseudo-element.
  let subject = 0
  let at = body.length
  scan(body, (i, c, depth) => {
    if (depth > 0) return
    if (/[\s>+~]/.test(c)) {
      subject = i + 1
      at = body.length
    } else if (c === ':') {
      const doubled = body[i + 1] === ':'
      if (at === body.length && (doubled || LEGACY_PSEUDO_ELEMENT.test(body.slice(i)))) at = i
      if (doubled) return i + 1
    }
  })
  if (subject >= body.length) throw new Error(`scope-css: a selector ends in a combinator: ${selector}`)
  return leading + body.slice(0, at) + IN_SCOPE + body.slice(at) + trailing
}

/** Rewrites every style rule's selector list in `css`; every other byte is
 *  copied as is. */
export function scopeCss(css: string): string {
  const pieces: string[] = []
  let copied = 0 // css[0, copied) is in `pieces`
  let prelude = 0 // where the text before the next `{` starts
  const blocks: Array<'style' | 'keyframes' | 'other'> = []
  scan(css, (i, c) => {
    if (c === ';' || c === '}') {
      if (c === '}') blocks.pop()
      prelude = i + 1
      return
    }
    if (c !== '{') return
    // A comment before the prelude stays as it is.
    const comment = css.lastIndexOf('*/', i)
    const start = comment >= prelude ? comment + 2 : prelude
    const head = css.slice(start, i).trim()
    const parent = blocks[blocks.length - 1]
    if (head.startsWith('@')) {
      blocks.push(/^@(-[a-z]+-)?keyframes\b/i.test(head) ? 'keyframes' : 'other')
    } else if (parent === 'keyframes') {
      blocks.push('other') // `from`, `to`, `50%`
    } else if (parent === 'style') {
      // Tailwind emits no nested style rules; fail rather than leave one unscoped.
      throw new Error(`scope-css: a nested style rule is not supported: ${head}`)
    } else {
      blocks.push('style')
      const selectors: string[] = []
      let from = start
      scan(css.slice(start, i), (j, ch, depth) => {
        if (depth === 0 && ch === ',') {
          selectors.push(css.slice(from, start + j))
          from = start + j + 1
        }
      })
      selectors.push(css.slice(from, i))
      pieces.push(css.slice(copied, start), selectors.map(scopeSelector).join(','))
      copied = i
    }
    prelude = i + 1
  })
  pieces.push(css.slice(copied))
  return pieces.join('')
}

/** Applies `scopeCss` to the page's stylesheet, after Tailwind has compiled it. */
export function scopeToMountRoot(): PluginOption {
  return {
    name: 'capsules:scope-to-mount-root',
    enforce: 'pre',
    transform(code, id) {
      return /\/src\/styles\/evidence\.css(\?|$)/.test(id) ? { code: scopeCss(code), map: null } : null
    }
  }
}
