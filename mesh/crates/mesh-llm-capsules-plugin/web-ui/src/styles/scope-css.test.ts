import { transform } from 'lightningcss'
import { describe, expect, it } from 'vitest'
import evidenceCss from '@/styles/evidence.css?inline'
import { SCOPE_ATTRIBUTE } from '@/lib/scope'
import { IN_SCOPE, scopeCss, scopeSelector } from '../../scope-css'

describe('scopeSelector', () => {
  it.each([
    ['.hidden', `.hidden${IN_SCOPE}`],
    ['.a:hover', `.a:hover${IN_SCOPE}`],
    ['.file\\:x::file-selector-button', `.file\\:x${IN_SCOPE}::file-selector-button`],
    ['.before\\:x:before', `.before\\:x${IN_SCOPE}:before`],
    ['::backdrop', `${IN_SCOPE}::backdrop`],
    ['*', `*${IN_SCOPE}`],
    ['.\\[\\&_tr\\]\\:border-b tr', `.\\[\\&_tr\\]\\:border-b tr${IN_SCOPE}`],
    ['.a > .b + .c ~ .d', `.a > .b + .c ~ .d${IN_SCOPE}`],
    [':where(.space-y-1 > :not(:last-child))', `:where(.space-y-1 > :not(:last-child))${IN_SCOPE}`],
    ['.a[data-x="b c::d"]', `.a[data-x="b c::d"]${IN_SCOPE}`],
    // A hex escape's terminating space is part of the class name, not a combinator.
    ['.\\32 xl\\:p-1', `.\\32 xl\\:p-1${IN_SCOPE}`],
    ['  .a  ', `  .a${IN_SCOPE}  `]
  ])('%s', (selector, scoped) => {
    expect(scopeSelector(selector)).toBe(scoped)
  })
})

describe('scopeCss', () => {
  it('rewrites only selectors: at-rules, keyframes, declarations and comments are copied as they are', () => {
    const css = [
      '/*! a, b { } */',
      '@layer properties;',
      '@layer utilities {',
      '  .a, .b:hover { color: red; content: "x, y { }"; }',
      '  @media (hover: hover) { .c:hover { color: blue; } }',
      '}',
      '@property --tw-x { syntax: "*"; inherits: false; }',
      '@keyframes enter { from { opacity: 0; } 50% { opacity: .5; } to { opacity: 1; } }',
      '@layer properties { @supports (x: y) { *, ::before { --tw-x: 0; } } }'
    ].join('\n')
    const scoped = scopeCss(css)
    expect(scoped).toContain(`.a${IN_SCOPE}, .b:hover${IN_SCOPE} { color: red; content: "x, y { }"; }`)
    expect(scoped).toContain(`.c:hover${IN_SCOPE} { color: blue; }`)
    expect(scoped).toContain(`*${IN_SCOPE}, ${IN_SCOPE}::before { --tw-x: 0; }`)
    expect(scoped.split(IN_SCOPE).join('')).toBe(css)
  })

  it('refuses a nested style rule rather than leaving it unscoped', () => {
    expect(() => scopeCss('.a { color: red; .b { color: blue; } }')).toThrow(/nested style rule/)
  })
})

describe('the injected stylesheet', () => {
  it('scopes every style rule to the page: each selector, read by a CSS parser, requires the scope', () => {
    const subjects: string[] = []
    transform({
      filename: 'evidence.css',
      code: Buffer.from(evidenceCss),
      visitor: {
        Selector(selector) {
          // The subject compound: after the last combinator.
          let start = selector.length
          while (start > 0 && selector[start - 1]!.type !== 'combinator') start--
          const scoped = selector.slice(start).some(
            (part) =>
              part.type === 'pseudo-class' &&
              part.kind === 'where' &&
              JSON.stringify(part.selectors) ===
                JSON.stringify([
                  [{ type: 'attribute', namespace: null, name: SCOPE_ATTRIBUTE, operation: null }],
                  [
                    { type: 'attribute', namespace: null, name: SCOPE_ATTRIBUTE, operation: null },
                    { type: 'combinator', value: 'descendant' },
                    { type: 'universal' }
                  ]
                ])
          )
          subjects.push(scoped ? 'scoped' : JSON.stringify(selector))
        }
      }
    })
    expect(subjects.length).toBeGreaterThan(100)
    expect(subjects.filter((subject) => subject !== 'scoped')).toEqual([])
  })
})
