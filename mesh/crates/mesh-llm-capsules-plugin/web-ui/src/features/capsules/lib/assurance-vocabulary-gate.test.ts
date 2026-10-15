// item 8: a grep-gate over rendered
// strings for the FULL retired-vocabulary list named in the ledger batch plan
// -- `pending`, `present-unverified`, `proven`, `rung`, `verified` (as a
// result), `score`, `rating`, `unilateral`, `bilateral`, `log_id`.
//
// This is intentionally a SEPARATE, narrower gate than
// `ledger-grep-gate.test.ts`, scoped to exactly the files this task owns
// (the assurance mapper/tone/labels/explanation and the chip components
// that render them), for two reasons:
//  1. `ledger-grep-gate.test.ts` only walks `.tsx` files. This task's new
//     prose lives partly in `.ts` (`security-checks-view.ts`,
//     `chip-explanation.ts`) -- those strings are invisible to that gate.
//  2. Three of the batch plan's words -- `verified`, `unilateral`,
//     `bilateral` -- have LEGITIMATE, currently-live uses elsewhere in this
//     feature outside this task's scope (e.g. `row.unilateral` the field,
//     ordinary prose like "recomputed and verified in your browser" in
//     `ExchangeInspector.tsx`, ordinary English "verified" in
//     `CapsuleCard.tsx`). A blanket file-wide ban on those three words
//     would fail on files this task does not touch and has no mandate to
//     rewrite. Scoping this gate to the files this task actually authored
//     lets it enforce the full list precisely where it applies, without
//     manufacturing false failures in other tickets' files. Flagged in the
//     DONE report for the batch integrator/PM to decide whether a
//     repo-wide version of this gate is wanted as a follow-on.
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { describe, expect, it } from 'vitest'

const LIB_DIR = dirname(import.meta.url.replace('file://', ''))
const CAPSULES_DIR = join(LIB_DIR, '..')

const OWNED_FILES = [
  join(LIB_DIR, 'assurance-tone.ts'),
  join(LIB_DIR, 'nine-properties.ts'),
  join(LIB_DIR, 'security-checks-view.ts'),
  join(LIB_DIR, 'chip-explanation.ts'),
  join(CAPSULES_DIR, 'components', 'SecurityChecksView.tsx'),
  join(CAPSULES_DIR, 'components', 'ChipExplanationPopover.tsx')
]

const FORBIDDEN_EXACT = ['present-unverified', 'log_id'] as const

const FORBIDDEN_WORDS = ['pending', 'proven', 'rung', 'score', 'rating', 'verified', 'unilateral', 'bilateral'] as const

function stripComments(src: string): string {
  let out = src.replace(/\/\*[\s\S]*?\*\//g, (m) => ' '.repeat(m.length))
  out = out.replace(/\/\/[^\n]*/g, (m) => ' '.repeat(m.length))
  return out
}

function extractProseStrings(src: string): string[] {
  const results: string[] = []
  let m: RegExpExecArray | null

  const dqRe = /"([^\n"\\]{4,})"/g
  while ((m = dqRe.exec(src)) !== null) results.push(m[1])

  const sqRe = /'([^\n'\\]{4,})'/g
  while ((m = sqRe.exec(src)) !== null) results.push(m[1])

  const tlRe = /`([^\n`\\${}]{4,})`/g
  while ((m = tlRe.exec(src)) !== null) results.push(m[1].trim())

  return results
}

describe('assurance-vocabulary-gate — retired vocabulary must not appear anywhere in this task’s owned files', () => {
  it('scans every owned file (sanity: the list itself is non-empty and every file exists)', () => {
    expect(OWNED_FILES.length).toBeGreaterThan(0)
    for (const f of OWNED_FILES) {
      expect(() => readFileSync(f, 'utf-8')).not.toThrow()
    }
  })

  it('contains no forbidden vocabulary, in string literals or comments', () => {
    const violations: string[] = []

    for (const filePath of OWNED_FILES) {
      const raw = readFileSync(filePath, 'utf-8')
      // Comments are stripped first, same as `ledger-grep-gate.test.ts`:
      // `assurance-tone.ts`'s own header comment legitimately discusses
      // the OLDER Pane A/B ladder vocabulary this task does not touch
      // (`verified`/`present-unverified`/...) -- that's documentation
      // about a different subsystem, not a rendered string this task
      // introduced.
      const proseStrings = extractProseStrings(stripComments(raw))

      for (const str of proseStrings) {
        for (const term of FORBIDDEN_EXACT) {
          if (str.includes(term)) {
            violations.push(`${filePath}: exact "${term}" found in "${str.slice(0, 100)}"`)
          }
        }
        for (const term of FORBIDDEN_WORDS) {
          const wordBoundary = new RegExp(`\\b${term}\\b`, 'i')
          if (wordBoundary.test(str)) {
            violations.push(`${filePath}: word "${term}" found in "${str.slice(0, 100)}"`)
          }
        }
      }
    }

    if (violations.length > 0) {
      throw new Error(`Forbidden vocabulary found:\n${violations.join('\n')}`)
    }
  })

  it('the five rendered results are exactly the manifesto lowercase set (no shouty caps, no retired synonym)', async () => {
    const { CHIP_LABEL } = await import('@/features/capsules/lib/assurance-tone')
    expect(Object.values(CHIP_LABEL).sort()).toEqual(
      ['established', 'failed', 'inconclusive', 'not checked', 'not present'].sort()
    )
  })
})
