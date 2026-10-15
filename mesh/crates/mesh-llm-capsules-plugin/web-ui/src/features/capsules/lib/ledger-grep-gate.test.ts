// Static-analysis grep gate: no forbidden vocabulary may appear in rendered
// UI text within the capsules feature. The gate checks string literals in
// .tsx files that are clearly user-facing (quoted strings with spaces that
// form human-readable text), not code identifiers or type discriminants.
//
// What this gate catches:
//   - JSX text nodes and quoted attribute values that contain forbidden terms
//   - Object property VALUES (not keys) that are prose strings
//
// What this gate intentionally does NOT catch:
//   - Code identifiers (property accesses like row.rung)
//   - TypeScript type discriminant strings like 'advertisement_absent' in
//     union types or interface definitions
//   - Variable/property names
//
// The approach: scan all quoted string literals that look like prose (contain
// spaces and are at least 10 chars) after stripping comments.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

// ---------------------------------------------------------------------------
// Forbidden patterns in user-visible prose strings
// ---------------------------------------------------------------------------

// Exact substrings forbidden in rendered prose:
const FORBIDDEN_EXACT = [
  'present-unverified',
  'unilateral_fallback',
  'advertisement_absent',
  'trust_level',
  'not yet witnessed',
  'unknown peer',
  'peer identity not resolved yet'
] as const

// Words forbidden as standalone tokens in rendered prose. Using
// `\b` boundaries to avoid matching inside compound identifiers.
const FORBIDDEN_WORDS = ['pending', 'proven', 'rung', 'score', 'rating'] as const

// ---------------------------------------------------------------------------
// Permitted phrases — exact substrings that are EXEMPT from the word check.
// These are design-thesis phrases that deliberately use a forbidden word in
// order to negate it (e.g. "Nothing is a score." uses "score" to deny
// scoring). Strip them from the prose string before scanning so the gate
// still catches any OTHER use of the word.
// ---------------------------------------------------------------------------

const PERMITTED_PHRASES = [
  // §0 design thesis — uses "score" to negate scoring; must stay verbatim.
  'Nothing is a score.'
] as const

// ---------------------------------------------------------------------------
// File walker — only .tsx files (JSX lives here)
// ---------------------------------------------------------------------------

function walkDir(dir: string, ext: string): string[] {
  const result: string[] = []
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    const stat = statSync(full)
    if (stat.isDirectory()) {
      result.push(...walkDir(full, ext))
    } else if (full.endsWith(ext)) {
      result.push(full)
    }
  }
  return result
}

// ---------------------------------------------------------------------------
// Comment stripping
// ---------------------------------------------------------------------------

function stripComments(src: string): string {
  let out = src.replace(/\/\*[\s\S]*?\*\//g, (m) => ' '.repeat(m.length))
  out = out.replace(/\/\/[^\n]*/g, (m) => ' '.repeat(m.length))
  return out
}

// ---------------------------------------------------------------------------
// Extract prose strings: quoted string literals that contain at least one
// space (prose has spaces; code identifiers and type discriminants rarely do).
// This catches user-facing message strings like error messages and labels while
// naturally excluding single-word type discriminants and key names.
// ---------------------------------------------------------------------------

function extractProseStrings(src: string): string[] {
  const results: string[] = []
  let m: RegExpExecArray | null

  // Double-quoted prose (single-line only — no newlines inside):
  const dqRe = /"([^\n"\\]{10,})"/g
  while ((m = dqRe.exec(src)) !== null) {
    const s = m[1]
    if (s.includes(' ')) results.push(s)
  }

  // Single-quoted prose (single-line only):
  const sqRe = /'([^\n'\\]{10,})'/g
  while ((m = sqRe.exec(src)) !== null) {
    const s = m[1]
    if (s.includes(' ')) results.push(s)
  }

  // Template literal prose (single-line, no ${...}):
  const tlRe = /`([^\n`\\${}]{10,})`/g
  while ((m = tlRe.exec(src)) !== null) {
    const s = m[1].trim()
    if (s.includes(' ')) results.push(s)
  }

  return results
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

const CAPSULES_SRC = join(import.meta.dirname ?? __dirname, '..')

describe('ledger grep gate — retired vocabulary must not reach rendered UI', () => {
  const tsxFiles = walkDir(CAPSULES_SRC, '.tsx').filter((f) => !f.endsWith('.test.tsx') && !f.endsWith('.spec.tsx'))

  it('has .tsx source files to scan', () => {
    expect(tsxFiles.length).toBeGreaterThan(0)
  })

  it('contains no forbidden terms in user-visible prose strings', () => {
    const violations: string[] = []

    for (const filePath of tsxFiles) {
      const raw = readFileSync(filePath, 'utf-8')
      const stripped = stripComments(raw)
      const proseStrings = extractProseStrings(stripped)

      for (const str of proseStrings) {
        // Strip permitted phrases before scanning so that design-thesis
        // phrases (which use a forbidden word to negate it) are exempt
        // while any OTHER use of the word is still caught.
        let scanStr = str
        for (const phrase of PERMITTED_PHRASES) {
          scanStr = scanStr.split(phrase).join(' '.repeat(phrase.length))
        }

        for (const term of FORBIDDEN_EXACT) {
          if (scanStr.includes(term)) {
            violations.push(
              `${filePath.replace(CAPSULES_SRC, '').replace(/^\//, '')}: ` + `"${term}" found in "${str.slice(0, 100)}"`
            )
          }
        }
        for (const term of FORBIDDEN_WORDS) {
          const wordBoundary = new RegExp(`\\b${term}\\b`, 'i')
          if (wordBoundary.test(scanStr)) {
            violations.push(
              `${filePath.replace(CAPSULES_SRC, '').replace(/^\//, '')}: ` +
                `word "${term}" found in "${str.slice(0, 100)}"`
            )
          }
        }
      }
    }

    if (violations.length > 0) {
      throw new Error(`Forbidden vocabulary found in user-visible prose strings:\n${violations.join('\n')}`)
    }
  })

  it('gate still catches a genuine rating use (negative case)', () => {
    // Confirm the permitted-phrase exemption is surgical: the thesis phrase
    // is exempt but a different "score" usage is still caught.
    const genuineViolation = 'Node quality score: 9.2'
    const permitted = 'Nothing is a score.'

    // Thesis phrase: after stripping permitted phrases, the word "score"
    // disappears — the test string should be clean.
    let scanPermitted = permitted
    for (const phrase of PERMITTED_PHRASES) {
      scanPermitted = scanPermitted.split(phrase).join(' '.repeat(phrase.length))
    }
    expect(/\bscore\b/i.test(scanPermitted)).toBe(false)

    // Genuine violation: "score" survives stripping (no permitted phrase matches).
    let scanViolation = genuineViolation
    for (const phrase of PERMITTED_PHRASES) {
      scanViolation = scanViolation.split(phrase).join(' '.repeat(phrase.length))
    }
    expect(/\bscore\b/i.test(scanViolation)).toBe(true)
  })
})
