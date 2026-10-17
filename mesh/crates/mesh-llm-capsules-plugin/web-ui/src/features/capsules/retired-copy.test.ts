// Copy pin: wording the Evidence surface
// retired must never come back. Two stale-base branches merged into the freeze
// line once and quietly re-imported it ("Ask a peer" went 0 -> 5), so this
// scans the SOURCE of every shipped Evidence file rather than trusting that
// each component test happens to render the right state. Comments are
// stripped first: history notes may quote a retired phrase; code and copy may
// not.
//
// In the plugin bundle this scans every non-test source the bundle ships
// (`src/`, vendored console primitives included). The fork's copy of this test
// also scanned the host's Rust pane (`capsule_panes_native.rs`); that pane is
// still host code on the fork, so its scan stays there and moves here with the
// pane.
import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { STANDING_WORD } from '@/features/capsules/lib/banned-terms'

const RETIRED_PHRASES = [
  // -> "Get the other side’s record" (7bc96e8a4)
  'Ask a peer',
  // -> "not held" (7bc96e8a4: the banned word "pending" left the on-screen state)
  'pending fetch',
  // retired by design: a calm verdict over records no one else confirmed
  'Nothing needs your attention',
  // the duplicate headline; the ruled vocabulary is "confirmed by the other side"
  'confirmed by anyone else',
  // -> "Their record hasn’t arrived yet."
  'Their capsule id: not given',
  // UX §8 -- rule 4: a banned word stays off
  // the screen even to deny it; rule 3: plain words; rule 5: never ask for
  // what isn't needed.
  `not a ${STANDING_WORD} signal`,
  STANDING_WORD,
  'judgement of the peer',
  'Nothing is a score',
  'a score',
  'xpand checks to fetch it', // both "Expand…" and "expand…"
  'Corroboration cannot come from you',
  'Their halves appear here',
  'cites your half by digest',
  'Ask them for their half',
  'Get the other side’s half',
  'covered by checkpoint (',
  // + look findings (2026-09-26): one
  // term per thing, plain words, no designer notation on the face.
  'Closed by the other side', // -> "Confirmed by the other side"
  'recomputed in browser', // -> "recomputed here"
  'none sealed', // Disputes judged -> "none"
  '{ yours ', // bracket strip -> "Yours ● sealed —— Theirs ● same"
  '✓ same request and answer as yours', // -> "✓ They recorded the same request and answer"
  'bound (self-asserted)', // -> "linked (self-asserted)" + what it established
  'serve-boundary path', // -> "the plugin at this node’s serving boundary"
  'Registration is a separate step', // said once, in step 1
  // 
  'no key bound', // records ARE signed by the node key -> "not linked to an owner"
  'open in Logs', // an inert control; returns only when it can link
  // The dead pill -> the
  // `Your records` button that opens the panel
  "This node's copy",
  'This node’s copy',
  // Design copy: plain words, no protocol terms on the
  // face -> "in a checkpoint" / "a witness holds it"; times in the viewer's
  // own zone, never a raw ISO stamp.
  'not registered',
  'Registered with',
  'registered no later than',
  'Register your checkpoints',
  'inclusion proof',
  'nothing here is registered',
  // Tooltip honesty: no claim the code doesn't back.
  'Everything here is checked on this machine',
  'checkable only by you',
  'The next one will be checked against it',
  'Register a checkpoint',
  'The response bytes hash to the digest',
  'disagree about the request or the answer'
] as const

const HERE = dirname(fileURLToPath(import.meta.url))
const EVIDENCE_UI_ROOT = resolve(HERE, '../..')
/** The plugin's Rust panes: what the page shows comes from here too. */
const PANE_SOURCES = [
  '../../src/evidence_panes.rs',
  '../../src/evidence_panes/settlement.rs',
].map((path) => resolve(EVIDENCE_UI_ROOT, path))

function shippedSources(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) return shippedSources(path)
    return /\.(ts|tsx)$/.test(entry.name) && !/\.test\.(ts|tsx)$/.test(entry.name) ? [path] : []
  })
}

/** Drop block and line comments; keep `://` inside strings (URLs). */
function withoutComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')
}

/** A Rust file's shipped part: everything before its test module. (A lone
 *  `#[cfg(test)]` on a helper is not the test module, so it doesn't cut.) */
function rustShippedPart(source: string): string {
  const tests = source.search(/#\[cfg\(test\)\]\s*mod \w+/)
  return tests === -1 ? source : source.slice(0, tests)
}

function offenders(files: Array<{ label: string; code: string }>): string[] {
  return files.flatMap(({ label, code }) =>
    RETIRED_PHRASES.filter((phrase) => code.includes(phrase)).map((phrase) => `${label}: "${phrase}"`)
  )
}

describe('retired Evidence copy stays retired', () => {
  it('scans a real, non-empty set of shipped Evidence sources', () => {
    const sources = shippedSources(EVIDENCE_UI_ROOT)
    expect(sources.length).toBeGreaterThan(20)
    expect(sources.some((path) => path.endsWith('register-mesh-plugin-ui.tsx'))).toBe(true)
    expect(sources.some((path) => path.endsWith('LedgerPage.tsx'))).toBe(true)
  })

  it('no shipped Evidence UI source carries a retired phrase', () => {
    const files = shippedSources(EVIDENCE_UI_ROOT).map((path) => ({
      label: relative(EVIDENCE_UI_ROOT, path),
      code: withoutComments(readFileSync(path, 'utf8'))
    }))
    expect(offenders(files)).toEqual([])
  })

  it('the Rust panes that feed the page carry no retired phrase', () => {
    const files = PANE_SOURCES.map((path) => ({
      label: relative(EVIDENCE_UI_ROOT, path),
      code: withoutComments(rustShippedPart(readFileSync(path, 'utf8')))
    }))
    // The shipped part is real code, not an empty cut.
    expect(files[0].code).toContain('fn build_pane_c_list_with_settlements')
    expect(files[1].code).toContain('fn payer_book')
    expect(offenders(files)).toEqual([])
  })
})
