// Client-side mirror of capsule-emit-mesh's `assurance_map.tone_for()`
//: the sidecar's three
// panes share ONE tone map so a state string -- whether the five-state
// property map (PASS/FAIL/NOT_PRESENT/NOT_CHECKED/INCONCLUSIVE) or the
// older per-rung ladder vocabulary Pane A/B still carry for values this task
// does not touch (verified/absent/present-unverified/failed/...) -- always
// resolves to the same colour here that it does in the Python-rendered
// panes. Never a second, drifting copy of the mapping's MEANING: this is a
// direct value-for-value port of assurance_map.py's CHIP_TONE + _LEGACY_TONE,
// kept in one place so updating one side is a reminder to update the other.
import type { StatusPillTone } from '@/components/ui/status-pill'

const CHIP_TONE: Record<string, StatusPillTone> = {
  PASS: 'good',
  FAIL: 'bad',
  NOT_PRESENT: 'neutral',
  NOT_CHECKED: 'neutral',
  INCONCLUSIVE: 'warn'
}

const LEGACY_TONE: Record<string, StatusPillTone> = {
  absent: 'neutral',
  unattested: 'neutral',
  pending: 'neutral',
  acknowledged_receipt: 'warn',
  self_measured: 'warn',
  os_measured: 'warn',
  'platform-attested': 'warn',
  unilateral: 'warn',
  verified: 'good',
  full_bilateral: 'good',
  tee_measured: 'good',
  attested: 'good',
  present: 'good',
  failed: 'bad',
  refused: 'bad',
  contradicted: 'bad'
}

/** Resolves any pane state string (map or legacy) to a `StatusPill` tone. */
export function toneForState(state: string | null | undefined): StatusPillTone {
  if (!state) return 'neutral'
  return CHIP_TONE[state] ?? LEGACY_TONE[state] ?? 'neutral'
}

export const CHIP_GLYPH: Record<string, string> = {
  PASS: '✓',
  FAIL: '✕',
  NOT_PRESENT: '∅',
  NOT_CHECKED: '…',
  INCONCLUSIVE: '?'
}

// Q1 (ruled 2026-09-11, two-sided ledger design §0): the five
// results render lowercase -- `established · failed · not present ·
// not checked · inconclusive` -- never the shouty-caps wire vocabulary
// (`PASS`/`FAIL`/...) the sidecar's `assurance_map.py` emits. The wire
// values stay as-is (`CHIP_TONE`/`CHIP_GLYPH` above keep keying on them --
// they're the sidecar's actual contract, not display text); this is the
// one place that turns a wire state into the word a person reads.
export const CHIP_LABEL: Record<string, string> = {
  PASS: 'established',
  FAIL: 'failed',
  NOT_PRESENT: 'not present',
  NOT_CHECKED: 'not checked',
  INCONCLUSIVE: 'inconclusive'
}

/** Resolves a wire state to its lowercase, manifesto-vocabulary label. */
export function labelForState(state: string | null | undefined): string {
  if (!state) return CHIP_LABEL.NOT_CHECKED
  return CHIP_LABEL[state] ?? state.toLowerCase().replace(/_/g, ' ')
}
