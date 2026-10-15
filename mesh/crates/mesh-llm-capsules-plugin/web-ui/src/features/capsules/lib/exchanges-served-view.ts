// Exchanges served / asked strip -- the
// quantity strip that sits above the Exchanges records table. Reads ONLY
// the fields `capsule_accountability_tab.build_served_summary_block` (which
// wraps `served_summary.build_served_summary`)
// actually emits on pane-a's `card.served_summary` -- never
// invents a number the fold doesn't produce. That fold counts SERVED
// EXCHANGES per model (`by_model[*].served`/`completed`/`failed`), not
// tokens, so this view labels them for what they are rather than borrowing
// "tokens" language the data doesn't support. There is no symmetric
// "asked" (this node AS REQUESTER) fold exposed yet -- rendered as an
// honest absence, the same discipline `asked_cell`'s "this node doesn't
// persist a send log yet" already uses on Pane B, never a fabricated zero.
import type { JsonRecord } from '@/features/capsules/api/types'

export type ServedAskedCoverage =
  | { kind: 'absent'; headline: string }
  | { kind: 'failed'; headline: string }
  | { kind: 'verified'; servedText: string; askedText: string; statement: string }

export const ASKED_ABSENT_TEXT = 'Asked of other nodes: not tracked by this fold yet.'

function asRecord(value: unknown): JsonRecord | null {
  return value !== null && typeof value === 'object' ? (value as JsonRecord) : null
}

function shortRoot(root: unknown): string {
  return typeof root === 'string' && root.length > 0 ? `${root.slice(0, 12)}…` : 'no root yet'
}

/** Derives the exchanges served / asked strip's coverage state from pane-a's raw `card`
 *  field. Total over any shape -- an absent/malformed block degrades to
 *  `'absent'`, never a thrown error or an invented number. */
export function servedAskedCoverage(card: JsonRecord | null | undefined): ServedAskedCoverage {
  const block = asRecord(card?.served_summary)
  const headline = typeof block?.text === 'string' ? block.text : 'No served-summary data available yet.'
  const state = typeof block?.state === 'string' ? block.state : 'NOT_CHECKED'

  // Never silently green: a summary that failed its own recompute+match
  // renders as failed, not folded into the same branch as a real count.
  if (state === 'FAIL' || state === 'failed') {
    return { kind: 'failed', headline }
  }

  const value = asRecord(block?.served_summary)
  if (!value || state === 'NOT_CHECKED' || state === 'absent') {
    return { kind: 'absent', headline }
  }

  const selection = asRecord(value.selection)
  const coverage = asRecord(value.coverage)
  const coveredEntries = typeof selection?.covered_entries === 'number' ? selection.covered_entries : null
  const mmrSize = typeof coverage?.mmr_size === 'number' ? coverage.mmr_size : null
  const witnesses = Array.isArray(coverage?.witnesses) ? (coverage.witnesses as unknown[]) : []
  const witnessed = coverage?.witnessed === true

  const rangeText = coveredEntries !== null ? `range-complete through entry ${coveredEntries}` : 'range not yet covered'
  const capturedText =
    mmrSize !== null
      ? `captured under the ${mmrSize}-leaf checkpoint ${shortRoot(coverage?.checkpoint_root)}`
      : // The coverage carried no mmr_size -- "not reported", never a false
        // "no checkpoint yet" (that would assert an absence the record does not
        // state). Same three-state honesty as the Integrity tab.
        'checkpoint coverage not reported'
  // Never fabricate a witness count when `witnessed` is false -- the
  // statement collapses to the honest "not reconciled" clause instead.
  const reconciledText = witnessed
    ? `reconciled against ${witnesses.length} witness${witnesses.length === 1 ? '' : 'es'}`
    : 'not reconciled'

  return {
    kind: 'verified',
    servedText: headline,
    askedText: ASKED_ABSENT_TEXT,
    statement: `${rangeText} · ${capturedText} · ${reconciledText}`
  }
}
