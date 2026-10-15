// The line a twin pair's row reads when no referee signed a verdict for it:
// "Not adjudicated: <why>". One line per reason the plugin gives
// (`referee::request`), and none of them counts against either machine.
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'

/** The pair's outcome as the plugin records it (`twin.referee_row`). */
export type RefereeRow =
  | { state: 'adjudicated'; verdict: string; referee: string; tier: number }
  | { state: 'not_adjudicated'; reason: string; because?: string }

const NOT_COMPARABLE_BECAUSE: Record<string, string> = {
  sampled: 'an answer was sampled (temperature above 0)',
  model_hash_differs: 'the two machines served different models',
  model_hash_unknown: 'a machine did not say which model it served',
  weights_differ: 'the two machines served different weights',
  weights_unknown: 'a machine did not say which weights it served'
}

/** Every reason the plugin gives, and its line. */
export const NOT_ADJUDICATED_LINES: Record<string, string> = {
  off: 'Not adjudicated: the independent check is turned off on this node.',
  host_does_not_mark_twins: 'Not adjudicated: no client marked these as a pair.',
  twins_agree: 'Not adjudicated: the two answers agree.',
  not_comparable: 'Not adjudicated: the two answers cannot be compared.',
  no_eligible_referee: 'Not adjudicated: no eligible referee.',
  referee_cannot_sign: 'Not adjudicated: the referee asked cannot sign a verdict.',
  referee_unreachable: 'Not adjudicated: the referee asked did not answer.'
}

/** A pair with no outcome recorded yet (its other half not held here). */
export const NOT_ADJUDICATED_YET = 'Not adjudicated yet.'

/** The line for a row that is not adjudicated; `null` for an adjudicated one. */
export function notAdjudicatedLine(row: RefereeRow | null | undefined): string | null {
  if (!row) return NOT_ADJUDICATED_YET
  if (row.state === 'adjudicated') return null
  if (row.reason === 'not_comparable' && row.because && NOT_COMPARABLE_BECAUSE[row.because]) {
    return `Not adjudicated: not comparable, ${NOT_COMPARABLE_BECAUSE[row.because]}.`
  }
  return NOT_ADJUDICATED_LINES[row.reason] ?? 'Not adjudicated.'
}

/** The pair's recorded outcome, from either of its rows. */
export function twinRefereeRow(rows: readonly ExchangeLedgerRow[]): RefereeRow | null {
  for (const row of rows) {
    const recorded = row.raw.twin?.referee_row
    if (recorded) return recorded
  }
  return null
}
