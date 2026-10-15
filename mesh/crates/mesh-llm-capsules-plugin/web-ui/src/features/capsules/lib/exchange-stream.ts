// Time-ordering + session-rail derivation for the two-sided Ledger stream
//. Pure, so L-N/L-O are unit-testable
// without mounting a component.
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'

/**
 * NORMATIVE — L-N. "The ledger is one stream ordered by time. Nothing
 * reorders it — not session, not role, not outcome." Timestamp (falling
 * back to `exchangeKey` only as a stable tie-break, never role or outcome)
 * is the ONLY thing this function reads to order rows. Newest first.
 */
export function sortStreamByTime<T extends { timestamp: string | null; exchangeKey: string }>(rows: readonly T[]): T[] {
  return [...rows].sort((a, b) => {
    const at = a.timestamp ?? ''
    const bt = b.timestamp ?? ''
    if (at !== bt) return at < bt ? 1 : -1
    return a.exchangeKey < b.exchangeKey ? 1 : -1
  })
}

export type RailSegment = {
  /** NORMATIVE — L-O. "Served rows carry no session, and that is a fact,
   *  not a gap." A served row never carries a rail, full stop, regardless
   *  of whether the record happens to carry a `sessionId`. */
  hasRail: boolean
  /** True when this row opens a new rail segment (its session differs from
   *  the immediately preceding rail-bearing row, or there was none). */
  isSegmentStart: boolean
}

/**
 * Builds one `RailSegment` per row, in the SAME order the rows were given
 * -- callers must sort with `sortStreamByTime` first (this function does
 * not sort; sorting again here would risk a second, silently different,
 * sort key from L-N's).
 */
export function buildRailSegments(rows: readonly ExchangeLedgerRow[]): RailSegment[] {
  let previousSessionId: string | null = null
  return rows.map((row) => {
    const hasRail = row.roleTag === 'ASKED' && row.sessionId !== null
    const isSegmentStart = hasRail && row.sessionId !== previousSessionId
    previousSessionId = hasRail ? row.sessionId : null
    return { hasRail, isSegmentStart }
  })
}
