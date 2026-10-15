// Sticky day headers over the Exchanges stream (design §3A:
// "Tuesday 23 Sep · 42 exchanges · 0 confirmed").
// Pure so the boundary/tally logic is unit-testable without mounting the
// page. Tallies are always computed over the FULL row set a caller passes
// (the filtered stream, never one page) -- a sticky header states the whole
// day's count, not a page fragment's, the same "never a page-scoped number
// presented as the full range" discipline `fullRangeLabel` already applies.
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'

export type DayTally = {
  label: string
  count: number
  confirmedCount: number
}

/** Calendar-day bucket key, UTC, from the row's own timestamp -- never the
 *  viewer's local timezone (would make the same row land in a different
 *  day's header depending on who's looking). `null`/unparseable timestamps
 *  bucket together under 'unknown' rather than each becoming a false lone
 *  "day". */
export function dayKeyForRow(row: Pick<ExchangeLedgerRow, 'timestamp'>): string {
  return row.timestamp ? row.timestamp.slice(0, 10) : 'unknown'
}

function dayLabelFor(dayKey: string): string {
  if (dayKey === 'unknown') return 'Date unavailable'
  const parsed = new Date(`${dayKey}T00:00:00Z`)
  if (Number.isNaN(parsed.getTime())) return dayKey
  const weekday = parsed.toLocaleDateString('en-US', { timeZone: 'UTC', weekday: 'long' })
  const month = parsed.toLocaleDateString('en-US', { timeZone: 'UTC', month: 'short' })
  return `${weekday} ${parsed.getUTCDate()} ${month}`
}

/** One tally per calendar day found in `rows`. `confirmedCount` uses the
 *  same CLOSED predicate as the section's own headline count
 *  (`deriveRightCellState(row).kind === 'closed'`, already resolved onto
 *  `row.rightCellState` by the time a row reaches this stream) -- never a
 *  second, drifting definition of "confirmed". */
export function dayTalliesByKey(rows: readonly ExchangeLedgerRow[]): Map<string, DayTally> {
  const tallies = new Map<string, DayTally>()
  for (const row of rows) {
    const key = dayKeyForRow(row)
    const confirmed = row.rightCellState.kind === 'closed' ? 1 : 0
    const existing = tallies.get(key)
    if (existing) {
      existing.count += 1
      existing.confirmedCount += confirmed
    } else {
      tallies.set(key, { confirmedCount: confirmed, count: 1, label: dayLabelFor(key) })
    }
  }
  return tallies
}

/** The set of indices (into `rows`, in the SAME order the stream renders
 *  them) where a new day header belongs -- the first row, plus every row
 *  whose day differs from the row immediately before it. Rows must already
 *  be time-ordered (`sortStreamByTime`); this function doesn't re-sort. */
export function dayHeaderIndices(rows: readonly ExchangeLedgerRow[]): ReadonlySet<number> {
  const indices = new Set<number>()
  let previousKey: string | null = null
  rows.forEach((row, index) => {
    const key = dayKeyForRow(row)
    if (key !== previousKey) {
      indices.add(index)
      previousKey = key
    }
  })
  return indices
}

/** The header's own rendered line -- vocabulary-ruled words only
 *  ("exchanges"/"confirmed"; never "balanced", the wallet's word). */
export function dayTallyLine(tally: DayTally): string {
  return `${tally.label} · ${tally.count} exchange${tally.count === 1 ? '' : 's'} · ${tally.confirmedCount} confirmed`
}
