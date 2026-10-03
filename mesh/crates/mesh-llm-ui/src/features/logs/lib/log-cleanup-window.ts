import type { LogEventCategory, LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'

const HOUR_MS = 60 * 60 * 1000

export type CleanupWindow = {
  readonly start: number
  readonly end: number
}

/**
 * Browser dates retain milliseconds while durable log timestamps retain
 * nanoseconds. Use the next millisecond as an exclusive server cutoff so every
 * record represented by the selected final millisecond remains eligible.
 */
export function cleanupWindowExclusiveEnd(end: number): string {
  return new Date(end + 1).toISOString()
}

export function cleanupWindowBounds(
  rows: readonly LogEventLedgerRow[],
  from: string | undefined,
  to: string | undefined,
  now = Date.now()
): CleanupWindow {
  const times = rows.map((row) => Date.parse(row.occurredAt)).filter(Number.isFinite)
  const parsedFrom = from === undefined ? Number.NaN : Date.parse(from)
  const parsedTo = to === undefined ? Number.NaN : Date.parse(to)
  const start = Number.isFinite(parsedFrom) ? parsedFrom : times.length > 0 ? Math.min(...times) : now - 24 * HOUR_MS
  const candidateEnd = Number.isFinite(parsedTo) ? parsedTo : times.length > 0 ? Math.max(...times) : now
  return { start, end: candidateEnd > start ? candidateEnd : start + HOUR_MS }
}

export function rowsInCleanupWindow(
  rows: readonly LogEventLedgerRow[],
  window: CleanupWindow,
  categories: ReadonlySet<LogEventCategory>
) {
  return rows.filter((row) => {
    const occurredAt = Date.parse(row.occurredAt)
    return (
      Number.isFinite(occurredAt) &&
      occurredAt >= window.start &&
      occurredAt <= window.end &&
      categories.has(row.category)
    )
  })
}
