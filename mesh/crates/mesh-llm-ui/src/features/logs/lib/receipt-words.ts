import type { LogOutcome } from '@/features/logs/api/schemas'
import wording from '@/features/logs/lib/receipt-wording-en.json'

/**
 * Receipt presentation rules R1 to R8 (capsule-viewer docs/design/receipt-presentation.md) say
 * what words a view of a request's evidence uses. The sentences live in receipt-wording-en.json,
 * a copy of capsule-viewer's wording pack; no sentence is typed into a component.
 */
type WordingKey = keyof typeof wording.entries

export function receiptWord(key: WordingKey, values: Readonly<Record<string, string>> = {}): string {
  return wording.entries[key].replace(/\{(\w+)\}/g, (placeholder, name: string) => values[name] ?? placeholder)
}

/** R5: group thousands with a comma by hand, so the text cannot change with the reader's locale. */
export function groupThousands(value: number): string {
  if (!Number.isInteger(value)) return String(value)
  const grouped = Math.abs(value)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ',')
  return value < 0 ? `-${grouped}` : grouped
}

/** R3: a recorded token such as `attempt_failed` read as words; the caller keeps the token small beside it. */
export function tokenWords(token: string): string {
  const spaced = token.replace(/[_-]+/g, ' ').trim()
  return spaced.length === 0 ? token : spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

export type SelfReportedChecks = {
  readonly errorsRecorded: number | undefined
  readonly attemptCount: number | undefined
}

/** R1: who reported, what it checked with each result, and that this view did not re-check. */
export function selfReportedLine({ errorsRecorded, attemptCount }: SelfReportedChecks): string {
  const errors =
    errorsRecorded === undefined ? 'not available' : errorsRecorded === 0 ? 'none' : groupThousands(errorsRecorded)
  const retries = attemptCount === undefined ? 'not available' : groupThousands(Math.max(attemptCount - 1, 0))
  return receiptWord('verify.self_reported', {
    verifier: "this node's own request log",
    checks: `errors recorded: ${errors}; retries: ${retries}`
  })
}

/** The console's plain words for each recorded request outcome (R3); the token stays small beside them. */
export const outcomeLabels: Record<LogOutcome, string> = {
  active: 'Active',
  completed: 'Completed',
  failed: 'Failed',
  rejected: 'Rejected',
  cancelled: 'Cancelled',
  dropped: 'Dropped'
}
