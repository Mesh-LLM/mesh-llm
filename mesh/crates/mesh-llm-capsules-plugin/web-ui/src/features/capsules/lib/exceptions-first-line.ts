// The Exchanges header's exceptions-first line --
// leads with what needs attention, never a flat "N exchanges" count that
// buries a failure below it. Kept pure/testable, matching the discipline
// `checksTextFor` (exchange-ledger.ts) already uses for the row-level
// Checks column: NEVER a count when clean (em-dash there; "Nothing needs
// your attention" here), the specific failing tally when not.
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'

export type ExceptionsFirstTally = {
  total: number
  /** Rows carrying at least one of the three exceptions below -- a row
   *  counted once here even if it trips more than one category, so this
   *  never overstates how many records actually need a look. */
  needingAttention: number
  /** A property/pair-reconciliation check literally failed its recompute
   *  (`row.hasIssue`) -- see `checksTextFor`. */
  failed: number
  /** Both sides hold a signed record and they disagree
   *  (`rightCellState.kind === 'contradicted'`). */
  mismatched: number
  /** Asked for their half; no reply yet (`rightCellState.kind ===
   *  'open_asked'`) -- distinct from never having asked at all. */
  askedUnanswered: number
  /** A real fetched-and-verified artifact from the other side
   *  (`rightCellState.kind === 'closed'`) -- the one count this headline
   *  used to imply ("all recomputed clean") without ever actually stating,
   *  finding 7 of the 2026-09-23 assessment. */
  confirmedByAnyoneElse: number
}

export function exceptionsFirstTally(rows: readonly ExchangeLedgerRow[]): ExceptionsFirstTally {
  let failed = 0
  let mismatched = 0
  let askedUnanswered = 0
  let needingAttention = 0
  let confirmedByAnyoneElse = 0

  for (const row of rows) {
    const isFailed = row.hasIssue
    const isMismatched = row.rightCellState.kind === 'contradicted'
    const isAskedUnanswered = row.rightCellState.kind === 'open_asked'
    if (isFailed) failed += 1
    if (isMismatched) mismatched += 1
    if (isAskedUnanswered) askedUnanswered += 1
    if (row.rightCellState.kind === 'closed') confirmedByAnyoneElse += 1
    if (isFailed || isMismatched || isAskedUnanswered) needingAttention += 1
  }

  return { total: rows.length, needingAttention, failed, mismatched, askedUnanswered, confirmedByAnyoneElse }
}

/** NORMATIVE — same range discipline as `windowBannerHeadline`/
 *  `fullRangeLabel` (exchange-pages.ts, L-K): every count above the table
 *  states the range it covers, appended only when a row actually carries a
 *  timestamp, never fabricated.
 *
 * **Finding 7 (2026-09-23 assessment) -- corrected.** The zero-exceptions
 * branch used to read as an all-clear ("...all sealed, all
 * recomputed clean" over 135 rows with zero counterparties confirmed and
 * zero checkpoints registered -- true only of THIS node's own self-checks,
 * placed where it reads as a verdict on the whole exchange. The 10-second
 * rule (`ledger-ux-from-the-user`) wants the role-aware truth stated
 * up front instead: what this node sealed, what the other side actually
 * confirmed, and whether any of it is registered.
 *
 * The last two facts now live in the
 * Exchanges headline directly above this line ("You sealed N · M confirmed
 * by the other side · registered/not registered"), in the ruled vocabulary.
 * Repeating them here in the retired "... by anyone else" wording was the duplicate
 * headline, so this line carries only what it adds: the range and the
 * exception breakdown. */
export function exceptionsFirstLine(tally: ExceptionsFirstTally, rangeLabel: string | null): string {
  const { total, needingAttention, failed, mismatched, askedUnanswered } = tally
  const noun = total === 1 ? 'exchange' : 'exchanges'
  const rangeSuffix = rangeLabel ? `, ${rangeLabel}` : ''
  const breakdown = `${failed} failed · ${mismatched} mismatched · ${askedUnanswered} asked-and-unanswered`

  if (needingAttention === 0) {
    return `${total} sealed by you${rangeSuffix}.`
  }

  const needNoun = needingAttention === 1 ? 'exchange needs' : 'exchanges need'
  return `${needingAttention} ${needNoun} your attention — ${total} ${noun}${rangeSuffix}. ${breakdown}.`
}

/** UX §3 "One headline": `5 exchanges, 26 Sep · 3 confirmed by the other side
 *  · 0 disagreements`. Registration is an Integrity fact and is not repeated
 *  here. `disagreements` is the CONTRADICTED count, the same gate the rows use. */
export function exchangesHeadline(
  total: number,
  confirmed: number,
  disagreements: number,
  rangeLabel: string | null
): string {
  const noun = total === 1 ? 'exchange' : 'exchanges'
  const rangeSuffix = rangeLabel ? `, ${rangeLabel}` : ''
  const disagreementNoun = disagreements === 1 ? 'disagreement' : 'disagreements'
  return `${total} ${noun}${rangeSuffix} · ${confirmed} confirmed by the other side · ${disagreements} ${disagreementNoun}`
}
