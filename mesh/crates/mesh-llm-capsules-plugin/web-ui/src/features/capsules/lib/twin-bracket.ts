// Pure view-model derivation for the TWIN bracket's COMPARISON block
// (v3 §5). Kept separate from `TwinBracket.tsx`
// so the honesty invariants below are unit-testable without mounting a
// component:
//   - the disclosure sentence's N is READ from the live configured rate
//     (a prop threaded from the wire payload), never a hardcoded constant;
//   - OBSERVE-ONLY (item 4): this module computes no verdict at all -- not
//     even a PASS/FAIL string -- because the underlying live dual-dispatch
//     that would justify one doesn't exist yet (see the Rust host's
//     `runtime::twin_sample` module doc). It surfaces only what the two
//     halves themselves recorded (parameters, raw response text for a real
//     diff), never an equality/inconclusive judgment call.
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import { parseVerdict, type ParsedVerdict } from '@/features/capsules/lib/adjudication-view'

/** The bracket's one statement of how the pair came to be: a client marked
 *  the two requests as one pair with the `x-mesh-twin-bracket` header. The
 *  host never sends a second copy of a request by itself, so nothing here
 *  says the comparison "ran automatically" or names a sampling rate. */
export const TWIN_PAIR_SENTENCE = 'A client marked these two requests as one pair (x-mesh-twin-bracket).'

/**
 * The COMPARISON block's parameters line -- "temp 0 · seed 1 · model
 * identity d41d… · settings KV F16/F16" (v3 §5) -- built from whatever
 * fields the bracket's rows actually carry. Any missing field is simply
 * skipped (never a fabricated placeholder); `null` when NEITHER row carries
 * any comparison data at all, so callers can omit the line entirely rather
 * than render an empty one.
 */
export function twinComparisonParametersLine(rows: readonly ExchangeLedgerRow[]): string | null {
  const comparison = rows.map((row) => row.raw.twin_comparison).find((value) => value != null)
  if (!comparison) return null

  const parts: string[] = []
  if (comparison.temperature !== undefined && comparison.temperature !== null) {
    parts.push(`temp ${comparison.temperature}`)
  }
  if (comparison.seed !== undefined && comparison.seed !== null) {
    parts.push(`seed ${comparison.seed}`)
  }
  if (comparison.model_identity_hash) {
    parts.push(`model identity ${comparison.model_identity_hash}`)
  }
  if (comparison.settings_label) {
    parts.push(`settings ${comparison.settings_label}`)
  }
  return parts.length > 0 ? parts.join(' · ') : null
}

/**
 * The raw response text held on each side of the bracket, for the ONE real
 * side-by-side diff this ledger can show (v3 §5: "Because you were the
 * requester to both, you hold both responses"). `[null, null]` (or a
 * partial pair) when either side hasn't populated `mine.text` -- never
 * invented content to diff against.
 */
export function twinResponseTexts(rows: readonly ExchangeLedgerRow[]): [string | null, string | null] {
  const [a, b] = rows
  return [a?.raw.mine.text ?? null, b?.raw.mine.text ?? null]
}

/** Whether the two providers gave the same answer, from the pane's twin
 *  facts (sealed answer-text digests compared, not a verdict). */
export type TwinAnswerState = 'same' | 'different' | 'not_compared'

export function twinAnswerState(rows: readonly ExchangeLedgerRow[]): TwinAnswerState {
  const facts = rows.map((row) => row.raw.twin?.same_answer)
  if (facts.length > 0 && facts.every((f) => f === true)) return 'same'
  if (facts.some((f) => f === false)) return 'different'
  return 'not_compared'
}

export const TWIN_ANSWER_LABEL: Record<TwinAnswerState, string> = {
  same: 'same answer',
  different: 'different answers',
  not_compared: 'not compared yet'
}

/** A referee's signed verdict for this pair, when one exists. */
export type TwinVerdict = ParsedVerdict | null

export function twinVerdict(rows: readonly ExchangeLedgerRow[]): { verdict: TwinVerdict; capsuleId: string | null } {
  for (const row of rows) {
    const verdict = parseVerdict(row.raw.twin?.verdict)
    if (verdict !== null) return { verdict, capsuleId: row.raw.twin?.verdict_capsule_id ?? null }
  }
  return { verdict: null, capsuleId: null }
}
