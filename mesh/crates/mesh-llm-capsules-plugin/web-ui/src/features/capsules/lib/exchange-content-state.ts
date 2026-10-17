// Toggle ① content (v3 §3) -- the two-sided
// row's content cells. Content is asymmetric by architecture (L-F): the
// requester holds the payload, the provider streams and retains nothing. So
// exactly one side is ever "populated" and which side flips with `Your role`
// (`role_tag`).
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import { rightCellText } from '@/features/capsules/lib/exchange-row-state'

export type YourContentStateKind = 'populated' | 'populated_deleted' | 'streamed_not_retained'

export type TheirContentStateKind =
  | 'not_visible_holds' // Case B -- the requester holds it, architecturally guaranteed
  | 'recorded_absence' // signed statement of absence held
  | 'not_asked' // never asked
  | 'unanswered' // asked, no reply
  | 'signed_refusal' // not one of v3 §3's three named sub-states -- see theirContentText

export type YourContentState = { kind: YourContentStateKind; date: string | null }
export type TheirContentState = { kind: TheirContentStateKind; date: string | null }

export type ContentToggleState = {
  your: YourContentState
  their: TheirContentState
}

/**
 * A SERVED row's `mine` side is never populated -- the provider streamed and
 * retained nothing, full stop, regardless of what the record carries. An
 * ASKED row's `mine` side holds both halves of the conversation, unless the
 * reader has since deleted their own copy (L-D, `mine.deleted`).
 *
 * The `their` side is NEVER populated with content, in either case (L-F) --
 * `theirs.state === 'present'` (a CLOSED/CONTRADICTED right-cell state) says
 * only that they hold an OUTCOME record citing your digest (L-G), never that
 * they hold your content. So an ASKED row's `their` cell always renders one
 * of the absence/claim states below, driven by the same `evidence_outcome`
 * wire vocabulary as the right cell (`exchange-row-state.ts`) -- an absent
 * or `not_asked` outcome degrades to `not_asked` (L-C: we hold no ask-log,
 * never invent one).
 */
export function deriveContentToggleState(row: PaneCRow): ContentToggleState {
  if (row.role_tag === 'SERVED') {
    return {
      your: { kind: 'streamed_not_retained', date: null },
      their: { kind: 'not_visible_holds', date: null }
    }
  }

  const your: YourContentState = row.mine.deleted
    ? { kind: 'populated_deleted', date: row.mine.deleted_date ?? null }
    : { kind: 'populated', date: null }

  const outcome = row.theirs.evidence_outcome
  let theirKind: TheirContentStateKind
  if (outcome === 'recorded_absence') theirKind = 'recorded_absence'
  else if (outcome === 'unanswered') theirKind = 'unanswered'
  else if (outcome === 'signed_refusal') theirKind = 'signed_refusal'
  else theirKind = 'not_asked'

  return {
    your,
    their: { kind: theirKind, date: row.theirs.evidence_outcome_date ?? null }
  }
}

function dateOrFallback(date: string | null): string {
  return date ?? 'date unavailable'
}

/** The `your` cell's rendered sentence when it is NOT the real populated
 *  content -- `null` for `populated`, where the caller renders the actual
 *  quoted text/digest instead (this module has no opinion on that shape). */
export function yourContentFixedText(state: YourContentState): string | null {
  switch (state.kind) {
    case 'populated':
      return null
    case 'populated_deleted':
      return `Content deleted ${dateOrFallback(state.date)} · record still verifies`
    case 'streamed_not_retained':
      return 'No content. You streamed this response and did not retain it.'
    default: {
      const exhaustiveCheck: never = state.kind
      return exhaustiveCheck
    }
  }
}

/** NORMATIVE L-H -- render the epistemic difference. Never write this in the
 *  voice of `yourContentFixedText`: every branch here is either a stated
 *  claim ("they state...") or an architectural fact about visibility, never
 *  a fact WE know about what they hold. A provider that secretly retained
 *  would look identical to one that didn't -- we cannot prove either. */
export function theirContentText(state: TheirContentState): string {
  switch (state.kind) {
    case 'not_visible_holds':
      return 'Their content — not visible to you. The requester holds it.'
    case 'recorded_absence':
      return `They state they hold no payload for this exchange — signed ${dateOrFallback(state.date)}.`
    case 'not_asked':
      // With push the default, "not asked" is no longer why the comparison
      // didn't happen -- nothing arrived or we didn't fetch. "Not compared"
      // is true under both; the row state (OPEN · not held) already says why.
      return 'Not compared. They would be expected to hold none.'
    case 'unanswered':
      return `Asked ${dateOrFallback(state.date)}. No reply yet.`
    case 'signed_refusal':
      // Not one of v3 §3's three named sub-states (which cover the
      // never-asked-for-content path) -- reuses the already-reviewed
      // right-cell copy verbatim rather than inventing new content-specific
      // wording for a case the design note doesn't draw out.
      return rightCellText({ kind: 'open_refused', date: state.date })
    default: {
      const exhaustiveCheck: never = state.kind
      return exhaustiveCheck
    }
  }
}

/** The action control that lives inside the `their` cell when expanded --
 *  only the never-asked state has one (v3 §3's table). */
export function theirContentAction(state: TheirContentState): string | null {
  return state.kind === 'not_asked' ? 'Ask them to state it' : null
}
