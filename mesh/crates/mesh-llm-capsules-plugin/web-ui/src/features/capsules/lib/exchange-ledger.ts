// Pure view-model derivation for the Exchanges table
// (Part 3) — kept separate from the
// table/column rendering so the honesty-backbone invariants (Checks is
// exceptions-first and NEVER a count; Confirmed is the double-entry fact,
// not registration) are unit-testable without mounting a table.
import type { PaneBRow, PaneCRow } from '@/features/capsules/api/sidecarTypes'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import { askTarget } from '@/features/capsules/lib/ask-for-record'
import { deriveContentToggleState, type ContentToggleState } from '@/features/capsules/lib/exchange-content-state'
import {
  deriveRightCellState,
  ledgerStateFilterValue,
  type LedgerStateFilterValue,
  type RightCellState
} from '@/features/capsules/lib/exchange-row-state'
import { NINE_PROPERTY_LABELS } from '@/features/capsules/lib/nine-properties'
import { peerExchangeIds } from '@/features/capsules/lib/peer-exchange-timeline'
import { peerDisplayId } from '@/features/capsules/lib/peer-row-view'

export type ExchangeLedgerRow = {
  exchangeKey: string
  timestamp: string | null
  roleTag: string
  counterparty: string | null
  /** The double-entry differentiator (not registration, which is a range
   *  fact that belongs on Integrity, never per-row). Derived from the ONE
   *  gate (`rightCellState.kind === 'closed'`), never a second structural
   *  predicate -- the old `theirs.state !== 'absent' && !unilateral` read
   *  was a dormant CLOSED-ish overclaim the gate never authorized. */
  confirmed: boolean
  /** This node served the exchange and no other side is recorded or named
   *  anywhere (`isLocalOnly`): there is no record of theirs to wait for, so
   *  the row is LOCAL, never OPEN, and no open or unconfirmed count holds it. */
  localOnly: boolean
  hasIssue: boolean
  /** Exceptions-first, NEVER a count — "—" when clean, else the specific
   *  failing property name(s), or the pair-reconciliation fallback when
   *  `has_issue` is driven by that check rather than a named property. */
  checksText: string
  /** The right cell's one of eight states (v3 §2's six + not-held/
   *  not-given) -- see `exchange-row-state.ts` for the derivation and its
   *  honesty limits. */
  rightCellState: RightCellState
  /** Toggle ① content state (v3 §3) -- see `exchange-content-state.ts`. */
  contentToggleState: ContentToggleState
  /** L-O: null for every served row and for any row the record itself
   *  carries no session for -- never invented. */
  sessionId: string | null
  /** `null` for every row that isn't one half
   *  of a client-marked twin pair (the overwhelming majority today). See
   *  `twin-bracket.ts` for how two rows sharing a non-null id become one
   *  bracket. */
  twinBracketId: string | null
  raw: PaneCRow
}

/** Pane C rows carry no counterparty identity field at all (`mine`/
 *  `theirs` are state + capsule_id only) -- the only real, non-invented way
 *  to name a counterparty is the SAME join `livePeerExchangeSources` already
 *  uses in the other direction: a Pane B peer's own `pair.details`/
 *  `expand.pair_ledger` names the exchange_ids it reconciled. This inverts
 *  that lookup into exchange_id -> peer display id, reusing `peerExchangeIds`
 *  verbatim rather than re-deriving the join. A Pane B row with no
 *  counterparty identity (`peerDisplayId` returns `null`) contributes NO
 *  entries -- its exchange_ids stay unindexed, which `buildExchangeLedgerRows`
 *  below reads as `counterparty: null`, never a synthetic placeholder. */
export function buildExchangeCounterpartyIndex(paneBRows: readonly PaneBRow[]): Map<string, string> {
  const index = new Map<string, string>()
  // An id two peers both list (a twin pair: one prompt, two providers,
  // one shared request digest) names neither: it is dropped, never "the
  // last peer written wins".
  const claimedTwice = new Set<string>()
  for (const row of paneBRows) {
    const displayId = peerDisplayId(row)
    if (displayId === null) continue
    for (const exchangeId of peerExchangeIds(row)) {
      const already = index.get(exchangeId)
      if (already !== undefined && already !== displayId) claimedTwice.add(exchangeId)
      index.set(exchangeId, displayId)
    }
  }
  for (const exchangeId of claimedTwice) index.delete(exchangeId)
  return index
}

function checksTextFor(row: PaneCRow): string {
  if (!row.has_issue) return '—'
  const properties = row.properties ?? {}
  const failing = Object.entries(properties)
    .filter(([, cell]) => cell?.state === 'FAIL')
    .map(([key]) => NINE_PROPERTY_LABELS[key] ?? key.replace(/_/g, ' '))
  // `has_issue` can also be driven by the pair-reconciliation check, which
  // is not one of the nine named properties this row's `properties` object
  // carries -- name that honestly too, never silently blank when there IS
  // an exception to report.
  return failing.length > 0 ? failing.join(', ') : 'pair reconciliation'
}

/** The peer a CLOSED (or contradicted) row's own half attributes, when a
 *  provenance-carrying counterparty half was pushed in
 *: `theirs.received_from` is the door's
 *  recorded sender of that half (`capsule_panes_native.rs::theirs_sibling_cell`,
 *  sourced from `received-provenance.jsonl`). A row that cites the peer's half
 *  by digest must attribute the peer, never read "counterparty not recorded".
 *  `null` when no half was received -- never a fabricated identity. */
function pushedHalfCounterparty(row: PaneCRow): string | null {
  const receivedFrom = row.theirs.received_from
  return typeof receivedFrom === 'string' && receivedFrom.length > 0 ? receivedFrom : null
}

/** A row this node served for pay: it holds its own book as the provider.
 *  This relies on the host sending a provider's payment events only for a
 *  request another node paid for (a node does not pay itself), so another
 *  node asked for it, whether or not that node's id is recorded. Nothing on
 *  this page checks that. */
export function isServedForPay(row: PaneCRow): boolean {
  return row.role_tag === 'SERVED' && row.provider_settlement != null
}

/** A row this node served with no other side anywhere: no counterparty (from
 *  the row, Pane B or a pushed half), no evidence of theirs of any kind
 *  (`open_not_asked`), its own record names no node to ask, and it was not
 *  served for pay (a payer is another side, recorded or not). The same
 *  conditions under which the row already reads "Local — no other side". */
export function isLocalOnly(
  row: PaneCRow,
  counterparty: string | null,
  state: RightCellState,
  ownRecord: CapsuleRecord | Record<string, unknown> | null | undefined
): boolean {
  return (
    row.role_tag === 'SERVED' &&
    counterparty === null &&
    state.kind === 'open_not_asked' &&
    askTarget(ownRecord) === null &&
    !isServedForPay(row)
  )
}

/** The state filter's bucket for a row: a local-only row has its own,
 *  `local`, rather than counting as `open`. */
export type ExchangeStateFilterValue = LedgerStateFilterValue | 'local'

export function exchangeStateFilterValue(row: ExchangeLedgerRow): ExchangeStateFilterValue {
  return row.localOnly ? 'local' : ledgerStateFilterValue(row.rightCellState)
}

export function buildExchangeLedgerRows(
  rows: readonly PaneCRow[],
  counterpartyIndex: ReadonlyMap<string, string>,
  /** This node's own records by `capsule_id`, as the rows render them. */
  recordsById: ReadonlyMap<string, CapsuleRecord> = new Map()
): ExchangeLedgerRow[] {
  return rows.map((row) => {
    const rightCellState = deriveRightCellState(row)
    const counterparty = row.counterparty ?? counterpartyIndex.get(row.exchange_key) ?? pushedHalfCounterparty(row)
    const ownRecord = (row.mine.capsule_id ? recordsById.get(row.mine.capsule_id) : undefined) ?? row.mine.record
    return {
      exchangeKey: row.exchange_key,
      timestamp: row.timestamp,
      roleTag: row.role_tag,
      // The row's own `counterparty` first (the pane's per-record peer
      // attribution -- names the server THIS row routed to, with the SAME
      // row key as the Peers table, D4(a); a twin pair's two rows share an
      // exchange key but not a counterparty); then the Pane B
      // pair-reconciliation join; last, the sender the record-push door
      // recorded for a locally-held counterparty half (older payloads).
      counterparty,
      // Derived from the ONE gate -- `closed` is the only state where their
      // half is held, signed, and cites ours. The retired structural read
      // (`theirs.state !== 'absent' && !unilateral`) was a dormant second
      // CLOSED-ish predicate; nothing may re-grow one.
      confirmed: rightCellState.kind === 'closed',
      localOnly: isLocalOnly(row, counterparty, rightCellState, ownRecord),
      hasIssue: row.has_issue,
      checksText: checksTextFor(row),
      rightCellState,
      contentToggleState: deriveContentToggleState(row),
      // L-O -- a served row structurally has no session (this node was never
      // party to the requester's conversation), regardless of what the
      // record carries.
      sessionId: row.role_tag === 'ASKED' ? (row.session_id ?? null) : null,
      twinBracketId: row.twin_bracket_id ?? null,
      raw: row
    }
  })
}
