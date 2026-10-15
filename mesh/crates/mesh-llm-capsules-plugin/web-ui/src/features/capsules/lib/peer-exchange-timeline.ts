// Phase 2 timeline data — per-exchange points for
// one peer, and the adjudication lookup that backs the "requested only"
// lane. Deliberately does NOT fabricate the two data feeds this codebase
// doesn't have yet: a peer's own exchanges with OTHER counterparties (no
// peer-fetch carrier exists -- same gap `peer_history_cell` documents) and
// per-event integrity markers like `changed_without_saying`/owner-binding
// on Pane B (those exist only on Pane A's own history view today). Both
// render as an honest "not yet checked" band via `theirChainSummary`
// rather than an invented mark or event.
import type { CapsuleRecord } from '@/features/capsules/api/types'
import type { PaneBRow, PaneCRow } from '@/features/capsules/api/sidecarTypes'

export type ExchangeDirection = 'requested' | 'served'
export type Reconciliation = 'verified' | 'failed' | 'missing' | 'unknown'

export type PeerExchangeSource = {
  exchangeId: string
  timestamp: string | null
  direction: ExchangeDirection
  mineCapsuleId: string | null
  theirsCapsuleId: string | null
}

export type AdjudicationDetail = {
  verdict: string
  marginTau: number | null
  refereeId: string | null
  adjudicationCapsuleId: string
}

export type PeerTimelinePoint = {
  exchangeId: string
  timestamp: string | null
  direction: ExchangeDirection
  reconciliation: Reconciliation
  mineCapsuleId: string | null
  theirsCapsuleId: string | null
  /** Only ever populated for `direction === 'requested'` -- the only side
   *  a twin can check (honesty backbone: the adjudication lane sits under
   *  requested points only). */
  adjudication: AdjudicationDetail | null
}

/** The exchange_ids this peer's `pair_cell` actually reconciled -- the one
 *  real, peer-scoped join key Pane B exposes (no `peer_id` field on Pane
 *  C rows, so this is how a peer's exchanges get picked out of the full
 *  exchange list). */
export function peerExchangeIds(row: PaneBRow): string[] {
  const ledger = row.expand?.pair_ledger ?? row.pair?.details ?? []
  return ledger.map((entry) => entry.exchange_id)
}

function normalizeReconciliation(state: string | undefined): Reconciliation {
  return state === 'verified' || state === 'failed' || state === 'missing' ? state : 'unknown'
}

/** Live mode: Pane C already lists every exchange with a real timestamp
 *  and role_tag (`ASKED` = you asked them = requested, `SERVED` = they
 *  asked you) -- filter it down to this peer's exchange_ids. */
export function livePeerExchangeSources(row: PaneBRow, paneCRows: readonly PaneCRow[]): PeerExchangeSource[] {
  const ids = new Set(peerExchangeIds(row))
  return paneCRows
    .filter((paneCRow) => ids.has(paneCRow.exchange_key))
    .map((paneCRow) => ({
      exchangeId: paneCRow.exchange_key,
      timestamp: paneCRow.timestamp,
      direction: paneCRow.role_tag === 'ASKED' ? ('requested' as const) : ('served' as const),
      mineCapsuleId: paneCRow.mine.capsule_id,
      theirsCapsuleId: paneCRow.theirs.capsule_id
    }))
}

/** Scans the local ledger for a sealed adjudication capsule naming
 *  *capsuleId* as either half (Phase 2 -- same
 *  detection `peer_accountability_tab._adjudications_about` uses
 *  server-side, re-derived client-side over the already-fetched raw
 *  ledger records so the drill-down can show verdict + margin_tau +
 *  referee identity). Returns `null` -- render NOT_CHECKED -- for both an
 *  exchange nobody adjudicated AND a no-verdict case (referee unreachable
 *  / owner absent): `seal_adjudication_capsule` returns `None` and seals
 *  nothing when there's no verdict, so real data cannot distinguish those
 *  two cases either. That's the honesty backbone's own rule (never a
 *  silent pass), not a UI simplification. */
export function findAdjudicationForCapsule(
  capsuleId: string | null,
  recordsById: ReadonlyMap<string, CapsuleRecord>
): AdjudicationDetail | null {
  if (!capsuleId) return null
  for (const record of recordsById.values()) {
    const adjudication = record.model_attestation?.compute_attestation?.adjudication
    if (!adjudication?.verdict) continue
    if (adjudication.half_a_capsule_id !== capsuleId && adjudication.half_b_capsule_id !== capsuleId) continue
    return {
      verdict: adjudication.verdict,
      marginTau: adjudication.margin_tau != null ? Number(adjudication.margin_tau) : null,
      refereeId: adjudication.referee_id ?? null,
      adjudicationCapsuleId: record.capsule_id ?? capsuleId
    }
  }
  return null
}

export function buildTimelinePoints(
  row: PaneBRow,
  sources: readonly PeerExchangeSource[],
  recordsById: ReadonlyMap<string, CapsuleRecord>
): PeerTimelinePoint[] {
  const reconciliationByExchangeId = new Map(
    (row.expand?.pair_ledger ?? row.pair?.details ?? []).map((entry) => [entry.exchange_id, entry.state])
  )
  return [...sources]
    .map((source) => ({
      exchangeId: source.exchangeId,
      timestamp: source.timestamp,
      direction: source.direction,
      reconciliation: normalizeReconciliation(reconciliationByExchangeId.get(source.exchangeId)),
      mineCapsuleId: source.mineCapsuleId,
      theirsCapsuleId: source.theirsCapsuleId,
      adjudication:
        source.direction === 'requested' ? findAdjudicationForCapsule(source.mineCapsuleId, recordsById) : null
    }))
    .sort((a, b) => (a.timestamp ?? '').localeCompare(b.timestamp ?? ''))
}
