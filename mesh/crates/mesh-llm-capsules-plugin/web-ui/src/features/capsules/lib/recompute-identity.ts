// Shared in-browser recompute for `content_binding` (capsule_id) and
// `producer_signature`, factored out of `CapsuleCard.tsx`'s effect so the
// Ledger Panes section (build item 2) can apply
// the SAME rule to sidecar-served rows: "the live tab keeps recomputing
// capsule_id + signature itself and must never trust a chip it did not
// recompute for those two properties" -- a pane's own JSON `verify_ok` /
// `properties.content_binding` / `properties.producer_signature` are
// display hints at best; this hook is the one and only place either
// property's real PASS/FAIL is decided for the panes.
import { useEffect, useState } from 'react'
import { fetchSignedStatement } from '@/features/capsules/api/client'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import { decodeSignedStatementB64, fetchPeerLedgerCapsule } from '@/features/capsules/api/peerLedgerFetchClient'
import { isDigestShaped, recomputeCapsuleId } from '@/features/capsules/lib/canonical'
import { ed25519PublicKeyFromSpkiPem, verifyCoseSign1 } from '@/features/capsules/lib/cose'

export type RecomputedIdentity = {
  /** `null` while pending or when recompute could not run at all (e.g. no full record available). */
  idMatch: boolean | null
  signatureOk: boolean | null
}

const PENDING: RecomputedIdentity = { idMatch: null, signatureOk: null }

/** Shared by `recompute()` (which fetches `mine`'s statement from the LOCAL
 *  ledger route) and `recomputeFromFetchedBytes` (peer bytes the mesh
 *  already delivered) -- one rule for "does capsule_id match", not two. */
export async function recomputeIdMatch(
  record: Record<string, unknown>,
  claimedCapsuleId: string | null
): Promise<boolean | null> {
  try {
    const recomputed = await recomputeCapsuleId(record)
    return claimedCapsuleId ? recomputed === claimedCapsuleId : null
  } catch {
    return null
  }
}

/**
 * Recomputes `capsule_id` + verifies the COSE_Sign1 signature for bytes
 * fetched from elsewhere than the LOCAL ledger route -- the peer-fetch case
 * (`fetchPeerLedgerCapsule`). Additive: `mine`'s own call sites are
 * untouched and keep using `recompute()`/`useRecomputedIdentity` below,
 * which still always fetches from the local route.
 */
export async function recomputeFromFetchedBytes(
  record: Record<string, unknown>,
  claimedCapsuleId: string | null,
  nodePubKeyPem: string,
  statementBytes: Uint8Array
): Promise<RecomputedIdentity> {
  const idMatch = await recomputeIdMatch(record, claimedCapsuleId)
  try {
    const publicKey = ed25519PublicKeyFromSpkiPem(nodePubKeyPem)
    const verified = verifyCoseSign1(statementBytes, publicKey)
    return { idMatch, signatureOk: verified.verified }
  } catch {
    return { idMatch, signatureOk: null }
  }
}

async function recompute(record: CapsuleRecord, nodePubKeyPem: string | null): Promise<RecomputedIdentity> {
  const idMatch = await recomputeIdMatch(record, record.capsule_id ?? null)

  if (!nodePubKeyPem || !record.capsule_id) return { idMatch, signatureOk: null }

  try {
    const statementBytes = await fetchSignedStatement(record.capsule_id)
    if (!statementBytes) return { idMatch, signatureOk: null }
    const publicKey = ed25519PublicKeyFromSpkiPem(nodePubKeyPem)
    const verified = verifyCoseSign1(statementBytes, publicKey)
    return { idMatch, signatureOk: verified.verified }
  } catch {
    return { idMatch, signatureOk: null }
  }
}

/**
 * Recomputes `capsule_id` from `record` and, when `nodePubKeyPem` is known,
 * verifies the detached COSE_Sign1 signed statement for it. `record` may be
 * `null` when a pane row did not carry the full record this node needs to
 * recompute against (e.g. a counterparty's half in Pane C) -- that is
 * reported as `null`/`null`, never silently upgraded to a trusted PASS.
 *
 * While a new `record` is in flight, this returns `PENDING` (derived at
 * render time from the key mismatch below, never from a synchronous
 * `setState` inside the effect) rather than the previous record's result.
 */
export function useRecomputedIdentity(record: CapsuleRecord | null, nodePubKeyPem: string | null): RecomputedIdentity {
  const recordKey = record?.capsule_id ?? null
  const [resolved, setResolved] = useState<{
    key: string | null
    nodePubKeyPem: string | null
    result: RecomputedIdentity
  }>({
    key: null,
    nodePubKeyPem: null,
    result: PENDING
  })

  useEffect(() => {
    if (!record) return undefined
    let cancelled = false
    void recompute(record, nodePubKeyPem).then((result) => {
      if (!cancelled) setResolved({ key: recordKey, nodePubKeyPem, result })
    })
    return () => {
      cancelled = true
    }
  }, [record, recordKey, nodePubKeyPem])

  if (!record) return PENDING
  if (resolved.key !== recordKey || resolved.nodePubKeyPem !== nodePubKeyPem) return PENDING
  return resolved.result
}

/** `theirs.state === 'NOT_CHECKED'` plus a real join key
 *  (`capsule_id`+`peer_id`) is the only fetchable shape `theirs_cell`
 *  (`capsule_panes_native.rs`) ever emits -- see that function's own doc
 *  comment. Anything else (absent, or NOT_CHECKED with a hole in the join
 *  key) has nowhere to fetch FROM. Finding 5 (2026-09-23 assessment): a
 *  peer-asserted `capsule_id` that isn't digest-shaped (a chat-completion
 *  id like `capsule-chatcmpl-1790147257740`, forwarded as-is by an upstream
 *  peer) is not a fetchable capsule id either -- `mesh_ledger_fetch` would
 *  only ever answer `not_found` for it, so the fetch action is disabled
 *  rather than offered and failing. */
export function peerFetchJoinKey(row: PaneCRow): { capsuleId: string; peerId: string } | null {
  if (row.theirs.state !== 'NOT_CHECKED') return null
  const capsuleId = row.theirs.capsule_id
  const peerId = row.theirs.peer_id
  if (!capsuleId || !peerId) return null
  if (!isDigestShaped(capsuleId)) return null
  return { capsuleId, peerId }
}

export type PeerRecomputeStatus = 'not_fetched' | 'fetching' | 'found' | 'not_found' | 'error'

export type PeerRecomputeState = {
  status: PeerRecomputeStatus
  idMatch: boolean | null
  signatureOk: boolean | null
  errorMessage?: string
  /** The peer's actual capsule record, present only once `status ===
   *  'found'` -- carried through (never re-derived) so a caller can compare
   *  a REAL field of theirs (`effect.request_digest`, `model_attestation.
   *  model_id`, ...) against ours, instead of mirroring our own value and
   *  labelling it `theirs` (`security-checks-view.ts`'s honesty rule:
   *  `✓ same` only after a real comparison). `null` at every other status --
   *  never a stand-in for bytes this browser hasn't actually received. */
  peerRecord: Record<string, unknown> | null
  /** Never true merely because a fetch is in flight or queued -- only once
   *  a peer's bytes actually arrived and were recomputed against, matching
   *  this file's own `actuallyRecomputed` discipline used elsewhere
   *  (`security-checks-view.ts`). */
  fetch: () => void
}

const PEER_NOT_FETCHED: Omit<PeerRecomputeState, 'fetch'> = {
  status: 'not_fetched',
  idMatch: null,
  signatureOk: null,
  peerRecord: null
}

/**
 * Fetches and recomputes a Pane-C row's `theirs` half on demand -- never
 * automatically on mount (a peer fetch is a real mesh call, not a free
 * local read, per the design doc's "off the hot path" ruling). Call
 * `.fetch()` to trigger it; `status` reports the real outcome honestly,
 * including `not_found`/`error` -- neither is ever rendered as a pass.
 * Same "derived at render time from a key mismatch, never a synchronous
 * `setState` outside an effect/handler" discipline as `useRecomputedIdentity`
 * above: a stored result whose key no longer matches this row's join key
 * (a different row, or the join key disappeared) is discarded at render
 * time rather than mutated in place, so a stale peer's result can never
 * leak onto a different capsule's row.
 */
export function usePeerLedgerRecompute(row: PaneCRow): PeerRecomputeState {
  const joinKey = peerFetchJoinKey(row)
  const rowKey = joinKey ? `${joinKey.peerId}:${joinKey.capsuleId}` : null
  const [resolved, setResolved] = useState<{ key: string | null; result: Omit<PeerRecomputeState, 'fetch'> }>({
    key: null,
    result: PEER_NOT_FETCHED
  })

  function triggerFetch() {
    if (!joinKey) return
    const key = rowKey
    setResolved({ key, result: { status: 'fetching', idMatch: null, signatureOk: null, peerRecord: null } })
    void fetchPeerLedgerCapsule(joinKey.peerId, joinKey.capsuleId).then(async (outcome) => {
      if (key !== rowKey) return
      if (outcome.kind === 'not_found') {
        setResolved({ key, result: { status: 'not_found', idMatch: null, signatureOk: null, peerRecord: null } })
        return
      }
      if (outcome.kind === 'error' || outcome.kind === 'transport_error') {
        setResolved({
          key,
          result: { status: 'error', idMatch: null, signatureOk: null, peerRecord: null, errorMessage: outcome.message }
        })
        return
      }
      const statementBytes = decodeSignedStatementB64(outcome.signedStatementB64)
      const identity = await recomputeFromFetchedBytes(
        outcome.capsule,
        joinKey.capsuleId,
        outcome.nodePubKeyPem,
        statementBytes
      )
      setResolved({
        key,
        result: {
          status: 'found',
          idMatch: identity.idMatch,
          signatureOk: identity.signatureOk,
          peerRecord: outcome.capsule
        }
      })
    })
  }

  const current = resolved.key === rowKey ? resolved.result : PEER_NOT_FETCHED
  return { ...current, fetch: triggerFetch }
}
