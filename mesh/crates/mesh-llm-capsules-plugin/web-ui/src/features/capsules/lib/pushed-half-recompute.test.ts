// The per-row recompute for pushed counterparty halves, end-to-end through the
// ONE gate. Every `capsule_id` here is a real recompute of the body it names
// (`recomputeCapsuleId`), never a stub -- the gate's id check runs for real.
// Rows are shaped the way `capsule_panes_native.rs` sends them, and the gate is
// called the way the list-path callers call it: `deriveRightCellState(row)`,
// no recompute or local record passed in.
import { describe, expect, it } from 'vitest'
import type { PaneBJson, PaneBRow, PaneCListJson, PaneCRow } from '@/features/capsules/api/sidecarTypes'
import { recomputeCapsuleId } from '@/features/capsules/lib/canonical'
import { deriveRightCellState } from '@/features/capsules/lib/exchange-row-state'
import { confirmedByOtherSide } from '@/features/capsules/lib/peer-row-view'
import {
  attachPushedHalfRecomputeToPaneB,
  attachPushedHalfRecomputeToPaneC
} from '@/features/capsules/lib/pushed-half-recompute'

const REQUEST_DIGEST = 'a'.repeat(64)
const RESPONSE_DIGEST = 'b'.repeat(64)
const PROVIDER_NODE = 'c'.repeat(64)

function half(opts: {
  role: string
  response?: string
  servedBy?: string | null
  nonce?: string
}): Record<string, unknown> {
  const servingProvenance: Record<string, unknown> = { exchange_id: `exch-${opts.role}` }
  if (opts.servedBy !== null) servingProvenance.served_by_node_id = opts.servedBy ?? PROVIDER_NODE
  return {
    timestamp: '2026-09-26T00:00:00Z',
    action_id: `mesh-poc/${opts.role}/${opts.nonce ?? '1'}`,
    effect: { request_digest: REQUEST_DIGEST, response_digest: opts.response ?? RESPONSE_DIGEST },
    model_attestation: {
      compute_attestation: { 'x-mesh-poc-v1': { role: opts.role, serving_provenance: servingProvenance } }
    }
  }
}

async function sealed(body: Record<string, unknown>): Promise<Record<string, unknown>> {
  return { ...body, capsule_id: await recomputeCapsuleId(body) }
}

/** A correlated pushed pair exactly as the pane sends it: both bodies, the
 *  door's verdict off our citing record. */
async function pushedPairRow(
  opts: { theirs?: Record<string, unknown>; claimedId?: string; signatureOk?: boolean | undefined } = {}
): Promise<PaneCRow> {
  const mine = await sealed(half({ role: 'requested' }))
  const theirsBody = opts.theirs ?? (await sealed(half({ role: 'served' })))
  const theirs: PaneCRow['theirs'] = {
    state: 'present-unverified',
    capsule_id: opts.claimedId ?? (theirsBody.capsule_id as string),
    role: 'served',
    received_from: 'endpoint-node-a',
    via: 'push',
    record: theirsBody
  }
  if (!('signatureOk' in opts)) theirs.signature_ok = true
  else if (opts.signatureOk !== undefined) theirs.signature_ok = opts.signatureOk
  return {
    exchange_key: `digest:${REQUEST_DIGEST}`,
    role_tag: 'ASKED',
    header_state: 'absent',
    properties: null,
    has_issue: false,
    mine: { state: 'present-unverified', capsule_id: mine.capsule_id as string, role: 'requested', record: mine },
    theirs,
    unilateral: false,
    digest_match: { state: 'verified' },
    timestamp: '2026-09-26T00:00:00Z'
  }
}

function paneC(rows: PaneCRow[]): PaneCListJson {
  return {
    row_count: rows.length,
    default_sort: 'timestamp',
    filters: [],
    rows,
    next_after_seq: null,
    archived_segments: []
  }
}

async function gate(row: PaneCRow): Promise<string> {
  const [attached] = (await attachPushedHalfRecomputeToPaneC(paneC([row]))).rows
  return deriveRightCellState(attached).kind
}

describe('attachPushedHalfRecomputeToPaneC -- the per-row recompute', () => {
  it('records id_match true when the held body recomputes to its claimed capsule_id', async () => {
    const [row] = (await attachPushedHalfRecomputeToPaneC(paneC([await pushedPairRow()]))).rows
    expect(row.theirs.id_match).toBe(true)
  })

  it('records id_match false when the held body does not recompute to its claimed id', async () => {
    const [row] = (await attachPushedHalfRecomputeToPaneC(paneC([await pushedPairRow({ claimedId: 'f'.repeat(64) })])))
      .rows
    expect(row.theirs.id_match).toBe(false)
  })

  it('leaves a row with no pushed body untouched (no id_match invented)', async () => {
    const lone: PaneCRow = { ...(await pushedPairRow()), theirs: { state: 'absent', capsule_id: null } }
    const [row] = (await attachPushedHalfRecomputeToPaneC(paneC([lone]))).rows
    expect(row.theirs).toEqual({ state: 'absent', capsule_id: null })
  })
})

describe('the ONE gate on a pushed half (list-path call, verdict ready on first read)', () => {
  it('provider-signed, id recomputes, both digests equal ours, same provider -> CLOSED', async () => {
    expect(await gate(await pushedPairRow())).toBe('closed')
  })

  it('before the recompute has run (id_match absent) the row is never CLOSED', async () => {
    const row = await pushedPairRow()
    expect(deriveRightCellState(row).kind).toBe('open_not_held')
  })

  // Required mutant 1: no provenance -> OPEN.
  it('MUTANT no provenance -> OPEN: a held body with no door verdict on a citing record never closes', async () => {
    const kind = await gate(await pushedPairRow({ signatureOk: undefined }))
    expect(kind).not.toBe('closed')
    expect(kind).not.toBe('contradicted')
  })

  // Required mutant 2: self-sealed sibling -> OPEN. The pane files a sibling we
  // sealed ourselves under `mine` (it has no citing record), so `theirs` is
  // absent -- exactly `pane_c_self_sealed_sibling_never_closes_even_when_digests_match`.
  it('MUTANT self-sealed sibling -> OPEN: our own second half is never a counterparty half', async () => {
    const pair = await pushedPairRow()
    const selfSealed: PaneCRow = { ...pair, theirs: { state: 'absent', capsule_id: null }, unilateral: true }
    expect(await gate(selfSealed)).toBe('open_not_asked')
  })

  // Required mutant 3: provenance ok, digests differ -> CONTRADICTED.
  it('MUTANT provenance ok, digests differ -> CONTRADICTED', async () => {
    const theirs = await sealed(half({ role: 'served', response: 'e'.repeat(64) }))
    expect(await gate(await pushedPairRow({ theirs }))).toBe('contradicted')
  })

  it('a body that does not recompute to its claimed id -> CONTRADICTED, never CLOSED', async () => {
    expect(await gate(await pushedPairRow({ claimedId: 'f'.repeat(64) }))).toBe('contradicted')
  })

  it('a door verdict other than true -> not CLOSED', async () => {
    expect(await gate(await pushedPairRow({ signatureOk: false }))).toBe('open_not_held')
  })

  it('PROVISIONAL provider check (i): a body naming a different server -> not CLOSED', async () => {
    const theirs = await sealed(half({ role: 'served', servedBy: 'd'.repeat(64) }))
    expect(await gate(await pushedPairRow({ theirs }))).toBe('open_not_held')
  })

  it('PROVISIONAL provider check (i): a body naming no server -> not CLOSED', async () => {
    const theirs = await sealed(half({ role: 'served', servedBy: null }))
    expect(await gate(await pushedPairRow({ theirs }))).toBe('open_not_held')
  })
})

describe('attachPushedHalfRecomputeToPaneB -- Peers reads the same gate', () => {
  async function peerRow(pair: PaneCRow): Promise<PaneBRow> {
    return {
      peer_id: 'key:71eb777777777777',
      exchange_count: 1,
      confirmed_siblings: [{ mine: pair.mine, theirs: pair.theirs, digest_match: pair.digest_match }]
    } as unknown as PaneBRow
  }
  async function attached(row: PaneBRow): Promise<PaneBRow> {
    const payload: PaneBJson = { peer_count: 1, rows: [row] }
    return (await attachPushedHalfRecomputeToPaneB(payload)).rows[0]
  }

  it('a verified pushed half counts as confirmed by the other side', async () => {
    const row = await attached(await peerRow(await pushedPairRow()))
    expect(row.confirmed_siblings?.[0].theirs.id_match).toBe(true)
    expect(confirmedByOtherSide(row).confirmed).toBe(1)
  })

  it('a half whose digests differ is not counted as confirmed', async () => {
    const theirs = await sealed(half({ role: 'served', response: 'e'.repeat(64) }))
    const row = await attached(await peerRow(await pushedPairRow({ theirs })))
    expect(confirmedByOtherSide(row).confirmed).toBe(0)
  })
})
