// Harness fixtures for the Peers tab (Phase 1) --
// so `pnpm dev` (:5173, harness data mode) renders a populated Peers
// column without a live sidecar/mesh. Two Pane B rows, matching the
// acceptance check: one clean peer, one peer carrying an alarm (a
// contradicted adjudication AND a failed chain-continuity check). Field
// shapes mirror `peer_accountability_tab.build_peer_row` on
// capsule-emit-mesh main verbatim -- see `sidecarTypes.ts`'s Pane B cell
// types. `PEER_TAB_HARNESS_MESH_PEERS` carries a THIRD peer with no
// matching Pane B row at all -- the Peers table's "advertised but
// unused" row group fixture.
import type { CapsuleRecord } from '@/features/capsules/api/types'
import type { ModelSummary, Peer } from '@/features/app-tabs/types'
import type { PaneBConfirmedSibling, PaneBJson, PaneBRow } from '@/features/capsules/api/sidecarTypes'
import type { PeerExchangeSource } from '@/features/capsules/lib/peer-exchange-timeline'
import { fixtureMineCell, fixtureTheirsCell } from '@/features/capsules/lib/pushed-half-fixtures'
import { LatencySource } from '@/lib/api/types'

const CLEAN_PEER_ID = 'node:aa11bb22cc33dd44'
const ALARMED_PEER_ID = 'node:ff99ee88dd77cc66'

/** A pushed counterparty half as `capsule_panes_native.rs` supplies it: both
 *  bodies, the door's verdict and the browser's id recompute. `verified`
 *  closes (clean); `failed` carries a differing digest and contradicts
 *  (mismatch). Same builders as the Pane C fixtures, so both panes render off
 *  the SAME gate. */
function confirmedSibling(state: 'verified' | 'failed', index: number): PaneBConfirmedSibling {
  return {
    mine: fixtureMineCell(`mine_half_${String(index).padStart(2, '0')}`),
    theirs: fixtureTheirsCell(state === 'verified' ? 'agrees' : 'disagrees', {
      capsuleId: `pushed_half_${String(index).padStart(2, '0')}`,
      receivedFrom: state === 'failed' ? ALARMED_PEER_ID : CLEAN_PEER_ID
    }),
    digest_match: { state }
  }
}

const CLEAN_PEER_ROW: PaneBRow = {
  peer_id: CLEAN_PEER_ID,
  node: {
    state: 'present',
    text: CLEAN_PEER_ID,
    peer_id: CLEAN_PEER_ID,
    member_kind: 'member',
    exchange_count: 24
  },
  rung: {
    state: 'present',
    text: 'full_bilateral',
    rung: 'full_bilateral',
    distinct_rungs: ['full_bilateral']
  },
  role: {
    state: 'present',
    text: 'both · 24 (16 you→them, 8 them→you)',
    role: 'both',
    you_to_them_count: 16,
    them_to_you_count: 8,
    exchange_count: 24
  },
  history: {
    state: 'verified',
    text: '5 bundle(s) verified, 41 checkpoint(s) (their count)',
    fetch_source: 'peer_evidence_client',
    history_summary: { verified_bundles: 5, checkpoint_count: 41 }
  },
  served: {
    state: 'NOT_CHECKED',
    text: "this node cannot fetch the peer's OWN served summary yet",
    source: 'self_derived'
  },
  pair: {
    state: 'verified',
    text: '16 pair(s) reconciled, 0 missing',
    verified: 16,
    failed: 0,
    missing: 0,
    details: Array.from({ length: 16 }, (_, i) => ({
      exchange_id: `exch-clean-${String(i).padStart(2, '0')}`,
      state: 'verified'
    }))
  },
  // 16 halves pushed in and closed through the ONE gate (the same 16 the
  // pair ledger reconciled) -- confirmed-by-other-side and MATCH both derive
  // from these, never a browser peer-fetch of the whole chain.
  confirmed_siblings: Array.from({ length: 16 }, (_, i) => confirmedSibling('verified', i)),
  verdicts: {
    state: 'present',
    text: '8 corroborated, 0 contradicted, 0 inconclusive (self-sealed)',
    tally: { corroborated: 8, contradicted: 0, inconclusive: 0 },
    source: 'self_sealed'
  },
  asked: { state: 'absent', text: "this node doesn't persist a send log yet", count: 0 },
  exchange_count: 24,
  first_seen: '2026-08-20T09:12:00Z',
  last_seen: '2026-09-08T16:58:05Z',
  expand: {
    pair_ledger: Array.from({ length: 16 }, (_, i) => ({
      exchange_id: `exch-clean-${String(i).padStart(2, '0')}`,
      state: 'verified'
    })),
    their_card: { state: 'NOT_CHECKED', text: "this node cannot fetch the peer's OWN history card yet" }
  }
}

const ALARMED_PEER_ROW: PaneBRow = {
  peer_id: ALARMED_PEER_ID,
  node: {
    state: 'present',
    text: ALARMED_PEER_ID,
    peer_id: ALARMED_PEER_ID,
    member_kind: 'member',
    exchange_count: 14
  },
  rung: {
    state: 'present',
    text: 'acknowledged_receipt',
    rung: 'acknowledged_receipt',
    distinct_rungs: ['acknowledged_receipt', 'full_bilateral']
  },
  role: {
    state: 'present',
    text: 'both · 14 (10 you→them, 4 them→you)',
    role: 'both',
    you_to_them_count: 10,
    them_to_you_count: 4,
    exchange_count: 14
  },
  history: {
    state: 'failed',
    text: 'fetch verification failed: chain diverged from the last checkpoint this node verified',
    mine_for_reference: {
      history: {
        state: 'verified',
        text: '9 checkpoint(s) since size 0',
        history_depth: 9,
        checkpoint_count: 9,
        cadence: {}
      },
      continuity: { state: 'failed', text: 'broken at mmr_size=32', unforked: false },
      witnessed: { state: 'absent', text: 'not witnessed', witnesses: [] }
    }
  },
  served: {
    state: 'NOT_CHECKED',
    text: "this node cannot fetch the peer's OWN served summary yet",
    source: 'self_derived'
  },
  pair: {
    state: 'failed',
    text: '1 pair(s) digest-mismatched, 9 reconciled',
    verified: 9,
    failed: 1,
    missing: 0,
    details: [
      { exchange_id: 'exch-alarm-07', state: 'failed' },
      // 'b' prefix -- distinct from the single called-out 'exch-alarm-07'
      // above, never colliding with it (a bare 00-08 range would repeat
      // '07' at i=7).
      ...Array.from({ length: 9 }, (_, i) => ({
        exchange_id: `exch-alarm-b${String(i).padStart(2, '0')}`,
        state: 'verified'
      }))
    ]
  },
  // 10 halves pushed in: 9 close clean through the gate, 1 disagrees on a
  // digest (the gate reads it CONTRADICTED -> a mismatch).
  confirmed_siblings: [
    confirmedSibling('failed', 0),
    ...Array.from({ length: 9 }, (_, i) => confirmedSibling('verified', i + 1))
  ],
  verdicts: {
    state: 'contradicted',
    text: '6 corroborated, 1 contradicted, 0 inconclusive (self-sealed)',
    tally: { corroborated: 6, contradicted: 1, inconclusive: 0 },
    adjudication_capsule_id: 'cap-alarmed-adjudication-0007',
    source: 'self_sealed'
  },
  asked: { state: 'absent', text: "this node doesn't persist a send log yet", count: 0 },
  exchange_count: 14,
  first_seen: '2026-08-25T11:40:00Z',
  last_seen: '2026-09-08T08:03:00Z',
  expand: {
    pair_ledger: [
      { exchange_id: 'exch-alarm-07', state: 'failed' },
      // 'b' prefix -- distinct from the single called-out 'exch-alarm-07'
      // above, never colliding with it (a bare 00-08 range would repeat
      // '07' at i=7).
      ...Array.from({ length: 9 }, (_, i) => ({
        exchange_id: `exch-alarm-b${String(i).padStart(2, '0')}`,
        state: 'verified'
      }))
    ],
    their_card: { state: 'NOT_CHECKED', text: "this node cannot fetch the peer's OWN history card yet" }
  }
}

export const HARNESS_PANE_B_PAYLOAD: PaneBJson = {
  peer_count: 2,
  default_sort: 'last_seen',
  rows: [CLEAN_PEER_ROW, ALARMED_PEER_ROW]
}

// Model names/quant/ctx match `dashboard-fixtures.ts`'s `MODELS` entries
// verbatim so the Chat harness catalog actually contains what "Route here"
// pre-selects -- otherwise the dropdown would fall back to Auto in harness
// mode even though the search param round-trips correctly.
export const PEER_TAB_HARNESS_MESH_PEERS: Peer[] = [
  {
    id: 'aa11bb22cc33dd44',
    hostname: 'clean-node.local',
    region: 'us-west',
    status: 'online',
    hostedModels: ['Qwen3.6-27B-UD'],
    sharePct: 12,
    latencyMs: 38,
    latencySource: LatencySource.DIRECT,
    loadPct: 22,
    shortId: 'aa11bb22'
  },
  {
    id: 'ff99ee88dd77cc66',
    hostname: 'alarmed-node.local',
    region: 'eu-central',
    status: 'online',
    hostedModels: ['Qwen3.6-35B-A3B-UD'],
    sharePct: 6,
    latencyMs: 145,
    latencySource: LatencySource.ESTIMATED,
    loadPct: 61,
    shortId: 'ff99ee88'
  },
  // no Pane B row matches this id anywhere:
  // the "Nodes advertised but unused" row group's one fixture. Mesh knows
  // about it (announced a model, is online) but this node has never
  // exchanged with it.
  {
    id: '1122334455667788',
    hostname: 'unused-node.local',
    region: 'ap-south',
    status: 'online',
    hostedModels: ['Qwen3.6-27B-UD'],
    sharePct: 4,
    latencyMs: 210,
    latencySource: LatencySource.ESTIMATED,
    loadPct: 9,
    shortId: '11223344'
  }
]

export const PEER_TAB_HARNESS_MESH_MODELS: ModelSummary[] = [
  {
    name: 'Qwen3.6-27B-UD',
    family: 'Qwen',
    size: '17.8 GB',
    context: '256k',
    status: 'warm',
    tags: [],
    quant: 'Q4_K_XL',
    ctxMaxK: 256
  },
  {
    name: 'Qwen3.6-35B-A3B-UD',
    family: 'Qwen',
    size: '22.1 GB',
    context: '256k',
    status: 'warm',
    tags: [],
    quant: 'Q4_K_XL',
    ctxMaxK: 256
  }
]

// ---------------------------------------------------------------------------
// Phase 2 -- per-exchange timeline fixtures. Sourced from the SAME
// exchange_ids the Phase 1 `pair.details`/`expand.pair_ledger` arrays
// above already name (never a separate invented set), with a mine-side
// capsule_id per exchange and two ledger records carrying a real
// `model_attestation.compute_attestation.adjudication` block -- one
// corroborated (clean peer), one contradicted, the latter's capsule_id
// matching `ALARMED_PEER_ROW.verdicts.adjudication_capsule_id` exactly so
// the inline alarm chip and the timeline drill-down cite the same record.
// Every other exchange has no matching ledger record on purpose: the
// acceptance check requires seeing a genuine NOT_CHECKED (no-verdict)
// case, not just the adjudicated ones.
// ---------------------------------------------------------------------------

function evenlySpacedTimestamps(startIso: string, endIso: string, count: number): string[] {
  const start = new Date(startIso).getTime()
  const end = new Date(endIso).getTime()
  if (count <= 1) return [startIso]
  const stepMs = (end - start) / (count - 1)
  return Array.from({ length: count }, (_, i) => new Date(start + stepMs * i).toISOString())
}

function mineCapsuleIdFor(exchangeId: string): string {
  return `mine_${exchangeId}`
}

function theirsCapsuleIdFor(exchangeId: string): string {
  return `theirs_${exchangeId}`
}

function buildExchangeSources(
  exchangeIds: readonly string[],
  timestamps: readonly string[],
  servedIndices: ReadonlySet<number>
): PeerExchangeSource[] {
  return exchangeIds.map((exchangeId, index) => ({
    exchangeId,
    timestamp: timestamps[index] ?? null,
    direction: servedIndices.has(index) ? 'served' : 'requested',
    mineCapsuleId: mineCapsuleIdFor(exchangeId),
    theirsCapsuleId: theirsCapsuleIdFor(exchangeId)
  }))
}

const CLEAN_EXCHANGE_IDS = Array.from({ length: 16 }, (_, i) => `exch-clean-${String(i).padStart(2, '0')}`)
const ALARMED_EXCHANGE_IDS = [
  'exch-alarm-07',
  ...Array.from({ length: 9 }, (_, i) => `exch-alarm-b${String(i).padStart(2, '0')}`)
]

export const PEER_TAB_HARNESS_EXCHANGE_SOURCES: Record<string, PeerExchangeSource[]> = {
  [CLEAN_PEER_ID]: buildExchangeSources(
    CLEAN_EXCHANGE_IDS,
    evenlySpacedTimestamps('2026-08-20T09:12:00Z', '2026-09-08T16:58:05Z', CLEAN_EXCHANGE_IDS.length),
    new Set([3, 6, 9, 12, 15]) // exch-clean-00 (index 0) stays "requested" -- it carries the corroborated verdict
  ),
  [ALARMED_PEER_ID]: buildExchangeSources(
    ALARMED_EXCHANGE_IDS,
    evenlySpacedTimestamps('2026-08-25T11:40:00Z', '2026-09-08T08:03:00Z', ALARMED_EXCHANGE_IDS.length),
    new Set([1, 4, 7]) // exch-alarm-07 (index 0) stays "requested" -- it carries the contradicted verdict
  )
}

const CLEAN_ADJUDICATED_EXCHANGE_ID = 'exch-clean-00'
const ALARMED_ADJUDICATED_EXCHANGE_ID = 'exch-alarm-07'

export const PEER_TAB_HARNESS_LEDGER_RECORDS: Map<string, CapsuleRecord> = new Map([
  [
    'cap-clean-adjudication-0001',
    {
      capsule_id: 'cap-clean-adjudication-0001',
      timestamp: PEER_TAB_HARNESS_EXCHANGE_SOURCES[CLEAN_PEER_ID][0]?.timestamp ?? undefined,
      model_attestation: {
        compute_attestation: {
          adjudication: {
            verdict: 'corroborated',
            margin: '0.96',
            margin_tau: '0.9',
            half_a_capsule_id: mineCapsuleIdFor(CLEAN_ADJUDICATED_EXCHANGE_ID),
            half_b_capsule_id: `twin_${CLEAN_ADJUDICATED_EXCHANGE_ID}`,
            referee_id: 'local-twin'
          }
        }
      }
    }
  ],
  [
    'cap-alarmed-adjudication-0007',
    {
      capsule_id: 'cap-alarmed-adjudication-0007',
      timestamp: PEER_TAB_HARNESS_EXCHANGE_SOURCES[ALARMED_PEER_ID][0]?.timestamp ?? undefined,
      model_attestation: {
        compute_attestation: {
          adjudication: {
            verdict: 'contradicted',
            margin: '0.41',
            margin_tau: '0.9',
            half_a_capsule_id: mineCapsuleIdFor(ALARMED_ADJUDICATED_EXCHANGE_ID),
            half_b_capsule_id: `twin_${ALARMED_ADJUDICATED_EXCHANGE_ID}`,
            referee_capsule_id: 'cap-alarmed-referee-0007',
            referee_id: 'mesh-referee-eu-2'
          }
        }
      }
    }
  ]
])
