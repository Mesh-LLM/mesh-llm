// Phase 2 — the adjudication lookup and timeline
// point builder. The key honesty property under test: a no-verdict
// exchange (no matching sealed adjudication capsule) always resolves to
// `adjudication: null` (renders NOT_CHECKED), and adjudication is never
// attached to a `served` point.
import { describe, expect, it } from 'vitest'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import type { PaneBRow, PaneCRow } from '@/features/capsules/api/sidecarTypes'
import {
  buildTimelinePoints,
  findAdjudicationForCapsule,
  livePeerExchangeSources,
  peerExchangeIds,
  type PeerExchangeSource
} from '@/features/capsules/lib/peer-exchange-timeline'

function baseRow(overrides: Partial<PaneBRow> = {}): PaneBRow {
  return {
    peer_id: 'node:test-peer',
    node: { state: 'present', text: '' },
    rung: { state: 'present', text: '' },
    role: { state: 'present', text: '', role: 'both', you_to_them_count: 0, them_to_you_count: 0, exchange_count: 0 },
    history: { state: 'NOT_CHECKED', text: null },
    served: { state: 'NOT_CHECKED', text: null },
    pair: { state: 'absent', text: null, verified: 0, failed: 0, missing: 0, details: [] },
    verdicts: { state: 'NOT_CHECKED', text: null },
    asked: { state: 'absent', text: null },
    exchange_count: 0,
    first_seen: null,
    last_seen: null,
    ...overrides
  }
}

describe('findAdjudicationForCapsule', () => {
  const recordsById = new Map<string, CapsuleRecord>([
    [
      'cap-adj-1',
      {
        capsule_id: 'cap-adj-1',
        model_attestation: {
          compute_attestation: {
            adjudication: {
              verdict: 'corroborated',
              margin_tau: '0.9',
              half_a_capsule_id: 'mine-1',
              half_b_capsule_id: 'twin-1',
              referee_id: 'local-twin'
            }
          }
        }
      }
    ]
  ])

  it('finds an adjudication naming the capsule as either half', () => {
    expect(findAdjudicationForCapsule('mine-1', recordsById)).toEqual({
      verdict: 'corroborated',
      marginTau: 0.9,
      refereeId: 'local-twin',
      adjudicationCapsuleId: 'cap-adj-1'
    })
  })

  it('returns null for an exchange nobody adjudicated (renders NOT_CHECKED, never invented)', () => {
    expect(findAdjudicationForCapsule('mine-unadjudicated', recordsById)).toBeNull()
  })

  it('returns null for a null capsule id without scanning', () => {
    expect(findAdjudicationForCapsule(null, recordsById)).toBeNull()
  })

  it('ignores a record whose adjudication block has no verdict', () => {
    const noVerdict = new Map<string, CapsuleRecord>([
      ['cap-x', { capsule_id: 'cap-x', model_attestation: { compute_attestation: {} } }]
    ])
    expect(findAdjudicationForCapsule('mine-1', noVerdict)).toBeNull()
  })
})

describe('livePeerExchangeSources', () => {
  it("filters Pane C to this peer's exchange_ids and maps role_tag to direction", () => {
    const row = baseRow({
      pair: {
        state: 'verified',
        text: '',
        verified: 1,
        failed: 0,
        missing: 0,
        details: [{ exchange_id: 'exch-1', state: 'verified' }]
      }
    })
    const paneCRows: PaneCRow[] = [
      {
        exchange_key: 'exch-1',
        role_tag: 'ASKED',
        header_state: 'ok',
        properties: null,
        has_issue: false,
        mine: { state: 'present', capsule_id: 'mine-1' },
        theirs: { state: 'present', capsule_id: 'theirs-1' },
        unilateral: false,
        timestamp: '2026-09-01T00:00:00Z'
      },
      {
        exchange_key: 'exch-not-this-peer',
        role_tag: 'SERVED',
        header_state: 'ok',
        properties: null,
        has_issue: false,
        mine: { state: 'present', capsule_id: 'mine-2' },
        theirs: { state: 'present', capsule_id: 'theirs-2' },
        unilateral: false,
        timestamp: '2026-09-02T00:00:00Z'
      }
    ]
    const sources = livePeerExchangeSources(row, paneCRows)
    expect(sources).toEqual([
      {
        exchangeId: 'exch-1',
        timestamp: '2026-09-01T00:00:00Z',
        direction: 'requested',
        mineCapsuleId: 'mine-1',
        theirsCapsuleId: 'theirs-1'
      }
    ])
  })
})

describe('buildTimelinePoints', () => {
  const row = baseRow({
    pair: {
      state: 'verified',
      text: '',
      verified: 1,
      failed: 1,
      missing: 0,
      details: [
        { exchange_id: 'exch-adjudicated', state: 'verified' },
        { exchange_id: 'exch-not-checked', state: 'verified' },
        { exchange_id: 'exch-served', state: 'failed' }
      ]
    }
  })
  const sources: PeerExchangeSource[] = [
    {
      exchangeId: 'exch-adjudicated',
      timestamp: '2026-09-01T00:00:00Z',
      direction: 'requested',
      mineCapsuleId: 'mine-1',
      theirsCapsuleId: 'theirs-1'
    },
    {
      exchangeId: 'exch-not-checked',
      timestamp: '2026-09-02T00:00:00Z',
      direction: 'requested',
      mineCapsuleId: 'mine-2',
      theirsCapsuleId: 'theirs-2'
    },
    {
      exchangeId: 'exch-served',
      timestamp: '2026-09-03T00:00:00Z',
      direction: 'served',
      mineCapsuleId: 'mine-3',
      theirsCapsuleId: 'theirs-3'
    }
  ]
  const recordsById = new Map<string, CapsuleRecord>([
    [
      'cap-adj-1',
      {
        capsule_id: 'cap-adj-1',
        model_attestation: {
          compute_attestation: {
            adjudication: { verdict: 'corroborated', half_a_capsule_id: 'mine-1', margin_tau: '0.9' }
          }
        }
      }
    ],
    // Names `mine-3` (a served, not requested, point) -- must never surface
    // as that point's adjudication even though it technically matches.
    [
      'cap-adj-3',
      {
        capsule_id: 'cap-adj-3',
        model_attestation: {
          compute_attestation: {
            adjudication: { verdict: 'corroborated', half_a_capsule_id: 'mine-3', margin_tau: '0.9' }
          }
        }
      }
    ]
  ])

  it('attaches a real adjudication to a requested point that has one', () => {
    const points = buildTimelinePoints(row, sources, recordsById)
    const adjudicated = points.find((p) => p.exchangeId === 'exch-adjudicated')
    expect(adjudicated?.adjudication?.verdict).toBe('corroborated')
    expect(adjudicated?.reconciliation).toBe('verified')
  })

  it('leaves a requested point with no matching adjudication as null (NOT_CHECKED), never fabricated', () => {
    const points = buildTimelinePoints(row, sources, recordsById)
    const notChecked = points.find((p) => p.exchangeId === 'exch-not-checked')
    expect(notChecked?.adjudication).toBeNull()
  })

  it('never attaches an adjudication to a served point, even when a matching record exists', () => {
    const points = buildTimelinePoints(row, sources, recordsById)
    const served = points.find((p) => p.exchangeId === 'exch-served')
    expect(served?.direction).toBe('served')
    expect(served?.adjudication).toBeNull()
  })

  it('sorts points chronologically', () => {
    const points = buildTimelinePoints(row, [...sources].reverse(), recordsById)
    expect(points.map((p) => p.exchangeId)).toEqual(['exch-adjudicated', 'exch-not-checked', 'exch-served'])
  })
})

describe('peerExchangeIds', () => {
  it('reads from expand.pair_ledger when present, falling back to pair.details', () => {
    const row = baseRow({
      pair: {
        state: 'verified',
        text: '',
        verified: 1,
        failed: 0,
        missing: 0,
        details: [{ exchange_id: 'from-pair', state: 'verified' }]
      },
      expand: { pair_ledger: [{ exchange_id: 'from-expand', state: 'verified' }] }
    })
    expect(peerExchangeIds(row)).toEqual(['from-expand'])
  })
})
