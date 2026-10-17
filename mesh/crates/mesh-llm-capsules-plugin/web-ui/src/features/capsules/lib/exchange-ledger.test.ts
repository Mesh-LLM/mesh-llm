import { describe, expect, it } from 'vitest'
import {
  buildExchangeCounterpartyIndex,
  buildExchangeLedgerRows,
  exchangeStateFilterValue,
  isServedForPay
} from '@/features/capsules/lib/exchange-ledger'
import type { CapsuleRecord } from '@/features/capsules/api/types'
import type { PaneBRow, PaneCRow, PayerBook } from '@/features/capsules/api/sidecarTypes'
import { fixtureMineCell, fixtureTheirsCell } from '@/features/capsules/lib/pushed-half-fixtures'

function paneCRow(overrides: Partial<PaneCRow>): PaneCRow {
  return {
    exchange_key: 'exch-1',
    role_tag: 'ASKED',
    header_state: 'ok',
    properties: null,
    has_issue: false,
    mine: { state: 'present', capsule_id: 'mine-1' },
    theirs: { state: 'present', capsule_id: 'theirs-1' },
    unilateral: false,
    timestamp: '2026-09-08T00:00:00Z',
    ...overrides
  }
}

describe('buildExchangeLedgerRows', () => {
  it('Checks is an em-dash for a clean row, never a count', () => {
    const [row] = buildExchangeLedgerRows([paneCRow({ has_issue: false })], new Map())
    expect(row.checksText).toBe('—')
    expect(row.checksText).not.toMatch(/\d\/\d/)
  })

  it('Checks names the specific failing property, never a count, when the row has an issue', () => {
    const [row] = buildExchangeLedgerRows(
      [
        paneCRow({
          has_issue: true,
          properties: {
            content_binding: { state: 'PASS' },
            checkpoint_signature: { state: 'FAIL', text: 'could not verify against the pinned key' }
          }
        })
      ],
      new Map()
    )
    expect(row.checksText).toBe('checkpoint signature')
    expect(row.checksText).not.toMatch(/\d\/\d/)
  })

  it('ADVERSARIAL: a NOT_CHECKED or NOT_PRESENT property is never named as a Checks exception — only a real FAIL is', () => {
    const [row] = buildExchangeLedgerRows(
      [
        paneCRow({
          // has_issue is false here on purpose: an honest NOT_CHECKED/
          // NOT_PRESENT property is never, by itself, an "issue" (the
          // backend's own has_issue rule this view trusts), so Checks
          // must never surface one as though it were a failing check.
          has_issue: false,
          properties: {
            content_binding: { state: 'PASS' },
            checkpoint_signature: { state: 'NOT_CHECKED' },
            identity_authority: { state: 'NOT_PRESENT' }
          }
        })
      ],
      new Map()
    )
    expect(row.checksText).toBe('—')
  })

  it('ADVERSARIAL: never corroborates a no-verdict case — a false has_issue with a stray FAIL-looking property still reports only real FAILs', () => {
    // Even if `has_issue` were true for an unrelated reason, checksTextFor
    // only ever names properties whose OWN state is the literal string
    // 'FAIL' -- NOT_CHECKED/NOT_PRESENT never qualify, so this can't drift
    // into treating "unknown" as "bad" (nor the inverse).
    const [row] = buildExchangeLedgerRows(
      [
        paneCRow({
          has_issue: true,
          properties: {
            content_binding: { state: 'NOT_CHECKED' },
            checkpoint_signature: { state: 'FAIL' },
            capture_coverage: { state: 'NOT_PRESENT' }
          }
        })
      ],
      new Map()
    )
    expect(row.checksText).toBe('checkpoint signature')
    expect(row.checksText).not.toMatch(/content binding|capture coverage/)
  })

  it('falls back to naming the pair-reconciliation check when has_issue is true but no named property failed', () => {
    const [row] = buildExchangeLedgerRows(
      [paneCRow({ has_issue: true, properties: { content_binding: { state: 'PASS' } } })],
      new Map()
    )
    expect(row.checksText).toBe('pair reconciliation')
  })

  it('Confirmed derives from the ONE gate (rightCellState closed), never the retired structural theirs-present read', () => {
    const rows = buildExchangeLedgerRows(
      [
        // Gate-closed: a pushed half whose body agrees with ours (signature,
        // capsule_id, both digests, provider).
        paneCRow({
          exchange_key: 'exch-confirmed',
          mine: fixtureMineCell(),
          theirs: fixtureTheirsCell('agrees'),
          digest_match: { state: 'verified' },
          unilateral: false
        }),
        // ADVERSARIAL (the dormant second predicate, pinned dead): theirs
        // "present" and not unilateral, but NO gate inputs -- the old
        // `theirs.state !== 'absent' && !unilateral` read called this
        // confirmed; the gate does not.
        paneCRow({
          exchange_key: 'exch-present-not-verified',
          theirs: { state: 'present', capsule_id: 't' },
          unilateral: false
        }),
        paneCRow({ exchange_key: 'exch-unilateral', theirs: { state: 'absent', capsule_id: null }, unilateral: true })
      ],
      new Map()
    )
    expect(rows.find((r) => r.exchangeKey === 'exch-confirmed')?.confirmed).toBe(true)
    expect(rows.find((r) => r.exchangeKey === 'exch-present-not-verified')?.confirmed).toBe(false)
    expect(rows.find((r) => r.exchangeKey === 'exch-unilateral')?.confirmed).toBe(false)
  })

  it('confirmed always equals rightCellState.kind === "closed" — one gate, one field, never divergent', () => {
    const rows = buildExchangeLedgerRows(
      [
        paneCRow({
          exchange_key: 'a',
          mine: fixtureMineCell(),
          theirs: fixtureTheirsCell('agrees'),
          digest_match: { state: 'verified' }
        }),
        paneCRow({ exchange_key: 'b' }),
        paneCRow({ exchange_key: 'c', theirs: { state: 'absent', capsule_id: null } })
      ],
      new Map()
    )
    for (const row of rows) {
      expect(row.confirmed).toBe(row.rightCellState.kind === 'closed')
    }
  })

  it('names the counterparty from the Pane B join, never inventing one when absent', () => {
    const index = new Map([['exch-1', 'node:aa11bb22']])
    const [named] = buildExchangeLedgerRows([paneCRow({ exchange_key: 'exch-1' })], index)
    const [unnamed] = buildExchangeLedgerRows([paneCRow({ exchange_key: 'exch-2' })], index)
    expect(named.counterparty).toBe('node:aa11bb22')
    expect(unnamed.counterparty).toBeNull()
  })

  it('D4(a): a requester row names the peer its own record routed to (row.counterparty), still OPEN', () => {
    const [row] = buildExchangeLedgerRows(
      [
        paneCRow({
          exchange_key: 'exch-asked',
          role_tag: 'ASKED',
          counterparty: 'node:a70d3967bea3b22f',
          theirs: { state: 'absent', capsule_id: null },
          unilateral: true
        })
      ],
      new Map()
    )
    expect(row.counterparty).toBe('node:a70d3967bea3b22f')
    // Naming whom we asked is NOT claiming their half.
    expect(row.confirmed).toBe(false)
    expect(row.rightCellState.kind).toBe('open_not_asked')
  })

  it("the row's own counterparty outranks the Pane B pair join; the door sender is the last resort", () => {
    const index = new Map([['exch-1', 'node:from-pane-b']])
    const [own] = buildExchangeLedgerRows(
      [paneCRow({ exchange_key: 'exch-1', counterparty: 'key:from-row' })],
      index
    )
    expect(own.counterparty).toBe('key:from-row')
    const [joined] = buildExchangeLedgerRows([paneCRow({ exchange_key: 'exch-1', counterparty: null })], index)
    expect(joined.counterparty).toBe('node:from-pane-b')

    const [doorFallback] = buildExchangeLedgerRows(
      [
        paneCRow({
          exchange_key: 'exch-2',
          theirs: { state: 'present-unverified', capsule_id: 't', received_from: 'e5ba9d1001' }
        })
      ],
      new Map()
    )
    expect(doorFallback.counterparty).toBe('e5ba9d1001')
  })

  it('a twin pair: two rows sharing one exchange key each name their own provider (Phase C finding)', () => {
    // Both providers' Peers rows list the shared digest; the old order let the
    // index (last writer wins) label BOTH rows as one provider.
    const index = new Map([['digest:twin', 'key:provider-a']])
    const rows = buildExchangeLedgerRows(
      [
        paneCRow({ exchange_key: 'digest:twin', counterparty: 'key:provider-a' }),
        paneCRow({ exchange_key: 'digest:twin#2', counterparty: 'key:provider-b' })
      ],
      index
    )
    expect(rows.map((r) => r.counterparty)).toEqual(['key:provider-a', 'key:provider-b'])
    const sameKey = buildExchangeLedgerRows(
      [
        paneCRow({ exchange_key: 'digest:twin', counterparty: 'key:provider-a' }),
        paneCRow({ exchange_key: 'digest:twin', counterparty: 'key:provider-b' })
      ],
      index
    )
    expect(sameKey.map((r) => r.counterparty)).toEqual(['key:provider-a', 'key:provider-b'])
  })
})

describe('a served exchange with no other side is LOCAL', () => {
  // As served on a live node for a chat sent from its own console: this node
  // served it, and nothing names or holds another side.
  const selfServed = (overrides: Partial<PaneCRow> = {}) =>
    paneCRow({
      role_tag: 'SERVED',
      header_state: 'absent',
      counterparty: null,
      mine: { state: 'present-unverified', capsule_id: 'mine-local', role: 'served' },
      theirs: { state: 'absent', capsule_id: null },
      unilateral: true,
      ...overrides
    })

  it('is local only, in its own filter bucket, and neither open nor confirmed', () => {
    const [row] = buildExchangeLedgerRows([selfServed()], new Map())
    expect(row.rightCellState.kind).toBe('open_not_asked')
    expect(row.localOnly).toBe(true)
    expect(row.confirmed).toBe(false)
    expect(exchangeStateFilterValue(row)).toBe('local')
  })

  it.each([
    ['the row names a counterparty', selfServed({ counterparty: 'key:2b4600' }), new Map<string, string>()],
    ['Pane B names a counterparty', selfServed(), new Map([['exch-1', 'node:peer']])],
    ['it is a row this node asked', selfServed({ role_tag: 'ASKED' }), new Map<string, string>()],
    [
      'the other side has stated something',
      selfServed({ theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'unanswered' } }),
      new Map<string, string>()
    ],
    [
      'a half of theirs was pushed',
      selfServed({ theirs: { state: 'present-unverified', capsule_id: 'theirs-1', received_from: 'peer-x' } }),
      new Map<string, string>()
    ]
  ])('is not local when %s, and then counts as it did before', (_, paneRow, index) => {
    const [row] = buildExchangeLedgerRows([paneRow], index)
    expect(row.localOnly).toBe(false)
    expect(exchangeStateFilterValue(row)).not.toBe('local')
  })

  it('is not local when it was served for pay, even with no payer recorded', () => {
    const providerBook: PayerBook = {
      observed_by: 'provider',
      who_paid: 'requester',
      state: 'settled',
      terms_digests: ['t'.repeat(64)],
      entries: []
    }
    const [row] = buildExchangeLedgerRows([selfServed({ provider_settlement: providerBook })], new Map())
    expect(isServedForPay(row.raw)).toBe(true)
    expect(row.localOnly).toBe(false)
    expect(exchangeStateFilterValue(row)).toBe('open')
    // No book, or an explicit null one, is not a paid exchange: still local.
    for (const provider_settlement of [null, undefined]) {
      const [free] = buildExchangeLedgerRows([selfServed({ provider_settlement })], new Map())
      expect(isServedForPay(free.raw)).toBe(false)
      expect(free.localOnly).toBe(true)
    }
  })

  it('is not local when its own record names the node that asked', () => {
    const record = {
      capsule_id: 'mine-local',
      effect: {},
      model_attestation: {
        compute_attestation: {
          'x-mesh-poc-v1': {
            role: 'served',
            client_nonce: 'nonce-1',
            serving_provenance: { requested_by_node_id: 'a'.repeat(64) }
          }
        }
      }
    } as unknown as CapsuleRecord
    const [row] = buildExchangeLedgerRows([selfServed()], new Map(), new Map([['mine-local', record]]))
    expect(row.localOnly).toBe(false)
    expect(exchangeStateFilterValue(row)).toBe('open')
  })

  it('the open count leaves it out', () => {
    const rows = buildExchangeLedgerRows(
      [selfServed({ exchange_key: 'local' }), selfServed({ exchange_key: 'remote', counterparty: 'key:2b4600' })],
      new Map()
    )
    expect(rows.filter((row) => exchangeStateFilterValue(row) === 'open').map((row) => row.exchangeKey)).toEqual([
      'remote'
    ])
    expect(rows.filter((row) => exchangeStateFilterValue(row) === 'local').map((row) => row.exchangeKey)).toEqual([
      'local'
    ])
  })
})

describe('buildExchangeCounterpartyIndex', () => {
  it("maps every exchange_id a peer reconciled to that peer's display id", () => {
    const peer: PaneBRow = {
      peer_id: 'node:peer-a',
      node: { state: 'present' },
      rung: { state: 'present' },
      role: { state: 'present' },
      history: { state: 'NOT_CHECKED' },
      served: { state: 'NOT_CHECKED' },
      pair: {
        state: 'verified',
        verified: 1,
        failed: 0,
        missing: 0,
        details: [{ exchange_id: 'exch-a', state: 'verified' }]
      },
      verdicts: { state: 'NOT_CHECKED' },
      asked: { state: 'absent' },
      exchange_count: 1,
      first_seen: null,
      last_seen: null
    }
    const index = buildExchangeCounterpartyIndex([peer])
    expect(index.get('exch-a')).toBe('node:peer-a')
    expect(index.has('exch-nonexistent')).toBe(false)
  })

  it('an exchange id two peers both list (a twin pair) names neither', () => {
    const peer = (peer_id: string): PaneBRow => ({
      peer_id,
      node: { state: 'present' },
      rung: { state: 'present' },
      role: { state: 'present' },
      history: { state: 'NOT_CHECKED' },
      served: { state: 'NOT_CHECKED' },
      pair: { state: 'verified', verified: 1, failed: 0, missing: 0, details: [{ exchange_id: 'digest:twin', state: 'verified' }] },
      verdicts: { state: 'NOT_CHECKED' },
      asked: { state: 'absent' },
      exchange_count: 1,
      first_seen: null,
      last_seen: null
    })
    const index = buildExchangeCounterpartyIndex([peer('key:provider-a'), peer('key:provider-b')])
    expect(index.has('digest:twin')).toBe(false)
  })

  it('ADVERSARIAL: an unattributed Pane B row (no peer_id, no node.peer_id) contributes no entries — never "unknown peer"', () => {
    const unattributed: PaneBRow = {
      peer_id: null,
      node: { state: 'present' },
      rung: { state: 'present' },
      role: { state: 'present' },
      history: { state: 'NOT_CHECKED' },
      served: { state: 'NOT_CHECKED' },
      pair: {
        state: 'verified',
        verified: 1,
        failed: 0,
        missing: 0,
        details: [{ exchange_id: 'exch-unattributed', state: 'verified' }]
      },
      verdicts: { state: 'NOT_CHECKED' },
      asked: { state: 'absent' },
      exchange_count: 1,
      first_seen: null,
      last_seen: null
    }
    const index = buildExchangeCounterpartyIndex([unattributed])
    expect(index.has('exch-unattributed')).toBe(false)
    expect([...index.values()]).not.toContain('unknown peer')
  })
})
