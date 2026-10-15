import { describe, expect, it } from 'vitest'
import { buildRailSegments, sortStreamByTime } from '@/features/capsules/lib/exchange-stream'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'

function streamRow(overrides: Partial<ExchangeLedgerRow>): ExchangeLedgerRow {
  return {
    exchangeKey: 'exch-1',
    timestamp: '2026-09-08T00:00:00Z',
    roleTag: 'ASKED',
    counterparty: null,
    confirmed: false,
    localOnly: false,
    hasIssue: false,
    checksText: '—',
    rightCellState: { kind: 'open_not_asked', date: null },
    contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'not_asked', date: null } },
    sessionId: null,
    twinBracketId: null,
    raw: {} as PaneCRow,
    ...overrides
  }
}

describe('sortStreamByTime — L-N', () => {
  it('orders newest first by timestamp only', () => {
    const rows = [
      streamRow({ exchangeKey: 'a', timestamp: '2026-09-08T10:00:00Z' }),
      streamRow({ exchangeKey: 'b', timestamp: '2026-09-08T12:00:00Z' }),
      streamRow({ exchangeKey: 'c', timestamp: '2026-09-08T11:00:00Z' })
    ]
    expect(sortStreamByTime(rows).map((r) => r.exchangeKey)).toEqual(['b', 'c', 'a'])
  })

  it('ADVERSARIAL — L-N: role and outcome never affect order, only time (and exchangeKey as a stable tie-break)', () => {
    const rows = [
      streamRow({
        exchangeKey: 'served-but-newer',
        timestamp: '2026-09-08T12:00:00Z',
        roleTag: 'SERVED',
        rightCellState: { kind: 'contradicted', date: null }
      }),
      streamRow({
        exchangeKey: 'asked-but-older',
        timestamp: '2026-09-08T10:00:00Z',
        roleTag: 'ASKED',
        rightCellState: { kind: 'closed', date: null }
      })
    ]
    // The newer SERVED/CONTRADICTED row still sorts before the older
    // ASKED/CLOSED row -- proves role/outcome play no part in ordering.
    expect(sortStreamByTime(rows).map((r) => r.exchangeKey)).toEqual(['served-but-newer', 'asked-but-older'])
  })

  it('falls back to exchangeKey only as a stable tie-break on equal timestamps', () => {
    const rows = [
      streamRow({ exchangeKey: 'z', timestamp: '2026-09-08T10:00:00Z' }),
      streamRow({ exchangeKey: 'a', timestamp: '2026-09-08T10:00:00Z' })
    ]
    expect(sortStreamByTime(rows).map((r) => r.exchangeKey)).toEqual(['z', 'a'])
  })
})

describe('buildRailSegments — L-N/L-O', () => {
  it('L-O: a served row never carries a rail, even when it carries a sessionId', () => {
    const rows = [streamRow({ roleTag: 'SERVED', sessionId: 'sess-a' })]
    expect(buildRailSegments(rows)[0].hasRail).toBe(false)
  })

  it('an asked row with no sessionId carries no rail either -- never an invented grouping', () => {
    const rows = [streamRow({ roleTag: 'ASKED', sessionId: null })]
    expect(buildRailSegments(rows)[0].hasRail).toBe(false)
  })

  it('consecutive asked rows sharing a session form one rail segment', () => {
    const rows = [
      streamRow({ exchangeKey: 'a', roleTag: 'ASKED', sessionId: 'sess-1' }),
      streamRow({ exchangeKey: 'b', roleTag: 'ASKED', sessionId: 'sess-1' })
    ]
    const segments = buildRailSegments(rows)
    expect(segments[0]).toEqual({ hasRail: true, isSegmentStart: true })
    expect(segments[1]).toEqual({ hasRail: true, isSegmentStart: false })
  })

  it('a served row interrupting a session breaks the rail -- the next asked row (even same session) starts a fresh segment', () => {
    const rows = [
      streamRow({ exchangeKey: 'a', roleTag: 'ASKED', sessionId: 'sess-1' }),
      streamRow({ exchangeKey: 'b', roleTag: 'SERVED', sessionId: null }),
      streamRow({ exchangeKey: 'c', roleTag: 'ASKED', sessionId: 'sess-1' })
    ]
    const segments = buildRailSegments(rows)
    expect(segments[0]).toEqual({ hasRail: true, isSegmentStart: true })
    expect(segments[1].hasRail).toBe(false)
    expect(segments[2]).toEqual({ hasRail: true, isSegmentStart: true })
  })

  it('switching to a different session starts a new segment', () => {
    const rows = [
      streamRow({ exchangeKey: 'a', roleTag: 'ASKED', sessionId: 'sess-1' }),
      streamRow({ exchangeKey: 'b', roleTag: 'ASKED', sessionId: 'sess-2' })
    ]
    const segments = buildRailSegments(rows)
    expect(segments[0].isSegmentStart).toBe(true)
    expect(segments[1].isSegmentStart).toBe(true)
  })
})
