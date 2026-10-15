import { describe, expect, it } from 'vitest'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import {
  dayHeaderIndices,
  dayKeyForRow,
  dayTalliesByKey,
  dayTallyLine
} from '@/features/capsules/lib/exchange-day-groups'

function row(overrides: Partial<ExchangeLedgerRow> = {}): ExchangeLedgerRow {
  return {
    exchangeKey: 'exch-1',
    timestamp: '2026-09-23T07:00:00Z',
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
    raw: { mine: { state: 'present', capsule_id: 'mine-1' }, theirs: { state: 'absent', capsule_id: null } } as never,
    ...overrides
  }
}

describe('dayKeyForRow', () => {
  it('buckets by the UTC calendar day, ignoring time-of-day', () => {
    expect(dayKeyForRow(row({ timestamp: '2026-09-23T00:00:01Z' }))).toBe('2026-09-23')
    expect(dayKeyForRow(row({ timestamp: '2026-09-23T23:59:59Z' }))).toBe('2026-09-23')
  })

  it('a null timestamp buckets under a stable "unknown" key, never a lone false day', () => {
    expect(dayKeyForRow(row({ timestamp: null }))).toBe('unknown')
  })
})

describe('dayTalliesByKey', () => {
  it('tallies count and confirmed-count per day, confirmed meaning CLOSED', () => {
    const rows = [
      row({ exchangeKey: 'e1', timestamp: '2026-09-23T07:00:00Z', rightCellState: { kind: 'closed', date: null } }),
      row({
        exchangeKey: 'e2',
        timestamp: '2026-09-23T08:00:00Z',
        rightCellState: { kind: 'open_not_asked', date: null }
      }),
      row({
        exchangeKey: 'e3',
        timestamp: '2026-09-22T08:00:00Z',
        rightCellState: { kind: 'contradicted', date: null }
      })
    ]
    const tallies = dayTalliesByKey(rows)
    expect(tallies.get('2026-09-23')).toEqual({ label: 'Wednesday 23 Sep', count: 2, confirmedCount: 1 })
    expect(tallies.get('2026-09-22')).toEqual({ label: 'Tuesday 22 Sep', count: 1, confirmedCount: 0 })
  })
})

describe('dayHeaderIndices', () => {
  it('marks index 0 and every index whose day differs from the row immediately before it', () => {
    const rows = [
      row({ exchangeKey: 'e1', timestamp: '2026-09-23T09:00:00Z' }),
      row({ exchangeKey: 'e2', timestamp: '2026-09-23T08:00:00Z' }),
      row({ exchangeKey: 'e3', timestamp: '2026-09-22T09:00:00Z' }),
      row({ exchangeKey: 'e4', timestamp: '2026-09-22T08:00:00Z' })
    ]
    expect([...dayHeaderIndices(rows)]).toEqual([0, 2])
  })

  it('an empty row set marks nothing', () => {
    expect(dayHeaderIndices([])).toEqual(new Set())
  })
})

describe('dayTallyLine', () => {
  it('renders the vocabulary-ruled words -- "exchanges"/"confirmed", never "balanced"', () => {
    const line = dayTallyLine({ label: 'Tuesday 23 Sep', count: 42, confirmedCount: 0 })
    expect(line).toBe('Tuesday 23 Sep · 42 exchanges · 0 confirmed')
    expect(line).not.toMatch(/balanced/i)
  })

  it('singularises "exchange" at a count of exactly one', () => {
    expect(dayTallyLine({ label: 'Tuesday 23 Sep', count: 1, confirmedCount: 1 })).toBe(
      'Tuesday 23 Sep · 1 exchange · 1 confirmed'
    )
  })
})
