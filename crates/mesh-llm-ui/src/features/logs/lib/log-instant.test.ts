import { describe, expect, it } from 'vitest'
import { compareLogInstants, sortByOccurredAt } from './log-instant'

type LogFixture = {
  readonly id: string
  readonly occurredAt: string
}

describe('compareLogInstants', () => {
  it('compares RFC3339 offsets by their numeric instant', () => {
    const offsetInstant = '2026-08-04T09:30:00-04:00'
    const utcInstant = '2026-08-04T12:45:00Z'

    expect(compareLogInstants(offsetInstant, utcInstant)).toBeGreaterThan(0)
    expect(compareLogInstants(utcInstant, offsetInstant)).toBeLessThan(0)
  })

  it('returns zero for equal instants with different offsets', () => {
    expect(compareLogInstants('2026-08-04T08:30:00-04:00', '2026-08-04T12:30:00Z')).toBe(0)
  })
})

describe('sortByOccurredAt', () => {
  it('sorts ascending by instant and preserves input order for equal instants', () => {
    const entries: readonly LogFixture[] = [
      { id: 'equal-first', occurredAt: '2026-08-04T08:30:00-04:00' },
      { id: 'utc-12:45', occurredAt: '2026-08-04T12:45:00Z' },
      { id: 'offset-13:30', occurredAt: '2026-08-04T09:30:00-04:00' },
      { id: 'equal-second', occurredAt: '2026-08-04T12:30:00Z' }
    ]

    expect(sortByOccurredAt(entries).map((entry) => entry.id)).toEqual([
      'equal-first',
      'equal-second',
      'utc-12:45',
      'offset-13:30'
    ])
  })

  it('does not mutate the input order', () => {
    const entries: readonly LogFixture[] = [
      { id: 'second', occurredAt: '2026-08-04T12:45:00Z' },
      { id: 'first', occurredAt: '2026-08-04T09:30:00-04:00' }
    ]

    expect(sortByOccurredAt(entries)).not.toBe(entries)
    expect(entries.map((entry) => entry.id)).toEqual(['second', 'first'])
  })
})
