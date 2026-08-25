import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import {
  LOG_EVENT_CATEGORIES,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import {
  BUCKET_INTERVALS,
  MAX_VOLUME_BUCKETS,
  PREFERRED_VOLUME_BUCKETS,
  VOLUME_TIME_RANGES,
  buildEventVolumeBuckets,
  defaultBucketIntervalKey,
  effectiveEventVolumeIntervalMs,
  formatBucketRange,
  formatBucketTick,
  formatClock
} from '@/features/logs/lib/log-volume'

// 2026-08-04T00:00:00Z .. 2026-08-04T23:59:59Z fixtures.
function utc(hours: number, minutes = 0, seconds = 0): number {
  return Date.UTC(2026, 7, 4, hours, minutes, seconds)
}

function requestAt(createdAt: string): LogRequest {
  return {
    requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000001'),
    outcome: 'completed',
    createdAt,
    terminalAt: undefined,
    route: 'reserve',
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable'
  }
}

function eventAt(category: LogEventCategory, occurredAt: string, index = 1): LogEventLedgerRow {
  if (category === 'requests') {
    return {
      type: 'request',
      id: `request:${index}`,
      occurredAt,
      category,
      request: requestAt(occurredAt)
    }
  }
  return {
    type: 'audit',
    id: `audit:${index}`,
    occurredAt,
    category,
    audit: {
      entryId: `audit-${index}`,
      occurredAt,
      source: 'runtime',
      code: `${category}_event`,
      sequence: index
    }
  }
}

function iso(ms: number): string {
  return new Date(ms).toISOString()
}

const FIVE_MINUTES = 300_000
const TWELVE_HOURS = 43_200_000
const ALL_CATEGORIES = new Set<LogEventCategory>(LOG_EVENT_CATEGORIES)

describe('buildEventVolumeBuckets', () => {
  it('buckets selected event categories into 5m stacks across a 12h window ending at now', () => {
    const now = utc(12, 10)
    const rows = [
      eventAt('requests', iso(utc(12, 0)), 1),
      eventAt('system', iso(utc(12, 1)), 2),
      eventAt('quic', iso(utc(12, 5)), 3),
      eventAt('gossip', iso(utc(12, 9, 59)), 4),
      eventAt('requests', iso(utc(12, 10)), 5)
    ]

    const buckets = buildEventVolumeBuckets(rows, ALL_CATEGORIES, {
      intervalMs: FIVE_MINUTES,
      rangeMs: TWELVE_HOURS,
      now
    })

    expect(buckets).toHaveLength(TWELVE_HOURS / FIVE_MINUTES + 1)
    expect(buckets[0].bucketStart).toBe(now - TWELVE_HOURS)
    expect(buckets.at(-1)?.bucketEnd).toBe(now + FIVE_MINUTES)
    expect(buckets.reduce((sum, bucket) => sum + bucket.total, 0)).toBe(5)

    const byStart = new Map(buckets.map((bucket) => [bucket.bucketStart, bucket]))
    expect(byStart.get(utc(12, 0))).toMatchObject({ requests: 1, system: 1, total: 2 })
    expect(byStart.get(utc(12, 5))).toMatchObject({ quic: 1, gossip: 1, total: 2 })
    expect(byStart.get(utc(12, 10))?.total).toBe(1) // 12:10:00
    expect(byStart.get(utc(0, 15))?.total).toBe(0) // zero-count bucket preserved
  })

  it('removes filtered categories from bucket totals', () => {
    const rows = [eventAt('requests', iso(utc(12, 0)), 1), eventAt('system', iso(utc(12, 0)), 2)]

    const buckets = buildEventVolumeBuckets(rows, new Set<LogEventCategory>(['system']), {
      intervalMs: FIVE_MINUTES,
      rangeMs: TWELVE_HOURS,
      now: utc(12, 0)
    })

    expect(buckets.at(-1)).toMatchObject({ requests: 0, system: 1, total: 1 })
  })

  it('spans earliest to latest selected event for the all-time range', () => {
    const rows = [eventAt('requests', iso(utc(9, 0)), 1), eventAt('system', iso(utc(21, 0)), 2)]
    const buckets = buildEventVolumeBuckets(rows, ALL_CATEGORIES, {
      intervalMs: 3_600_000,
      rangeMs: Number.POSITIVE_INFINITY,
      now: utc(12, 0)
    })

    expect(buckets).toHaveLength(13)
    expect(buckets[0].bucketStart).toBe(utc(9, 0))
    expect(buckets[0].total).toBe(1)
    expect(buckets.at(-1)?.bucketStart).toBe(utc(21, 0))
    expect(buckets.at(-1)?.total).toBe(1)
  })

  it('caps sparse 90-day all-time data even when one-minute buckets are requested', () => {
    const minute = 60_000
    const start = Date.UTC(2026, 4, 1, 0, 0, 0)
    const end = start + 90 * 24 * 60 * minute
    const buckets = buildEventVolumeBuckets(
      [eventAt('requests', iso(start), 1), eventAt('requests', iso(end), 2)],
      ALL_CATEGORIES,
      {
        intervalMs: minute,
        rangeMs: Number.POSITIVE_INFINITY,
        now: end
      }
    )

    expect(buckets.length).toBeLessThanOrEqual(MAX_VOLUME_BUCKETS)
    expect(buckets.reduce((sum, bucket) => sum + bucket.total, 0)).toBe(2)
    expect(effectiveEventVolumeIntervalMs(buckets, minute)).toBeGreaterThan(minute)
  })

  it('collapses events sharing one bucket in all-time mode', () => {
    const rows = [eventAt('requests', iso(utc(12, 0)), 1), eventAt('system', iso(utc(12, 30)), 2)]
    const buckets = buildEventVolumeBuckets(rows, ALL_CATEGORIES, {
      intervalMs: 3_600_000,
      rangeMs: Number.POSITIVE_INFINITY,
      now: utc(12, 0)
    })

    expect(buckets).toHaveLength(1)
    expect(buckets[0].total).toBe(2)
  })

  it('includes requests exactly on the range boundaries', () => {
    const now = utc(12, 0)
    const rows = [eventAt('requests', iso(now - TWELVE_HOURS), 1), eventAt('requests', iso(now), 2)]
    const buckets = buildEventVolumeBuckets(rows, ALL_CATEGORIES, {
      intervalMs: FIVE_MINUTES,
      rangeMs: TWELVE_HOURS,
      now
    })

    expect(buckets[0].total).toBe(1)
    expect(buckets.at(-1)?.total).toBe(1)
  })

  it('excludes requests outside the window while preserving the frame', () => {
    const now = utc(12, 0)
    const rows = [eventAt('requests', iso(now - TWELVE_HOURS - FIVE_MINUTES))]
    const buckets = buildEventVolumeBuckets(rows, ALL_CATEGORIES, {
      intervalMs: FIVE_MINUTES,
      rangeMs: TWELVE_HOURS,
      now
    })

    expect(buckets).toHaveLength(TWELVE_HOURS / FIVE_MINUTES + 1)
    expect(buckets.reduce((sum, bucket) => sum + bucket.total, 0)).toBe(0)
  })

  it('returns no buckets for empty rows', () => {
    expect(
      buildEventVolumeBuckets([], ALL_CATEGORIES, {
        intervalMs: FIVE_MINUTES,
        rangeMs: TWELVE_HOURS,
        now: utc(12, 0)
      })
    ).toEqual([])
  })

  it('returns no buckets for a non-positive interval', () => {
    const rows = [eventAt('requests', iso(utc(12, 0)))]
    expect(
      buildEventVolumeBuckets(rows, ALL_CATEGORIES, { intervalMs: 0, rangeMs: TWELVE_HOURS, now: utc(12, 0) })
    ).toEqual([])
  })

  it('ignores unparseable timestamps', () => {
    const rows = [eventAt('requests', 'not-a-date', 1), eventAt('requests', iso(utc(12, 0)), 2)]
    const buckets = buildEventVolumeBuckets(rows, ALL_CATEGORIES, {
      intervalMs: 3_600_000,
      rangeMs: Number.POSITIVE_INFINITY,
      now: utc(12, 0)
    })
    expect(buckets).toHaveLength(1)
    expect(buckets[0].total).toBe(1)
  })
})

describe('time label formatters', () => {
  it('formats a clock label with an AM/PM period', () => {
    expect(formatClock(utc(10, 25))).toMatch(/^\d{1,2}:25 (AM|PM)$/)
    expect(formatClock(utc(22, 5))).toMatch(/^\d{1,2}:05 (AM|PM)$/)
  })

  it('formats a bucket range with an en dash', () => {
    expect(formatBucketRange(utc(10, 25), utc(10, 30))).toMatch(/^\d{1,2}:25 (AM|PM)\u2013\d{1,2}:30 (AM|PM)$/)
  })

  it('drops minutes for hourly bucket ticks', () => {
    expect(formatBucketTick(utc(22, 0), 3_600_000)).toMatch(/^\d{1,2} (AM|PM)$/)
    expect(formatBucketTick(utc(10, 25), FIVE_MINUTES)).toMatch(/^\d{1,2}:25 (AM|PM)$/)
  })
})

describe('defaultBucketIntervalKey', () => {
  it('pairs each selectable time range with a legible bucket interval', () => {
    expect(defaultBucketIntervalKey(3_600_000)).toBe('1m')
    expect(defaultBucketIntervalKey(21_600_000)).toBe('5m')
    expect(defaultBucketIntervalKey(43_200_000)).toBe('15m')
    expect(defaultBucketIntervalKey(86_400_000)).toBe('30m')
    expect(defaultBucketIntervalKey(604_800_000)).toBe('1h')
  })

  it('falls back to the coarsest interval for lifetime and degenerate windows', () => {
    expect(defaultBucketIntervalKey(Number.POSITIVE_INFINITY)).toBe('1h')
    expect(defaultBucketIntervalKey(Number.NaN)).toBe('1h')
    expect(defaultBucketIntervalKey(0)).toBe('1h')
    expect(defaultBucketIntervalKey(-1)).toBe('1h')
  })

  it('keeps every finite range at the preferred bar count or the coarsest interval available', () => {
    const coarsest = BUCKET_INTERVALS[BUCKET_INTERVALS.length - 1]
    for (const range of VOLUME_TIME_RANGES) {
      if (!Number.isFinite(range.ms)) continue
      const key = defaultBucketIntervalKey(range.ms)
      const intervalMs = BUCKET_INTERVALS.find((option) => option.value === key)?.ms ?? 0
      expect(intervalMs).toBeGreaterThan(0)
      expect(range.ms / intervalMs <= PREFERRED_VOLUME_BUCKETS || key === coarsest.value).toBe(true)
    }
  })
})
