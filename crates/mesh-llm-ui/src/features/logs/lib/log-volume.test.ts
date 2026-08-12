import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import {
  MAX_VOLUME_BUCKETS,
  buildRequestVolumeBuckets,
  effectiveRequestVolumeIntervalMs,
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

function iso(ms: number): string {
  return new Date(ms).toISOString()
}

const FIVE_MINUTES = 300_000
const TWELVE_HOURS = 43_200_000

describe('buildRequestVolumeBuckets', () => {
  it('buckets requests into 5m buckets across a 12h window ending at now', () => {
    const now = utc(12, 10)
    const rows = [
      requestAt(iso(utc(12, 0))),
      requestAt(iso(utc(12, 1))),
      requestAt(iso(utc(12, 5))),
      requestAt(iso(utc(12, 9, 59))),
      requestAt(iso(utc(12, 10)))
    ]

    const buckets = buildRequestVolumeBuckets(rows, { intervalMs: FIVE_MINUTES, rangeMs: TWELVE_HOURS, now })

    expect(buckets).toHaveLength(TWELVE_HOURS / FIVE_MINUTES + 1)
    expect(buckets[0].bucketStart).toBe(now - TWELVE_HOURS)
    expect(buckets.at(-1)?.bucketEnd).toBe(now + FIVE_MINUTES)
    expect(buckets.reduce((sum, bucket) => sum + bucket.total, 0)).toBe(5)

    const byStart = new Map(buckets.map((bucket) => [bucket.bucketStart, bucket]))
    expect(byStart.get(utc(12, 0))?.total).toBe(2) // 12:00:00 + 12:01:00
    expect(byStart.get(utc(12, 5))?.total).toBe(2) // 12:05:00 + 12:09:59
    expect(byStart.get(utc(12, 10))?.total).toBe(1) // 12:10:00
    expect(byStart.get(utc(0, 15))?.total).toBe(0) // zero-count bucket preserved
  })

  it('spans earliest to latest request for the all-time range', () => {
    const rows = [requestAt(iso(utc(9, 0))), requestAt(iso(utc(21, 0)))]
    const buckets = buildRequestVolumeBuckets(rows, {
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
    const buckets = buildRequestVolumeBuckets([requestAt(iso(start)), requestAt(iso(end))], {
      intervalMs: minute,
      rangeMs: Number.POSITIVE_INFINITY,
      now: end
    })

    expect(buckets.length).toBeLessThanOrEqual(MAX_VOLUME_BUCKETS)
    expect(buckets.reduce((sum, bucket) => sum + bucket.total, 0)).toBe(2)
    expect(effectiveRequestVolumeIntervalMs(buckets, minute)).toBeGreaterThan(minute)
  })

  it('collapses requests sharing one bucket in all-time mode', () => {
    const rows = [requestAt(iso(utc(12, 0))), requestAt(iso(utc(12, 30)))]
    const buckets = buildRequestVolumeBuckets(rows, {
      intervalMs: 3_600_000,
      rangeMs: Number.POSITIVE_INFINITY,
      now: utc(12, 0)
    })

    expect(buckets).toHaveLength(1)
    expect(buckets[0].total).toBe(2)
  })

  it('includes requests exactly on the range boundaries', () => {
    const now = utc(12, 0)
    const rows = [requestAt(iso(now - TWELVE_HOURS)), requestAt(iso(now))]
    const buckets = buildRequestVolumeBuckets(rows, { intervalMs: FIVE_MINUTES, rangeMs: TWELVE_HOURS, now })

    expect(buckets[0].total).toBe(1)
    expect(buckets.at(-1)?.total).toBe(1)
  })

  it('excludes requests outside the window while preserving the frame', () => {
    const now = utc(12, 0)
    const rows = [requestAt(iso(now - TWELVE_HOURS - FIVE_MINUTES))]
    const buckets = buildRequestVolumeBuckets(rows, { intervalMs: FIVE_MINUTES, rangeMs: TWELVE_HOURS, now })

    expect(buckets).toHaveLength(TWELVE_HOURS / FIVE_MINUTES + 1)
    expect(buckets.reduce((sum, bucket) => sum + bucket.total, 0)).toBe(0)
  })

  it('returns no buckets for empty rows', () => {
    expect(buildRequestVolumeBuckets([], { intervalMs: FIVE_MINUTES, rangeMs: TWELVE_HOURS, now: utc(12, 0) })).toEqual(
      []
    )
  })

  it('returns no buckets for a non-positive interval', () => {
    const rows = [requestAt(iso(utc(12, 0)))]
    expect(buildRequestVolumeBuckets(rows, { intervalMs: 0, rangeMs: TWELVE_HOURS, now: utc(12, 0) })).toEqual([])
  })

  it('ignores unparseable timestamps', () => {
    const rows = [requestAt('not-a-date'), requestAt(iso(utc(12, 0)))]
    const buckets = buildRequestVolumeBuckets(rows, {
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
