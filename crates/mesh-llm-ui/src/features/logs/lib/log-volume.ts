import {
  LOG_EVENT_CATEGORIES,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'

export type BucketIntervalKey = '1m' | '5m' | '15m' | '30m' | '1h'
export type VolumeTimeRangeKey = '1h' | '6h' | '12h' | '24h' | '7d' | 'selected' | 'all'

export const BUCKET_INTERVALS: readonly { value: BucketIntervalKey; label: string; ms: number }[] = [
  { value: '1m', label: '1m', ms: 60_000 },
  { value: '5m', label: '5m', ms: 300_000 },
  { value: '15m', label: '15m', ms: 900_000 },
  { value: '30m', label: '30m', ms: 1_800_000 },
  { value: '1h', label: '1h', ms: 3_600_000 }
]

export const VOLUME_TIME_RANGES: readonly { value: VolumeTimeRangeKey; label: string; ms: number }[] = [
  { value: '1h', label: 'Last hour', ms: 3_600_000 },
  { value: '6h', label: 'Last 6 hours', ms: 21_600_000 },
  { value: '12h', label: 'Last 12 hours', ms: 43_200_000 },
  { value: '24h', label: 'Last 24 hours', ms: 86_400_000 },
  { value: '7d', label: 'Last week', ms: 604_800_000 },
  { value: 'all', label: 'Lifetime', ms: Number.POSITIVE_INFINITY }
]

export const MAX_VOLUME_BUCKETS = 480

export type EventVolumeBucket = {
  readonly bucketStart: number
  readonly bucketEnd: number
  readonly label: string
  readonly total: number
  readonly requests: number
  readonly system: number
  readonly quic: number
  readonly gossip: number
  readonly iroh: number
}

/**
 * Bucket loaded log events by category into fixed-width time buckets.
 *
 * `now` is injected so callers can render deterministically in tests. Finite
 * `rangeMs` windows are anchored to `[now - rangeMs, now]`; an infinite range
 * spans the earliest to the latest selected event. Buckets are emitted contiguously
 * (including zero-count buckets) so the bar chart renders a true timeline.
 */
export function buildEventVolumeBuckets(
  rows: readonly LogEventLedgerRow[],
  categories: ReadonlySet<LogEventCategory>,
  options: { readonly intervalMs: number; readonly rangeMs: number; readonly now: number }
): EventVolumeBucket[] {
  const { intervalMs, rangeMs, now } = options
  if (intervalMs <= 0) return []

  const events: Array<{ readonly category: LogEventCategory; readonly timestamp: number }> = []
  for (const row of rows) {
    if (!categories.has(row.category)) continue
    const parsed = Date.parse(row.occurredAt)
    if (!Number.isNaN(parsed)) events.push({ category: row.category, timestamp: parsed })
  }
  if (events.length === 0) return []

  let startBoundary: number
  let endBoundary: number
  if (Number.isFinite(rangeMs)) {
    startBoundary = now - rangeMs
    endBoundary = now
  } else {
    startBoundary = events[0].timestamp
    endBoundary = events[0].timestamp
    for (const { timestamp } of events) {
      if (timestamp < startBoundary) startBoundary = timestamp
      if (timestamp > endBoundary) endBoundary = timestamp
    }
  }

  const effectiveIntervalMs = boundedIntervalMs(startBoundary, endBoundary, intervalMs)
  const firstIndex = Math.floor(startBoundary / effectiveIntervalMs)
  const lastIndex = Math.floor(endBoundary / effectiveIntervalMs)
  const bucketCount = lastIndex - firstIndex + 1
  const categoryTotals: Record<LogEventCategory, number[]> = {
    requests: new Array<number>(bucketCount).fill(0),
    system: new Array<number>(bucketCount).fill(0),
    quic: new Array<number>(bucketCount).fill(0),
    gossip: new Array<number>(bucketCount).fill(0),
    iroh: new Array<number>(bucketCount).fill(0)
  }

  for (const { category, timestamp } of events) {
    if (timestamp < startBoundary || timestamp > endBoundary) continue
    const index = Math.floor(timestamp / effectiveIntervalMs) - firstIndex
    if (index >= 0 && index < bucketCount) categoryTotals[category][index] += 1
  }

  const buckets: EventVolumeBucket[] = []
  for (let index = 0; index < bucketCount; index += 1) {
    const bucketStart = (firstIndex + index) * effectiveIntervalMs
    const counts = Object.fromEntries(
      LOG_EVENT_CATEGORIES.map((category) => [category, categoryTotals[category][index]])
    ) as Record<LogEventCategory, number>
    buckets.push({
      bucketStart,
      bucketEnd: bucketStart + effectiveIntervalMs,
      label: formatBucketTick(bucketStart, effectiveIntervalMs),
      total: LOG_EVENT_CATEGORIES.reduce((sum, category) => sum + counts[category], 0),
      ...counts
    })
  }
  return buckets
}

function boundedIntervalMs(startBoundary: number, endBoundary: number, requestedIntervalMs: number): number {
  let intervalMs = requestedIntervalMs
  while (Math.floor(endBoundary / intervalMs) - Math.floor(startBoundary / intervalMs) + 1 > MAX_VOLUME_BUCKETS) {
    const currentCount = Math.floor(endBoundary / intervalMs) - Math.floor(startBoundary / intervalMs) + 1
    intervalMs *= Math.max(2, Math.ceil(currentCount / MAX_VOLUME_BUCKETS))
  }
  return intervalMs
}

export function effectiveEventVolumeIntervalMs(buckets: readonly EventVolumeBucket[], fallback: number): number {
  const first = buckets[0]
  return first ? first.bucketEnd - first.bucketStart : fallback
}

export function formatBucketInterval(intervalMs: number): string {
  if (intervalMs % 3_600_000 === 0) return `${intervalMs / 3_600_000}h`
  if (intervalMs % 60_000 === 0) return `${intervalMs / 60_000}m`
  return `${Math.round(intervalMs / 1_000)}s`
}

export function formatClock(ms: number): string {
  const date = new Date(ms)
  const rawHours = date.getHours()
  const minutes = date.getMinutes()
  const period = rawHours >= 12 ? 'PM' : 'AM'
  const hours = rawHours % 12 === 0 ? 12 : rawHours % 12
  return `${hours}:${String(minutes).padStart(2, '0')} ${period}`
}

function formatHour(ms: number): string {
  const date = new Date(ms)
  const rawHours = date.getHours()
  const period = rawHours >= 12 ? 'PM' : 'AM'
  const hours = rawHours % 12 === 0 ? 12 : rawHours % 12
  return `${hours} ${period}`
}

export function formatBucketTick(bucketStart: number, intervalMs: number): string {
  return intervalMs >= 3_600_000 ? formatHour(bucketStart) : formatClock(bucketStart)
}

export function formatBucketRange(bucketStart: number, bucketEnd: number): string {
  return `${formatClock(bucketStart)}\u2013${formatClock(bucketEnd)}`
}
