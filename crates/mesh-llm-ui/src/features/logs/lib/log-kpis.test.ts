import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { buildLogKpiMetrics } from '@/features/logs/lib/log-kpis'

function requestAt(hour: number, outcome: LogRequest['outcome']): LogRequest {
  return {
    requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000001'),
    outcome,
    createdAt: new Date(Date.UTC(2026, 7, 4, hour)).toISOString(),
    terminalAt: undefined,
    route: 'reserve',
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable'
  }
}

describe('buildLogKpiMetrics', () => {
  it('orders buckets oldest-to-newest and separates outcome trends', () => {
    const metrics = buildLogKpiMetrics(
      [requestAt(0, 'completed'), requestAt(2, 'failed'), requestAt(11, 'active'), requestAt(11, 'completed')],
      Date.UTC(2026, 7, 4, 11, 30)
    )

    expect(metrics.totalValues).toEqual([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2])
    expect(metrics.completedValues).toEqual([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    expect(metrics.failedValues).toEqual([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expect(metrics.activeValues).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    expect(metrics.totalCount).toBe(4)
    expect(metrics.activeCount).toBe(1)
  })

  it('does not count stale active or terminal records outside the wall-clock window', () => {
    const metrics = buildLogKpiMetrics(
      [requestAt(0, 'active'), requestAt(11, 'completed')],
      Date.UTC(2026, 7, 5, 11, 30)
    )

    expect(metrics.totalCount).toBe(0)
    expect(metrics.activeCount).toBe(0)
    expect(metrics.completedCount).toBe(0)
  })

  it('spans all twelve trend buckets across the selected seven-day window', () => {
    const end = Date.UTC(2026, 7, 8, 0, 0)
    const range = 7 * 24 * 60 * 60 * 1_000
    const metrics = buildLogKpiMetrics(
      [
        { ...requestAt(0, 'completed'), createdAt: new Date(end - range).toISOString() },
        { ...requestAt(0, 'failed'), createdAt: new Date(end).toISOString() }
      ],
      end,
      range
    )

    expect(metrics.totalValues).toHaveLength(12)
    expect(metrics.totalValues[0]).toBe(1)
    expect(metrics.totalValues[11]).toBe(1)
    expect(metrics.totalValues.reduce((sum, value) => sum + value, 0)).toBe(2)
  })

  it('uses the loaded minimum and maximum for an all-time trend', () => {
    const metrics = buildLogKpiMetrics(
      [requestAt(0, 'completed'), requestAt(11, 'active')],
      Date.UTC(2026, 7, 4, 12),
      Number.POSITIVE_INFINITY
    )

    expect(metrics.totalValues[0]).toBe(1)
    expect(metrics.totalValues[11]).toBe(1)
    expect(metrics.totalCount).toBe(2)
  })
})
