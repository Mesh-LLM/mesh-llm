import type { LogRequest } from '@/features/logs/api/schemas'

const KPI_BUCKET_COUNT = 12
const KPI_BUCKET_MS = 60 * 60 * 1_000

function isFailedOutcome(outcome?: string): boolean {
  return outcome === 'failed' || outcome === 'rejected' || outcome === 'dropped'
}

export function buildLogKpiMetrics(
  rows: readonly LogRequest[],
  now = Date.now(),
  rangeMs = KPI_BUCKET_COUNT * KPI_BUCKET_MS
) {
  const totalValues = new Array<number>(KPI_BUCKET_COUNT).fill(0)
  const completedValues = new Array<number>(KPI_BUCKET_COUNT).fill(0)
  const failedValues = new Array<number>(KPI_BUCKET_COUNT).fill(0)
  const activeValues = new Array<number>(KPI_BUCKET_COUNT).fill(0)
  let totalCount = 0
  let completedCount = 0
  let failedCount = 0
  let activeCount = 0
  const validTimestamps = rows.flatMap((row) => {
    const timestamp = Date.parse(row.createdAt)
    return Number.isNaN(timestamp) ? [] : [timestamp]
  })
  if (validTimestamps.length > 0) {
    const selectedStart = Number.isFinite(rangeMs) ? now - rangeMs : Math.min(...validTimestamps)
    const selectedEnd = Number.isFinite(rangeMs) ? now : Math.max(...validTimestamps)
    const bucketWidth = Math.max((selectedEnd - selectedStart) / KPI_BUCKET_COUNT, 1)
    for (const row of rows) {
      const timestamp = Date.parse(row.createdAt)
      if (Number.isNaN(timestamp)) continue
      const isInSelectedRange = timestamp >= selectedStart && timestamp <= selectedEnd
      if (!isInSelectedRange) continue
      const index = Math.min(Math.floor((timestamp - selectedStart) / bucketWidth), KPI_BUCKET_COUNT - 1)
      totalValues[index] += 1
      totalCount += 1
      if (row.outcome === 'completed') {
        completedValues[index] += 1
        completedCount += 1
      }
      if (isFailedOutcome(row.outcome)) {
        failedValues[index] += 1
        failedCount += 1
      }
      if (row.outcome === 'active') {
        activeValues[index] += 1
        activeCount += 1
      }
    }
  }

  const share = (count: number) => (totalCount > 0 ? `${((count / totalCount) * 100).toFixed(1)}%` : '—')

  return {
    totalValues,
    completedValues,
    failedValues,
    activeValues,
    totalCount,
    completedCount,
    failedCount,
    activeCount,
    completedShare: share(completedCount),
    failedShare: share(failedCount),
    activeShare: share(activeCount)
  }
}
