import type { LogCleanupPreviewRequest, LogsRequestQuery } from '@/features/logs/api/client'
import type { LogCleanupOutcome } from '@/features/logs/api/schemas'

function isCleanupOutcome(value: string | undefined): value is LogCleanupOutcome {
  return value !== undefined && ['completed', 'failed', 'rejected', 'cancelled', 'dropped'].includes(value)
}

export function supportsCleanup(query: LogsRequestQuery) {
  return (
    (query.source === undefined || query.source === 'durable') &&
    (query.outcome === undefined || isCleanupOutcome(query.outcome))
  )
}

export function cleanupScopeFromQuery(
  query: LogsRequestQuery
): Pick<LogCleanupPreviewRequest, 'source' | 'from' | 'to' | 'route' | 'model' | 'provider' | 'engine' | 'outcome'> {
  return {
    source: query.source === 'durable' ? 'durable' : undefined,
    from: query.from,
    to: query.to,
    route: query.route,
    model: query.model,
    provider: query.provider,
    engine: query.engine,
    outcome: isCleanupOutcome(query.outcome) ? query.outcome : undefined
  }
}
