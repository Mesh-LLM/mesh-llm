import type { LogCleanupPreviewRequest, LogsRequestQuery } from '@/features/logs/api/client'
import { withLedgerRouteExclusions } from '@/features/logs/api/ledger-route-exclusions'
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
): Pick<
  LogCleanupPreviewRequest,
  'source' | 'from' | 'to' | 'route' | 'excludeRoute' | 'model' | 'provider' | 'engine' | 'outcome'
> {
  const scopedQuery = withLedgerRouteExclusions(query)
  return {
    source: scopedQuery.source === 'durable' ? 'durable' : undefined,
    from: scopedQuery.from,
    to: scopedQuery.to,
    route: scopedQuery.route,
    excludeRoute: scopedQuery.excludeRoute,
    model: scopedQuery.model,
    provider: scopedQuery.provider,
    engine: scopedQuery.engine,
    outcome: isCleanupOutcome(scopedQuery.outcome) ? scopedQuery.outcome : undefined
  }
}
