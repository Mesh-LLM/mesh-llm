import type { LogsRequestQuery } from './client'

const LEDGER_ROUTE_EXCLUSIONS = {
  excludeRoute: 'models',
  excludeRoutePrefix: 'management_'
} as const satisfies LogsRequestQuery

export function withLedgerRouteExclusions(query: LogsRequestQuery): LogsRequestQuery {
  return { ...query, ...LEDGER_ROUTE_EXCLUSIONS }
}
