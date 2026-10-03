import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { useDataMode, type DataMode } from '@/lib/data-mode'
import { LogsApiClient } from '@/features/logs/api/client'
import type { LogAuditQuery, LogsCapability, LogsRequestQuery } from '@/features/logs/api/client'
import type { LogsPage, LogRequest } from '@/features/logs/api/schemas'
import { withLedgerRouteExclusions } from '@/features/logs/api/ledger-route-exclusions'

export const LEDGER_PAGE_SIZE = 100
export const LEDGER_MAX_RECORDS = 1_000

export async function loadCompleteLedger(
  query: LogsRequestQuery,
  mode: DataMode
): Promise<LogsCapability<LogsPage<LogRequest>>> {
  const client = new LogsApiClient()
  const scopedQuery = withLedgerRouteExclusions(query)
  const items: LogRequest[] = []
  let cursor = scopedQuery.cursor
  while (items.length < LEDGER_MAX_RECORDS) {
    const result = await client.listRequests({ ...scopedQuery, cursor, limit: LEDGER_PAGE_SIZE }, mode)
    if (result.state === 'unsupported') return result
    const remaining = LEDGER_MAX_RECORDS - items.length
    items.push(...result.value.items.slice(0, remaining))
    if (result.value.items.length === 0 && result.value.nextCursor !== undefined) {
      return { state: 'supported', value: { items, nextCursor: undefined } }
    }
    if (result.value.nextCursor === undefined && result.value.items.length <= remaining) {
      return { state: 'supported', value: { items, nextCursor: undefined } }
    }
    cursor = result.value.nextCursor
    if (cursor === undefined) break
  }
  return { state: 'supported', value: { items, nextCursor: cursor, incomplete: true } }
}

function requestQueryKey(query: LogsRequestQuery) {
  return {
    cursor: query.cursor?.toString(),
    limit: query.limit,
    from: query.from,
    to: query.to,
    route: query.route,
    excludeRoute: query.excludeRoute,
    excludeRoutePrefix: query.excludeRoutePrefix,
    model: query.model,
    provider: query.provider,
    engine: query.engine,
    status: query.status,
    outcome: query.outcome,
    source: query.source,
    sort: query.sort
  }
}

export const logsKeys = {
  all: ['logs'],
  ledger: (query: LogsRequestQuery, mode: DataMode) => [
    ...logsKeys.all,
    'ledger',
    requestQueryKey(withLedgerRouteExclusions(query)),
    mode
  ],
  audit: (query: LogAuditQuery, mode: DataMode) => [
    ...logsKeys.all,
    'audit',
    { ...query, cursor: query.cursor?.toString() },
    mode
  ]
}

export function useLogsLedgerQuery(query: LogsRequestQuery) {
  const dataMode = useDataMode()
  const retainedSuccessfulData = useRef<LogsCapability<LogsPage<LogRequest>> | undefined>(undefined)
  const result = useQuery({
    queryKey: logsKeys.ledger(query, dataMode.mode),
    queryFn: () => loadCompleteLedger(query, dataMode.mode as DataMode),
    placeholderData: (previousData) => previousData ?? retainedSuccessfulData.current,
    staleTime: 10_000
  })

  useEffect(() => {
    if (result.data !== undefined && !result.isPlaceholderData) retainedSuccessfulData.current = result.data
  }, [result.data, result.isPlaceholderData])

  return result
}
