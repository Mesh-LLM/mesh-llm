import { useQuery } from '@tanstack/react-query'
import { useDataMode, type DataMode } from '@/lib/data-mode'
import { LogsApiClient } from '@/features/logs/api/client'
import { LogPageCursor, type LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest, LogsPage } from '@/features/logs/api/schemas'

const DETAIL_PAGE_SIZE = 50
export const DETAIL_ITEM_LIMIT = 250

/**
 * Collect request-detail pages up to a strict client-side ceiling. The returned
 * cursor is intentionally preserved when more data exists so consumers can
 * state that the diagnostic view is incomplete instead of silently treating a
 * server-default first page as the whole request history.
 */
export async function loadBoundedDetailPages<T>(
  fetchPage: (cursor: LogPageCursor | undefined, limit: number) => Promise<LogsPage<T>>
): Promise<LogsPage<T>> {
  const items: T[] = []
  const seenCursors = new Set<string>()
  let cursor: LogPageCursor | undefined

  while (items.length < DETAIL_ITEM_LIMIT) {
    const page = await fetchPage(cursor, Math.min(DETAIL_PAGE_SIZE, DETAIL_ITEM_LIMIT - items.length))
    items.push(...page.items.slice(0, DETAIL_ITEM_LIMIT - items.length))
    if (!page.nextCursor) return { items, nextCursor: undefined }
    if (page.items.length === 0) throw new Error('Logs detail pagination returned an empty page with a cursor')

    const nextCursor = page.nextCursor.toString()
    if (seenCursors.has(nextCursor)) throw new Error('Logs detail pagination returned a repeated cursor')
    seenCursors.add(nextCursor)
    cursor = LogPageCursor.parse(nextCursor)
  }

  return { items, nextCursor: cursor }
}

export const logRequestDetailsKeys = {
  all: ['logs', 'request-details'],
  summary: (requestId: LogRequestId, mode: DataMode) => [
    ...logRequestDetailsKeys.all,
    'summary',
    requestId.toString(),
    mode
  ],
  events: (requestId: LogRequestId, mode: DataMode) => [
    ...logRequestDetailsKeys.all,
    'events',
    requestId.toString(),
    mode
  ],
  artifacts: (requestId: LogRequestId, mode: DataMode) => [
    ...logRequestDetailsKeys.all,
    'artifacts',
    requestId.toString(),
    mode
  ],
  attempts: (requestId: LogRequestId, mode: DataMode) => [
    ...logRequestDetailsKeys.all,
    'attempts',
    requestId.toString(),
    mode
  ]
}

/**
 * Read one request summary, optionally seeded with the ledger row the caller
 * already holds. The seed is treated as immediately stale, so the inspector
 * paints real data on the first frame and the authoritative record still
 * arrives from a background refetch.
 */
export function useLogRequestSummaryQuery(requestId: LogRequestId, knownRequest?: LogRequest) {
  const dataMode = useDataMode()
  return useQuery({
    queryKey: logRequestDetailsKeys.summary(requestId, dataMode.mode),
    queryFn: () => new LogsApiClient().getRequest(requestId, dataMode.mode as DataMode),
    initialData: knownRequest,
    initialDataUpdatedAt: 0,
    staleTime: 10_000
  })
}

export function useLogRequestEventsQuery(requestId: LogRequestId, enabled: boolean) {
  const dataMode = useDataMode()
  return useQuery({
    queryKey: logRequestDetailsKeys.events(requestId, dataMode.mode),
    queryFn: () => {
      const client = new LogsApiClient()
      return loadBoundedDetailPages((cursor, limit) =>
        client.listRequestEvents(requestId, { cursor, limit }, dataMode.mode as DataMode)
      )
    },
    enabled,
    staleTime: 10_000
  })
}

export function useLogRequestArtifactsQuery(requestId: LogRequestId, enabled: boolean) {
  const dataMode = useDataMode()
  return useQuery({
    queryKey: logRequestDetailsKeys.artifacts(requestId, dataMode.mode),
    queryFn: () => {
      const client = new LogsApiClient()
      return loadBoundedDetailPages((cursor, limit) =>
        client.listRequestArtifacts(requestId, { cursor, limit }, dataMode.mode as DataMode)
      )
    },
    enabled,
    staleTime: 10_000
  })
}

export function useLogRequestAttemptsQuery(requestId: LogRequestId, enabled: boolean) {
  const dataMode = useDataMode()
  return useQuery({
    queryKey: logRequestDetailsKeys.attempts(requestId, dataMode.mode),
    queryFn: () => {
      const client = new LogsApiClient()
      return loadBoundedDetailPages((cursor, limit) =>
        client.listProxy({ requestId, cursor, limit }, dataMode.mode as DataMode)
      )
    },
    enabled,
    staleTime: 10_000
  })
}
