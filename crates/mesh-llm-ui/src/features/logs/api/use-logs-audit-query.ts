import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { LogsApiClient, type LogAuditQuery, type LogsCapability } from '@/features/logs/api/client'
import type { LogAuditEntry, LogAuditPage } from '@/features/logs/api/schemas'
import { logsKeys } from '@/features/logs/api/use-logs-ledger-query'
import { useDataMode, type DataMode } from '@/lib/data-mode'

export const AUDIT_PAGE_SIZE = 100
export const AUDIT_MAX_RECORDS = 1_000
const AUDIT_MAX_PAGE_REQUESTS = 100

export async function loadCompleteAudits(query: LogAuditQuery, mode: DataMode): Promise<LogsCapability<LogAuditPage>> {
  const client = new LogsApiClient()
  const items: LogAuditEntry[] = []
  const entryIds = new Set<string>()
  const requestedCursors = new Set<string>()
  let cursor = query.cursor
  let pageRequests = 0
  let truncatedAtCap = false

  while (items.length < AUDIT_MAX_RECORDS && pageRequests < AUDIT_MAX_PAGE_REQUESTS) {
    const cursorKey = cursor?.toString() ?? '<initial>'
    if (requestedCursors.has(cursorKey)) {
      return { state: 'supported', value: { items, nextCursor: cursor, incomplete: true } }
    }
    requestedCursors.add(cursorKey)
    pageRequests += 1

    const result = await client.listAudits({ ...query, cursor, limit: AUDIT_PAGE_SIZE }, mode)
    if (result.state === 'unsupported') return result

    for (const entry of result.value.items) {
      if (entryIds.has(entry.entryId)) continue
      if (items.length === AUDIT_MAX_RECORDS) {
        truncatedAtCap = true
        break
      }
      entryIds.add(entry.entryId)
      items.push(entry)
    }

    cursor = result.value.nextCursor
    if (cursor === undefined) {
      return {
        state: 'supported',
        value: { items, nextCursor: undefined, ...(truncatedAtCap ? { incomplete: true } : {}) }
      }
    }
  }

  return { state: 'supported', value: { items, nextCursor: cursor, incomplete: true } }
}

export function useLogsAuditQuery(query: LogAuditQuery = {}) {
  const dataMode = useDataMode()
  const retainedSuccessfulData = useRef<LogsCapability<LogAuditPage> | undefined>(undefined)
  const result = useQuery({
    queryKey: logsKeys.audit(query, dataMode.mode),
    queryFn: () => loadCompleteAudits(query, dataMode.mode),
    placeholderData: (previousData) => previousData ?? retainedSuccessfulData.current,
    staleTime: 10_000
  })

  useEffect(() => {
    if (result.data !== undefined && !result.isPlaceholderData) retainedSuccessfulData.current = result.data
  }, [result.data, result.isPlaceholderData])

  return result
}
