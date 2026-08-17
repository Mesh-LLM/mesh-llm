import type { LogLifecycleEvent, LogProxyAttempt } from '@/features/logs/api/schemas'
import {
  DEFAULT_LOG_REQUEST_DETAIL_TAB,
  type LogRequestDetailTab,
  normalizeLogRequestDetailTab
} from '@/features/logs/lib/log-inspector'
import { sortByOccurredAt } from '@/features/logs/lib/log-instant'
import { type LogsLedgerSearch, parseLogsLedgerSearch } from '@/features/logs/lib/log-search'

export { isLogRequestDetailTab, type LogRequestDetailTab } from '@/features/logs/lib/log-inspector'

export type LogRequestDetailsSearch = LogsLedgerSearch & {
  readonly tab?: LogRequestDetailTab
}

export function parseLogRequestDetailsSearch(search: Record<string, unknown>): LogRequestDetailsSearch {
  const ledgerSearch = parseLogsLedgerSearch(search)
  const tab = normalizeLogRequestDetailTab(search.tab) ?? DEFAULT_LOG_REQUEST_DETAIL_TAB
  return { ...ledgerSearch, tab }
}

export function ledgerSearchFromDetails(search: LogRequestDetailsSearch): LogsLedgerSearch {
  const { tab: _tab, ...ledgerSearch } = search
  return ledgerSearch
}

export function sortLifecycleEvents(events: readonly LogLifecycleEvent[]): LogLifecycleEvent[] {
  return sortByOccurredAt(events)
}

export function sortProxyAttempts(attempts: readonly LogProxyAttempt[]): LogProxyAttempt[] {
  return sortByOccurredAt(attempts)
}

export function isStreamEvent(event: LogLifecycleEvent): boolean {
  return (
    event.kind === 'stream_started' ||
    event.kind === 'stream_chunk' ||
    event.kind === 'stream_completed' ||
    event.kind === 'stream_error'
  )
}

export function isErrorEvent(event: LogLifecycleEvent): boolean {
  return (
    event.kind === 'attempt_failed' ||
    event.kind === 'stream_error' ||
    event.kind === 'audit_error' ||
    event.kind === 'failed'
  )
}

export function artifactMatchesTab(kind: string, tab: 'request' | 'response' | 'errors'): boolean {
  const normalized = kind.toLowerCase()
  return normalized.includes(tab === 'errors' ? 'error' : tab)
}
