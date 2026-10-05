import type { LogAuditEntry, LogRequest } from '@/features/logs/api/schemas'
import { compareLogInstants } from '@/features/logs/lib/log-instant'

export const LOG_EVENT_CATEGORIES = ['requests', 'system', 'quic', 'gossip', 'iroh'] as const

export type LogEventCategory = (typeof LOG_EVENT_CATEGORIES)[number]
export type OperationalLogEventCategory = Exclude<LogEventCategory, 'requests'>

export type RequestLogEvent = {
  readonly type: 'request'
  readonly id: string
  readonly occurredAt: string
  readonly category: 'requests'
  readonly request: LogRequest
}

export type AuditLogEvent = {
  readonly type: 'audit'
  readonly id: string
  readonly occurredAt: string
  readonly category: OperationalLogEventCategory
  readonly audit: LogAuditEntry
}

export type LogEventLedgerRow = RequestLogEvent | AuditLogEvent

export type LogEventCategoryOption = {
  readonly value: LogEventCategory
  readonly count: number
}

export function classifyAuditCategory(code: string): OperationalLogEventCategory {
  const normalized = code.toLowerCase()
  if (normalized.startsWith('gossip_')) return 'gossip'
  if (normalized.startsWith('mesh_quic_') || normalized.startsWith('mesh_control_')) return 'quic'
  return 'system'
}

export function mergeLogEventWindow(
  requests: readonly LogRequest[],
  audits: readonly LogAuditEntry[],
  limit?: number
): LogEventLedgerRow[] {
  const requestsById = new Map<string, LogRequest>()
  for (const request of requests) {
    const requestId = request.requestId.toString()
    const previous = requestsById.get(requestId)
    if (previous === undefined || (previous.source !== 'active' && request.source === 'active')) {
      requestsById.set(requestId, request)
    }
  }

  const requestRows: RequestLogEvent[] = [...requestsById.values()].map((request) => ({
    type: 'request',
    id: `request:${request.requestId.toString()}`,
    occurredAt: request.createdAt,
    category: 'requests',
    request
  }))
  const auditRows: AuditLogEvent[] = audits.map((audit) => ({
    type: 'audit',
    id: `audit:${audit.entryId}`,
    occurredAt: audit.occurredAt,
    category: classifyAuditCategory(audit.code),
    audit
  }))

  return [...requestRows, ...auditRows]
    .sort(compareLogEventRows)
    .slice(0, limit === undefined ? undefined : Math.max(0, limit))
}

export function filterLogEventRows(
  rows: readonly LogEventLedgerRow[],
  categories: ReadonlySet<LogEventCategory>
): LogEventLedgerRow[] {
  return rows.filter((row) => categories.has(row.category))
}

export function logEventCategoryOptions(rows: readonly LogEventLedgerRow[]): LogEventCategoryOption[] {
  const counts = new Map<LogEventCategory, number>()
  for (const row of rows) counts.set(row.category, (counts.get(row.category) ?? 0) + 1)

  return LOG_EVENT_CATEGORIES.filter((value) => value !== 'iroh' || counts.has('iroh')).map((value) => ({
    value,
    count: counts.get(value) ?? 0
  }))
}

export function formatLogEventTimestamp(value: string): string {
  const timestamp = new Date(value)
  return Number.isNaN(timestamp.getTime()) ? value : timestamp.toLocaleString()
}

function compareLogEventRows(left: LogEventLedgerRow, right: LogEventLedgerRow): number {
  const occurredAtComparison = compareLogInstants(right.occurredAt, left.occurredAt)
  if (occurredAtComparison !== 0) return occurredAtComparison
  if (left.type !== right.type) return left.type === 'request' ? -1 : 1
  return left.id.localeCompare(right.id)
}
