import { formatLogEventTimestamp, type LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'
import { formatEndpointId } from '@/features/logs/lib/log-client-info'

export function logEventSearchText(row: LogEventLedgerRow): string {
  switch (row.type) {
    case 'request':
      return [
        row.request.requestId.toString(),
        row.occurredAt,
        formatLogEventTimestamp(row.occurredAt),
        row.category,
        row.request.model,
        row.request.provider,
        row.request.engine,
        row.request.route,
        row.request.source,
        row.request.outcome,
        row.request.callerEndpointId,
        row.request.callerEndpointId ? formatEndpointId(row.request.callerEndpointId) : undefined,
        row.request.callerAddr,
        row.request.callerPathType
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
    case 'audit':
      return [
        row.audit.entryId,
        row.occurredAt,
        formatLogEventTimestamp(row.occurredAt),
        row.category,
        row.audit.code,
        row.audit.source,
        row.audit.severity,
        row.audit.subjectKind,
        row.audit.subjectId,
        row.audit.subjectKind === 'mesh_peer' && row.audit.subjectId
          ? formatEndpointId(row.audit.subjectId)
          : undefined,
        row.audit.remoteAddr,
        row.audit.pathType,
        row.audit.operationId,
        row.audit.requestId,
        row.audit.reasonCode,
        row.audit.outcome,
        String(row.audit.sequence)
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
    default:
      return assertNever(row)
  }
}

function assertNever(value: never): never {
  throw new Error(`Unhandled log event search row: ${String(value)}`)
}
