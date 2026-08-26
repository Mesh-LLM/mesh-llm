import { formatNetworkPathType, formatRequestCaller } from '@/features/logs/lib/log-client-info'
import type { LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'

export type LogEventOriginLines = {
  readonly identity: string
  readonly path?: string
}

export function logEventOriginLines(row: LogEventLedgerRow): LogEventOriginLines {
  switch (row.type) {
    case 'request': {
      const provider = row.request.provider === 'mesh' ? row.request.provider : undefined
      const caller = formatRequestCaller(row.request)
      const path = row.request.callerPathType ? formatNetworkPathType(row.request.callerPathType) : undefined
      const pathSuffix = path ? ` · ${path}` : undefined
      const callerIdentity = pathSuffix && caller?.endsWith(pathSuffix) ? caller.slice(0, -pathSuffix.length) : caller
      const identity =
        [provider, callerIdentity]
          .filter((value): value is string => value !== undefined && value.length > 0)
          .join(' · ') || '—'

      return path ? { identity, path } : { identity }
    }
    case 'audit':
      return { identity: row.audit.source }
    default:
      return assertNever(row)
  }
}

export function logEventOriginLabel(row: LogEventLedgerRow): string {
  const { identity, path } = logEventOriginLines(row)
  return path ? `${identity} · ${path}` : identity
}

function assertNever(value: never): never {
  throw new Error(`Unhandled log event origin: ${String(value)}`)
}
