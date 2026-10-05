import type { ReactNode } from 'react'
import type { LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'
import { logEventOriginLines } from '@/features/logs/lib/log-event-origin'

export function LogEventLedgerOrigin({ row }: { readonly row: LogEventLedgerRow }): ReactNode {
  const origin = logEventOriginLines(row)
  switch (row.type) {
    case 'request': {
      return (
        <div className="flex min-w-0 flex-col font-mono">
          <div className="break-words text-[length:var(--density-type-caption)] text-foreground" data-log-origin-caller>
            {origin.identity}
          </div>
          {origin.path ? (
            <div className="type-caption text-fg-faint" data-log-origin-path>
              {origin.path}
            </div>
          ) : null}
        </div>
      )
    }
    case 'audit':
      return <span className="font-mono text-fg-dim">{origin.identity}</span>
    default:
      return assertNever(row)
  }
}

function assertNever(value: never): never {
  throw new Error(`Unhandled log event origin: ${String(value)}`)
}
