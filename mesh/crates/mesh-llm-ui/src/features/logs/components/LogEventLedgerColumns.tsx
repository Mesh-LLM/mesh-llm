import { CircleCheckBig, CircleSlash, CircleX, Info, LoaderCircle, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { DataTableColumnHeader } from '@/components/ui/data-table-column-header'
import type { ColumnDef } from '@/components/ui/data-table'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { LogAuditEntry, LogAuditSeverity, LogOutcome } from '@/features/logs/api/schemas'
import { LogEventCategoryBadge } from '@/features/logs/components/LogEventCategoryBadge'
import { LogEventLedgerOrigin } from '@/features/logs/components/LogEventLedgerOrigin'
import { formatLogEventTimestamp, type LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'
import { logEventOriginLabel } from '@/features/logs/lib/log-event-origin'

export const LOG_EVENT_LEDGER_COLUMN_LABELS = {
  occurredAt: 'Occurred',
  category: 'Category',
  event: 'Event',
  context: 'Context',
  origin: 'Origin',
  state: 'State'
} as const

export function buildLogEventLedgerColumns(): ColumnDef<LogEventLedgerRow>[] {
  return [
    {
      accessorKey: 'occurredAt',
      header: ({ column }) => (
        <DataTableColumnHeader column={column} title={LOG_EVENT_LEDGER_COLUMN_LABELS.occurredAt} />
      ),
      cell: ({ row }) => (
        <time className="font-mono tabular-nums text-fg-dim" dateTime={row.original.occurredAt}>
          {formatLogEventTimestamp(row.original.occurredAt)}
        </time>
      )
    },
    {
      accessorKey: 'category',
      header: ({ column }) => <DataTableColumnHeader column={column} title={LOG_EVENT_LEDGER_COLUMN_LABELS.category} />,
      cell: ({ row }) => <LogEventCategoryBadge category={row.original.category} />
    },
    {
      id: 'state',
      accessorFn: stateLabel,
      header: ({ column }) => <DataTableColumnHeader column={column} title={LOG_EVENT_LEDGER_COLUMN_LABELS.state} />,
      cell: ({ row }) => stateCell(row.original)
    },
    {
      id: 'origin',
      accessorFn: logEventOriginLabel,
      header: ({ column }) => <DataTableColumnHeader column={column} title={LOG_EVENT_LEDGER_COLUMN_LABELS.origin} />,
      cell: ({ row }) => <LogEventLedgerOrigin row={row.original} />
    },
    {
      id: 'event',
      accessorFn: eventLabel,
      header: ({ column }) => <DataTableColumnHeader column={column} title={LOG_EVENT_LEDGER_COLUMN_LABELS.event} />,
      cell: ({ row }) => eventCell(row.original)
    },
    {
      id: 'context',
      accessorFn: contextLabel,
      header: ({ column }) => <DataTableColumnHeader column={column} title={LOG_EVENT_LEDGER_COLUMN_LABELS.context} />,
      cell: ({ row }) => contextCell(row.original)
    }
  ]
}

function eventCell(row: LogEventLedgerRow): ReactNode {
  switch (row.type) {
    case 'request':
      return (
        <div className="min-w-0 break-all font-mono font-medium text-foreground">
          {row.request.requestId.toString()}
        </div>
      )
    case 'audit':
      return <div className="min-w-0 break-words font-mono font-medium text-foreground">{row.audit.code}</div>
    default:
      return assertNever(row)
  }
}

function contextCell(row: LogEventLedgerRow): ReactNode {
  switch (row.type) {
    case 'request':
      return machineFields([
        { label: 'Model', value: row.request.model },
        { label: 'Route', value: row.request.route }
      ])
    case 'audit': {
      const typedFields = [
        ...(row.audit.subjectId ? [{ label: subjectLabel(row.audit.subjectKind), value: row.audit.subjectId }] : []),
        ...(row.audit.operationId ? [{ label: 'Operation', value: row.audit.operationId }] : []),
        ...(row.audit.requestId ? [{ label: 'Request', value: row.audit.requestId }] : [])
      ]
      return machineFields(
        typedFields.length > 0
          ? typedFields
          : [
              { label: 'Sequence', value: String(row.audit.sequence) },
              { label: 'Entry ID', value: row.audit.entryId }
            ]
      )
    }
    default:
      return assertNever(row)
  }
}

function stateCell(row: LogEventLedgerRow): ReactNode {
  switch (row.type) {
    case 'request': {
      return (
        <StatusBadge size="caption" tone={requestTone(row.request.outcome)}>
          {outcomeIcon(row.request.outcome)}
          {row.request.outcome}
        </StatusBadge>
      )
    }
    case 'audit':
      return (
        <StatusBadge size="caption" tone={auditTone(row.audit.severity)}>
          {row.audit.severity === undefined ? null : auditSeverityIcon(row.audit.severity)}
          {row.audit.severity ?? 'Not provided'}
        </StatusBadge>
      )
    default:
      return assertNever(row)
  }
}

type MachineField = {
  readonly label: string
  readonly value: string | undefined
}

function machineFields(fields: readonly MachineField[]): ReactNode {
  return (
    <div className="min-w-0 space-y-1 font-mono">
      {fields.map((field) => (
        <div className="flex min-w-0 items-baseline gap-1.5" key={field.label}>
          <span className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.07em] text-primary">
            {field.label}
          </span>
          <span className="break-words text-fg-dim">{field.value ?? '—'}</span>
        </div>
      ))}
    </div>
  )
}

function eventLabel(row: LogEventLedgerRow): string {
  return row.type === 'request' ? row.request.requestId.toString() : row.audit.code
}

function contextLabel(row: LogEventLedgerRow): string {
  return row.type === 'request'
    ? `${row.request.model ?? ''} ${row.request.route ?? ''}`
    : `${row.audit.subjectId ?? ''} ${row.audit.operationId ?? ''} ${row.audit.requestId ?? ''} ${row.audit.sequence} ${row.audit.entryId}`
}

function subjectLabel(kind: LogAuditEntry['subjectKind']): string {
  switch (kind) {
    case 'model':
      return 'Model'
    case 'runtime_instance':
      return 'Instance'
    case 'cli_command':
      return 'Command'
    case 'runtime':
      return 'Runtime'
    case 'mesh_peer':
      return 'Peer'
    default:
      return 'Subject'
  }
}

function stateLabel(row: LogEventLedgerRow): string {
  return row.type === 'request' ? row.request.outcome : (row.audit.severity ?? '')
}

function requestTone(outcome: LogOutcome): StatusBadgeTone {
  switch (outcome) {
    case 'active':
      return 'accent'
    case 'completed':
      return 'good'
    case 'failed':
    case 'rejected':
    case 'dropped':
      return 'bad'
    case 'cancelled':
      return 'warn'
    default:
      return assertNever(outcome)
  }
}

function outcomeIcon(outcome: LogOutcome): ReactNode {
  switch (outcome) {
    case 'completed':
      return <CircleCheckBig aria-hidden="true" className="size-3" />
    case 'failed':
    case 'rejected':
    case 'dropped':
      return <CircleX aria-hidden="true" className="size-3" />
    case 'cancelled':
      return <CircleSlash aria-hidden="true" className="size-3" />
    case 'active':
      return <LoaderCircle aria-hidden="true" className="size-3 animate-spin motion-reduce:animate-none" />
    default:
      return assertNever(outcome)
  }
}

function auditSeverityIcon(severity: LogAuditSeverity): ReactNode {
  switch (severity) {
    case 'info':
      return <Info aria-hidden="true" className="size-3" />
    case 'warning':
      return <TriangleAlert aria-hidden="true" className="size-3" />
    case 'error':
      return <CircleX aria-hidden="true" className="size-3" />
    default:
      return assertNever(severity)
  }
}

function auditTone(severity: LogAuditSeverity | undefined): StatusBadgeTone {
  switch (severity) {
    case 'info':
    case undefined:
      return 'muted'
    case 'warning':
      return 'warn'
    case 'error':
      return 'bad'
    default:
      return assertNever(severity)
  }
}

function assertNever(value: never): never {
  throw new Error(`Unhandled log event value: ${String(value)}`)
}
