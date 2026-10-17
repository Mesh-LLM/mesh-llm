// Part 3 — column definitions for the
// Exchanges table, mirroring the Logs feature's `buildLogEventLedgerColumns`
// pattern (DataTableColumnHeader for sortable headers, StatusPill for
// state). NON-OVERLAP INVARIANT: every column `id` here must stay disjoint
// from the Logs ledger's column keys except `exchange_id` -- see
// `ledger-logs-non-overlap.test.ts`. No column here may show routing,
// retry, status, or category -- those are Logs concepts.
import type { ColumnDef } from '@/components/ui/data-table'
import { DataTableColumnHeader } from '@/components/ui/data-table-column-header'
import { StatusPill } from '@/components/ui/status-pill'
import { ExchangeIdCell } from '@/features/capsules/components/ExchangeIdCell'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'

export const EXCHANGE_COLUMN_LABELS = {
  time: 'Time',
  exchange_id: 'Exchange ID',
  counterparty: 'Counterparty',
  your_role: 'Your role',
  confirmed: 'Confirmed',
  checks: 'Checks'
} as const

function formatExchangeTimestamp(timestamp: string | null): string {
  if (!timestamp) return 'timestamp unavailable'
  const match = timestamp.match(/T(\d{2}:\d{2}:\d{2})Z?/)
  return match ? `${timestamp.slice(0, 10)} ${match[1]}Z` : timestamp
}

function roleLabel(roleTag: string): string {
  return roleTag === 'SERVED' ? 'Served' : roleTag === 'ASKED' ? 'Asked' : roleTag
}

export function buildExchangeColumns(): ColumnDef<ExchangeLedgerRow>[] {
  return [
    {
      id: 'time',
      accessorFn: (row) => row.timestamp ?? '',
      header: ({ column }) => <DataTableColumnHeader column={column} title={EXCHANGE_COLUMN_LABELS.time} />,
      cell: ({ row }) => (
        <time className="font-mono tabular-nums text-fg-dim" dateTime={row.original.timestamp ?? undefined}>
          {formatExchangeTimestamp(row.original.timestamp)}
        </time>
      )
    },
    {
      id: 'exchange_id',
      accessorFn: (row) => row.exchangeKey,
      header: ({ column }) => <DataTableColumnHeader column={column} title={EXCHANGE_COLUMN_LABELS.exchange_id} />,
      cell: ({ row }) => <ExchangeIdCell exchangeKey={row.original.exchangeKey} />
    },
    {
      id: 'counterparty',
      accessorFn: (row) => row.counterparty ?? '',
      header: ({ column }) => <DataTableColumnHeader column={column} title={EXCHANGE_COLUMN_LABELS.counterparty} />,
      cell: ({ row }) =>
        row.original.counterparty ? (
          <span className="font-mono text-fg-dim">{row.original.counterparty}</span>
        ) : (
          <span className="text-fg-faint">—</span>
        )
    },
    {
      id: 'your_role',
      accessorFn: (row) => roleLabel(row.roleTag),
      header: ({ column }) => <DataTableColumnHeader column={column} title={EXCHANGE_COLUMN_LABELS.your_role} />,
      cell: ({ row }) => <span className="text-fg-dim">{roleLabel(row.original.roleTag)}</span>
    },
    {
      id: 'confirmed',
      accessorFn: (row) => (row.confirmed ? 1 : 0),
      header: ({ column }) => <DataTableColumnHeader column={column} title={EXCHANGE_COLUMN_LABELS.confirmed} />,
      cell: ({ row }) => (
        <StatusPill
          dot
          label={row.original.confirmed ? 'confirmed' : row.original.localOnly ? 'this node only' : 'not yet confirmed'}
          tone={row.original.confirmed ? 'good' : 'neutral'}
        />
      )
    },
    {
      id: 'checks',
      accessorFn: (row) => row.checksText,
      header: ({ column }) => <DataTableColumnHeader column={column} title={EXCHANGE_COLUMN_LABELS.checks} />,
      cell: ({ row }) =>
        row.original.hasIssue ? (
          <StatusPill label={row.original.checksText} tone="bad" />
        ) : (
          <span className="text-fg-faint">{row.original.checksText}</span>
        )
    }
  ]
}
