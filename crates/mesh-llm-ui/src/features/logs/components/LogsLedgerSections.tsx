/* eslint-disable react-refresh/only-export-components */

import { useEffect } from 'react'
import { Activity, CircleCheckBig, CircleX, Database } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { type FilterCategory, type FilterValueOption } from '@/components/ui/FilterPopover'
import { Sparkline } from '@/components/ui/Sparkline'
import type { StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { TanStackTable as DataTableTanStackTableType } from '@/components/ui/data-table'
import type { LogsLiveConnectionState } from '@/features/logs/api/use-logs-live-recovery'
import type { LogRequest } from '@/features/logs/api/schemas'
import {
  formatLogEventTimestamp,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import { buildLogKpiMetrics } from '@/features/logs/lib/log-kpis'
import type { VolumeTimeRangeKey } from '@/features/logs/lib/log-volume'
import { RELATIVE_TIME_PRESETS, toLogsRequestQuery, type LogsLedgerSearch } from '@/features/logs/lib/log-search'

export type LedgerFilterKey = 'category'

export const ledgerFilterCategories: Array<FilterCategory<LedgerFilterKey>> = [{ key: 'category', label: 'Category' }]

export function activeFilterGroupCount(search: LogsLedgerSearch) {
  return [
    search.timeRange ?? (search.from || search.to ? 'window' : undefined),
    search.model,
    search.provider,
    search.engine,
    search.route,
    search.source,
    search.outcome,
    search.categories
  ].filter(Boolean).length
}

export function selectedCategories(
  categories: LogsLedgerSearch['categories'],
  options: readonly FilterValueOption[]
): Set<LogEventCategory> {
  if (categories === 'none') return new Set()
  if (categories) return new Set(categories)
  return new Set(options.map((option) => option.value).filter(isLogEventCategory))
}

export function isLogEventCategory(value: string): value is LogEventCategory {
  return value === 'requests' || value === 'system' || value === 'quic' || value === 'gossip' || value === 'iroh'
}

export function eventSearchText(row: LogEventLedgerRow): string {
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
        row.request.outcome
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
        row.audit.operationId,
        row.audit.requestId,
        row.audit.reasonCode,
        row.audit.outcome,
        String(row.audit.sequence)
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()
  }
}

export function eventRowAriaLabel(row: LogEventLedgerRow): string {
  return row.type === 'request'
    ? `Inspect request ${row.request.requestId.toString()}`
    : `Inspect operational event ${row.audit.code}`
}

export function eventRowId(row: LogEventLedgerRow): string {
  return row.id
}

export function liveStateLabel(state: LogsLiveConnectionState) {
  switch (state) {
    case 'connected':
      return 'Live'
    case 'reconnecting':
      return 'Reconnecting'
    case 'polling':
      return 'Polling'
    case 'gap':
      return 'Recovering gap'
    case 'stale':
      return 'Live data stale'
  }
}

export function liveStateTone(state: LogsLiveConnectionState): StatusBadgeTone {
  switch (state) {
    case 'connected':
      return 'good'
    case 'reconnecting':
    case 'polling':
    case 'gap':
      return 'accent'
    case 'stale':
      return 'warn'
  }
}

/* ------------------------------------------------------------------ */
/* KPI helpers & components                                             */
/* ------------------------------------------------------------------ */

type KpiTileProps = {
  readonly Icon: LucideIcon
  readonly label: string
  readonly valueText: string
  readonly valueColor: string
  readonly secondaryLabel?: string
  readonly sparklineValues: number[]
  readonly sparklineColor: string
  readonly sparklineLabel: string
}

function KpiTile({
  Icon,
  label,
  valueText,
  valueColor,
  secondaryLabel,
  sparklineValues,
  sparklineColor,
  sparklineLabel
}: KpiTileProps) {
  return (
    <div className="panel-shell min-w-0 rounded-[var(--radius-lg)] border border-border bg-panel px-[var(--panel-x)] py-[var(--panel-y)]">
      <div className="flex items-center gap-1.5">
        <Icon className="size-3.5 shrink-0" style={{ color: valueColor }} aria-hidden="true" />
        <span className="type-label truncate text-fg-faint">{label}</span>
      </div>
      <div
        className="mt-[var(--panel-y,12px)] font-mono text-[length:var(--density-type-headline)] font-semibold leading-none tracking-tight"
        style={{ color: valueColor }}
      >
        {valueText}
      </div>
      <Sparkline
        className="mt-[calc(var(--panel-y,12px)*0.667)] h-5 w-full max-w-full"
        values={sparklineValues}
        color={sparklineColor}
        width={200}
        height={20}
        preserveAspectRatio="none"
        strokeWidth={1.5}
        ariaLabel={sparklineLabel}
      />
      {secondaryLabel ? <div className="mt-1 type-caption text-fg-dim">{secondaryLabel}</div> : null}
    </div>
  )
}

export function selectedRange(
  requestScope: Pick<ReturnType<typeof toLogsRequestQuery>, 'from' | 'to'>,
  timeRange: LogsLedgerSearch['timeRange'],
  now: number
): {
  readonly label: string
  readonly rangeMs: number
  readonly endMs: number
  readonly chartRange: VolumeTimeRangeKey
} {
  const preset = RELATIVE_TIME_PRESETS.find((candidate) => candidate.value === (timeRange ?? ''))
  const resolvedFrom = requestScope.from ? Date.parse(requestScope.from) : undefined
  const resolvedTo = requestScope.to ? Date.parse(requestScope.to) : undefined
  const endMs = resolvedTo !== undefined && !Number.isNaN(resolvedTo) ? resolvedTo : now
  const rangeMs =
    resolvedFrom !== undefined && !Number.isNaN(resolvedFrom) ? endMs - resolvedFrom : Number.POSITIVE_INFINITY
  return {
    label: preset?.label ?? (requestScope.from || requestScope.to ? 'Custom time range' : 'Lifetime'),
    rangeMs,
    endMs,
    chartRange:
      timeRange === '1h' || timeRange === '6h' || timeRange === '12h' || timeRange === '24h' || timeRange === '7d'
        ? timeRange
        : requestScope.from || requestScope.to
          ? 'selected'
          : 'all'
  }
}

export function KpiStrip({
  rows,
  complete,
  range
}: {
  readonly rows: readonly LogRequest[]
  readonly complete: boolean
  readonly range: ReturnType<typeof selectedRange>
}) {
  const counts = buildLogKpiMetrics(rows, range.endMs, range.rangeMs)
  const scopeDescription = complete
    ? `Selected range: ${range.label} · retained records only`
    : `Selected range: ${range.label} · first 1,000 retained records`
  const tiles: KpiTileProps[] = [
    {
      Icon: Database,
      label: 'Total',
      valueText: String(counts.totalCount),
      valueColor: 'var(--color-foreground)',
      secondaryLabel: complete ? 'Retained records' : 'First 1,000 records',
      sparklineValues: counts.totalValues,
      sparklineColor: 'var(--color-foreground)',
      sparklineLabel: `Retained request records over ${range.label}`
    },
    {
      Icon: CircleCheckBig,
      label: 'Completed',
      valueText: String(counts.completedCount),
      valueColor: 'var(--color-good)',
      secondaryLabel: counts.completedShare,
      sparklineValues: counts.completedValues,
      sparklineColor: 'var(--color-good)',
      sparklineLabel: `Completed requests trend · ${range.label}`
    },
    {
      Icon: CircleX,
      label: 'Failed',
      valueText: String(counts.failedCount),
      valueColor: 'var(--color-bad)',
      secondaryLabel: counts.failedShare,
      sparklineValues: counts.failedValues,
      sparklineColor: 'var(--color-bad)',
      sparklineLabel: `Failed requests trend · ${range.label}`
    },
    {
      Icon: Activity,
      label: 'Active',
      valueText: String(counts.activeCount),
      valueColor: 'var(--color-accent)',
      secondaryLabel: counts.activeShare,
      sparklineValues: counts.activeValues,
      sparklineColor: 'var(--color-accent)',
      sparklineLabel: `Active requests trend · ${range.label}`
    }
  ]
  return (
    <section aria-labelledby="request-records-heading">
      <div className="mb-3 flex min-w-0 flex-col gap-0.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-4">
        <h2 className="type-panel-title text-foreground" id="request-records-heading">
          Request records
        </h2>
        <p className="type-caption text-fg-dim">{scopeDescription}</p>
      </div>
      <div className="grid grid-cols-1 gap-[calc(var(--shell-normal)*1.25)] sm:grid-cols-2 xl:grid-cols-4">
        {tiles.map((tile) => (
          <KpiTile key={tile.label} {...tile} />
        ))}
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ */
/* Table capture helper                                                */
/* ------------------------------------------------------------------ */

type TableCaptureProps = {
  readonly table: DataTableTanStackTableType<LogEventLedgerRow>
  readonly onCapture: (table: DataTableTanStackTableType<LogEventLedgerRow> | null) => void
}

export function TableCapture({ table, onCapture }: TableCaptureProps) {
  useEffect(() => {
    onCapture(table)
    return () => onCapture(null)
  }, [table, onCapture])
  return null
}

/* ------------------------------------------------------------------ */
/* LogsLedger                                                          */
/* ------------------------------------------------------------------ */
