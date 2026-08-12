import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ArrowLeftRight,
  Calendar,
  CircleCheckBig,
  CircleX,
  Database,
  RotateCcw,
  Search as SearchIcon,
  X,
  Logs
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { FilterPopover, type FilterCategory, type FilterValueOption } from '@/components/ui/FilterPopover'
import { InfoBanner } from '@/components/ui/InfoBanner'
import { Input } from '@/components/ui/input'
import { NativeSelect } from '@/components/ui/NativeSelect'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Sparkline } from '@/components/ui/Sparkline'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import { DataTable, TanStackTable as DataTableTanStackTableType } from '@/components/ui/data-table'
import { DataTablePagination } from '@/components/ui/data-table-pagination'
import { DataTableViewOptions } from '@/components/ui/data-table-view-options'
import { useLogsAuditQuery } from '@/features/logs/api/use-logs-audit-query'
import { useLogsLedgerQuery } from '@/features/logs/api/use-logs-ledger-query'
import { useLogsLiveRecovery, type LogsLiveConnectionState } from '@/features/logs/api/use-logs-live-recovery'
import {
  buildLogEventLedgerColumns,
  LOG_EVENT_LEDGER_COLUMN_LABELS
} from '@/features/logs/components/LogEventLedgerColumns'
import { LogEventInspector } from '@/features/logs/components/LogEventInspector'
import { LogOperations } from '@/features/logs/components/LogOperations'
import { RequestsOverTimeChart } from '@/features/logs/components/RequestsOverTimeChart'
import type { LogRequest } from '@/features/logs/api/schemas'
import type { LoggingStatus } from '@/lib/api/types'
import {
  filterLogEventRows,
  formatLogEventTimestamp,
  logEventCategoryOptions,
  mergeLogEventWindow,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import { compareLogInstants } from '@/features/logs/lib/log-instant'
import { buildLogKpiMetrics } from '@/features/logs/lib/log-kpis'
import type { VolumeTimeRangeKey } from '@/features/logs/lib/log-volume'
import { useAdvancingChartClock } from '@/features/logs/lib/use-advancing-chart-clock'
import {
  RELATIVE_TIME_PRESETS,
  closeLogInspector,
  logInspectorFromSearch,
  openLogInspector,
  resetLogsSearch,
  toLogsRequestQuery,
  updateLogCategories,
  updateLogsTimeRange,
  type LogsLedgerSearch
} from '@/features/logs/lib/log-search'

type LedgerFilterKey = 'category'

type LogsLedgerProps = {
  readonly search: LogsLedgerSearch
  readonly onSearchChange: (search: LogsLedgerSearch) => void
  readonly onMaintenanceMutationSucceeded?: () => void
  readonly loggingStatus?: LoggingStatus
}

const ledgerFilterCategories: Array<FilterCategory<LedgerFilterKey>> = [{ key: 'category', label: 'Category' }]

function activeFilterGroupCount(search: LogsLedgerSearch) {
  return [
    search.timeRange,
    search.model,
    search.provider,
    search.engine,
    search.route,
    search.source,
    search.outcome,
    search.categories
  ].filter(Boolean).length
}

function selectedCategories(search: LogsLedgerSearch, options: readonly FilterValueOption[]): Set<LogEventCategory> {
  if (search.categories === 'none') return new Set()
  if (search.categories) return new Set(search.categories)
  return new Set(options.map((option) => option.value).filter(isLogEventCategory))
}

function isLogEventCategory(value: string): value is LogEventCategory {
  return value === 'requests' || value === 'system' || value === 'quic' || value === 'gossip' || value === 'iroh'
}

function eventSearchText(row: LogEventLedgerRow): string {
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

function eventRowAriaLabel(row: LogEventLedgerRow): string {
  return row.type === 'request'
    ? `Inspect request ${row.request.requestId.toString()}`
    : `Inspect operational event ${row.audit.code}`
}

function liveStateLabel(state: LogsLiveConnectionState) {
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

function liveStateTone(state: LogsLiveConnectionState): StatusBadgeTone {
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

function selectedRange(
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
    label: preset?.label ?? (requestScope.from || requestScope.to ? 'Custom time range' : 'All time'),
    rangeMs,
    endMs,
    chartRange:
      timeRange === '1h' || timeRange === '6h' || timeRange === '24h' || timeRange === '7d'
        ? timeRange
        : requestScope.from || requestScope.to
          ? 'selected'
          : 'all'
  }
}

function KpiStrip({
  rows,
  complete,
  range
}: {
  readonly rows: readonly LogRequest[]
  readonly complete: boolean
  readonly range: ReturnType<typeof selectedRange>
}) {
  const counts = buildLogKpiMetrics(rows, range.endMs, range.rangeMs)
  const tiles: KpiTileProps[] = [
    {
      Icon: Database,
      label: 'Total requests',
      valueText: String(counts.totalCount),
      valueColor: 'var(--color-foreground)',
      secondaryLabel: complete ? range.label : `First 1,000 matching records · ${range.label}`,
      sparklineValues: counts.totalValues,
      sparklineColor: 'var(--color-foreground)',
      sparklineLabel: `Total requests trend · ${range.label}`
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
    <section
      className="grid grid-cols-1 gap-[calc(var(--shell-normal)*1.25)] sm:grid-cols-2 xl:grid-cols-4"
      aria-label="Request summary"
    >
      {tiles.map((tile) => (
        <KpiTile key={tile.label} {...tile} />
      ))}
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

function TableCapture({ table, onCapture }: TableCaptureProps) {
  useEffect(() => {
    onCapture(table)
    return () => onCapture(null)
  }, [table, onCapture])
  return null
}

/* ------------------------------------------------------------------ */
/* LogsLedger                                                          */
/* ------------------------------------------------------------------ */

export function LogsLedger({ search, onSearchChange, onMaintenanceMutationSucceeded, loggingStatus }: LogsLedgerProps) {
  const ledgerNow = useAdvancingChartClock()
  const requestScope = useMemo(() => toLogsRequestQuery(search, ledgerNow), [ledgerNow, search])
  const auditBounds = useMemo(
    () => ({ from: requestScope.from, to: requestScope.to }),
    [requestScope.from, requestScope.to]
  )
  const requestQuery = useLogsLedgerQuery(requestScope)
  const auditQuery = useLogsAuditQuery(auditBounds)
  const { refetch } = requestQuery
  const { refetch: refetchAudit } = auditQuery
  const requestResult = requestQuery.data
  const auditResult = auditQuery.data
  const hydrate = useCallback(async () => refetch(), [refetch])
  const hydrateAudit = useCallback(async () => refetchAudit(), [refetchAudit])
  const live = useLogsLiveRecovery({
    enabled: requestResult?.state === 'supported',
    auditEnabled: auditResult?.state === 'supported',
    search,
    hydrate,
    hydrateAudit
  })
  const liveStatusLabel = requestQuery.isFetching
    ? 'Updating'
    : live.state === 'polling' && !live.pollingEnabled
      ? 'Polling paused'
      : liveStateLabel(live.state)
  const liveStatusTone = requestQuery.isFetching
    ? 'accent'
    : live.state === 'polling' && !live.pollingEnabled
      ? 'muted'
      : liveStateTone(live.state)
  const allRequestRows = useMemo(
    () => (requestResult?.state === 'supported' ? requestResult.value.items : []),
    [requestResult]
  )
  const [showManagementRecords, setShowManagementRecords] = useState(false)
  const requestRows = useMemo(
    () =>
      showManagementRecords ? allRequestRows : allRequestRows.filter((row) => !row.route?.startsWith('management_')),
    [allRequestRows, showManagementRecords]
  )
  const selectedLedgerRange = useMemo(
    () => selectedRange(requestScope, search.timeRange, ledgerNow),
    [ledgerNow, requestScope, search.timeRange]
  )
  const auditEntries = useMemo(() => (auditResult?.state === 'supported' ? auditResult.value.items : []), [auditResult])
  const filteredAuditEntries = useMemo(() => {
    if (!auditBounds.from && !auditBounds.to) return auditEntries
    return auditEntries.filter(
      (entry) =>
        (auditBounds.from === undefined || compareLogInstants(entry.occurredAt, auditBounds.from) >= 0) &&
        (auditBounds.to === undefined || compareLogInstants(entry.occurredAt, auditBounds.to) <= 0)
    )
  }, [auditBounds.from, auditBounds.to, auditEntries])
  const mergedRows = useMemo(
    () => mergeLogEventWindow(requestRows, filteredAuditEntries),
    [filteredAuditEntries, requestRows]
  )
  const categoryOptions = useMemo<FilterValueOption[]>(() => logEventCategoryOptions(mergedRows), [mergedRows])
  const selectedCategoryValues = useMemo(() => selectedCategories(search, categoryOptions), [categoryOptions, search])
  const optionsByCategory = useMemo<Record<LedgerFilterKey, FilterValueOption[]>>(
    () => ({ category: categoryOptions }),
    [categoryOptions]
  )
  const selectedValuesByCategory = useMemo<Record<LedgerFilterKey, Set<string>>>(
    () => ({ category: selectedCategoryValues }),
    [selectedCategoryValues]
  )
  const categoryFilterGroups = search.categories === undefined ? 0 : 1
  const activeFilterGroups = activeFilterGroupCount(search)
  const categoryRows = useMemo(
    () => filterLogEventRows(mergedRows, selectedCategoryValues),
    [mergedRows, selectedCategoryValues]
  )
  const columns = useMemo(() => buildLogEventLedgerColumns(), [])
  const [table, setTable] = useState<DataTableTanStackTableType<LogEventLedgerRow> | null>(null)
  const [eventQuery, setEventQuery] = useState('')
  const tableRegionRef = useRef<HTMLDivElement>(null)
  const restoredFocusIdRef = useRef<string | undefined>(undefined)
  const trimmedQuery = useMemo(() => eventQuery.trim().toLowerCase(), [eventQuery])
  const visibleRows = useMemo(
    () => (trimmedQuery ? categoryRows.filter((row) => eventSearchText(row).includes(trimmedQuery)) : categoryRows),
    [categoryRows, trimmedQuery]
  )
  const hasSupportedWindow = requestResult?.state === 'supported' || auditResult?.state === 'supported'
  const requestedInspector = logInspectorFromSearch(search)
  const inspector =
    requestedInspector?.type === 'request' && requestResult?.state !== 'supported' ? undefined : requestedInspector

  const handleSetTable = useCallback((next: DataTableTanStackTableType<LogEventLedgerRow> | null) => {
    setTable(next)
  }, [])
  const handleEventOpen = useCallback(
    (event: LogEventLedgerRow) => {
      const nextSearch =
        event.type === 'request' ? { ...search, focusRequestId: event.request.requestId.toString() } : search
      const inspectorSelection =
        event.type === 'request'
          ? { type: 'request' as const, id: event.request.requestId.toString() }
          : { type: 'audit' as const, id: event.audit.entryId }
      onSearchChange(openLogInspector(nextSearch, inspectorSelection))
    },
    [onSearchChange, search]
  )

  useEffect(() => {
    if (inspector) return
    if (!search.focusRequestId) {
      restoredFocusIdRef.current = undefined
      return
    }
    if (restoredFocusIdRef.current === search.focusRequestId || visibleRows.length === 0) return
    const activeElement = document.activeElement
    if (activeElement instanceof HTMLInputElement || activeElement instanceof HTMLTextAreaElement) return
    const label = `Inspect request ${search.focusRequestId}`
    const rows = tableRegionRef.current?.querySelectorAll<HTMLElement>('tr[aria-label]') ?? []
    const row = [...rows].find((element) => element.getAttribute('aria-label') === label)
    if (!row) return
    restoredFocusIdRef.current = search.focusRequestId
    row.focus()
  }, [inspector, search.focusRequestId, visibleRows.length])

  return (
    <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-[calc(var(--shell-normal)*2)]">
      <InfoBanner
        action={
          requestResult?.state === 'supported' ? (
            <LogOperations
              operation="cleanup"
              onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded}
              query={requestScope}
            />
          ) : undefined
        }
        actionClassName="basis-full justify-start pl-[58px] pt-1 sm:basis-auto sm:justify-end sm:pl-0 sm:pt-0"
        className="flex-wrap items-start sm:flex-nowrap sm:items-center"
        description="Monitor request activity and operational events from this MeshLLM host."
        leadingIcon={<Logs aria-hidden="true" className="size-4" />}
        leadingIconClassName="size-[38px]"
        status={
          hasSupportedWindow ? (
            <div aria-live="polite" className="flex flex-wrap items-center gap-2">
              {requestResult?.state === 'supported' ? (
                live.state === 'polling' ? (
                  <button
                    aria-label="Fallback log polling"
                    aria-pressed={live.pollingEnabled}
                    className="inline-flex cursor-pointer appearance-none rounded-full border-0 bg-transparent p-0 text-inherit focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                    onClick={live.togglePolling}
                    type="button"
                  >
                    <StatusBadge dot size="caption" tone={liveStatusTone}>
                      {liveStatusLabel}
                    </StatusBadge>
                  </button>
                ) : (
                  <StatusBadge dot size="caption" tone={liveStatusTone}>
                    {liveStatusLabel}
                  </StatusBadge>
                )
              ) : null}
              <StatusBadge tone="muted" size="caption">
                Local only
              </StatusBadge>
              {loggingStatus ? (
                <StatusBadge
                  size="caption"
                  tone={
                    loggingStatus.capture_mode === 'redacted_artifacts' && !loggingStatus.artifact_capture_ready
                      ? 'warn'
                      : 'muted'
                  }
                >
                  {captureStatusLabel(loggingStatus)}
                </StatusBadge>
              ) : null}
            </div>
          ) : undefined
        }
        title="System logs"
        titleId="logs-ledger-title"
        titleLevel="h1"
      />

      {requestResult?.state === 'supported' ? (
        <RequestsOverTimeChart
          now={selectedLedgerRange.endMs}
          rows={requestRows}
          selectedRange={selectedLedgerRange.chartRange}
          selectedRangeMs={selectedLedgerRange.rangeMs}
        />
      ) : null}

      {requestResult?.state === 'supported' ? (
        <KpiStrip complete={!requestResult.value.incomplete} range={selectedLedgerRange} rows={requestRows} />
      ) : null}

      {requestResult?.state === 'supported' && requestResult.value.incomplete ? (
        <Alert className="border-warn/40 bg-warn/5" role="status">
          <AlertTitle>Ledger window is bounded</AlertTitle>
          <AlertDescription>
            The server returned more than 1,000 matching records. The table, chart, and KPIs show the first 1,000 only;
            narrow the filters for complete totals.
          </AlertDescription>
        </Alert>
      ) : null}

      {auditResult?.state === 'supported' && auditResult.value.incomplete ? (
        <Alert className="border-warn/40 bg-warn/5" role="status">
          <AlertTitle>Operational window is bounded</AlertTitle>
          <AlertDescription>
            The server returned more than 1,000 matching operational records. The unified table shows the first 1,000
            only; narrow the time range for a complete operational window.
          </AlertDescription>
        </Alert>
      ) : null}

      {requestQuery.isLoading && auditQuery.isLoading ? (
        <div
          role="status"
          className="panel-shell min-h-[14rem] rounded-[var(--radius)] border border-border bg-panel p-[var(--panel-x)]"
        >
          <div className="type-label text-fg-faint">Loading event ledger</div>
          <p className="type-body mt-2 text-fg-dim">Retrieving the independent request and operational windows.</p>
        </div>
      ) : null}

      {requestQuery.isError ? (
        <Alert
          className="panel-shell rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-bad)_35%,var(--color-border))] bg-panel p-[var(--panel-x)]"
          variant="destructive"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <AlertTitle className="type-panel-title text-foreground">Request window could not be loaded</AlertTitle>
              <AlertDescription className="type-caption mt-1 text-fg-dim">
                The local logging service did not return a usable response.
              </AlertDescription>
            </div>
            <Button
              className="ui-control gap-1.5"
              onClick={() => void requestQuery.refetch()}
              size="sm"
              variant="outline"
            >
              Retry requests
            </Button>
          </div>
        </Alert>
      ) : null}

      {auditQuery.isError ? (
        <Alert
          className="panel-shell rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-bad)_35%,var(--color-border))] bg-panel p-[var(--panel-x)]"
          variant="destructive"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <AlertTitle className="type-panel-title text-foreground">
                Operational window could not be loaded
              </AlertTitle>
              <AlertDescription className="type-caption mt-1 text-fg-dim">
                The local logging service did not return a usable operational response.
              </AlertDescription>
            </div>
            <Button
              className="ui-control gap-1.5"
              onClick={() => void auditQuery.refetch()}
              size="sm"
              variant="outline"
            >
              Retry operations
            </Button>
          </div>
        </Alert>
      ) : null}

      {requestResult?.state === 'unsupported' ? (
        <div
          className="panel-shell rounded-[var(--radius)] border border-border bg-panel p-[var(--panel-x)]"
          role="status"
        >
          <div className="type-panel-title text-foreground">Request window unavailable</div>
          <p className="type-body mt-1 max-w-[68ch] text-fg-dim">
            This MeshLLM host does not expose the local logs API. Upgrade the host to inspect request history here.
          </p>
        </div>
      ) : null}

      {auditResult?.state === 'unsupported' ? (
        <div
          className="panel-shell rounded-[var(--radius)] border border-border bg-panel p-[var(--panel-x)]"
          role="status"
        >
          <div className="type-panel-title text-foreground">Operational window unavailable</div>
          <p className="type-body mt-1 max-w-[68ch] text-fg-dim">
            This host does not expose bounded operational audit entries. Request events remain available when supported.
          </p>
        </div>
      ) : null}

      {hasSupportedWindow ? (
        <Card className="overflow-hidden rounded-[var(--radius-lg)] border-border bg-panel shadow-none">
          <section
            aria-label="Event log controls"
            className="flex flex-wrap items-start justify-between gap-3 border-b border-border-soft px-4 py-3"
          >
            <div className="min-w-0">
              <p className="type-caption font-mono text-fg-dim">
                {visibleRows.length === mergedRows.length
                  ? visibleRows.length
                  : `${visibleRows.length} of ${mergedRows.length}`}{' '}
                events in this bounded loaded window
              </p>
              <p className="mt-1 type-caption text-fg-faint">
                {requestRows.length} request records and {filteredAuditEntries.length} operational records load
                independently.
              </p>
              <p className="mt-1 flex items-center gap-1 type-caption text-fg-faint lg:hidden">
                <ArrowLeftRight aria-hidden="true" className="size-3" />
                Scroll horizontally for all columns.
              </p>
            </div>
            <div className="flex w-full min-w-0 flex-wrap items-center gap-2 sm:w-auto">
              <div className="relative min-w-0 basis-full sm:w-64 sm:basis-auto">
                <SearchIcon
                  className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-faint"
                  aria-hidden="true"
                />
                <Input
                  aria-label="Search loaded event window"
                  className="ui-control h-8 w-full rounded-[var(--radius)] border-border-soft pl-9 pr-9 text-[length:var(--density-type-caption)]"
                  value={eventQuery}
                  onChange={(event) => setEventQuery(event.target.value)}
                  placeholder="Search ID, code, model, source..."
                />
                {eventQuery ? (
                  <Button
                    aria-label="Clear event search"
                    onClick={() => setEventQuery('')}
                    size="icon"
                    variant="ghost"
                    className="ui-control-ghost absolute right-1.5 top-1/2 h-7 w-7 -translate-y-1/2 rounded-[var(--radius-sm)] text-fg-faint hover:text-foreground"
                  >
                    <X className="size-3.5" aria-hidden="true" />
                  </Button>
                ) : null}
              </div>
              <div className="flex items-center gap-1.5 text-[length:var(--density-type-caption)] text-fg-dim">
                <Calendar className="size-3.5 text-fg-faint" aria-hidden="true" />
                <NativeSelect
                  ariaLabel="Filter logs by time range"
                  className="w-[11.5rem] min-w-0 pl-7"
                  name="logs-time-range"
                  onValueChange={(value) => {
                    const preset = RELATIVE_TIME_PRESETS.find((option) => option.value === value)
                    if (preset) onSearchChange(updateLogsTimeRange(search, preset.value))
                  }}
                  options={RELATIVE_TIME_PRESETS}
                  value={search.timeRange ?? ''}
                />
              </div>
              <Button
                className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
                disabled={activeFilterGroups === 0 && !search.cursor && !trimmedQuery}
                onClick={() => {
                  onSearchChange(resetLogsSearch(search))
                  setEventQuery('')
                }}
                size="sm"
                variant="outline"
              >
                <RotateCcw className="size-3.5" aria-hidden="true" />
                Reset view
              </Button>
              <Button
                aria-pressed={showManagementRecords}
                className="ui-control h-8 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
                onClick={() => setShowManagementRecords((current) => !current)}
                size="sm"
                variant="outline"
              >
                {showManagementRecords ? 'Hide management records' : 'Show management records'}
              </Button>
              <FilterPopover
                activeFilterGroups={categoryFilterGroups}
                categories={ledgerFilterCategories}
                contentLabel="Event log filters"
                formatOptionLabel={(value) =>
                  value === 'requests'
                    ? 'Requests'
                    : value === 'quic'
                      ? 'QUIC'
                      : `${value[0]?.toUpperCase()}${value.slice(1)}`
                }
                id="logs-ledger-filters"
                itemLabel="events"
                onClear={() => onSearchChange(updateLogCategories(search, undefined))}
                onSelectAll={() => onSearchChange(updateLogCategories(search, undefined))}
                onSelectNone={() => onSearchChange(updateLogCategories(search, []))}
                onValueChange={(_key, value, checked) => {
                  if (!isLogEventCategory(value)) return
                  const nextCategories = new Set(selectedCategoryValues)
                  if (checked) nextCategories.add(value)
                  else nextCategories.delete(value)
                  onSearchChange(updateLogCategories(search, [...nextCategories]))
                }}
                optionsByCategory={optionsByCategory}
                selectedValuesByCategory={selectedValuesByCategory}
                title="Event categories"
                totalCount={mergedRows.length}
                triggerLabel="Filter event logs"
                visibleCount={visibleRows.length}
              />
              {table ? <DataTableViewOptions columnLabels={LOG_EVENT_LEDGER_COLUMN_LABELS} table={table} /> : null}
              {requestResult?.state === 'supported' ? <LogOperations operation="export" query={requestScope} /> : null}
            </div>
          </section>

          <ScrollArea
            ref={tableRegionRef}
            className="max-h-[71rem]"
            horizontal
            viewportLabel="Scrollable event columns"
          >
            <DataTable
              ariaLabel="MeshLLM event logs"
              columns={columns}
              data={visibleRows}
              defaultPageSize={20}
              emptyMessage={
                activeFilterGroups > 0 || trimmedQuery
                  ? 'No events match this loaded window.'
                  : 'No request or operational events are loaded yet.'
              }
              enablePagination
              footerClassName=""
              getRowAriaLabel={eventRowAriaLabel}
              getRowId={(row) => row.id}
              onRowActivate={handleEventOpen}
              tableClassName="min-w-[780px] text-[length:var(--density-type-caption-lg)] [&_td]:py-3 [&_thead]:bg-transparent"
            >
              {(tableInstance) => <TableCapture onCapture={handleSetTable} table={tableInstance} />}
            </DataTable>
          </ScrollArea>

          {table && visibleRows.length > 0 ? (
            <nav aria-label="Loaded event rows" className="border-t border-border-soft">
              <DataTablePagination table={table} />
            </nav>
          ) : null}
        </Card>
      ) : null}

      <LogEventInspector
        auditEntries={auditEntries}
        inspector={inspector}
        onClose={() => onSearchChange(closeLogInspector(search))}
        onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded}
        onRequestTabChange={(tab) => onSearchChange({ ...search, tab })}
        requestTab={search.tab ?? 'overview'}
      />
    </div>
  )
}

function captureStatusLabel(status: LoggingStatus): string {
  switch (status.capture_mode) {
    case 'metadata_only':
      return 'Payloads · Metadata only'
    case 'redacted_artifacts':
      return status.artifact_capture_ready ? 'Payloads · Redacted · Ready' : 'Payloads · Redacted · Unavailable'
    case 'unavailable':
      return 'Payloads · Unavailable'
  }
}
