import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeftRight, RotateCcw, Search as SearchIcon, X, Logs } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { FilterPopover, type FilterValueOption } from '@/components/ui/FilterPopover'
import { InfoBanner } from '@/components/ui/InfoBanner'
import { Input } from '@/components/ui/input'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { DataTable, TanStackTable as DataTableTanStackTableType } from '@/components/ui/data-table'
import { DataTablePagination } from '@/components/ui/data-table-pagination'
import { DataTableViewOptions } from '@/components/ui/data-table-view-options'
import { useLogsAuditQuery } from '@/features/logs/api/use-logs-audit-query'
import { useLogsLedgerQuery } from '@/features/logs/api/use-logs-ledger-query'
import { useLogsLiveRecovery } from '@/features/logs/api/use-logs-live-recovery'
import { LogAuditCursor } from '@/features/logs/api/ids'
import {
  buildLogEventLedgerColumns,
  LOG_EVENT_LEDGER_COLUMN_LABELS
} from '@/features/logs/components/LogEventLedgerColumns'
import { EventsOverTimeChart } from '@/features/logs/components/EventsOverTimeChart'
import { LogEventInspector } from '@/features/logs/components/LogEventInspector'
import { LogOperations } from '@/features/logs/components/LogOperations'
import { LogsLedgerLoadingGhost } from '@/features/logs/components/LogsLedgerLoadingGhost'
import { LogsSchemaCompatibilityAlert } from '@/features/logs/components/LogsSchemaCompatibilityAlert'
import type { LoggingStatus } from '@/lib/api/types'
import {
  filterLogEventRows,
  logEventCategoryOptions,
  mergeLogEventWindow,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import { compareLogInstants } from '@/features/logs/lib/log-instant'
import { resolveLogsSchemaCompatibility } from '@/features/logs/lib/logs-schema-compatibility'
import { useAdvancingChartClock } from '@/features/logs/lib/use-advancing-chart-clock'
import {
  closeLogInspector,
  logInspectorFromSearch,
  openLogInspector,
  resetLogsSearch,
  toLogsRequestQuery,
  updateLogCategories,
  updateLogsTimeRange,
  type LogsLedgerSearch
} from '@/features/logs/lib/log-search'

import {
  KpiStrip,
  type LedgerFilterKey,
  LogWindowRecoveryAlert,
  TableCapture,
  activeFilterGroupCount,
  eventRowAriaLabel,
  eventRowId,
  eventSearchText,
  isLogEventCategory,
  ledgerFilterCategories,
  liveStateLabel,
  liveStateTone,
  selectedCategories,
  selectedRange
} from '@/features/logs/components/LogsLedgerSections'

type LogsLedgerProps = {
  readonly search: LogsLedgerSearch
  readonly onSearchChange: (search: LogsLedgerSearch) => void
  readonly onMaintenanceMutationSucceeded?: () => void
  readonly loggingStatus?: LoggingStatus
}
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
  const schemaCompatibility = resolveLogsSchemaCompatibility(loggingStatus, requestQuery.error, auditQuery.error)
  const hydrate = useCallback(async () => refetch(), [refetch])
  const hydrateAudit = useCallback(async () => refetchAudit(), [refetchAudit])
  const auditCursor = useMemo(() => {
    if (auditResult?.state !== 'supported' || auditResult.value.items.length === 0) return undefined
    const sequence = auditResult.value.items.reduce((latest, entry) => Math.max(latest, entry.sequence), 0)
    return LogAuditCursor.parse(`a1:${sequence}`)
  }, [auditResult])
  const live = useLogsLiveRecovery({
    enabled: requestResult?.state === 'supported',
    authoritativeSnapshot: requestResult?.state === 'supported' ? requestResult.value : undefined,
    auditEnabled: auditResult?.state === 'supported',
    search,
    hydrate,
    hydrateAudit,
    auditCursor
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
  const allRequestRows = useMemo(() => {
    const rows = requestResult?.state === 'supported' ? requestResult.value.items : []
    const excluded = new Set(live.excludedRequestIds ?? [])
    const merged = new Map(
      rows.filter((row) => !excluded.has(row.requestId.toString())).map((row) => [row.requestId.toString(), row])
    )
    for (const update of live.requestUpdates ?? []) merged.set(update.requestId.toString(), update)
    return [...merged.values()].sort((left, right) => right.createdAt.localeCompare(left.createdAt))
  }, [live.excludedRequestIds, live.requestUpdates, requestResult])
  const requestRows = useMemo(
    () => allRequestRows.filter((row) => row.route !== 'models' && !row.route?.startsWith('management_')),
    [allRequestRows]
  )
  const selectedLedgerRange = useMemo(
    () => selectedRange(requestScope, search.timeRange, ledgerNow),
    [ledgerNow, requestScope, search.timeRange]
  )
  const auditEntries = useMemo(() => {
    const rows = auditResult?.state === 'supported' ? auditResult.value.items : []
    const merged = new Map(rows.map((entry) => [entry.entryId, entry]))
    for (const entry of live.auditEntries ?? []) merged.set(entry.entryId, entry)
    return [...merged.values()].sort((left, right) => right.occurredAt.localeCompare(left.occurredAt))
  }, [auditResult, live.auditEntries])
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
  const selectedCategoryValues = useMemo(
    () => selectedCategories(search.categories, categoryOptions),
    [categoryOptions, search.categories]
  )
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
              rows={mergedRows}
              selectedCategories={selectedCategoryValues}
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

      {hasSupportedWindow ? (
        <EventsOverTimeChart
          now={selectedLedgerRange.endMs}
          onSelectedRangeChange={(range) => {
            if (range === 'selected') return
            onSearchChange(updateLogsTimeRange(search, range === 'all' ? '' : range))
          }}
          rows={categoryRows}
          selectedCategories={selectedCategoryValues}
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

      {requestQuery.isLoading && auditQuery.isLoading && !schemaCompatibility ? <LogsLedgerLoadingGhost /> : null}

      {schemaCompatibility ? <LogsSchemaCompatibilityAlert {...schemaCompatibility} /> : null}

      <LogWindowRecoveryAlert
        sources={[
          {
            id: 'requests',
            label: 'Request history',
            error: requestQuery.isError && !schemaCompatibility,
            fetching: requestQuery.isFetching,
            loading: requestQuery.isLoading,
            hasLoadedData: requestResult?.state === 'supported',
            onRetry: () => void requestQuery.refetch()
          },
          {
            id: 'operations',
            label: 'Operational events',
            error: auditQuery.isError && !schemaCompatibility,
            fetching: auditQuery.isFetching,
            loading: auditQuery.isLoading,
            hasLoadedData: auditResult?.state === 'supported',
            onRetry: () => void auditQuery.refetch()
          }
        ]}
      />

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

          <div
            aria-label="Scrollable event columns"
            className="overflow-x-auto focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent"
            ref={tableRegionRef}
            role="region"
            tabIndex={0}
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
              getRowId={eventRowId}
              onRowActivate={handleEventOpen}
              tableClassName="min-w-[780px] text-[length:var(--density-type-caption-lg)] [&_td]:py-3 [&_thead]:bg-transparent"
              tableWrapperClassName="overflow-visible"
            >
              {(tableInstance) => <TableCapture onCapture={handleSetTable} table={tableInstance} />}
            </DataTable>
          </div>

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
