import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeftRight, RotateCcw, Search as SearchIcon, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { FilterPopover, type FilterValueOption } from '@/components/ui/FilterPopover'
import { Input } from '@/components/ui/input'
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
import { LogsLedgerHeader } from '@/features/logs/components/LogsLedgerHeader'
import { LogsLedgerLoadingGhost } from '@/features/logs/components/LogsLedgerLoadingGhost'
import { LogsSchemaCompatibilityAlert } from '@/features/logs/components/LogsSchemaCompatibilityAlert'
import type { LoggingStatus } from '@/lib/api/types'
import {
  filterLogEventRows,
  logEventCategoryOptions,
  mergeLogEventWindow,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import { logEventSearchText } from '@/features/logs/lib/log-event-search'
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
  updateLogsTimeWindow,
  clearLogsTimeWindow,
  type LogsLedgerSearch
} from '@/features/logs/lib/log-search'

import {
  KpiStrip,
  type LedgerFilterKey,
  TableCapture,
  activeFilterGroupCount,
  eventRowAriaLabel,
  eventRowId,
  isLogEventCategory,
  ledgerFilterCategories,
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
  const tableRegionRef = useRef<HTMLElement>(null)
  const restoredFocusIdRef = useRef<string | undefined>(undefined)
  const trimmedQuery = useMemo(() => eventQuery.trim().toLowerCase(), [eventQuery])
  const visibleRows = useMemo(
    () => (trimmedQuery ? categoryRows.filter((row) => logEventSearchText(row).includes(trimmedQuery)) : categoryRows),
    [categoryRows, trimmedQuery]
  )
  const currentPageTimeWindow = useMemo(() => {
    const currentRows = table?.getRowModel().rows ?? []
    if (currentRows.length === 0) return undefined
    const timestamps = currentRows.map((row) => Date.parse(row.original.occurredAt))
    return { from: Math.min(...timestamps), to: Math.max(...timestamps) }
  }, [table])
  const hasSupportedWindow = requestResult?.state === 'supported' || auditResult?.state === 'supported'
  const windowLoading =
    requestQuery.isLoading || auditQuery.isLoading || requestQuery.isPlaceholderData || auditQuery.isPlaceholderData
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
      <LogsLedgerHeader
        cleanup={
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
        hasSupportedWindow={hasSupportedWindow}
        live={{
          state: live.state,
          fallbackPollingActive: live.fallbackPollingActive,
          pollingEnabled: live.pollingEnabled,
          togglePolling: live.togglePolling
        }}
        loggingStatus={loggingStatus}
        operationsBounded={auditResult?.state === 'supported' && auditResult.value.incomplete === true}
        requestBounded={requestResult?.state === 'supported' && requestResult.value.incomplete === true}
        sources={[
          {
            id: 'requests',
            label: 'Request history',
            error: requestQuery.isError && !schemaCompatibility,
            fetching: requestQuery.isFetching,
            hasLoadedData: requestResult?.state === 'supported',
            refetch: requestQuery.refetch
          },
          {
            id: 'operations',
            label: 'Operational events',
            error: auditQuery.isError && !schemaCompatibility,
            fetching: auditQuery.isFetching,
            hasLoadedData: auditResult?.state === 'supported',
            refetch: auditQuery.refetch
          }
        ]}
      />

      {hasSupportedWindow ? (
        <EventsOverTimeChart
          currentPageTimeWindow={currentPageTimeWindow}
          loading={windowLoading}
          now={selectedLedgerRange.endMs}
          onBucketSelect={({ from, to }) => onSearchChange(updateLogsTimeWindow(search, from, to))}
          onClearBucketSelection={() => onSearchChange(clearLogsTimeWindow(search))}
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

      {requestQuery.isLoading && auditQuery.isLoading && !schemaCompatibility ? <LogsLedgerLoadingGhost /> : null}

      {schemaCompatibility ? <LogsSchemaCompatibilityAlert {...schemaCompatibility} /> : null}

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

          <section
            aria-label="Scrollable event columns"
            className="overflow-x-auto focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent"
            ref={tableRegionRef}
            // biome-ignore lint/a11y/noNoninteractiveTabindex: overflow regions need a tab stop for keyboard scrolling.
            tabIndex={0}
          >
            <DataTable
              ariaLabel="MeshLLM event logs"
              columns={columns}
              data={visibleRows}
              defaultPageSize={10}
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
          </section>

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
        requestRows={requestRows}
        requestTab={search.tab ?? 'overview'}
      />
    </div>
  )
}
