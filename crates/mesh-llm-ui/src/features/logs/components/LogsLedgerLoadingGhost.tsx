import { LiveLoadingGhostRoot } from '@/components/ui/LiveLoadingGhostRoot'
import { LoadingGhostBlock } from '@/components/ui/LoadingGhostBlock'

const CHART_BAR_STACKS = [
  ['h-7 bg-accent/35', 'h-4 bg-accent-contrast/35', 'h-2 bg-border-soft'],
  ['h-11 bg-accent/35', 'h-6 bg-accent-contrast/35', 'h-3 bg-border-soft'],
  ['h-5 bg-accent/35', 'h-9 bg-accent-contrast/35', 'h-2 bg-border-soft'],
  ['h-14 bg-accent/35', 'h-5 bg-accent-contrast/35', 'h-3 bg-border-soft'],
  ['h-9 bg-accent/35', 'h-12 bg-accent-contrast/35', 'h-2 bg-border-soft'],
  ['h-16 bg-accent/35', 'h-7 bg-accent-contrast/35', 'h-4 bg-border-soft'],
  ['h-6 bg-accent/35', 'h-4 bg-accent-contrast/35', 'h-2 bg-border-soft'],
  ['h-12 bg-accent/35', 'h-8 bg-accent-contrast/35', 'h-3 bg-border-soft'],
  ['h-8 bg-accent/35', 'h-13 bg-accent-contrast/35', 'h-2 bg-border-soft']
] as const

const KPI_GHOSTS = [
  { label: 'w-24', value: 'w-12', meta: 'w-20', line: 'w-[78%]' },
  { label: 'w-20', value: 'w-10', meta: 'w-14', line: 'w-[62%]' },
  { label: 'w-14', value: 'w-8', meta: 'w-12', line: 'w-[45%]' },
  { label: 'w-16', value: 'w-9', meta: 'w-16', line: 'w-[58%]' }
] as const

const LEDGER_ROWS = [
  { time: 'w-14', title: 'w-40', detail: 'w-24', model: 'w-24', source: 'w-20', outcome: 'w-14', age: 'w-10' },
  { time: 'w-11', title: 'w-28', detail: 'w-36', model: 'w-16', source: 'w-28', outcome: 'w-12', age: 'w-8' },
  { time: 'w-16', title: 'w-48', detail: 'w-20', model: 'w-28', source: 'w-16', outcome: 'w-16', age: 'w-12' },
  { time: 'w-12', title: 'w-32', detail: 'w-28', model: 'w-20', source: 'w-24', outcome: 'w-12', age: 'w-9' },
  { time: 'w-14', title: 'w-36', detail: 'w-16', model: 'w-32', source: 'w-18', outcome: 'w-14', age: 'w-11' }
] as const

function LogsChartLoadingGhost() {
  return (
    <section
      className="w-full rounded-[var(--radius)] border border-border-soft bg-panel px-[var(--panel-x)] py-[var(--panel-y)]"
      data-loading-region="logs-chart"
    >
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <LoadingGhostBlock className="h-4 w-40" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-[min(20rem,82vw)] max-w-full" shimmer />
        </div>
        <div className="grid w-full grid-cols-2 gap-2 sm:flex sm:w-auto">
          <LoadingGhostBlock className="h-8 w-full sm:w-[6.5rem]" shimmer />
          <LoadingGhostBlock className="h-8 w-full sm:w-40" shimmer />
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
        {['w-16', 'w-20', 'w-14', 'w-18'].map((width) => (
          <div className="flex items-center gap-1.5" key={width}>
            <LoadingGhostBlock className="size-2 rounded-[2px]" />
            <LoadingGhostBlock className={`h-3 ${width}`} shimmer />
          </div>
        ))}
      </div>

      <div className="relative mt-4 h-[170px] overflow-hidden rounded-[var(--radius-sm)] border border-border-soft bg-background px-3 pb-7 pt-4">
        <div className="pointer-events-none absolute inset-x-3 top-8 border-t border-border-soft" />
        <div className="pointer-events-none absolute inset-x-3 top-1/2 border-t border-border-soft" />
        <div className="pointer-events-none absolute inset-x-3 bottom-7 border-t border-border" />
        <div className="absolute bottom-7 left-3 top-4 flex flex-col justify-between">
          <LoadingGhostBlock className="h-2.5 w-6" />
          <LoadingGhostBlock className="h-2.5 w-5" />
          <LoadingGhostBlock className="h-2.5 w-4" />
        </div>
        <div className="absolute inset-x-8 bottom-7 top-4 flex items-end justify-between gap-1">
          {CHART_BAR_STACKS.map((stack, index) => (
            <div className="flex h-full w-[clamp(8px,2.5%,14px)] flex-col justify-end gap-px" key={index}>
              {stack.map((segment) => (
                <span className={`block w-full rounded-[2px] border border-panel ${segment}`} key={segment} />
              ))}
            </div>
          ))}
        </div>
        <div className="absolute inset-x-8 bottom-2 flex justify-between gap-2">
          <LoadingGhostBlock className="h-2.5 w-9" />
          <LoadingGhostBlock className="h-2.5 w-10" />
          <LoadingGhostBlock className="h-2.5 w-8" />
          <LoadingGhostBlock className="h-2.5 w-10" />
        </div>
      </div>
    </section>
  )
}

function LogsKpiLoadingGhost() {
  return (
    <section aria-label="Request summary" data-loading-region="logs-kpis">
      <div className="mb-3 flex items-center justify-between gap-4">
        <LoadingGhostBlock className="h-4 w-28" shimmer />
        <LoadingGhostBlock className="hidden h-3 w-56 sm:block" />
      </div>
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4 xl:gap-[calc(var(--shell-normal)*1.25)]">
        {KPI_GHOSTS.map((ghost, index) => (
          <div
            className="panel-shell min-w-0 rounded-[var(--radius-lg)] border border-border bg-panel px-[var(--panel-x)] py-[var(--panel-y)]"
            key={index}
          >
            <div className="flex items-center gap-1.5">
              <LoadingGhostBlock className="size-3.5 shrink-0 rounded-[2px]" />
              <LoadingGhostBlock className={`h-3 ${ghost.label} max-w-[calc(100%-1.25rem)]`} shimmer />
            </div>
            <LoadingGhostBlock className={`mt-[var(--panel-y,12px)] h-5 ${ghost.value}`} shimmer />
            <LoadingGhostBlock className={`mt-3 h-2.5 ${ghost.line} max-w-full`} />
            <LoadingGhostBlock className={`mt-2 h-2.5 ${ghost.meta}`} shimmer />
          </div>
        ))}
      </div>
    </section>
  )
}

function LogsLedgerTableRowsLoadingGhost() {
  return (
    <div className="divide-y divide-border-soft">
      {LEDGER_ROWS.map((row, index) => (
        <div
          className="grid min-w-[780px] grid-cols-[5rem_minmax(0,1.5fr)_minmax(0,.85fr)_minmax(0,.8fr)_minmax(0,.65fr)_minmax(0,.6fr)] items-center gap-3 px-4 py-3"
          key={index}
        >
          <LoadingGhostBlock className={`h-3 ${row.time} max-w-full`} />
          <div className="min-w-0 space-y-2">
            <LoadingGhostBlock className={`h-3 ${row.title} max-w-full`} shimmer />
            <LoadingGhostBlock className={`h-2.5 ${row.detail} max-w-[70%]`} />
          </div>
          <LoadingGhostBlock className={`h-3 ${row.model} max-w-full`} />
          <LoadingGhostBlock className={`h-3 ${row.source} max-w-full`} />
          <LoadingGhostBlock className={`h-5 ${row.outcome} max-w-full rounded-full`} shimmer />
          <LoadingGhostBlock className={`h-3 ${row.age} max-w-full`} />
        </div>
      ))}
    </div>
  )
}

function LogsLedgerTableLoadingGhost() {
  return (
    <section
      className="overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel shadow-none"
      data-loading-region="logs-ledger"
    >
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border-soft px-4 py-3">
        <div className="min-w-0 flex-1">
          <LoadingGhostBlock className="h-3 w-[min(17rem,74vw)] max-w-full" shimmer />
          <LoadingGhostBlock className="mt-2 h-3 w-[min(24rem,86vw)] max-w-full" />
          <LoadingGhostBlock className="mt-2 h-3 w-36 max-w-full lg:hidden" />
        </div>
        <div className="grid w-full grid-cols-2 gap-2 sm:flex sm:w-auto sm:flex-wrap sm:items-center">
          <LoadingGhostBlock className="h-8 w-full sm:w-64" shimmer />
          <LoadingGhostBlock className="h-8 w-full sm:w-24" shimmer />
          <LoadingGhostBlock className="h-8 w-full sm:w-28" shimmer />
          <LoadingGhostBlock className="h-8 w-full sm:w-28" />
          <LoadingGhostBlock className="h-8 w-full sm:w-20" />
        </div>
      </div>

      <div data-loading-region="logs-ledger-table">
        <div className="overflow-x-auto">
          <div className="grid min-w-[780px] grid-cols-[5rem_minmax(0,1.5fr)_minmax(0,.85fr)_minmax(0,.8fr)_minmax(0,.65fr)_minmax(0,.6fr)] gap-3 border-b border-border-soft px-4 py-2.5">
            <LoadingGhostBlock className="h-3 w-12 max-w-full" shimmer />
            <LoadingGhostBlock className="h-3 w-16 max-w-full" shimmer />
            <LoadingGhostBlock className="h-3 w-14 max-w-full" />
            <LoadingGhostBlock className="h-3 w-14 max-w-full" />
            <LoadingGhostBlock className="h-3 w-16 max-w-full" />
            <LoadingGhostBlock className="h-3 w-10 max-w-full" />
          </div>
          <LogsLedgerTableRowsLoadingGhost />
        </div>
      </div>
      <div
        className="flex flex-wrap items-center justify-end gap-3 border-t border-border-soft px-[var(--panel-x)] py-[var(--panel-y)]"
        data-loading-region="logs-ledger-pagination"
      >
        <LoadingGhostBlock className="h-3 w-20" shimmer />
        <LoadingGhostBlock className="h-8 w-[4.5rem]" shimmer />
        <LoadingGhostBlock className="h-3 w-16" />
        <div className="flex items-center gap-1">
          {Array.from({ length: 4 }, (_, index) => (
            <LoadingGhostBlock className="size-8" key={index} />
          ))}
        </div>
      </div>
    </section>
  )
}

export function LogsLedgerLoadingGhost() {
  return (
    <LiveLoadingGhostRoot>
      <div aria-label="Loading system logs" data-loading-region="logs-loading" role="status">
        <span className="sr-only">Loading system logs</span>
        <div aria-hidden="true" className="flex min-w-0 flex-col gap-[calc(var(--shell-normal)*2)]">
          <LogsChartLoadingGhost />
          <LogsKpiLoadingGhost />
          <LogsLedgerTableLoadingGhost />
        </div>
      </div>
    </LiveLoadingGhostRoot>
  )
}
