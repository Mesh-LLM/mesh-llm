import * as SliderPrimitive from '@radix-ui/react-slider'
import { useMemo, type RefObject } from 'react'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
  LOG_EVENT_CATEGORIES,
  logEventCategoryOptions,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import { rowsInCleanupWindow, type CleanupWindow } from '@/features/logs/lib/log-cleanup-window'

const BUCKET_COUNT = 28

const categoryLabels: Record<LogEventCategory, string> = {
  requests: 'Requests',
  system: 'System',
  quic: 'QUIC',
  gossip: 'Gossip',
  iroh: 'Iroh'
}

const categoryColors: Record<LogEventCategory, string> = {
  requests: 'color-mix(in oklab, var(--color-accent) 58%, var(--color-foreground))',
  system: 'color-mix(in oklab, var(--color-accent-contrast) 58%, var(--color-foreground))',
  quic: 'color-mix(in oklab, var(--color-accent) 34%, var(--color-foreground))',
  gossip: 'color-mix(in oklab, var(--color-accent-contrast) 34%, var(--color-foreground))',
  iroh: 'var(--color-fg-dim)'
}

type LogCleanupWindowProps = {
  readonly bounds: CleanupWindow
  readonly categories: readonly LogEventCategory[]
  readonly onCategoriesChange: (categories: LogEventCategory[]) => void
  readonly onWindowChange: (window: CleanupWindow) => void
  readonly rows: readonly LogEventLedgerRow[]
  readonly startThumbRef: RefObject<HTMLSpanElement | null>
  readonly window: CleanupWindow
}

type CleanupBucket = {
  readonly start: number
  readonly end: number
  readonly counts: Record<LogEventCategory, number>
  readonly total: number
}

function emptyCounts(): Record<LogEventCategory, number> {
  return { requests: 0, system: 0, quic: 0, gossip: 0, iroh: 0 }
}

function cleanupBuckets(rows: readonly LogEventLedgerRow[], bounds: CleanupWindow): CleanupBucket[] {
  const duration = Math.max(1, bounds.end - bounds.start)
  const bucketDuration = duration / BUCKET_COUNT
  const buckets = Array.from({ length: BUCKET_COUNT }, (_, index) => ({
    start: bounds.start + index * bucketDuration,
    end: bounds.start + (index + 1) * bucketDuration,
    counts: emptyCounts(),
    total: 0
  }))

  for (const row of rows) {
    const occurredAt = Date.parse(row.occurredAt)
    if (!Number.isFinite(occurredAt) || occurredAt < bounds.start || occurredAt > bounds.end) continue
    const index = Math.min(BUCKET_COUNT - 1, Math.floor(((occurredAt - bounds.start) / duration) * BUCKET_COUNT))
    buckets[index].counts[row.category] += 1
    buckets[index].total += 1
  }
  return buckets
}

function formatWindowInstant(value: number) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  }).format(value)
}

function categoryDescription(category: LogEventCategory, count: number, selected: boolean) {
  const disposition =
    category === 'requests'
      ? selected
        ? 'selected for cleanup preview'
        : 'not selected for cleanup preview'
      : selected
        ? 'shown in chart and retained during cleanup'
        : 'hidden from chart and retained during cleanup'
  return `${categoryLabels[category]} chart layer, ${count} loaded ${count === 1 ? 'event' : 'events'}, ${disposition}`
}

export function LogCleanupWindow({
  bounds,
  categories,
  onCategoriesChange,
  onWindowChange,
  rows,
  startThumbRef,
  window
}: LogCleanupWindowProps) {
  const options = useMemo(() => logEventCategoryOptions(rows), [rows])
  const optionCategories = LOG_EVENT_CATEGORIES.filter(
    (category) => category !== 'iroh' || options.some((option) => option.value === 'iroh')
  )
  const selectedCategories = useMemo(() => new Set(categories), [categories])
  const windowRows = useMemo(() => rowsInCleanupWindow(rows, window, new Set(LOG_EVENT_CATEGORIES)), [rows, window])
  const selectedRows = useMemo(
    () => rowsInCleanupWindow(rows, window, selectedCategories),
    [rows, window, selectedCategories]
  )
  const requestCount = useMemo(() => selectedRows.filter((row) => row.category === 'requests').length, [selectedRows])
  const buckets = useMemo(() => cleanupBuckets(rows, bounds), [rows, bounds])
  const maxBucketTotal = useMemo(() => Math.max(1, ...buckets.map((bucket) => bucket.total)), [buckets])
  const step = useMemo(() => Math.max(1, Math.round((bounds.end - bounds.start) / 240)), [bounds])
  const selectedSummary = useMemo(
    () =>
      `${selectedRows.length} loaded ${selectedRows.length === 1 ? 'event' : 'events'} shown; ${requestCount} request ${requestCount === 1 ? 'event' : 'events'} in the loaded view. Server preview identifies removable terminal request groups.`,
    [selectedRows, requestCount]
  )

  return (
    <section aria-labelledby="cleanup-window-title" className="space-y-3">
      <div>
        <h3 className="type-panel-title text-foreground" id="cleanup-window-title">
          Select a time window
        </h3>
        <p className="mt-1 type-caption text-fg-dim">
          Drag either edge to narrow the loaded history. The server preview confirms what can be removed.
        </p>
      </div>

      <div className="rounded-[var(--radius)] border border-border-soft bg-panel-strong/55 px-3 pb-3 pt-3.5">
        <div aria-hidden="true" className="flex h-20 items-end gap-px">
          {buckets.map((bucket) => {
            const selected = bucket.end >= window.start && bucket.start <= window.end
            return (
              <div
                className="flex h-full min-w-0 flex-1 flex-col justify-end overflow-hidden rounded-t-[2px]"
                key={bucket.start}
                style={{ opacity: selected ? 1 : 0.2 }}
              >
                {LOG_EVENT_CATEGORIES.map((category) => {
                  const count = selectedCategories.has(category) ? bucket.counts[category] : 0
                  if (count === 0) return null
                  return (
                    <span
                      className="block min-h-px w-full"
                      key={category}
                      style={{
                        background: categoryColors[category],
                        height: `${(count / maxBucketTotal) * 100}%`
                      }}
                    />
                  )
                })}
              </div>
            )
          })}
        </div>

        <SliderPrimitive.Root
          aria-label="Cleanup time window"
          className="relative mt-1 flex h-11 w-full touch-none select-none items-center"
          max={bounds.end}
          min={bounds.start}
          minStepsBetweenThumbs={1}
          onValueChange={([start = bounds.start, end = bounds.end]) => onWindowChange({ start, end })}
          step={step}
          value={[window.start, window.end]}
        >
          <SliderPrimitive.Track className="relative h-1.5 grow overflow-hidden rounded-full bg-border">
            <SliderPrimitive.Range className="absolute h-full bg-accent" />
          </SliderPrimitive.Track>
          <SliderPrimitive.Thumb
            aria-label="Window start"
            aria-valuetext={formatWindowInstant(window.start)}
            className="relative block size-[3.15rem] rounded-full bg-transparent outline-none after:absolute after:left-1/2 after:top-1/2 after:size-4 after:-translate-x-1/2 after:-translate-y-1/2 after:rounded-full after:border-2 after:border-panel after:bg-accent after:shadow-[var(--shadow-slider-thumb)] after:transition-transform hover:after:scale-110 focus-visible:after:ring-2 focus-visible:after:ring-accent focus-visible:after:ring-offset-2 focus-visible:after:ring-offset-panel"
            ref={startThumbRef}
          />
          <SliderPrimitive.Thumb
            aria-label="Window end"
            aria-valuetext={formatWindowInstant(window.end)}
            className="relative block size-[3.15rem] rounded-full bg-transparent outline-none after:absolute after:left-1/2 after:top-1/2 after:size-4 after:-translate-x-1/2 after:-translate-y-1/2 after:rounded-full after:border-2 after:border-panel after:bg-accent after:shadow-[var(--shadow-slider-thumb)] after:transition-transform hover:after:scale-110 focus-visible:after:ring-2 focus-visible:after:ring-accent focus-visible:after:ring-offset-2 focus-visible:after:ring-offset-panel"
          />
        </SliderPrimitive.Root>

        <div className="flex items-start justify-between gap-4 font-mono text-[length:var(--density-type-annotation)] tabular-nums text-fg-dim">
          <span>{formatWindowInstant(window.start)}</span>
          <span className="text-right">{formatWindowInstant(window.end)}</span>
        </div>
      </div>

      <div>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="type-panel-title text-foreground">Chart layers</h3>
          <span className="type-caption text-fg-dim" role="status">
            {windowRows.length} loaded in window
          </span>
        </div>
        <ToggleGroup
          aria-label="Log categories shown in the chart"
          className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4"
          onValueChange={(values) => onCategoriesChange(values as LogEventCategory[])}
          type="multiple"
          value={[...categories]}
        >
          {optionCategories.map((category) => {
            const count = windowRows.filter((row) => row.category === category).length
            return (
              <ToggleGroupItem
                aria-label={categoryDescription(category, count, selectedCategories.has(category))}
                className="ui-control h-auto min-h-11 justify-between gap-2 rounded-[var(--radius)] px-2.5 py-2 text-left data-[state=on]:border-accent data-[state=on]:bg-[color:color-mix(in_oklab,var(--color-accent)_9%,var(--color-panel-strong))] data-[state=on]:text-foreground data-[state=on]:shadow-none"
                key={category}
                value={category}
              >
                <span className="min-w-0">
                  <span className="flex items-center gap-1.5 type-caption">
                    <span
                      aria-hidden="true"
                      className="size-1.5 shrink-0 rounded-full"
                      style={{ background: categoryColors[category] }}
                    />
                    {categoryLabels[category]}
                  </span>
                  <span className="mt-0.5 block type-annotation text-fg-faint">
                    {category === 'requests'
                      ? selectedCategories.has(category)
                        ? 'Selected for removal'
                        : 'Not selected'
                      : 'Chart only · retained'}
                  </span>
                </span>
                <span className="font-mono text-[length:var(--density-type-caption)] tabular-nums">{count}</span>
              </ToggleGroupItem>
            )
          })}
        </ToggleGroup>
        <div className="mt-2 flex items-baseline justify-between gap-3 rounded-[var(--radius)] bg-panel-strong/55 px-3 py-2">
          <p className="type-caption text-fg-dim">
            <span className="font-medium text-foreground">{requestCount}</span> loaded request{' '}
            {requestCount === 1 ? 'event' : 'events'} in this window. Server review identifies removable terminal
            request groups.
          </p>
        </div>
        <p className="mt-2 type-caption text-fg-dim">
          Select Requests to include terminal request history in the cleanup preview. Operational layers only change the
          chart and stay retained.
        </p>
        <span className="sr-only" aria-live="polite">
          {selectedSummary}
        </span>
      </div>
    </section>
  )
}
