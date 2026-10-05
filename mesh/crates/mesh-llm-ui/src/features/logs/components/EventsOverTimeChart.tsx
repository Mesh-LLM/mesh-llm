import { useCallback, useId, useMemo, useState, type KeyboardEvent } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceArea,
  XAxis,
  YAxis,
  type MouseHandlerDataParam,
  type TooltipContentProps
} from 'recharts'
import { Loader2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
  type ChartTooltipPayloadItem
} from '@/components/ui/chart'
import { NativeSelect } from '@/components/ui/NativeSelect'
import {
  LOG_EVENT_CATEGORY_COLORS,
  LOG_EVENT_CATEGORY_LABELS,
  LOG_EVENT_CATEGORY_MARKER_CLASS
} from '@/features/logs/lib/log-event-category-style'
import {
  LOG_EVENT_CATEGORIES,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import {
  BUCKET_INTERVALS,
  VOLUME_TIME_RANGES,
  buildEventVolumeBuckets,
  defaultBucketIntervalKey,
  effectiveEventVolumeIntervalMs,
  formatBucketInterval,
  formatBucketRange,
  formatBucketTick,
  type BucketIntervalKey,
  type VolumeTimeRangeKey
} from '@/features/logs/lib/log-volume'
import { hasVisibleEventVolumeTooltip } from '@/features/logs/components/events-over-time-chart-tooltip'
import { useAdvancingChartClock } from '@/features/logs/lib/use-advancing-chart-clock'

type EventsOverTimeChartProps = {
  readonly rows: readonly LogEventLedgerRow[]
  readonly selectedCategories: ReadonlySet<LogEventCategory>
  readonly currentPageTimeWindow?: { readonly from: number; readonly to: number }
  /** The ledger filter window; selecting a new ledger range resets the chart to the same window. */
  readonly selectedRange?: VolumeTimeRangeKey
  /** Exact duration for a custom ledger window represented by `selected`. */
  readonly selectedRangeMs?: number
  /** Promotes the chart selector to the owning page's time-range control. */
  readonly onSelectedRangeChange?: (range: VolumeTimeRangeKey) => void
  /** Narrows the owning ledger to a bucket the operator clicked. */
  readonly onBucketSelect?: (window: { readonly from: string; readonly to: string }) => void
  /** Restores the ledger when a clicked bucket window is the active filter. */
  readonly onClearBucketSelection?: () => void
  /** Marks the window as still loading; shown in a reserved slot so the controls never move. */
  readonly loading?: boolean
  /** Test seam: overrides the wall clock used to anchor the time window. */
  readonly now?: number
}

const chartConfig = {
  requests: { label: LOG_EVENT_CATEGORY_LABELS.requests, color: LOG_EVENT_CATEGORY_COLORS.requests },
  system: { label: LOG_EVENT_CATEGORY_LABELS.system, color: LOG_EVENT_CATEGORY_COLORS.system },
  quic: { label: LOG_EVENT_CATEGORY_LABELS.quic, color: LOG_EVENT_CATEGORY_COLORS.quic },
  gossip: { label: LOG_EVENT_CATEGORY_LABELS.gossip, color: LOG_EVENT_CATEGORY_COLORS.gossip },
  iroh: { label: LOG_EVENT_CATEGORY_LABELS.iroh, color: LOG_EVENT_CATEGORY_COLORS.iroh }
} satisfies ChartConfig

export function EventsOverTimeChart({
  rows,
  selectedCategories,
  currentPageTimeWindow,
  now,
  selectedRange,
  selectedRangeMs,
  onSelectedRangeChange,
  onBucketSelect,
  onClearBucketSelection,
  loading = false
}: EventsOverTimeChartProps) {
  const [rangeSelection, setRangeSelection] = useState<{
    readonly filter: VolumeTimeRangeKey | undefined
    readonly value: VolumeTimeRangeKey
  }>({ filter: selectedRange, value: selectedRange ?? '12h' })
  const rangeKey = rangeSelection.filter === selectedRange ? rangeSelection.value : (selectedRange ?? '12h')
  const rangeMs =
    rangeKey === 'selected'
      ? (selectedRangeMs ?? Number.POSITIVE_INFINITY)
      : (VOLUME_TIME_RANGES.find((option) => option.value === rangeKey)?.ms ?? 43_200_000)

  const [intervalSelection, setIntervalSelection] = useState<{
    readonly range: VolumeTimeRangeKey
    readonly value: BucketIntervalKey
  }>()
  const intervalKey =
    intervalSelection?.range === rangeKey ? intervalSelection.value : defaultBucketIntervalKey(rangeMs)
  const intervalMs = BUCKET_INTERVALS.find((option) => option.value === intervalKey)?.ms ?? 300_000
  const liveCurrent = useAdvancingChartClock(now === undefined)
  const current = now ?? liveCurrent
  const activeCategories = useMemo(
    () => LOG_EVENT_CATEGORIES.filter((category) => selectedCategories.has(category)),
    [selectedCategories]
  )

  const data = useMemo(
    () => buildEventVolumeBuckets(rows, selectedCategories, { intervalMs, rangeMs, now: current }),
    [rows, selectedCategories, intervalMs, rangeMs, current]
  )
  const populatedBuckets = useMemo(
    () => data.flatMap((bucket, index) => (bucket.total > 0 ? [{ bucket, index }] : [])),
    [data]
  )
  const totalEvents = useMemo(() => data.reduce((sum, bucket) => sum + bucket.total, 0), [data])
  const totalsByCategory = useMemo(
    () =>
      Object.fromEntries(
        LOG_EVENT_CATEGORIES.map((category) => [category, data.reduce((sum, bucket) => sum + bucket[category], 0)])
      ) as Record<LogEventCategory, number>,
    [data]
  )
  const effectiveIntervalMs = effectiveEventVolumeIntervalMs(data, intervalMs)
  const wasAutoBucketed = effectiveIntervalMs > intervalMs
  const currentPageBucketWindow = useMemo(
    () => overlappingBucketWindow(data, currentPageTimeWindow),
    [currentPageTimeWindow, data]
  )

  const chartListboxId = useId().replace(/:/g, '')
  const [activeBucketStart, setActiveBucketStart] = useState<number | undefined>(undefined)
  const [previousPopulatedBuckets, setPreviousPopulatedBuckets] = useState(populatedBuckets)
  const [hoveredIndex, setHoveredIndex] = useState<number | undefined>(undefined)
  const activeOption =
    populatedBuckets.find(({ bucket }) => bucket.bucketStart === activeBucketStart) ?? populatedBuckets[0]
  const activeOptionIndex = activeBucketStart === undefined ? undefined : activeOption?.index
  const activeOptionId = activeOption ? `${chartListboxId}-bucket-${activeOption.bucket.bucketStart}` : undefined
  const highlightedIndex = hoveredIndex ?? activeOptionIndex
  const highlightedBucket = highlightedIndex === undefined ? undefined : data[highlightedIndex]
  const chartLabel = `Events over time stacked bar chart. Showing ${activeCategories
    .map((category) => chartConfig[category].label)
    .join(', ')}.`
  const bucketIndexOf = useCallback(
    (nextState: MouseHandlerDataParam) => {
      const rawIndex = nextState.activeTooltipIndex
      const index =
        typeof rawIndex === 'number' ? rawIndex : rawIndex === null || rawIndex === '' ? undefined : Number(rawIndex)
      return index !== undefined && Number.isInteger(index) && data[index]?.total ? index : undefined
    },
    [data]
  )
  const handleChartMouseMove = useCallback(
    (nextState: MouseHandlerDataParam) => setHoveredIndex(bucketIndexOf(nextState)),
    [bucketIndexOf]
  )
  const handleChartMouseLeave = useCallback(() => setHoveredIndex(undefined), [])
  const selectBucket = useCallback(
    (index: number | undefined) => {
      const bucket = index === undefined ? undefined : data[index]
      if (!onBucketSelect || bucket === undefined) return
      onBucketSelect({
        from: new Date(bucket.bucketStart).toISOString(),
        to: new Date(bucket.bucketEnd - 1).toISOString()
      })
    },
    [data, onBucketSelect]
  )
  const handleChartClick = useCallback(
    (nextState: MouseHandlerDataParam) => selectBucket(bucketIndexOf(nextState)),
    [bucketIndexOf, selectBucket]
  )
  const handleChartFocus = useCallback(() => {
    setActiveBucketStart((currentBucketStart) =>
      populatedBuckets.some(({ bucket }) => bucket.bucketStart === currentBucketStart)
        ? currentBucketStart
        : populatedBuckets[0]?.bucket.bucketStart
    )
  }, [populatedBuckets])
  const handleChartKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        const activePosition = populatedBuckets.findIndex(({ index }) => index === activeOption?.index)
        const direction = event.key === 'ArrowLeft' ? -1 : 1
        const nextPosition = Math.min(Math.max(activePosition + direction, 0), populatedBuckets.length - 1)
        const nextOption = populatedBuckets[nextPosition]
        if (!nextOption) return
        event.preventDefault()
        setActiveBucketStart(nextOption.bucket.bucketStart)
        return
      }
      if (event.key !== 'Enter' && event.key !== ' ') return
      if (!activeOption) return
      event.preventDefault()
      selectBucket(activeOption.index)
    },
    [activeOption, populatedBuckets, selectBucket]
  )

  const renderTooltip = useCallback((tooltipProps: TooltipContentProps) => {
    if (!hasVisibleEventVolumeTooltip(tooltipProps.payload as readonly ChartTooltipPayloadItem[] | undefined))
      return null
    return (
      <ChartTooltipContent
        active={tooltipProps.active}
        label={tooltipProps.label}
        payload={tooltipProps.payload as unknown as readonly ChartTooltipPayloadItem[]}
        formatter={(value) => `${String(value)} ${Number(value) === 1 ? 'event' : 'events'}`}
        labelFormatter={(_label, payload) => {
          const first = payload[0]?.payload
          return formatBucketRange(Number(first?.bucketStart), Number(first?.bucketEnd))
        }}
        labelKey="label"
      />
    )
  }, [])

  if (previousPopulatedBuckets !== populatedBuckets) {
    setPreviousPopulatedBuckets(populatedBuckets)
    if (
      activeBucketStart !== undefined &&
      !populatedBuckets.some(({ bucket }) => bucket.bucketStart === activeBucketStart)
    ) {
      setActiveBucketStart(populatedBuckets[0]?.bucket.bucketStart)
    }
  }

  return (
    <Card
      aria-describedby="events-over-time-description"
      aria-labelledby="events-over-time-title"
      className="w-full rounded-[var(--radius)] border-border-soft bg-panel px-[var(--panel-x)] py-[var(--panel-y)] shadow-none"
      role="region"
    >
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <h2 className="type-panel-title text-foreground" id="events-over-time-title">
            Events Over Time
          </h2>
          <p className="type-caption mt-1 text-fg-dim" id="events-over-time-description">
            Loaded event volume by category and time bucket
            {wasAutoBucketed ? ` · Auto-bucketed to ${formatBucketInterval(effectiveIntervalMs)}` : ''}
            {currentPageBucketWindow ? (
              <span className="text-fg-faint">
                {' · '}Accent band marks current table page:{' '}
                {formatBucketRange(currentPageBucketWindow.from, currentPageBucketWindow.to)}.
              </span>
            ) : null}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span aria-live="polite" className="flex size-4 shrink-0 items-center justify-center">
            {loading ? (
              <>
                <Loader2 aria-hidden="true" className="size-3.5 animate-spin text-fg-faint" />
                <span className="sr-only">Loading system logs</span>
              </>
            ) : null}
          </span>
          {onClearBucketSelection && selectedRange === 'selected' ? (
            <Button onClick={onClearBucketSelection} size="sm" variant="outline">
              <X aria-hidden="true" className="size-3.5" />
              Clear window
            </Button>
          ) : null}
          <NativeSelect
            ariaLabel="Bucket interval"
            className="w-[6.5rem] min-w-0"
            name="volume-bucket-interval"
            onValueChange={(value) => setIntervalSelection({ range: rangeKey, value: value as BucketIntervalKey })}
            options={BUCKET_INTERVALS.map(({ value, label }) => ({ value, label }))}
            value={intervalKey}
          />
          <NativeSelect
            ariaLabel="Chart time range"
            className="w-[11.5rem] min-w-0"
            name="volume-time-range"
            onValueChange={(value) => {
              const range = value as VolumeTimeRangeKey
              setRangeSelection({ filter: selectedRange, value: range })
              onSelectedRangeChange?.(range)
            }}
            options={[
              ...VOLUME_TIME_RANGES.map(({ value, label }) => ({ value, label })),
              ...(selectedRange === 'selected' ? [{ value: 'selected', label: 'Selected range' }] : [])
            ]}
            value={rangeKey}
          />
        </div>
      </div>

      {activeCategories.length > 0 ? (
        <ul aria-label="Visible event categories" className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
          {activeCategories.map((category) => (
            <li className="inline-flex items-center gap-1.5 type-caption text-fg-dim" key={category}>
              <span
                aria-hidden="true"
                className={`size-2 ${LOG_EVENT_CATEGORY_MARKER_CLASS[category]}`}
                style={{ backgroundColor: chartConfig[category].color }}
              />
              <span>{chartConfig[category].label}</span>
              <span className="font-mono tabular-nums text-fg">{totalsByCategory[category]}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {activeCategories.length === 0 ? (
        <div className="flex h-[170px] items-center justify-center">
          <p className="type-caption text-fg-dim">Select an event category to display the chart.</p>
        </div>
      ) : totalEvents === 0 ? (
        <div className="flex h-[170px] items-center justify-center">
          <p className="type-caption text-fg-dim">No selected events during the chart time range.</p>
        </div>
      ) : (
        <div className="mt-4 h-[170px] w-full">
          <div
            aria-activedescendant={onBucketSelect ? activeOptionId : undefined}
            aria-label={onBucketSelect ? chartLabel : undefined}
            aria-orientation={onBucketSelect ? 'horizontal' : undefined}
            className={
              onBucketSelect
                ? 'h-full w-full rounded-[var(--radius)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'
                : 'h-full w-full'
            }
            onFocus={onBucketSelect ? handleChartFocus : undefined}
            onKeyDown={onBucketSelect ? handleChartKeyDown : undefined}
            role={onBucketSelect ? 'listbox' : undefined}
            tabIndex={onBucketSelect ? 0 : undefined}
          >
            <ChartContainer
              aria-hidden={onBucketSelect ? true : undefined}
              aria-label={onBucketSelect ? undefined : chartLabel}
              className="h-full w-full"
              config={chartConfig}
              role={onBucketSelect ? undefined : 'img'}
            >
              <BarChart
                accessibilityLayer={false}
                data={data}
                margin={{ top: 8, right: 4, left: 0, bottom: 0 }}
                barCategoryGap={1.5}
                className={onBucketSelect ? 'cursor-pointer' : undefined}
                onClick={handleChartClick}
                onMouseLeave={handleChartMouseLeave}
                onMouseMove={handleChartMouseMove}
              >
                <CartesianGrid vertical={false} stroke="var(--color-border-soft)" />
                <XAxis
                  axisLine={false}
                  dataKey="bucketStart"
                  minTickGap={48}
                  tick={{ fill: 'var(--color-fg-faint)', fontSize: 11 }}
                  tickFormatter={(value: number) => formatBucketTick(value, effectiveIntervalMs)}
                  tickLine={false}
                  tickMargin={8}
                />
                <YAxis
                  allowDecimals={false}
                  axisLine={false}
                  tick={{ fill: 'var(--color-fg-faint)', fontSize: 11 }}
                  tickLine={false}
                  tickFormatter={(value: number) => (value >= 1000 ? `${Math.round(value / 1000)}k` : String(value))}
                  width={36}
                />
                {currentPageBucketWindow ? (
                  <ReferenceArea
                    className="pointer-events-none"
                    fill="var(--color-accent)"
                    fillOpacity={0.08}
                    ifOverflow="hidden"
                    x1={currentPageBucketWindow.from}
                    x2={currentPageBucketWindow.lastBucketStart}
                  />
                ) : null}
                <ChartTooltip
                  content={renderTooltip}
                  cursor={highlightedBucket?.total ? { fill: 'var(--color-fg-faint)', fillOpacity: 0.08 } : false}
                  isAnimationActive={false}
                />
                {activeCategories.map((category, categoryIndex) => (
                  <Bar
                    dataKey={category}
                    fill={`var(--color-${category})`}
                    isAnimationActive={false}
                    key={category}
                    maxBarSize={8}
                    radius={categoryIndex === activeCategories.length - 1 ? [2, 2, 0, 0] : 0}
                    stackId="events"
                  >
                    {data.map((bucket, index) => (
                      <Cell
                        key={`${category}-${bucket.bucketStart}`}
                        fillOpacity={highlightedIndex === undefined || highlightedIndex === index ? 1 : 0.7}
                      />
                    ))}
                  </Bar>
                ))}
              </BarChart>
            </ChartContainer>
            {onBucketSelect
              ? populatedBuckets.map(({ bucket }) => (
                  <span
                    aria-selected={activeOption?.bucket.bucketStart === bucket.bucketStart}
                    className="sr-only"
                    id={`${chartListboxId}-bucket-${bucket.bucketStart}`}
                    key={bucket.bucketStart}
                    role="option"
                  >
                    {formatBucketRange(bucket.bucketStart, bucket.bucketEnd)}: {bucket.total}{' '}
                    {bucket.total === 1 ? 'event' : 'events'}
                  </span>
                ))
              : null}
          </div>
        </div>
      )}
    </Card>
  )
}

function overlappingBucketWindow(
  buckets: readonly { readonly bucketStart: number; readonly bucketEnd: number }[],
  window: { readonly from: number; readonly to: number } | undefined
): { readonly from: number; readonly to: number; readonly lastBucketStart: number } | undefined {
  if (!window) return undefined

  const first = buckets.find((bucket) => bucket.bucketEnd > window.from && bucket.bucketStart <= window.to)
  const last = buckets.findLast((bucket) => bucket.bucketEnd > window.from && bucket.bucketStart <= window.to)
  return first && last ? { from: first.bucketStart, to: last.bucketEnd, lastBucketStart: last.bucketStart } : undefined
}
