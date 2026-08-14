import { useCallback, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, XAxis, YAxis } from 'recharts'
import { Card } from '@/components/ui/card'
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import { NativeSelect } from '@/components/ui/NativeSelect'
import {
  LOG_EVENT_CATEGORIES,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import {
  BUCKET_INTERVALS,
  VOLUME_TIME_RANGES,
  buildEventVolumeBuckets,
  effectiveEventVolumeIntervalMs,
  formatBucketInterval,
  formatBucketRange,
  type BucketIntervalKey,
  type VolumeTimeRangeKey
} from '@/features/logs/lib/log-volume'
import { useAdvancingChartClock } from '@/features/logs/lib/use-advancing-chart-clock'

type EventsOverTimeChartProps = {
  readonly rows: readonly LogEventLedgerRow[]
  readonly selectedCategories: ReadonlySet<LogEventCategory>
  /** The ledger filter window; selecting a new ledger range resets the chart to the same window. */
  readonly selectedRange?: VolumeTimeRangeKey
  /** Exact duration for a custom ledger window represented by `selected`. */
  readonly selectedRangeMs?: number
  /** Test seam: overrides the wall clock used to anchor the time window. */
  readonly now?: number
}

const chartConfig = {
  requests: {
    label: 'Requests',
    color: 'color-mix(in oklab, var(--color-accent) 58%, var(--color-foreground))'
  },
  system: {
    label: 'System',
    color: 'color-mix(in oklab, var(--color-accent-contrast) 58%, var(--color-foreground))'
  },
  quic: {
    label: 'QUIC',
    color: 'color-mix(in oklab, var(--color-accent) 34%, var(--color-foreground))'
  },
  gossip: {
    label: 'Gossip',
    color: 'color-mix(in oklab, var(--color-accent-contrast) 34%, var(--color-foreground))'
  },
  iroh: { label: 'Iroh', color: 'var(--color-fg-dim)' }
} satisfies ChartConfig

export function EventsOverTimeChart({
  rows,
  selectedCategories,
  now,
  selectedRange,
  selectedRangeMs
}: EventsOverTimeChartProps) {
  const [intervalKey, setIntervalKey] = useState<BucketIntervalKey>('5m')
  const [rangeSelection, setRangeSelection] = useState<{
    readonly filter: VolumeTimeRangeKey | undefined
    readonly value: VolumeTimeRangeKey
  }>({ filter: selectedRange, value: selectedRange ?? '12h' })
  const rangeKey = rangeSelection.filter === selectedRange ? rangeSelection.value : (selectedRange ?? '12h')

  const intervalMs = BUCKET_INTERVALS.find((option) => option.value === intervalKey)?.ms ?? 300_000
  const rangeMs =
    rangeKey === 'selected'
      ? (selectedRangeMs ?? Number.POSITIVE_INFINITY)
      : (VOLUME_TIME_RANGES.find((option) => option.value === rangeKey)?.ms ?? 43_200_000)
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

  const [activeIndex, setActiveIndex] = useState<number | undefined>(undefined)
  const handleBarMouseEnter = useCallback((_entry: unknown, index: number) => setActiveIndex(index), [])
  const handleBarMouseLeave = useCallback(() => setActiveIndex(undefined), [])

  return (
    <Card
      aria-labelledby="events-over-time-title"
      className="w-full rounded-[var(--radius)] border-border-soft bg-panel px-[var(--panel-x)] py-[var(--panel-y)] shadow-none"
      role="region"
    >
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="min-w-0">
          <h2 className="type-panel-title text-foreground" id="events-over-time-title">
            Events Over Time
          </h2>
          <p className="type-caption mt-1 text-fg-dim">
            Loaded event volume by category and time bucket
            {wasAutoBucketed ? ` · Auto-bucketed to ${formatBucketInterval(effectiveIntervalMs)}` : ''}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <NativeSelect
            ariaLabel="Bucket interval"
            className="w-[6.5rem] min-w-0"
            name="volume-bucket-interval"
            onValueChange={(value) => setIntervalKey(value as BucketIntervalKey)}
            options={BUCKET_INTERVALS.map(({ value, label }) => ({ value, label }))}
            value={intervalKey}
          />
          <NativeSelect
            ariaLabel="Chart time range"
            className="w-[10rem] min-w-0"
            name="volume-time-range"
            onValueChange={(value) => setRangeSelection({ filter: selectedRange, value: value as VolumeTimeRangeKey })}
            options={[
              ...VOLUME_TIME_RANGES.map(({ value, label }) => ({ value, label })),
              ...(selectedRange === 'selected' ? [{ value: 'selected', label: 'Selected range' }] : [])
            ]}
            value={rangeKey}
          />
        </div>
      </div>

      {activeCategories.length > 0 ? (
        <ul aria-label="Visible event categories" className="mt-3 flex flex-wrap gap-x-4 gap-y-2" role="list">
          {activeCategories.map((category) => (
            <li className="inline-flex items-center gap-1.5 type-caption text-fg-dim" key={category}>
              <span
                aria-hidden="true"
                className="size-2 rounded-[2px]"
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
          <ChartContainer
            aria-label={`Events over time stacked bar chart. Showing ${activeCategories
              .map((category) => chartConfig[category].label)
              .join(', ')}.`}
            className="h-full w-full"
            config={chartConfig}
            role="img"
          >
            <BarChart data={data} margin={{ top: 8, right: 4, left: 0, bottom: 0 }} barCategoryGap={1.5}>
              <CartesianGrid vertical={false} stroke="var(--color-border-soft)" />
              <XAxis
                axisLine={false}
                dataKey="label"
                minTickGap={48}
                tick={{ fill: 'var(--color-fg-faint)', fontSize: 11 }}
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
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    formatter={(value) => `${String(value)} ${Number(value) === 1 ? 'event' : 'events'}`}
                    labelFormatter={(_label, payload) => {
                      const first = payload[0]?.payload
                      return formatBucketRange(Number(first?.bucketStart), Number(first?.bucketEnd))
                    }}
                    labelKey="label"
                  />
                }
                cursor={{ fill: 'var(--color-fg-faint)', fillOpacity: 0.08 }}
              />
              {activeCategories.map((category, categoryIndex) => (
                <Bar
                  dataKey={category}
                  fill={`var(--color-${category})`}
                  isAnimationActive={false}
                  key={category}
                  maxBarSize={8}
                  onMouseEnter={handleBarMouseEnter}
                  onMouseLeave={handleBarMouseLeave}
                  radius={categoryIndex === activeCategories.length - 1 ? [2, 2, 0, 0] : 0}
                  stackId="events"
                  stroke="var(--color-panel)"
                  strokeWidth={1}
                >
                  {data.map((bucket, index) => (
                    <Cell
                      key={`${category}-${bucket.bucketStart}`}
                      fillOpacity={activeIndex === undefined || activeIndex === index ? 1 : 0.35}
                    />
                  ))}
                </Bar>
              ))}
            </BarChart>
          </ChartContainer>
        </div>
      )}
    </Card>
  )
}
