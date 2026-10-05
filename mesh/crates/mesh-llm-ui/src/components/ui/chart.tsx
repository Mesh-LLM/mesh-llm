import * as React from 'react'
import type { ComponentProps, ReactNode } from 'react'
import * as RechartsPrimitive from 'recharts'
import { cn } from '@/lib/cn'

export type ChartConfig = {
  readonly [key: string]: {
    readonly label?: ReactNode
    readonly icon?: React.ComponentType<{ className?: string }>
    readonly color?: string
  }
}

export type ChartTooltipPayloadItem = {
  readonly dataKey?: string | number
  readonly name?: string
  readonly value?: unknown
  readonly color?: string
  readonly payload?: Record<string, unknown>
}

type ChartContextValue = { readonly config: ChartConfig }

const ChartContext = React.createContext<ChartContextValue | null>(null)

function useChart() {
  const context = React.useContext(ChartContext)
  if (!context) throw new Error('useChart must be used within a <ChartContainer />')
  return context
}

export const ChartContainer = React.forwardRef<
  HTMLDivElement,
  Omit<ComponentProps<'div'>, 'children'> & {
    readonly config: ChartConfig
    readonly children: ComponentProps<typeof RechartsPrimitive.ResponsiveContainer>['children']
  }
>(({ id, className, config, children, ...props }, ref) => {
  const uniqueId = React.useId()
  const chartId = `chart-${id ?? uniqueId.replace(/:/g, '')}`
  return (
    <ChartContext.Provider value={{ config }}>
      <div
        data-chart={chartId}
        ref={ref}
        className={cn(
          'flex justify-center text-[length:var(--density-type-caption)]',
          '[&_.recharts-cartesian-axis-tick_text]:fill-fg-faint',
          '[&_.recharts-cartesian-grid_line]:stroke-border-soft',
          '[&_.recharts-layer]:outline-none',
          '[&_.recharts-rectangle.recharts-tooltip-cursor]:fill-fg-faint',
          '[&_.recharts-surface]:outline-none',
          className
        )}
        {...props}
      >
        <ChartStyle id={chartId} config={config} />
        <RechartsPrimitive.ResponsiveContainer>{children}</RechartsPrimitive.ResponsiveContainer>
      </div>
    </ChartContext.Provider>
  )
})
ChartContainer.displayName = 'ChartContainer'

function ChartStyle({ id, config }: { readonly id: string; readonly config: ChartConfig }) {
  const colorConfig = Object.entries(config).filter(([, item]) => item.color)
  if (colorConfig.length === 0) return null
  const css = `[data-chart="${id}"] {\n${colorConfig
    .map(([key, item]) => `  --color-${key}: ${item.color};`)
    .join('\n')}\n}`
  return <style>{css}</style>
}

export function ChartTooltip(props: ComponentProps<typeof RechartsPrimitive.Tooltip>) {
  return <RechartsPrimitive.Tooltip {...props} />
}

type ChartTooltipContentProps = {
  readonly active?: boolean
  readonly payload?: readonly ChartTooltipPayloadItem[]
  readonly label?: unknown
  readonly className?: string
  readonly hideLabel?: boolean
  readonly hideIndicator?: boolean
  readonly hideName?: boolean
  readonly indicator?: 'dot' | 'line' | 'none'
  readonly labelKey?: string
  readonly nameKey?: string
  readonly labelFormatter?: (label: unknown, payload: readonly ChartTooltipPayloadItem[]) => ReactNode
  readonly formatter?: (value: unknown, name: string, item: ChartTooltipPayloadItem, index: number) => ReactNode
}

export function ChartTooltipContent({
  active,
  payload,
  label,
  className,
  hideLabel = false,
  hideIndicator = false,
  hideName = false,
  indicator = 'dot',
  labelKey,
  nameKey,
  labelFormatter,
  formatter
}: ChartTooltipContentProps) {
  const { config } = useChart()
  if (!active || !payload || payload.length === 0) return null

  const firstItem = payload[0]
  const dataKey = firstItem?.dataKey
  const configKey = labelKey ?? nameKey ?? (typeof dataKey === 'string' ? dataKey : undefined)
  const configItem = configKey ? config[configKey] : undefined
  const resolvedLabel = labelKey && firstItem?.payload ? firstItem.payload[labelKey] : label

  return (
    <div
      className={cn(
        'rounded-[var(--radius)] border border-border-soft bg-panel-strong px-3 py-2 shadow-surface-low',
        className
      )}
    >
      {!hideLabel && resolvedLabel != null ? (
        <div className="mb-1.5 text-[length:var(--density-type-label)] font-medium text-fg">
          {labelFormatter ? labelFormatter(resolvedLabel, payload) : String(resolvedLabel)}
        </div>
      ) : null}
      <div className="flex flex-col gap-1">
        {payload.map((item, index) => {
          const itemName = nameKey && item.payload ? String(item.payload[nameKey]) : item.name
          const itemConfig = itemName ? config[itemName] : undefined
          const color = item.color ?? itemConfig?.color ?? configItem?.color
          const displayName = hideName ? undefined : (itemConfig?.label ?? itemName)
          return (
            <div
              key={`chart-item-${String(itemName ?? '')}-${String(dataKey ?? '')}-${index}`}
              className="flex items-center gap-2"
            >
              {!hideIndicator && indicator !== 'none' ? (
                <span
                  aria-hidden="true"
                  className={cn('shrink-0 rounded-full', indicator === 'dot' ? 'size-2' : 'h-0.5 w-3.5')}
                  style={{ backgroundColor: color }}
                />
              ) : null}
              {displayName ? <span className="text-fg-dim">{displayName}</span> : null}
              <span className={cn('font-mono font-medium tabular-nums text-fg', displayName ? 'ml-auto' : 'ml-1')}>
                {formatter ? formatter(item.value, itemName ?? '', item, index) : String(item.value)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
