import '@testing-library/jest-dom/vitest'

import { act, render, renderHook, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ChartTooltipPayloadItem } from '@/components/ui/chart'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { EventsOverTimeChart } from '@/features/logs/components/EventsOverTimeChart'
import { hasVisibleEventVolumeTooltip } from '@/features/logs/components/events-over-time-chart-tooltip'
import {
  LOG_EVENT_CATEGORIES,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'
import { useAdvancingChartClock } from '@/features/logs/lib/use-advancing-chart-clock'

const NOW = Date.UTC(2026, 7, 4, 12, 0, 0)

function requestAt(createdAt: string): LogRequest {
  return {
    requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000001'),
    outcome: 'completed',
    createdAt,
    terminalAt: undefined,
    route: 'reserve',
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable'
  }
}

function eventAt(category: LogEventCategory, occurredAt: string, index = 1): LogEventLedgerRow {
  if (category === 'requests') {
    return {
      type: 'request',
      id: `request:${index}`,
      occurredAt,
      category,
      request: requestAt(occurredAt)
    }
  }
  return {
    type: 'audit',
    id: `audit:${index}`,
    occurredAt,
    category,
    audit: {
      entryId: `audit-${index}`,
      occurredAt,
      source: 'runtime',
      code: `${category}_event`,
      sequence: index
    }
  }
}

function iso(ms: number): string {
  return new Date(ms).toISOString()
}

const ALL_CATEGORIES = new Set<LogEventCategory>(LOG_EVENT_CATEGORIES)
const EMPTY_MESSAGE = 'No selected events during the chart time range.'

describe('EventsOverTimeChart', () => {
  beforeEach(() => {
    class ResizeObserverStub {
      constructor(private readonly callback: ResizeObserverCallback) {}

      observe(target: Element) {
        this.callback(
          [
            {
              target,
              contentRect: {
                width: 640,
                height: 170,
                top: 0,
                right: 640,
                bottom: 170,
                left: 0,
                x: 0,
                y: 0,
                toJSON: () => ({})
              } as DOMRectReadOnly
            } as ResizeObserverEntry
          ],
          this as unknown as ResizeObserver
        )
      }

      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal('ResizeObserver', ResizeObserverStub)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('advances finite windows on a minute-aligned clock and cleans up its timer', () => {
    vi.useFakeTimers({ now: NOW + 30_000 })
    const { result, unmount } = renderHook(() => useAdvancingChartClock())

    expect(result.current).toBe(NOW + 30_000)
    expect(vi.getTimerCount()).toBe(1)
    act(() => vi.advanceTimersByTime(30_000))
    expect(result.current).toBe(NOW + 60_000)
    expect(vi.getTimerCount()).toBe(1)

    unmount()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('refreshes the clock immediately when updates are re-enabled', () => {
    vi.useFakeTimers({ now: NOW })
    const { result, rerender } = renderHook(({ enabled }) => useAdvancingChartClock(enabled), {
      initialProps: { enabled: false }
    })

    act(() => vi.advanceTimersByTime(15_000))
    expect(result.current).toBe(NOW)

    rerender({ enabled: true })
    expect(result.current).toBe(NOW + 15_000)
  })

  it('renders the card header with bucket and time range selectors', () => {
    render(<EventsOverTimeChart rows={[]} selectedCategories={ALL_CATEGORIES} now={NOW} />)

    expect(screen.getByText('Events Over Time')).toBeInTheDocument()
    expect(screen.getByText('Loaded event volume by category and time bucket')).toBeInTheDocument()

    const bucketSelect = screen.getByLabelText('Bucket interval') as HTMLSelectElement
    const rangeSelect = screen.getByLabelText('Chart time range') as HTMLSelectElement
    expect(bucketSelect.value).toBe('5m')
    expect(rangeSelect.value).toBe('12h')
  })

  it('reports time-range changes to an owning page', async () => {
    const user = userEvent.setup()
    const onSelectedRangeChange = vi.fn()
    render(
      <EventsOverTimeChart
        onSelectedRangeChange={onSelectedRangeChange}
        rows={[]}
        selectedCategories={ALL_CATEGORIES}
        selectedRange="all"
        now={NOW}
      />
    )

    await user.selectOptions(screen.getByLabelText('Chart time range'), '6h')

    expect(onSelectedRangeChange).toHaveBeenCalledWith('6h')
  })

  it('shows the empty state when there are no rows', () => {
    render(<EventsOverTimeChart rows={[]} selectedCategories={ALL_CATEGORIES} now={NOW} />)

    expect(screen.getByText(EMPTY_MESSAGE)).toBeInTheDocument()
    expect(screen.queryByLabelText(/Events over time stacked bar chart/)).not.toBeInTheDocument()
  })

  it('shows the empty state when every selected event falls outside the window', () => {
    const rows = [eventAt('requests', iso(NOW - 13 * 3_600_000))]
    render(<EventsOverTimeChart rows={rows} selectedCategories={ALL_CATEGORIES} now={NOW} />)

    expect(screen.getByText(EMPTY_MESSAGE)).toBeInTheDocument()
  })

  it('shows a filter-specific empty state when no categories are selected', () => {
    render(<EventsOverTimeChart rows={[]} selectedCategories={new Set()} now={NOW} />)

    expect(screen.getByText('Select an event category to display the chart.')).toBeInTheDocument()
    expect(screen.queryByRole('list', { name: 'Visible event categories' })).not.toBeInTheDocument()
  })

  it('uses a truthful seven-day chart window', () => {
    const rows = [eventAt('requests', iso(NOW - 8 * 24 * 3_600_000))]
    render(<EventsOverTimeChart rows={rows} selectedCategories={ALL_CATEGORIES} now={NOW} selectedRange="7d" />)

    expect(screen.getByLabelText('Chart time range')).toHaveValue('7d')
    expect(screen.getByRole('option', { name: 'Last week' })).toBeInTheDocument()
    expect(screen.getByText(EMPTY_MESSAGE)).toBeInTheDocument()
  })

  it('renders stacked category series and totals for selected events inside the window', () => {
    const rows = [
      eventAt('requests', iso(NOW - 10 * 60_000), 1),
      eventAt('system', iso(NOW - 5 * 60_000), 2),
      eventAt('quic', iso(NOW), 3)
    ]
    render(<EventsOverTimeChart rows={rows} selectedCategories={ALL_CATEGORIES} now={NOW} />)

    expect(screen.queryByText(EMPTY_MESSAGE)).not.toBeInTheDocument()
    expect(screen.getByLabelText(/Events over time stacked bar chart/)).toHaveAccessibleName(
      'Events over time stacked bar chart. Showing Requests, System, QUIC, Gossip, Iroh.'
    )
    const legend = screen.getByRole('list', { name: 'Visible event categories' })
    expect(legend).toHaveTextContent('Requests1')
    expect(legend).toHaveTextContent('System1')
    expect(legend).toHaveTextContent('QUIC1')
    expect(legend).toHaveTextContent('Gossip0')
  })

  it('uses a stable, differentiated series palette and marker shapes', () => {
    const rows = LOG_EVENT_CATEGORIES.map((category, index) =>
      eventAt(category, iso(NOW - index * 5 * 60_000), index + 1)
    )
    render(<EventsOverTimeChart rows={rows} selectedCategories={ALL_CATEGORIES} now={NOW} />)

    const markers = within(screen.getByRole('list', { name: 'Visible event categories' }))
      .getAllByRole('listitem')
      .map((item) => item.querySelector<HTMLElement>('[aria-hidden="true"]'))

    expect(markers.every((marker) => marker !== null)).toBe(true)
    expect(new Set(markers.map((marker) => marker?.getAttribute('style'))).size).toBe(5)
    expect(new Set(markers.map((marker) => marker?.className)).size).toBe(5)
    expect(markers.every((marker) => marker?.getAttribute('style')?.includes('var(--color-log-'))).toBe(true)
    expect(markers.map((marker) => marker?.getAttribute('style')).join(' ')).not.toContain('var(--color-accent)')
  })

  it('suppresses zero-volume tooltip payloads while retaining populated buckets', () => {
    const zeroBucket = { bucketStart: NOW - 60_000, bucketEnd: NOW, total: 0 }
    const populatedBucket = { bucketStart: NOW - 60_000, bucketEnd: NOW, total: 1 }

    expect(hasVisibleEventVolumeTooltip([{ payload: zeroBucket } as ChartTooltipPayloadItem])).toBe(false)
    expect(hasVisibleEventVolumeTooltip([{ payload: populatedBucket } as ChartTooltipPayloadItem])).toBe(true)
    expect(hasVisibleEventVolumeTooltip(undefined)).toBe(false)
  })

  it('removes filtered categories from the chart legend and accessible series list', () => {
    const rows = [eventAt('requests', iso(NOW), 1), eventAt('system', iso(NOW), 2)]
    render(<EventsOverTimeChart rows={rows} selectedCategories={new Set<LogEventCategory>(['system'])} now={NOW} />)

    const legend = screen.getByRole('list', { name: 'Visible event categories' })
    expect(legend).toHaveTextContent('System1')
    expect(legend).not.toHaveTextContent('Requests')
    expect(screen.getByLabelText(/Events over time stacked bar chart/)).toHaveAccessibleName(
      'Events over time stacked bar chart. Showing System.'
    )
  })

  it('switches the bucket interval and time range via the selectors', async () => {
    const user = userEvent.setup()
    render(<EventsOverTimeChart rows={[]} selectedCategories={ALL_CATEGORIES} now={NOW} />)

    const bucketSelect = screen.getByLabelText('Bucket interval') as HTMLSelectElement
    const rangeSelect = screen.getByLabelText('Chart time range') as HTMLSelectElement

    await user.selectOptions(bucketSelect, '1h')
    expect(bucketSelect.value).toBe('1h')

    await user.selectOptions(rangeSelect, '24h')
    expect(rangeSelect.value).toBe('24h')
  })

  it('reports automatic bucket promotion for sparse 90-day endpoints', async () => {
    const user = userEvent.setup()
    const start = NOW - 90 * 24 * 60 * 60 * 1_000
    render(
      <EventsOverTimeChart
        rows={[eventAt('requests', iso(start), 1), eventAt('requests', iso(NOW), 2)]}
        selectedCategories={ALL_CATEGORIES}
        now={NOW}
      />
    )

    await user.selectOptions(screen.getByLabelText('Bucket interval'), '1m')
    await user.selectOptions(screen.getByLabelText('Chart time range'), 'all')

    expect(screen.getByText(/Auto-bucketed to/)).toBeInTheDocument()
    expect(screen.getByLabelText(/Events over time stacked bar chart/)).toBeInTheDocument()
  })
})
