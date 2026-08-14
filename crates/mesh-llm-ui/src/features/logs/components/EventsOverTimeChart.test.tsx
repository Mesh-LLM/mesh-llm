import '@testing-library/jest-dom/vitest'

import { act, render, renderHook, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { EventsOverTimeChart } from '@/features/logs/components/EventsOverTimeChart'
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
      observe() {}
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
