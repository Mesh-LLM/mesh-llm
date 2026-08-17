import '@testing-library/jest-dom/vitest'

import { act, render, renderHook, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { RequestsOverTimeChart } from '@/features/logs/components/RequestsOverTimeChart'
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

function iso(ms: number): string {
  return new Date(ms).toISOString()
}

const EMPTY_MESSAGE = 'No requests during the selected time range.'

describe('RequestsOverTimeChart', () => {
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
    render(<RequestsOverTimeChart rows={[]} now={NOW} />)

    expect(screen.getByText('Requests Over Time')).toBeInTheDocument()
    expect(screen.getByText('Request volume by time bucket')).toBeInTheDocument()

    const bucketSelect = screen.getByLabelText('Bucket interval') as HTMLSelectElement
    const rangeSelect = screen.getByLabelText('Chart time range') as HTMLSelectElement
    expect(bucketSelect.value).toBe('5m')
    expect(rangeSelect.value).toBe('12h')
  })

  it('shows the empty state when there are no rows', () => {
    render(<RequestsOverTimeChart rows={[]} now={NOW} />)

    expect(screen.getByText(EMPTY_MESSAGE)).toBeInTheDocument()
    expect(screen.queryByLabelText('Requests over time bar chart')).not.toBeInTheDocument()
  })

  it('shows the empty state when every request falls outside the window', () => {
    const rows = [requestAt(iso(NOW - 13 * 3_600_000))]
    render(<RequestsOverTimeChart rows={rows} now={NOW} />)

    expect(screen.getByText(EMPTY_MESSAGE)).toBeInTheDocument()
  })

  it('uses a truthful seven-day chart window', () => {
    const rows = [requestAt(iso(NOW - 8 * 24 * 3_600_000))]
    render(<RequestsOverTimeChart rows={rows} now={NOW} selectedRange="7d" />)

    expect(screen.getByLabelText('Chart time range')).toHaveValue('7d')
    expect(screen.getByRole('option', { name: 'Last week' })).toBeInTheDocument()
    expect(screen.getByText(EMPTY_MESSAGE)).toBeInTheDocument()
  })

  it('renders the chart frame when requests fall inside the window', () => {
    const rows = [requestAt(iso(NOW - 10 * 60_000)), requestAt(iso(NOW - 5 * 60_000)), requestAt(iso(NOW))]
    render(<RequestsOverTimeChart rows={rows} now={NOW} />)

    expect(screen.queryByText(EMPTY_MESSAGE)).not.toBeInTheDocument()
    expect(screen.getByLabelText('Requests over time bar chart')).toBeInTheDocument()
  })

  it('switches the bucket interval and time range via the selectors', async () => {
    const user = userEvent.setup()
    render(<RequestsOverTimeChart rows={[]} now={NOW} />)

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
    render(<RequestsOverTimeChart rows={[requestAt(iso(start)), requestAt(iso(NOW))]} now={NOW} />)

    await user.selectOptions(screen.getByLabelText('Bucket interval'), '1m')
    await user.selectOptions(screen.getByLabelText('Chart time range'), 'all')

    expect(screen.getByText(/Auto-bucketed to/)).toBeInTheDocument()
    expect(screen.getByLabelText('Requests over time bar chart')).toBeInTheDocument()
  })
})
