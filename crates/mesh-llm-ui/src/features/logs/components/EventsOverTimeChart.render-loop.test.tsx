import '@testing-library/jest-dom/vitest'

import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import { EventsOverTimeChart } from '@/features/logs/components/EventsOverTimeChart'
import {
  LOG_EVENT_CATEGORIES,
  type LogEventCategory,
  type LogEventLedgerRow
} from '@/features/logs/lib/log-event-ledger'

const NOW = Date.UTC(2026, 7, 4, 12, 0, 0)

function requestAt(createdAt: string): LogRequest {
  return {
    requestId: '00000000-0000-4000-8000-000000000001' as unknown as LogRequestId,
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

function eventAt(category: LogEventCategory, occurredAt: string): LogEventLedgerRow {
  if (category === 'requests') {
    return {
      type: 'request',
      id: `request:${occurredAt}`,
      occurredAt,
      category,
      request: requestAt(occurredAt)
    }
  }
  return {
    type: 'audit',
    id: `audit:${occurredAt}`,
    occurredAt,
    category,
    audit: {
      entryId: `audit-${occurredAt}`,
      occurredAt,
      source: 'runtime',
      code: `${category}_event`,
      sequence: 1
    }
  }
}

function iso(ms: number): string {
  return new Date(ms).toISOString()
}

/** Fresh row array on every call, mirroring the live log stream replacing rows. */
function freshRows(): LogEventLedgerRow[] {
  return [
    eventAt('requests', iso(NOW - 10 * 60_000)),
    eventAt('system', iso(NOW - 5 * 60_000)),
    eventAt('quic', iso(NOW))
  ]
}

class RecordingResizeObserverStub {
  static instances: RecordingResizeObserverStub[] = []

  constructor(private readonly callback: ResizeObserverCallback) {
    RecordingResizeObserverStub.instances.push(this)
  }

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
        } as unknown as ResizeObserverEntry
      ],
      this as unknown as ResizeObserver
    )
  }

  unobserve() {}

  disconnect() {}

  fireResize(width: number, height: number) {
    const target = document.createElement('div')
    this.callback(
      [
        {
          target,
          contentRect: {
            width,
            height,
            top: 0,
            right: width,
            bottom: height,
            left: 0,
            x: 0,
            y: 0,
            toJSON: () => ({})
          } as DOMRectReadOnly
        } as unknown as ResizeObserverEntry
      ],
      this as unknown as ResizeObserver
    )
  }
}

const ALL_CATEGORIES = new Set<LogEventCategory>(LOG_EVENT_CATEGORIES)

describe('EventsOverTimeChart recharts render-loop regression', () => {
  beforeEach(() => {
    RecordingResizeObserverStub.instances = []
    vi.stubGlobal('ResizeObserver', RecordingResizeObserverStub)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('stays within React update limits across repeated measurements and parent updates', () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    let rows = freshRows()
    const { rerender } = render(<EventsOverTimeChart rows={rows} selectedCategories={ALL_CATEGORIES} now={NOW} />)

    // ChartContainer (chart.tsx) renders a div carrying the aria-label.
    const container = screen.getByLabelText(/^events over time$/i)
    expect(container).toBeInTheDocument()

    const observers = RecordingResizeObserverStub.instances
    expect(observers.length).toBeGreaterThan(0)

    // Constant-size measurement cycles with fresh parent props (live log stream):
    // a re-dispatch feedback loop re-enters React's update machinery here.
    for (let cycle = 0; cycle < 30; cycle++) {
      rows = freshRows()
      act(() => {
        for (const observer of observers) observer.fireResize(640, 170)
        rerender(<EventsOverTimeChart rows={rows} selectedCategories={ALL_CATEGORIES} now={NOW} />)
      })
    }

    // Oscillating-size cycles exercise the size-change path as well.
    for (let cycle = 0; cycle < 20; cycle++) {
      rows = freshRows()
      const width = cycle % 2 === 0 ? 644 : 636
      act(() => {
        for (const observer of observers) observer.fireResize(width, 170)
        rerender(<EventsOverTimeChart rows={rows} selectedCategories={ALL_CATEGORIES} now={NOW} />)
      })
    }

    const depthErrors = consoleErrorSpy.mock.calls.filter((call) =>
      /maximum update depth exceeded/i.test(String(call[0]))
    )
    expect(depthErrors).toEqual([])

    // Functionality retained after the stress: chart and legend still rendered.
    expect(container).toBeInTheDocument()
    const legend = screen.getByRole('list', { name: 'Visible event categories' })
    expect(legend).toHaveTextContent('Requests1')

    consoleErrorSpy.mockRestore()
  })
})
