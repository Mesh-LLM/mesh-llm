// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogProxyAttempt } from '@/features/logs/api/schemas'
import { LogRoutingAttemptsTimeline } from '@/features/logs/components/LogRoutingAttemptsTimeline'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

function attempt(attemptId: string, occurredAt: string, overrides: Partial<LogProxyAttempt> = {}): LogProxyAttempt {
  return {
    attemptId,
    requestId: REQUEST_ID,
    occurredAt,
    target: 'opaque',
    provider: undefined,
    engine: undefined,
    startedAt: undefined,
    completedAt: undefined,
    statusCode: undefined,
    ...overrides
  }
}

describe('LogRoutingAttemptsTimeline', () => {
  it('lists attempts in occurrence order under the routing attempts label', () => {
    render(
      <LogRoutingAttemptsTimeline
        attempts={[
          attempt('retry-secondary', '2026-08-04T12:00:02.000Z'),
          attempt('retry-primary', '2026-08-04T12:00:01.000Z')
        ]}
        emptyMessage={undefined}
      />
    )

    const timeline = screen.getByRole('list', { name: 'Routing attempts timeline' })
    expect(timeline.textContent).toMatch(/retry-primary[\s\S]*retry-secondary/)
  })

  it('labels recorded statuses Success or Failed and shows the HTTP line and duration', () => {
    render(
      <LogRoutingAttemptsTimeline
        attempts={[
          attempt('att-ok', '2026-08-04T12:00:00.000Z', {
            startedAt: '2026-08-04T12:00:00.000Z',
            completedAt: '2026-08-04T12:00:00.843Z',
            statusCode: 200
          }),
          attempt('att-bad', '2026-08-04T12:00:01.000Z', {
            startedAt: '2026-08-04T12:00:01.000Z',
            completedAt: '2026-08-04T12:00:02.000Z',
            statusCode: 502
          })
        ]}
        emptyMessage={undefined}
      />
    )

    expect(screen.getByText('Success')).toBeInTheDocument()
    expect(screen.getByText('Failed')).toBeInTheDocument()
    expect(screen.getByText('HTTP 200')).toBeInTheDocument()
    expect(screen.getByText('HTTP 502')).toBeInTheDocument()
    expect(screen.getByText('843ms')).toBeInTheDocument()
    expect(screen.getByText('1s')).toBeInTheDocument()
  })

  it('labels a started unfinished attempt In progress without an HTTP line', () => {
    render(
      <LogRoutingAttemptsTimeline
        attempts={[attempt('att-live', '2026-08-04T12:00:00.000Z', { startedAt: '2026-08-04T12:00:00.000Z' })]}
        emptyMessage={undefined}
      />
    )

    expect(screen.getByText('In progress')).toBeInTheDocument()
    expect(screen.queryByText(/^HTTP /)).not.toBeInTheDocument()
  })

  it('labels attempts with no recorded status explicitly', () => {
    render(
      <LogRoutingAttemptsTimeline
        attempts={[attempt('att-bare', '2026-08-04T12:00:00.000Z')]}
        emptyMessage={undefined}
      />
    )

    expect(screen.getByText('Status not recorded')).toBeInTheDocument()
  })

  it('shows the provider and engine route pair when recorded', () => {
    render(
      <LogRoutingAttemptsTimeline
        attempts={[attempt('att-1', '2026-08-04T12:00:00.000Z', { provider: 'mesh-routed', engine: 'skippy' })]}
        emptyMessage={undefined}
      />
    )

    expect(screen.getByText('mesh-routed / skippy')).toBeInTheDocument()
  })

  it('falls back to the attempt target with word-boundary wrapping when no route pair exists', () => {
    const target = 'https://peer-a.mesh.invalid/v1/chat/completions'
    render(
      <LogRoutingAttemptsTimeline
        attempts={[attempt('att-1', '2026-08-04T12:00:00.000Z', { target })]}
        emptyMessage={undefined}
      />
    )

    const route = screen.getByText(target, { exact: true })
    expect(route).toHaveClass('break-words')
    expect(route).not.toHaveClass('break-all')
  })

  it('renders the millisecond start to completion range and start-only ranges', () => {
    render(
      <LogRoutingAttemptsTimeline
        attempts={[
          attempt('att-ranged', '2026-08-04T12:00:00.000Z', {
            startedAt: '2026-08-04T12:00:00.327Z',
            completedAt: '2026-08-04T12:00:01.843Z'
          }),
          attempt('att-started', '2026-08-04T12:00:02.000Z', { startedAt: '2026-08-04T12:00:02.100Z' })
        ]}
        emptyMessage={undefined}
      />
    )

    const ranged = screen.getByText(/\d{1,2}:\d{2}:\d{2}\.327/)
    expect(ranged.tagName).toBe('TIME')
    expect(ranged).toHaveAttribute('dateTime', '2026-08-04T12:00:00.327Z')
    expect(ranged.parentElement?.textContent).toMatch(/→/)
    expect(screen.getByText(/\d{1,2}:\d{2}:\d{2}\.843/)).toBeInTheDocument()

    const startedOnly = screen.getByText(/\d{1,2}:\d{2}:\d{2}\.100/)
    expect(startedOnly.parentElement?.textContent).not.toMatch(/→/)
  })

  it('renders the empty message only when one is provided', () => {
    const { rerender } = render(
      <LogRoutingAttemptsTimeline attempts={[]} emptyMessage="No proxy attempts were retained for this request." />
    )
    expect(screen.getByText('No proxy attempts were retained for this request.')).toBeInTheDocument()
    expect(screen.queryByRole('list')).not.toBeInTheDocument()

    rerender(<LogRoutingAttemptsTimeline attempts={[]} emptyMessage={undefined} />)
    expect(screen.queryByText('No proxy attempts were retained for this request.')).not.toBeInTheDocument()
  })
})
