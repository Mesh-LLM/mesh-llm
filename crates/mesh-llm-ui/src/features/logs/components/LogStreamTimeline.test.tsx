// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogEventId, LogRequestId } from '@/features/logs/api/ids'
import type { LogLifecycleEvent } from '@/features/logs/api/schemas'
import { LogStreamTimeline } from '@/features/logs/components/LogStreamTimeline'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

let eventOrdinal = 0

function event(
  kind: LogLifecycleEvent['kind'],
  occurredAt: string,
  overrides: Partial<LogLifecycleEvent> = {}
): LogLifecycleEvent {
  eventOrdinal += 1
  return {
    eventId: LogEventId.parse(`00000000-0000-4000-8000-${String(eventOrdinal).padStart(12, '0')}`),
    requestId: REQUEST_ID,
    occurredAt,
    kind,
    model: undefined,
    provider: undefined,
    engine: undefined,
    attemptId: undefined,
    statusCode: undefined,
    durationMs: undefined,
    tokens: undefined,
    ...overrides
  }
}

describe('LogStreamTimeline', () => {
  it('lists every stream lifecycle event in occurrence order under the stream timeline label', () => {
    render(
      <LogStreamTimeline
        emptyMessage={undefined}
        events={[
          event('stream_completed', '2026-08-04T12:00:03.000Z'),
          event('stream_started', '2026-08-04T12:00:00.000Z'),
          event('stream_chunk', '2026-08-04T12:00:01.000Z')
        ]}
      />
    )

    const timeline = screen.getByRole('list', { name: 'Stream timeline' })
    expect(timeline.textContent).toMatch(/stream_started[\s\S]*stream_chunk[\s\S]*stream_completed/)
  })

  it('renders delta pills measured from the previous event and omits the first entry pill', () => {
    render(
      <LogStreamTimeline
        emptyMessage={undefined}
        events={[
          event('stream_started', '2026-08-04T12:00:00.000Z'),
          event('stream_chunk', '2026-08-04T12:00:00.084Z'),
          event('stream_chunk', '2026-08-04T12:00:00.111Z')
        ]}
      />
    )

    const timeline = screen.getByRole('list', { name: 'Stream timeline' })
    const entries = within(timeline).getAllByRole('listitem')
    expect(entries).toHaveLength(3)
    expect(within(entries[0]!).queryByText(/^\+/)).not.toBeInTheDocument()
    expect(within(entries[1]!).getByText('+84ms')).toBeInTheDocument()
    expect(within(entries[2]!).getByText('+27ms')).toBeInTheDocument()
  })

  it('labels legacy tokens as completion tokens beneath the timestamp', () => {
    render(
      <LogStreamTimeline
        emptyMessage={undefined}
        events={[event('stream_chunk', '2026-08-04T12:00:00.000Z', { attemptId: 'att-1', tokens: 128 })]}
      />
    )

    expect(screen.getByText('att-1')).toBeInTheDocument()
    expect(screen.getByText('Completion tokens: 128')).toBeInTheDocument()
    expect(screen.queryByText('128 tokens total')).not.toBeInTheDocument()
  })

  it('renders an explicit structured prompt, completion, and total breakdown when retained', () => {
    render(
      <LogStreamTimeline
        emptyMessage={undefined}
        events={[
          event('stream_completed', '2026-08-04T12:00:00.000Z', {
            promptTokens: 100,
            completionTokens: 28,
            totalTokens: 128
          })
        ]}
      />
    )

    expect(screen.getByText('Prompt tokens: 100 · Completion tokens: 28 · Total tokens: 128')).toBeInTheDocument()
  })

  it('tints the kind label with the event tone text color', () => {
    render(<LogStreamTimeline emptyMessage={undefined} events={[event('stream_error', '2026-08-04T12:00:00.000Z')]} />)

    expect(screen.getByText('stream_error')).toHaveClass('text-bad')
  })

  it('keeps millisecond precision on semantic timestamps', () => {
    render(
      <LogStreamTimeline emptyMessage={undefined} events={[event('stream_started', '2026-08-04T12:00:01.557Z')]} />
    )

    const stamp = screen.getByText(/\d{1,2}:\d{2}:\d{2}\.557/)
    expect(stamp.tagName).toBe('TIME')
    expect(stamp).toHaveAttribute('dateTime', '2026-08-04T12:00:01.557Z')
  })

  it('renders the empty message only when one is provided', () => {
    const { rerender } = render(
      <LogStreamTimeline emptyMessage="No lifecycle or stream markers were retained for this request." events={[]} />
    )
    expect(screen.getByText('No lifecycle or stream markers were retained for this request.')).toBeInTheDocument()
    expect(screen.queryByRole('list')).not.toBeInTheDocument()

    rerender(<LogStreamTimeline emptyMessage={undefined} events={[]} />)
    expect(screen.queryByText('No lifecycle or stream markers were retained for this request.')).not.toBeInTheDocument()
  })
})
