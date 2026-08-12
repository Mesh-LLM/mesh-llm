// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogEventId, LogRequestId } from '@/features/logs/api/ids'
import type { LogLifecycleEvent, LogProxyAttempt } from '@/features/logs/api/schemas'
import { LogRequestEvidenceTimeline } from '@/features/logs/components/LogRequestEvidenceTimeline'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

function event(occurredAt: string): LogLifecycleEvent {
  return {
    eventId: LogEventId.parse('00000000-0000-4000-8000-000000000001'),
    requestId: REQUEST_ID,
    occurredAt,
    kind: 'failed',
    model: undefined,
    provider: undefined,
    engine: undefined,
    attemptId: undefined,
    statusCode: undefined,
    durationMs: undefined,
    tokens: undefined
  }
}

function attempt(attemptId: string, occurredAt: string, target = attemptId): LogProxyAttempt {
  return {
    attemptId,
    requestId: REQUEST_ID,
    occurredAt,
    target,
    provider: undefined,
    engine: undefined,
    startedAt: occurredAt,
    completedAt: occurredAt,
    statusCode: 502
  }
}

describe('LogRequestEvidenceTimeline', () => {
  it('orders combined offset timestamps by instant with stable equal-instant ties', () => {
    // Given
    const attempts = [
      attempt('later-attempt', '2026-08-04T10:00:00-02:00'),
      attempt('tied-attempt', '2026-08-04T13:00:00+01:00')
    ]
    const events = [event('2026-08-04T11:00:00Z')]

    // When
    render(
      <LogRequestEvidenceTimeline
        ariaLabel="Offset evidence"
        attemptEmptyMessage={undefined}
        attempts={attempts}
        eventEmptyMessage={undefined}
        events={events}
      />
    )

    // Then
    expect(screen.getByRole('list', { name: 'Offset evidence' }).textContent).toMatch(
      /failed[\s\S]*later-attempt[\s\S]*tied-attempt/
    )
  })

  it('wraps the attempt target at word boundaries instead of breaking mid-token', () => {
    // Given
    const target = 'https://peer-a.mesh.invalid/v1/chat/completions'
    const attempts = [attempt('attempt-1', '2026-08-04T10:00:00-02:00', target)]

    // When
    render(
      <LogRequestEvidenceTimeline
        ariaLabel="Target wrap"
        attemptEmptyMessage={undefined}
        attempts={attempts}
        eventEmptyMessage={undefined}
        events={[event('2026-08-04T11:00:00Z')]}
      />
    )

    // Then
    const targetCode = screen.getByText(target, { exact: true })
    expect(targetCode).toHaveClass('break-words')
    expect(targetCode).not.toHaveClass('break-all')
  })
})
