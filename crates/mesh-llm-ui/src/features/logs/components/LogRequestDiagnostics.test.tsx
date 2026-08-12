// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { LogArtifactId, LogEventId, LogRequestId } from '@/features/logs/api/ids'
import type { LogArtifact, LogLifecycleEvent, LogProxyAttempt, LogRequest } from '@/features/logs/api/schemas'
import { LogRequestDiagnostics } from '@/features/logs/components/LogRequestDiagnostics'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

function request(overrides: Partial<LogRequest> = {}): LogRequest {
  return {
    requestId: REQUEST_ID,
    outcome: 'completed',
    createdAt: '2026-08-04T12:00:00Z',
    terminalAt: '2026-08-04T12:00:04Z',
    route: 'reserve',
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable',
    ...overrides
  }
}

type ArtifactOptions = {
  readonly sequence: number
  readonly kind: string
  readonly occurredAt?: string
  readonly contentState?: LogArtifact['contentState']
  readonly redacted?: boolean
  readonly truncated?: boolean
}

function artifact(options: ArtifactOptions): LogArtifact {
  const contentState = options.contentState ?? 'available'
  const base = {
    artifactId: LogArtifactId.parse(`00000000-0000-4000-8000-${options.sequence.toString().padStart(12, '0')}`),
    requestId: REQUEST_ID,
    occurredAt: options.occurredAt ?? '2026-08-04T12:00:02Z',
    kind: options.kind,
    mediaKind: 'application/json',
    checksum: undefined,
    bytes: 384,
    version: 2,
    redacted: options.redacted ?? true,
    truncated: options.truncated ?? false
  }
  switch (contentState) {
    case 'available':
      return { ...base, contentState, contentBase64: undefined }
    case 'unavailable':
    case 'missing':
    case 'corrupt':
      return { ...base, contentState, contentBase64: undefined }
  }
}

type EventOptions = {
  readonly sequence: number
  readonly kind: LogLifecycleEvent['kind']
  readonly occurredAt: string
}

function event(options: EventOptions): LogLifecycleEvent {
  return {
    eventId: LogEventId.parse(`00000000-0000-4000-8000-${options.sequence.toString().padStart(12, '0')}`),
    requestId: REQUEST_ID,
    occurredAt: options.occurredAt,
    kind: options.kind,
    model: undefined,
    provider: undefined,
    engine: undefined,
    attemptId: undefined,
    statusCode: undefined,
    durationMs: undefined,
    tokens: undefined
  }
}

function attempt(sequence: number): LogProxyAttempt {
  return {
    attemptId: `attempt-${sequence}`,
    requestId: REQUEST_ID,
    occurredAt: `2026-08-04T12:00:0${sequence}Z`,
    target: 'opaque',
    provider: 'reserve-a',
    engine: 'skippy',
    startedAt: `2026-08-04T12:00:0${sequence}Z`,
    completedAt: `2026-08-04T12:00:0${sequence}Z`,
    statusCode: 502
  }
}

type DiagnosticsFixture = {
  readonly request: LogRequest
  readonly events?: readonly LogLifecycleEvent[]
  readonly attempts?: readonly LogProxyAttempt[]
  readonly artifacts?: readonly LogArtifact[]
}

function renderDiagnostics(fixture: DiagnosticsFixture) {
  return render(
    <LogRequestDiagnostics
      artifacts={fixture.artifacts ?? []}
      artifactsError={false}
      artifactsLoading={false}
      attempts={fixture.attempts ?? []}
      attemptsError={false}
      attemptsLoading={false}
      events={fixture.events ?? []}
      eventsError={false}
      eventsLoading={false}
      request={fixture.request}
      requestError={false}
      requestLoading={false}
    />
  )
}

function expectMetric(summary: HTMLElement, label: string, value: string): void {
  expect(within(summary).getByText(label).parentElement).toHaveTextContent(value)
}

describe('LogRequestDiagnostics', () => {
  it('summarizes successful typed retention evidence without viewing artifact bodies', () => {
    // Given
    const artifacts = [
      artifact({ sequence: 1, kind: 'request_body' }),
      artifact({
        sequence: 2,
        kind: 'response_archive',
        contentState: 'unavailable',
        redacted: false
      }),
      artifact({
        sequence: 3,
        kind: 'request_trace',
        contentState: 'missing',
        truncated: true
      }),
      artifact({
        sequence: 4,
        kind: 'response_trace',
        contentState: 'corrupt',
        redacted: false
      })
    ]

    // When
    renderDiagnostics({ request: request(), artifacts })

    // Then
    const summary = screen.getByLabelText('Diagnostic summary')
    expectMetric(summary, 'Request source', 'durable')
    expectMetric(summary, 'Artifact records', '4')
    expectMetric(summary, 'Redacted', '2 of 4')
    expectMetric(summary, 'Truncated', '1 of 4')
    expectMetric(summary, 'Content states', '1 available · 1 unavailable · 1 missing · 1 corrupt')
    expectMetric(summary, 'Artifact access', 'Metadata only; body content not requested')
    for (const value of [
      '1 available · 1 unavailable · 1 missing · 1 corrupt',
      'Metadata only; body content not requested'
    ]) {
      expect(within(summary).getByText(value, { exact: true })).toHaveClass('break-words')
    }
    // The 6-metric success grid fills 3 full rows of 2; no trailing span needed.
    expect(within(summary).getByText('Artifact access').parentElement).not.toHaveClass('sm:col-span-2')
    expect(within(screen.getByRole('region', { name: 'Terminal record' })).getByText(/HTTP 200/)).toBeInTheDocument()
  })

  it('places the typed failed-request summary before its ordered evidence', () => {
    // Given
    const failedRequest = request({
      outcome: 'failed',
      terminalAt: '2026-08-04T12:00:03Z',
      statusCode: 502
    })
    const events = [
      event({
        sequence: 1,
        kind: 'admitted',
        occurredAt: '2026-08-04T12:00:00Z'
      }),
      event({
        sequence: 2,
        kind: 'attempt_failed',
        occurredAt: '2026-08-04T12:00:01Z'
      }),
      event({
        sequence: 3,
        kind: 'failed',
        occurredAt: '2026-08-04T12:00:03Z'
      })
    ]
    const artifacts = [
      artifact({ sequence: 5, kind: 'error_trace', contentState: 'corrupt' }),
      artifact({ sequence: 6, kind: 'request_body' })
    ]

    // When
    renderDiagnostics({
      request: failedRequest,
      events,
      attempts: [attempt(1), attempt(2)],
      artifacts
    })

    // Then
    const summary = screen.getByLabelText('Diagnostic summary')
    expectMetric(summary, 'Outcome', 'failed')
    expectMetric(summary, 'HTTP status', '502')
    expectMetric(summary, 'Provider', 'reserve-a')
    expectMetric(summary, 'Engine', 'skippy')
    expectMetric(summary, 'Duration', '3.00 s')
    expectMetric(summary, 'Attempt count', '2')
    expectMetric(summary, 'Diagnostic markers', '3')
    expect(within(summary).getByText('reserve-a', { exact: true })).toHaveClass('break-words')
    // The 7-metric failed grid has a trailing row of one cell that spans both columns.
    expect(within(summary).getByText('Diagnostic markers').parentElement).toHaveClass('sm:col-span-2')
    const timeline = screen.getByRole('list', {
      name: 'Ordered diagnostic evidence'
    })
    expect(summary.compareDocumentPosition(timeline) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('keeps sparse diagnostics explicit without inventing unavailable fields', () => {
    // Given / When
    renderDiagnostics({
      request: request({
        outcome: 'failed',
        terminalAt: undefined,
        provider: undefined,
        engine: undefined,
        statusCode: undefined,
        source: 'active'
      })
    })

    // Then
    const summary = screen.getByLabelText('Diagnostic summary')
    expectMetric(summary, 'HTTP status', 'Not recorded')
    expectMetric(summary, 'Provider', 'Not recorded')
    expectMetric(summary, 'Engine', 'Not recorded')
    expectMetric(summary, 'Duration', 'Not recorded')
    expectMetric(summary, 'Attempt count', '0')
    expectMetric(summary, 'Diagnostic markers', '0')
    expect(
      screen.queryByText(/error type|error message|time to live|retention expires|body loaded/i)
    ).not.toBeInTheDocument()
  })

  it('orders diagnostic artifacts by instant while preserving equal-instant input order', () => {
    // Given
    const artifacts = [
      artifact({
        sequence: 7,
        kind: 'error_later',
        occurredAt: '2026-08-04T10:00:00-02:00'
      }),
      artifact({
        sequence: 8,
        kind: 'error_earlier',
        occurredAt: '2026-08-04T11:00:00Z'
      }),
      artifact({
        sequence: 9,
        kind: 'error_tied',
        occurredAt: '2026-08-04T13:00:00+01:00'
      })
    ]

    // When
    renderDiagnostics({ request: request({ outcome: 'failed' }), artifacts })

    // Then
    expect(screen.getByRole('region', { name: 'Error artifacts' }).textContent).toMatch(
      /error_earlier[\s\S]*error_later[\s\S]*error_tied/
    )
  })
})
