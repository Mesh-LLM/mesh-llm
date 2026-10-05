import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LogArtifactId, LogEventId, LogPageCursor, LogRequestId } from '@/features/logs/api/ids'
import type { LogArtifact, LogLifecycleEvent, LogProxyAttempt, LogRequest } from '@/features/logs/api/schemas'
import {
  HARNESS_LOG_FIXTURES,
  HARNESS_LOG_SCENARIO_IDS,
  generateArtifacts,
  generateLifecycleEvents,
  generateProxyAttempts
} from '@/features/logs/lib/log-fixtures'
import type { LogRequestDetailTab } from '@/features/logs/lib/log-request-details'

const hooks = vi.hoisted(() => ({
  summary: vi.fn(),
  events: vi.fn(),
  artifacts: vi.fn(),
  attempts: vi.fn()
}))

vi.mock('@/features/logs/api/use-log-request-details-query', () => ({
  useLogRequestSummaryQuery: (...args: unknown[]) => hooks.summary(...args),
  useLogRequestEventsQuery: (...args: unknown[]) => hooks.events(...args),
  useLogRequestArtifactsQuery: (...args: unknown[]) => hooks.artifacts(...args),
  useLogRequestAttemptsQuery: (...args: unknown[]) => hooks.attempts(...args)
}))

import { LogRequestDetails } from '@/features/logs/components/LogRequestDetails'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

function request(): LogRequest {
  return {
    requestId: REQUEST_ID,
    outcome: 'failed',
    createdAt: '2026-08-04T12:00:00Z',
    terminalAt: '2026-08-04T12:00:03Z',
    route: 'reserve',
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    statusCode: 502,
    source: 'durable'
  }
}

function event(eventId: string, kind: LogLifecycleEvent['kind'], occurredAt: string): LogLifecycleEvent {
  return {
    eventId: LogEventId.parse(eventId),
    requestId: REQUEST_ID,
    occurredAt,
    kind,
    model: undefined,
    provider: undefined,
    engine: undefined,
    attemptId: kind === 'attempt_failed' ? 'attempt-two' : undefined,
    statusCode: 502,
    durationMs: 12,
    tokens: 3
  }
}

function artifact(
  kind: string,
  contentState: LogArtifact['contentState'],
  redacted = true,
  artifactId = '00000000-0000-4000-8000-000000000011'
): LogArtifact {
  const base = {
    artifactId: LogArtifactId.parse(artifactId),
    requestId: REQUEST_ID,
    occurredAt: '2026-08-04T12:00:02Z',
    kind,
    mediaKind: 'application/json',
    checksum: 'sha256:0123456789abcdef',
    bytes: 384,
    version: 2,
    redacted,
    truncated: true
  }
  if (contentState === 'available') return { ...base, contentState, contentBase64: 'ZXhhbXBsZQ==' }
  return { ...base, contentState, contentBase64: undefined }
}

function attempt(attemptId: string, occurredAt: string): LogProxyAttempt {
  return {
    attemptId,
    requestId: REQUEST_ID,
    occurredAt,
    target: 'opaque',
    provider: 'reserve-a',
    engine: 'skippy',
    startedAt: occurredAt,
    completedAt: occurredAt,
    statusCode: 502
  }
}

function ready<T>(data: T) {
  return { data, isLoading: false, isError: false }
}

function renderDetails(tab: LogRequestDetailTab = 'overview') {
  return render(<LogRequestDetails onBack={vi.fn()} onTabChange={vi.fn()} requestId={REQUEST_ID} tab={tab} />)
}

function renderFixtureDetails(
  requestId: LogRequestId,
  tab: LogRequestDetailTab,
  artifacts: readonly LogArtifact[] = generateArtifacts(requestId.toString())
) {
  const summary = HARNESS_LOG_FIXTURES.find((request) => request.requestId.toString() === requestId.toString())
  if (summary === undefined) throw new Error(`Missing harness request ${requestId.toString()}`)
  hooks.summary.mockReturnValue(ready(summary))
  hooks.events.mockReturnValue(ready({ items: generateLifecycleEvents(requestId.toString()) }))
  hooks.artifacts.mockReturnValue(ready({ items: artifacts }))
  hooks.attempts.mockReturnValue(ready({ items: generateProxyAttempts(requestId.toString()) }))
  return render(<LogRequestDetails onBack={vi.fn()} onTabChange={vi.fn()} requestId={requestId} tab={tab} />)
}

describe('LogRequestDetails', () => {
  beforeEach(() => {
    hooks.summary.mockReset()
    hooks.events.mockReset()
    hooks.artifacts.mockReset()
    hooks.attempts.mockReset()
    hooks.summary.mockReturnValue(ready(request()))
    hooks.events.mockReturnValue(
      ready({
        items: [
          event('00000000-0000-4000-8000-000000000003', 'stream_chunk', '2026-08-04T12:00:03Z'),
          event('00000000-0000-4000-8000-000000000002', 'attempt_failed', '2026-08-04T12:00:02Z')
        ]
      })
    )
    hooks.artifacts.mockReturnValue(
      ready({
        items: [
          artifact('request', 'available'),
          artifact('response', 'missing', true, '00000000-0000-4000-8000-000000000012'),
          artifact('<img src=x onerror=error>', 'corrupt', true, '00000000-0000-4000-8000-000000000013')
        ]
      })
    )
    hooks.attempts.mockReturnValue(
      ready({ items: [attempt('second', '2026-08-04T12:00:04Z'), attempt('first', '2026-08-04T12:00:01Z')] })
    )
  })

  it('loads the summary and Overview metadata while focusing the request heading', () => {
    renderDetails()

    expect(screen.getByRole('heading', { name: REQUEST_ID.toString() })).toHaveFocus()
    expect(hooks.events).toHaveBeenCalledWith(REQUEST_ID, true)
    expect(hooks.artifacts).toHaveBeenCalledWith(REQUEST_ID, true)
    expect(hooks.attempts).toHaveBeenCalledWith(REQUEST_ID, true)
    expect(screen.getByRole('button', { name: 'Copy Request ID' })).toBeInTheDocument()
  })

  it('renders exactly four underlined tabs and preserves keyboard activation', async () => {
    const user = userEvent.setup()
    const onTabChange = vi.fn()
    render(<LogRequestDetails onBack={vi.fn()} onTabChange={onTabChange} requestId={REQUEST_ID} tab="overview" />)

    const tabList = screen.getByRole('tablist', { name: 'Request detail tabs' })
    const tabs = within(tabList).getAllByRole('tab')
    expect(tabs.map((tab) => tab.textContent)).toEqual(['Overview', 'Payloads', 'Timeline', 'Diagnostics'])
    expect(tabList.parentElement).toHaveClass('panel-divider', 'border-b', 'bg-transparent')
    expect(tabs[0]).toHaveAttribute('data-active', 'true')
    expect(tabs[0]?.style.borderBottomColor).toBe('var(--color-accent)')
    expect(tabs[1]).not.toHaveAttribute('data-active')
    expect(tabs[1]?.style.borderBottomColor).toBe('transparent')

    tabs[0]?.focus()
    await user.keyboard('{ArrowRight}')
    expect(onTabChange).toHaveBeenCalledWith('payloads')
  })

  const queryCases = [
    { name: 'Overview', tab: 'overview', events: true, artifacts: true, attempts: true },
    { name: 'Payloads', tab: 'payloads', events: false, artifacts: true, attempts: false },
    { name: 'Timeline', tab: 'timeline', events: true, artifacts: false, attempts: true },
    { name: 'Diagnostics', tab: 'diagnostics', events: true, artifacts: true, attempts: true }
  ] as const

  it.each(queryCases)('consolidates retained-data query enables for $name', ({ tab, events, artifacts, attempts }) => {
    renderDetails(tab)

    expect(hooks.summary).toHaveBeenCalledWith(REQUEST_ID, undefined)
    expect(hooks.events).toHaveBeenCalledWith(REQUEST_ID, events)
    expect(hooks.artifacts).toHaveBeenCalledWith(REQUEST_ID, artifacts)
    expect(hooks.attempts).toHaveBeenCalledWith(REQUEST_ID, attempts)
  })

  it('renders hostile error artifact text as text rather than markup', () => {
    hooks.artifacts.mockReturnValue(ready({ items: [artifact('error-<img src=x onerror=error>', 'corrupt')] }))
    renderDetails('diagnostics')

    expect(screen.getByText('error-<img src=x onerror=error>')).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })

  it('discloses when bounded diagnostics are incomplete', () => {
    hooks.events.mockReturnValue(ready({ items: [], nextCursor: LogPageCursor.parse('250') }))

    renderDetails('diagnostics')

    expect(
      screen.getByText('This request exceeds the bounded diagnostic limit. The records shown below are incomplete.')
    ).toHaveAttribute('role', 'status')
  })

  it.each([
    ['loading', { data: undefined, isLoading: true, isError: false }, /Loading request summary/],
    ['error', { data: undefined, isLoading: false, isError: true }, /request summary could not be loaded/]
  ])('renders the request summary %s state', (_state, state, label) => {
    hooks.summary.mockReturnValue(state)
    renderDetails()

    expect(screen.getByText(label)).toBeInTheDocument()
  })

  it('returns to the ledger context with a labeled back action', async () => {
    const user = userEvent.setup()
    const onBack = vi.fn()
    render(<LogRequestDetails onBack={onBack} onTabChange={vi.fn()} requestId={REQUEST_ID} tab="overview" />)

    await user.click(screen.getByRole('button', { name: 'Back to logs' }))
    expect(onBack).toHaveBeenCalledOnce()
  })

  it('combines ordered stream markers and routing attempts in Timeline', () => {
    renderFixtureDetails(HARNESS_LOG_SCENARIO_IDS.completedMesh, 'timeline')

    const streamTimeline = screen.getByRole('list', { name: 'Stream timeline' })
    expect(streamTimeline.textContent).toMatch(/stream_started[\s\S]*stream_chunk[\s\S]*stream_completed/)
    const attemptsTimeline = screen.getByRole('list', { name: 'Routing attempts timeline' })
    expect(attemptsTimeline.textContent).toMatch(/mesh-primary[\s\S]*mesh-routed \/ skippy[\s\S]*Success/)
  })

  it('shows an explicit successful state in Diagnostics for a completed request', () => {
    renderFixtureDetails(HARNESS_LOG_SCENARIO_IDS.completedMesh, 'diagnostics')

    const diagnostics = screen.getByRole('tabpanel')
    const successState = within(diagnostics).getByText('No errors', { exact: true })
    expect(successState.closest('[role="status"]')).toHaveAttribute('data-diagnostic-state', 'success')
  })

  it('renders failed lifecycle markers, routing attempts, and error artifacts from the retry scenario', () => {
    renderFixtureDetails(HARNESS_LOG_SCENARIO_IDS.failedRetry, 'diagnostics')

    const diagnostics = screen.getByRole('tabpanel')
    for (const kind of ['stream_error', 'attempt_failed', 'audit_error', 'failed']) {
      expect(within(diagnostics).getAllByText(kind).length).toBeGreaterThan(0)
    }
    expect(within(diagnostics).getByText('error_diagnostic')).toBeInTheDocument()
    expect(within(diagnostics).getByText('error_trace')).toBeInTheDocument()
    expect(within(diagnostics).getByText('corrupt')).toBeInTheDocument()
    expect(within(diagnostics).getByText('missing')).toBeInTheDocument()
    expect(within(diagnostics).getByText('2048 B / v4')).toBeInTheDocument()
    expect(within(diagnostics).getByText('Not recorded')).toBeInTheDocument()
    expect(diagnostics).toHaveTextContent('http://peer-b.mesh.invalid:9337')
    expect(diagnostics).toHaveTextContent('https://peer-b.mesh.invalid')
    expect(diagnostics.textContent).toMatch(/retry-primary[\s\S]*retry-secondary/)
  })

  it('sorts the harness retry attempts and renders the no-attempt routing profile', () => {
    const retry = renderFixtureDetails(HARNESS_LOG_SCENARIO_IDS.failedRetry, 'timeline')
    expect(screen.getByRole('list', { name: 'Routing attempts timeline' }).textContent).toMatch(
      /retry-primary[\s\S]*retry-secondary/
    )

    retry.unmount()
    renderFixtureDetails(HARNESS_LOG_SCENARIO_IDS.rejectedAdmission, 'timeline')
    expect(screen.getByText('No proxy attempts were retained for this request.')).toBeInTheDocument()
  })

  it('renders completed stream markers and sparse summary fallbacks from harness fixtures', () => {
    const stream = renderFixtureDetails(HARNESS_LOG_SCENARIO_IDS.completedMesh, 'timeline')
    for (const kind of ['stream_started', 'stream_chunk', 'stream_completed']) {
      expect(screen.getByText(kind)).toBeInTheDocument()
    }

    stream.unmount()
    renderFixtureDetails(HARNESS_LOG_SCENARIO_IDS.completedSparse, 'overview')
    expect(screen.getAllByText('Not recorded').length).toBeGreaterThanOrEqual(5)
    expect(screen.getAllByText('Completed').length).toBeGreaterThan(0)
    expect(screen.getAllByText('durable').length).toBeGreaterThan(0)
  })
})
