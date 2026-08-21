import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { LogRequest } from '@/features/logs/api/schemas'
import { LogRequestOverview } from '@/features/logs/components/LogRequestOverview'
import { formatDurationMs } from '@/features/logs/components/LogRequestOverviewDerivations'
import {
  HARNESS_LOG_FIXTURES,
  HARNESS_LOG_SCENARIO_IDS,
  generateArtifacts,
  generateLifecycleEvents,
  generateProxyAttempts
} from '@/features/logs/lib/log-fixtures'

function requestFixture(requestId: string): LogRequest {
  const request = HARNESS_LOG_FIXTURES.find((candidate) => candidate.requestId.toString() === requestId)
  if (request === undefined) throw new Error(`Missing request fixture ${requestId}`)
  return request
}

function retained<T>(items: readonly T[]) {
  return { items, loading: false, error: false }
}

describe('LogRequestOverview', () => {
  it('treats non-finite durations as unrecorded', () => {
    expect(formatDurationMs(Number.NaN)).toBe('Not recorded')
    expect(formatDurationMs(Number.POSITIVE_INFINITY)).toBe('Not recorded')
  })

  it('presents the six reference metrics and retained request evidence without artifact bodies', () => {
    const requestId = HARNESS_LOG_SCENARIO_IDS.completedMesh.toString()
    const request = requestFixture(requestId)
    const events = generateLifecycleEvents(requestId)
    const artifacts = generateArtifacts(requestId)
    const attempts = generateProxyAttempts(requestId)

    render(
      <LogRequestOverview
        artifacts={retained(artifacts)}
        attempts={retained(attempts)}
        events={retained(events)}
        request={request}
      />
    )

    const metrics = screen.getByLabelText('Request metrics')
    for (const label of [
      'Status',
      'Duration',
      'Provider',
      'Model',
      'Attempts / retries',
      'Stream / completion tokens'
    ]) {
      expect(within(metrics).getByText(label)).toBeInTheDocument()
    }
    expect(metrics).toHaveTextContent('Completed')
    expect(within(metrics).getByTestId('request-outcome')).toHaveTextContent('Completed')
    expect(within(metrics).getByTestId('request-http-status')).toHaveTextContent('HTTP 200')
    expect(metrics).toHaveTextContent('57.0 s')
    expect(metrics).toHaveTextContent('openai_frontend')
    expect(metrics).toHaveTextContent('Qwen3-30B-A3B-Q4_K_M.gguf')
    expect(metrics).toHaveTextContent('1 attempt / 0 retries')
    expect(metrics).toHaveTextContent('3 stream events / 612 completion tokens')
    expect(metrics).not.toHaveTextContent('1,352 tokens')

    const streamLabel = within(metrics).getByText('Stream / completion tokens')
    expect(streamLabel).toHaveClass('break-words')
    expect(streamLabel.closest('dt')).toHaveClass('items-start')
    expect(within(metrics).getByText('Status').closest('dt')).toHaveClass('items-start')
    expect(within(metrics).getByTestId('metric-icon-Status')).toHaveAttribute('data-metric-icon-tone', 'good')
    expect(within(metrics).getByTestId('metric-icon-Status')).not.toHaveClass('border')
    expect(within(metrics).getByTestId('metric-icon-Model')).toHaveAttribute('data-metric-icon-tone', 'contrast')
    expect(within(metrics).getByTestId('metric-icon-Stream / completion tokens')).toHaveAttribute(
      'data-metric-icon-tone',
      'accent'
    )

    const metadata = screen.getByRole('region', { name: 'Request metadata' })
    for (const value of [requestId, 'chat_completions', 'chat_completion_stream', '200', 'durable']) {
      expect(metadata).toHaveTextContent(value)
    }
    expect(within(metadata).getByText('Record source', { exact: true }).parentElement).toHaveClass('sm:col-span-2')
    expect(within(metadata).getByText('Record source', { exact: true }).parentElement).not.toHaveClass('xl:col-span-1')

    const artifactSummary = screen.getByRole('region', { name: 'Artifact retention' })
    expect(artifactSummary).toHaveTextContent('4')
    expect(artifactSummary).toHaveTextContent('3 available · 1 unavailable')
    expect(artifactSummary).toHaveTextContent('3 of 4')
    expect(artifactSummary).toHaveTextContent('1 of 4')
    expect(artifactSummary).toHaveTextContent('1,992 B')
    expect(artifactSummary).toHaveTextContent('v1 · v2 · v3')
    expect(artifactSummary).toHaveTextContent('durable')
    expect(artifactSummary).not.toHaveTextContent(/expiry|ttl|body loaded/i)
    expect(within(artifactSummary).getByText('Request source', { exact: true }).parentElement).toHaveClass(
      'sm:col-span-2'
    )
    expect(within(artifactSummary).getByText('3 available · 1 unavailable', { exact: true })).toHaveClass('break-words')
    expect(within(artifactSummary).getByText('3 available · 1 unavailable', { exact: true })).not.toHaveClass(
      'break-all'
    )

    const lifecycle = screen.getByRole('list', { name: 'Lifecycle events' })
    expect(lifecycle.textContent).toMatch(/admitted[\s\S]*stream_started[\s\S]*stream_completed[\s\S]*completed/)
    expect(lifecycle.querySelectorAll('time')).toHaveLength(events.length)

    const routing = screen.getByRole('list', { name: 'Routing attempts' })
    for (const value of ['mesh-primary', 'https://peer-a.mesh.invalid', 'mesh-routed', 'skippy', '200', '48.0 s']) {
      expect(routing).toHaveTextContent(value)
    }
    expect(routing.querySelectorAll('time')).toHaveLength(2)
  })

  it('derives retries as attempt count minus one and uses precise sparse states', () => {
    const retryId = HARNESS_LOG_SCENARIO_IDS.failedRetry.toString()
    const retry = render(
      <LogRequestOverview
        artifacts={retained(generateArtifacts(retryId))}
        attempts={retained(generateProxyAttempts(retryId))}
        events={retained(generateLifecycleEvents(retryId))}
        request={requestFixture(retryId)}
      />
    )
    expect(screen.getByLabelText('Request metrics')).toHaveTextContent('2 attempts / 1 retry')

    retry.unmount()
    const sparseId = HARNESS_LOG_SCENARIO_IDS.completedSparse.toString()
    render(
      <LogRequestOverview
        artifacts={retained([])}
        attempts={retained([])}
        events={retained([])}
        request={requestFixture(sparseId)}
      />
    )

    expect(screen.getByText('No artifact metadata was retained for this request.')).toBeInTheDocument()
    expect(screen.getByText('No lifecycle events were retained for this request.')).toBeInTheDocument()
    expect(screen.getByText('No routing attempts were retained for this request.')).toBeInTheDocument()
    expect(screen.getByLabelText('Request metrics')).toHaveTextContent('0 attempts / 0 retries')
    expect(screen.getByLabelText('Request metrics')).toHaveTextContent('Not recorded')
  })

  it('keeps an error HTTP status secondary to the failed outcome', () => {
    const requestId = HARNESS_LOG_SCENARIO_IDS.failedRetry.toString()
    render(
      <LogRequestOverview
        artifacts={retained([])}
        attempts={retained([])}
        events={retained([])}
        request={requestFixture(requestId)}
      />
    )

    const metrics = screen.getByLabelText('Request metrics')
    expect(within(metrics).getByTestId('request-outcome')).toHaveTextContent('Failed')
    expect(within(metrics).getByTestId('request-http-status')).toHaveTextContent('HTTP 502')
  })
})
