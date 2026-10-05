// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  generateArtifacts,
  generateLifecycleEvents,
  generateProxyAttempts,
  HARNESS_LOG_FIXTURES,
  HARNESS_LOG_SCENARIO_IDS
} from '@/features/logs/lib/log-fixtures'

const queries = vi.hoisted(() => ({
  summary: vi.fn(),
  events: vi.fn(),
  artifacts: vi.fn(),
  attempts: vi.fn(),
  artifactContent: vi.fn()
}))

vi.mock('@/features/logs/api/use-log-request-details-query', () => ({
  useLogRequestSummaryQuery: (...args: unknown[]) => queries.summary(...args),
  useLogRequestEventsQuery: (...args: unknown[]) => queries.events(...args),
  useLogRequestArtifactsQuery: (...args: unknown[]) => queries.artifacts(...args),
  useLogRequestAttemptsQuery: (...args: unknown[]) => queries.attempts(...args)
}))

vi.mock('@/features/logs/api/use-log-artifact-content-query', () => ({
  useLogArtifactContentQuery: (...args: unknown[]) => queries.artifactContent(...args)
}))

import { LogRequestDetails } from '@/features/logs/components/LogRequestDetails'

function ready<T>(data: T) {
  return { data, isLoading: false, isError: false }
}

describe('LogRequestDetails Diagnostics query ownership', () => {
  beforeEach(() => {
    for (const query of Object.values(queries)) query.mockReset()
    const requestId = HARNESS_LOG_SCENARIO_IDS.completedMesh
    const summary = HARNESS_LOG_FIXTURES.find((request) => request.requestId.toString() === requestId.toString())
    if (summary === undefined) throw new Error('Missing completed request fixture')
    queries.summary.mockReturnValue(ready(summary))
    queries.events.mockReturnValue(ready({ items: generateLifecycleEvents(requestId.toString()) }))
    queries.artifacts.mockReturnValue(ready({ items: generateArtifacts(requestId.toString()) }))
    queries.attempts.mockReturnValue(ready({ items: generateProxyAttempts(requestId.toString()) }))
  })

  it('uses retained metadata queries without owning artifact body queries', () => {
    // Given
    const requestId = HARNESS_LOG_SCENARIO_IDS.completedMesh

    // When
    render(<LogRequestDetails onBack={vi.fn()} onTabChange={vi.fn()} requestId={requestId} tab="diagnostics" />)

    // Then
    expect(queries.summary).toHaveBeenCalledWith(requestId, undefined)
    expect(queries.events).toHaveBeenCalledWith(requestId, true)
    expect(queries.artifacts).toHaveBeenCalledWith(requestId, true)
    expect(queries.attempts).toHaveBeenCalledWith(requestId, true)
    expect(queries.artifactContent).not.toHaveBeenCalled()
    expect(screen.getByRole('region', { name: 'Request diagnostics' })).toBeInTheDocument()
  })
})
