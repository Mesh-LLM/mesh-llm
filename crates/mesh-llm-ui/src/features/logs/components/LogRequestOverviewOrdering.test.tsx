import '@testing-library/jest-dom/vitest'

import { render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { LogLifecycleEvent, LogProxyAttempt, LogRequest } from '@/features/logs/api/schemas'
import { LogRequestOverview } from '@/features/logs/components/LogRequestOverview'
import {
  HARNESS_LOG_FIXTURES,
  HARNESS_LOG_SCENARIO_IDS,
  generateLifecycleEvents,
  generateProxyAttempts
} from '@/features/logs/lib/log-fixtures'
import { compareLogInstants } from '@/features/logs/lib/log-instant'

vi.mock('@/features/logs/lib/log-instant', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/features/logs/lib/log-instant')>()
  return { ...actual, compareLogInstants: vi.fn(actual.compareLogInstants) }
})

const compareLogInstantsMock = vi.mocked(compareLogInstants)

function requestFixture(requestId: string): LogRequest {
  const request = HARNESS_LOG_FIXTURES.find((candidate) => candidate.requestId.toString() === requestId)
  if (request === undefined) throw new Error(`Missing request fixture ${requestId}`)
  return request
}

function retained<T>(items: readonly T[]) {
  return { items, loading: false, error: false }
}

describe('LogRequestOverview instant ordering', () => {
  it('orders lifecycle events by canonical instant and preserves equal-instant input order', () => {
    const requestId = HARNESS_LOG_SCENARIO_IDS.completedMesh.toString()
    const [equalFirst, laterUtc, latestOffset, equalSecond] = generateLifecycleEvents(requestId)
    const events: readonly LogLifecycleEvent[] = [
      { ...equalFirst, kind: 'admitted', occurredAt: '2026-08-04T08:30:00-04:00', tokens: undefined },
      { ...laterUtc, kind: 'route_selected', occurredAt: '2026-08-04T12:45:00Z', tokens: undefined },
      { ...latestOffset, kind: 'completed', occurredAt: '2026-08-04T09:30:00-04:00', tokens: undefined },
      { ...equalSecond, kind: 'attempt_started', occurredAt: '2026-08-04T12:30:00Z', tokens: undefined }
    ]
    compareLogInstantsMock.mockClear()

    render(
      <LogRequestOverview
        artifacts={retained([])}
        attempts={retained([])}
        events={retained(events)}
        request={requestFixture(requestId)}
      />
    )

    const lifecycle = screen.getByRole('list', { name: 'Lifecycle events' })
    expect(
      within(lifecycle)
        .getAllByRole('listitem')
        .map((item) => item.getAttribute('data-event-kind'))
    ).toEqual(['admitted', 'attempt_started', 'route_selected', 'completed'])
    expect(compareLogInstantsMock).toHaveBeenCalled()
  })

  it('orders attempts by started instant with occurred-at fallback and preserves equal-instant input order', () => {
    const requestId = HARNESS_LOG_SCENARIO_IDS.failedRetry.toString()
    const [equalFirst, fallbackUtc] = generateProxyAttempts(requestId)
    const attempts: readonly LogProxyAttempt[] = [
      {
        ...equalFirst,
        attemptId: 'equal-first',
        occurredAt: '2026-08-04T12:00:00Z',
        startedAt: '2026-08-04T08:30:00-04:00',
        completedAt: undefined
      },
      {
        ...fallbackUtc,
        attemptId: 'utc-12:45-fallback',
        occurredAt: '2026-08-04T12:45:00Z',
        startedAt: undefined,
        completedAt: undefined
      },
      {
        ...equalFirst,
        attemptId: 'offset-13:30',
        occurredAt: '2026-08-04T12:00:00Z',
        startedAt: '2026-08-04T09:30:00-04:00',
        completedAt: undefined
      },
      {
        ...fallbackUtc,
        attemptId: 'equal-second',
        occurredAt: '2026-08-04T12:00:00Z',
        startedAt: '2026-08-04T12:30:00Z',
        completedAt: undefined
      }
    ]
    compareLogInstantsMock.mockClear()

    render(
      <LogRequestOverview
        artifacts={retained([])}
        attempts={retained(attempts)}
        events={retained([])}
        request={requestFixture(requestId)}
      />
    )

    const routing = screen.getByRole('list', { name: 'Routing attempts' })
    expect(
      within(routing)
        .getAllByRole('listitem')
        .map((item) => item.querySelector('dd')?.textContent)
    ).toEqual(['equal-first', 'equal-second', 'utc-12:45-fallback', 'offset-13:30'])
    expect(compareLogInstantsMock).toHaveBeenCalled()
  })

  it('keeps the first token-bearing event when canonical instants are equal', () => {
    const requestId = HARNESS_LOG_SCENARIO_IDS.completedMesh.toString()
    const [equalFirst, equalSecond] = generateLifecycleEvents(requestId)
    const events: readonly LogLifecycleEvent[] = [
      { ...equalFirst, kind: 'stream_chunk', occurredAt: '2026-08-04T08:30:00-04:00', tokens: 111 },
      { ...equalSecond, kind: 'stream_completed', occurredAt: '2026-08-04T12:30:00Z', tokens: 222 }
    ]
    compareLogInstantsMock.mockClear()

    render(
      <LogRequestOverview
        artifacts={retained([])}
        attempts={retained([])}
        events={retained(events)}
        request={requestFixture(requestId)}
      />
    )

    const metrics = screen.getByLabelText('Request metrics')
    expect(metrics).toHaveTextContent('2 stream events / 111 completion tokens')
    expect(metrics).not.toHaveTextContent('222 completion tokens')
    expect(compareLogInstantsMock).toHaveBeenCalledTimes(2)
  })
})
