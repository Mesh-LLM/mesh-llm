import { describe, expect, it } from 'vitest'
import { LogEventId, LogRequestId } from '@/features/logs/api/ids'
import type { LogLifecycleEvent, LogProxyAttempt } from '@/features/logs/api/schemas'
import {
  ledgerSearchFromDetails,
  parseLogRequestDetailsSearch,
  sortLifecycleEvents,
  sortProxyAttempts
} from '@/features/logs/lib/log-request-details'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

function event(eventId: string, occurredAt: string): LogLifecycleEvent {
  return {
    eventId: LogEventId.parse(eventId),
    requestId: REQUEST_ID,
    occurredAt,
    kind: 'stream_chunk',
    model: undefined,
    provider: undefined,
    engine: undefined,
    attemptId: undefined,
    statusCode: undefined,
    durationMs: undefined,
    tokens: undefined
  }
}

function attempt(attemptId: string, occurredAt: string): LogProxyAttempt {
  return {
    attemptId,
    requestId: REQUEST_ID,
    occurredAt,
    target: 'opaque',
    provider: undefined,
    engine: undefined,
    startedAt: undefined,
    completedAt: undefined,
    statusCode: undefined
  }
}

describe('log request details helpers', () => {
  it('preserves ledger context while restoring a recognized inspector tab', () => {
    const search = parseLogRequestDetailsSearch({
      provider: 'reserve-a',
      cursor: 'opaque-page',
      trail: 'older-page',
      tab: 'routing'
    })

    expect(search).toMatchObject({
      provider: 'reserve-a',
      cursor: 'opaque-page',
      trail: ['older-page'],
      tab: 'timeline'
    })
    expect(ledgerSearchFromDetails(search)).toEqual({
      provider: 'reserve-a',
      cursor: 'opaque-page',
      trail: ['older-page']
    })
  })

  it('orders timeline records chronologically without mutating their input lists', () => {
    const events = [
      event('00000000-0000-4000-8000-000000000003', '2026-08-04T12:00:03Z'),
      event('00000000-0000-4000-8000-000000000002', '2026-08-04T12:00:02Z')
    ]
    const attempts = [attempt('second', '2026-08-04T12:00:04Z'), attempt('first', '2026-08-04T12:00:01Z')]

    expect(sortLifecycleEvents(events).map((item) => item.eventId.toString())).toEqual([
      '00000000-0000-4000-8000-000000000002',
      '00000000-0000-4000-8000-000000000003'
    ])
    expect(sortProxyAttempts(attempts).map((item) => item.attemptId)).toEqual(['first', 'second'])
    expect(events[0]?.eventId.toString()).toBe('00000000-0000-4000-8000-000000000003')
  })

  it('orders offset timestamps by instant and preserves equal-instant input order', () => {
    // Given
    const events = [
      event('00000000-0000-4000-8000-000000000002', '2026-08-04T10:00:00-02:00'),
      event('00000000-0000-4000-8000-000000000003', '2026-08-04T11:00:00Z'),
      event('00000000-0000-4000-8000-000000000004', '2026-08-04T13:00:00+01:00')
    ]
    const attempts = [
      attempt('later', '2026-08-04T10:00:00-02:00'),
      attempt('earlier', '2026-08-04T11:00:00Z'),
      attempt('tied', '2026-08-04T13:00:00+01:00')
    ]

    // When / Then
    expect(sortLifecycleEvents(events).map((item) => item.eventId.toString())).toEqual([
      '00000000-0000-4000-8000-000000000003',
      '00000000-0000-4000-8000-000000000002',
      '00000000-0000-4000-8000-000000000004'
    ])
    expect(sortProxyAttempts(attempts).map((item) => item.attemptId)).toEqual(['earlier', 'later', 'tied'])
  })
})
