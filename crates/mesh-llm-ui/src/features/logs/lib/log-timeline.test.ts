import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogEventKind, LogProxyAttempt } from '@/features/logs/api/schemas'
import {
  attemptDurationMs,
  attemptOutcomeLabel,
  attemptRouteLabel,
  attemptStatus,
  attemptTone,
  elapsedMilliseconds,
  eventTone,
  formatLogTimestampMs
} from './log-timeline'

const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

function attempt(overrides: Partial<LogProxyAttempt> = {}): LogProxyAttempt {
  return {
    attemptId: 'att-1',
    requestId: REQUEST_ID,
    occurredAt: '2026-08-04T12:00:00Z',
    target: 'https://peer-a.mesh.invalid',
    provider: undefined,
    engine: undefined,
    startedAt: undefined,
    completedAt: undefined,
    statusCode: undefined,
    ...overrides
  }
}

describe('eventTone', () => {
  it('maps every lifecycle kind to its timeline tone', () => {
    const expected: Record<LogEventKind, string> = {
      admitted: 'muted',
      route_selected: 'accent',
      attempt_started: 'accent',
      stream_started: 'accent',
      stream_chunk: 'accent',
      attempt_completed: 'good',
      stream_completed: 'good',
      completed: 'good',
      cancelled: 'warn',
      attempt_failed: 'bad',
      stream_error: 'bad',
      audit_error: 'bad',
      failed: 'bad',
      rejected: 'bad',
      dropped: 'bad'
    }
    for (const [kind, tone] of Object.entries(expected)) {
      expect(eventTone(kind as LogEventKind)).toBe(tone)
    }
  })
})

describe('attemptTone', () => {
  it('is good for recorded status codes below 400 and bad for 400 and above', () => {
    expect(attemptTone(attempt({ statusCode: 200 }))).toBe('good')
    expect(attemptTone(attempt({ statusCode: 399 }))).toBe('good')
    expect(attemptTone(attempt({ statusCode: 400 }))).toBe('bad')
    expect(attemptTone(attempt({ statusCode: 503 }))).toBe('bad')
  })

  it('is accent while a started attempt has not completed and muted otherwise', () => {
    expect(attemptTone(attempt({ startedAt: '2026-08-04T12:00:00Z' }))).toBe('accent')
    expect(attemptTone(attempt())).toBe('muted')
    expect(attemptTone(attempt({ startedAt: '2026-08-04T12:00:00Z', completedAt: '2026-08-04T12:00:01Z' }))).toBe(
      'muted'
    )
  })
})

describe('attemptStatus', () => {
  it('prefers the HTTP status, then in-progress, then the unrecorded fallback', () => {
    expect(attemptStatus(attempt({ statusCode: 502 }))).toBe('HTTP 502')
    expect(attemptStatus(attempt({ startedAt: '2026-08-04T12:00:00Z' }))).toBe('In progress')
    expect(attemptStatus(attempt())).toBe('Status not recorded')
  })
})

describe('attemptOutcomeLabel', () => {
  it('labels recorded statuses Success below 400 and Failed at or above 400', () => {
    expect(attemptOutcomeLabel(attempt({ statusCode: 200 }))).toBe('Success')
    expect(attemptOutcomeLabel(attempt({ statusCode: 404 }))).toBe('Failed')
  })

  it('labels started unfinished attempts In progress and everything else not recorded', () => {
    expect(attemptOutcomeLabel(attempt({ startedAt: '2026-08-04T12:00:00Z' }))).toBe('In progress')
    expect(attemptOutcomeLabel(attempt())).toBe('Status not recorded')
    expect(
      attemptOutcomeLabel(attempt({ startedAt: '2026-08-04T12:00:00Z', completedAt: '2026-08-04T12:00:01Z' }))
    ).toBe('Status not recorded')
  })
})

describe('elapsedMilliseconds', () => {
  it('returns the millisecond delta between two instants', () => {
    expect(elapsedMilliseconds('2026-08-04T12:00:00.000Z', '2026-08-04T12:00:00.084Z')).toBe(84)
  })

  it('returns zero for equal instants and undefined when the delta is negative', () => {
    expect(elapsedMilliseconds('2026-08-04T12:00:00Z', '2026-08-04T12:00:00Z')).toBe(0)
    expect(elapsedMilliseconds('2026-08-04T12:00:01Z', '2026-08-04T12:00:00Z')).toBeUndefined()
  })

  it('returns undefined when either bound is missing', () => {
    expect(elapsedMilliseconds(undefined, '2026-08-04T12:00:00Z')).toBeUndefined()
    expect(elapsedMilliseconds('2026-08-04T12:00:00Z', undefined)).toBeUndefined()
  })
})

describe('attemptDurationMs', () => {
  it('measures completedAt minus startedAt when both parse in order', () => {
    expect(
      attemptDurationMs(attempt({ startedAt: '2026-08-04T12:00:00.000Z', completedAt: '2026-08-04T12:00:00.843Z' }))
    ).toBe(843)
  })

  it('is undefined when a bound is missing or the range is inverted', () => {
    expect(attemptDurationMs(attempt({ startedAt: '2026-08-04T12:00:00Z' }))).toBeUndefined()
    expect(attemptDurationMs(attempt({ completedAt: '2026-08-04T12:00:00Z' }))).toBeUndefined()
    expect(
      attemptDurationMs(attempt({ startedAt: '2026-08-04T12:00:01Z', completedAt: '2026-08-04T12:00:00Z' }))
    ).toBeUndefined()
  })
})

describe('attemptRouteLabel', () => {
  it('joins provider and engine with a slash separator', () => {
    expect(attemptRouteLabel(attempt({ provider: 'mesh-routed', engine: 'skippy' }))).toBe('mesh-routed / skippy')
  })

  it('uses whichever of provider or engine is defined', () => {
    expect(attemptRouteLabel(attempt({ provider: 'mesh-routed' }))).toBe('mesh-routed')
    expect(attemptRouteLabel(attempt({ engine: 'skippy' }))).toBe('skippy')
  })

  it('falls back to the attempt target when neither provider nor engine is recorded', () => {
    expect(attemptRouteLabel(attempt({ target: 'http://peer-b.mesh.invalid:9337' }))).toBe(
      'http://peer-b.mesh.invalid:9337'
    )
  })
})

describe('formatLogTimestampMs', () => {
  it('renders a time with three fractional second digits', () => {
    expect(formatLogTimestampMs('2026-08-04T12:00:01.200Z')).toMatch(/\d{1,2}:\d{2}:\d{2}\.200/)
  })

  it('pads sub-second precision to exactly three digits', () => {
    expect(formatLogTimestampMs('2026-08-04T12:00:01Z')).toMatch(/\.\d{3}/)
  })
})
