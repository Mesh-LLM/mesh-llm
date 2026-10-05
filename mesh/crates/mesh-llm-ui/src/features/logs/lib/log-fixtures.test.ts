import { describe, expect, it } from 'vitest'
import {
  parseLogArtifact,
  parseLogLifecycleEvent,
  parseLogProxyAttempt,
  parseLogRequest,
  type LogArtifact,
  type LogEventKind,
  type LogLifecycleEvent,
  type LogOutcome,
  type LogProxyAttempt,
  type LogRequest
} from '@/features/logs/api/schemas'
import { artifactMatchesTab } from '@/features/logs/lib/log-request-details'
import {
  HARNESS_LOG_FIXTURES,
  HARNESS_REFERENCE_TIME,
  generateArtifacts,
  generateLifecycleEvents,
  generateProxyAttempts
} from '@/features/logs/lib/log-fixtures'

const EXPECTED_OUTCOMES = [
  'active',
  'completed',
  'failed',
  'rejected',
  'cancelled',
  'dropped'
] satisfies readonly LogOutcome[]

const EXPECTED_EVENT_KINDS = [
  'admitted',
  'route_selected',
  'attempt_started',
  'attempt_completed',
  'attempt_failed',
  'backend_stream_first_item',
  'stream_started',
  'stream_chunk',
  'stream_completed',
  'usage_recorded',
  'stream_error',
  'audit_error',
  'completed',
  'failed',
  'rejected',
  'cancelled',
  'dropped'
] satisfies readonly LogEventKind[]

const OPENAI_OPERATION_LABELS = [
  'models',
  'chat_completion',
  'chat_completion_stream',
  'completion',
  'completion_stream',
  'responses',
  'responses_stream'
] as const

function requestWire(request: LogRequest) {
  return {
    ...request,
    requestId: request.requestId.toString(),
    terminalAt: request.terminalAt ?? null,
    route: request.route ?? null,
    model: request.model ?? null,
    provider: request.provider ?? null,
    engine: request.engine ?? null,
    statusCode: request.statusCode ?? null
  }
}

function eventWire(event: LogLifecycleEvent) {
  return {
    ...event,
    eventId: event.eventId.toString(),
    requestId: event.requestId.toString(),
    model: event.model ?? null,
    provider: event.provider ?? null,
    engine: event.engine ?? null,
    attemptId: event.attemptId ?? null,
    statusCode: event.statusCode ?? null,
    durationMs: event.durationMs ?? null,
    tokens: event.tokens ?? null,
    promptTokens: event.promptTokens ?? null,
    cachedPromptTokens: event.cachedPromptTokens ?? null,
    completionTokens: event.completionTokens ?? null,
    totalTokens: event.totalTokens ?? null
  }
}

function artifactWire(artifact: LogArtifact) {
  return {
    ...artifact,
    artifactId: artifact.artifactId.toString(),
    requestId: artifact.requestId.toString(),
    mediaKind: artifact.mediaKind ?? null,
    checksum: artifact.checksum ?? null,
    contentBase64: artifact.contentBase64 ?? null
  }
}

function attemptWire(attempt: LogProxyAttempt) {
  return {
    ...attempt,
    requestId: attempt.requestId.toString(),
    provider: attempt.provider ?? null,
    engine: attempt.engine ?? null,
    startedAt: attempt.startedAt ?? null,
    completedAt: attempt.completedAt ?? null,
    statusCode: attempt.statusCode ?? null
  }
}

function expectUnique(values: readonly string[]) {
  expect(new Set(values).size).toBe(values.length)
}

describe('logs harness fixtures', () => {
  it('anchors default fixture timestamps to a rolling reference instant inside the ledger time presets', () => {
    const referenceTimeMs = Date.parse(HARNESS_REFERENCE_TIME)
    const newestRequestTimeMs = Math.max(...HARNESS_LOG_FIXTURES.map((request) => Date.parse(request.createdAt)))
    const oldestRequestTimeMs = Math.min(...HARNESS_LOG_FIXTURES.map((request) => Date.parse(request.createdAt)))

    expect(Number.isFinite(referenceTimeMs)).toBe(true)
    expect(Date.now() - referenceTimeMs).toBeLessThan(60 * 60_000)
    expect(referenceTimeMs - newestRequestTimeMs).toBe(60_000)
    expect(referenceTimeMs - oldestRequestTimeMs).toBeLessThan(12 * 60 * 60_000)
    expect(Date.now() - oldestRequestTimeMs).toBeLessThan(24 * 60 * 60_000)
  })

  it('covers every ledger outcome, source, optional metadata state, and populated-table volume', () => {
    expect(HARNESS_LOG_FIXTURES.length).toBeGreaterThan(20)
    expect(new Set(HARNESS_LOG_FIXTURES.map((request) => request.outcome))).toEqual(new Set(EXPECTED_OUTCOMES))
    expect(new Set(HARNESS_LOG_FIXTURES.map((request) => request.source))).toEqual(new Set(['active', 'durable']))
    expectUnique(HARNESS_LOG_FIXTURES.map((request) => request.requestId.toString()))

    for (const key of [
      'terminalAt',
      'route',
      'model',
      'provider',
      'engine',
      'statusCode',
      'callerEndpointId',
      'callerAddr',
      'callerPathType'
    ] as const) {
      expect(HARNESS_LOG_FIXTURES.some((request) => request[key] === undefined)).toBe(true)
      expect(HARNESS_LOG_FIXTURES.some((request) => request[key] !== undefined)).toBe(true)
    }

    for (const request of HARNESS_LOG_FIXTURES) {
      expect(parseLogRequest(requestWire(request))).toEqual(request)
    }

    const relayRequest = HARNESS_LOG_FIXTURES.find((request) => request.callerPathType === 'relay')
    expect(relayRequest?.callerEndpointId).toBeDefined()
    expect(relayRequest?.callerAddr).toBeUndefined()
  })

  it('uses normalized OpenAI and management operations plus raw mesh defaults', () => {
    const engines = new Set(HARNESS_LOG_FIXTURES.map((request) => request.engine))
    const routes = new Set(HARNESS_LOG_FIXTURES.map((request) => request.route))

    for (const operation of OPENAI_OPERATION_LABELS) expect(engines.has(operation)).toBe(true)
    expect(routes.has('chat_completions')).toBe(true)
    expect(routes.has('management_get_status')).toBe(true)
    expect(routes.has('management_post')).toBe(true)
    expect(
      HARNESS_LOG_FIXTURES.some((request) => request.provider === 'mesh' && request.engine === 'raw_ingress')
    ).toBe(true)
    expect(HARNESS_LOG_FIXTURES.some((request) => request.route?.includes('.'))).toBe(false)
  })

  it('covers every lifecycle kind with unique schema-valid records linked to their request', () => {
    const events = HARNESS_LOG_FIXTURES.flatMap((request) => generateLifecycleEvents(request.requestId.toString()))

    expect(new Set(events.map((event) => event.kind))).toEqual(new Set(EXPECTED_EVENT_KINDS))
    expectUnique(events.map((event) => event.eventId.toString()))
    expect(events.some((event) => event.attemptId === undefined)).toBe(true)
    expect(events.some((event) => event.attemptId !== undefined)).toBe(true)
    expect(events.some((event) => event.statusCode === undefined && event.durationMs === undefined)).toBe(true)
    expect(events.some((event) => event.statusCode !== undefined && event.durationMs !== undefined)).toBe(true)

    for (const event of events) {
      expect(HARNESS_LOG_FIXTURES.some((request) => request.requestId.toString() === event.requestId.toString())).toBe(
        true
      )
      expect(parseLogLifecycleEvent(eventWire(event))).toEqual(event)
    }
  })

  it('covers every artifact tab and content state with deterministic schema-valid metadata', () => {
    const requestId = HARNESS_LOG_FIXTURES[0]?.requestId.toString()
    expect(requestId).toBeDefined()
    if (requestId === undefined) return

    expect(generateArtifacts(requestId)).toEqual(generateArtifacts(requestId))

    const artifacts = HARNESS_LOG_FIXTURES.flatMap((request) => generateArtifacts(request.requestId.toString()))
    expect(new Set(artifacts.map((artifact) => artifact.contentState))).toEqual(
      new Set(['available', 'unavailable', 'missing', 'corrupt'])
    )
    expectUnique(artifacts.map((artifact) => artifact.artifactId.toString()))
    expect(artifacts.some((artifact) => artifactMatchesTab(artifact.kind, 'request'))).toBe(true)
    expect(artifacts.some((artifact) => artifactMatchesTab(artifact.kind, 'response'))).toBe(true)
    expect(artifacts.some((artifact) => artifactMatchesTab(artifact.kind, 'errors'))).toBe(true)
    expect(artifacts.some((artifact) => artifact.checksum === undefined)).toBe(true)
    expect(artifacts.some((artifact) => artifact.truncated)).toBe(true)
    expect(artifacts.some((artifact) => !artifact.truncated)).toBe(true)

    for (const artifact of artifacts) {
      expect(artifact.version).toBeGreaterThanOrEqual(1)
      if (artifact.checksum !== undefined) expect(artifact.checksum).toMatch(/^sha256:[0-9a-f]{64}$/)
      if (artifact.contentState === 'available') expect(artifact.redacted).toBe(true)
      expect(parseLogArtifact(artifactWire(artifact))).toEqual(artifact)
    }
  })

  it('covers zero, one, and retry routing profiles using schema-valid safe targets', () => {
    const attemptsByRequest = HARNESS_LOG_FIXTURES.map((request) => generateProxyAttempts(request.requestId.toString()))
    const attempts = attemptsByRequest.flat()

    expect(attemptsByRequest.some((requestAttempts) => requestAttempts.length === 0)).toBe(true)
    expect(attemptsByRequest.some((requestAttempts) => requestAttempts.length === 1)).toBe(true)
    expect(attemptsByRequest.some((requestAttempts) => requestAttempts.length > 1)).toBe(true)
    expect(attempts.some((attempt) => attempt.provider === undefined && attempt.engine === undefined)).toBe(true)
    expect(attempts.some((attempt) => attempt.completedAt === undefined && attempt.statusCode === undefined)).toBe(true)

    for (const attempt of attempts) {
      expect(parseLogProxyAttempt(attemptWire(attempt))).toEqual(attempt)
    }
  })

  it('returns empty detail collections for an unknown request', () => {
    const unknownRequestId = '00000000-0000-4000-8000-ffffffffffff'

    expect(generateLifecycleEvents(unknownRequestId)).toEqual([])
    expect(generateArtifacts(unknownRequestId)).toEqual([])
    expect(generateProxyAttempts(unknownRequestId)).toEqual([])
  })
})
