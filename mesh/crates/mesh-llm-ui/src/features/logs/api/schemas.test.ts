import { describe, expect, it } from 'vitest'
import { LogAuditCursor, LogReplayCursor } from './ids'
import {
  LogsDtoError,
  parseLogCleanupReceipt,
  parseLogDeleteReceipt,
  parseLogExport,
  parseLogArtifact,
  parseLogAuditPage,
  parseLogLifecycleEvent,
  parseLogProxyAttempt,
  parseLogRequest,
  parseLogRequestPage
} from './schemas'
import { parseLogsSseFrame } from './sse'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'
const EVENT_ID = '00000000-0000-4000-8000-000000000002'
const ARTIFACT_ID = '00000000-0000-4000-8000-000000000003'
const AUDIT_ID = '00000000-0000-4000-8000-000000000004'
const TIMESTAMP = '2026-08-04T12:00:00Z'

function requestDto() {
  return {
    requestId: REQUEST_ID,
    outcome: 'completed',
    createdAt: TIMESTAMP,
    terminalAt: TIMESTAMP,
    route: 'chat',
    model: 'model-a',
    provider: 'local',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable'
  }
}

function artifactDto(
  contentState: string,
  redacted: boolean,
  contentBase64: string | null,
  unavailableReason: string | null = null
) {
  return {
    artifactId: ARTIFACT_ID,
    requestId: REQUEST_ID,
    occurredAt: TIMESTAMP,
    kind: 'request',
    mediaKind: 'text/plain',
    checksum: 'sha256:abc',
    bytes: 5,
    version: 1,
    redacted,
    truncated: false,
    contentState,
    unavailableReason,
    contentBase64
  }
}

function cleanupReceiptDto() {
  return {
    operationId: EVENT_ID,
    auditId: AUDIT_ID,
    cutoffBefore: TIMESTAMP,
    requestLimit: 1,
    scope: {
      source: 'durable',
      cutoffBefore: TIMESTAMP,
      requestLimit: 1,
      from: '2026-08-01T00:00:00Z',
      to: TIMESTAMP,
      route: 'reserve',
      excludeRoute: 'models',
      model: 'Qwen/Qwen3',
      provider: 'reserve-a',
      engine: 'skippy',
      outcome: 'completed'
    },
    state: 'previewed',
    hasMore: false,
    selectionFingerprint: 'safe',
    planned: { requests: 1, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 1 },
    executed: { requests: 0, events: 0, artifacts: 0, proxyRecords: 0, databaseRows: 0 },
    artifactDeletion: { removed: 0, failed: 0 }
  }
}

describe('logs DTO boundary parsers', () => {
  it('parses valid request, event, page, proxy, and every artifact state', () => {
    const request = parseLogRequest(requestDto())
    const page = parseLogRequestPage({ items: [requestDto()], nextCursor: 'opaque-next-page' })
    const event = parseLogLifecycleEvent({
      eventId: EVENT_ID,
      requestId: REQUEST_ID,
      occurredAt: TIMESTAMP,
      kind: 'completed',
      model: 'model-a',
      provider: null,
      engine: null,
      attemptId: null,
      statusCode: 200,
      durationMs: 12,
      tokens: 5,
      promptTokens: 11,
      completionTokens: 5,
      totalTokens: 16
    })
    const proxy = parseLogProxyAttempt({
      attemptId: EVENT_ID,
      requestId: REQUEST_ID,
      occurredAt: TIMESTAMP,
      target: 'https://example.test:9443',
      provider: null,
      engine: null,
      startedAt: null,
      completedAt: null,
      statusCode: null
    })

    expect(request.requestId.toString()).toBe(REQUEST_ID)
    expect(page.nextCursor?.toString()).toBe('opaque-next-page')
    expect(event.eventId.toString()).toBe(EVENT_ID)
    expect(event).toMatchObject({ tokens: 5, promptTokens: 11, completionTokens: 5, totalTokens: 16 })
    expect(proxy.target).toBe('https://example.test:9443')
    expect(parseLogArtifact(artifactDto('available', true, 'SGVsbG8=')).contentState).toBe('available')
    for (const unavailableReason of [
      'streaming_response_not_assembled',
      'response_body_not_bounded',
      'capture_content_limit_exceeded',
      'capture_memory_budget_exceeded',
      'artifact_capture_disabled',
      'artifact_capture_failed'
    ] as const) {
      expect(parseLogArtifact(artifactDto('unavailable', false, null, unavailableReason))).toMatchObject({
        contentState: 'unavailable',
        unavailableReason
      })
    }
    expect(parseLogArtifact(artifactDto('missing', false, null)).contentState).toBe('missing')
    expect(parseLogArtifact(artifactDto('corrupt', false, null)).contentState).toBe('corrupt')
  })

  it('parses bounded token usage while accepting older event DTOs without it', () => {
    const usage = parseLogLifecycleEvent({
      eventId: EVENT_ID,
      requestId: REQUEST_ID,
      occurredAt: TIMESTAMP,
      kind: 'usage_recorded',
      model: null,
      provider: null,
      engine: null,
      attemptId: null,
      statusCode: null,
      durationMs: null,
      tokens: null,
      promptTokens: 21,
      cachedPromptTokens: 13,
      completionTokens: 8,
      totalTokens: 29
    })
    const legacy = parseLogLifecycleEvent({
      eventId: EVENT_ID,
      requestId: REQUEST_ID,
      occurredAt: TIMESTAMP,
      kind: 'completed',
      model: null,
      provider: null,
      engine: null,
      attemptId: null,
      statusCode: 200,
      durationMs: 12,
      tokens: null
    })

    expect(usage).toMatchObject({
      promptTokens: 21,
      cachedPromptTokens: 13,
      completionTokens: 8,
      totalTokens: 29
    })
    expect(legacy.totalTokens).toBeUndefined()
  })

  it('rejects unknown event versions, malformed cursors, unsafe proxy URLs, and inconsistent artifacts', () => {
    expect(() => LogReplayCursor.parse('v2:1.2.3')).toThrow()
    expect(() => LogReplayCursor.parse('v1:1.not-a-number.3')).toThrow()
    expect(() =>
      parseLogLifecycleEvent({
        eventId: EVENT_ID,
        requestId: REQUEST_ID,
        occurredAt: TIMESTAMP,
        kind: 'future_event',
        model: null,
        provider: null,
        engine: null,
        attemptId: null,
        statusCode: null,
        durationMs: null,
        tokens: null
      })
    ).toThrow(LogsDtoError)
    expect(() =>
      parseLogProxyAttempt({
        attemptId: EVENT_ID,
        requestId: REQUEST_ID,
        occurredAt: TIMESTAMP,
        target: 'https://user:secret@example.test/private?token=secret',
        provider: null,
        engine: null,
        startedAt: null,
        completedAt: null,
        statusCode: null
      })
    ).toThrow(LogsDtoError)
    expect(() =>
      parseLogProxyAttempt({
        attemptId: EVENT_ID,
        requestId: REQUEST_ID,
        occurredAt: TIMESTAMP,
        target: 'https://example.test:0',
        provider: null,
        engine: null,
        startedAt: null,
        completedAt: null,
        statusCode: null
      })
    ).toThrow(LogsDtoError)
    expect(() => parseLogArtifact(artifactDto('available', false, 'SGVsbG8='))).toThrow(LogsDtoError)
    expect(() => parseLogArtifact(artifactDto('missing', false, 'SGVsbG8='))).toThrow(LogsDtoError)
    expect(() => parseLogArtifact(artifactDto('unavailable', false, null, 'secret_internal_failure'))).toThrow(
      LogsDtoError
    )
    expect(() => parseLogArtifact(artifactDto('missing', false, null, 'streaming_response_not_assembled'))).toThrow(
      LogsDtoError
    )
  })
})

describe('logs operation DTO parser', () => {
  it('requires a strict durable cleanup scope and valid audit ID on maintenance receipts', () => {
    const receipt = cleanupReceiptDto()

    const parsed = parseLogCleanupReceipt(receipt)
    expect(parsed.auditId.toString()).toBe(AUDIT_ID)
    expect(parsed.scope).toMatchObject({
      source: 'durable',
      excludeRoute: 'models',
      model: 'Qwen/Qwen3',
      outcome: 'completed'
    })
    const { auditId: _auditId, ...missingAuditId } = receipt
    expect(() => parseLogCleanupReceipt(missingAuditId)).toThrow(LogsDtoError)
    expect(() => parseLogCleanupReceipt({ ...receipt, auditId: 'audit:/private/secret' })).toThrow(LogsDtoError)
    const { scope: _scope, ...missingScope } = receipt
    expect(() => parseLogCleanupReceipt(missingScope)).toThrow(LogsDtoError)
    expect(() => parseLogCleanupReceipt({ ...receipt, scope: { ...receipt.scope, source: 'active' } })).toThrow(
      LogsDtoError
    )
    expect(() => parseLogCleanupReceipt({ ...receipt, scope: { ...receipt.scope, outcome: 'active' } })).toThrow(
      LogsDtoError
    )
    expect(() =>
      parseLogCleanupReceipt({ ...receipt, scope: { ...receipt.scope, model: '/private/model?token=secret' } })
    ).toThrow(LogsDtoError)
    expect(() =>
      parseLogCleanupReceipt({ ...receipt, scope: { ...receipt.scope, excludeRoute: '/private/models' } })
    ).toThrow(LogsDtoError)
    expect(() => parseLogCleanupReceipt({ ...receipt, scope: { ...receipt.scope, requestLimit: 2 } })).toThrow(
      LogsDtoError
    )
    expect(() => parseLogCleanupReceipt({ ...receipt, scope: { ...receipt.scope, cursor: 'opaque-page' } })).toThrow(
      LogsDtoError
    )
    expect(() =>
      parseLogDeleteReceipt({
        operationId: EVENT_ID,
        requestId: REQUEST_ID,
        state: 'completed',
        selectionFingerprint: 'safe',
        planned: receipt.planned,
        executed: receipt.executed,
        artifactDeletion: receipt.artifactDeletion
      })
    ).toThrow(LogsDtoError)
  })

  it('distinguishes completed, pending, and partial delete receipts', () => {
    const base = {
      operationId: EVENT_ID,
      requestId: REQUEST_ID,
      selectionFingerprint: 'safe',
      planned: cleanupReceiptDto().planned,
      executed: cleanupReceiptDto().executed,
      artifactDeletion: cleanupReceiptDto().artifactDeletion
    }

    const completed = parseLogDeleteReceipt({ ...base, auditId: AUDIT_ID, state: 'completed' })
    expect(completed.state).toBe('completed')
    expect(completed.auditId?.toString()).toBe(AUDIT_ID)
    expect(parseLogDeleteReceipt({ ...base, state: 'pending' })).toMatchObject({
      state: 'pending',
      auditId: undefined
    })
    expect(parseLogDeleteReceipt({ ...base, auditId: null, state: 'pending' })).toMatchObject({
      state: 'pending',
      auditId: undefined
    })
    const partial = parseLogDeleteReceipt({
      ...base,
      auditId: AUDIT_ID,
      state: 'partial',
      artifactDeletion: { removed: 0, failed: 1, failureClass: 'io' }
    })
    expect(partial).toMatchObject({ state: 'partial', artifactDeletion: { failed: 1 } })
    expect(partial.auditId?.toString()).toBe(AUDIT_ID)
    expect(parseLogDeleteReceipt({ ...base, state: 'partial' })).toMatchObject({
      state: 'partial',
      auditId: undefined
    })
    expect(() => parseLogDeleteReceipt({ ...base, auditId: AUDIT_ID, state: 'pending' })).toThrow(LogsDtoError)
    expect(() => parseLogDeleteReceipt({ ...base, state: 'completed' })).toThrow(LogsDtoError)
  })

  it('parses bounded metadata-only export results without treating artifact payloads as UI content', () => {
    const exportResult = parseLogExport({
      items: [
        {
          summary: requestDto(),
          events: [],
          artifacts: [artifactDto('available', true, null)],
          childIncomplete: false
        }
      ],
      nextCursor: null,
      truncated: true,
      retryRequired: false,
      artifactContentIncluded: false
    })

    expect(exportResult.truncated).toBe(true)
    expect(exportResult.artifactContentIncluded).toBe(false)
    expect(exportResult.items[0]?.artifacts[0]?.contentBase64).toBeUndefined()
  })

  it('rejects hostile operation DTOs before they reach controls', () => {
    expect(() =>
      parseLogCleanupReceipt({
        ...cleanupReceiptDto(),
        artifactDeletion: { removed: 0, failed: 0, failureClass: 'path:/private/secret' }
      })
    ).toThrow(LogsDtoError)
    expect(() =>
      parseLogExport({
        items: [],
        nextCursor: null,
        truncated: false,
        retryRequired: false,
        artifactContentIncluded: 'yes'
      })
    ).toThrow(LogsDtoError)
  })
})

describe('dedicated logs SSE frame parser', () => {
  it('accepts real maintenance audit entries from the logs API in pages and SSE', () => {
    const entry = {
      entryId: 'audit-maintenance-0001',
      occurredAt: TIMESTAMP,
      source: 'logs_api',
      code: 'logging_cleanup_completed',
      severity: 'info',
      sequence: 42
    }

    expect(parseLogAuditPage({ items: [entry], nextCursor: null }).items).toEqual([entry])
    expect(parseLogsSseFrame({ event: 'audit_entry', lastEventId: 'a1:42', data: JSON.stringify(entry) })).toEqual({
      type: 'audit_entry',
      cursor: LogAuditCursor.parse('a1:42'),
      entry
    })
    expect(() =>
      parseLogsSseFrame({ event: 'audit_entry', lastEventId: 'a1:41', data: JSON.stringify(entry) })
    ).toThrow(LogsDtoError)
  })

  it('accepts the bounded typed audit context while preserving old-row compatibility', () => {
    const oldEntry = {
      entryId: 'audit-old',
      occurredAt: TIMESTAMP,
      source: 'runtime',
      code: 'runtime_ready',
      sequence: 1
    }
    const typedEntry = {
      ...oldEntry,
      entryId: 'audit-typed',
      sequence: 2,
      contextVersion: 1,
      subjectKind: 'model',
      subjectId: 'unsloth/Qwen3.5-4B-GGUF',
      operationId: 'runtime-instance-7',
      requestId: REQUEST_ID,
      reasonCode: 'model_loaded',
      outcome: 'ready',
      durationMs: 412,
      numericSummaries: { layers: 36 },
      commandSummary: 'mesh-llm load name [REDACTED] --root-relay [REDACTED]'
    }

    const page = parseLogAuditPage({ items: [oldEntry, typedEntry], nextCursor: null })
    expect(page.items[0]).toEqual(oldEntry)
    expect(page.items[1]).toEqual(typedEntry)
    expect(page.items[0]?.commandSummary).toBeUndefined()
    expect(page.items[1]?.commandSummary).toBe('mesh-llm load name [REDACTED] --root-relay [REDACTED]')

    const sse = parseLogsSseFrame({
      event: 'audit_entry',
      lastEventId: 'a1:2',
      data: JSON.stringify(typedEntry)
    })
    expect(sse).toMatchObject({
      type: 'audit_entry',
      entry: { commandSummary: 'mesh-llm load name [REDACTED] --root-relay [REDACTED]' }
    })
  })

  it('rejects malformed command summaries at REST and SSE boundaries', () => {
    const malformedSummaries = [
      'mesh-llm load private-value',
      'mesh-llm models list --json --json',
      'mesh-llm models --json list',
      'mesh-llm load name [REDACTED] name [REDACTED]',
      'mesh-llm gpus run-benchmark --backend cuda --json --json',
      'mesh-llm load name [REDACTED] --relay private-relay',
      'mesh-llm  models list --json'
    ]

    for (const [index, commandSummary] of malformedSummaries.entries()) {
      const malformedEntry = {
        entryId: `audit-malformed-summary-${index}`,
        occurredAt: TIMESTAMP,
        source: 'cli',
        code: 'command_completed',
        sequence: index + 3,
        commandSummary
      }

      expect(() => parseLogAuditPage({ items: [malformedEntry], nextCursor: null })).toThrow(LogsDtoError)
      expect(() =>
        parseLogsSseFrame({
          event: 'audit_entry',
          lastEventId: `a1:${index + 3}`,
          data: JSON.stringify(malformedEntry)
        })
      ).toThrow(LogsDtoError)
    }
  })

  it('parses lifecycle, gap, and typed stream-error frames', () => {
    const event = parseLogsSseFrame({
      event: 'log_event',
      lastEventId: 'v1:2.0.0',
      data: JSON.stringify({
        eventId: EVENT_ID,
        requestId: REQUEST_ID,
        occurredAt: TIMESTAMP,
        channel: 'requests',
        sequence: 2,
        kind: 'completed'
      })
    })
    const gap = parseLogsSseFrame({
      event: 'replay_gap',
      lastEventId: 'v1:2.0.0',
      data: JSON.stringify({
        channel: 'requests',
        fromSequence: 1,
        toSequence: 2,
        recovery: { endpoint: '/api/logs/requests', cursor: 'next-page' }
      })
    })
    const error = parseLogsSseFrame({
      event: 'stream_error',
      lastEventId: 'v1:2.0.0',
      data: JSON.stringify({ code: 'invalid_event' })
    })

    expect(event.type).toBe('log_event')
    expect(gap.type).toBe('replay_gap')
    expect(error).toEqual({ type: 'stream_error', cursor: LogReplayCursor.parse('v1:2.0.0'), code: 'invalid_event' })
  })

  it('parses the additive request summary carried by current lifecycle frames', () => {
    const frame = parseLogsSseFrame({
      event: 'log_event',
      lastEventId: 'v1:3.0.0',
      data: JSON.stringify({
        eventId: EVENT_ID,
        requestId: REQUEST_ID,
        occurredAt: TIMESTAMP,
        channel: 'requests',
        sequence: 3,
        kind: 'completed',
        request: {
          requestId: REQUEST_ID,
          outcome: 'completed',
          createdAt: TIMESTAMP,
          terminalAt: TIMESTAMP,
          route: 'chat_completions',
          model: 'Qwen3',
          provider: 'mesh',
          engine: 'skippy',
          statusCode: 200,
          source: 'active'
        }
      })
    })

    expect(frame.type).toBe('log_event')
    if (frame.type !== 'log_event') throw new Error('expected lifecycle frame')
    expect(frame.event.request).toEqual(
      expect.objectContaining({ outcome: 'completed', route: 'chat_completions', statusCode: 200 })
    )
  })

  it('parses invalid-event stream errors with either valid cursor family', () => {
    // Given
    const auditInput = {
      event: 'stream_error',
      lastEventId: 'a1:42',
      data: JSON.stringify({ code: 'invalid_event' })
    }
    const lifecycleInput = {
      event: 'stream_error',
      lastEventId: 'v1:2.0.0',
      data: JSON.stringify({ code: 'invalid_event' })
    }

    // When
    const auditFrame = parseLogsSseFrame(auditInput)
    const lifecycleFrame = parseLogsSseFrame(lifecycleInput)

    // Then
    expect(auditFrame).toEqual({ type: 'stream_error', cursor: LogAuditCursor.parse('a1:42'), code: 'invalid_event' })
    expect(lifecycleFrame).toEqual({
      type: 'stream_error',
      cursor: LogReplayCursor.parse('v1:2.0.0'),
      code: 'invalid_event'
    })
  })

  it('parses audit reconciliation failures only with a valid audit cursor', () => {
    // Given
    const input = {
      event: 'stream_error',
      lastEventId: 'a1:43',
      data: JSON.stringify({ code: 'audit_reconcile_failed' })
    }

    // When
    const frame = parseLogsSseFrame(input)

    // Then
    expect(frame).toEqual({
      type: 'stream_error',
      cursor: LogAuditCursor.parse('a1:43'),
      code: 'audit_reconcile_failed'
    })
  })

  it('rejects an audit reconciliation failure paired with a lifecycle cursor', () => {
    // Given
    const input = {
      event: 'stream_error',
      lastEventId: 'v1:2.0.0',
      data: JSON.stringify({ code: 'audit_reconcile_failed' })
    }

    // When / Then
    expect(() => parseLogsSseFrame(input)).toThrow()
  })

  it.each([
    ['audit invalid-event cursor', 'a1:not-a-sequence', 'invalid_event'],
    ['lifecycle invalid-event cursor', 'v1:2.0', 'invalid_event'],
    ['audit reconciliation cursor', 'a1:not-a-sequence', 'audit_reconcile_failed']
  ])('rejects a malformed %s', (_label, lastEventId, code) => {
    // Given
    const input = {
      event: 'stream_error',
      lastEventId,
      data: JSON.stringify({ code })
    }

    // When / Then
    expect(() => parseLogsSseFrame(input)).toThrow()
  })

  it('rejects unknown audit stream-error codes', () => {
    // Given
    const input = {
      event: 'stream_error',
      lastEventId: 'a1:44',
      data: JSON.stringify({ code: 'future_error' })
    }

    // When / Then
    expect(() => parseLogsSseFrame(input)).toThrow(LogsDtoError)
  })

  it('parses audit replay gaps from the shared replay_gap event name', () => {
    expect(
      parseLogsSseFrame({
        event: 'replay_gap',
        lastEventId: 'a1:42',
        data: JSON.stringify({
          channel: 'audit',
          fromSequence: 40,
          toSequence: 42,
          recovery: { endpoint: '/api/logs/audit', cursor: 'a1:42' }
        })
      })
    ).toEqual({
      type: 'audit_gap',
      cursor: LogAuditCursor.parse('a1:42'),
      fromSequence: 40,
      toSequence: 42,
      recoveryCursor: 'a1:42'
    })

    expect(() =>
      parseLogsSseFrame({
        event: 'replay_gap',
        lastEventId: 'a1:42',
        data: JSON.stringify({
          channel: 'audit',
          fromSequence: 43,
          toSequence: 42,
          recovery: { endpoint: '/api/logs/audit', cursor: 'a1:42' }
        })
      })
    ).toThrow(LogsDtoError)
  })

  it.each([
    ['omitted', { endpoint: '/api/logs/requests' }],
    ['null', { endpoint: '/api/logs/requests', cursor: null }]
  ])('accepts an %s recovery cursor as unavailable', (_label, recovery) => {
    const gap = parseLogsSseFrame({
      event: 'replay_gap',
      lastEventId: 'v1:2.0.0',
      data: JSON.stringify({
        channel: 'requests',
        fromSequence: 1,
        toSequence: 2,
        recovery
      })
    })

    expect(gap).toMatchObject({ type: 'replay_gap', gap: { recovery: { cursor: undefined } } })
  })

  it('rejects unknown SSE types and malformed IDs before they reach feature state', () => {
    expect(() => parseLogsSseFrame({ event: 'unknown', lastEventId: 'v1:0.0.0', data: '{}' })).toThrow(LogsDtoError)
    expect(() =>
      parseLogsSseFrame({
        event: 'log_event',
        lastEventId: 'v1:malformed.0.0',
        data: JSON.stringify({
          eventId: EVENT_ID,
          requestId: REQUEST_ID,
          occurredAt: TIMESTAMP,
          channel: 'requests',
          sequence: 1,
          kind: 'completed'
        })
      })
    ).toThrow()
  })
})
