import type { LogRequestId } from '@/features/logs/api/ids'
import type { LogEventKind, LogLifecycleEvent } from '@/features/logs/api/schemas'
import { HARNESS_LOG_FIXTURES, HARNESS_LOG_SCENARIO_IDS } from './requests'
import { fixtureEventId, harnessTimestamp } from './support'

type EventFixtureInput = {
  readonly requestId: LogRequestId
  readonly ordinal: number
  readonly occurredMinutesAgo: number
  readonly kind: LogEventKind
  readonly model?: string
  readonly provider?: string
  readonly engine?: string
  readonly attemptId?: string
  readonly statusCode?: number
  readonly durationMs?: number
  readonly tokens?: number
}

function eventFixture(input: EventFixtureInput): LogLifecycleEvent {
  return {
    eventId: fixtureEventId(input.requestId, input.ordinal),
    requestId: input.requestId,
    occurredAt: harnessTimestamp(input.occurredMinutesAgo),
    kind: input.kind,
    model: input.model,
    provider: input.provider,
    engine: input.engine,
    attemptId: input.attemptId,
    statusCode: input.statusCode,
    durationMs: input.durationMs,
    tokens: input.tokens
  }
}

const SCENARIO_EVENTS = new Map<string, readonly LogLifecycleEvent[]>([
  [
    HARNESS_LOG_SCENARIO_IDS.completedMesh.toString(),
    [
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 9,
        occurredMinutesAgo: 1.05,
        kind: 'completed',
        attemptId: 'mesh-primary',
        statusCode: 200,
        durationMs: 57_000,
        tokens: 612
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 1,
        occurredMinutesAgo: 2,
        kind: 'admitted'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 2,
        occurredMinutesAgo: 1.95,
        kind: 'route_selected',
        model: 'Qwen3-30B-A3B-Q4_K_M.gguf',
        provider: 'mesh-routed'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 3,
        occurredMinutesAgo: 1.9,
        kind: 'attempt_started',
        engine: 'skippy',
        attemptId: 'mesh-primary'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 4,
        occurredMinutesAgo: 1.75,
        kind: 'stream_started',
        attemptId: 'mesh-primary'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 5,
        occurredMinutesAgo: 1.5,
        kind: 'stream_chunk',
        attemptId: 'mesh-primary',
        tokens: 128
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 6,
        occurredMinutesAgo: 1.2,
        kind: 'stream_completed',
        attemptId: 'mesh-primary',
        durationMs: 54_000,
        tokens: 612
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 7,
        occurredMinutesAgo: 1.1,
        kind: 'attempt_completed',
        engine: 'skippy',
        attemptId: 'mesh-primary',
        statusCode: 200,
        durationMs: 56_000,
        tokens: 612
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.failedRetry.toString(),
    [
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 10,
        occurredMinutesAgo: 3,
        kind: 'failed',
        attemptId: 'retry-secondary',
        statusCode: 502,
        durationMs: 58_000
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 1,
        occurredMinutesAgo: 4,
        kind: 'admitted'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 2,
        occurredMinutesAgo: 3.95,
        kind: 'route_selected',
        model: 'DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf',
        provider: 'mesh-routed'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 3,
        occurredMinutesAgo: 3.9,
        kind: 'attempt_started',
        engine: 'skippy',
        attemptId: 'retry-primary'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 4,
        occurredMinutesAgo: 3.7,
        kind: 'attempt_failed',
        engine: 'skippy',
        attemptId: 'retry-primary',
        statusCode: 503,
        durationMs: 12_000
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 5,
        occurredMinutesAgo: 3.6,
        kind: 'attempt_started',
        engine: 'skippy',
        attemptId: 'retry-secondary'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 6,
        occurredMinutesAgo: 3.45,
        kind: 'stream_started',
        attemptId: 'retry-secondary'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 7,
        occurredMinutesAgo: 3.2,
        kind: 'stream_error',
        attemptId: 'retry-secondary',
        statusCode: 502,
        tokens: 41
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 8,
        occurredMinutesAgo: 3.1,
        kind: 'attempt_failed',
        engine: 'skippy',
        attemptId: 'retry-secondary',
        statusCode: 502,
        durationMs: 42_000,
        tokens: 41
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 9,
        occurredMinutesAgo: 3.05,
        kind: 'audit_error',
        attemptId: 'retry-secondary',
        statusCode: 502
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.activeStream.toString(),
    [
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.activeStream,
        ordinal: 1,
        occurredMinutesAgo: 1,
        kind: 'admitted'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.activeStream,
        ordinal: 2,
        occurredMinutesAgo: 0.9,
        kind: 'route_selected',
        model: 'Qwen3-8B-Q4_K_M.gguf',
        provider: 'mesh-routed'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.activeStream,
        ordinal: 3,
        occurredMinutesAgo: 0.8,
        kind: 'attempt_started',
        engine: 'skippy',
        attemptId: 'active-primary'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.activeStream,
        ordinal: 4,
        occurredMinutesAgo: 0.6,
        kind: 'stream_started',
        attemptId: 'active-primary'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.activeStream,
        ordinal: 5,
        occurredMinutesAgo: 0.2,
        kind: 'stream_chunk',
        attemptId: 'active-primary',
        tokens: 96
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.rejectedAdmission.toString(),
    [
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.rejectedAdmission,
        ordinal: 1,
        occurredMinutesAgo: 6,
        kind: 'admitted'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.rejectedAdmission,
        ordinal: 2,
        occurredMinutesAgo: 5.9,
        kind: 'rejected',
        statusCode: 400,
        durationMs: 80
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.cancelledClient.toString(),
    [
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.cancelledClient,
        ordinal: 1,
        occurredMinutesAgo: 8,
        kind: 'admitted'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.cancelledClient,
        ordinal: 2,
        occurredMinutesAgo: 7.9,
        kind: 'route_selected',
        model: 'Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        provider: 'mesh-routed'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.cancelledClient,
        ordinal: 3,
        occurredMinutesAgo: 7.8,
        kind: 'attempt_started',
        engine: 'native',
        attemptId: 'cancelled-primary'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.cancelledClient,
        ordinal: 4,
        occurredMinutesAgo: 7,
        kind: 'cancelled',
        attemptId: 'cancelled-primary',
        statusCode: 499,
        durationMs: 48_000,
        tokens: 73
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.droppedCapacity.toString(),
    [
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.droppedCapacity,
        ordinal: 1,
        occurredMinutesAgo: 10,
        kind: 'admitted'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.droppedCapacity,
        ordinal: 2,
        occurredMinutesAgo: 9.9,
        kind: 'route_selected',
        model: 'Qwen3-235B-A22B-Q4_K_M.gguf'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.droppedCapacity,
        ordinal: 3,
        occurredMinutesAgo: 9.8,
        kind: 'dropped',
        statusCode: 503,
        durationMs: 110
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.completedSparse.toString(),
    [
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedSparse,
        ordinal: 1,
        occurredMinutesAgo: 18,
        kind: 'admitted'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedSparse,
        ordinal: 2,
        occurredMinutesAgo: 17.5,
        kind: 'completed'
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.failedOpaque.toString(),
    [
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedOpaque,
        ordinal: 1,
        occurredMinutesAgo: 32,
        kind: 'admitted'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedOpaque,
        ordinal: 2,
        occurredMinutesAgo: 31.8,
        kind: 'attempt_started',
        attemptId: 'opaque-attempt'
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedOpaque,
        ordinal: 3,
        occurredMinutesAgo: 31.1,
        kind: 'attempt_failed',
        attemptId: 'opaque-attempt',
        statusCode: 500
      }),
      eventFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedOpaque,
        ordinal: 4,
        occurredMinutesAgo: 31,
        kind: 'failed',
        statusCode: 500,
        durationMs: 60_000
      })
    ]
  ]
])

function basicLifecycle(requestId: LogRequestId): readonly LogLifecycleEvent[] {
  const request = HARNESS_LOG_FIXTURES.find((item) => item.requestId.toString() === requestId.toString())
  if (request === undefined) return []
  if (request.outcome === 'active') {
    return [eventFixture({ requestId, ordinal: 1, occurredMinutesAgo: 0, kind: 'admitted' })]
  }
  return [
    {
      eventId: fixtureEventId(requestId, 1),
      requestId,
      occurredAt: request.createdAt,
      kind: 'admitted',
      model: undefined,
      provider: undefined,
      engine: undefined,
      attemptId: undefined,
      statusCode: undefined,
      durationMs: undefined,
      tokens: undefined
    },
    {
      eventId: fixtureEventId(requestId, 2),
      requestId,
      occurredAt: request.terminalAt ?? request.createdAt,
      kind: request.outcome,
      model: request.model,
      provider: request.provider,
      engine: request.engine,
      attemptId: undefined,
      statusCode: request.statusCode,
      durationMs: undefined,
      tokens: undefined
    }
  ]
}

export function generateLifecycleEvents(requestId: string): readonly LogLifecycleEvent[] {
  const explicit = SCENARIO_EVENTS.get(requestId)
  if (explicit !== undefined) return explicit
  const request = HARNESS_LOG_FIXTURES.find((item) => item.requestId.toString() === requestId)
  return request === undefined ? [] : basicLifecycle(request.requestId)
}
