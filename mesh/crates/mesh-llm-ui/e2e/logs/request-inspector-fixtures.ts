export const REQUEST_INSPECTOR_IDS = {
  completed: '10000000-0000-4000-8000-000000000001',
  malformed: '10000000-0000-4000-8000-000000000002',
  empty: '10000000-0000-4000-8000-000000000003',
  failed: '10000000-0000-4000-8000-000000000004',
  active: '10000000-0000-4000-8000-000000000005',
  transient: '10000000-0000-4000-8000-000000000006',
  streaming: '10000000-0000-4000-8000-000000000007'
} as const

export const REQUEST_INSPECTOR_ARTIFACT_IDS = {
  request: '30000000-0000-4000-8000-000000000001',
  response: '30000000-0000-4000-8000-000000000002',
  malformed: '30000000-0000-4000-8000-000000000003',
  missing: '30000000-0000-4000-8000-000000000004',
  unavailable: '30000000-0000-4000-8000-000000000005',
  corrupt: '30000000-0000-4000-8000-000000000006',
  errorCorrupt: '30000000-0000-4000-8000-000000000007',
  errorMissing: '30000000-0000-4000-8000-000000000008',
  streamingResponse: '30000000-0000-4000-8000-000000000009'
} as const

export const REQUEST_INSPECTOR_SHELL_STATUS = {
  node_id: 'request-inspector-e2e',
  node_state: 'serving',
  peers: [],
  models: [],
  gpus: []
}

const CALLER_ENDPOINT_ID = '9f0c4cbe8cb7a8d5d577c20e50ef03fd2f63a2e7fd9897c155823bcbb281bb04'

type EventFixture = readonly [
  sequence: number,
  occurredAt: string,
  kind: string,
  attemptId?: string,
  statusCode?: number,
  tokens?: number
]
type AttemptFixture = readonly [attemptId: string, occurredAt: string, target: string, statusCode: number]
type ArtifactFixture = readonly [
  artifactId: string,
  kind: string,
  contentState: 'available' | 'unavailable' | 'missing' | 'corrupt',
  bytes?: number,
  version?: number
]

type WireArtifact = ReturnType<typeof artifact>
export type RequestInspectorScenario = {
  readonly summary: ReturnType<typeof summary>
  readonly events: readonly ReturnType<typeof event>[]
  readonly attempts: readonly ReturnType<typeof attempt>[]
  readonly artifacts: readonly WireArtifact[]
}

function timestamp(minute: number, second: string): string {
  return `2026-08-04T12:${String(minute).padStart(2, '0')}:${second}Z`
}

function summary(requestId: string, outcome: 'active' | 'completed' | 'failed', minute: number) {
  return {
    requestId,
    outcome,
    createdAt: timestamp(minute, '00'),
    terminalAt: outcome === 'active' ? null : timestamp(minute, '04'),
    route: 'reserve',
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    statusCode: outcome === 'completed' ? 200 : outcome === 'failed' ? 502 : null,
    source: outcome === 'active' ? 'active' : 'durable'
  }
}

function event(requestId: string, fixture: EventFixture) {
  return {
    eventId: `20000000-0000-4000-8000-${String(fixture[0]).padStart(12, '0')}`,
    requestId,
    occurredAt: fixture[1],
    kind: fixture[2],
    model: 'Qwen3',
    provider: 'reserve-a',
    engine: 'skippy',
    attemptId: fixture[3] ?? null,
    statusCode: fixture[4] ?? null,
    durationMs: null,
    tokens: fixture[5] ?? null
  }
}

function attempt(requestId: string, fixture: AttemptFixture) {
  return {
    attemptId: fixture[0],
    requestId,
    occurredAt: fixture[1],
    target: fixture[2],
    provider: 'reserve-a',
    engine: 'skippy',
    startedAt: fixture[1],
    completedAt: fixture[1],
    statusCode: fixture[3]
  }
}

function artifact(requestId: string, fixture: ArtifactFixture) {
  const contentBase64: string | null = null
  return {
    artifactId: fixture[0],
    requestId,
    occurredAt: '2026-08-04T12:00:02Z',
    kind: fixture[1],
    mediaKind: 'application/json',
    checksum: null,
    bytes: fixture[3] ?? 384,
    version: fixture[4] ?? 2,
    redacted: true,
    truncated: false,
    contentState: fixture[2],
    unavailableReason: fixture[2] === 'unavailable' ? 'streaming_response_not_assembled' : null,
    contentBase64
  }
}

function encoded(text: string): string {
  return Buffer.from(text, 'utf8').toString('base64')
}

const REQUEST = REQUEST_INSPECTOR_IDS
const ARTIFACT = REQUEST_INSPECTOR_ARTIFACT_IDS
const completedArtifacts = [
  artifact(REQUEST.completed, [ARTIFACT.request, 'request_body', 'available']),
  artifact(REQUEST.completed, [ARTIFACT.response, 'response_body', 'available']),
  artifact(REQUEST.completed, [ARTIFACT.missing, 'request_archive_missing', 'missing']),
  artifact(REQUEST.completed, [ARTIFACT.unavailable, 'response_archive_unavailable', 'unavailable']),
  artifact(REQUEST.completed, [ARTIFACT.corrupt, 'payload_archive_corrupt', 'corrupt'])
] as const
const malformedArtifact = artifact(REQUEST.malformed, [ARTIFACT.malformed, 'request_body', 'available'])
const failedArtifacts = [
  artifact(REQUEST.failed, [ARTIFACT.errorCorrupt, 'error_diagnostic', 'corrupt', 2048, 4]),
  artifact(REQUEST.failed, [ARTIFACT.errorMissing, 'error_trace', 'missing'])
] as const
const streamingResponseArtifact = {
  ...artifact(REQUEST.streaming, [ARTIFACT.streamingResponse, 'response_body', 'available']),
  mediaKind: 'text/event-stream'
}

export const REQUEST_INSPECTOR_STREAM_HOSTILE_TEXT = '<img src=stream onerror=alert(4)><script>alert(5)</script>'

export const REQUEST_INSPECTOR_SCENARIOS: Readonly<Record<string, RequestInspectorScenario>> = {
  [REQUEST.completed]: {
    summary: {
      ...summary(REQUEST.completed, 'completed', 0),
      callerEndpointId: CALLER_ENDPOINT_ID,
      callerAddr: '203.0.113.24:48712',
      callerPathType: 'remote_quic_http'
    },
    events: [
      event(REQUEST.completed, [0, timestamp(0, '00.500'), 'admitted']),
      event(REQUEST.completed, [1, timestamp(0, '01.200'), 'stream_started']),
      event(REQUEST.completed, [2, timestamp(0, '02'), 'stream_chunk']),
      event(REQUEST.completed, [3, timestamp(0, '03'), 'stream_completed', undefined, undefined, 42])
    ],
    attempts: [
      attempt(REQUEST.completed, ['mesh-primary', timestamp(0, '00.900'), 'https://peer-a.mesh.invalid', 200])
    ],
    artifacts: completedArtifacts
  },
  [REQUEST.malformed]: {
    summary: summary(REQUEST.malformed, 'completed', 1),
    events: [],
    attempts: [],
    artifacts: [malformedArtifact]
  },
  [REQUEST.empty]: {
    summary: summary(REQUEST.empty, 'completed', 2),
    events: [],
    attempts: [],
    artifacts: []
  },
  [REQUEST.failed]: {
    summary: summary(REQUEST.failed, 'failed', 3),
    events: [
      event(REQUEST.failed, [4, timestamp(3, '01.200'), 'attempt_failed', 'retry-primary', 503]),
      event(REQUEST.failed, [5, timestamp(3, '02.200'), 'stream_error', 'retry-secondary', 502]),
      event(REQUEST.failed, [6, timestamp(3, '02.500'), 'audit_error', undefined, 502]),
      event(REQUEST.failed, [7, timestamp(3, '03'), 'failed', undefined, 502])
    ],
    attempts: [
      attempt(REQUEST.failed, ['retry-primary', timestamp(3, '01'), 'http://peer-b.mesh.invalid:9337', 503]),
      attempt(REQUEST.failed, ['retry-secondary', timestamp(3, '02'), 'https://peer-b.mesh.invalid', 502])
    ],
    artifacts: failedArtifacts
  },
  [REQUEST.active]: {
    summary: summary(REQUEST.active, 'active', 4),
    events: [],
    attempts: [],
    artifacts: []
  },
  [REQUEST.transient]: {
    summary: { ...summary(REQUEST.transient, 'completed', 5), source: 'active' },
    events: [],
    attempts: [],
    artifacts: []
  },
  [REQUEST.streaming]: {
    summary: summary(REQUEST.streaming, 'completed', 6),
    events: [],
    attempts: [],
    artifacts: [streamingResponseArtifact]
  }
}

export const REQUEST_INSPECTOR_ARTIFACT_DETAILS: Readonly<Record<string, WireArtifact>> = {
  [ARTIFACT.request]: {
    ...completedArtifacts[0],
    contentBase64: encoded(
      JSON.stringify({
        model: 'Qwen3',
        image: '<img src=payload onerror=alert(2)>',
        script: '<script>globalThis.compromised=true</script>'
      })
    )
  },
  [ARTIFACT.response]: {
    ...completedArtifacts[1],
    contentBase64: encoded(JSON.stringify({ choices: [{ role: 'assistant' }], complete: true }))
  },
  [ARTIFACT.malformed]: {
    ...malformedArtifact,
    contentBase64: encoded('{"broken": "<img src=malformed onerror=alert(3)>"')
  },
  [ARTIFACT.streamingResponse]: {
    ...streamingResponseArtifact,
    contentBase64: encoded(
      [
        'event: delta',
        'id: stream-1',
        'data: {"delta":"hello"}',
        '',
        `data: ${REQUEST_INSPECTOR_STREAM_HOSTILE_TEXT}`,
        '',
        'event: done',
        'data: [DONE]',
        ''
      ].join('\n')
    )
  }
}

export function requestDeleteReceipt(requestId: string) {
  return {
    operationId: '40000000-0000-4000-8000-000000000001',
    auditId: '40000000-0000-4000-8000-000000000002',
    requestId,
    state: 'completed',
    selectionFingerprint: 'request-inspector-delete',
    planned: { requests: 1, events: 4, artifacts: 5, proxyRecords: 1, databaseRows: 11 },
    executed: { requests: 1, events: 4, artifacts: 5, proxyRecords: 1, databaseRows: 11 },
    artifactDeletion: { removed: 5, failed: 0 }
  }
}
