import type { LogRequestId } from '@/features/logs/api/ids'
import type { LogProxyAttempt } from '@/features/logs/api/schemas'
import { HARNESS_LOG_SCENARIO_IDS } from './requests'
import { harnessTimestamp } from './support'

type ProxyAttemptFixtureInput = {
  readonly requestId: LogRequestId
  readonly attemptId: string
  readonly occurredMinutesAgo: number
  readonly target: string
  readonly provider: string | undefined
  readonly engine: string | undefined
  readonly startedMinutesAgo: number | undefined
  readonly completedMinutesAgo: number | undefined
  readonly statusCode: number | undefined
}

function proxyAttemptFixture(input: ProxyAttemptFixtureInput): LogProxyAttempt {
  return {
    attemptId: input.attemptId,
    requestId: input.requestId,
    occurredAt: harnessTimestamp(input.occurredMinutesAgo),
    target: input.target,
    provider: input.provider,
    engine: input.engine,
    startedAt: input.startedMinutesAgo === undefined ? undefined : harnessTimestamp(input.startedMinutesAgo),
    completedAt: input.completedMinutesAgo === undefined ? undefined : harnessTimestamp(input.completedMinutesAgo),
    statusCode: input.statusCode
  }
}

const SCENARIO_ATTEMPTS = new Map<string, readonly LogProxyAttempt[]>([
  [
    HARNESS_LOG_SCENARIO_IDS.completedMesh.toString(),
    [
      proxyAttemptFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        attemptId: 'mesh-primary',
        occurredMinutesAgo: 1.9,
        target: 'https://peer-a.mesh.invalid',
        provider: 'mesh-routed',
        engine: 'skippy',
        startedMinutesAgo: 1.9,
        completedMinutesAgo: 1.1,
        statusCode: 200
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.failedRetry.toString(),
    [
      proxyAttemptFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        attemptId: 'retry-secondary',
        occurredMinutesAgo: 3.6,
        target: 'https://peer-b.mesh.invalid',
        provider: 'mesh-routed',
        engine: 'skippy',
        startedMinutesAgo: 3.6,
        completedMinutesAgo: 3.1,
        statusCode: 502
      }),
      proxyAttemptFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        attemptId: 'retry-primary',
        occurredMinutesAgo: 3.9,
        target: 'http://peer-b.mesh.invalid:9337',
        provider: 'mesh-routed',
        engine: 'skippy',
        startedMinutesAgo: 3.9,
        completedMinutesAgo: 3.7,
        statusCode: 503
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.activeStream.toString(),
    [
      proxyAttemptFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.activeStream,
        attemptId: 'active-primary',
        occurredMinutesAgo: 0.8,
        target: 'http://127.0.0.1:9337',
        provider: 'mesh-routed',
        engine: 'skippy',
        startedMinutesAgo: 0.8,
        completedMinutesAgo: undefined,
        statusCode: undefined
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.cancelledClient.toString(),
    [
      proxyAttemptFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.cancelledClient,
        attemptId: 'cancelled-primary',
        occurredMinutesAgo: 7.8,
        target: 'opaque',
        provider: undefined,
        engine: undefined,
        startedMinutesAgo: 7.8,
        completedMinutesAgo: undefined,
        statusCode: undefined
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.completedLocal.toString(),
    [
      proxyAttemptFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedLocal,
        attemptId: 'local-primary',
        occurredMinutesAgo: 11.9,
        target: 'http://127.0.0.1:9447',
        provider: 'local-native',
        engine: 'native',
        startedMinutesAgo: 11.9,
        completedMinutesAgo: 11,
        statusCode: 200
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.completedSparse.toString(),
    [
      proxyAttemptFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedSparse,
        attemptId: 'opaque-metadata',
        occurredMinutesAgo: 17.8,
        target: 'opaque',
        provider: undefined,
        engine: undefined,
        startedMinutesAgo: undefined,
        completedMinutesAgo: undefined,
        statusCode: undefined
      })
    ]
  ]
])

export function generateProxyAttempts(requestId: string): readonly LogProxyAttempt[] {
  return SCENARIO_ATTEMPTS.get(requestId) ?? []
}
