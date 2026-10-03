import type { LogRequestId } from '@/features/logs/api/ids'
import type { LogArtifact, LogArtifactUnavailableReason } from '@/features/logs/api/schemas'
import { HARNESS_LOG_SCENARIO_IDS } from './requests'
import {
  REDACTED_REQUEST_CONTENT,
  REDACTED_RESPONSE_CONTENT,
  fixtureArtifactId,
  fixtureChecksum,
  harnessTimestamp
} from './support'

type ArtifactFixtureCommon = {
  readonly requestId: LogRequestId
  readonly ordinal: number
  readonly occurredMinutesAgo: number
  readonly kind: string
  readonly mediaKind: string | undefined
  readonly bytes: number
  readonly version: number
  readonly truncated: boolean
  readonly checksumRecorded: boolean
}

type ArtifactFixtureInput = ArtifactFixtureCommon &
  (
    | { readonly contentState: 'available'; readonly contentBase64?: string }
    | {
        readonly contentState: 'unavailable'
        readonly unavailableReason: LogArtifactUnavailableReason
        readonly redacted: boolean
      }
    | {
        readonly contentState: 'missing' | 'corrupt'
        readonly redacted: boolean
      }
  )

function artifactFixture(input: ArtifactFixtureInput): LogArtifact {
  const artifactId = fixtureArtifactId(input.requestId, input.ordinal)
  const base = {
    artifactId,
    requestId: input.requestId,
    occurredAt: harnessTimestamp(input.occurredMinutesAgo),
    kind: input.kind,
    mediaKind: input.mediaKind,
    checksum: input.checksumRecorded ? fixtureChecksum(artifactId) : undefined,
    bytes: input.bytes,
    version: input.version,
    truncated: input.truncated
  }
  if (input.contentState === 'available') {
    return { ...base, redacted: true, contentState: 'available', contentBase64: input.contentBase64 }
  }
  if (input.contentState === 'unavailable') {
    return {
      ...base,
      redacted: input.redacted,
      contentState: 'unavailable',
      unavailableReason: input.unavailableReason,
      contentBase64: undefined
    }
  }
  return { ...base, redacted: input.redacted, contentState: input.contentState, contentBase64: undefined }
}

const SCENARIO_ARTIFACTS = new Map<string, readonly LogArtifact[]>([
  [
    HARNESS_LOG_SCENARIO_IDS.completedMesh.toString(),
    [
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 1,
        occurredMinutesAgo: 1.98,
        kind: 'request_body',
        mediaKind: 'application/json',
        bytes: 648,
        version: 1,
        truncated: false,
        checksumRecorded: true,
        contentState: 'available',
        contentBase64: REDACTED_REQUEST_CONTENT
      }),
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 2,
        occurredMinutesAgo: 1.96,
        kind: 'request_headers',
        mediaKind: 'application/json',
        bytes: 192,
        version: 2,
        truncated: true,
        checksumRecorded: true,
        contentState: 'available'
      }),
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 3,
        occurredMinutesAgo: 1.08,
        kind: 'response_body',
        mediaKind: 'application/json',
        bytes: 1_024,
        version: 3,
        truncated: false,
        checksumRecorded: true,
        contentState: 'available',
        contentBase64: REDACTED_RESPONSE_CONTENT
      }),
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedMesh,
        ordinal: 4,
        occurredMinutesAgo: 1.06,
        kind: 'response_usage',
        mediaKind: 'application/json',
        bytes: 128,
        version: 1,
        truncated: false,
        checksumRecorded: false,
        contentState: 'unavailable',
        unavailableReason: 'artifact_capture_disabled',
        redacted: false
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.failedRetry.toString(),
    [
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 1,
        occurredMinutesAgo: 3.98,
        kind: 'request_body',
        mediaKind: 'application/json',
        bytes: 712,
        version: 1,
        truncated: false,
        checksumRecorded: true,
        contentState: 'available',
        contentBase64: REDACTED_REQUEST_CONTENT
      }),
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 2,
        occurredMinutesAgo: 3.1,
        kind: 'response_body',
        mediaKind: 'application/json',
        bytes: 96,
        version: 2,
        truncated: true,
        checksumRecorded: true,
        contentState: 'unavailable',
        unavailableReason: 'streaming_response_not_assembled',
        redacted: false
      }),
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 3,
        occurredMinutesAgo: 3.05,
        kind: 'error_diagnostic',
        mediaKind: 'text/plain',
        bytes: 2_048,
        version: 4,
        truncated: true,
        checksumRecorded: true,
        contentState: 'corrupt',
        redacted: true
      }),
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.failedRetry,
        ordinal: 4,
        occurredMinutesAgo: 3.02,
        kind: 'error_trace',
        mediaKind: undefined,
        bytes: 0,
        version: 1,
        truncated: false,
        checksumRecorded: false,
        contentState: 'missing',
        redacted: false
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.activeStream.toString(),
    [
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.activeStream,
        ordinal: 1,
        occurredMinutesAgo: 0.98,
        kind: 'request_body',
        mediaKind: 'application/json',
        bytes: 384,
        version: 1,
        truncated: false,
        checksumRecorded: true,
        contentState: 'available'
      })
    ]
  ],
  [
    HARNESS_LOG_SCENARIO_IDS.completedSparse.toString(),
    [
      artifactFixture({
        requestId: HARNESS_LOG_SCENARIO_IDS.completedSparse,
        ordinal: 1,
        occurredMinutesAgo: 17.5,
        kind: 'response_metadata',
        mediaKind: undefined,
        bytes: 0,
        version: 1,
        truncated: false,
        checksumRecorded: false,
        contentState: 'missing',
        redacted: false
      })
    ]
  ]
])

export function generateArtifacts(requestId: string): readonly LogArtifact[] {
  return SCENARIO_ARTIFACTS.get(requestId) ?? []
}
