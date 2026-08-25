import { LogArtifactId, LogEventId, type LogRequestId } from '@/features/logs/api/ids'

// Captured once at module load so fixture offsets stay stable for the session
// while the dataset keeps falling inside the ledger's rolling time presets.
const HARNESS_REFERENCE_TIME_MS = Date.now()

export const HARNESS_REFERENCE_TIME = new Date(HARNESS_REFERENCE_TIME_MS).toISOString()

export const REDACTED_REQUEST_CONTENT =
  'eyJtb2RlbCI6IlF3ZW4zIiwibWVzc2FnZXMiOlt7InJvbGUiOiJ1c2VyIiwiY29udGVudCI6IltSRURBQ1RFRF0ifV19'
export const REDACTED_RESPONSE_CONTENT =
  'eyJjaG9pY2VzIjpbeyJtZXNzYWdlIjp7InJvbGUiOiJhc3Npc3RhbnQiLCJjb250ZW50IjoiW1JFREFDVEVEXSJ9LCJmaW5pc2hfcmVhc29uIjoic3RvcCJ9XX0='

export function harnessTimestamp(minutesAgo: number): string {
  return new Date(HARNESS_REFERENCE_TIME_MS - minutesAgo * 60_000).toISOString()
}

function derivedUuid(requestId: LogRequestId, namespace: '1' | '2', ordinal: number): string {
  const requestHex = requestId.toString().replaceAll('-', '')
  const ordinalHex = ordinal.toString(16).padStart(2, '0').slice(-2)
  return `${requestHex.slice(0, 8)}-${requestHex.slice(8, 12)}-4${namespace}${ordinalHex}-8${requestHex.slice(17, 20)}-${requestHex.slice(20, 32)}`
}

export function fixtureEventId(requestId: LogRequestId, ordinal: number): LogEventId {
  return LogEventId.parse(derivedUuid(requestId, '1', ordinal))
}

export function fixtureArtifactId(requestId: LogRequestId, ordinal: number): LogArtifactId {
  return LogArtifactId.parse(derivedUuid(requestId, '2', ordinal))
}

export function fixtureChecksum(artifactId: LogArtifactId): string {
  const artifactHex = artifactId.toString().replaceAll('-', '')
  return `sha256:${artifactHex}${artifactHex}`
}
