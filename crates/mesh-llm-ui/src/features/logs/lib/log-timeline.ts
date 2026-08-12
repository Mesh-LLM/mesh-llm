import type { StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { LogEventKind, LogProxyAttempt } from '@/features/logs/api/schemas'

/** Stream-lifecycle event kinds that belong in the stream timeline. */
const STREAM_LIFECYCLE_KINDS = new Set<string>(['stream_started', 'stream_chunk', 'stream_completed', 'stream_error'])

export function isStreamLifecycleEvent(kind: LogEventKind): boolean {
  return STREAM_LIFECYCLE_KINDS.has(kind) || kind === ('first_token' as never)
}

export function eventTone(kind: LogEventKind): StatusBadgeTone {
  switch (kind) {
    case 'admitted':
      return 'muted'
    case 'route_selected':
    case 'attempt_started':
    case 'stream_started':
    case 'stream_chunk':
      return 'accent'
    case 'attempt_completed':
    case 'stream_completed':
    case 'completed':
      return 'good'
    case 'cancelled':
      return 'warn'
    case 'attempt_failed':
    case 'stream_error':
    case 'audit_error':
    case 'failed':
    case 'rejected':
    case 'dropped':
      return 'bad'
    default:
      return assertNever(kind)
  }
}

export function attemptTone(attempt: LogProxyAttempt): StatusBadgeTone {
  if (attempt.statusCode !== undefined) return attempt.statusCode >= 400 ? 'bad' : 'good'
  return attempt.startedAt !== undefined && attempt.completedAt === undefined ? 'accent' : 'muted'
}

export function attemptStatus(attempt: LogProxyAttempt): string {
  if (attempt.statusCode !== undefined) return `HTTP ${attempt.statusCode}`
  if (attempt.startedAt !== undefined && attempt.completedAt === undefined) return 'In progress'
  return 'Status not recorded'
}

export function attemptOutcomeLabel(attempt: LogProxyAttempt): string {
  if (attempt.statusCode !== undefined) return attempt.statusCode >= 400 ? 'Failed' : 'Success'
  if (attempt.startedAt !== undefined && attempt.completedAt === undefined) return 'In progress'
  return 'Status not recorded'
}

export function elapsedMilliseconds(start: string | undefined, end: string | undefined): number | undefined {
  if (start === undefined || end === undefined) return undefined
  const elapsed = Date.parse(end) - Date.parse(start)
  return elapsed >= 0 ? elapsed : undefined
}

export function attemptDurationMs(attempt: LogProxyAttempt): number | undefined {
  return elapsedMilliseconds(attempt.startedAt, attempt.completedAt)
}

export function attemptRouteLabel(attempt: LogProxyAttempt): string {
  const parts = [attempt.provider, attempt.engine].filter((part): part is string => part !== undefined)
  return parts.length > 0 ? parts.join(' / ') : attempt.target
}

export function formatLogTimestampMs(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3
  })
}

function assertNever(value: never): never {
  throw new Error(`Unhandled request timeline value: ${String(value)}`)
}
