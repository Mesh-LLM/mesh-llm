import type { LogLifecycleEvent, LogProxyAttempt, LogRequest } from '@/features/logs/api/schemas'
import { compareLogInstants } from '@/features/logs/lib/log-instant'
import { completionTokenCount } from '@/features/logs/lib/log-token-usage'

export type RetainedQueryState<T> = {
  readonly items: readonly T[] | undefined
  readonly loading: boolean
  readonly error: boolean
}

const STREAM_EVENT_KINDS = new Set<LogLifecycleEvent['kind']>([
  'stream_started',
  'stream_chunk',
  'stream_completed',
  'stream_error'
])

export function formatTimestamp(value: string): string {
  return new Date(value).toLocaleString()
}

export function formatClockTime(value: string): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3
  })
}

export function machineValue(value: string | number | undefined): string {
  return value === undefined ? 'Not recorded' : String(value)
}

export function formatDurationMs(durationMs: number | undefined): string {
  if (durationMs === undefined || !Number.isFinite(durationMs) || durationMs < 0) return 'Not recorded'
  if (durationMs < 1_000) return `${durationMs} ms`
  if (durationMs < 10_000) return `${(durationMs / 1_000).toFixed(2)} s`
  if (durationMs < 60_000) return `${(durationMs / 1_000).toFixed(1)} s`

  const totalMinutes = Math.floor(durationMs / 60_000)
  const seconds = Math.floor((durationMs % 60_000) / 1_000)
  if (totalMinutes < 60) return `${totalMinutes}m ${seconds}s`

  const hours = Math.floor(totalMinutes / 60)
  return `${hours}h ${totalMinutes % 60}m`
}

export function formatRequestDuration(request: LogRequest): string {
  if (request.terminalAt === undefined) return request.outcome === 'active' ? 'In progress' : 'Not recorded'
  return formatDurationMs(Date.parse(request.terminalAt) - Date.parse(request.createdAt))
}

function countLabel(count: number, singular: string, plural: string): string {
  return `${count.toLocaleString()} ${count === 1 ? singular : plural}`
}

export function formatAttemptEvidence(attempts: readonly LogProxyAttempt[] | undefined): string {
  if (attempts === undefined) return 'Not recorded'
  const retries = Math.max(attempts.length - 1, 0)
  return `${countLabel(attempts.length, 'attempt', 'attempts')} / ${countLabel(retries, 'retry', 'retries')}`
}

function latestTokenCount(events: readonly LogLifecycleEvent[]): number | undefined {
  const latest = events.reduce<LogLifecycleEvent | undefined>((latest, event) => {
    if (completionTokenCount(event) === undefined) return latest
    if (latest === undefined || compareLogInstants(event.occurredAt, latest.occurredAt) > 0) return event
    return latest
  }, undefined)
  return latest === undefined ? undefined : completionTokenCount(latest)
}

export function formatStreamEvidence(events: readonly LogLifecycleEvent[] | undefined): string {
  if (events === undefined) return 'Not recorded'
  const streamCount = events.filter((event) => STREAM_EVENT_KINDS.has(event.kind)).length
  const tokens = latestTokenCount(events)
  if (streamCount === 0 && tokens === undefined) return 'Not recorded'

  const streamEvidence = countLabel(streamCount, 'stream event', 'stream events')
  const tokenEvidence =
    tokens === undefined
      ? 'completion tokens not recorded'
      : countLabel(tokens, 'completion token', 'completion tokens')
  return `${streamEvidence} / ${tokenEvidence}`
}

export function attemptDurationMs(attempt: LogProxyAttempt): number | undefined {
  if (attempt.startedAt === undefined || attempt.completedAt === undefined) return undefined
  return Date.parse(attempt.completedAt) - Date.parse(attempt.startedAt)
}
