import { LogAuditCursor, LogReplayCursor, LogRequestId, type LogReplayChannel } from './ids'
import {
  LogsDtoError,
  parseReplayEvent,
  parseReplayGap,
  parseAuditEntry,
  parseAuditGap,
  type LogAuditEntry,
  type ParsedReplayEvent,
  type ParsedReplayGap
} from './schemas'

export type LogsSseFilterKey = 'from' | 'to' | 'route' | 'model' | 'provider' | 'engine' | 'outcome'

export type LogsStreamErrorFrame =
  | {
      readonly type: 'stream_error'
      readonly cursor: LogReplayCursor | LogAuditCursor
      readonly code: 'invalid_event'
    }
  | {
      readonly type: 'stream_error'
      readonly cursor: LogAuditCursor
      readonly code: 'audit_reconcile_failed'
    }

export type LogsStreamErrorCode = LogsStreamErrorFrame['code']

export type LogsSseFilter = {
  readonly key: LogsSseFilterKey
  readonly value: string
}

export type LogsSseSubscription = {
  readonly channels: readonly LogReplayChannel[]
  readonly filters?: readonly LogsSseFilter[]
  readonly requestIds?: readonly LogRequestId[]
  readonly cursor?: LogReplayCursor
  readonly audit?: {
    readonly cursor?: LogAuditCursor
    readonly source?: string
    readonly severity?: string
  }
}

export type LogsSseFrame =
  | { readonly type: 'log_event'; readonly cursor: LogReplayCursor; readonly event: ParsedReplayEvent }
  | { readonly type: 'replay_gap'; readonly cursor: LogReplayCursor; readonly gap: ParsedReplayGap }
  | LogsStreamErrorFrame
  | {
      readonly type: 'audit_entry'
      readonly cursor: LogAuditCursor
      readonly entry: LogAuditEntry
    }
  | {
      readonly type: 'audit_gap'
      readonly cursor: LogAuditCursor
      readonly fromSequence: number
      readonly toSequence: number
      readonly recoveryCursor?: string
    }

export type LogsSseFrameInput = {
  readonly event: string
  readonly lastEventId: string
  readonly data: string
}

function parseJson(data: string): unknown {
  try {
    return JSON.parse(data)
  } catch {
    throw new LogsDtoError()
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object'
}

function parseStreamError(input: unknown, lastEventId: string): LogsStreamErrorFrame {
  if (!isRecord(input)) throw new LogsDtoError()

  const code = input['code']
  switch (code) {
    case 'invalid_event': {
      switch (lastEventId.slice(0, 3)) {
        case 'a1:':
          return { type: 'stream_error', cursor: LogAuditCursor.parse(lastEventId), code }
        case 'v1:':
          return { type: 'stream_error', cursor: LogReplayCursor.parse(lastEventId), code }
        default:
          throw new LogsDtoError()
      }
    }
    case 'audit_reconcile_failed':
      return { type: 'stream_error', cursor: LogAuditCursor.parse(lastEventId), code }
    default:
      throw new LogsDtoError()
  }
}

export function parseLogsSseFrame(input: LogsSseFrameInput): LogsSseFrame {
  const data = parseJson(input.data)
  switch (input.event) {
    case 'log_event': {
      const cursor = LogReplayCursor.parse(input.lastEventId)
      const event = parseReplayEvent(data)
      if (cursor.sequence(event.channel) !== BigInt(event.sequence)) throw new LogsDtoError()
      return { type: 'log_event', cursor, event }
    }
    case 'replay_gap': {
      if (input.lastEventId.startsWith('a1:')) {
        const cursor = LogAuditCursor.parse(input.lastEventId)
        const gap = parseAuditGap(data)
        return {
          type: 'audit_gap',
          cursor,
          fromSequence: gap.fromSequence,
          toSequence: gap.toSequence,
          recoveryCursor: gap.recovery.cursor ?? undefined
        }
      }
      const cursor = LogReplayCursor.parse(input.lastEventId)
      return { type: 'replay_gap', cursor, gap: parseReplayGap(data) }
    }
    case 'stream_error':
      return parseStreamError(data, input.lastEventId)
    case 'audit_entry': {
      const cursor = LogAuditCursor.parse(input.lastEventId)
      const entry = parseAuditEntry(data)
      if (cursor.sequence() !== BigInt(entry.sequence)) throw new LogsDtoError()
      return { type: 'audit_entry', cursor, entry }
    }
    default:
      throw new LogsDtoError()
  }
}

export function serializeLogsSseSubscription(subscription: LogsSseSubscription) {
  const query = new URLSearchParams()
  if (subscription.audit) {
    query.set('audit', '1')
    if (subscription.audit.cursor) query.set('cursor', subscription.audit.cursor.toString())
    if (subscription.audit.source) query.set('source', subscription.audit.source)
    if (subscription.audit.severity) query.set('severity', subscription.audit.severity)
  } else {
    for (const channel of subscription.channels) query.append('channel', channel)
    for (const filter of subscription.filters ?? []) query.append('filter', `${filter.key}:${filter.value}`)
    for (const requestId of subscription.requestIds ?? []) query.append('filter', `request_id:${requestId.toString()}`)
    if (subscription.cursor) query.set('cursor', subscription.cursor.toString())
  }
  return query.toString()
}
