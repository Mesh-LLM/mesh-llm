import { LogEventId, LogRequestId } from '@/features/logs/api/ids'
import type { LogLifecycleEvent } from '@/features/logs/api/schemas'

export const REQUEST_ID = LogRequestId.parse('00000000-0000-4000-8000-000000000001')

export function event(id: number, occurredAt: string, kind: LogLifecycleEvent['kind']): LogLifecycleEvent {
  return {
    eventId: LogEventId.parse(`00000000-0000-4000-8000-${String(id).padStart(12, '0')}`),
    requestId: REQUEST_ID,
    occurredAt,
    kind,
    model: undefined,
    provider: undefined,
    engine: undefined,
    attemptId: undefined,
    statusCode: undefined,
    durationMs: undefined,
    tokens: undefined
  }
}
