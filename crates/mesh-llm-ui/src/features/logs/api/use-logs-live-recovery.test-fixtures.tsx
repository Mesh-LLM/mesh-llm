import { act, renderHook } from '@testing-library/react'
import { vi } from 'vitest'
import { useLogsLiveRecovery, type LogsEventSourceFactory } from '@/features/logs/api/use-logs-live-recovery'
import type { LogsLedgerSearch } from '@/features/logs/lib/log-search'

export const REQUEST_A = '00000000-0000-4000-8000-000000000001'
export const REQUEST_B = '00000000-0000-4000-8000-000000000002'
export const unsupportedEventSourceFactory: LogsEventSourceFactory = () => {
  throw new Error('unsupported')
}

export type Listener = (event: MessageEvent<string>) => void

export class FakeEventSource {
  readonly listeners = new Map<string, Listener>()
  readonly url: string
  closed = false
  onopen: ((event: Event) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
  }

  addEventListener(type: string, listener: Listener) {
    this.listeners.set(type, listener)
  }

  close() {
    this.closed = true
  }

  open() {
    this.onopen?.(new Event('open'))
  }

  error() {
    this.onerror?.(new Event('error'))
  }

  emit(type: string, data: string, lastEventId: string) {
    const event = new MessageEvent<string>(type, { data })
    Object.defineProperty(event, 'lastEventId', { value: lastEventId })
    this.listeners.get(type)?.(event)
  }
}

export function eventData(requestId: string, eventId: string, sequence: number, occurredAt = '2026-08-04T12:00:00Z') {
  return JSON.stringify({
    eventId,
    requestId,
    occurredAt,
    channel: 'requests',
    sequence,
    kind: 'completed',
    model: null,
    provider: null,
    engine: null,
    attemptId: null,
    statusCode: null,
    durationMs: null,
    tokens: null
  })
}

export function projectedEventData(
  requestId: string,
  eventId: string,
  sequence: number,
  outcome = 'completed',
  route = 'chat_completions'
) {
  const event = JSON.parse(eventData(requestId, eventId, sequence)) as Record<string, unknown>
  return JSON.stringify({
    ...event,
    request: {
      requestId,
      outcome,
      createdAt: '2026-08-04T12:00:00Z',
      terminalAt: '2026-08-04T12:00:00Z',
      route,
      model: 'Qwen3',
      provider: 'reserve-a',
      engine: null,
      statusCode: 200,
      source: 'active'
    }
  })
}

export function auditData(sequence: number) {
  return JSON.stringify({
    entryId: `audit-${sequence}`,
    occurredAt: '2026-08-04T12:00:00Z',
    source: 'logs_api',
    code: 'logging_cleanup_completed',
    severity: 'info',
    sequence
  })
}

export function renderLive(options: Partial<Parameters<typeof useLogsLiveRecovery>[0]> = {}) {
  const sources: FakeEventSource[] = []
  const factory: LogsEventSourceFactory = (url) => {
    const source = new FakeEventSource(url)
    sources.push(source)
    return source
  }
  const hydrate = vi.fn(async () => undefined)
  const hydrateAudit = vi.fn(async () => undefined)
  const search: LogsLedgerSearch = options.search ?? { model: 'Qwen3', provider: 'reserve-a' }
  const initialProps: { search: LogsLedgerSearch; authoritativeSnapshot?: unknown } = {
    search,
    authoritativeSnapshot: options.authoritativeSnapshot
  }
  const result = renderHook(
    (input: { search: LogsLedgerSearch; authoritativeSnapshot?: unknown }) =>
      useLogsLiveRecovery({
        enabled: options.enabled ?? true,
        search: input.search,
        hydrate,
        authoritativeSnapshot: input.authoritativeSnapshot,
        auditEnabled: options.auditEnabled,
        hydrateAudit: options.hydrateAudit ?? hydrateAudit,
        eventSourceFactory: options.eventSourceFactory ?? factory,
        channels: options.channels
      }),
    { initialProps }
  )
  return { ...result, hydrate, hydrateAudit, sources }
}

export async function flush() {
  await act(async () => {
    await Promise.resolve()
  })
}
