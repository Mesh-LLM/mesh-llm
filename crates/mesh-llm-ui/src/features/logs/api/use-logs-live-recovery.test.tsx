import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useLogsLiveRecovery, type LogsEventSourceFactory } from '@/features/logs/api/use-logs-live-recovery'
import type { LogsLedgerSearch } from '@/features/logs/lib/log-search'

const REQUEST_A = '00000000-0000-4000-8000-000000000001'
const REQUEST_B = '00000000-0000-4000-8000-000000000002'
const unsupportedEventSourceFactory: LogsEventSourceFactory = () => {
  throw new Error('unsupported')
}

type Listener = (event: MessageEvent<string>) => void

class FakeEventSource {
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

function eventData(requestId: string, eventId: string, sequence: number, occurredAt = '2026-08-04T12:00:00Z') {
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

function projectedEventData(
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

function auditData(sequence: number) {
  return JSON.stringify({
    entryId: `audit-${sequence}`,
    occurredAt: '2026-08-04T12:00:00Z',
    source: 'logs_api',
    code: 'logging_cleanup_completed',
    severity: 'info',
    sequence
  })
}

function renderLive(options: Partial<Parameters<typeof useLogsLiveRecovery>[0]> = {}) {
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

async function flush() {
  await act(async () => {
    await Promise.resolve()
  })
}

afterEach(() => {
  vi.useRealTimers()
})

describe('useLogsLiveRecovery', () => {
  it('preserves lifecycle gap recovery when the initial hydration is still in flight', async () => {
    let resolveInitial: (() => void) | undefined
    let calls = 0
    const hydrate = vi.fn(() => {
      calls += 1
      if (calls > 1) return Promise.resolve()
      return new Promise<void>((resolve) => {
        resolveInitial = resolve
      })
    })
    const sources: FakeEventSource[] = []
    const factory: LogsEventSourceFactory = (url) => {
      const source = new FakeEventSource(url)
      sources.push(source)
      return source
    }
    const { result } = renderHook(() =>
      useLogsLiveRecovery({
        enabled: true,
        search: { replayCursor: 'v1:0.0.0' },
        hydrate,
        eventSourceFactory: factory
      })
    )
    act(() => sources[0]?.open())
    act(() =>
      sources[0]?.emit(
        'replay_gap',
        JSON.stringify({
          channel: 'requests',
          fromSequence: 1,
          toSequence: 2,
          recovery: { endpoint: '/api/logs/requests', cursor: 'next-page' }
        }),
        'v1:2.0.0'
      )
    )
    expect(result.current.state).toBe('gap')

    resolveInitial?.()
    await flush()
    await flush()

    expect(hydrate).toHaveBeenCalledTimes(1)
    expect(result.current.state).toBe('connected')
  })

  it('preserves audit gap recovery when the initial hydration is still in flight', async () => {
    let resolveInitial: (() => void) | undefined
    let calls = 0
    const hydrateAudit = vi.fn(() => {
      calls += 1
      if (calls > 1) return Promise.resolve()
      return new Promise<void>((resolve) => {
        resolveInitial = resolve
      })
    })
    const { result, sources } = renderLive({ auditEnabled: true, hydrateAudit })
    act(() => {
      sources[0]?.open()
      sources[1]?.open()
    })
    act(() =>
      sources[1]?.emit(
        'replay_gap',
        JSON.stringify({
          channel: 'audit',
          fromSequence: 1,
          toSequence: 2,
          recovery: { endpoint: '/api/logs/audit', cursor: 'a1:2' }
        }),
        'a1:2'
      )
    )
    expect(result.current.state).toBe('gap')

    resolveInitial?.()
    await flush()
    await flush()

    expect(hydrateAudit).toHaveBeenCalledTimes(1)
    expect(result.current.state).toBe('connected')
  })

  it('accepts server-reconciled cross-process audit rows without browser polling', async () => {
    vi.useFakeTimers()
    const { hydrateAudit, result, sources } = renderLive({ enabled: false, auditEnabled: true })
    await flush()
    act(() => sources[0]?.open())
    expect(result.current.state).toBe('connected')
    expect(hydrateAudit).not.toHaveBeenCalled()

    act(() => sources[0]?.emit('audit_entry', auditData(1), 'a1:1'))
    await flush()

    expect(hydrateAudit).not.toHaveBeenCalled()
    expect(result.current.auditEntries.map((entry) => entry.entryId)).toEqual(['audit-1'])
    expect(result.current.state).toBe('connected')
  })

  it('does not hydrate, open a stream, or schedule timers while logs are unsupported', async () => {
    vi.useFakeTimers()
    const { hydrate, sources } = renderLive({ enabled: false })

    await flush()
    expect(hydrate).not.toHaveBeenCalled()
    expect(sources).toHaveLength(0)
    expect(vi.getTimerCount()).toBe(0)

    act(() => vi.advanceTimersByTime(60_000))
    await flush()
    expect(hydrate).not.toHaveBeenCalled()
    expect(sources).toHaveLength(0)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('keeps the independent audit stream live when request logs are unsupported', async () => {
    const { hydrate, hydrateAudit, result, sources } = renderLive({ enabled: false, auditEnabled: true })
    await flush()

    expect(sources).toHaveLength(1)
    expect(sources[0]?.url).toBe('/api/logs/events?audit=1')
    expect(hydrate).not.toHaveBeenCalled()
    expect(hydrateAudit).not.toHaveBeenCalled()
    act(() => sources[0]?.open())
    expect(result.current.state).toBe('connected')
  })

  it('keeps relative subscriptions rolling so later live events are accepted', async () => {
    vi.useFakeTimers({ now: new Date('2026-08-04T12:30:00Z') })
    const search: LogsLedgerSearch = {
      timeRange: '7d' as const,
      model: 'Qwen3',
      provider: 'reserve-a',
      engine: 'skippy',
      route: 'chat',
      outcome: 'completed'
    }
    const { hydrate, rerender, sources, result, unmount } = renderLive({
      channels: ['requests', 'operations'],
      search
    })

    await flush()
    expect(hydrate).not.toHaveBeenCalled()
    expect(sources[0]?.url).toBe(
      '/api/logs/events?channel=requests&channel=operations&filter=model%3AQwen3&filter=provider%3Areserve-a&filter=engine%3Askippy&filter=route%3Achat&filter=outcome%3Acompleted'
    )
    act(() => sources[0]?.open())
    expect(result.current.state).toBe('connected')
    rerender({ search })
    expect(sources).toHaveLength(1)
    expect(sources[0]?.closed).toBe(false)

    act(() => vi.advanceTimersByTime(8 * 24 * 60 * 60 * 1_000))
    act(() =>
      sources[0]?.emit(
        'log_event',
        eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1, '2026-08-12T12:30:00Z'),
        'v1:1.0.0'
      )
    )
    await flush()

    expect(result.current.liveRequestIds).toEqual([REQUEST_A])
    expect(hydrate).toHaveBeenCalledTimes(1)
    expect(sources).toHaveLength(1)
    unmount()
    expect(sources[0]?.closed).toBe(true)
  })

  it('uses an independent audit stream with entry, gap, error, cursor, and disconnected polling recovery', async () => {
    vi.useFakeTimers()
    const { hydrate, hydrateAudit, result, sources } = renderLive({ auditEnabled: true })
    await flush()

    expect(sources).toHaveLength(2)
    expect(sources[0]?.url).toContain('channel=requests&channel=operations')
    expect(sources[1]?.url).toBe('/api/logs/events?audit=1')
    expect(hydrate).not.toHaveBeenCalled()
    expect(hydrateAudit).not.toHaveBeenCalled()

    act(() => {
      sources[0]?.open()
      sources[1]?.open()
    })
    expect(result.current.state).toBe('connected')

    act(() => sources[1]?.emit('audit_entry', auditData(1), 'a1:1'))
    await flush()
    expect(hydrateAudit).not.toHaveBeenCalled()
    expect(hydrate).not.toHaveBeenCalled()
    expect(result.current.auditEntries.map((entry) => entry.entryId)).toEqual(['audit-1'])

    act(() =>
      sources[1]?.emit(
        'replay_gap',
        JSON.stringify({
          channel: 'audit',
          fromSequence: 2,
          toSequence: 3,
          recovery: { endpoint: '/api/logs/audit', cursor: 'a1:3' }
        }),
        'a1:3'
      )
    )
    expect(result.current.state).toBe('gap')
    await flush()
    expect(hydrateAudit).toHaveBeenCalledTimes(1)
    expect(result.current.state).toBe('connected')

    act(() => sources[1]?.emit('stream_error', JSON.stringify({ code: 'invalid_event' }), 'a1:3'))
    expect(result.current.state).toBe('stale')
    await flush()
    expect(hydrateAudit).toHaveBeenCalledTimes(2)
    expect(result.current.state).toBe('connected')
    act(() => vi.advanceTimersByTime(5_000))
    await flush()
    expect(hydrateAudit).toHaveBeenCalledTimes(2)
  })

  it('does not re-hydrate when the audit stream fails to reconnect a second time while already polling', async () => {
    vi.useFakeTimers()
    const { hydrateAudit, result, sources } = renderLive({ enabled: false, auditEnabled: true })
    await flush()
    const source = sources[0]

    act(() => source?.error())
    expect(result.current.state).toBe('reconnecting')
    act(() => vi.advanceTimersByTime(1_000))
    await flush()
    expect(result.current.state).toBe('polling')
    expect(hydrateAudit).toHaveBeenCalledTimes(1)

    // Native EventSource retries on its own schedule and calls onerror again on
    // every failed attempt. A second failure while already polling must not
    // re-enter startPolling and fire a duplicate hydrate — the reconciliation
    // interval from the first entry is still live and owns future refreshes.
    act(() => source?.error())
    act(() => vi.advanceTimersByTime(1_000))
    await flush()
    expect(result.current.state).toBe('polling')
    expect(hydrateAudit).toHaveBeenCalledTimes(1)

    // The reconciliation interval from the first entry must still be the one
    // driving refreshes — the second failure should not have restarted or
    // dropped it.
    act(() => vi.advanceTimersByTime(5_000))
    await flush()
    expect(hydrateAudit).toHaveBeenCalledTimes(2)
  })

  it('serializes route and reconnects while source remains unsupported', async () => {
    const { rerender, sources } = renderLive({ search: { route: 'reserve', source: 'active' } })
    await flush()

    expect(sources[0]?.url).toBe('/api/logs/events?channel=requests&channel=operations&filter=route%3Areserve')
    rerender({ search: { route: 'mesh', source: 'durable' } })

    expect(sources[0]?.closed).toBe(true)
    expect(sources[1]?.url).toBe('/api/logs/events?channel=requests&channel=operations&filter=route%3Amesh')
  })

  it('preserves absolute from/to bounds and all supported filters across a deterministic reconnect', async () => {
    const before = {
      from: '2026-08-03T00:00:00Z',
      to: '2026-08-04T00:00:00Z',
      model: 'Qwen3',
      provider: 'reserve-a',
      engine: 'skippy',
      route: 'chat',
      source: 'durable',
      outcome: 'completed'
    } satisfies LogsLedgerSearch
    const after = { ...before, to: '2026-08-05T00:00:00Z', outcome: 'failed' } satisfies LogsLedgerSearch
    const { rerender, sources } = renderLive({ search: before })

    await flush()
    expect(sources[0]?.url).toBe(
      '/api/logs/events?channel=requests&channel=operations&filter=from%3A2026-08-03T00%3A00%3A00Z&filter=to%3A2026-08-04T00%3A00%3A00Z&filter=model%3AQwen3&filter=provider%3Areserve-a&filter=engine%3Askippy&filter=route%3Achat&filter=outcome%3Acompleted'
    )

    rerender({ search: after })

    expect(sources[0]?.closed).toBe(true)
    expect(sources[1]?.url).toBe(
      '/api/logs/events?channel=requests&channel=operations&filter=from%3A2026-08-03T00%3A00%3A00Z&filter=to%3A2026-08-05T00%3A00%3A00Z&filter=model%3AQwen3&filter=provider%3Areserve-a&filter=engine%3Askippy&filter=route%3Achat&filter=outcome%3Afailed'
    )
  })

  it('reconnects with the new outcome when only the outcome filter changes', async () => {
    const { rerender, sources } = renderLive({ search: { outcome: 'active' } })
    await flush()

    expect(sources[0]?.url).toBe('/api/logs/events?channel=requests&channel=operations&filter=outcome%3Aactive')
    rerender({ search: { outcome: 'completed' } })

    expect(sources[0]?.closed).toBe(true)
    expect(sources[1]?.url).toBe('/api/logs/events?channel=requests&channel=operations&filter=outcome%3Acompleted')
  })

  it('merges new request IDs in order while suppressing repeated sequence and event frames', async () => {
    const { hydrate, sources, result } = renderLive()
    await flush()
    const source = sources[0]
    act(() => source?.open())
    act(() =>
      source?.emit(
        'log_event',
        eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1, '2026-08-04T12:00:01Z'),
        'v1:1.0.0'
      )
    )
    await flush()
    act(() =>
      source?.emit(
        'log_event',
        eventData(REQUEST_B, '00000000-0000-4000-8000-000000000004', 2, '2026-08-04T12:00:02Z'),
        'v1:2.0.0'
      )
    )
    act(() =>
      source?.emit(
        'log_event',
        eventData(REQUEST_B, '00000000-0000-4000-8000-000000000004', 2, '2026-08-04T12:00:02Z'),
        'v1:2.0.0'
      )
    )

    expect(result.current.liveRequestIds).toEqual([REQUEST_A, REQUEST_B])
    expect(hydrate).toHaveBeenCalledTimes(2)
  })

  it('merges projected request rows without refreshing the ledger and removes rows that leave the filter', async () => {
    const search = { model: 'Qwen3', provider: 'reserve-a', outcome: 'active' } satisfies LogsLedgerSearch
    const { hydrate, sources, result } = renderLive({ search })
    const source = sources[0]

    act(() =>
      source?.emit(
        'log_event',
        projectedEventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1, 'active'),
        'v1:1.0.0'
      )
    )
    expect(result.current.requestUpdates.map((request) => request.requestId.toString())).toEqual([REQUEST_A])
    expect(result.current.excludedRequestIds).toEqual([])
    expect(hydrate).not.toHaveBeenCalled()

    act(() =>
      source?.emit(
        'log_event',
        projectedEventData(REQUEST_A, '00000000-0000-4000-8000-000000000004', 2, 'completed'),
        'v1:2.0.0'
      )
    )
    expect(result.current.requestUpdates).toEqual([])
    expect(result.current.excludedRequestIds).toEqual([REQUEST_A])
    expect(hydrate).not.toHaveBeenCalled()
  })

  it('discards projected rows superseded by an authoritative recovery', async () => {
    const { hydrate, sources, result } = renderLive()
    const source = sources[0]
    act(() =>
      source?.emit('log_event', projectedEventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1), 'v1:1.0.0')
    )
    expect(result.current.requestUpdates).toHaveLength(1)

    act(() =>
      source?.emit(
        'replay_gap',
        JSON.stringify({
          channel: 'requests',
          fromSequence: 2,
          toSequence: 2,
          recovery: { endpoint: '/api/logs/requests', cursor: null }
        }),
        'v1:2.0.0'
      )
    )
    await flush()

    expect(hydrate).toHaveBeenCalledTimes(1)
    expect(result.current.requestUpdates).toEqual([])
  })

  it('discards projected rows when an external authoritative query is replaced', () => {
    const initialSnapshot = { items: [] }
    const { rerender, sources, result } = renderLive({ authoritativeSnapshot: initialSnapshot })
    act(() =>
      sources[0]?.emit(
        'log_event',
        projectedEventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1),
        'v1:1.0.0'
      )
    )
    expect(result.current.requestUpdates).toHaveLength(1)

    rerender({ search: { model: 'Qwen3', provider: 'reserve-a' }, authoritativeSnapshot: { items: [] } })

    expect(result.current.requestUpdates).toEqual([])
  })

  it('orders retained request IDs oldest-first by instant across offsets', async () => {
    // Given
    const { sources, result } = renderLive()
    await flush()
    const source = sources[0]

    // When
    act(() =>
      source?.emit(
        'log_event',
        eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1, '2026-08-04T12:00:00Z'),
        'v1:1.0.0'
      )
    )
    await flush()
    act(() =>
      source?.emit(
        'log_event',
        eventData(REQUEST_B, '00000000-0000-4000-8000-000000000004', 2, '2026-08-04T10:30:00-02:00'),
        'v1:2.0.0'
      )
    )

    // Then
    expect(result.current.liveRequestIds).toEqual([REQUEST_A, REQUEST_B])
  })

  it('preserves arrival order for equal instants across offsets', async () => {
    // Given
    const { sources, result } = renderLive()
    await flush()
    const source = sources[0]

    // When
    act(() =>
      source?.emit(
        'log_event',
        eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1, '2026-08-04T12:00:00Z'),
        'v1:1.0.0'
      )
    )
    await flush()
    act(() =>
      source?.emit(
        'log_event',
        eventData(REQUEST_B, '00000000-0000-4000-8000-000000000004', 2, '2026-08-04T10:00:00-02:00'),
        'v1:2.0.0'
      )
    )

    // Then
    expect(result.current.liveRequestIds).toEqual([REQUEST_A, REQUEST_B])
  })

  it('hydrates later lifecycle events for the same request while keeping the live request projection deduped', async () => {
    const { hydrate, sources, result } = renderLive()
    await flush()
    const source = sources[0]
    act(() => source?.emit('log_event', eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1), 'v1:1.0.0'))
    await flush()
    act(() => source?.emit('log_event', eventData(REQUEST_A, '00000000-0000-4000-8000-000000000004', 2), 'v1:2.0.0'))
    await flush()

    expect(result.current.liveRequestIds).toEqual([REQUEST_A])
    expect(hydrate).toHaveBeenCalledTimes(2)
  })

  it('keeps the native EventSource instance for reconnect and falls back to bounded polling only while disconnected', async () => {
    vi.useFakeTimers()
    const { hydrate, sources, result } = renderLive()
    await flush()
    const source = sources[0]
    act(() => source?.error())
    expect(result.current.state).toBe('reconnecting')
    act(() => vi.advanceTimersByTime(1_000))
    expect(result.current.state).toBe('polling')
    act(() => vi.advanceTimersByTime(15_000))
    expect(sources).toHaveLength(1)
    act(() => source?.open())
    expect(result.current.state).toBe('connected')
    expect(hydrate.mock.calls.length).toBeLessThanOrEqual(2)
  })

  it('pauses only future fallback interval hydrations without replacing the source or timer', async () => {
    vi.useFakeTimers()
    const { hydrate, sources, result } = renderLive()
    await flush()
    const source = sources[0]
    act(() => source?.error())
    act(() => vi.advanceTimersByTime(1_000))
    await flush()
    // Entering polling hydrates once immediately, before any interval tick.
    expect(hydrate).toHaveBeenCalledTimes(1)
    const timerCount = vi.getTimerCount()
    const togglePolling = result.current.togglePolling

    expect(result.current.pollingEnabled).toBe(true)
    act(() => result.current.togglePolling())

    expect(result.current.pollingEnabled).toBe(false)
    expect(result.current.togglePolling).toBe(togglePolling)
    expect(vi.getTimerCount()).toBe(timerCount)
    act(() => vi.advanceTimersByTime(15_000))
    await flush()
    expect(hydrate).toHaveBeenCalledTimes(1)
    expect(sources).toHaveLength(1)
    expect(source?.closed).toBe(false)
  })

  it('resumes on the next existing interval boundary without hydrating immediately', async () => {
    vi.useFakeTimers()
    const { hydrate, sources, result } = renderLive()
    await flush()
    act(() => sources[0]?.error())
    act(() => vi.advanceTimersByTime(1_000))
    const timerCount = vi.getTimerCount()
    act(() => result.current.togglePolling())
    act(() => vi.advanceTimersByTime(10_000))
    await flush()
    const pausedHydrationCount = hydrate.mock.calls.length

    act(() => result.current.togglePolling())

    expect(result.current.pollingEnabled).toBe(true)
    expect(hydrate).toHaveBeenCalledTimes(pausedHydrationCount)
    expect(vi.getTimerCount()).toBe(timerCount)
    act(() => vi.advanceTimersByTime(4_999))
    expect(hydrate).toHaveBeenCalledTimes(pausedHydrationCount)
    act(() => vi.advanceTimersByTime(1))
    await flush()
    expect(hydrate).toHaveBeenCalledTimes(pausedHydrationCount + 1)
    expect(sources).toHaveLength(1)
  })

  it('keeps retained EventSource events and native recovery active while interval polling is paused', async () => {
    vi.useFakeTimers()
    const { hydrate, sources, result } = renderLive()
    await flush()
    const source = sources[0]
    act(() => source?.error())
    act(() => vi.advanceTimersByTime(1_000))
    act(() => result.current.togglePolling())
    const hydrationCount = hydrate.mock.calls.length

    act(() => source?.emit('log_event', eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1), 'v1:1.0.0'))
    await flush()

    expect(hydrate).toHaveBeenCalledTimes(hydrationCount + 1)
    expect(result.current.liveRequestIds).toEqual([REQUEST_A])
    act(() => source?.open())
    expect(result.current.state).toBe('connected')
    expect(result.current.pollingEnabled).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
    expect(sources).toHaveLength(1)
    expect(source?.closed).toBe(false)
  })

  it('recovers replay gaps authoritatively while fallback interval polling is paused', async () => {
    vi.useFakeTimers()
    const { hydrate, sources, result } = renderLive()
    await flush()
    act(() => sources[0]?.error())
    act(() => vi.advanceTimersByTime(1_000))
    act(() => result.current.togglePolling())
    const hydrationCount = hydrate.mock.calls.length

    act(() =>
      sources[0]?.emit(
        'replay_gap',
        JSON.stringify({
          channel: 'requests',
          fromSequence: 3,
          toSequence: 4,
          recovery: { endpoint: '/api/logs/requests', cursor: null }
        }),
        'v1:4.0.0'
      )
    )
    expect(result.current.state).toBe('gap')
    await flush()

    expect(hydrate).toHaveBeenCalledTimes(hydrationCount + 1)
    expect(result.current.state).toBe('connected')
    expect(result.current.pollingEnabled).toBe(false)
  })

  it('uses polling when the dedicated stream cannot be constructed and never overlaps hydration', async () => {
    vi.useFakeTimers()
    let resolveHydration: (() => void) | undefined
    const hydrate = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveHydration = resolve
        })
    )
    const result = renderHook(() =>
      useLogsLiveRecovery({
        enabled: true,
        search: {},
        hydrate,
        eventSourceFactory: unsupportedEventSourceFactory
      })
    )

    expect(result.result.current.state).toBe('polling')
    act(() => vi.advanceTimersByTime(15_000))
    expect(hydrate).toHaveBeenCalledTimes(1)
    act(() => resolveHydration?.())
    await flush()
    expect(hydrate).toHaveBeenCalledTimes(2)
  })

  it('refetches authoritatively after a replay gap', async () => {
    const { hydrate, sources, result } = renderLive()
    await flush()
    act(() => sources[0]?.open())
    act(() =>
      sources[0]?.emit(
        'replay_gap',
        JSON.stringify({
          channel: 'requests',
          fromSequence: 3,
          toSequence: 4,
          recovery: { endpoint: '/api/logs/requests', cursor: null }
        }),
        'v1:4.0.0'
      )
    )
    expect(result.current.state).toBe('gap')
    await flush()
    expect(hydrate).toHaveBeenCalledTimes(1)
    expect(result.current.state).toBe('connected')
  })

  it('closes and reopens on filter change with the last received cursor and fresh dedupe state', async () => {
    const { hydrate, rerender, result, sources } = renderLive({ search: { provider: 'reserve-a' } })
    await flush()
    act(() =>
      sources[0]?.emit('log_event', eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1), 'v1:1.0.0')
    )
    await flush()
    rerender({ search: { provider: 'reserve-b' } })

    expect(sources[0]?.closed).toBe(true)
    expect(sources[1]?.url).toBe(
      '/api/logs/events?channel=requests&channel=operations&filter=provider%3Areserve-b&cursor=v1%3A1.0.0'
    )
    act(() =>
      sources[1]?.emit('log_event', eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1), 'v1:1.0.0')
    )
    await flush()
    expect(result.current.liveRequestIds).toEqual([REQUEST_A])
    expect(hydrate).toHaveBeenCalledTimes(2)
  })

  /**
   * Regression guard for the `/logs` render loop (React error #185).
   *
   * Every collection this hook returns must keep a stable reference across a
   * render that changed nothing. `auditEntries` returned a fresh `[]` literal
   * when audit was disabled, which invalidated the whole ledger memo chain
   * (auditEntries -> filteredAuditEntries -> mergedRows -> categoryRows) on
   * every render. That handed `<BarChart>` a new `data` array each render, so
   * recharts' ChartDataContextProvider re-dispatched `setChartData`, and
   * react-redux v9's synchronous `defaultNoopBatch` notified subscribers
   * inline — re-rendering the tree that minted the next `[]`. Self-sustaining
   * until React's 50-nested-update ceiling tripped the error boundary.
   *
   * This asserts the mechanism (referential stability), not the symptom, so it
   * fails on the fresh-literal pattern regardless of which consumer breaks.
   */
  it('returns referentially stable collections across renders when audit is disabled', async () => {
    const { rerender, result } = renderLive({ auditEnabled: false })
    await flush()

    const first = {
      auditEntries: result.current.auditEntries,
      liveRequestIds: result.current.liveRequestIds,
      requestUpdates: result.current.requestUpdates,
      excludedRequestIds: result.current.excludedRequestIds
    }

    // A no-op re-render with identical inputs must not mint new collections.
    rerender({ search: { model: 'Qwen3', provider: 'reserve-a' } })
    await flush()

    expect(result.current.auditEntries).toBe(first.auditEntries)
    expect(result.current.liveRequestIds).toBe(first.liveRequestIds)
    expect(result.current.requestUpdates).toBe(first.requestUpdates)
    expect(result.current.excludedRequestIds).toBe(first.excludedRequestIds)
  })

  it('keeps auditEntries referentially stable while live request events arrive', async () => {
    const { result, sources } = renderLive({ auditEnabled: false })
    await flush()
    const initialAuditEntries = result.current.auditEntries

    // Request traffic must not invalidate the (disabled) audit collection.
    act(() =>
      sources[0]?.emit('log_event', eventData(REQUEST_A, '00000000-0000-4000-8000-000000000003', 1), 'v1:1.0.0')
    )
    await flush()

    expect(result.current.auditEntries).toBe(initialAuditEntries)
    expect(result.current.auditEntries).toEqual([])
  })
})
