import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { LogsApiClient } from '@/features/logs/api/client'
import { LogAuditCursor, LogReplayCursor, type LogReplayChannel } from '@/features/logs/api/ids'
import { parseLogsSseFrame, type LogsSseFilter } from '@/features/logs/api/sse'
import type { LogAuditEntry, LogRequest } from '@/features/logs/api/schemas'
import { sortByOccurredAt } from '@/features/logs/lib/log-instant'
import { resolveRelativeTime, type LogsLedgerSearch } from '@/features/logs/lib/log-search'

const POLL_INTERVAL_MS = 5_000
const FALLBACK_DELAY_MS = 1_000
const DEFAULT_CHANNELS: readonly LogReplayChannel[] = ['requests', 'operations']
/**
 * Returned when audit streaming is disabled. Must be a shared module-level
 * value, not a fresh `[]` literal: a new identity per render invalidates the
 * ledger memo chain, which hands the events chart a new `data` array on every
 * render and drives recharts into a synchronous re-dispatch loop until React's
 * nested-update ceiling trips the `/logs` error boundary.
 */
const EMPTY_AUDIT_ENTRIES: readonly LogAuditEntry[] = []

export type LogsLiveConnectionState = 'connected' | 'reconnecting' | 'polling' | 'gap' | 'stale'

type LogsEventSource = {
  close: () => void
  onopen: ((event: Event) => void) | null
  onerror: ((event: Event) => void) | null
  addEventListener: (type: string, listener: (event: MessageEvent<string>) => void) => void
}

export type LogsEventSourceFactory = (url: string) => LogsEventSource

export type LogsLiveRecoveryOptions = {
  readonly enabled: boolean
  readonly search: LogsLedgerSearch
  readonly hydrate: () => Promise<unknown>
  readonly authoritativeSnapshot?: unknown
  readonly auditEnabled?: boolean
  readonly hydrateAudit?: () => Promise<unknown>
  readonly auditCursor?: LogAuditCursor
  readonly channels?: readonly LogReplayChannel[]
  readonly eventSourceFactory?: LogsEventSourceFactory
}

export type LogsLiveRecovery = {
  readonly state: LogsLiveConnectionState
  readonly liveRequestIds: readonly string[]
  readonly requestUpdates: readonly LogRequest[]
  readonly excludedRequestIds: readonly string[]
  readonly auditEntries: readonly LogAuditEntry[]
  readonly pollingEnabled: boolean
  readonly togglePolling: () => void
}

type LiveRequest = {
  readonly requestId: string
  readonly occurredAt: string
  readonly request?: LogRequest
  readonly included: boolean
  readonly revision: number
}

type LiveRequestState = {
  readonly subscriptionKey: string
  readonly entries: readonly LiveRequest[]
}

function eventSourceFactory(url: string): LogsEventSource {
  return new EventSource(url)
}

function parseReplayCursor(value: string | undefined) {
  if (!value) return undefined
  try {
    return LogReplayCursor.parse(value)
  } catch {
    return undefined
  }
}

function activeFilterScope(
  from: string | undefined,
  to: string | undefined,
  timeRange: LogsLedgerSearch['timeRange'],
  model: string | undefined,
  provider: string | undefined,
  engine: string | undefined,
  route: string | undefined,
  source: string | undefined,
  outcome: string | undefined
): string[] {
  const entries: Array<[string, string | undefined]> = [
    ['from', from],
    ['to', to],
    ['timeRange', timeRange],
    ['model', model],
    ['provider', provider],
    ['engine', engine],
    ['route', route],
    ['source', source],
    ['outcome', outcome]
  ]
  return entries.flatMap(([key, value]) => (value ? [`${key}:${value}`] : []))
}

function streamFilters(
  from: string | undefined,
  to: string | undefined,
  model: string | undefined,
  provider: string | undefined,
  engine: string | undefined,
  route: string | undefined,
  outcome: string | undefined
): LogsSseFilter[] {
  const entries: Array<[LogsSseFilter['key'], string | undefined]> = [
    ['from', from],
    ['to', to],
    ['model', model],
    ['provider', provider],
    ['engine', engine],
    ['route', route],
    ['outcome', outcome]
  ]
  return entries.flatMap(([key, value]) => (value ? [{ key, value }] : []))
}

function subscriptionKey(
  channels: readonly LogReplayChannel[],
  filterScope: readonly string[],
  replayCursor: string | undefined
) {
  return `${channels.join(',')}|${filterScope.join('|')}|${replayCursor ?? ''}`
}

function mergeLiveRequests(current: readonly LiveRequest[], next: LiveRequest): LiveRequest[] {
  return sortByOccurredAt([...current.filter((entry) => entry.requestId !== next.requestId), next]).slice(-32)
}

function mergeAuditEntries(current: readonly LogAuditEntry[], next: LogAuditEntry): LogAuditEntry[] {
  return sortByOccurredAt([...current.filter((entry) => entry.entryId !== next.entryId), next]).slice(-64)
}

function requestMatchesSearch(request: LogRequest, search: LogsLedgerSearch) {
  const bounds = search.timeRange ? resolveRelativeTime(search.timeRange) : { from: search.from, to: search.to }
  const createdAt = Date.parse(request.createdAt)
  const from = bounds?.from ? Date.parse(bounds.from) : undefined
  const to = bounds?.to ? Date.parse(bounds.to) : undefined
  if (from !== undefined && createdAt < from) return false
  if (to !== undefined && createdAt > to) return false
  if (search.model && request.model !== search.model) return false
  if (search.provider && request.provider !== search.provider) return false
  if (search.engine && request.engine !== search.engine) return false
  if (search.route && request.route !== search.route) return false
  if (search.source && request.source !== search.source) return false
  if (search.outcome && request.outcome !== search.outcome) return false
  return true
}

function combinedConnectionState(
  lifecycle: LogsLiveConnectionState,
  audit: LogsLiveConnectionState,
  lifecycleEnabled: boolean,
  auditEnabled: boolean
): LogsLiveConnectionState {
  const states = [
    ...(lifecycleEnabled ? [lifecycle] : []),
    ...(auditEnabled ? [audit] : [])
  ] satisfies LogsLiveConnectionState[]
  if (states.length === 0) return 'reconnecting'
  for (const candidate of ['stale', 'gap', 'polling', 'reconnecting'] as const) {
    if (states.includes(candidate)) return candidate
  }
  return 'connected'
}

export function useLogsLiveRecovery({
  enabled,
  search,
  hydrate,
  authoritativeSnapshot,
  auditEnabled = false,
  hydrateAudit = hydrate,
  auditCursor,
  channels = DEFAULT_CHANNELS,
  eventSourceFactory: createEventSource = eventSourceFactory
}: LogsLiveRecoveryOptions): LogsLiveRecovery {
  const [lifecycleState, setLifecycleState] = useState<LogsLiveConnectionState>('reconnecting')
  const [auditState, setAuditState] = useState<LogsLiveConnectionState>('reconnecting')
  const [liveRequests, setLiveRequests] = useState<LiveRequestState>({ subscriptionKey: '', entries: [] })
  const [liveAuditEntries, setLiveAuditEntries] = useState<readonly LogAuditEntry[]>([])
  const [pollingEnabled, setPollingEnabled] = useState(true)
  const pollingEnabledRef = useRef(true)
  const sequenceByChannelRef = useRef(new Map<LogReplayChannel, number>())
  const eventIdsRef = useRef(new Set<string>())
  const requestIdsRef = useRef(new Set<string>())
  const liveRevisionRef = useRef(0)
  const hydrateInFlightRef = useRef(false)
  const hydratePendingRef = useRef(false)
  const hydratePendingClearGapRef = useRef(false)
  const latestCursorRef = useRef<LogReplayCursor | LogAuditCursor | undefined>(undefined)
  const latestAuditCursorRef = useRef<LogAuditCursor | undefined>(undefined)
  const auditSequenceRef = useRef<bigint>(0n)
  const auditHydrateInFlightRef = useRef(false)
  const auditHydratePendingRef = useRef(false)
  const auditHydratePendingClearGapRef = useRef(false)
  const hydrateAuditRef = useRef(hydrateAudit)
  const searchRef = useRef(search)
  const restoredCursorValueRef = useRef<string | undefined>(undefined)
  const previousAuthoritativeSnapshotRef = useRef(authoritativeSnapshot)

  useEffect(() => {
    hydrateAuditRef.current = hydrateAudit
  }, [hydrateAudit])

  useEffect(() => {
    searchRef.current = search
  }, [search])

  useEffect(() => {
    const previous = previousAuthoritativeSnapshotRef.current
    previousAuthoritativeSnapshotRef.current = authoritativeSnapshot
    if (previous === authoritativeSnapshot) return
    const authoritativeRevision = liveRevisionRef.current
    setLiveRequests((current) => ({
      subscriptionKey: current.subscriptionKey,
      entries: current.entries.filter((entry) => entry.request === undefined || entry.revision > authoritativeRevision)
    }))
  }, [authoritativeSnapshot])

  useEffect(() => {
    if (
      auditCursor &&
      (!latestAuditCursorRef.current || auditCursor.sequence() > latestAuditCursorRef.current.sequence())
    ) {
      latestAuditCursorRef.current = auditCursor
      auditSequenceRef.current = auditCursor.sequence()
    }
  }, [auditCursor])

  const togglePolling = useCallback(() => {
    setPollingEnabled((current) => {
      const next = !current
      pollingEnabledRef.current = next
      return next
    })
  }, [])

  // Presets are deliberately not serialized as terminal SSE bounds. A `to` bound
  // captured when this hook mounted would reject every later event. Hydration
  // resolves the rolling preset for each request; explicit deep-link bounds stay
  // fixed so reconnects retain the user's exact historical scope.
  const streamTimeBounds = useMemo(() => ({ from: search.from, to: search.to }), [search.from, search.to])
  const filterScope = useMemo(
    () =>
      activeFilterScope(
        streamTimeBounds.from,
        streamTimeBounds.to,
        search.timeRange,
        search.model,
        search.provider,
        search.engine,
        search.route,
        search.source,
        search.outcome
      ),
    [
      streamTimeBounds.from,
      streamTimeBounds.to,
      search.timeRange,
      search.model,
      search.provider,
      search.engine,
      search.route,
      search.source,
      search.outcome
    ]
  )

  const key = subscriptionKey(channels, filterScope, search.replayCursor)
  const subscriptionFilters = useMemo(
    () =>
      streamFilters(
        streamTimeBounds.from,
        streamTimeBounds.to,
        search.model,
        search.provider,
        search.engine,
        search.route,
        search.outcome
      ),
    [
      streamTimeBounds.from,
      streamTimeBounds.to,
      search.model,
      search.provider,
      search.engine,
      search.route,
      search.outcome
    ]
  )

  useEffect(() => {
    if (!enabled) return

    let disposed = false
    let source: LogsEventSource | undefined
    let pollingTimer: number | undefined
    let fallbackTimer: number | undefined

    sequenceByChannelRef.current = new Map()
    eventIdsRef.current = new Set()
    requestIdsRef.current = new Set()
    liveRevisionRef.current = 0
    if (restoredCursorValueRef.current !== search.replayCursor) {
      restoredCursorValueRef.current = search.replayCursor
      latestCursorRef.current = parseReplayCursor(search.replayCursor)
    }

    const clearPolling = () => {
      if (pollingTimer === undefined) return
      window.clearInterval(pollingTimer)
      pollingTimer = undefined
    }

    const clearFallback = () => {
      if (fallbackTimer === undefined) return
      window.clearTimeout(fallbackTimer)
      fallbackTimer = undefined
    }

    const closeSource = () => {
      if (!source) return
      source.onopen = null
      source.onerror = null
      source.close()
      source = undefined
    }

    const hydrateAuthoritatively = (clearGap: boolean) => {
      if (disposed) return
      if (hydrateInFlightRef.current) {
        hydratePendingRef.current = true
        hydratePendingClearGapRef.current ||= clearGap
        return
      }
      const authoritativeRevision = liveRevisionRef.current
      hydrateInFlightRef.current = true
      void Promise.resolve(hydrate())
        .then(() => {
          if (!disposed) {
            setLiveRequests((current) => ({
              subscriptionKey: current.subscriptionKey,
              entries: current.entries.filter(
                (entry) => entry.request === undefined || entry.revision > authoritativeRevision
              )
            }))
          }
          if (!disposed && clearGap) setLifecycleState(source ? 'connected' : 'polling')
        })
        .catch(() => {
          if (!disposed) setLifecycleState('stale')
        })
        .finally(() => {
          hydrateInFlightRef.current = false
          if (!disposed && hydratePendingRef.current) {
            hydratePendingRef.current = false
            const pendingClearGap = hydratePendingClearGapRef.current
            hydratePendingClearGapRef.current = false
            hydrateAuthoritatively(pendingClearGap)
          }
        })
    }

    const startPolling = () => {
      if (pollingTimer !== undefined) return
      setLifecycleState('polling')
      pollingTimer = window.setInterval(() => {
        if (pollingEnabledRef.current) hydrateAuthoritatively(false)
      }, POLL_INTERVAL_MS)
    }

    const queuePollingFallback = () => {
      if (fallbackTimer !== undefined) return
      setLifecycleState('reconnecting')
      fallbackTimer = window.setTimeout(() => {
        fallbackTimer = undefined
        startPolling()
      }, FALLBACK_DELAY_MS)
    }

    const acceptEvent = (event: MessageEvent<string>) => {
      if (disposed) return
      try {
        const frame = parseLogsSseFrame({ event: event.type, lastEventId: event.lastEventId, data: event.data })
        latestCursorRef.current = frame.cursor
        if (frame.type === 'replay_gap') {
          setLifecycleState('gap')
          hydrateAuthoritatively(true)
          return
        }
        if (frame.type === 'stream_error') {
          setLifecycleState('stale')
          hydrateAuthoritatively(true)
          return
        }
        if (frame.type !== 'log_event') {
          setLifecycleState('stale')
          hydrateAuthoritatively(true)
          return
        }

        const channelSequence = sequenceByChannelRef.current.get(frame.event.channel)
        if (channelSequence !== undefined && frame.event.sequence <= channelSequence) return
        sequenceByChannelRef.current.set(frame.event.channel, frame.event.sequence)

        const eventId = frame.event.eventId.toString()
        if (eventIdsRef.current.has(eventId)) return
        eventIdsRef.current.add(eventId)

        const requestId = frame.event.requestId.toString()
        requestIdsRef.current.add(requestId)
        const revision = ++liveRevisionRef.current
        const projectedRequest = frame.event.request
        if (projectedRequest) {
          setLiveRequests((current) => ({
            subscriptionKey: key,
            entries: mergeLiveRequests(current.subscriptionKey === key ? current.entries : [], {
              requestId,
              occurredAt: frame.event.occurredAt,
              request: projectedRequest,
              included: requestMatchesSearch(projectedRequest, searchRef.current),
              revision
            })
          }))
        } else {
          // Compatibility with older daemons whose additive SSE projection is
          // absent. Current servers never take this full-ledger recovery path.
          setLiveRequests((current) => ({
            subscriptionKey: key,
            entries: mergeLiveRequests(current.subscriptionKey === key ? current.entries : [], {
              requestId,
              occurredAt: frame.event.occurredAt,
              included: true,
              revision
            })
          }))
          hydrateAuthoritatively(false)
        }
      } catch {
        setLifecycleState('stale')
        hydrateAuthoritatively(true)
      }
    }

    const url = new LogsApiClient().logsEventSourceUrl({
      channels,
      filters: subscriptionFilters,
      cursor: latestCursorRef.current instanceof LogReplayCursor ? latestCursorRef.current : undefined
    })
    try {
      const connectedSource = createEventSource(url)
      source = connectedSource
      connectedSource.onopen = () => {
        if (disposed) return
        clearFallback()
        clearPolling()
        setLifecycleState('connected')
      }
      connectedSource.onerror = () => {
        if (!disposed) queuePollingFallback()
      }
      connectedSource.addEventListener('log_event', acceptEvent)
      connectedSource.addEventListener('replay_gap', acceptEvent)
      connectedSource.addEventListener('stream_error', acceptEvent)
    } catch {
      startPolling()
    }

    return () => {
      disposed = true
      clearFallback()
      clearPolling()
      closeSource()
    }
  }, [channels, createEventSource, enabled, filterScope, hydrate, key, search.replayCursor, subscriptionFilters])

  useEffect(() => {
    if (!auditEnabled) return

    let disposed = false
    let source: LogsEventSource | undefined
    let reconciliationTimer: number | undefined
    let fallbackTimer: number | undefined

    const clearReconciliation = () => {
      if (reconciliationTimer === undefined) return
      window.clearInterval(reconciliationTimer)
      reconciliationTimer = undefined
    }
    const clearFallback = () => {
      if (fallbackTimer === undefined) return
      window.clearTimeout(fallbackTimer)
      fallbackTimer = undefined
    }
    const closeSource = () => {
      if (!source) return
      source.onopen = null
      source.onerror = null
      source.close()
      source = undefined
    }
    const hydrateAuditAuthoritatively = (clearGap: boolean) => {
      if (disposed) return
      if (auditHydrateInFlightRef.current) {
        auditHydratePendingRef.current = true
        auditHydratePendingClearGapRef.current ||= clearGap
        return
      }
      auditHydrateInFlightRef.current = true
      void Promise.resolve(hydrateAuditRef.current())
        .then(() => {
          if (!disposed && clearGap) setAuditState(source ? 'connected' : 'polling')
        })
        .catch(() => {
          if (!disposed) setAuditState('stale')
        })
        .finally(() => {
          auditHydrateInFlightRef.current = false
          if (!disposed && auditHydratePendingRef.current) {
            auditHydratePendingRef.current = false
            const pendingClearGap = auditHydratePendingClearGapRef.current
            auditHydratePendingClearGapRef.current = false
            hydrateAuditAuthoritatively(pendingClearGap)
          }
        })
    }
    const startReconciliation = () => {
      if (reconciliationTimer !== undefined) return
      reconciliationTimer = window.setInterval(() => {
        if (pollingEnabledRef.current) hydrateAuditAuthoritatively(false)
      }, POLL_INTERVAL_MS)
    }
    const startPolling = () => {
      setAuditState('polling')
      startReconciliation()
    }
    const queuePollingFallback = () => {
      if (fallbackTimer !== undefined) return
      setAuditState('reconnecting')
      fallbackTimer = window.setTimeout(() => {
        fallbackTimer = undefined
        startPolling()
      }, FALLBACK_DELAY_MS)
    }
    const acceptAuditEvent = (event: MessageEvent<string>) => {
      if (disposed) return
      try {
        const frame = parseLogsSseFrame({ event: event.type, lastEventId: event.lastEventId, data: event.data })
        if (!(frame.cursor instanceof LogAuditCursor)) {
          setAuditState('stale')
          hydrateAuditAuthoritatively(true)
          return
        }
        latestAuditCursorRef.current = frame.cursor
        if (frame.type === 'audit_gap') {
          setAuditState('gap')
          hydrateAuditAuthoritatively(true)
          return
        }
        if (frame.type === 'stream_error') {
          setAuditState('stale')
          hydrateAuditAuthoritatively(true)
          return
        }
        if (frame.type !== 'audit_entry') {
          setAuditState('stale')
          hydrateAuditAuthoritatively(true)
          return
        }
        const sequence = BigInt(frame.entry.sequence)
        if (sequence <= auditSequenceRef.current) return
        auditSequenceRef.current = sequence
        setLiveAuditEntries((current) => mergeAuditEntries(current, frame.entry))
      } catch {
        setAuditState('stale')
        hydrateAuditAuthoritatively(true)
      }
    }

    const url = new LogsApiClient().logsEventSourceUrl({
      channels: [],
      audit: { cursor: latestAuditCursorRef.current }
    })
    try {
      const connectedSource = createEventSource(url)
      source = connectedSource
      connectedSource.onopen = () => {
        if (disposed) return
        clearFallback()
        clearReconciliation()
        setAuditState('connected')
      }
      connectedSource.onerror = () => {
        if (!disposed) queuePollingFallback()
      }
      connectedSource.addEventListener('audit_entry', acceptAuditEvent)
      connectedSource.addEventListener('replay_gap', acceptAuditEvent)
      connectedSource.addEventListener('stream_error', acceptAuditEvent)
    } catch {
      startPolling()
    }

    return () => {
      disposed = true
      clearFallback()
      clearReconciliation()
      closeSource()
    }
  }, [auditEnabled, createEventSource])

  const state = combinedConnectionState(lifecycleState, auditState, enabled, auditEnabled)
  const activeLiveRequests = useMemo(
    () => (enabled && liveRequests.subscriptionKey === key ? liveRequests.entries : []),
    [enabled, key, liveRequests]
  )
  const liveRequestIds = useMemo(() => activeLiveRequests.map((entry) => entry.requestId), [activeLiveRequests])
  const requestUpdates = useMemo(
    () => activeLiveRequests.flatMap((entry) => (entry.included && entry.request ? [entry.request] : [])),
    [activeLiveRequests]
  )
  const excludedRequestIds = useMemo(
    () => activeLiveRequests.flatMap((entry) => (entry.included ? [] : [entry.requestId])),
    [activeLiveRequests]
  )

  return {
    state,
    liveRequestIds,
    requestUpdates,
    excludedRequestIds,
    auditEntries: auditEnabled ? liveAuditEntries : EMPTY_AUDIT_ENTRIES,
    pollingEnabled,
    togglePolling
  }
}
