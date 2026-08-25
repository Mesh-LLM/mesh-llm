import { useEffect, useRef, useState } from 'react'
import { LogsApiClient } from '@/features/logs/api/client'
import { LogAuditCursor } from '@/features/logs/api/ids'
import type { LogAuditEntry } from '@/features/logs/api/schemas'
import { parseLogsSseFrame } from '@/features/logs/api/sse'
import { sortByOccurredAt } from '@/features/logs/lib/log-instant'
import * as auditTerminal from './audit-terminal-recovery'
import type { LogsEventSourceFactory, LogsLiveConnectionState } from './use-logs-live-recovery'

const POLL_INTERVAL_MS = 5_000
const FALLBACK_DELAY_MS = 1_000
/**
 * A stable empty value prevents the disabled audit stream from invalidating
 * consumers that memoize the returned entry list by identity.
 */
const EMPTY_AUDIT_ENTRIES: readonly LogAuditEntry[] = []

type AuditEventSource = ReturnType<LogsEventSourceFactory>
type AuditHydrationRequest = { readonly kind: 'standard'; readonly clearGap: boolean } | { readonly kind: 'terminal' }
type AuditLiveRecoveryOptions = {
  readonly enabled: boolean
  readonly hydrate: () => Promise<unknown>
  readonly cursor: LogAuditCursor | undefined
  readonly pollingEnabledRef: { readonly current: boolean }
  readonly eventSourceFactory: LogsEventSourceFactory
}
function mergeAuditEntries(current: readonly LogAuditEntry[], next: LogAuditEntry): LogAuditEntry[] {
  return sortByOccurredAt([...current.filter((entry) => entry.entryId !== next.entryId), next]).slice(-64)
}

export function useAuditLiveRecovery({
  enabled,
  hydrate,
  cursor,
  pollingEnabledRef,
  eventSourceFactory
}: AuditLiveRecoveryOptions) {
  const [state, setState] = useState<LogsLiveConnectionState>('reconnecting')
  const [liveEntries, setLiveEntries] = useState<readonly LogAuditEntry[]>([])
  const [fallbackPollingActive, setFallbackPollingActive] = useState(false)
  const latestCursorRef = useRef<LogAuditCursor | undefined>(undefined)
  const sequenceRef = useRef<bigint>(0n)
  const hydrateInFlightRef = useRef(false)
  const hydratePendingRequestRef = useRef<AuditHydrationRequest | undefined>(undefined)
  const hydrateRef = useRef(hydrate)

  useEffect(() => {
    hydrateRef.current = hydrate
  }, [hydrate])

  useEffect(() => {
    if (cursor && (!latestCursorRef.current || cursor.sequence() > latestCursorRef.current.sequence())) {
      latestCursorRef.current = cursor
      sequenceRef.current = cursor.sequence()
    }
  }, [cursor])

  useEffect(() => {
    if (!enabled) return

    let disposed = false
    let source: AuditEventSource | undefined
    let reconciliationTimer: number | undefined
    let fallbackTimer: number | undefined
    let terminalRecovery: auditTerminal.AuditTerminalRecovery<AuditEventSource> | undefined

    const clearRecoveryTimers = () => {
      if (fallbackTimer !== undefined) window.clearTimeout(fallbackTimer)
      fallbackTimer = undefined
      if (reconciliationTimer !== undefined) window.clearInterval(reconciliationTimer)
      reconciliationTimer = undefined
      setFallbackPollingActive(false)
    }
    const hydrateAuthoritatively = (request: AuditHydrationRequest) => {
      if (disposed) return
      if (hydrateInFlightRef.current) {
        const pending = hydratePendingRequestRef.current
        if (request.kind === 'terminal' || pending === undefined) {
          hydratePendingRequestRef.current = request
        } else if (pending.kind === 'standard') {
          hydratePendingRequestRef.current = {
            kind: 'standard',
            clearGap: pending.clearGap || request.clearGap
          }
        }
        return
      }
      hydrateInFlightRef.current = true
      void Promise.resolve(hydrateRef.current())
        .then(() => {
          if (disposed) return
          if (request.kind === 'terminal') {
            finishTerminalHydration(true)
          } else if (request.clearGap && terminalRecovery === undefined) {
            setState(source ? 'connected' : 'polling')
          }
        })
        .catch(() => {
          if (disposed) return
          setState('stale')
          if (request.kind === 'terminal') finishTerminalHydration(false)
        })
        .finally(() => {
          if (disposed) return
          hydrateInFlightRef.current = false
          const pending = hydratePendingRequestRef.current
          hydratePendingRequestRef.current = undefined
          if (pending) hydrateAuthoritatively(pending)
        })
    }
    const startPolling = () => {
      setState(source ? 'reconnecting' : 'polling')
      if (reconciliationTimer !== undefined) return
      if (pollingEnabledRef.current) hydrateAuthoritatively({ kind: 'standard', clearGap: false })
      reconciliationTimer = window.setInterval(() => {
        if (pollingEnabledRef.current) hydrateAuthoritatively({ kind: 'standard', clearGap: false })
      }, POLL_INTERVAL_MS)
      setFallbackPollingActive(true)
    }

    function applyTerminalTransition(transition: auditTerminal.AuditTerminalRecoveryTransition<AuditEventSource>) {
      terminalRecovery = transition.recovery
      if (transition.shouldMarkStale) setState('stale')
      if (transition.shouldReconnect) {
        setState('reconnecting')
        connectAuditSource()
        return
      }
      if (transition.recovery.phase === 'failed' && source === undefined) startPolling()
    }

    function finishTerminalHydration(succeeded: boolean) {
      if (!terminalRecovery) return
      applyTerminalTransition(auditTerminal.completeAuditTerminalHydration(terminalRecovery, succeeded))
    }

    const markForReconciliation = (nextState: 'gap' | 'stale') => {
      setState(nextState)
      hydrateAuthoritatively({ kind: 'standard', clearGap: true })
    }

    const acceptAuditEvent = (connectedSource: AuditEventSource, event: MessageEvent<string>) => {
      if (disposed || source !== connectedSource) return
      try {
        const frame = parseLogsSseFrame({ event: event.type, lastEventId: event.lastEventId, data: event.data })
        if (!(frame.cursor instanceof LogAuditCursor)) {
          markForReconciliation('stale')
          return
        }
        latestCursorRef.current = frame.cursor
        if (frame.type === 'audit_gap') {
          markForReconciliation('gap')
          return
        }
        if (frame.type === 'stream_error') {
          setState('stale')
          if (frame.code === 'audit_reconcile_failed') {
            const transition = auditTerminal.beginAuditTerminalRecovery(terminalRecovery, connectedSource)
            if (transition.recovery === terminalRecovery) return
            clearRecoveryTimers()
            applyTerminalTransition(transition)
            hydrateAuthoritatively({ kind: 'terminal' })
          } else {
            hydrateAuthoritatively({ kind: 'standard', clearGap: true })
          }
          return
        }
        if (frame.type !== 'audit_entry') {
          markForReconciliation('stale')
          return
        }
        const sequence = BigInt(frame.entry.sequence)
        if (sequence <= sequenceRef.current) return
        sequenceRef.current = sequence
        setLiveEntries((current) => mergeAuditEntries(current, frame.entry))
      } catch {
        markForReconciliation('stale')
      }
    }

    function connectAuditSource(): void {
      const url = new LogsApiClient().logsEventSourceUrl({
        channels: [],
        audit: { cursor: latestCursorRef.current }
      })
      try {
        const connectedSource = eventSourceFactory(url)
        source = connectedSource
        connectedSource.onopen = () => {
          if (disposed || source !== connectedSource) return
          clearRecoveryTimers()
          if (terminalRecovery?.phase === 'replaced') terminalRecovery = undefined
          setState('connected')
        }
        connectedSource.onerror = () => {
          if (disposed || source !== connectedSource) return
          const recovery = terminalRecovery
          if (!recovery || recovery.source !== connectedSource) {
            if (fallbackTimer !== undefined) return
            setState('reconnecting')
            fallbackTimer = window.setTimeout(() => {
              fallbackTimer = undefined
              startPolling()
            }, FALLBACK_DELAY_MS)
            return
          }
          connectedSource.onopen = null
          connectedSource.onerror = null
          connectedSource.close()
          source = undefined
          applyTerminalTransition(auditTerminal.completeAuditTerminalEof(recovery))
        }
        const acceptConnectedAuditEvent = (event: MessageEvent<string>) => acceptAuditEvent(connectedSource, event)
        connectedSource.addEventListener('audit_entry', acceptConnectedAuditEvent)
        connectedSource.addEventListener('replay_gap', acceptConnectedAuditEvent)
        connectedSource.addEventListener('stream_error', acceptConnectedAuditEvent)
      } catch {
        startPolling()
      }
    }
    connectAuditSource()

    return () => {
      disposed = true
      hydrateInFlightRef.current = false
      hydratePendingRequestRef.current = undefined
      clearRecoveryTimers()
      if (source) {
        source.onopen = null
        source.onerror = null
        source.close()
        source = undefined
      }
    }
  }, [enabled, eventSourceFactory, pollingEnabledRef])

  return {
    state,
    entries: enabled ? liveEntries : EMPTY_AUDIT_ENTRIES,
    fallbackPollingActive: enabled && fallbackPollingActive
  }
}
