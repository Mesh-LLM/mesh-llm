import { useMemo } from 'react'
import type {
  LogArtifact,
  LogEventKind,
  LogLifecycleEvent,
  LogProxyAttempt,
  LogRequest
} from '@/features/logs/api/schemas'
import { LogDiagnosticArtifactList } from '@/features/logs/components/LogDiagnosticArtifactList'
import {
  type LogRequestDiagnosticEvidence,
  LogRequestDiagnosticSummary
} from '@/features/logs/components/LogRequestDiagnosticSummary'
import { artifactMatchesTab, isErrorEvent } from '@/features/logs/lib/log-request-details'
import { LogRequestEvidenceTimeline } from './LogRequestEvidenceTimeline'

export type LogRequestDiagnosticsProps = {
  readonly request: LogRequest | undefined
  readonly events: readonly LogLifecycleEvent[] | undefined
  readonly attempts: readonly LogProxyAttempt[] | undefined
  readonly artifacts: readonly LogArtifact[] | undefined
  readonly requestLoading: boolean
  readonly requestError: boolean
  readonly eventsLoading: boolean
  readonly eventsError: boolean
  readonly attemptsLoading: boolean
  readonly attemptsError: boolean
  readonly artifactsLoading: boolean
  readonly artifactsError: boolean
}

const interruptedEventKinds: readonly LogEventKind[] = ['rejected', 'cancelled', 'dropped']

function isDiagnosticEvent(event: LogLifecycleEvent): boolean {
  return isErrorEvent(event) || interruptedEventKinds.includes(event.kind)
}

type RequestQueryStateKind = 'loading' | 'error' | 'empty'
const requestQueryCopy = {
  loading: 'Loading request diagnostics.',
  error: 'Request diagnostics could not be loaded from the local log service.',
  empty: 'The request summary is unavailable, so terminal diagnostics cannot be determined.'
} satisfies Record<RequestQueryStateKind, string>

function RequestQueryState({ state }: { readonly state: RequestQueryStateKind }) {
  return (
    <div
      className="min-w-0 break-words rounded-[var(--radius)] border border-border bg-panel px-[var(--panel-x)] py-[var(--panel-y)] type-body text-fg-dim"
      data-diagnostic-state={state}
      role={state === 'error' ? 'alert' : 'status'}
    >
      {requestQueryCopy[state]}
    </div>
  )
}

function QueryNotice({
  label,
  loading,
  error
}: {
  readonly label: string
  readonly loading: boolean
  readonly error: boolean
}) {
  if (!loading && !error) return null
  return (
    <p className="min-w-0 break-words type-caption text-fg-dim" role={error ? 'alert' : 'status'}>
      {error ? `${label} could not be loaded.` : `Loading ${label}.`}
    </p>
  )
}

export function LogRequestDiagnostics(props: LogRequestDiagnosticsProps) {
  const diagnosticEvents = useMemo(() => (props.events ?? []).filter(isDiagnosticEvent), [props.events])
  const errorArtifacts = useMemo(
    () => (props.artifacts ?? []).filter((artifact) => artifactMatchesTab(artifact.kind, 'errors')),
    [props.artifacts]
  )
  const eventsReady = !props.eventsLoading && !props.eventsError
  const attemptsReady = !props.attemptsLoading && !props.attemptsError
  const artifactsReady = !props.artifactsLoading && !props.artifactsError
  const evidenceComplete = eventsReady && attemptsReady && artifactsReady
  const hasErrorEvidence =
    diagnosticEvents.length > 0 ||
    (props.attempts ?? []).some((attempt) => attempt.statusCode !== undefined && attempt.statusCode >= 400) ||
    errorArtifacts.length > 0

  if (props.requestError) return <RequestQueryState state="error" />
  if (props.requestLoading) return <RequestQueryState state="loading" />
  if (props.request === undefined) return <RequestQueryState state="empty" />

  const diagnosticEvidence: LogRequestDiagnosticEvidence = {
    artifacts: artifactsReady ? (props.artifacts ?? []) : undefined,
    attemptCount: attemptsReady ? (props.attempts ?? []).length : undefined,
    diagnosticMarkerCount: eventsReady && artifactsReady ? diagnosticEvents.length + errorArtifacts.length : undefined,
    evidenceComplete,
    hasErrorEvidence
  }
  const successful =
    props.request.outcome === 'completed' && diagnosticEvidence.evidenceComplete && !diagnosticEvidence.hasErrorEvidence

  return (
    <section aria-label="Request diagnostics" className="min-w-0 space-y-[var(--shell-normal)]">
      <LogRequestDiagnosticSummary evidence={diagnosticEvidence} request={props.request} successful={successful} />

      {successful ? null : (
        <>
          {!eventsReady || !attemptsReady ? (
            <div className="grid min-w-0 gap-2 rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 px-[var(--panel-x)] py-[var(--panel-y)] sm:grid-cols-2">
              <QueryNotice
                error={props.eventsError}
                label="Diagnostic lifecycle evidence"
                loading={props.eventsLoading}
              />
              <QueryNotice error={props.attemptsError} label="Routing attempts" loading={props.attemptsLoading} />
            </div>
          ) : null}
          <LogRequestEvidenceTimeline
            ariaLabel="Ordered diagnostic evidence"
            attemptEmptyMessage={
              attemptsReady ? 'No retry or proxy attempts were retained for this request.' : undefined
            }
            attempts={attemptsReady ? (props.attempts ?? []) : []}
            eventEmptyMessage={eventsReady ? 'No failed or interrupted lifecycle markers were retained.' : undefined}
            events={eventsReady ? diagnosticEvents : []}
          />
          <LogDiagnosticArtifactList
            artifacts={errorArtifacts}
            error={props.artifactsError}
            loading={props.artifactsLoading}
          />
        </>
      )}
    </section>
  )
}
