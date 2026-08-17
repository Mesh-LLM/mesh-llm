import { Route, Workflow } from 'lucide-react'
import type { LogLifecycleEvent, LogProxyAttempt } from '@/features/logs/api/schemas'
import { tokenUsageEntries } from '@/features/logs/lib/log-token-usage'
import {
  attemptDurationMs,
  formatDurationMs,
  formatTimestamp,
  machineValue,
  type RetainedQueryState
} from '@/features/logs/components/LogRequestOverviewDerivations'
import { LogRequestOverviewPanel } from '@/features/logs/components/LogRequestOverviewPanel'
import { compareLogInstants } from '@/features/logs/lib/log-instant'

type LogRequestOverviewEvidenceProps = {
  readonly attempts: RetainedQueryState<LogProxyAttempt>
  readonly events: RetainedQueryState<LogLifecycleEvent>
}

function EvidenceState({
  loading,
  error,
  absent,
  empty
}: {
  readonly loading: boolean
  readonly error: boolean
  readonly absent: string
  readonly empty: string
}) {
  const message = loading ? `Loading ${absent}.` : error ? `${absent} could not be loaded.` : empty
  return (
    <p className="type-body p-[var(--panel-x)] text-fg-dim" role={error ? 'alert' : 'status'}>
      {message}
    </p>
  )
}

function EventDetail({ label, value }: { readonly label: string; readonly value: string | number | undefined }) {
  if (value === undefined) return null
  return (
    <div className="min-w-0">
      <dt className="type-label text-fg-faint">{label}</dt>
      <dd className="mt-0.5 break-all font-mono type-caption text-foreground">{value}</dd>
    </div>
  )
}

function LifecycleEvents({ query }: { readonly query: RetainedQueryState<LogLifecycleEvent> }) {
  if (query.items === undefined) {
    return (
      <EvidenceState
        absent="lifecycle events"
        empty="Lifecycle events were not recorded."
        error={query.error}
        loading={query.loading}
      />
    )
  }
  if (query.items.length === 0) {
    return (
      <EvidenceState
        absent="lifecycle events"
        empty="No lifecycle events were retained for this request."
        error={false}
        loading={false}
      />
    )
  }
  const events = query.items
    .map((event, index) => ({ event, index }))
    .sort(
      (left, right) => compareLogInstants(left.event.occurredAt, right.event.occurredAt) || left.index - right.index
    )
    .map(({ event }) => event)
  return (
    <ol aria-label="Lifecycle events" className="divide-y divide-border-soft">
      {events.map((event) => (
        <li className="min-w-0 px-[var(--panel-x)] py-[var(--panel-y)]" key={event.eventId.toString()}>
          <div className="flex min-w-0 flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
            <code className="break-all font-mono type-caption text-foreground">{event.kind}</code>
            <time className="font-mono tabular-nums type-caption text-fg-dim" dateTime={event.occurredAt}>
              {formatTimestamp(event.occurredAt)}
            </time>
          </div>
          <dl className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            <EventDetail label="Attempt ID" value={event.attemptId} />
            <EventDetail label="Model" value={event.model} />
            <EventDetail label="Provider" value={event.provider} />
            <EventDetail label="Engine" value={event.engine} />
            <EventDetail label="HTTP status" value={event.statusCode} />
            <EventDetail
              label="Duration"
              value={event.durationMs === undefined ? undefined : formatDurationMs(event.durationMs)}
            />
            {tokenUsageEntries(event).map(({ label, value }) => (
              <EventDetail key={label} label={label} value={value} />
            ))}
          </dl>
        </li>
      ))}
    </ol>
  )
}

function AttemptTime({ label, value }: { readonly label: string; readonly value: string | undefined }) {
  return (
    <div className="min-w-0">
      <dt className="type-label text-fg-faint">{label}</dt>
      <dd className="mt-0.5 break-all font-mono tabular-nums type-caption text-foreground">
        {value === undefined ? 'Not recorded' : <time dateTime={value}>{formatTimestamp(value)}</time>}
      </dd>
    </div>
  )
}

function RoutingAttempts({ query }: { readonly query: RetainedQueryState<LogProxyAttempt> }) {
  if (query.items === undefined) {
    return (
      <EvidenceState
        absent="routing attempts"
        empty="Routing attempts were not recorded."
        error={query.error}
        loading={query.loading}
      />
    )
  }
  if (query.items.length === 0) {
    return (
      <EvidenceState
        absent="routing attempts"
        empty="No routing attempts were retained for this request."
        error={false}
        loading={false}
      />
    )
  }
  const attempts = query.items
    .map((attempt, index) => ({ attempt, index }))
    .sort(
      (left, right) =>
        compareLogInstants(
          left.attempt.startedAt ?? left.attempt.occurredAt,
          right.attempt.startedAt ?? right.attempt.occurredAt
        ) || left.index - right.index
    )
    .map(({ attempt }) => attempt)
  return (
    <ol aria-label="Routing attempts" className="divide-y divide-border-soft">
      {attempts.map((attempt) => (
        <li className="min-w-0 px-[var(--panel-x)] py-[var(--panel-y)]" key={attempt.attemptId}>
          <dl className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            <EventDetail label="Attempt ID" value={attempt.attemptId} />
            <EventDetail label="Target" value={attempt.target} />
            <EventDetail label="Provider" value={machineValue(attempt.provider)} />
            <EventDetail label="Engine" value={machineValue(attempt.engine)} />
            <EventDetail label="HTTP status" value={machineValue(attempt.statusCode)} />
            <AttemptTime label="Started" value={attempt.startedAt} />
            <AttemptTime label="Completed" value={attempt.completedAt} />
            <EventDetail label="Duration" value={formatDurationMs(attemptDurationMs(attempt))} />
          </dl>
        </li>
      ))}
    </ol>
  )
}

export function LogRequestOverviewEvidence({ attempts, events }: LogRequestOverviewEvidenceProps) {
  return (
    <>
      <LogRequestOverviewPanel
        ariaLabel="Request lifecycle"
        description="Canonical retained event kinds and timestamps in chronological order."
        icon={Workflow}
        title="Lifecycle events"
      >
        <LifecycleEvents query={events} />
      </LogRequestOverviewPanel>
      <LogRequestOverviewPanel
        ariaLabel="Request routing attempts"
        description="Retained targets, providers, engines, statuses, and timing for each attempt."
        icon={Route}
        title="Routing attempts"
      >
        <RoutingAttempts query={attempts} />
      </LogRequestOverviewPanel>
    </>
  )
}
