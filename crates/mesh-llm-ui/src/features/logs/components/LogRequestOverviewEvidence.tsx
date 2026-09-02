import { Route, Workflow } from 'lucide-react'
import type { LogLifecycleEvent, LogProxyAttempt } from '@/features/logs/api/schemas'
import {
  attemptDurationMs,
  formatDurationMs,
  formatTimestamp,
  machineValue,
  type RetainedQueryState
} from '@/features/logs/components/LogRequestOverviewDerivations'
import { LogRequestLifecycleStrip } from '@/features/logs/components/LogRequestLifecycleStrip'
import { LogRequestOverviewPanel } from '@/features/logs/components/LogRequestOverviewPanel'
import { compareLogInstants } from '@/features/logs/lib/log-instant'

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
  return <LogRequestLifecycleStrip events={query.items} />
}

export function LogRequestLifecycleOverview({ events }: { readonly events: RetainedQueryState<LogLifecycleEvent> }) {
  return (
    <LogRequestOverviewPanel ariaLabel="Request lifecycle" icon={Workflow} title="Lifecycle timeline">
      <LifecycleEvents query={events} />
    </LogRequestOverviewPanel>
  )
}

export function LogRequestRoutingOverview({ attempts }: { readonly attempts: RetainedQueryState<LogProxyAttempt> }) {
  return (
    <LogRequestOverviewPanel
      ariaLabel="Request routing attempts"
      description="Retained targets, providers, engines, statuses, and timing for each attempt."
      icon={Route}
      title="Routing attempts"
    >
      <RoutingAttempts query={attempts} />
    </LogRequestOverviewPanel>
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
