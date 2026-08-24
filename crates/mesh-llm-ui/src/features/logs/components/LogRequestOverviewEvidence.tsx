import { Route, Workflow } from 'lucide-react'
import type { LogEventKind, LogLifecycleEvent, LogProxyAttempt } from '@/features/logs/api/schemas'
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

function lifecycleLabel(kind: LogLifecycleEvent['kind']): string {
  return kind
    .split('_')
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(' ')
}

function lifecycleElapsed(events: readonly LogLifecycleEvent[], index: number): string | undefined {
  if (index === 0) return undefined
  const previous = events[index - 1]
  const current = events[index]
  if (previous === undefined || current === undefined) return undefined
  const elapsed = Date.parse(current.occurredAt) - Date.parse(previous.occurredAt)
  return Number.isFinite(elapsed) && elapsed >= 0 ? `+${formatDurationMs(elapsed)}` : undefined
}

const LIFECYCLE_TONES: Readonly<Record<LogEventKind, 'accent' | 'bad' | 'good'>> = {
  admitted: 'accent',
  route_selected: 'accent',
  attempt_started: 'accent',
  attempt_completed: 'good',
  attempt_failed: 'bad',
  backend_stream_first_item: 'accent',
  stream_started: 'accent',
  stream_chunk: 'accent',
  stream_completed: 'good',
  usage_recorded: 'accent',
  stream_error: 'bad',
  audit_error: 'bad',
  completed: 'good',
  failed: 'bad',
  rejected: 'bad',
  cancelled: 'bad',
  dropped: 'bad'
}

function lifecycleTone(kind: LogLifecycleEvent['kind']): 'accent' | 'bad' | 'good' {
  return LIFECYCLE_TONES[kind]
}

const LIFECYCLE_NODE_CLASS = {
  accent:
    'border-[color:color-mix(in_oklab,var(--color-accent)_48%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-accent)_16%,var(--color-panel))] text-accent',
  bad: 'border-[color:color-mix(in_oklab,var(--color-bad)_48%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-bad)_14%,var(--color-panel))] text-bad',
  good: 'border-[color:color-mix(in_oklab,var(--color-good)_48%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-good)_14%,var(--color-panel))] text-good'
} as const

function lifecycleEvidence(event: LogLifecycleEvent): string | undefined {
  const tokenEvidence = tokenUsageEntries(event).at(-1)
  const values = [
    event.attemptId ? `attempt ${event.attemptId}` : undefined,
    event.statusCode === undefined ? undefined : `HTTP ${event.statusCode}`,
    event.durationMs === undefined ? undefined : formatDurationMs(event.durationMs),
    tokenEvidence ? `${tokenEvidence.value} ${tokenEvidence.label.toLowerCase()}` : undefined
  ].filter((value): value is string => value !== undefined)
  return values.length === 0 ? undefined : values.join(' · ')
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
    <div className="overflow-x-auto">
      <ol aria-label="Lifecycle events" className="flex w-max min-w-full px-[var(--panel-x)] py-5">
        {events.map((event, index) => {
          const tone = lifecycleTone(event.kind)
          const elapsed = lifecycleElapsed(events, index)
          const evidence = lifecycleEvidence(event)
          return (
            <li className="min-w-40 flex-1" key={event.eventId.toString()}>
              <div className="flex items-center">
                <span
                  aria-hidden="true"
                  className={`grid size-7 shrink-0 place-items-center rounded-full border ${LIFECYCLE_NODE_CLASS[tone]}`}
                >
                  <span className="size-1.5 rounded-full bg-current" />
                </span>
                {index < events.length - 1 ? (
                  <span aria-hidden="true" className="relative h-px flex-1 bg-border">
                    {events[index + 1] ? (
                      <span className="absolute left-1/2 top-0 -translate-x-1/2 -translate-y-1/2 bg-panel px-1.5 font-mono tabular-nums type-micro text-fg-faint">
                        {lifecycleElapsed(events, index + 1)}
                      </span>
                    ) : null}
                  </span>
                ) : null}
              </div>
              <div className="mt-3 min-w-0 pr-4">
                <div className="type-panel-title text-foreground">{lifecycleLabel(event.kind)}</div>
                <code className="mt-1 block break-all font-mono type-micro text-fg-faint">{event.kind}</code>
                <time
                  className="mt-2 block font-mono tabular-nums type-caption text-fg-dim"
                  dateTime={event.occurredAt}
                >
                  {formatTimestamp(event.occurredAt)}
                </time>
                {evidence ? <p className="mt-1.5 break-words font-mono type-micro text-fg-faint">{evidence}</p> : null}
                {elapsed === undefined ? null : <span className="sr-only">Elapsed {elapsed}</span>}
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

export function LogRequestLifecycleOverview({ events }: { readonly events: RetainedQueryState<LogLifecycleEvent> }) {
  return (
    <LogRequestOverviewPanel
      ariaLabel="Request lifecycle"
      description="Retained request events, ordered from first record to terminal state."
      icon={Workflow}
      title="Lifecycle timeline"
    >
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
