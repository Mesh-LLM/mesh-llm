import { useMemo } from 'react'
import { StatusBadge } from '@/components/ui/StatusBadge'
import type { LogProxyAttempt } from '@/features/logs/api/schemas'
import { sortByOccurredAt } from '@/features/logs/lib/log-instant'
import {
  attemptDurationMs,
  attemptOutcomeLabel,
  attemptRouteLabel,
  attemptTone,
  formatLogTimestampMs
} from '@/features/logs/lib/log-timeline'
import { formatElapsedMs } from '@/lib/format-duration'

export type LogRoutingAttemptsTimelineProps = {
  readonly attempts: readonly LogProxyAttempt[]
  readonly emptyMessage: string | undefined
}

function AttemptCard({ attempt }: { readonly attempt: LogProxyAttempt }) {
  const durationMs = attemptDurationMs(attempt)
  const startedAt = attempt.startedAt
  const completedAt = attempt.completedAt
  const isInProgress = attempt.statusCode === undefined && startedAt !== undefined && completedAt === undefined

  return (
    <article className="min-w-0 rounded-[var(--radius-lg)] border border-border-soft bg-panel-strong/80 p-3">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-accent-contrast" />
        <code className="font-mono text-[length:var(--density-type-caption-lg)] font-medium text-accent-contrast">
          {attempt.attemptId}
        </code>
        <code className="min-w-0 break-words font-mono text-[length:var(--density-type-caption)] text-fg-dim">
          {attemptRouteLabel(attempt)}
        </code>
        <span className="ml-auto flex shrink-0 items-center gap-2">
          <StatusBadge size="caption" tone={isInProgress ? 'accent' : attemptTone(attempt)}>
            {attemptOutcomeLabel(attempt)}
          </StatusBadge>
          {!isInProgress && durationMs !== undefined ? (
            <span className="font-mono text-[length:var(--density-type-caption)] tabular-nums text-foreground">
              {formatElapsedMs(durationMs)}
            </span>
          ) : null}
        </span>
      </div>
      {!isInProgress && attempt.statusCode !== undefined ? (
        <div className="mt-1.5 font-mono text-[length:var(--density-type-caption)] tabular-nums text-fg-dim">
          HTTP {attempt.statusCode}
        </div>
      ) : null}
      {startedAt !== undefined ? (
        <div className="mt-1 font-mono text-[length:var(--density-type-caption)] tabular-nums text-fg-faint">
          <time dateTime={startedAt}>{formatLogTimestampMs(startedAt)}</time>
          {!isInProgress && completedAt !== undefined ? (
            <>
              {' → '}
              <time dateTime={completedAt}>{formatLogTimestampMs(completedAt)}</time>
            </>
          ) : null}
        </div>
      ) : null}
    </article>
  )
}

export function LogRoutingAttemptsTimeline({ attempts, emptyMessage }: LogRoutingAttemptsTimelineProps) {
  const ordered = useMemo(() => sortByOccurredAt(attempts), [attempts])

  return (
    <section className="mt-6">
      <h2 className="type-panel-title text-foreground">Routing attempts timeline</h2>
      {ordered.length === 0 && emptyMessage !== undefined ? (
        <p className="mt-2 rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 p-2 type-caption text-fg-dim">
          {emptyMessage}
        </p>
      ) : null}
      {ordered.length > 0 ? (
        <ol aria-label="Routing attempts timeline" className="mt-3 space-y-2.5">
          {ordered.map((attempt) => (
            <li className="min-w-0" key={attempt.attemptId}>
              <AttemptCard attempt={attempt} />
            </li>
          ))}
        </ol>
      ) : null}
    </section>
  )
}
