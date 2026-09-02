import { Milestone, Radio, Route } from 'lucide-react'
import { useMemo } from 'react'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { LogLifecycleEvent, LogProxyAttempt } from '@/features/logs/api/schemas'
import { formatTokenUsage } from '@/features/logs/lib/log-token-usage'
import { sortByOccurredAt } from '@/features/logs/lib/log-instant'
import { isStreamEvent } from '@/features/logs/lib/log-request-details'
import { attemptStatus, attemptTone, elapsedMilliseconds, eventTone } from '@/features/logs/lib/log-timeline'
import { formatElapsedMs } from '@/lib/format-duration'

type EvidenceEntry =
  | {
      readonly type: 'event'
      readonly occurredAt: string
      readonly event: LogLifecycleEvent
    }
  | {
      readonly type: 'attempt'
      readonly occurredAt: string
      readonly attempt: LogProxyAttempt
    }

type LogRequestEvidenceTimelineProps = {
  readonly ariaLabel: string
  readonly events: readonly LogLifecycleEvent[]
  readonly attempts: readonly LogProxyAttempt[]
  readonly eventEmptyMessage: string | undefined
  readonly attemptEmptyMessage: string | undefined
}

const dotClass: Record<StatusBadgeTone, string> = {
  muted: 'bg-fg-faint',
  accent: 'bg-accent',
  good: 'bg-good',
  warn: 'bg-warn',
  bad: 'bg-bad'
}

function orderedEvidence(events: readonly LogLifecycleEvent[], attempts: readonly LogProxyAttempt[]): EvidenceEntry[] {
  return sortByOccurredAt([
    ...attempts.map((attempt) => ({
      type: 'attempt' as const,
      occurredAt: attempt.occurredAt,
      attempt
    })),
    ...events.map((event) => ({
      type: 'event' as const,
      occurredAt: event.occurredAt,
      event
    }))
  ])
}

function EventEvidence({
  event,
  deltaMs
}: {
  readonly event: LogLifecycleEvent
  readonly deltaMs: number | undefined
}) {
  const tone = eventTone(event.kind)
  const Icon = isStreamEvent(event) ? Radio : Milestone
  const tokenUsage = formatTokenUsage(event)
  const hasMetadata =
    event.model !== undefined ||
    event.provider !== undefined ||
    event.engine !== undefined ||
    event.statusCode !== undefined ||
    event.durationMs !== undefined ||
    tokenUsage !== undefined

  return (
    <div className="min-w-0 pb-3">
      <div className="flex flex-wrap items-start gap-2">
        <Icon aria-hidden="true" className="mt-0.5 size-3.5 shrink-0 text-fg-faint" />
        <StatusBadge size="caption" tone={tone}>
          {event.kind}
        </StatusBadge>
        {event.attemptId !== undefined ? (
          <code className="break-all font-mono text-[length:var(--density-type-caption)] text-fg-dim">
            attempt {event.attemptId}
          </code>
        ) : null}
        <div className="ml-auto flex shrink-0 items-center gap-2 font-mono text-[length:var(--density-type-caption)] text-fg-faint">
          {deltaMs !== undefined ? <span>{formatElapsedMs(deltaMs, { prefix: '+' })}</span> : null}
          <time dateTime={event.occurredAt}>{new Date(event.occurredAt).toLocaleTimeString()}</time>
        </div>
      </div>
      {hasMetadata ? (
        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[length:var(--density-type-caption)] text-fg-faint">
          {event.model !== undefined ? <span>model {event.model}</span> : null}
          {event.provider !== undefined ? <span>provider {event.provider}</span> : null}
          {event.engine !== undefined ? <span>engine {event.engine}</span> : null}
          {event.statusCode !== undefined ? <span>HTTP {event.statusCode}</span> : null}
          {event.durationMs !== undefined ? <span>{formatElapsedMs(event.durationMs)}</span> : null}
          {tokenUsage !== undefined ? <span>{tokenUsage}</span> : null}
        </div>
      ) : null}
    </div>
  )
}

function AttemptEvidence({
  attempt,
  deltaMs
}: {
  readonly attempt: LogProxyAttempt
  readonly deltaMs: number | undefined
}) {
  const durationMs = elapsedMilliseconds(attempt.startedAt, attempt.completedAt)
  const tone = attemptTone(attempt)
  return (
    <article className="mb-3 min-w-0 rounded-[var(--radius)] border border-border-soft bg-panel-strong/60 p-3">
      <div className="flex flex-wrap items-start gap-2">
        <Route aria-hidden="true" className="mt-0.5 size-3.5 shrink-0 text-accent" />
        <div className="min-w-0 flex-1">
          <code className="block break-words font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
            {attempt.target}
          </code>
          <code className="mt-1 block break-all font-mono text-[length:var(--density-type-caption)] text-fg-dim">
            {attempt.attemptId}
          </code>
        </div>
        <StatusBadge size="caption" tone={tone}>
          {attemptStatus(attempt)}
        </StatusBadge>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[length:var(--density-type-caption)] text-fg-faint">
        {attempt.provider !== undefined ? <span>provider {attempt.provider}</span> : null}
        {attempt.engine !== undefined ? <span>engine {attempt.engine}</span> : null}
        {durationMs !== undefined ? <span>{formatElapsedMs(durationMs)}</span> : null}
        {deltaMs !== undefined ? <span>{formatElapsedMs(deltaMs, { prefix: '+' })}</span> : null}
        <time dateTime={attempt.occurredAt}>{new Date(attempt.occurredAt).toLocaleTimeString()}</time>
      </div>
    </article>
  )
}

export function LogRequestEvidenceTimeline({
  ariaLabel,
  events,
  attempts,
  eventEmptyMessage,
  attemptEmptyMessage
}: LogRequestEvidenceTimelineProps) {
  const entries = useMemo(() => orderedEvidence(events, attempts), [attempts, events])
  const showEventEmpty = events.length === 0 && eventEmptyMessage !== undefined
  const showAttemptEmpty = attempts.length === 0 && attemptEmptyMessage !== undefined

  return (
    <>
      {showEventEmpty || showAttemptEmpty ? (
        <div className="mb-3 grid gap-2 sm:grid-cols-2">
          {showEventEmpty ? (
            <p className="rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 p-2 type-caption text-fg-dim">
              {eventEmptyMessage}
            </p>
          ) : null}
          {showAttemptEmpty ? (
            <p className="rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 p-2 type-caption text-fg-dim">
              {attemptEmptyMessage}
            </p>
          ) : null}
        </div>
      ) : null}
      {entries.length > 0 ? (
        <ol
          aria-label={ariaLabel}
          className="relative before:absolute before:bottom-4 before:left-[calc(var(--shell-compact)/2)] before:top-4 before:w-px before:-translate-x-1/2 before:bg-border-soft before:content-['']"
        >
          {entries.map((entry, index) => {
            const previous = index > 0 ? entries[index - 1] : undefined
            const deltaMs = elapsedMilliseconds(previous?.occurredAt, entry.occurredAt)
            const tone = entry.type === 'event' ? eventTone(entry.event.kind) : attemptTone(entry.attempt)
            return (
              <li
                className="relative grid grid-cols-[var(--shell-compact)_minmax(0,1fr)] gap-3"
                key={
                  entry.type === 'event'
                    ? `event-${entry.event.eventId.toString()}`
                    : `attempt-${entry.attempt.attemptId}`
                }
              >
                <span
                  aria-hidden="true"
                  className={`relative mt-2 size-2 justify-self-center rounded-full ring-2 ring-panel ${dotClass[tone]}`}
                />
                {entry.type === 'event' ? (
                  <EventEvidence deltaMs={deltaMs} event={entry.event} />
                ) : (
                  <AttemptEvidence attempt={entry.attempt} deltaMs={deltaMs} />
                )}
              </li>
            )
          })}
        </ol>
      ) : null}
    </>
  )
}
