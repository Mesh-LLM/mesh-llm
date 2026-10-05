import { type CSSProperties, useMemo } from 'react'
import type { LogLifecycleEvent } from '@/features/logs/api/schemas'
import { sortByOccurredAt } from '@/features/logs/lib/log-instant'
import { elapsedMilliseconds, formatLogTimestampMs, isStreamLifecycleEvent } from '@/features/logs/lib/log-timeline'
import { formatTokenUsage } from '@/features/logs/lib/log-token-usage'
import { formatElapsedMs } from '@/lib/format-duration'

export type LogStreamTimelineProps = {
  readonly events: readonly LogLifecycleEvent[]
  readonly emptyMessage: string | undefined
}

function StreamTimelineEntry({
  event,
  deltaMs
}: {
  readonly event: LogLifecycleEvent
  readonly deltaMs: number | undefined
}) {
  const isError = event.kind === 'stream_error'
  const tokenUsage = formatTokenUsage(event)

  // Marker style: filled blue center + outer ring (accent/primary) for normal, bad tone for errors
  const markerStyle: CSSProperties = isError
    ? {
        backgroundColor: 'var(--color-bad)',
        boxShadow: `0 0 0 2px color-mix(in oklab, var(--color-bad) 30%, transparent)`
      }
    : {
        backgroundColor: 'var(--color-accent)',
        boxShadow: `0 0 0 2px color-mix(in oklab, var(--color-accent) 25%, transparent)`
      }

  // Event name tone: accent for normal stream events, bad for errors
  const eventToneClass = isError ? 'text-bad' : 'text-foreground'

  return (
    <li className="relative grid grid-cols-[16px_minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-x-3 py-2.5 before:absolute before:left-[7.5px] before:top-1/2 before:h-[calc(100%+12px)] before:w-px before:bg-accent/20 before:content-[''] last:before:hidden">
      {/* Marker */}
      <span
        aria-hidden="true"
        className="relative z-10 size-3.5 justify-self-center rounded-full"
        style={markerStyle}
      />

      {/* Event name + attempt id */}
      <div className="min-w-0">
        <code className={`font-mono text-[length:var(--density-type-caption-lg)] font-semibold ${eventToneClass}`}>
          {event.kind}
        </code>
        {event.attemptId !== undefined ? (
          <code className="mt-0.5 block font-mono text-[length:var(--density-type-caption)] text-fg-dim">
            {event.attemptId}
          </code>
        ) : null}
      </div>

      {/* Centered delta chip (empty placeholder keeps the first row aligned) */}
      {deltaMs !== undefined ? (
        <span className="shrink-0 justify-self-center rounded-md bg-accent/[.1] px-2 py-0.5 font-mono text-[length:var(--density-type-caption)] font-medium tabular-nums text-accent">
          {formatElapsedMs(deltaMs, { prefix: '+' })}
        </span>
      ) : (
        <span aria-hidden="true" />
      )}

      {/* Right-aligned timestamp + tokens */}
      <div className="min-w-0 text-right font-mono text-[length:var(--density-type-caption)] tabular-nums">
        <time className="block text-fg-dim" dateTime={event.occurredAt}>
          {formatLogTimestampMs(event.occurredAt)}
        </time>
        {tokenUsage !== undefined ? <span className="mt-0.5 block text-fg-faint">{tokenUsage}</span> : null}
      </div>
    </li>
  )
}

export function LogStreamTimeline({ events, emptyMessage }: LogStreamTimelineProps) {
  const streamEvents = useMemo(() => sortByOccurredAt(events.filter((e) => isStreamLifecycleEvent(e.kind))), [events])

  return (
    <section>
      <h2 className="type-panel-title text-foreground">Stream timeline</h2>
      {streamEvents.length === 0 && emptyMessage !== undefined ? (
        <p className="mt-2 rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 p-2 type-caption text-fg-dim">
          {emptyMessage}
        </p>
      ) : null}
      {streamEvents.length > 0 ? (
        <ol aria-label="Stream timeline" className="mt-3 space-y-1">
          {streamEvents.map((event, index) => {
            const previous = index > 0 ? streamEvents[index - 1] : undefined
            return (
              <StreamTimelineEntry
                deltaMs={elapsedMilliseconds(previous?.occurredAt, event.occurredAt)}
                event={event}
                key={event.eventId.toString()}
              />
            )
          })}
        </ol>
      ) : null}
    </section>
  )
}
