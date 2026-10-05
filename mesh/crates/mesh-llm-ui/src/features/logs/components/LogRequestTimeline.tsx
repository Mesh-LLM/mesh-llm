import type { LogLifecycleEvent, LogProxyAttempt } from '@/features/logs/api/schemas'
import { LogRoutingAttemptsTimeline } from './LogRoutingAttemptsTimeline'
import { LogStreamTimeline } from './LogStreamTimeline'

export type LogRequestTimelineProps = {
  readonly events: readonly LogLifecycleEvent[] | undefined
  readonly attempts: readonly LogProxyAttempt[] | undefined
  readonly eventsLoading: boolean
  readonly eventsError: boolean
  readonly attemptsLoading: boolean
  readonly attemptsError: boolean
}

function QueryNotice({
  subject,
  loading,
  error
}: {
  readonly subject: 'lifecycle evidence' | 'routing attempts'
  readonly loading: boolean
  readonly error: boolean
}) {
  if (error) {
    return (
      <p
        className="min-w-0 break-words rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 px-[var(--panel-x)] py-[var(--panel-y)] type-caption text-fg-dim"
        role="alert"
      >
        {subject === 'lifecycle evidence'
          ? 'Lifecycle evidence could not be loaded from the local log service.'
          : 'Routing attempts could not be loaded from the local log service.'}
      </p>
    )
  }
  if (loading) {
    return (
      <p
        className="min-w-0 break-words rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 px-[var(--panel-x)] py-[var(--panel-y)] type-caption text-fg-dim"
        role="status"
      >
        Loading {subject}.
      </p>
    )
  }
  return null
}

export function LogRequestTimeline({
  events,
  attempts,
  eventsLoading,
  eventsError,
  attemptsLoading,
  attemptsError
}: LogRequestTimelineProps) {
  const eventsReady = !eventsLoading && !eventsError
  const attemptsReady = !attemptsLoading && !attemptsError

  return (
    <div className="grid min-w-0 gap-[var(--shell-normal)]">
      {eventsReady ? (
        <LogStreamTimeline
          emptyMessage="No lifecycle or stream markers were retained for this request."
          events={events ?? []}
        />
      ) : (
        <QueryNotice error={eventsError} loading={eventsLoading} subject="lifecycle evidence" />
      )}
      {attemptsReady ? (
        <LogRoutingAttemptsTimeline
          attempts={attempts ?? []}
          emptyMessage="No proxy attempts were retained for this request."
        />
      ) : (
        <QueryNotice error={attemptsError} loading={attemptsLoading} subject="routing attempts" />
      )}
    </div>
  )
}
