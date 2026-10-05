import type { LogEventKind, LogLifecycleEvent } from '@/features/logs/api/schemas'
import { compareLogInstants } from '@/features/logs/lib/log-instant'
import { formatElapsedMs } from '@/lib/format-duration'

export type LifecycleNode = {
  readonly kind: LogEventKind
  readonly key: string
  readonly count: number
  readonly firstOccurredAt: string
  readonly lastOccurredAt: string
  readonly elapsed: string | undefined
}

function orderedEvents(events: readonly LogLifecycleEvent[]): readonly LogLifecycleEvent[] {
  return events
    .map((event, index) => ({ event, index }))
    .sort(
      (left, right) => compareLogInstants(left.event.occurredAt, right.event.occurredAt) || left.index - right.index
    )
    .map(({ event }) => event)
}

function elapsedBetween(from: string, to: string): string | undefined {
  const delta = Date.parse(to) - Date.parse(from)
  if (!Number.isFinite(delta) || delta < 0) return undefined
  return formatElapsedMs(delta, { prefix: '+' })
}

export function lifecycleNodes(events: readonly LogLifecycleEvent[]): readonly LifecycleNode[] {
  const nodes: LifecycleNode[] = []
  for (const event of orderedEvents(events)) {
    const previous = nodes.at(-1)
    if (previous && previous.kind === event.kind) {
      nodes[nodes.length - 1] = {
        ...previous,
        count: previous.count + 1,
        lastOccurredAt: event.occurredAt
      }
      continue
    }
    nodes.push({
      kind: event.kind,
      key: event.eventId.toString(),
      count: 1,
      firstOccurredAt: event.occurredAt,
      lastOccurredAt: event.occurredAt,
      elapsed: previous ? elapsedBetween(previous.lastOccurredAt, event.occurredAt) : undefined
    })
  }
  return nodes
}
