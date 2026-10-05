import {
  Activity,
  Ban,
  Check,
  CircleCheck,
  CircleSlash,
  CircleX,
  Gauge,
  Inbox,
  Link2,
  Radio,
  Share2,
  ShieldAlert,
  TriangleAlert,
  Unplug,
  X,
  Zap,
  type LucideIcon
} from 'lucide-react'
import type { LogEventKind } from '@/features/logs/api/schemas'
import { formatClockTime } from '@/features/logs/components/LogRequestOverviewDerivations'
import type { LifecycleNode } from './log-request-lifecycle-data'
import {
  connectorPositionStyle,
  incomingConnectorPositionStyle,
  outgoingConnectorPositionStyle
} from './log-request-lifecycle-layout'

type LifecycleTone = 'accent' | 'bad' | 'good' | 'warn'

type LifecyclePresentation = {
  readonly icon: LucideIcon
  readonly label: string
  readonly tone: LifecycleTone
}

const LIFECYCLE_PRESENTATION: Readonly<Record<LogEventKind, LifecyclePresentation>> = {
  admitted: { icon: Inbox, label: 'Received', tone: 'good' },
  route_selected: { icon: Share2, label: 'Routed', tone: 'good' },
  attempt_started: { icon: Link2, label: 'Connected', tone: 'good' },
  backend_stream_first_item: { icon: Zap, label: 'First token', tone: 'accent' },
  stream_started: { icon: Radio, label: 'Stream started', tone: 'accent' },
  stream_chunk: { icon: Activity, label: 'Chunks', tone: 'accent' },
  usage_recorded: { icon: Gauge, label: 'Usage', tone: 'accent' },
  stream_completed: { icon: CircleCheck, label: 'Stream done', tone: 'good' },
  attempt_completed: { icon: CircleCheck, label: 'Attempt done', tone: 'good' },
  completed: { icon: Check, label: 'Completed', tone: 'good' },
  attempt_failed: { icon: CircleX, label: 'Attempt failed', tone: 'bad' },
  stream_error: { icon: TriangleAlert, label: 'Stream error', tone: 'bad' },
  audit_error: { icon: ShieldAlert, label: 'Audit error', tone: 'bad' },
  failed: { icon: X, label: 'Failed', tone: 'bad' },
  rejected: { icon: Ban, label: 'Rejected', tone: 'warn' },
  cancelled: { icon: CircleSlash, label: 'Cancelled', tone: 'warn' },
  dropped: { icon: Unplug, label: 'Dropped', tone: 'bad' }
}

const NODE_TONE_CLASS: Readonly<Record<LifecycleTone, string>> = {
  accent: 'bg-[color:color-mix(in_oklab,var(--color-accent)_22%,var(--color-panel))] text-accent',
  bad: 'bg-[color:color-mix(in_oklab,var(--color-bad)_22%,var(--color-panel))] text-bad-text',
  good: 'bg-[color:color-mix(in_oklab,var(--color-good)_22%,var(--color-panel))] text-good-text',
  warn: 'bg-[color:color-mix(in_oklab,var(--color-warn)_22%,var(--color-panel))] text-warn-text'
}

const CONNECTOR_TONE_CLASS: Readonly<Record<LifecycleTone, string>> = {
  accent: 'bg-[color:color-mix(in_oklab,var(--color-accent)_55%,transparent)]',
  bad: 'bg-[color:color-mix(in_oklab,var(--color-bad)_55%,transparent)]',
  good: 'bg-[color:color-mix(in_oklab,var(--color-good)_55%,transparent)]',
  warn: 'bg-[color:color-mix(in_oklab,var(--color-warn)_55%,transparent)]'
}

function lifecyclePresentation(kind: LogEventKind): LifecyclePresentation {
  return LIFECYCLE_PRESENTATION[kind]
}

function ElapsedSeparator({
  continuation = false,
  elapsed
}: {
  readonly continuation?: boolean
  readonly elapsed: string
}) {
  return (
    <span
      aria-label={`Elapsed ${elapsed}${continuation ? ' from previous lifecycle page' : ''}`}
      className={
        continuation
          ? 'sr-only'
          : 'type-label absolute bottom-full left-1/2 mb-1.5 -translate-x-1/2 whitespace-nowrap font-mono normal-case! tracking-normal! tabular-nums text-fg-faint'
      }
      role="separator"
    >
      <span aria-hidden="true">{elapsed}</span>
    </span>
  )
}

export function LifecycleNodeItem({
  continuationElapsed,
  node,
  nextElapsed,
  showConnector,
  showIncoming,
  showOutgoing
}: {
  readonly continuationElapsed: string | undefined
  readonly node: LifecycleNode
  readonly nextElapsed: string | undefined
  readonly showConnector: boolean
  readonly showIncoming: boolean
  readonly showOutgoing: boolean
}) {
  const { icon: Icon, label, tone } = lifecyclePresentation(node.kind)

  return (
    <li className="flex min-w-0 flex-1 flex-col items-center" data-event-kind={node.kind}>
      <div className="relative flex w-full items-center justify-center">
        {showIncoming ? (
          <span
            aria-hidden="true"
            className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-fg-dim"
            data-lifecycle-edge="incoming"
            style={incomingConnectorPositionStyle()}
          />
        ) : null}
        <span
          aria-hidden="true"
          className={`grid size-9 shrink-0 place-items-center rounded-full ${NODE_TONE_CLASS[tone]}`}
          data-event-tone={tone}
        >
          <Icon className="size-4" />
        </span>
        {showConnector ? (
          <span
            className={`absolute top-1/2 h-px -translate-y-1/2 ${CONNECTOR_TONE_CLASS[tone]}`}
            style={connectorPositionStyle()}
          >
            {nextElapsed === undefined ? null : <ElapsedSeparator elapsed={nextElapsed} />}
          </span>
        ) : null}
        {showOutgoing ? (
          <span
            aria-hidden="true"
            className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-fg-dim"
            data-lifecycle-edge="outgoing"
            style={outgoingConnectorPositionStyle()}
          />
        ) : null}
      </div>
      <div className="mt-2.5 min-w-0 w-full text-center">
        {continuationElapsed === undefined ? null : <ElapsedSeparator continuation elapsed={continuationElapsed} />}
        <p
          aria-label={
            node.count > 1
              ? `${node.count} ${label.toLowerCase()}, last at ${formatClockTime(node.lastOccurredAt)}`
              : undefined
          }
          className="type-caption truncate text-foreground"
        >
          {label}
          {node.count > 1 ? (
            <span aria-hidden="true" className="text-fg-faint">
              {' '}
              ×{node.count}
            </span>
          ) : null}
        </p>
        <time
          className="type-label mt-0.5 block break-words font-mono normal-case! tracking-normal! tabular-nums text-fg-dim"
          dateTime={node.firstOccurredAt}
          title={node.firstOccurredAt}
        >
          {formatClockTime(node.firstOccurredAt)}
        </time>
      </div>
    </li>
  )
}
