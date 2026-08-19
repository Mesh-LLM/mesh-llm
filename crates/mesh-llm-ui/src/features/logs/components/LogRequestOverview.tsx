import {
  Activity,
  Ban,
  BrainCircuit,
  CircleCheckBig,
  CircleMinus,
  CircleSlash,
  CircleX,
  Clock3,
  Gauge,
  Route,
  Server,
  type LucideIcon
} from 'lucide-react'
import type { ReactNode } from 'react'
import type { StatusBadgeTone } from '@/components/ui/StatusBadge'
import type {
  LogArtifact,
  LogLifecycleEvent,
  LogOutcome,
  LogProxyAttempt,
  LogRequest
} from '@/features/logs/api/schemas'
import {
  formatAttemptEvidence,
  formatRequestDuration,
  formatStreamEvidence,
  machineValue,
  type RetainedQueryState
} from '@/features/logs/components/LogRequestOverviewDerivations'
import {
  LogRequestLifecycleOverview,
  LogRequestRoutingOverview
} from '@/features/logs/components/LogRequestOverviewEvidence'
import { LogRequestOverviewMetadata } from '@/features/logs/components/LogRequestOverviewMetadata'

type LogRequestOverviewProps = {
  readonly request: LogRequest
  readonly artifacts: RetainedQueryState<LogArtifact>
  readonly attempts: RetainedQueryState<LogProxyAttempt>
  readonly events: RetainedQueryState<LogLifecycleEvent>
}

type OutcomePresentation = {
  readonly icon: LucideIcon
  readonly label: string
  readonly tone: StatusBadgeTone
}

type MetricCellProps = {
  readonly children: ReactNode
  readonly icon: LucideIcon
  readonly iconTone?: MetricIconTone
  readonly label: string
}

type MetricIconTone = StatusBadgeTone | 'contrast'

const metricIconToneClass: Record<MetricIconTone, string> = {
  muted: 'bg-[color:color-mix(in_oklab,var(--color-fg-faint)_8%,transparent)] text-fg-faint',
  accent: 'bg-[color:color-mix(in_oklab,var(--color-accent)_10%,transparent)] text-accent',
  contrast:
    'bg-[color:color-mix(in_oklab,var(--color-accent-contrast)_10%,transparent)] text-[color:var(--color-accent-contrast)]',
  good: 'bg-[color:color-mix(in_oklab,var(--color-good)_10%,transparent)] text-good',
  warn: 'bg-[color:color-mix(in_oklab,var(--color-warn)_10%,transparent)] text-warn',
  bad: 'bg-[color:color-mix(in_oklab,var(--color-bad)_10%,transparent)] text-bad'
}

const toneTextClass: Record<StatusBadgeTone, string> = {
  muted: 'text-fg-dim',
  accent: 'text-accent',
  good: 'text-good-text',
  warn: 'text-warn-text',
  bad: 'text-bad-text'
}

const outcomePresentation: Record<LogOutcome, OutcomePresentation> = {
  active: { icon: Activity, label: 'Active', tone: 'accent' },
  completed: { icon: CircleCheckBig, label: 'Completed', tone: 'good' },
  failed: { icon: CircleX, label: 'Failed', tone: 'bad' },
  rejected: { icon: Ban, label: 'Rejected', tone: 'bad' },
  cancelled: { icon: CircleSlash, label: 'Cancelled', tone: 'warn' },
  dropped: { icon: CircleMinus, label: 'Dropped', tone: 'bad' }
}

function MetricCell({ children, icon: Icon, iconTone = 'muted', label }: MetricCellProps) {
  return (
    <div className="min-w-0 bg-panel px-4 py-4">
      <dt className="flex min-w-0 items-start gap-2.5 text-fg-faint">
        <span
          aria-hidden="true"
          className={`mt-0.5 grid size-7 shrink-0 place-items-center rounded-[var(--radius-sm)] ${metricIconToneClass[iconTone]}`}
          data-metric-icon-tone={iconTone}
          data-testid={`metric-icon-${label}`}
        >
          <Icon aria-hidden="true" className="size-3.5" />
        </span>
        <span className="type-label min-w-0 break-words">{label}</span>
      </dt>
      <dd className="mt-3 min-w-0">{children}</dd>
    </div>
  )
}

function requestStatusTone(statusCode: number | undefined): StatusBadgeTone {
  if (statusCode === undefined) return 'muted'
  return statusCode >= 400 ? 'bad' : 'good'
}

function MachineValue({ children }: { readonly children: ReactNode }) {
  return (
    <span className="block min-w-0 break-words font-mono tabular-nums text-[length:var(--density-type-caption-lg)] text-foreground">
      {children}
    </span>
  )
}

export function LogRequestOverview({ request, artifacts, attempts, events }: LogRequestOverviewProps) {
  const presentation = outcomePresentation[request.outcome]
  const OutcomeIcon = presentation.icon
  const httpStatusTone = requestStatusTone(request.statusCode)

  return (
    <section aria-label="Request overview" className="flex min-w-0 flex-col gap-[var(--shell-normal)]">
      <dl
        aria-label="Request metrics"
        className="grid min-w-0 grid-cols-2 gap-px overflow-hidden rounded-[var(--radius)] border border-border-soft bg-border-soft lg:grid-cols-3 xl:grid-cols-6"
      >
        <MetricCell icon={OutcomeIcon} iconTone={presentation.tone} label="Status">
          <div data-metric-outcome={request.outcome} data-testid="request-outcome">
            <div className="type-panel-title text-foreground">
              <span>{presentation.label}</span>
            </div>
            <div
              className={`mt-1 font-mono type-caption ${toneTextClass[httpStatusTone]}`}
              data-metric-http-status
              data-testid="request-http-status"
            >
              {request.statusCode === undefined ? 'HTTP status not recorded' : `HTTP ${request.statusCode}`}
            </div>
          </div>
        </MetricCell>
        <MetricCell icon={Clock3} iconTone="muted" label="Duration">
          <MachineValue>{formatRequestDuration(request)}</MachineValue>
        </MetricCell>
        <MetricCell icon={Server} iconTone="accent" label="Provider">
          <MachineValue>{machineValue(request.provider)}</MachineValue>
        </MetricCell>
        <MetricCell icon={BrainCircuit} iconTone="contrast" label="Model">
          <MachineValue>{machineValue(request.model)}</MachineValue>
        </MetricCell>
        <MetricCell icon={Route} iconTone="muted" label="Attempts / retries">
          <MachineValue>{formatAttemptEvidence(attempts.items)}</MachineValue>
        </MetricCell>
        <MetricCell icon={Gauge} iconTone="accent" label="Stream / completion tokens">
          <MachineValue>{formatStreamEvidence(events.items)}</MachineValue>
        </MetricCell>
      </dl>
      <LogRequestLifecycleOverview events={events} />
      <LogRequestOverviewMetadata artifacts={artifacts} request={request} />
      <LogRequestRoutingOverview attempts={attempts} />
    </section>
  )
}
