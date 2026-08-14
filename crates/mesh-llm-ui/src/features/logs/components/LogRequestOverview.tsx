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
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
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
  readonly label: string
}

const outcomePresentation: Record<LogOutcome, OutcomePresentation> = {
  active: { icon: Activity, label: 'Active', tone: 'accent' },
  completed: { icon: CircleCheckBig, label: 'Completed', tone: 'good' },
  failed: { icon: CircleX, label: 'Failed', tone: 'bad' },
  rejected: { icon: Ban, label: 'Rejected', tone: 'bad' },
  cancelled: { icon: CircleSlash, label: 'Cancelled', tone: 'warn' },
  dropped: { icon: CircleMinus, label: 'Dropped', tone: 'bad' }
}

function MetricCell({ children, icon: Icon, label }: MetricCellProps) {
  return (
    <div className="min-w-0 bg-panel px-4 py-4">
      <dt className="flex items-center gap-2.5 text-fg-faint">
        <span className="grid size-7 shrink-0 place-items-center rounded-[var(--radius-sm)] border border-[color:color-mix(in_oklab,var(--color-accent)_28%,var(--color-border-soft))] bg-[color:color-mix(in_oklab,var(--color-accent)_8%,var(--color-panel))] text-accent">
          <Icon aria-hidden="true" className="size-3.5" />
        </span>
        <span className="type-label">{label}</span>
      </dt>
      <dd className="mt-3 min-w-0">{children}</dd>
    </div>
  )
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

  return (
    <section aria-label="Request overview" className="flex min-w-0 flex-col gap-[var(--shell-normal)]">
      <dl
        aria-label="Request metrics"
        className="grid min-w-0 grid-cols-2 gap-px overflow-hidden rounded-[var(--radius)] border border-border-soft bg-border-soft lg:grid-cols-3 xl:grid-cols-6"
      >
        <MetricCell icon={Activity} label="Status">
          <StatusBadge size="caption" tone={presentation.tone}>
            <OutcomeIcon aria-hidden="true" className="size-3" />
            {presentation.label}
          </StatusBadge>
        </MetricCell>
        <MetricCell icon={Clock3} label="Duration">
          <MachineValue>{formatRequestDuration(request)}</MachineValue>
        </MetricCell>
        <MetricCell icon={Server} label="Provider">
          <MachineValue>{machineValue(request.provider)}</MachineValue>
        </MetricCell>
        <MetricCell icon={BrainCircuit} label="Model">
          <MachineValue>{machineValue(request.model)}</MachineValue>
        </MetricCell>
        <MetricCell icon={Route} label="Attempts / retries">
          <MachineValue>{formatAttemptEvidence(attempts.items)}</MachineValue>
        </MetricCell>
        <MetricCell icon={Gauge} label="Stream / completion tokens">
          <MachineValue>{formatStreamEvidence(events.items)}</MachineValue>
        </MetricCell>
      </dl>
      <LogRequestLifecycleOverview events={events} />
      <LogRequestOverviewMetadata artifacts={artifacts} request={request} />
      <LogRequestRoutingOverview attempts={attempts} />
    </section>
  )
}
