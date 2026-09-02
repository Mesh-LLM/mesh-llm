import { CircleCheckBig, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { LogArtifact, LogOutcome, LogRequest } from '@/features/logs/api/schemas'
import { formatRequestDuration, machineValue } from '@/features/logs/components/LogRequestOverviewDerivations'
import { trailingRowSpanClass } from '@/features/logs/lib/log-grid'
import { cn } from '@/lib/cn'

export type LogRequestDiagnosticEvidence = {
  readonly artifacts: readonly LogArtifact[] | undefined
  readonly attemptCount: number | undefined
  readonly diagnosticMarkerCount: number | undefined
  readonly evidenceComplete: boolean
  readonly hasErrorEvidence: boolean
}

export type LogRequestDiagnosticSummaryProps = {
  readonly request: LogRequest
  readonly evidence: LogRequestDiagnosticEvidence
  readonly successful: boolean
}

type DiagnosticMetric = {
  readonly label: string
  readonly value: ReactNode
}

const CONTENT_STATES = [
  'available',
  'unavailable',
  'missing',
  'corrupt'
] as const satisfies readonly LogArtifact['contentState'][]

const outcomeDetail = {
  active: 'The request has not reached a terminal outcome. Retained diagnostics may still change.',
  completed: 'The request completed, but retained evidence is still loading or unavailable.',
  failed: 'Review the ordered failure markers, retry targets, terminal status, and error artifacts below.',
  rejected: 'The request was rejected before normal execution completed.',
  cancelled: 'The caller cancelled this request before normal completion.',
  dropped: 'The request was dropped before a normal terminal response was produced.'
} satisfies Record<LogOutcome, string>

const outcomeTone = {
  active: 'accent',
  completed: 'warn',
  failed: 'bad',
  rejected: 'bad',
  cancelled: 'warn',
  dropped: 'bad'
} satisfies Record<LogOutcome, StatusBadgeTone>

const successDetail = 'The request completed without retained failure, retry, or error-artifact evidence.'
const retainedDetail = 'The request completed, but a failed attempt, error marker, or error artifact was retained.'

function contentStateSummary(artifacts: readonly LogArtifact[] | undefined): string {
  if (artifacts === undefined) return 'Not available'
  if (artifacts.length === 0) return 'None retained'
  return CONTENT_STATES.map((state) => ({
    state,
    count: artifacts.filter((artifact) => artifact.contentState === state).length
  }))
    .filter(({ count }) => count > 0)
    .map(({ state, count }) => `${count.toLocaleString()} ${state}`)
    .join(' · ')
}

function countedArtifacts(artifacts: readonly LogArtifact[] | undefined, field: 'redacted' | 'truncated'): string {
  if (artifacts === undefined) return 'Not available'
  const count = artifacts.filter((artifact) => artifact[field]).length
  return `${count.toLocaleString()} of ${artifacts.length.toLocaleString()}`
}

function countValue(value: number | undefined): string {
  return value === undefined ? 'Not available' : value.toLocaleString()
}

function SummaryGrid({ metrics }: { readonly metrics: readonly DiagnosticMetric[] }) {
  return (
    <dl
      aria-label="Diagnostic summary"
      className="grid min-w-0 gap-px overflow-hidden rounded-[var(--radius)] bg-border-soft sm:grid-cols-2"
    >
      {metrics.map((metric, metricIndex) => (
        <div
          className={cn(
            'min-w-0 bg-panel px-[var(--panel-x)] py-[var(--panel-y)]',
            trailingRowSpanClass(metrics.length, metricIndex, 2, 'sm')
          )}
          key={metric.label}
        >
          <dt className="type-label text-fg-faint">{metric.label}</dt>
          <dd className="mt-1 min-w-0 break-words font-mono tabular-nums text-[length:var(--density-type-caption-lg)] text-foreground [overflow-wrap:anywhere]">
            {metric.value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function successMetrics(
  request: LogRequest,
  artifacts: readonly LogArtifact[] | undefined
): readonly DiagnosticMetric[] {
  return [
    { label: 'Request source', value: request.source },
    { label: 'Artifact records', value: countValue(artifacts?.length) },
    { label: 'Redacted', value: countedArtifacts(artifacts, 'redacted') },
    { label: 'Truncated', value: countedArtifacts(artifacts, 'truncated') },
    { label: 'Content states', value: contentStateSummary(artifacts) },
    {
      label: 'Artifact access',
      value: 'Metadata only; body content not requested'
    }
  ]
}

function evidenceMetrics(request: LogRequest, evidence: LogRequestDiagnosticEvidence): readonly DiagnosticMetric[] {
  return [
    { label: 'Outcome', value: request.outcome },
    { label: 'HTTP status', value: machineValue(request.statusCode) },
    { label: 'Provider', value: machineValue(request.provider) },
    { label: 'Engine', value: machineValue(request.engine) },
    { label: 'Duration', value: formatRequestDuration(request) },
    { label: 'Attempt count', value: countValue(evidence.attemptCount) },
    {
      label: 'Diagnostic markers',
      value: countValue(evidence.diagnosticMarkerCount)
    }
  ]
}

function TerminalRecord({ request }: { readonly request: LogRequest }) {
  return (
    <section
      aria-label="Terminal record"
      className="min-w-0 rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 px-[var(--panel-x)] py-[var(--panel-y)]"
    >
      <h3 className="type-label text-fg-faint">Terminal record</h3>
      <p className="mt-2 min-w-0 font-mono text-[length:var(--density-type-caption)] text-foreground [overflow-wrap:anywhere]">
        {machineValue(request.terminalAt)} / HTTP {machineValue(request.statusCode)}
      </p>
      <p className="mt-1 min-w-0 font-mono text-[length:var(--density-type-caption)] text-fg-dim [overflow-wrap:anywhere]">
        route {machineValue(request.route)} / model {machineValue(request.model)}
      </p>
      <p className="mt-1 min-w-0 font-mono text-[length:var(--density-type-caption)] text-fg-dim [overflow-wrap:anywhere]">
        provider {machineValue(request.provider)} / engine {machineValue(request.engine)}
      </p>
    </section>
  )
}

export function LogRequestDiagnosticSummary({ request, evidence, successful }: LogRequestDiagnosticSummaryProps) {
  const hasRetainedEvidence = request.outcome === 'completed' && evidence.evidenceComplete && evidence.hasErrorEvidence
  const summaryState = successful ? 'success' : request.outcome === 'completed' ? 'attention' : request.outcome
  const summaryTitle = successful
    ? 'No errors'
    : hasRetainedEvidence
      ? 'Diagnostic evidence retained'
      : request.outcome === 'completed'
        ? 'Diagnostics incomplete'
        : `Request ${request.outcome}`
  const summaryDetail = successful
    ? successDetail
    : hasRetainedEvidence
      ? retainedDetail
      : outcomeDetail[request.outcome]
  const summaryTone = successful ? 'good' : outcomeTone[request.outcome]
  const SummaryIcon = successful ? CircleCheckBig : TriangleAlert
  const metrics = successful ? successMetrics(request, evidence.artifacts) : evidenceMetrics(request, evidence)

  return (
    <>
      <div
        className="flex min-w-0 flex-wrap items-start gap-2 rounded-[var(--radius)] border border-border bg-panel px-[var(--panel-x)] py-[var(--panel-y)] sm:gap-3"
        data-diagnostic-state={summaryState}
        role="status"
      >
        <SummaryIcon
          aria-hidden="true"
          className={`mt-0.5 size-4 shrink-0 ${successful ? 'text-good' : 'text-fg-faint'}`}
        />
        <div className="min-w-0 flex-1">
          <h2 className="type-panel-title text-foreground">{summaryTitle}</h2>
          <p className="mt-1 break-words type-caption text-fg-dim">{summaryDetail}</p>
        </div>
        <StatusBadge className="max-w-full shrink-0" dot size="caption" tone={summaryTone}>
          {request.outcome}
        </StatusBadge>
      </div>
      <SummaryGrid metrics={metrics} />
      <TerminalRecord request={request} />
    </>
  )
}
