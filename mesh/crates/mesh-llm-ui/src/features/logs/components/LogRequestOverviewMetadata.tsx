import { Archive, Database } from 'lucide-react'
import type { ReactNode } from 'react'
import type { LogArtifact, LogRequest } from '@/features/logs/api/schemas'
import {
  formatTimestamp,
  machineValue,
  type RetainedQueryState
} from '@/features/logs/components/LogRequestOverviewDerivations'
import { LogRequestOverviewPanel } from '@/features/logs/components/LogRequestOverviewPanel'
import { deriveArtifactCounts } from '@/features/logs/lib/log-artifact-counts'
import { trailingRowSpanClass } from '@/features/logs/lib/log-grid'
import { cn } from '@/lib/cn'

type LogRequestOverviewMetadataProps = {
  readonly artifacts: RetainedQueryState<LogArtifact>
  readonly request: LogRequest
}

type MetadataField = {
  readonly label: string
  readonly value: ReactNode
}

function MetadataGrid({
  fields,
  wideColumns = 2
}: {
  readonly fields: readonly MetadataField[]
  readonly wideColumns?: 2 | 3
}) {
  return (
    <dl className={cn('grid gap-px bg-border-soft sm:grid-cols-2', wideColumns === 3 && 'lg:grid-cols-3')}>
      {fields.map((field, fieldIndex) => {
        const smSpan = trailingRowSpanClass(fields.length, fieldIndex, 2, 'sm')
        const lgSpan = wideColumns === 3 ? trailingRowSpanClass(fields.length, fieldIndex, 3, 'lg') : undefined
        return (
          <div
            className={cn(
              'min-w-0 bg-panel px-[var(--panel-x)] py-[var(--panel-y)]',
              smSpan,
              // The sm span above stays active at lg unless overridden; reset it
              // when the 3-column layout doesn't need a span of its own here.
              wideColumns === 3 && (lgSpan ?? (smSpan ? 'lg:col-span-1' : undefined))
            )}
            key={field.label}
          >
            <dt className="type-label text-fg-faint">{field.label}</dt>
            <dd className="mt-1 break-words font-mono tabular-nums text-[length:var(--density-type-caption-lg)] text-foreground">
              {field.value}
            </dd>
          </div>
        )
      })}
    </dl>
  )
}

function requestMetadata(request: LogRequest): readonly MetadataField[] {
  return [
    { label: 'Request ID', value: request.requestId.toString() },
    { label: 'Created', value: <time dateTime={request.createdAt}>{formatTimestamp(request.createdAt)}</time> },
    {
      label: 'Terminal',
      value:
        request.terminalAt === undefined ? (
          'Not recorded'
        ) : (
          <time dateTime={request.terminalAt}>{formatTimestamp(request.terminalAt)}</time>
        )
    },
    { label: 'Route', value: machineValue(request.route) },
    { label: 'Model', value: machineValue(request.model) },
    { label: 'Provider', value: machineValue(request.provider) },
    { label: 'Engine', value: machineValue(request.engine) },
    { label: 'HTTP status', value: machineValue(request.statusCode) },
    { label: 'Record source', value: request.source }
  ]
}

function ArtifactSummary({
  artifacts,
  source
}: {
  readonly artifacts: RetainedQueryState<LogArtifact>
  readonly source: LogRequest['source']
}) {
  if (artifacts.items === undefined) {
    const message = artifacts.loading
      ? 'Loading artifact metadata.'
      : artifacts.error
        ? 'Artifact metadata could not be loaded.'
        : 'Artifact metadata was not recorded.'
    return (
      <p className="type-body p-[var(--panel-x)] text-fg-dim" role={artifacts.error ? 'alert' : 'status'}>
        {message}
      </p>
    )
  }
  if (artifacts.items.length === 0) {
    return (
      <p className="type-body p-[var(--panel-x)] text-fg-dim" role="status">
        No artifact metadata was retained for this request.
      </p>
    )
  }

  const items = artifacts.items
  const counts = deriveArtifactCounts(items)
  const fields: readonly MetadataField[] = [
    { label: 'Artifact records', value: counts.total.toLocaleString() },
    { label: 'Content states', value: counts.contentStates },
    { label: 'Redacted', value: `${counts.redacted.toLocaleString()} of ${counts.total.toLocaleString()}` },
    { label: 'Truncated', value: `${counts.truncated.toLocaleString()} of ${counts.total.toLocaleString()}` },
    { label: 'Stored bytes', value: `${counts.bytes.toLocaleString()} B` },
    { label: 'Versions', value: counts.versions.map((version) => `v${version}`).join(' · ') },
    { label: 'Request source', value: source }
  ]
  return <MetadataGrid fields={fields} />
}

export function LogRequestOverviewMetadata({ artifacts, request }: LogRequestOverviewMetadataProps) {
  return (
    <div className="grid min-w-0 gap-[var(--shell-normal)] xl:grid-cols-[minmax(0,1.7fr)_minmax(20rem,1fr)]">
      <LogRequestOverviewPanel
        ariaLabel="Request metadata"
        description="Canonical fields retained with the request summary."
        icon={Database}
        title="Request metadata"
      >
        <MetadataGrid fields={requestMetadata(request)} wideColumns={3} />
      </LogRequestOverviewPanel>
      <LogRequestOverviewPanel
        ariaLabel="Artifact retention"
        description="Retention, redaction, truncation, byte, and version metadata for this request."
        icon={Archive}
        title="Artifact retention"
      >
        <ArtifactSummary artifacts={artifacts} source={request.source} />
      </LogRequestOverviewPanel>
    </div>
  )
}
