import { Archive, Database } from 'lucide-react'
import type { ReactNode } from 'react'
import type { LogArtifact, LogRequest } from '@/features/logs/api/schemas'
import {
  formatTimestamp,
  machineValue,
  type RetainedQueryState
} from '@/features/logs/components/LogRequestOverviewDerivations'
import { LogRequestOverviewPanel } from '@/features/logs/components/LogRequestOverviewPanel'
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

const CONTENT_STATES = [
  'available',
  'unavailable',
  'missing',
  'corrupt'
] as const satisfies readonly LogArtifact['contentState'][]

function MetadataGrid({ fields }: { readonly fields: readonly MetadataField[] }) {
  return (
    <dl className="grid gap-px bg-border-soft sm:grid-cols-2 xl:grid-cols-3">
      {fields.map((field, fieldIndex) => (
        <div
          className={cn(
            'min-w-0 bg-panel px-[var(--panel-x)] py-[var(--panel-y)]',
            trailingRowSpanClass(fields.length, fieldIndex, 2, 'sm'),
            trailingRowSpanClass(fields.length, fieldIndex, 3, 'xl')
          )}
          key={field.label}
        >
          <dt className="type-label text-fg-faint">{field.label}</dt>
          <dd className="mt-1 break-words font-mono tabular-nums text-[length:var(--density-type-caption-lg)] text-foreground">
            {field.value}
          </dd>
        </div>
      ))}
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
  const contentStates = CONTENT_STATES.map((state) => ({
    state,
    count: items.filter((item) => item.contentState === state).length
  }))
    .filter(({ count }) => count > 0)
    .map(({ state, count }) => `${count.toLocaleString()} ${state}`)
    .join(' · ')
  const redacted = items.filter((item) => item.redacted).length
  const truncated = items.filter((item) => item.truncated).length
  const bytes = items.reduce((total, item) => total + item.bytes, 0)
  const versions = [...new Set(items.map((item) => item.version))].sort((left, right) => left - right)
  const fields: readonly MetadataField[] = [
    { label: 'Artifact records', value: items.length.toLocaleString() },
    { label: 'Content states', value: contentStates },
    { label: 'Redacted', value: `${redacted.toLocaleString()} of ${items.length.toLocaleString()}` },
    { label: 'Truncated', value: `${truncated.toLocaleString()} of ${items.length.toLocaleString()}` },
    { label: 'Stored bytes', value: `${bytes.toLocaleString()} B` },
    { label: 'Versions', value: versions.map((version) => `v${version}`).join(' · ') },
    { label: 'Request source', value: source }
  ]
  return <MetadataGrid fields={fields} />
}

export function LogRequestOverviewMetadata({ artifacts, request }: LogRequestOverviewMetadataProps) {
  return (
    <>
      <LogRequestOverviewPanel
        ariaLabel="Request metadata"
        description="Canonical fields retained with the request summary."
        icon={Database}
        title="Request metadata"
      >
        <MetadataGrid fields={requestMetadata(request)} />
      </LogRequestOverviewPanel>
      <LogRequestOverviewPanel
        ariaLabel="Artifact retention"
        description="Retention, redaction, truncation, byte, and version metadata for this request."
        icon={Archive}
        title="Artifact retention"
      >
        <ArtifactSummary artifacts={artifacts} source={request.source} />
      </LogRequestOverviewPanel>
    </>
  )
}
