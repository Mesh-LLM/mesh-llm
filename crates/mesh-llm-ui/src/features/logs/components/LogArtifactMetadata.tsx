import { type ReactNode, useId } from 'react'
import type { LogArtifact } from '@/features/logs/api/schemas'
import { LogArtifactDownloadControl } from '@/features/logs/components/LogArtifactDownloadControl'
import { LogArtifactStateBadges } from '@/features/logs/components/LogArtifactStateBadges'
import type { ClassifiedLogArtifacts } from '@/features/logs/lib/log-payload-content'

export type LogArtifactMetadataProps = {
  readonly artifact: LogArtifact
}

export type LogArtifactInventoryProps = {
  readonly classified: ClassifiedLogArtifacts
}

function formatTimestamp(value: string): string {
  const timestamp = new Date(value)
  return Number.isNaN(timestamp.getTime()) ? value : timestamp.toLocaleString()
}

function MetadataValue({ label, children }: { readonly label: string; readonly children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="type-label text-fg-faint">{label}</dt>
      <dd className="mt-1 break-words font-mono text-[length:var(--density-type-caption)] text-fg-dim">{children}</dd>
    </div>
  )
}

function unavailableReasonText(artifact: Extract<LogArtifact, { readonly contentState: 'unavailable' }>) {
  switch (artifact.unavailableReason) {
    case 'streaming_response_not_assembled':
      return 'Streaming response was not assembled for retention.'
    case 'response_body_not_bounded':
      return 'Response body exceeded the bounded capture policy.'
    case 'capture_content_limit_exceeded':
      return 'Artifact content exceeded the configured capture limit.'
    case 'capture_memory_budget_exceeded':
      return 'Artifact capture exceeded the configured memory budget.'
    case 'artifact_capture_disabled':
      return 'Artifact capture was disabled when this record was created.'
    case 'artifact_capture_failed':
      return 'Artifact capture failed before content could be retained.'
    default:
      return 'No specific capture reason was recorded.'
  }
}

export function LogArtifactMetadata({ artifact }: LogArtifactMetadataProps) {
  return (
    <article className="py-3" data-artifact-state={artifact.contentState}>
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="break-all font-mono text-[length:var(--density-type-caption-lg)] font-semibold text-foreground">
            {artifact.kind}
          </div>
          <div className="mt-1 break-all font-mono text-[length:var(--density-type-caption)] text-fg-faint">
            {artifact.artifactId.toString()}
          </div>
        </div>
        <LogArtifactStateBadges artifact={artifact} />
      </header>

      <dl className="mt-3 grid gap-x-[var(--shell-normal)] gap-y-2 sm:grid-cols-2 lg:grid-cols-4">
        <MetadataValue label="Captured">
          <time dateTime={artifact.occurredAt}>{formatTimestamp(artifact.occurredAt)}</time>
        </MetadataValue>
        <MetadataValue label="Media">{artifact.mediaKind ?? 'Unknown'}</MetadataValue>
        <MetadataValue label="Bytes / version">
          {artifact.bytes} B / v{artifact.version}
        </MetadataValue>
        <MetadataValue label="Request ID">{artifact.requestId.toString()}</MetadataValue>
        <div className="min-w-0 sm:col-span-2 lg:col-span-4">
          <dt className="type-label text-fg-faint">Checksum</dt>
          <dd className="mt-1 break-all font-mono text-[length:var(--density-type-caption)] text-fg-dim">
            {artifact.checksum ?? 'Not recorded'}
          </dd>
        </div>
      </dl>

      {artifact.contentState === 'unavailable' ? (
        <p className="mt-3 type-caption text-fg-dim" role="status">
          <span className="type-label text-fg-faint">Content unavailable: </span>
          {unavailableReasonText(artifact)}
        </p>
      ) : null}

      {artifact.contentState === 'available' && artifact.redacted ? (
        <div className="mt-3">
          <LogArtifactDownloadControl artifact={artifact} />
        </div>
      ) : null}
    </article>
  )
}

export function LogArtifactInventory({ classified }: LogArtifactInventoryProps) {
  const titleId = useId()
  const groups = [
    ['Request artifacts', classified.request],
    ['Response artifacts', classified.response],
    ['Error artifacts', classified.error],
    ['Unclassified artifacts', classified.unclassified]
  ] as const
  if (groups.every(([, group]) => group.artifacts.length === 0)) return null

  return (
    <section
      aria-labelledby={titleId}
      className="rounded-[var(--radius-lg)] border border-border bg-panel px-[var(--panel-x)]"
    >
      <header className="border-b border-border-soft py-[var(--panel-y)]">
        <h3 className="type-panel-title text-foreground" id={titleId}>
          Retained artifact metadata
        </h3>
        <p className="type-caption mt-1 text-fg-dim">
          All records remain visible, including non-primary and diagnostic artifacts.
        </p>
      </header>
      {groups.map(([label, group]) =>
        group.artifacts.length === 0 ? null : (
          <section className="border-b border-border-soft py-2 last:border-b-0" key={label}>
            <h4 className="type-label text-fg-faint">{label}</h4>
            <div className="divide-y divide-border-soft">
              {group.artifacts.map((artifact) => (
                <LogArtifactMetadata artifact={artifact} key={artifact.artifactId.toString()} />
              ))}
            </div>
          </section>
        )
      )}
    </section>
  )
}
