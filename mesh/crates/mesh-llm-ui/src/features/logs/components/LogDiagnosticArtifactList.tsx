import { DatabaseZap } from 'lucide-react'
import { useMemo } from 'react'
import type { LogArtifact } from '@/features/logs/api/schemas'
import { LogArtifactStateBadges } from '@/features/logs/components/LogArtifactStateBadges'
import { sortByOccurredAt } from '@/features/logs/lib/log-instant'

export type LogDiagnosticArtifactListProps = {
  readonly artifacts: readonly LogArtifact[]
  readonly loading: boolean
  readonly error: boolean
}

function ArtifactQueryState({ loading, error }: { readonly loading: boolean; readonly error: boolean }) {
  if (!loading && !error) return null
  return (
    <p className="type-caption text-fg-dim" role={error ? 'alert' : 'status'}>
      {error ? 'Error artifact metadata could not be loaded.' : 'Loading error artifact metadata.'}
    </p>
  )
}

function formatTimestamp(value: string): string {
  return new Date(value).toLocaleString()
}

function DiagnosticArtifact({ artifact }: { readonly artifact: LogArtifact }) {
  return (
    <article className="rounded-[var(--radius)] border border-border-soft bg-panel-strong/40 p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <code className="min-w-0 break-all font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
          {artifact.kind}
        </code>
        <LogArtifactStateBadges artifact={artifact} />
      </div>
      <dl className="mt-3 grid gap-x-[var(--shell-normal)] gap-y-2 sm:grid-cols-2">
        <div className="min-w-0">
          <dt className="type-label text-fg-faint">Captured</dt>
          <dd className="mt-1 break-words font-mono text-[length:var(--density-type-caption)] text-fg-dim">
            <time dateTime={artifact.occurredAt}>{formatTimestamp(artifact.occurredAt)}</time>
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="type-label text-fg-faint">Bytes / version</dt>
          <dd className="mt-1 break-words font-mono text-[length:var(--density-type-caption)] text-fg-dim">
            {artifact.bytes} B / v{artifact.version}
          </dd>
        </div>
        <div className="min-w-0 sm:col-span-2">
          <dt className="type-label text-fg-faint">Checksum</dt>
          <dd className="mt-1 break-all font-mono text-[length:var(--density-type-caption)] text-fg-dim">
            {artifact.checksum ?? 'Not recorded'}
          </dd>
        </div>
      </dl>
    </article>
  )
}

export function LogDiagnosticArtifactList({ artifacts, loading, error }: LogDiagnosticArtifactListProps) {
  const orderedArtifacts = useMemo(() => sortByOccurredAt(artifacts), [artifacts])
  const ready = !loading && !error

  return (
    <section aria-label="Error artifacts" className="rounded-[var(--radius)] border border-border-soft bg-panel p-3">
      <div className="flex items-center gap-2">
        <DatabaseZap aria-hidden="true" className="size-4 text-fg-faint" />
        <h3 className="type-panel-title text-foreground">Error artifacts</h3>
      </div>
      <p className="mt-1 type-caption text-fg-dim">
        Retained metadata only. Diagnostics never requests or renders artifact body content.
      </p>
      <div className="mt-3">
        <ArtifactQueryState error={error} loading={loading} />
        {ready && orderedArtifacts.length === 0 ? (
          <p className="type-caption text-fg-dim" role="status">
            No error artifact metadata was retained.
          </p>
        ) : null}
        {ready && orderedArtifacts.length > 0 ? (
          <div className="grid gap-2">
            {orderedArtifacts.map((artifact) => (
              <DiagnosticArtifact artifact={artifact} key={artifact.artifactId.toString()} />
            ))}
          </div>
        ) : null}
      </div>
    </section>
  )
}
