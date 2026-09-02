import type { LogAuditEntry } from '@/features/logs/api/schemas'
import { cn } from '@/lib/cn'

export function LogAuditMetadata({ audit }: { readonly audit: LogAuditEntry }) {
  const metadataFields: Array<readonly [string, string]> = [
    ['Entry ID', audit.entryId],
    ['Source', audit.source],
    ['Occurred', audit.occurredAt],
    ['Sequence', String(audit.sequence)],
    ...(audit.subjectKind ? ([['Subject kind', audit.subjectKind]] as const) : []),
    ...(audit.subjectId ? ([['Subject ID', audit.subjectId]] as const) : []),
    ...(audit.operationId ? ([['Operation ID', audit.operationId]] as const) : []),
    ...(audit.requestId ? ([['Request ID', audit.requestId]] as const) : []),
    ...(audit.reasonCode ? ([['Reason', audit.reasonCode]] as const) : []),
    ...(audit.durationMs !== undefined ? ([['Duration', `${audit.durationMs} ms`]] as const) : []),
    ...Object.entries(audit.numericSummaries ?? {}).map(([key, value]) => [`Summary · ${key}`, String(value)] as const)
  ]

  return (
    <div className="min-w-0">
      <section aria-labelledby="audit-metadata-heading" className="min-w-0 px-4 py-4 sm:px-5">
        <h2 className="type-panel-title text-foreground" id="audit-metadata-heading">
          Event metadata
        </h2>
        {audit.commandSummary ? (
          <div className="mt-3 rounded-[var(--radius)] border border-border-soft bg-panel-strong px-3 py-3">
            <div className="type-label mb-1.5 text-fg-faint">Command</div>
            <div className="min-w-0 font-mono text-[length:var(--density-type-control-lg)] text-foreground [overflow-wrap:anywhere]">
              <code>{audit.commandSummary}</code>
            </div>
          </div>
        ) : null}
        <dl className="mt-3 grid min-w-0 gap-x-6 gap-y-0 sm:grid-cols-2">
          {metadataFields.map(([label, value]) => {
            const isFullRow = label === 'Request ID'
            return (
              <div
                className={cn(
                  'min-w-0 border-t border-border-soft py-2.5 sm:grid sm:grid-cols-[minmax(5.75rem,max-content)_minmax(0,1fr)] sm:items-baseline sm:gap-3',
                  isFullRow && 'sm:col-span-2'
                )}
                key={label}
              >
                <dt className="type-label text-fg-faint">{label}</dt>
                <dd className="mt-1 min-w-0 font-mono text-[length:var(--density-type-caption-lg)] text-foreground [overflow-wrap:anywhere] sm:mt-0">
                  {value}
                </dd>
              </div>
            )
          })}
        </dl>
      </section>
    </div>
  )
}
