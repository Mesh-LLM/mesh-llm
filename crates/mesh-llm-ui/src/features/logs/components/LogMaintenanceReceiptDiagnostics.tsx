import type { LogCleanupReceipt, LogDeleteReceipt } from '@/features/logs/api/schemas'
import { hasRetryableArtifactWork } from '@/features/logs/components/LogMaintenanceReceiptEligibility'
import { cn } from '@/lib/cn'

type LogMaintenanceReceipt = LogCleanupReceipt | LogDeleteReceipt

export function LogMaintenanceReceiptDiagnostics({
  className,
  receipt,
  showCounts = true
}: {
  readonly className?: string
  readonly receipt: LogMaintenanceReceipt
  readonly showCounts?: boolean
}) {
  const partial = hasRetryableArtifactWork(receipt)
  return (
    <div
      className={cn(
        'mt-3 space-y-2 rounded-[var(--radius)] border border-border-soft bg-panel-strong/60 p-3',
        className
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="type-label text-fg-faint">Operation ID</span>
        <code className="break-all font-mono text-[length:var(--density-type-caption)] text-foreground">
          {receipt.operationId.toString()}
        </code>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="type-label text-fg-faint">Audit ID</span>
        <code className="break-all font-mono text-[length:var(--density-type-caption)] text-foreground">
          {receipt.auditId?.toString() ?? 'Not assigned'}
        </code>
      </div>
      {showCounts ? (
        <p className="type-caption text-fg-dim">
          Planned {receipt.planned.requests} request(s), {receipt.planned.events} event(s), and{' '}
          {receipt.planned.artifacts} artifact record(s). Executed {receipt.executed.databaseRows} database row
          change(s).
        </p>
      ) : null}
      {partial ? (
        <p className="type-caption text-warn" role="status">
          Partial cascade: {receipt.artifactDeletion.removed} artifact file(s) removed and{' '}
          {receipt.artifactDeletion.failed} could not be removed
          {receipt.artifactDeletion.failureClass ? ` (${receipt.artifactDeletion.failureClass})` : ''}.
        </p>
      ) : null}
      {receipt.state === 'pending' ? (
        <p className="type-caption text-warn" role="status">
          Deletion is durably prepared but still pending. Retry with this operation ID to check or resume execution.
        </p>
      ) : null}
    </div>
  )
}
