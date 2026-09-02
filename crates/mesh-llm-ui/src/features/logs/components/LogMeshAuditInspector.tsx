import type { LogAuditEntry } from '@/features/logs/api/schemas'
import { LogNetworkIdentityBand } from '@/features/logs/components/LogNetworkIdentityBand'
import { humanizeAuditCode, meshAuditPresentation } from '@/features/logs/lib/log-mesh-audit-presentation'
import { formatElapsedMs } from '@/lib/format-duration'

function policyVerdict(audit: LogAuditEntry, fallback: string): string {
  if (audit.code !== 'gossip_policy_rejected' || !audit.reasonCode) return fallback
  if (audit.reasonCode === 'owner_attestation_required') {
    return 'This node rejected the peer because owner attestation was required and the peer offered none.'
  }
  return `This node rejected the peer because ${humanizeAuditCode(audit.reasonCode)}.`
}

function peerOccurrenceCount(audit: LogAuditEntry, auditEntries: readonly LogAuditEntry[]): number | undefined {
  if (!audit.subjectId) return undefined
  return auditEntries.filter((entry) => entry.subjectKind === 'mesh_peer' && entry.subjectId === audit.subjectId).length
}

function MeshAuditSignals({ audit }: { readonly audit: LogAuditEntry }) {
  const fields: Array<readonly [string, string]> = [
    ...Object.entries(audit.numericSummaries ?? {}).map(
      ([key, value]) => [humanizeAuditCode(key), String(value)] as const
    ),
    ...(audit.reasonCode ? ([['Reason', humanizeAuditCode(audit.reasonCode)]] as const) : []),
    ...(audit.outcome ? ([['Outcome', audit.outcome]] as const) : []),
    ...(audit.durationMs === undefined ? [] : ([['Duration', formatElapsedMs(audit.durationMs)]] as const))
  ]
  if (fields.length === 0) return null

  return (
    <section aria-label="Signals" className="border-b border-border-soft px-4 py-4 sm:px-5">
      <h2 className="type-panel-title text-foreground">Signals</h2>
      <dl className="mt-2 grid gap-x-6 sm:grid-cols-2">
        {fields.map(([label, value]) => (
          <div
            className="min-w-0 border-t border-border-soft py-2 sm:grid sm:grid-cols-[minmax(5.75rem,max-content)_minmax(0,1fr)] sm:items-baseline sm:gap-3"
            key={label}
          >
            <dt className="type-label capitalize text-fg-faint">{label}</dt>
            <dd className="mt-1 break-words font-mono type-caption text-foreground sm:mt-0">{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function MeshAuditMetadata({ audit }: { readonly audit: LogAuditEntry }) {
  const fields: Array<readonly [string, string]> = [
    ['Entry ID', audit.entryId],
    ['Raw code', audit.code],
    ['Source', audit.source],
    ['Occurred', audit.occurredAt],
    ['Sequence', String(audit.sequence)],
    ...(audit.operationId ? ([['Operation ID', audit.operationId]] as const) : []),
    ...(audit.requestId ? ([['Request ID', audit.requestId]] as const) : [])
  ]

  return (
    <section aria-labelledby="audit-metadata-heading" className="px-4 py-4 sm:px-5">
      <h2 className="type-panel-title text-foreground" id="audit-metadata-heading">
        Event metadata
      </h2>
      <dl className="mt-2 grid gap-x-6 sm:grid-cols-2">
        {fields.map(([label, value]) => (
          <div
            className="min-w-0 border-t border-border-soft py-2 sm:grid sm:grid-cols-[minmax(5.75rem,max-content)_minmax(0,1fr)] sm:items-baseline sm:gap-3"
            key={label}
          >
            <dt className="type-label text-fg-faint">{label}</dt>
            <dd className="mt-1 break-words font-mono type-caption text-foreground sm:mt-0">{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

export function LogMeshAuditInspector({
  audit,
  auditEntries
}: {
  readonly audit: LogAuditEntry
  readonly auditEntries: readonly LogAuditEntry[]
}) {
  const presentation = meshAuditPresentation(audit.code)
  const hasPeerContext = Boolean(audit.subjectId || audit.remoteAddr || audit.pathType)

  return (
    <div className="flex min-w-0 flex-col">
      {hasPeerContext ? (
        <LogNetworkIdentityBand
          address={audit.remoteAddr}
          endpointId={audit.subjectId}
          occurrenceCount={peerOccurrenceCount(audit, auditEntries)}
          pathType={audit.pathType}
          title="Peer"
        />
      ) : null}
      <section aria-label="Verdict" className="border-b border-border-soft px-4 py-4 sm:px-5">
        <h2 className="type-panel-title text-foreground">Verdict</h2>
        <p className="type-body mt-2 text-foreground">{policyVerdict(audit, presentation.verdict)}</p>
        <p className="type-body mt-1 text-fg-dim">{presentation.meaning}</p>
        {hasPeerContext ? null : (
          <p className="type-caption mt-2 text-fg-faint">This older record does not include peer context.</p>
        )}
      </section>
      <MeshAuditSignals audit={audit} />
      <MeshAuditMetadata audit={audit} />
    </div>
  )
}
