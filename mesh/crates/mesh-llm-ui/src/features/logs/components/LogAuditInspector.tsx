import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import {
  SharedModalBody,
  SharedModalDescription,
  SharedModalHeader,
  SharedModalTitle
} from '@/components/ui/SharedModal'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import { Button } from '@/components/ui/button'
import type { LogAuditEntry } from '@/features/logs/api/schemas'
import { LogAuditMetadata } from '@/features/logs/components/LogAuditMetadata'
import { LogEventCategoryBadge } from '@/features/logs/components/LogEventCategoryBadge'
import { LogMeshAuditInspector } from '@/features/logs/components/LogMeshAuditInspector'
import { classifyAuditCategory, type OperationalLogEventCategory } from '@/features/logs/lib/log-event-ledger'
import { meshAuditPresentation } from '@/features/logs/lib/log-mesh-audit-presentation'

type LogAuditInspectorProps = {
  readonly audit: LogAuditEntry | undefined
  readonly auditEntries: readonly LogAuditEntry[]
  readonly code: string
}

const SEVERITY_TONES: Readonly<Record<string, StatusBadgeTone>> = {
  info: 'muted',
  warning: 'warn',
  error: 'bad'
}

function severityTone(value: string): StatusBadgeTone {
  return SEVERITY_TONES[value.trim().toLowerCase()] ?? 'muted'
}

function isPeerCategory(category: OperationalLogEventCategory): category is 'gossip' | 'quic' {
  return category === 'gossip' || category === 'quic'
}

function AuditInspectorHeader({
  audit,
  category,
  code
}: {
  readonly audit: LogAuditEntry | undefined
  readonly category: OperationalLogEventCategory
  readonly code: string
}) {
  const peerCategory = audit !== undefined && isPeerCategory(category)
  const title = peerCategory ? meshAuditPresentation(code).title : code

  return (
    <SharedModalHeader className="relative min-w-0 shrink-0 pr-16 lg:pr-14">
      <SharedModalTitle aria-label={`Operational event ${title}`} className="min-w-0 break-words">
        {title}
      </SharedModalTitle>
      <SharedModalDescription>
        {peerCategory
          ? 'Peer identity, connection evidence, and the operational meaning of this event.'
          : 'Recorded state, timing, source, and related identifiers for this event.'}
      </SharedModalDescription>
      {peerCategory ? (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <LogEventCategoryBadge category={category} />
          <StatusBadge dot size="caption" tone={severityTone(audit.severity ?? 'info')}>
            {audit.severity ?? 'Not provided'}
          </StatusBadge>
        </div>
      ) : null}
      {!peerCategory && audit !== undefined ? (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <LogEventCategoryBadge category={category} />
          <StatusBadge dot tone={severityTone(audit.severity ?? 'info')}>
            {audit.severity ?? 'Not provided'}
          </StatusBadge>
        </div>
      ) : null}
      <DialogPrimitive.Close asChild>
        <Button
          aria-label="Close inspector"
          className="ui-control-ghost absolute right-2 top-2 size-13 rounded-[var(--radius)] text-fg-dim lg:right-4 lg:top-4 lg:size-8"
          size="icon"
          type="button"
          variant="ghost"
        >
          <X aria-hidden="true" className="size-4" />
        </Button>
      </DialogPrimitive.Close>
    </SharedModalHeader>
  )
}

function AuditOutsideWindow() {
  return (
    <div className="px-4 py-5 sm:px-5" role="status">
      <div className="type-panel-title text-foreground">Operational event is outside the loaded window</div>
      <p className="type-body mt-1 text-fg-dim">Return to the ledger and load a window containing this entry.</p>
    </div>
  )
}

export function LogAuditInspector({ audit, auditEntries, code }: LogAuditInspectorProps) {
  const category = classifyAuditCategory(code)
  return (
    <>
      <AuditInspectorHeader audit={audit} category={category} code={code} />
      <SharedModalBody
        aria-label="Operational event metadata"
        className="min-h-0 flex-1 overflow-y-auto p-0"
        role="region"
      >
        {audit === undefined ? (
          <AuditOutsideWindow />
        ) : isPeerCategory(category) ? (
          <LogMeshAuditInspector audit={audit} auditEntries={auditEntries} />
        ) : (
          <LogAuditMetadata audit={audit} />
        )}
      </SharedModalBody>
    </>
  )
}
