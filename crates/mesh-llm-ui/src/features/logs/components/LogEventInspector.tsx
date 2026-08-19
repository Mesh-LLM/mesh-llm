import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { useRef } from 'react'
import {
  SharedModal,
  SharedModalBody,
  SharedModalContent,
  SharedModalDescription,
  SharedModalHeader,
  SharedModalTitle
} from '@/components/ui/SharedModal'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import { Button } from '@/components/ui/button'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogAuditEntry } from '@/features/logs/api/schemas'
import { LogRequestDetails } from '@/features/logs/components/LogRequestDetails'
import { LogRequestInspectorHeader } from '@/features/logs/components/LogRequestInspectorHeader'
import type { LogInspector, LogRequestDetailTab } from '@/features/logs/lib/log-inspector'

type LogEventInspectorProps = {
  readonly inspector: LogInspector | undefined
  readonly auditEntries: readonly LogAuditEntry[]
  readonly requestTab: LogRequestDetailTab
  readonly onClose: () => void
  readonly onRequestTabChange: (tab: LogRequestDetailTab) => void
  readonly onMaintenanceMutationSucceeded?: () => void
}

const INSPECTOR_FRAME_BASE_CLASS =
  'flex h-dvh w-full flex-col overflow-hidden rounded-none border-0 sm:w-[calc(100vw-2rem)] sm:rounded-[var(--radius-lg)] sm:border'
const REQUEST_INSPECTOR_FRAME_CLASS = `${INSPECTOR_FRAME_BASE_CLASS} sm:h-[min(calc(100dvh-3rem),54rem)] sm:max-w-[1120px]`
const AUDIT_INSPECTOR_FRAME_CLASS = `${INSPECTOR_FRAME_BASE_CLASS} sm:h-auto sm:max-h-[min(calc(100dvh-4rem),50rem)] sm:max-w-[720px]`

export function LogEventInspector({
  inspector,
  auditEntries,
  requestTab,
  onClose,
  onRequestTabChange,
  onMaintenanceMutationSucceeded
}: LogEventInspectorProps) {
  const returnFocusRef = useRef<HTMLElement | null>(null)

  return (
    <SharedModal
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      open={inspector !== undefined}
    >
      {inspector ? (
        <SharedModalContent
          className={`${inspector.type === 'audit' ? AUDIT_INSPECTOR_FRAME_CLASS : REQUEST_INSPECTOR_FRAME_CLASS} data-[state=closed]:zoom-out-100 data-[state=open]:zoom-in-100`}
          data-request-inspector-shell={inspector.type === 'request' ? 'fixed' : undefined}
          onCloseAutoFocus={(event) => {
            if (!returnFocusRef.current) return
            event.preventDefault()
            returnFocusRef.current.focus()
          }}
          onOpenAutoFocus={() => {
            returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
          }}
        >
          <InspectorContent
            auditEntries={auditEntries}
            inspector={inspector}
            onClose={onClose}
            onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded}
            onRequestTabChange={onRequestTabChange}
            requestTab={requestTab}
          />
        </SharedModalContent>
      ) : null}
    </SharedModal>
  )
}

function AuditInspectorHeader({ code }: { readonly code: string }) {
  return (
    <SharedModalHeader className="relative min-w-0 shrink-0 pr-16 lg:pr-14">
      <SharedModalTitle aria-label={`Operational event ${code}`} className="min-w-0 break-words">
        {code}
      </SharedModalTitle>
      <SharedModalDescription>
        Recorded state, timing, source, and related identifiers for this event.
      </SharedModalDescription>
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

function InspectorContent({
  inspector,
  auditEntries,
  requestTab,
  onClose,
  onRequestTabChange,
  onMaintenanceMutationSucceeded
}: LogEventInspectorProps & { readonly inspector: LogInspector }) {
  switch (inspector.type) {
    case 'request': {
      const requestId = LogRequestId.parse(inspector.id)
      return (
        <>
          <LogRequestInspectorHeader requestId={requestId} />
          <SharedModalBody className="flex min-h-0 flex-1 overflow-hidden p-0">
            <LogRequestDetails
              embedded
              onBack={onClose}
              onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded}
              onTabChange={onRequestTabChange}
              requestId={requestId}
              tab={requestTab}
            />
          </SharedModalBody>
        </>
      )
    }
    case 'audit': {
      const audit = auditEntries.find((entry) => entry.entryId === inspector.id)
      const code = audit?.code ?? inspector.id
      return (
        <>
          <AuditInspectorHeader code={code} />
          <SharedModalBody
            aria-label="Operational event metadata"
            className="min-h-0 flex-1 overflow-y-auto p-0"
            role="region"
          >
            {audit ? <AuditMetadata audit={audit} /> : <AuditOutsideWindow />}
          </SharedModalBody>
        </>
      )
    }
    default:
      return assertNever(inspector)
  }
}

function AuditMetadata({ audit }: { readonly audit: LogAuditEntry }) {
  const statusFields: Array<{
    readonly kind: 'severity' | 'outcome'
    readonly label: string
    readonly value: string
  }> = [
    { kind: 'severity', label: 'Severity', value: audit.severity ?? 'Not provided' },
    ...(audit.outcome ? ([{ kind: 'outcome' as const, label: 'Outcome', value: audit.outcome }] as const) : [])
  ]
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
    <div className="flex min-w-0 flex-col">
      <section
        aria-labelledby="audit-state-heading"
        className="border-b border-border-soft bg-panel-strong px-4 py-4 sm:px-5"
      >
        <h2 className="type-panel-title text-foreground" id="audit-state-heading">
          Event state
        </h2>
        <dl className="mt-3 flex min-w-0 flex-wrap gap-x-8 gap-y-3">
          {statusFields.map(({ kind, label, value }) => (
            <div className="min-w-[8rem]" key={label}>
              <dt className="type-label text-fg-faint">{label}</dt>
              <dd className="mt-1">
                <StatusBadge dot size="caption" tone={statusTone(kind, value)}>
                  {value}
                </StatusBadge>
              </dd>
            </div>
          ))}
        </dl>
      </section>
      <section aria-labelledby="audit-metadata-heading" className="min-w-0 px-4 py-4 sm:px-5">
        <h2 className="type-panel-title text-foreground" id="audit-metadata-heading">
          Event metadata
        </h2>
        <dl className="mt-2 grid min-w-0 gap-x-6 sm:grid-cols-2">
          {metadataFields.map(([label, value]) => (
            <div
              className="min-w-0 border-t border-border-soft py-2.5 sm:grid sm:grid-cols-[minmax(5.75rem,max-content)_minmax(0,1fr)] sm:items-baseline sm:gap-3"
              key={label}
            >
              <dt className="type-label text-fg-faint">{label}</dt>
              <dd className="mt-1 min-w-0 font-mono text-[length:var(--density-type-caption-lg)] text-foreground sm:mt-0 [overflow-wrap:anywhere]">
                {value}
              </dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
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

function statusTone(kind: 'severity' | 'outcome', value: string): StatusBadgeTone {
  const normalized = value.trim().toLowerCase()
  return kind === 'severity' ? (SEVERITY_TONES[normalized] ?? 'muted') : (OUTCOME_TONES[normalized] ?? 'muted')
}

const SEVERITY_TONES: Readonly<Record<string, StatusBadgeTone>> = {
  info: 'muted',
  warning: 'warn',
  error: 'bad'
}

const OUTCOME_TONES: Readonly<Record<string, StatusBadgeTone>> = {
  active: 'accent',
  running: 'accent',
  started: 'warn',
  pending: 'warn',
  loading: 'warn',
  cancelled: 'warn',
  canceled: 'warn',
  completed: 'good',
  accepted: 'good',
  ready: 'good',
  success: 'good',
  succeeded: 'good',
  healthy: 'good',
  available: 'good',
  failed: 'bad',
  error: 'bad',
  rejected: 'bad',
  denied: 'bad',
  blocked: 'bad',
  dropped: 'bad'
}

function assertNever(value: never): never {
  throw new Error(`Unhandled log inspector: ${String(value)}`)
}
