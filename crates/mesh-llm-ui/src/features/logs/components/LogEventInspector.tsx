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

const INSPECTOR_FRAME_CLASS =
  'flex h-dvh w-full flex-col overflow-hidden rounded-none border-0 sm:h-[min(calc(100dvh-4rem),50rem)] sm:w-[calc(100vw-2rem)] sm:max-w-[720px] sm:rounded-[var(--radius-lg)] sm:border'

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
          className={`${INSPECTOR_FRAME_CLASS} data-[state=closed]:zoom-out-100 data-[state=open]:zoom-in-100`}
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

function AuditInspectorHeader({ title }: { readonly title: string }) {
  return (
    <SharedModalHeader className="relative min-w-0 shrink-0 pr-16 lg:pr-14">
      <SharedModalTitle className="min-w-0 break-words">{title}</SharedModalTitle>
      <SharedModalDescription>Privacy-safe operational metadata from the bounded loaded window.</SharedModalDescription>
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
      const title = `Operational event ${audit?.code ?? inspector.id}`
      return (
        <>
          <AuditInspectorHeader title={title} />
          <SharedModalBody
            aria-label="Operational event metadata"
            className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-5"
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
  const fields = [
    ['Entry ID', audit.entryId],
    ['Occurred', audit.occurredAt],
    ['Source', audit.source],
    ['Code', audit.code],
    ['Severity', audit.severity ?? 'Not provided'],
    ['Sequence', String(audit.sequence)]
  ] as const

  return (
    <dl className="grid gap-x-[var(--shell-normal)] gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
      {fields.map(([label, value]) => (
        <div className="min-w-0" key={label}>
          <dt className="type-label text-fg-faint">{label}</dt>
          <dd className="mt-1 break-words font-mono text-[length:var(--density-type-caption-lg)] text-foreground">
            {value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function AuditOutsideWindow() {
  return (
    <div role="status">
      <div className="type-panel-title text-foreground">Operational event is outside the loaded window</div>
      <p className="type-body mt-1 text-fg-dim">Return to the ledger and load a window containing this entry.</p>
    </div>
  )
}

function assertNever(value: never): never {
  throw new Error(`Unhandled log inspector: ${String(value)}`)
}
