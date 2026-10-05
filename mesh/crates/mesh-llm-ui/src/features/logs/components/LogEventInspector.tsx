import { useRef } from 'react'
import { SharedModal, SharedModalBody, SharedModalContent } from '@/components/ui/SharedModal'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogAuditEntry, LogRequest } from '@/features/logs/api/schemas'
import { LogAuditInspector } from '@/features/logs/components/LogAuditInspector'
import { LogRequestDetails } from '@/features/logs/components/LogRequestDetails'
import { LogRequestInspectorHeader } from '@/features/logs/components/LogRequestInspectorHeader'
import type { LogInspector, LogRequestDetailTab } from '@/features/logs/lib/log-inspector'

type LogEventInspectorProps = {
  readonly inspector: LogInspector | undefined
  readonly auditEntries: readonly LogAuditEntry[]
  readonly requestRows?: readonly LogRequest[]
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
  requestRows,
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
            requestRows={requestRows}
            requestTab={requestTab}
          />
        </SharedModalContent>
      ) : null}
    </SharedModal>
  )
}

function InspectorContent({
  inspector,
  auditEntries,
  requestRows,
  requestTab,
  onClose,
  onRequestTabChange,
  onMaintenanceMutationSucceeded
}: LogEventInspectorProps & { readonly inspector: LogInspector }) {
  switch (inspector.type) {
    case 'request': {
      const requestId = LogRequestId.parse(inspector.id)
      const knownRequest = requestRows?.find((row) => row.requestId.toString() === requestId.toString())
      return (
        <>
          <LogRequestInspectorHeader knownRequest={knownRequest} requestId={requestId} />
          <SharedModalBody className="flex min-h-0 flex-1 overflow-hidden p-0">
            <LogRequestDetails
              embedded
              knownRequest={knownRequest}
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
      return <LogAuditInspector audit={audit} auditEntries={auditEntries} code={code} />
    }
    default:
      return assertNever(inspector)
  }
}

function assertNever(value: never): never {
  throw new Error(`Unhandled log inspector: ${String(value)}`)
}
