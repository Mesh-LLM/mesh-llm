import { X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SharedModalActionStrip } from '@/components/ui/SharedModal'
import type { LogRequest } from '@/features/logs/api/schemas'
import { LogRequestDeleteControl } from '@/features/logs/components/LogRequestDeleteControl'

type LogRequestInspectorFooterProps = {
  readonly request: LogRequest | undefined
  readonly onClose: () => void
  readonly onMaintenanceMutationSucceeded?: () => void
}

const INSPECTOR_ACTION_CLASS = 'min-h-11 w-full min-w-0 gap-1.5 sm:w-auto sm:min-w-24 lg:min-h-8'

export function LogRequestInspectorFooter({
  request,
  onClose,
  onMaintenanceMutationSucceeded
}: LogRequestInspectorFooterProps) {
  const canDelete = request?.source === 'durable' && request.outcome !== 'active'

  return (
    <SharedModalActionStrip
      aria-label="Request inspector actions"
      className="min-w-0 shrink-0 items-stretch gap-2 px-4 py-2.5 sm:flex-wrap sm:items-center sm:px-5 sm:py-3"
      role="contentinfo"
    >
      <Button
        className={`ui-control ${INSPECTOR_ACTION_CLASS}`}
        onClick={onClose}
        size="sm"
        type="button"
        variant="outline"
      >
        <X aria-hidden="true" className="size-3.5" />
        Close
      </Button>
      {canDelete ? (
        <LogRequestDeleteControl
          onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded}
          requestId={request.requestId}
          triggerClassName={INSPECTOR_ACTION_CLASS}
        />
      ) : null}
    </SharedModalActionStrip>
  )
}
