import { X } from 'lucide-react'
import { SharedModalActionStrip } from '@/components/ui/SharedModal'
import { Button } from '@/components/ui/button'
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
      className="shrink-0 items-stretch px-4 sm:items-center sm:px-5"
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
