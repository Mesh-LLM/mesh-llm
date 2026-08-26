import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { CopyInstructionRow } from '@/components/ui/CopyInstructionRow'
import { SharedModalDescription, SharedModalHeader, SharedModalTitle } from '@/components/ui/SharedModal'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import type { LogRequestId } from '@/features/logs/api/ids'
import type { LogOutcome, LogRequest } from '@/features/logs/api/schemas'
import { useLogRequestSummaryQuery } from '@/features/logs/api/use-log-request-details-query'

type LogRequestInspectorHeaderProps = {
  readonly requestId: LogRequestId
  readonly knownRequest?: LogRequest
}

type OutcomePresentation = {
  readonly label: string
  readonly tone: StatusBadgeTone
}

const OUTCOME_PRESENTATION: Record<LogOutcome, OutcomePresentation> = {
  active: { label: 'Active', tone: 'accent' },
  completed: { label: 'Completed', tone: 'good' },
  failed: { label: 'Failed', tone: 'bad' },
  rejected: { label: 'Rejected', tone: 'bad' },
  cancelled: { label: 'Cancelled', tone: 'warn' },
  dropped: { label: 'Dropped', tone: 'bad' }
}

export function LogRequestInspectorHeader({ requestId, knownRequest }: LogRequestInspectorHeaderProps) {
  const summaryQuery = useLogRequestSummaryQuery(requestId, knownRequest)
  const outcome = summaryQuery.data ? OUTCOME_PRESENTATION[summaryQuery.data.outcome] : undefined

  return (
    <SharedModalHeader className="relative min-w-0 shrink-0 px-4 pb-3 pt-3 sm:px-5 sm:pb-4 sm:pt-4.5">
      <div className="flex min-w-0 flex-wrap items-start gap-3 pr-16 lg:pr-12">
        <SharedModalTitle className="min-w-0 flex-1 break-words">Request Inspector</SharedModalTitle>
        {outcome ? (
          <StatusBadge className="max-w-full shrink-0" dot size="caption" tone={outcome.tone}>
            {outcome.label}
          </StatusBadge>
        ) : null}
      </div>
      <SharedModalDescription className="mt-1.5 max-w-3xl break-words pr-16 lg:pr-0">
        Inspect the request overview, payloads, timeline, and diagnostics.
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
      <div className="mt-2.5 min-w-0 max-w-3xl sm:mt-3">
        <CopyInstructionRow label="Request ID" value={requestId.toString()} />
      </div>
    </SharedModalHeader>
  )
}
