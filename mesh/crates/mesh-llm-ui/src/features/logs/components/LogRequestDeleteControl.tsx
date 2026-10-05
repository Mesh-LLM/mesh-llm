import * as DialogPrimitive from '@radix-ui/react-dialog'
import { Trash2 } from 'lucide-react'
import { useId, useRef, useState } from 'react'
import {
  SharedModal,
  SharedModalActionStrip,
  SharedModalBody,
  SharedModalContent,
  SharedModalDescription,
  SharedModalHeader,
  SharedModalTitle
} from '@/components/ui/SharedModal'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { LogsApiClient } from '@/features/logs/api/client'
import { LogOperationId, type LogRequestId } from '@/features/logs/api/ids'
import type { LogDeleteReceipt } from '@/features/logs/api/schemas'
import { LogMaintenanceReceiptDiagnostics } from '@/features/logs/components/LogMaintenanceReceiptDiagnostics'
import { canRetryDeleteReceipt } from '@/features/logs/components/LogMaintenanceReceiptEligibility'
import { cn } from '@/lib/cn'

type ActionState = { readonly message: string; readonly tone: 'success' | 'error' } | undefined

type LogRequestDeleteControlProps = {
  readonly requestId: LogRequestId
  readonly onMaintenanceMutationSucceeded?: () => void
  readonly triggerClassName?: string
}

type FrozenLogOperation = {
  readonly operationId: LogOperationId
  readonly reason: string
}

function newOperationId() {
  return LogOperationId.create()
}

function isReasonValid(reason: string) {
  return reason.trim().length > 0
}

function actionError(error: unknown) {
  return error instanceof Error ? error.message : 'The local log service did not complete the operation.'
}

export function LogRequestDeleteControl({
  requestId,
  onMaintenanceMutationSucceeded,
  triggerClassName
}: LogRequestDeleteControlProps) {
  const reasonInputId = useId()
  const [open, setOpen] = useState(false)
  const [reason, setReason] = useState('')
  const [receipt, setReceipt] = useState<LogDeleteReceipt>()
  const [operation, setOperation] = useState<FrozenLogOperation>()
  const [action, setAction] = useState<ActionState>()
  const [pending, setPending] = useState(false)
  const triggerRef = useRef<HTMLButtonElement | null>(null)

  async function submitDeletion(nextOperation: FrozenLogOperation) {
    setPending(true)
    setAction(undefined)
    try {
      const nextReceipt = await new LogsApiClient().deleteRequest(requestId, nextOperation)
      setReceipt(nextReceipt)
      setOperation(
        (currentOperation) => currentOperation ?? { operationId: nextReceipt.operationId, reason: nextOperation.reason }
      )
      setAction({
        tone: 'success',
        message:
          nextReceipt.state === 'pending'
            ? 'Deletion accepted and still pending. Retry to check or resume this operation.'
            : nextReceipt.state === 'partial'
              ? 'Request records removed; artifact cleanup is partial and retryable.'
              : 'Request removed.'
      })
      if (nextReceipt.state !== 'pending') onMaintenanceMutationSucceeded?.()
    } catch (error) {
      setAction({ tone: 'error', message: actionError(error) })
    } finally {
      setPending(false)
    }
  }

  async function deleteRequest() {
    if (!isReasonValid(reason)) return
    await submitDeletion({ operationId: newOperationId(), reason: reason.trim() })
  }

  async function retryDeletion() {
    if (!receipt || !operation || !canRetryDeleteReceipt(receipt)) return
    await submitDeletion(operation)
  }

  return (
    <>
      <Button
        ref={triggerRef}
        className={cn('ui-control-destructive', triggerClassName)}
        onClick={() => setOpen(true)}
        size="sm"
        type="button"
        variant="outline"
      >
        <Trash2 className="size-3.5" aria-hidden="true" />
        Delete terminal request
      </Button>
      <SharedModal open={open} onOpenChange={setOpen}>
        <SharedModalContent
          onCloseAutoFocus={(event) => {
            if (!triggerRef.current) return
            event.preventDefault()
            triggerRef.current.focus()
          }}
        >
          <SharedModalHeader>
            <SharedModalTitle>Delete terminal request?</SharedModalTitle>
            <SharedModalDescription>
              This removes the selected durable request and its retained child records. Review and confirm with an audit
              reason.
            </SharedModalDescription>
          </SharedModalHeader>
          <SharedModalBody className="space-y-3">
            <code className="block break-all font-mono text-[length:var(--density-type-caption)] text-fg-dim">
              {requestId.toString()}
            </code>
            {!receipt ? (
              <label
                className="grid gap-1.5 text-[length:var(--density-type-caption)] text-fg-dim"
                htmlFor={reasonInputId}
              >
                <span className="type-label text-fg-faint">Required audit reason</span>
                <Input
                  id={reasonInputId}
                  onChange={(event) => setReason(event.currentTarget.value)}
                  placeholder="Why remove this request?"
                  value={reason}
                />
              </label>
            ) : null}
            {receipt ? <LogMaintenanceReceiptDiagnostics receipt={receipt} /> : null}
            {action ? (
              <p className={`type-caption ${action.tone === 'error' ? 'text-bad' : 'text-good'}`} role="status">
                {action.message}
              </p>
            ) : null}
          </SharedModalBody>
          <SharedModalActionStrip>
            <DialogPrimitive.Close asChild>
              <Button className="ui-control" size="sm" type="button" variant="outline">
                Cancel
              </Button>
            </DialogPrimitive.Close>
            {!receipt ? (
              <Button
                className="ui-control-destructive"
                disabled={!isReasonValid(reason) || pending}
                onClick={() => void deleteRequest()}
                size="sm"
                type="button"
                variant="outline"
              >
                {pending ? 'Deleting…' : 'Confirm deletion'}
              </Button>
            ) : canRetryDeleteReceipt(receipt) ? (
              <Button
                className="ui-control-destructive"
                disabled={!operation || pending}
                onClick={() => void retryDeletion()}
                size="sm"
                type="button"
                variant="outline"
              >
                {pending ? 'Retrying…' : 'Retry deletion'}
              </Button>
            ) : null}
          </SharedModalActionStrip>
        </SharedModalContent>
      </SharedModal>
    </>
  )
}
