import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useState, type RefObject } from 'react'
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
import { LogsApiClient, type LogsRequestQuery } from '@/features/logs/api/client'
import { LogOperationId } from '@/features/logs/api/ids'
import type { LogCleanupReceipt } from '@/features/logs/api/schemas'
import { cleanupScopeFromQuery, supportsCleanup } from '@/features/logs/components/LogCleanupScope'
import { LogMaintenanceReceiptDiagnostics } from '@/features/logs/components/LogMaintenanceReceiptDiagnostics'
import { hasRetryableArtifactWork } from '@/features/logs/components/LogMaintenanceReceiptEligibility'

type ActionState = { readonly message: string; readonly tone: 'success' | 'error' } | undefined

type LogCleanupDialogProps = {
  readonly open: boolean
  readonly onOpenChange: (open: boolean) => void
  readonly onMaintenanceMutationSucceeded?: () => void
  readonly query: LogsRequestQuery
  readonly returnFocusRef: RefObject<HTMLButtonElement | null>
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

export function LogCleanupDialog({
  open,
  onOpenChange,
  onMaintenanceMutationSucceeded,
  query,
  returnFocusRef
}: LogCleanupDialogProps) {
  const [cutoffBefore, setCutoffBefore] = useState('')
  const [requestLimit, setRequestLimit] = useState('')
  const [reason, setReason] = useState('')
  const [preview, setPreview] = useState<LogCleanupReceipt>()
  const [operation, setOperation] = useState<FrozenLogOperation>()
  const [action, setAction] = useState<ActionState>()
  const [pending, setPending] = useState(false)
  const parsedLimit = Number(requestLimit)
  const validScope =
    supportsCleanup(query) &&
    !Number.isNaN(Date.parse(cutoffBefore)) &&
    Number.isSafeInteger(parsedLimit) &&
    parsedLimit > 0

  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen) {
      setPreview(undefined)
      setOperation(undefined)
      setAction(undefined)
    }
    onOpenChange(nextOpen)
  }

  async function previewCleanup() {
    if (!validScope || !isReasonValid(reason)) return
    setPending(true)
    setAction(undefined)
    try {
      const nextOperation = { operationId: newOperationId(), reason: reason.trim() }
      const receipt = await new LogsApiClient().previewCleanup({
        operationId: nextOperation.operationId,
        cutoffBefore,
        requestLimit: parsedLimit,
        ...cleanupScopeFromQuery(query),
        reason: nextOperation.reason
      })
      setPreview(receipt)
      setOperation({ operationId: receipt.operationId, reason: nextOperation.reason })
    } catch (error) {
      setAction({ tone: 'error', message: actionError(error) })
    } finally {
      setPending(false)
    }
  }

  async function runCleanup() {
    if (!preview || !operation) return
    setPending(true)
    setAction(undefined)
    try {
      const receipt = await new LogsApiClient().runCleanup(operation)
      setPreview(receipt)
      setAction({
        tone: 'success',
        message: receipt.state === 'partial' ? 'Cleanup completed with diagnostics.' : 'Cleanup completed.'
      })
      onMaintenanceMutationSucceeded?.()
    } catch (error) {
      setAction({ tone: 'error', message: actionError(error) })
    } finally {
      setPending(false)
    }
  }

  return (
    <SharedModal open={open} onOpenChange={handleOpenChange}>
      <SharedModalContent
        onCloseAutoFocus={(event) => {
          if (!returnFocusRef.current) return
          event.preventDefault()
          returnFocusRef.current.focus()
        }}
      >
        <SharedModalHeader>
          <SharedModalTitle>{preview ? 'Confirm scoped cleanup' : 'Preview scoped cleanup'}</SharedModalTitle>
          <SharedModalDescription>
            {preview
              ? 'Review the recorded selection before the server executes this same audited operation.'
              : 'Cleanup applies only to terminal records before the supplied cutoff, within the server-validated request scope.'}
          </SharedModalDescription>
        </SharedModalHeader>
        <SharedModalBody className="space-y-3">
          {!preview ? (
            <>
              <label
                className="grid gap-1.5 text-[length:var(--density-type-caption)] text-fg-dim"
                htmlFor="log-cleanup-cutoff"
              >
                <span className="type-label text-fg-faint">Delete terminal logs before</span>
                <Input
                  id="log-cleanup-cutoff"
                  className="border-border bg-panel-strong font-mono"
                  onChange={(event) => setCutoffBefore(event.currentTarget.value)}
                  placeholder="2026-08-01T00:00:00Z"
                  value={cutoffBefore}
                />
              </label>
              <label
                className="grid gap-1.5 text-[length:var(--density-type-caption)] text-fg-dim"
                htmlFor="log-cleanup-limit"
              >
                <span className="type-label text-fg-faint">Request scope</span>
                <Input
                  id="log-cleanup-limit"
                  inputMode="numeric"
                  min="1"
                  onChange={(event) => setRequestLimit(event.currentTarget.value)}
                  placeholder="Number of matching requests"
                  type="number"
                  value={requestLimit}
                />
              </label>
              <label
                className="grid gap-1.5 text-[length:var(--density-type-caption)] text-fg-dim"
                htmlFor="log-cleanup-reason"
              >
                <span className="type-label text-fg-faint">Required audit reason</span>
                <Input
                  id="log-cleanup-reason"
                  onChange={(event) => setReason(event.currentTarget.value)}
                  placeholder="Why is this scoped cleanup needed?"
                  value={reason}
                />
              </label>
            </>
          ) : (
            <>
              <p className="type-caption text-fg-dim">
                Server-recorded durable scope: cutoff{' '}
                <code className="font-mono text-foreground">{preview.scope.cutoffBefore}</code> · up to{' '}
                {preview.scope.requestLimit} request(s) ·{' '}
                {preview.hasMore ? 'more matching records remain' : 'no additional matching records'}.
              </p>
              <p className="type-caption text-fg-dim">
                Filters:{' '}
                {[
                  preview.scope.from && `from ${preview.scope.from}`,
                  preview.scope.to && `to ${preview.scope.to}`,
                  preview.scope.route && `route ${preview.scope.route}`,
                  preview.scope.model && `model ${preview.scope.model}`,
                  preview.scope.provider && `provider ${preview.scope.provider}`,
                  preview.scope.engine && `engine ${preview.scope.engine}`,
                  preview.scope.outcome && `outcome ${preview.scope.outcome}`
                ]
                  .filter(Boolean)
                  .join(' · ') || 'none'}
              </p>
              <LogMaintenanceReceiptDiagnostics receipt={preview} />
            </>
          )}
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
          {preview?.state === 'previewed' ? (
            <Button
              className="ui-control-destructive"
              disabled={!operation || pending}
              onClick={() => void runCleanup()}
              size="sm"
              type="button"
              variant="outline"
            >
              {pending ? 'Cleaning…' : 'Confirm cleanup'}
            </Button>
          ) : preview && hasRetryableArtifactWork(preview) ? (
            <Button
              className="ui-control-destructive"
              disabled={!operation || pending}
              onClick={() => void runCleanup()}
              size="sm"
              type="button"
              variant="outline"
            >
              {pending ? 'Retrying…' : 'Retry cleanup'}
            </Button>
          ) : !preview ? (
            <Button
              className="ui-control-primary"
              disabled={!validScope || !isReasonValid(reason) || pending}
              onClick={() => void previewCleanup()}
              size="sm"
              type="button"
            >
              {pending ? 'Previewing…' : 'Preview cleanup'}
            </Button>
          ) : null}
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}
