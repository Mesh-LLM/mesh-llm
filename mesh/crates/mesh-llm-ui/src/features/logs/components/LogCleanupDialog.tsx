import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useMemo, useRef, useState, type RefObject } from 'react'
import {
  SharedModal,
  SharedModalActionStrip,
  SharedModalBody,
  SharedModalContent,
  SharedModalDescription,
  SharedModalHeader
} from '@/components/ui/SharedModal'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { LogsApiClient, type LogsRequestQuery } from '@/features/logs/api/client'
import { LogOperationId } from '@/features/logs/api/ids'
import type { LogCleanupReceipt, LogMaintenanceCounts } from '@/features/logs/api/schemas'
import { cleanupScopeFromQuery, supportsCleanup } from '@/features/logs/components/LogCleanupScope'
import { LogCleanupWindow } from '@/features/logs/components/LogCleanupWindow'
import { LogMaintenanceReceiptDiagnostics } from '@/features/logs/components/LogMaintenanceReceiptDiagnostics'
import { hasRetryableArtifactWork } from '@/features/logs/components/LogMaintenanceReceiptEligibility'
import type { LogEventCategory, LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'
import {
  cleanupWindowBounds,
  cleanupWindowExclusiveEnd,
  type CleanupWindow
} from '@/features/logs/lib/log-cleanup-window'

type ActionState = { readonly message: string; readonly tone: 'success' | 'warning' | 'error' } | undefined

type LogCleanupDialogProps = {
  readonly open: boolean
  readonly onOpenChange: (open: boolean) => void
  readonly onMaintenanceMutationSucceeded?: () => void
  readonly query: LogsRequestQuery
  readonly returnFocusRef: RefObject<HTMLButtonElement | null>
  readonly rows: readonly LogEventLedgerRow[]
  readonly initialCategories: ReadonlySet<LogEventCategory>
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

const CLEANUP_REQUEST_LIMIT = 100

function formatWindow(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short'
  }).format(new Date(value))
}

function countLabel(value: number, singular: string, plural = `${singular}s`) {
  return `${value} ${value === 1 ? singular : plural}`
}

function linkedRecordTotal(counts: LogMaintenanceCounts) {
  return counts.events + counts.artifacts + counts.proxyRecords
}

function actionToneClass(tone: NonNullable<ActionState>['tone']) {
  if (tone === 'error') return 'text-bad'
  if (tone === 'warning') return 'text-warn'
  return 'text-good'
}

function cleanupDialogTitle(preview: LogCleanupReceipt | undefined) {
  if (!preview) return 'Choose logs to remove'
  if (preview.state === 'previewed' && preview.planned.databaseRows === 0) return 'Nothing to remove'
  if (preview.state === 'previewed') return 'Review log cleanup'
  if (preview.state === 'partial') return 'Cleanup needs attention'
  return 'Cleanup complete'
}

function cleanupDialogDescription(preview: LogCleanupReceipt | undefined) {
  if (!preview) return 'Select a time window and review the result before any terminal request records are deleted.'
  if (preview.state === 'previewed' && preview.planned.databaseRows === 0) {
    return 'The server found no removable terminal request groups in this window.'
  }
  if (preview.state === 'previewed') return 'Nothing has been deleted yet. Confirm what will be removed and retained.'
  if (preview.state === 'partial' && preview.artifactDeletion.failed > 0) {
    return 'Request records were removed, but some linked files still need attention.'
  }
  if (preview.state === 'partial')
    return 'The server completed only part of this batch. Review the result before continuing.'
  return 'The selected terminal request records have been removed.'
}

export function LogCleanupDialog({
  open,
  onOpenChange,
  onMaintenanceMutationSucceeded,
  query,
  returnFocusRef,
  rows,
  initialCategories
}: LogCleanupDialogProps) {
  const bounds = useMemo(() => cleanupWindowBounds(rows, query.from, query.to), [query.from, query.to, rows])
  const [window, setWindow] = useState<CleanupWindow>(bounds)
  const initialCleanupCategories = useMemo(
    () => [...new Set<LogEventCategory>(['requests', ...initialCategories])],
    [initialCategories]
  )
  const [categories, setCategories] = useState<LogEventCategory[]>(initialCleanupCategories)
  const [reason, setReason] = useState('')
  const [preview, setPreview] = useState<LogCleanupReceipt>()
  const [operation, setOperation] = useState<FrozenLogOperation>()
  const [action, setAction] = useState<ActionState>()
  const [pending, setPending] = useState(false)
  const titleRef = useRef<HTMLHeadingElement | null>(null)
  const windowStartRef = useRef<HTMLSpanElement | null>(null)
  const previewIsEmpty = preview?.state === 'previewed' && preview.planned.databaseRows === 0
  const validScope =
    supportsCleanup(query) &&
    categories.includes('requests') &&
    Number.isFinite(window.start) &&
    Number.isFinite(window.end) &&
    window.start < window.end

  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen && pending) return
    if (!nextOpen) {
      setWindow(bounds)
      setCategories(initialCleanupCategories)
      setPreview(undefined)
      setOperation(undefined)
      setAction(undefined)
      setReason('')
    }
    onOpenChange(nextOpen)
  }

  function editSelection() {
    setPreview(undefined)
    setOperation(undefined)
    setAction(undefined)
    requestAnimationFrame(() => windowStartRef.current?.focus())
  }

  async function previewCleanup() {
    if (!validScope || !isReasonValid(reason)) return
    setPending(true)
    setAction(undefined)
    try {
      const nextOperation = { operationId: newOperationId(), reason: reason.trim() }
      const from = new Date(window.start).toISOString()
      const to = cleanupWindowExclusiveEnd(window.end)
      const receipt = await new LogsApiClient().previewCleanup({
        operationId: nextOperation.operationId,
        ...cleanupScopeFromQuery(query),
        cutoffBefore: to,
        requestLimit: CLEANUP_REQUEST_LIMIT,
        from,
        to,
        reason: nextOperation.reason
      })
      setPreview(receipt)
      setOperation({ operationId: receipt.operationId, reason: nextOperation.reason })
      requestAnimationFrame(() => titleRef.current?.focus())
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
        tone: receipt.state === 'partial' ? 'warning' : 'success',
        message:
          receipt.state === 'partial'
            ? receipt.artifactDeletion.failed > 0
              ? `Cleanup removed ${countLabel(receipt.executed.requests, 'request group')}; ${countLabel(receipt.artifactDeletion.failed, 'linked file')} ${receipt.artifactDeletion.failed === 1 ? 'still needs' : 'still need'} attention.`
              : `Cleanup changed ${countLabel(receipt.executed.databaseRows, 'record')}, but the server reported a partial result. Review the audit details before continuing.`
            : 'Log cleanup completed.'
      })
      requestAnimationFrame(() => titleRef.current?.focus())
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
        aria-label="Review log cleanup"
        className="flex max-h-[min(820px,calc(100dvh-2rem))] w-[min(640px,calc(100vw-1.5rem))] flex-col"
        onCloseAutoFocus={(event) => {
          if (!returnFocusRef.current) return
          event.preventDefault()
          returnFocusRef.current.focus()
        }}
      >
        <SharedModalHeader>
          <h2
            className="text-[length:var(--density-type-headline)] font-semibold leading-5 tracking-[-0.02em] text-fg"
            ref={titleRef}
            tabIndex={-1}
          >
            {cleanupDialogTitle(preview)}
          </h2>
          <SharedModalDescription>{cleanupDialogDescription(preview)}</SharedModalDescription>
        </SharedModalHeader>
        <SharedModalBody aria-busy={pending} className="min-h-0 flex-1 space-y-4 overflow-y-auto">
          {!preview ? (
            <>
              <LogCleanupWindow
                bounds={bounds}
                categories={categories}
                onCategoriesChange={setCategories}
                onWindowChange={setWindow}
                rows={rows}
                startThumbRef={windowStartRef}
                window={window}
              />
              <label
                className="grid gap-1.5 text-[length:var(--density-type-caption)] text-fg-dim"
                htmlFor="log-cleanup-reason"
              >
                <span className="type-label text-fg-faint">Reason for removal</span>
                <Input
                  aria-describedby="log-cleanup-reason-help"
                  id="log-cleanup-reason"
                  onChange={(event) => setReason(event.currentTarget.value)}
                  placeholder="Why are these logs being removed?"
                  value={reason}
                />
                <span className="type-annotation text-fg-faint" id="log-cleanup-reason-help">
                  Required for the audit trail. The reason is hidden from the review screen.
                </span>
              </label>
            </>
          ) : (
            <>
              <section aria-labelledby="cleanup-plan-title" className="space-y-3">
                {previewIsEmpty ? (
                  <div
                    className="rounded-[var(--radius)] border border-border-soft bg-panel-strong/55 px-4 py-4"
                    role="status"
                  >
                    <h3 className="type-panel-title text-foreground" id="cleanup-plan-title">
                      No terminal request groups matched
                    </h3>
                    <p className="mt-1 type-caption text-fg-dim">
                      Adjust the time window to search again. No logs will be deleted from this result.
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <h3 className="type-panel-title text-foreground" id="cleanup-plan-title">
                          {countLabel(
                            preview.state === 'previewed' ? preview.planned.requests : preview.executed.requests,
                            'terminal request group'
                          )}{' '}
                          {preview.state === 'previewed' ? 'will be removed' : 'removed'}
                        </h3>
                        <p className="mt-1 type-caption text-fg-dim">
                          {countLabel(
                            linkedRecordTotal(preview.state === 'previewed' ? preview.planned : preview.executed),
                            'linked record'
                          )}{' '}
                          {preview.state === 'previewed' ? 'will be removed with them.' : 'removed with them.'}
                        </p>
                      </div>
                      <span className="rounded-full border border-border px-2 py-1 type-caption text-fg-dim">
                        {preview.hasMore ? 'Additional request groups remain' : 'All matching request groups included'}
                      </span>
                    </div>

                    <div className="grid gap-2 sm:grid-cols-[1.15fr_2fr]">
                      <dl className="rounded-[var(--radius)] border border-bad/35 bg-bad/5 px-3 py-3">
                        <dt className="type-label text-fg-faint">Terminal request groups</dt>
                        <dd className="mt-1 font-mono text-[length:var(--density-type-display)] font-semibold leading-none tabular-nums text-foreground">
                          {preview.state === 'previewed' ? preview.planned.requests : preview.executed.requests}
                        </dd>
                      </dl>
                      <dl className="grid grid-cols-3 overflow-hidden rounded-[var(--radius)] border border-border-soft bg-panel-strong/55">
                        {[
                          [
                            'Lifecycle events',
                            preview.state === 'previewed' ? preview.planned.events : preview.executed.events
                          ],
                          [
                            'Artifact records',
                            preview.state === 'previewed' ? preview.planned.artifacts : preview.executed.artifacts
                          ],
                          [
                            'Proxy records',
                            preview.state === 'previewed' ? preview.planned.proxyRecords : preview.executed.proxyRecords
                          ]
                        ].map(([label, value], index) => (
                          <div
                            className={index > 0 ? 'border-l border-border-soft px-3 py-3' : 'px-3 py-3'}
                            key={String(label)}
                          >
                            <dt className="type-label text-fg-faint">{label}</dt>
                            <dd className="mt-1 font-mono text-[length:var(--density-type-title)] font-semibold tabular-nums text-foreground">
                              {value}
                            </dd>
                          </div>
                        ))}
                      </dl>
                    </div>

                    <p className="type-caption text-fg-dim">
                      {countLabel(
                        preview.state === 'previewed' ? preview.planned.databaseRows : preview.executed.databaseRows,
                        'total record'
                      )}{' '}
                      {preview.state === 'previewed' ? 'in this batch' : 'changed in this batch'}
                      {preview.hasMore
                        ? `; the server limits each cleanup to ${preview.scope.requestLimit} request groups`
                        : ''}
                      .
                    </p>
                  </>
                )}

                <div className="rounded-[var(--radius)] border border-good/30 bg-good/5 px-3 py-2.5">
                  <p className="type-caption font-medium text-foreground">Operational events stay retained.</p>
                  <p className="mt-0.5 type-caption text-fg-dim">
                    System, QUIC, Gossip, and Iroh events are not removed by this cleanup.
                  </p>
                </div>

                <div className="rounded-[var(--radius)] bg-panel-strong/55 px-3 py-2.5">
                  <div className="flex items-start justify-between gap-3">
                    <dl className="grid min-w-0 flex-1 gap-2 sm:grid-cols-2">
                      <div>
                        <dt className="type-label text-fg-faint">Selected window</dt>
                        <dd className="mt-1 type-caption text-foreground">
                          {formatWindow(preview.scope.from ?? preview.scope.cutoffBefore)} to{' '}
                          {formatWindow(preview.scope.to ?? preview.scope.cutoffBefore)}
                        </dd>
                      </div>
                      <div>
                        <dt className="type-label text-fg-faint">Audit reason</dt>
                        <dd className="mt-1 type-caption text-foreground">Recorded and hidden from this screen</dd>
                      </div>
                    </dl>
                    {preview.state === 'previewed' && !previewIsEmpty ? (
                      <Button
                        className="ui-control h-8 shrink-0 !text-xs"
                        disabled={pending}
                        onClick={editSelection}
                        size="sm"
                        type="button"
                        variant="outline"
                      >
                        Edit
                      </Button>
                    ) : null}
                  </div>
                </div>
              </section>

              <p className="type-caption text-fg-dim">
                Applied filters:{' '}
                {[
                  preview.scope.route && `route ${preview.scope.route}`,
                  preview.scope.model && `model ${preview.scope.model}`,
                  preview.scope.provider && `provider ${preview.scope.provider}`,
                  preview.scope.engine && `engine ${preview.scope.engine}`,
                  preview.scope.outcome && `outcome ${preview.scope.outcome}`
                ]
                  .filter(Boolean)
                  .join(' · ') || 'none'}
              </p>
              <details
                className="rounded-[var(--radius)] border border-border-soft px-3 py-2.5"
                open={preview.state === 'partial' || undefined}
              >
                <summary className="cursor-pointer type-caption font-medium text-fg-dim">Audit details</summary>
                <LogMaintenanceReceiptDiagnostics
                  className="mt-2 border-0 bg-transparent p-0"
                  receipt={preview}
                  showCounts={false}
                />
              </details>
            </>
          )}
          <span className="sr-only" aria-live="polite">
            {preview?.state === 'previewed'
              ? previewIsEmpty
                ? 'Server preview complete. No terminal request groups matched.'
                : `Server preview complete. ${countLabel(preview.planned.requests, 'terminal request group')} ready for review.`
              : ''}
          </span>
          {action ? (
            <p className={`type-caption ${actionToneClass(action.tone)}`} role="status">
              {action.message}
            </p>
          ) : null}
        </SharedModalBody>
        <SharedModalActionStrip>
          <DialogPrimitive.Close asChild>
            <Button
              className="ui-control min-h-[3.15rem] !text-xs sm:min-h-0"
              disabled={pending}
              size="sm"
              type="button"
              variant="outline"
            >
              {preview && preview.state !== 'previewed' ? 'Close' : 'Cancel'}
            </Button>
          </DialogPrimitive.Close>
          {previewIsEmpty ? (
            <Button
              className="ui-control-primary min-h-[3.15rem] !text-xs sm:min-h-0"
              disabled={pending}
              onClick={editSelection}
              size="sm"
              type="button"
            >
              Adjust window
            </Button>
          ) : null}
          {preview?.state === 'previewed' && !previewIsEmpty ? (
            <Button
              className="ui-control-destructive min-h-[3.15rem] !text-xs sm:min-h-0"
              disabled={!operation || pending}
              onClick={() => void runCleanup()}
              size="sm"
              type="button"
              variant="outline"
            >
              {pending ? 'Deleting…' : 'Delete this batch'}
            </Button>
          ) : preview && hasRetryableArtifactWork(preview) ? (
            <Button
              className="ui-control-destructive min-h-[3.15rem] !text-xs sm:min-h-0"
              disabled={!operation || pending}
              onClick={() => void runCleanup()}
              size="sm"
              type="button"
              variant="outline"
            >
              {pending ? 'Retrying…' : 'Retry file removal'}
            </Button>
          ) : preview && preview.state !== 'previewed' && preview.hasMore ? (
            <Button
              className="ui-control-primary min-h-[3.15rem] !text-xs sm:min-h-0"
              disabled={pending}
              onClick={() => void previewCleanup()}
              size="sm"
              type="button"
            >
              {pending ? 'Checking…' : 'Review next batch'}
            </Button>
          ) : !preview ? (
            <Button
              className="ui-control-primary min-h-[3.15rem] !text-xs sm:min-h-0"
              disabled={!validScope || !isReasonValid(reason) || pending}
              onClick={() => void previewCleanup()}
              size="sm"
              type="button"
            >
              {pending ? 'Checking…' : 'Review deletion'}
            </Button>
          ) : null}
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}
