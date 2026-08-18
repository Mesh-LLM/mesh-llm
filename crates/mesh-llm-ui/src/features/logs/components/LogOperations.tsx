import * as DialogPrimitive from '@radix-ui/react-dialog'
import { Download, Trash2 } from 'lucide-react'
import { useMemo, useRef, useState, type RefObject } from 'react'
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
import type { LogExport } from '@/features/logs/api/schemas'
import { LogCleanupDialog } from '@/features/logs/components/LogCleanupDialog'
import { supportsCleanup } from '@/features/logs/components/LogCleanupScope'
import type { LogEventCategory, LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'

type ActionState = { readonly message: string; readonly tone: 'success' | 'error' } | undefined

type LogOperationsProps = {
  readonly operation: 'cleanup' | 'export'
  readonly query: LogsRequestQuery
  readonly onMaintenanceMutationSucceeded?: () => void
  readonly rows?: readonly LogEventLedgerRow[]
  readonly selectedCategories?: ReadonlySet<LogEventCategory>
}

type CleanupSnapshot = {
  readonly generation: number
  readonly rows: readonly LogEventLedgerRow[]
  readonly selectedCategories: ReadonlySet<LogEventCategory>
  readonly from: string | undefined
  readonly to: string | undefined
}

const DEFAULT_CLEANUP_CATEGORIES = new Set<LogEventCategory>(['requests', 'system', 'quic', 'gossip'])

function isReasonValid(reason: string) {
  return reason.trim().length > 0
}

function actionError(error: unknown) {
  return error instanceof Error ? error.message : 'The local log service did not complete the operation.'
}

function downloadExport(exportResult: LogExport) {
  const blob = new Blob([JSON.stringify(exportResult, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = 'mesh-llm-log-export.json'
  anchor.click()
  URL.revokeObjectURL(url)
}

function ExportDialog({
  open,
  onOpenChange,
  query,
  returnFocusRef
}: {
  readonly open: boolean
  readonly onOpenChange: (open: boolean) => void
  readonly query: LogsRequestQuery
  readonly returnFocusRef: RefObject<HTMLButtonElement | null>
}) {
  const [reason, setReason] = useState('')
  const [action, setAction] = useState<ActionState>()
  const [pending, setPending] = useState(false)

  async function exportLogs() {
    if (!isReasonValid(reason)) return
    setPending(true)
    setAction(undefined)
    try {
      // The UI deliberately exports metadata only. Artifact body inclusion is
      // server-capture gated and must never be inferred from client state.
      const exportResult = await new LogsApiClient().exportRequests(query, {
        reason: reason.trim(),
        includeArtifacts: false
      })
      downloadExport(exportResult)
      setAction({
        tone: 'success',
        message: exportResult.truncated
          ? 'A bounded partial export was downloaded. Narrow the retained filter context before retrying.'
          : 'Bounded log export downloaded.'
      })
    } catch (error) {
      setAction({ tone: 'error', message: actionError(error) })
    } finally {
      setPending(false)
    }
  }

  return (
    <SharedModal open={open} onOpenChange={onOpenChange}>
      <SharedModalContent
        onCloseAutoFocus={(event) => {
          if (!returnFocusRef.current) return
          event.preventDefault()
          returnFocusRef.current.focus()
        }}
      >
        <SharedModalHeader>
          <SharedModalTitle>Export current log view</SharedModalTitle>
          <SharedModalDescription>
            The server applies its bounded export limit to the current durable filters and cursor. Artifact bodies stay
            excluded.
          </SharedModalDescription>
        </SharedModalHeader>
        <SharedModalBody>
          <label
            className="grid gap-1.5 text-[length:var(--density-type-caption)] text-fg-dim"
            htmlFor="log-export-reason"
          >
            <span className="type-label text-fg-faint">Required audit reason</span>
            <Input
              id="log-export-reason"
              aria-describedby="log-export-metadata-note"
              className="border-border bg-panel-strong"
              onChange={(event) => setReason(event.currentTarget.value)}
              placeholder="Why is this export needed?"
              value={reason}
            />
          </label>
          <p className="mt-2 type-caption text-fg-dim" id="log-export-metadata-note">
            Metadata-only export. Retained artifact payloads are never loaded or included by this control.
          </p>
          {action ? (
            <p className={`mt-3 type-caption ${action.tone === 'error' ? 'text-bad' : 'text-good'}`} role="status">
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
          <Button
            className="ui-control-primary"
            disabled={!isReasonValid(reason) || pending}
            onClick={() => void exportLogs()}
            size="sm"
            type="button"
          >
            {pending ? 'Exporting…' : 'Download export'}
          </Button>
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}

export function LogOperations({
  operation,
  query,
  onMaintenanceMutationSucceeded,
  rows = [],
  selectedCategories = DEFAULT_CLEANUP_CATEGORIES
}: LogOperationsProps) {
  const [open, setOpen] = useState(false)
  const [cleanupSnapshot, setCleanupSnapshot] = useState<CleanupSnapshot>()
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const fallbackSnapshot = useMemo<CleanupSnapshot>(
    () => ({ generation: 0, rows, selectedCategories, from: query.from, to: query.to }),
    [rows, selectedCategories, query.from, query.to]
  )

  switch (operation) {
    case 'export':
      return (
        <div className="flex flex-wrap items-center gap-2">
          <Button
            ref={triggerRef}
            className="ui-control h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
            disabled={query.source !== undefined}
            onClick={() => setOpen(true)}
            size="sm"
            type="button"
            variant="outline"
          >
            <Download className="size-3.5" aria-hidden="true" />
            Export view
          </Button>
          {query.source !== undefined ? (
            <span className="type-caption text-fg-dim">Clear source selection to export durable records.</span>
          ) : null}
          <ExportDialog open={open} onOpenChange={setOpen} query={query} returnFocusRef={triggerRef} />
        </div>
      )
    case 'cleanup': {
      const snapshot = cleanupSnapshot ?? fallbackSnapshot
      const cleanupQuery = { ...query, from: snapshot.from, to: snapshot.to }
      return (
        <div className="flex flex-wrap items-center gap-2">
          <Button
            ref={triggerRef}
            className="ui-control-destructive h-8 gap-1.5 rounded-[var(--radius)] px-2.5 text-[length:var(--density-type-caption)]"
            disabled={!supportsCleanup(query)}
            onClick={() => {
              setCleanupSnapshot({
                generation: Date.now(),
                rows: [...rows],
                selectedCategories: new Set(selectedCategories),
                from: query.from,
                to: query.to
              })
              setOpen(true)
            }}
            size="sm"
            type="button"
            variant="outline"
          >
            <Trash2 className="size-3.5" aria-hidden="true" />
            Clean up logs
          </Button>
          {!supportsCleanup(query) ? (
            <span className="type-caption text-fg-dim">
              Clear active-source or non-terminal outcome filters before removing durable logs.
            </span>
          ) : null}
          <LogCleanupDialog
            key={`${snapshot.from ?? ''}:${snapshot.to ?? ''}:${[...snapshot.selectedCategories].join(',')}:${snapshot.generation}`}
            open={open}
            onMaintenanceMutationSucceeded={onMaintenanceMutationSucceeded}
            onOpenChange={(nextOpen) => {
              setOpen(nextOpen)
              if (!nextOpen) setCleanupSnapshot(undefined)
            }}
            query={cleanupQuery}
            returnFocusRef={triggerRef}
            rows={snapshot.rows}
            initialCategories={snapshot.selectedCategories}
          />
        </div>
      )
    }
  }
}
