// `Clean up records` (ported unchanged from the mesh-llm console fork): exactly
// three choices, each with its consequence, and NO single-record delete --
// removing one sealed record is a rewrite, not a cleanup, and the dialog says
// so. Every choice seals a record of itself on the plugin side
// (`owner_maintenance.rs`); the result line names the record it sealed.
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useQueryClient } from '@tanstack/react-query'
import { useId, useState } from 'react'
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
import { runCleanup, type CleanupAction, type RecordsStatus } from '@/features/capsules/api/recordsClient'
import { HoverChip } from '@/features/capsules/components/HoverChip'
import { RECORDS_STATUS_QUERY_KEY } from '@/features/capsules/lib/use-your-records'
import {
  CLEANUP_IS_ON_THE_RECORD,
  CLEANUP_OPTIONS,
  NO_SINGLE_RECORD_DELETE,
  NO_STORED_TEXT,
  START_NEW_LOG_CONFIRM,
  cleanupBlockedReason,
  cleanupResultMessage
} from '@/features/capsules/lib/your-records'

type Outcome = { tone: 'success' | 'error'; message: string } | undefined

export type CleanUpRecordsDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  status: RecordsStatus | null
  sample: boolean
}

export function CleanUpRecordsDialog({ open, onOpenChange, status, sample }: CleanUpRecordsDialogProps) {
  const queryClient = useQueryClient()
  const groupName = useId()
  // Nothing to delete when no prompt or answer text is kept: that option is
  // off, and the dialog opens on one that can run.
  const noStoredText = status?.stored_text_count === 0
  const [picked, setChoice] = useState<CleanupAction>('delete_stored_text')
  const choice: CleanupAction = noStoredText && picked === 'delete_stored_text' ? 'rebuild_index' : picked
  const [confirmed, setConfirmed] = useState(false)
  const [pending, setPending] = useState(false)
  const [outcome, setOutcome] = useState<Outcome>()
  const option = CLEANUP_OPTIONS.find((o) => o.action === choice) ?? CLEANUP_OPTIONS[0]
  const blocked = cleanupBlockedReason(choice, { sample, status, confirmed })

  async function run() {
    if (blocked) return
    setPending(true)
    setOutcome(undefined)
    try {
      const result = await runCleanup(choice)
      setOutcome({ tone: 'success', message: cleanupResultMessage(result) })
      // The cleanup sealed a record: every count on the tab moves by one.
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: RECORDS_STATUS_QUERY_KEY }),
        queryClient.invalidateQueries({ queryKey: ['capsules', 'ledger'] }),
        queryClient.invalidateQueries({ queryKey: ['ledger'] })
      ])
    } catch (error) {
      setOutcome({ tone: 'error', message: error instanceof Error ? error.message : 'The node did not complete it.' })
    } finally {
      setPending(false)
    }
  }

  const button = (
    <Button
      className={choice === 'rebuild_index' ? 'ui-control-primary' : 'ui-control-destructive'}
      disabled={blocked !== null || pending}
      onClick={() => void run()}
      size="sm"
      type="button"
      variant={choice === 'rebuild_index' ? 'default' : 'outline'}
    >
      {pending ? 'Working…' : option.button}
    </Button>
  )

  return (
    <SharedModal
      open={open}
      onOpenChange={(next) => {
        onOpenChange(next)
        if (!next) {
          setOutcome(undefined)
          setConfirmed(false)
        }
      }}
    >
      <SharedModalContent className="w-[min(560px,calc(100vw-2rem))]" aria-describedby="clean-up-records-description">
        <SharedModalHeader>
          <SharedModalTitle>Clean up records</SharedModalTitle>
          <SharedModalDescription id="clean-up-records-description">{CLEANUP_IS_ON_THE_RECORD}</SharedModalDescription>
        </SharedModalHeader>
        <SharedModalBody className="flex flex-col gap-3 text-sm">
          <fieldset className="flex flex-col gap-2">
            <legend className="sr-only">Choose a cleanup</legend>
            {CLEANUP_OPTIONS.map((o) => (
              <label
                className={`flex cursor-pointer gap-3 rounded-[var(--radius)] border px-3 py-2.5 ${
                  choice === o.action ? 'border-accent bg-panel-strong' : 'border-border-soft'
                }`}
                data-cleanup-option={o.action}
                key={o.action}
              >
                <input
                  checked={choice === o.action}
                  className="mt-1"
                  disabled={noStoredText && o.action === 'delete_stored_text'}
                  name={groupName}
                  onChange={() => {
                    setChoice(o.action)
                    setOutcome(undefined)
                  }}
                  type="radio"
                  value={o.action}
                />
                <span className="flex flex-col gap-0.5">
                  <span className="font-medium text-foreground">{o.title}</span>
                  <span className="text-fg-dim">
                    {noStoredText && o.action === 'delete_stored_text' ? NO_STORED_TEXT : o.consequence}
                  </span>
                </span>
              </label>
            ))}
          </fieldset>
          {choice === 'start_new_log' ? (
            <label className="flex items-start gap-2 text-fg-dim">
              <input
                checked={confirmed}
                className="mt-1"
                onChange={(event) => setConfirmed(event.currentTarget.checked)}
                type="checkbox"
              />
              {START_NEW_LOG_CONFIRM}
            </label>
          ) : null}
          <p className="type-caption text-fg-dim" data-testid="no-single-record-delete">
            {NO_SINGLE_RECORD_DELETE}
          </p>
          {outcome ? (
            <p className={`type-caption ${outcome.tone === 'error' ? 'text-bad' : 'text-good'}`} role="status">
              {outcome.message}
            </p>
          ) : null}
        </SharedModalBody>
        <SharedModalActionStrip>
          <DialogPrimitive.Close asChild>
            <Button className="ui-control" size="sm" type="button" variant="outline">
              Close
            </Button>
          </DialogPrimitive.Close>
          {blocked ? (
            <HoverChip census="action:disabled_reason" label={blocked}>
              <span className="inline-flex">{button}</span>
            </HoverChip>
          ) : (
            button
          )}
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}
