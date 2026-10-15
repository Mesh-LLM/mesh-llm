// A referee's signed verdict record, opened from the twin bracket, a judged
// node's row, or the referee's own row. Shows the record as sealed and the
// plugin's check of its signature; the page never re-judges it.
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { useQuery } from '@tanstack/react-query'
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
import { fetchVerdictRecord } from '@/features/capsules/api/verdictClient'
import {
  shortId,
  verdictRecordFacts,
  verdictSentence,
  verdictSignatureText
} from '@/features/capsules/lib/adjudication-view'

export type VerdictRecordDialogProps = {
  capsuleId: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function VerdictRecordDialog({ capsuleId, open, onOpenChange }: VerdictRecordDialogProps) {
  const record = useQuery({
    queryKey: ['capsules', 'verdict-record', capsuleId],
    queryFn: () => fetchVerdictRecord(capsuleId),
    enabled: open,
    retry: false,
    staleTime: 60_000
  })
  const facts = record.data ? verdictRecordFacts(record.data) : null

  return (
    <SharedModal onOpenChange={onOpenChange} open={open}>
      <SharedModalContent aria-describedby="verdict-record-description" className="w-[min(560px,calc(100vw-2rem))]">
        <SharedModalHeader>
          <SharedModalTitle>The referee’s verdict</SharedModalTitle>
          <SharedModalDescription id="verdict-record-description">
            A referee’s signed record of its judgment, as it was sealed. Record {shortId(capsuleId)}.
          </SharedModalDescription>
        </SharedModalHeader>
        <SharedModalBody className="flex flex-col gap-2 text-sm">
          {record.isPending ? <p className="text-fg-dim">Loading the record…</p> : null}
          {record.isError ? (
            <p className="text-fg-dim" role="status">
              This node couldn’t load the verdict record, so nothing about it is checked here.
            </p>
          ) : null}
          {record.data && facts ? (
            <>
              <p data-testid="verdict-record-verdict">
                {verdictSentence(facts.verdict)}
              </p>
              <p data-testid="verdict-record-signature">{verdictSignatureText(record.data)}</p>
              {record.data.signed_by_key_id ? (
                <p className="text-fg-dim">Signed by key {shortId(record.data.signed_by_key_id)}.</p>
              ) : null}
              {facts.halves.length > 0 ? (
                <p className="text-fg-dim">
                  The answers it judged: {facts.halves.map((id) => shortId(id)).join(' and ')}.
                </p>
              ) : null}
              {facts.refereeCapsuleId ? (
                <p className="text-fg-dim">The referee’s own answer: {shortId(facts.refereeCapsuleId)}.</p>
              ) : null}
            </>
          ) : null}
        </SharedModalBody>
        <SharedModalActionStrip>
          <DialogPrimitive.Close asChild>
            <Button className="ui-control" size="sm" type="button" variant="outline">
              Close
            </Button>
          </DialogPrimitive.Close>
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}
