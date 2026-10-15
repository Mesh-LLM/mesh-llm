// (Ported unchanged from the mesh-llm console fork.) The `Your records` panel
// (replaces the dead `This node's copy` pill): where the records live, how
// many, the last checkpoint, the four sharing switches with one "what leaves"
// sentence each, and the evidence file. The switches are changed in
// Configuration (UX §7.3); here they are read.
import * as DialogPrimitive from '@radix-ui/react-dialog'
import type { ReactNode } from 'react'
import { Download } from 'lucide-react'
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
import { StatusBadge } from '@/components/ui/StatusBadge'
import type { RecordsStatus } from '@/features/capsules/api/recordsClient'
import { InfoHover } from '@/features/capsules/components/InfoHover'
import {
  NEW_LOG_PENDING,
  SHARING_GOVERNS,
  lastCheckpointFact,
  recordsLocationFact,
  sharingRows
} from '@/features/capsules/lib/your-records'

export type YourRecordsDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  status: RecordsStatus | null
  sample: boolean
  recordCount: number
  coveredRecords: number | null
  checkpointNoLaterThan: string | null
  onExport: () => void
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[8.5rem_1fr] gap-3">
      <dt className="type-label text-fg-faint">{term}</dt>
      <dd className="min-w-0 break-words text-foreground">{children}</dd>
    </div>
  )
}

export function YourRecordsDialog({
  open,
  onOpenChange,
  status,
  sample,
  recordCount,
  coveredRecords,
  checkpointNoLaterThan,
  onExport
}: YourRecordsDialogProps) {
  return (
    <SharedModal open={open} onOpenChange={onOpenChange}>
      <SharedModalContent
        aria-describedby="your-records-description"
        className="w-[min(560px,calc(100vw-2rem))]"
        // Focus the panel, not its first (i): a focused (i) opens its tooltip
        // over the facts the person opened the panel to read.
        onOpenAutoFocus={(event) => {
          event.preventDefault()
          ;(event.currentTarget as HTMLElement | null)?.focus()
        }}
      >
        <SharedModalHeader>
          <SharedModalTitle>Your records</SharedModalTitle>
          <SharedModalDescription id="your-records-description">
            The records this node keeps, sealed and checkpointed.
          </SharedModalDescription>
        </SharedModalHeader>
        <SharedModalBody className="flex max-h-[70vh] flex-col gap-5 overflow-y-auto text-sm">
          <dl className="flex flex-col gap-2" data-testid="your-records-facts">
            <Fact term="Where">
              <span className={status ? 'font-mono text-xs' : 'text-fg-dim'}>
                {recordsLocationFact(status, sample)}
              </span>
            </Fact>
            <Fact term="How many">
              {recordCount} record{recordCount === 1 ? '' : 's'}
            </Fact>
            <Fact term="Last checkpoint">{lastCheckpointFact(coveredRecords, checkpointNoLaterThan)}</Fact>
          </dl>
          {status?.new_history_pending ? (
            <p className="rounded border border-border-soft bg-panel-strong px-3 py-2 text-fg-dim" role="status">
              {NEW_LOG_PENDING}
            </p>
          ) : null}

          {(status?.ledger_repairs ?? []).length > 0 ? (
            <p className="text-[var(--color-bad-text)]" data-testid="torn-write" role="status">
              {`This log's last line was cut short (a write that did not finish) ${status?.ledger_repairs?.length === 1 ? 'once' : `${status?.ledger_repairs?.length} times`}, most recently at ${status?.ledger_repairs?.[status.ledger_repairs.length - 1]?.at}; each time it was removed back to the last whole record, and every record before it still checks.`}
            </p>
          ) : null}

          <section aria-labelledby="what-you-share-title" className="flex flex-col gap-2">
            <h3 className="type-label text-fg-faint" id="what-you-share-title">
              What you share
            </h3>
            {(status?.setting_problems ?? []).map((problem) => (
              <p className="text-[var(--color-bad-text)]" data-testid="setting-problem" key={problem} role="status">
                {problem}
              </p>
            ))}
            {status?.config_problem ? (
              <p className="text-[var(--color-bad-text)]" data-testid="config-problem" role="status">
                {status.config_problem}
              </p>
            ) : null}
            <ul className="flex flex-col divide-y divide-border-soft" data-testid="what-you-share">
              {sharingRows(status).map((row) => (
                <li className="flex flex-col gap-1 py-2" data-sharing-switch={row.key} key={row.key}>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-foreground">{row.label}</span>
                    <span className="inline-flex items-center gap-1">
                      <StatusBadge size="caption" tone="muted">
                        {row.state ?? 'not shown'}
                        {row.source === 'default' ? ' · default' : ''}
                      </StatusBadge>
                      <InfoHover census={`share:${row.key}`} describes={row.label} label={SHARING_GOVERNS[row.key]} />
                    </span>
                  </div>
                  <p className="text-fg-dim">{row.whatLeaves}</p>
                </li>
              ))}
            </ul>
            <p className="type-caption text-fg-dim">
              Change these in{' '}
              {/* A plain link: this component stays router-free, like the
                 Peers tab's link to Network. */}
              <a className="underline underline-offset-2 hover:text-foreground" href="/configuration/plugins">
                Configuration › Plugins
              </a>
              .
            </p>
          </section>
        </SharedModalBody>
        <SharedModalActionStrip>
          <DialogPrimitive.Close asChild>
            <Button className="ui-control" size="sm" type="button" variant="outline">
              Close
            </Button>
          </DialogPrimitive.Close>
          <Button className="ui-control-primary gap-1.5" onClick={onExport} size="sm" type="button">
            <Download aria-hidden="true" className="size-3.5" />
            Export evidence file
          </Button>
        </SharedModalActionStrip>
      </SharedModalContent>
    </SharedModal>
  )
}
