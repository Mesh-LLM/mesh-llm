// One line on an exchange row about a referee's verdict: on a judged node,
// the verdict delivered about this exchange; on the referee node, the
// verdict it issued for the call it answered. Nothing renders when the row
// carries neither.
import { useState } from 'react'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'
import { VerdictRecordDialog } from '@/features/capsules/components/VerdictRecordDialog'
import { deliveredVerdictLine, issuedVerdictLine, parseVerdict } from '@/features/capsules/lib/adjudication-view'

export function RowVerdictLine({ row }: { row: PaneCRow }) {
  const [open, setOpen] = useState(false)
  const delivered = row.adjudication
  const issued = row.adjudication_issued
  const text = delivered ? deliveredVerdictLine(delivered) : issued ? issuedVerdictLine(issued) : null
  const capsuleId = delivered?.verdict_capsule_id ?? issued?.verdict_capsule_id ?? null
  if (text === null || capsuleId === null) return null
  const verdict = parseVerdict((delivered ?? issued)?.verdict)
  const againstYou = delivered?.about_this_node === true && verdict?.kind === 'contradicted'
  return (
    <p
      className={`flex flex-wrap items-center gap-x-2 text-xs ${againstYou ? 'text-bad' : 'text-fg-dim'}`}
      data-row-verdict={delivered ? 'delivered' : 'issued'}
    >
      <span>{text}</span>
      <button className="underline underline-offset-2 hover:text-foreground" onClick={() => setOpen(true)} type="button">
        View the verdict
      </button>
      {open ? <VerdictRecordDialog capsuleId={capsuleId} onOpenChange={setOpen} open={open} /> : null}
    </p>
  )
}
