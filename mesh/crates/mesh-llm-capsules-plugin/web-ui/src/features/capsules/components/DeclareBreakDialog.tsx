// L3.5 action vocabulary — "Declare the break" shows both consequences
// before confirming; cause is a picker. Shared by every FAIL-state actions
// block in the Ledger (never "Repair" — evidence is added, not fixed).
import { useState } from 'react'

export type DeclareBreakDialogProps = {
  onConfirm: (cause: string) => void
  onCancel: () => void
}

export function DeclareBreakDialog({ onConfirm, onCancel }: DeclareBreakDialogProps) {
  const [cause, setCause] = useState('unknown')

  return (
    <div className="mt-2 rounded border border-border/60 bg-card p-3 text-xs text-fg-dim">
      <p className="mb-2 font-medium text-foreground">Declaring this break has two consequences:</p>
      <ol className="mb-3 ml-3 flex list-decimal flex-col gap-1">
        <li>This exchange will be marked as broken in your local record.</li>
        <li>Other nodes you exchange with will be able to see that you declared a break here.</li>
      </ol>
      <div className="mb-3 flex items-center gap-2">
        <label htmlFor="declare-cause" className="shrink-0">
          Cause:
        </label>
        <select
          id="declare-cause"
          value={cause}
          onChange={(e) => setCause(e.target.value)}
          className="rounded border border-border/70 bg-card px-1 py-0.5 text-xs text-foreground"
        >
          <option value="restored_from_backup">Restored from backup</option>
          <option value="reinstalled">Reinstalled</option>
          <option value="unknown">Unknown</option>
        </select>
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => onConfirm(cause)}
          className="rounded border border-border/60 px-2 py-0.5 text-xs text-foreground hover:bg-card"
        >
          Confirm
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded border border-border/60 px-2 py-0.5 text-xs text-fg-dim hover:bg-card"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
