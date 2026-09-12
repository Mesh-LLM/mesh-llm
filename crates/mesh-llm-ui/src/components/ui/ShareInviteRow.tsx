import { Share2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { shareButtonLabel, useShareInvite } from '@/lib/useShareInvite'

type ShareInviteRowProps = {
  label: string
  hint: string
  shareText: string
  shareTitle?: string
  disabled?: boolean
}

/**
 * One-tap handoff of a join invite to the platform share sheet, falling back to
 * the clipboard when no share sheet exists (most desktop browsers).
 */
export function ShareInviteRow({ label, hint, shareText, shareTitle, disabled = false }: ShareInviteRowProps) {
  const { share, outcome, canUseShareSheet } = useShareInvite()

  return (
    <div className="rounded-[var(--radius)] border border-accent/40 bg-panel-strong px-3 py-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="type-label text-fg-faint">{label}</div>
          <div className="mt-1 text-[length:var(--density-type-caption)] text-fg-faint">{hint}</div>
        </div>
        <Button
          aria-label={`Share ${label}`}
          className="ui-control inline-flex h-13 min-h-11 shrink-0 items-center gap-1.5 rounded-[var(--radius)] border px-2.5 py-1 text-[length:var(--density-type-caption)] font-medium lg:h-8 lg:min-h-8"
          disabled={disabled}
          onClick={() => {
            if (disabled) return
            void share(shareText, shareTitle)
          }}
          size="sm"
          type="button"
        >
          <Share2 className="size-[11px]" aria-hidden="true" />
          {disabled ? 'Unavailable' : shareButtonLabel(outcome, canUseShareSheet)}
        </Button>
      </div>
    </div>
  )
}
