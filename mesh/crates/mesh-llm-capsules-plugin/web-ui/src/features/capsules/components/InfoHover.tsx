// A small (i) info glyph that reveals an already-blessed honest sentence on
// hover AND keyboard focus, so the terse column headers / tiles / chips / cells
// keep the honest detail without carrying it on the face. NEVER a click target:
// the detail is supplementary, not an action.
//
// Reuses the shared Radix `Tooltip` primitive (`components/ui/tooltip.tsx`) --
// which opens on hover and on keyboard focus with no click, exactly the
// interaction this needs -- rather than inventing a popover. On top of it we
// wire a PERSISTENT `aria-describedby` from the glyph to a visually-hidden copy
// of the same sentence: Radix links the description only while the tooltip is
// open, but the honest sentence should be available to a screen reader whenever
// the glyph is reachable, not only mid-hover. The page already mounts a
// `TooltipProvider` (LedgerPage), so the shared `Tooltip` works here directly.
//
// The tooltip copy lives in `tooltip-copy.ts`, one sentence per chip type.
import { useId } from 'react'
import { Info } from 'lucide-react'
import { Tooltip } from '@/components/ui/tooltip'

export type InfoHoverProps = {
  /** The honest sentence, moved off the face. Rendered in the tooltip AND in a
   *  visually-hidden node the glyph's `aria-describedby` points at. */
  label: string
  /** Names what the (i) explains, for the glyph's own accessible name
   *  (e.g. "About Confirmed by the other side"). */
  describes: string
  /** Optional tooltip side; defaults to the shared primitive's "top". */
  side?: 'top' | 'right' | 'bottom' | 'left'
  className?: string
  /** The chip type this (i) explains, for the tooltip census
   *  (`tooltip-census.test.tsx`). */
  census?: string
}

export function InfoHover({ label, describes, side = 'top', className, census }: InfoHoverProps) {
  const descriptionId = useId()
  return (
    <span className={`inline-flex items-center ${className ?? ''}`} data-census-chip={census}>
      <Tooltip content={label} side={side}>
        <button
          aria-describedby={descriptionId}
          aria-label={`About ${describes}`}
          className="ui-control inline-flex cursor-help appearance-none items-center rounded-full border-0 bg-transparent p-0 align-middle text-fg-faint outline-none hover:text-fg-dim focus-visible:ring-2 focus-visible:ring-accent"
          // The (i) is never an action -- keep a stray click/keydown from
          // bubbling to a clickable ancestor (e.g. a peer row that opens the
          // inspector). Hover and focus already reveal the copy; no click.
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') event.stopPropagation()
          }}
          type="button"
        >
          <Info aria-hidden="true" className="size-3.5" />
        </button>
      </Tooltip>
      {/* The same honest sentence, always available to assistive tech via the
         glyph's `aria-describedby` -- not gated on the tooltip being open. */}
      <span className="sr-only" id={descriptionId}>
        {label}
      </span>
    </span>
  )
}
