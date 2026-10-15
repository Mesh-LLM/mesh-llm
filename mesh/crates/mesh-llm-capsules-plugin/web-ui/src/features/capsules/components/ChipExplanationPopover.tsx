// v3 §4: every chip opens the four-part
// explanation on click -- a popover anchored to the chip, never a modal
// (same "inline, never a dialog" rule the rest of this view lives by).
import { useId, type ReactNode } from 'react'
import { Tooltip } from '@/components/ui/tooltip'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import type { ChecksSideCell } from '@/features/capsules/lib/security-checks-view'
import { explanationFor } from '@/features/capsules/lib/chip-explanation'
import { CHECK_CHIP_TOOLTIPS, checkChipTooltipKey } from '@/features/capsules/lib/tooltip-copy'

export function ChipExplanationPopover({
  propertyKey,
  cell,
  factKey,
  detail,
  children
}: {
  propertyKey: string
  cell: ChecksSideCell
  factKey?: 'binding' | 'authority'
  /** p2 item 2: the sentence that used to sit beside the chip, word for
   *  word, now the tooltip's second line. */
  detail?: string
  children: ReactNode
}) {
  const explanation = explanationFor(propertyKey, cell, factKey)
  // UX §8 rule 2: hover (and focus) for the one-line meaning, click for the
  // four-part explanation -- both, never one or the other.
  const hoverKey = checkChipTooltipKey(propertyKey, factKey)
  const meaning = CHECK_CHIP_TOOLTIPS[hoverKey]
  const hover = [meaning, detail].filter(Boolean).join(' ')
  const hoverId = useId()
  return (
    <Popover>
      <span className="inline-flex" data-census-chip={meaning ? `check_chip:${hoverKey}` : undefined}>
        <Tooltip
          content={
            detail ? (
              <>
                {meaning}
                <span className="mt-1 block text-fg-dim">{detail}</span>
              </>
            ) : (
              meaning
            )
          }
        >
          <PopoverTrigger asChild>
            <button
              aria-describedby={hover ? hoverId : undefined}
              className="ui-control cursor-pointer appearance-none border-0 bg-transparent p-0 text-left"
              type="button"
            >
              {children}
            </button>
          </PopoverTrigger>
        </Tooltip>
        {hover ? (
          <span className="sr-only" id={hoverId}>
            {hover}
          </span>
        ) : null}
      </span>
      <PopoverContent className="flex flex-col gap-2 text-xs" data-chip-explanation={propertyKey}>
        <p>
          <span className="font-medium text-fg-faint">What this means: </span>
          {explanation.whatItMeans}
        </p>
        <p>
          <span className="font-medium text-fg-faint">What this view found: </span>
          {explanation.whatThisFound}
        </p>
        <p>
          <span className="font-medium text-fg-faint">What it does not establish: </span>
          {explanation.whatItDoesNotEstablish}
        </p>
        <p>
          <span className="font-medium text-fg-faint">How to change it: </span>
          {explanation.howToChange}
        </p>
      </PopoverContent>
    </Popover>
  )
}
