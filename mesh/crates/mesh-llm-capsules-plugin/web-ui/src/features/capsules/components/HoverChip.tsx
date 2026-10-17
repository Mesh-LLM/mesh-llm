// A chip that explains itself on hover and keyboard focus, with no extra (i)
// button: the chip IS the trigger. For chips that already do something on
// click (the row's check chips jump into the checks panel) or that sit where
// row button counts are pinned. Same accessibility promise as `InfoHover`: the
// sentence is always reachable through `aria-describedby`, not only while the
// tooltip is open.
import { cloneElement, useId, type ReactElement } from 'react'
import { Tooltip } from '@/components/ui/tooltip'

export type HoverChipProps = {
  /** One plain sentence (`tooltip-copy.ts`). */
  label: string
  /** The chip type this is, for the tooltip census (`tooltip-census.test.tsx`). */
  census: string
  /** The chip element itself; it receives the describedby link and, if it
   *  isn't focusable already, `tabIndex={0}` so keyboard users get the hover. */
  children: ReactElement<{ 'aria-describedby'?: string; tabIndex?: number }>
  side?: 'top' | 'right' | 'bottom' | 'left'
}

export function HoverChip({ label, census, children, side = 'top' }: HoverChipProps) {
  const descriptionId = useId()
  const chip = cloneElement(children, {
    'aria-describedby': descriptionId,
    tabIndex: children.props.tabIndex ?? 0
  })
  return (
    <span className="inline-flex items-center" data-census-chip={census}>
      <Tooltip content={label} side={side}>
        {chip}
      </Tooltip>
      <span className="sr-only" id={descriptionId}>
        {label}
      </span>
    </span>
  )
}
