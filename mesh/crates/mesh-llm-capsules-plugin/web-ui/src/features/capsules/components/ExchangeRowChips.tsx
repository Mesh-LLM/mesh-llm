// The entry row's chip strip (design
// §3A) -- `content · sig · inclusion · registered · theirs`, each a link
// into the `▸ checks` panel's matching property row. `role="link"` (not
// `button`) deliberately: every existing row test counts the row's ACTION
// buttons (`getAllByRole('button')`) as a load-bearing invariant
// ("no whole-row click target... the two
// toggles are the entire detail surface"); a chip jumps to detail already on
// the page rather than performing an action, which is the same "link vs.
// button" distinction the design doc's own words ("a chip is a link into the
// expansion") draw.
import {
  entryRowChipMark,
  entryRowChipPropertyKey,
  ENTRY_ROW_CHIP_LABEL,
  ENTRY_ROW_CHIP_ORDER
} from '@/features/capsules/lib/entry-row-chips'
import type { EntryRowChipMark } from '@/features/capsules/lib/entry-row-chips'
import type { ChecksRow } from '@/features/capsules/lib/security-checks-view'
import { HoverChip } from '@/features/capsules/components/HoverChip'
import { ENTRY_CHIP_RESULT_TOOLTIPS } from '@/features/capsules/lib/tooltip-copy'

const MARK_COLOR: Record<EntryRowChipMark, string> = {
  '✓': 'var(--color-good-text)',
  '✗': 'var(--color-bad-text)',
  '–': 'var(--color-fg-faint)'
}

export function ExchangeRowChips({
  checks,
  onChipActivate
}: {
  /** The row's check results -- the same `buildChecksRows` output the
   *  `▸ checks` panel renders, so the two can never disagree. */
  checks: readonly ChecksRow[]
  /** The property key (`security-checks-view.ts`'s `ChecksRow.key`) this
   *  chip links to, not the chip's own short label. */
  onChipActivate: (propertyKey: string) => void
}) {
  return (
    <div aria-label="checks summary" className="flex flex-wrap items-center gap-2.5" role="group">
      {ENTRY_ROW_CHIP_ORDER.map((chip) => {
        const mark = entryRowChipMark(checks, chip)
        const label = ENTRY_ROW_CHIP_LABEL[chip]
        const propertyKey = entryRowChipPropertyKey(chip)
        // Hover for the meaning, click for the full check (UX §8 rule 2).
        return (
          <HoverChip census={`entry_chip:${chip}`} key={chip} label={ENTRY_CHIP_RESULT_TOOLTIPS[chip][mark]}>
            <span
              aria-label={`${label}: jump to that check`}
              className="ui-control-ghost inline-flex cursor-pointer items-center gap-1 font-mono text-xs"
              onClick={() => onChipActivate(propertyKey)}
              onKeyDown={(event) => {
                if (event.key !== 'Enter' && event.key !== ' ') return
                event.preventDefault()
                onChipActivate(propertyKey)
              }}
              role="link"
              tabIndex={0}
            >
              <span className="text-fg-dim">{label}</span>
              <span style={{ color: MARK_COLOR[mark] }}>{mark}</span>
            </span>
          </HoverChip>
        )
      })}
    </div>
  )
}
