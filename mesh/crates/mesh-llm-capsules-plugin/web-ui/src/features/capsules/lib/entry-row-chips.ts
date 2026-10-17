// Row-level chip strip (design §3A):
// `content · sig · inclusion · registered · theirs`, a condensed three-mark
// summary of five named properties -- distinct from the `▸ checks` panel's
// own five-state detail (`assurance-tone.ts`'s CHIP_GLYPH/CHIP_TONE), which
// this strip links into rather than duplicates.
import type { ChecksRow } from '@/features/capsules/lib/security-checks-view'

export type EntryRowChipKey = 'content' | 'sig' | 'inclusion' | 'registered' | 'theirs'
export type EntryRowChipMark = '✓' | '✗' | '–'

export const ENTRY_ROW_CHIP_ORDER: readonly EntryRowChipKey[] = ['content', 'sig', 'inclusion', 'registered', 'theirs']

/** The words each chip shows on the row face: plain, never the property's
 *  engineer name (UX §7.6-2). Console copy. */
export const ENTRY_ROW_CHIP_LABEL: Record<EntryRowChipKey, string> = {
  content: 'words match',
  sig: 'signed',
  inclusion: 'in a checkpoint',
  registered: 'witnessed',
  theirs: 'their record'
}

// The one property each chip names, verbatim design §3A -- never re-derive
// this mapping ad hoc at a call site.
const CHIP_PROPERTY_KEY: Record<EntryRowChipKey, string> = {
  content: 'content_binding',
  sig: 'producer_signature',
  inclusion: 'local_inclusion',
  registered: 'external_registration',
  theirs: 'outcome_corroboration'
}

/** The `security-checks-view.ts` `ChecksRow.key` (== the nine/ten-property
 *  map's key) this chip is a link into -- also the id fragment
 *  `checkRowDomId` (`exchange-pages.ts`) targets for the jump. */
export function entryRowChipPropertyKey(chip: EntryRowChipKey): string {
  return CHIP_PROPERTY_KEY[chip]
}

/** PASS -> green check, FAIL -> red cross, everything else (NOT_PRESENT /
 *  NOT_CHECKED / INCONCLUSIVE, or a property the checks don't carry) -> the
 *  neutral dash. Finding 1 (look 2026-09-26): the strip reads the SAME check
 *  results the `▸ checks` panel renders (`buildChecksRows`, this node's side)
 *  -- never the pane's `properties` map, which the native pane leaves null,
 *  so the strip read all dashes while the panel below it said established. */
export function entryRowChipMark(checks: readonly ChecksRow[], chip: EntryRowChipKey): EntryRowChipMark {
  const state = checks.find((row) => row.key === CHIP_PROPERTY_KEY[chip])?.yours?.state
  if (state === 'PASS') return '✓'
  if (state === 'FAIL') return '✗'
  return '–'
}
