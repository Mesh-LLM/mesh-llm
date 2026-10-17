// The Peers drill's "Referee verdicts about them": four counts, each opening
// the verdict records it is made of. Pure, so each line is unit-testable.
// Never one number standing for the peer: the four are shown side by side.
import type { PaneBRow, RefereeVerdictBucketKey } from '@/features/capsules/api/sidecarTypes'

export type RefereeVerdictCountItem = {
  key: RefereeVerdictBucketKey
  label: string
  count: number
  verdictCapsuleIds: string[]
}

const ORDER: ReadonlyArray<[RefereeVerdictBucketKey, string]> = [
  ['corroborated', 'corroborated'],
  ['contradicted', 'contradicted'],
  ['inconclusive', 'inconclusive'],
  ['not_comparable', 'not comparable']
]

/** `null` when the plugin sent no counts (an older plugin): the page says the
 *  counts aren't shown, never four zeros it didn't read. */
export function refereeVerdictCountItems(row: PaneBRow): RefereeVerdictCountItem[] | null {
  const counts = row.referee_verdicts
  if (!counts) return null
  return ORDER.map(([key, label]) => {
    const ids = counts[key]?.verdict_capsule_ids ?? []
    return { key, label, count: ids.length, verdictCapsuleIds: ids }
  })
}
