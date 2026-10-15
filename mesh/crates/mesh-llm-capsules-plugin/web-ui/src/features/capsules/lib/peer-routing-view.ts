// §7.5 "should I stop dealing with anyone?": the drill's "Your dealings with
// them" lines. Ported from the mesh-llm console fork's
// `lib/peer-routing-view.ts` (`dealingsLines` + `disputeParts`, unchanged).
// The console file's block-store half is not here: blocking a peer needs the
// host's local block list, which this page does not reach (see
// `ROUTING_NOT_ON_THIS_PAGE`).
import type { PaneBRow } from '@/features/capsules/api/sidecarTypes'
import { adjudicationSummary, confirmedByOtherSide, matchTally } from '@/features/capsules/lib/peer-row-view'

/** §7.5 "your dealings with them": exchanges, how many they confirmed with
 *  their own record, how many records differ, and disputes judged. Counts
 *  with their denominators; never a single number standing for the peer. */
export function dealingsLines(row: PaneBRow): string[] {
  const confirmed = confirmedByOtherSide(row)
  const tally = matchTally(row)
  const disputes = adjudicationSummary(row)
  const exchanges = row.exchange_count ?? 0
  return [
    `${exchanges} ${exchanges === 1 ? 'exchange' : 'exchanges'} with you · they confirmed ${confirmed.confirmed} of ${confirmed.total}`,
    `Same request & answer: ${tally.clean} · ${tally.mismatch} differ`,
    disputes.notChecked
      ? 'Disputes judged: none'
      : `Disputes judged: ${disputes.checked} of ${disputes.denominator}${disputeParts(disputes)}`
  ]
}

function disputeParts(summary: ReturnType<typeof adjudicationSummary>): string {
  const parts: string[] = []
  if (summary.corroborated) parts.push(`${summary.corroborated} corroborated`)
  if (summary.contradicted) parts.push(`${summary.contradicted} contradicted`)
  if (summary.inconclusive) parts.push(`${summary.inconclusive} inconclusive`)
  return parts.length ? ` · ${parts.join(' · ')}` : ''
}
