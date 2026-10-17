// A peer's twin pairs, from the Exchanges rows (pane C) that name it: the same
// request sent to this peer and to another node, marked as one pair by the
// client that sent it. Stated as facts only -- how many pairs, in how many the
// two sealed answers differed byte for byte, and whether a referee ruled --
// never a judgement of the peer.
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'

export type PeerTwinTally = {
  /** Complete pairs (both rows present) one of whose rows this peer served. */
  pairs: number
  /** Pairs whose two answers differed (`twin.same_answer === false`). */
  differed: number
  /** Pairs a referee has signed a verdict for. */
  adjudicated: number
}

/** `null` when no complete twin pair names this peer. */
export function peerTwinTally(peerId: string, rows: readonly PaneCRow[]): PeerTwinTally | null {
  const brackets = new Map<string, { differed: boolean; adjudicated: boolean }>()
  for (const row of rows) {
    const twin = row.twin
    if (row.counterparty !== peerId || !twin?.bracket_id || twin.other_row == null) continue
    const pair = brackets.get(twin.bracket_id) ?? { differed: false, adjudicated: false }
    pair.differed ||= twin.same_answer === false
    pair.adjudicated ||= !!twin.verdict
    brackets.set(twin.bracket_id, pair)
  }
  if (brackets.size === 0) return null
  const pairs = [...brackets.values()]
  return {
    pairs: pairs.length,
    differed: pairs.filter((pair) => pair.differed).length,
    adjudicated: pairs.filter((pair) => pair.adjudicated).length
  }
}

/** `Twin pairs: 3 · answers differed: 3 · not adjudicated`. */
export function peerTwinText(tally: PeerTwinTally): string {
  const ruled =
    tally.adjudicated === 0
      ? 'not adjudicated'
      : tally.adjudicated === tally.pairs
        ? 'adjudicated'
        : `${tally.adjudicated} adjudicated, ${tally.pairs - tally.adjudicated} not`
  return `Twin pairs: ${tally.pairs} · answers differed: ${tally.differed} · ${ruled}`
}
