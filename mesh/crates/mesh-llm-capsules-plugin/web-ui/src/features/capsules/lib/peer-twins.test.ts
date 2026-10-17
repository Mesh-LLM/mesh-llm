import { describe, expect, it } from 'vitest'
import type { PaneCRow, TwinRowFacts } from '@/features/capsules/api/sidecarTypes'
import { dealtWithRowView } from '@/features/capsules/lib/peer-row-view'
import { peerTwinTally, peerTwinText } from '@/features/capsules/lib/peer-twins'

/** One row of a twin pair: this node asked `counterparty`, as one half of
 *  bracket `bracket`. */
function twinRow(key: string, counterparty: string, twin: Partial<TwinRowFacts> & { bracket_id: string }): PaneCRow {
  return {
    exchange_key: key,
    role_tag: 'ASKED',
    counterparty,
    header_state: 'ok',
    properties: null,
    has_issue: false,
    mine: { state: 'present', capsule_id: `${key}-mine` },
    theirs: { state: 'present', capsule_id: `${key}-theirs` },
    unilateral: false,
    timestamp: '2026-10-05T21:00:00Z',
    twin_bracket_id: twin.bracket_id,
    twin: { same_answer: null, other_row: null, ...twin }
  }
}

/** Three brackets, each sent to a second node and to the adversary; the
 *  answers differed in all three. */
const DEMO_ROWS: PaneCRow[] = [1, 2, 3].flatMap((i) => [
  twinRow(`second-${i}`, 'key:second', { bracket_id: `b${i}`, same_answer: false, other_row: `adversary-${i}` }),
  twinRow(`adversary-${i}`, 'node:adversary', { bracket_id: `b${i}`, same_answer: false, other_row: `second-${i}` })
])

const JUDGEMENT_WORDS = /dishonest|honest|bad|flag|lie|lied|cheat|wrong|suspicious|fail|alarm|trust/i

describe('peerTwinTally', () => {
  it('a peer in three twin pairs whose answers all differed: the facts, and no verdict', () => {
    const tally = peerTwinTally('node:adversary', DEMO_ROWS)
    expect(tally).toEqual({ pairs: 3, differed: 3, adjudicated: 0 })
    const text = peerTwinText(tally!)
    expect(text).toBe('Twin pairs: 3 · answers differed: 3 · not adjudicated')
    expect(text).not.toMatch(JUDGEMENT_WORDS)
  })

  it('the other node in the same pairs reads the same: a difference belongs to the pair, not to one side', () => {
    expect(peerTwinTally('key:second', DEMO_ROWS)).toEqual(peerTwinTally('node:adversary', DEMO_ROWS))
  })

  it('a peer in no twin pair has no line at all', () => {
    expect(peerTwinTally('node:other', DEMO_ROWS)).toBeNull()
    expect(peerTwinTally('node:other', [])).toBeNull()
  })

  it('counts each pair once, a matching pair as not differed, and only complete pairs', () => {
    const rows = [
      twinRow('a1', 'node:p', { bracket_id: 'same', same_answer: true, other_row: 'x1' }),
      twinRow('a2', 'node:p', { bracket_id: 'differ', same_answer: false, other_row: 'x2' }),
      // the same bracket seen twice for this peer is still one pair
      twinRow('a3', 'node:p', { bracket_id: 'differ', same_answer: false, other_row: 'x3' }),
      // a bracket whose other half never arrived is not a pair
      twinRow('a4', 'node:p', { bracket_id: 'lone' })
    ]
    expect(peerTwinTally('node:p', rows)).toEqual({ pairs: 2, differed: 1, adjudicated: 0 })
  })

  it('says how many pairs a referee ruled on, without saying what it ruled', () => {
    const one = [
      twinRow('a1', 'node:p', { bracket_id: 'b1', same_answer: false, other_row: 'x1', verdict: 'contradicted:node:p' }),
      twinRow('a2', 'node:p', { bracket_id: 'b2', same_answer: false, other_row: 'x2' })
    ]
    expect(peerTwinText(peerTwinTally('node:p', one)!)).toBe('Twin pairs: 2 · answers differed: 2 · 1 adjudicated, 1 not')
    const all = [twinRow('a1', 'node:p', { bracket_id: 'b1', same_answer: true, other_row: 'x1', verdict: 'corroborated' })]
    expect(peerTwinText(peerTwinTally('node:p', all)!)).toBe('Twin pairs: 1 · answers differed: 0 · adjudicated')
  })

  it("a peer's row view carries the line only when it has pairs", () => {
    const paneB = (peerId: string) =>
      ({
        peer_id: peerId,
        node: { state: 'present', text: peerId, peer_id: peerId, exchange_count: 3 },
        exchange_count: 3,
        first_seen: null,
        last_seen: null
      }) as unknown as Parameters<typeof dealtWithRowView>[0]
    expect(dealtWithRowView(paneB('node:adversary'), undefined, DEMO_ROWS).twins).toBe(
      'Twin pairs: 3 · answers differed: 3 · not adjudicated'
    )
    expect(dealtWithRowView(paneB('node:other'), undefined, DEMO_ROWS).twins).toBeNull()
    expect(dealtWithRowView(paneB('node:adversary')).twins).toBeNull()
  })
})
