import { describe, expect, it } from 'vitest'
import type { AskedOfYouEntry, PaneBRow } from '@/features/capsules/api/sidecarTypes'
import { HARNESS_PANE_B_PAYLOAD } from '@/features/capsules/lib/peer-fixtures'
import {
  askedOfYouView,
  dealingsText,
  kindCountText,
  othersSayView,
  reasonLabel,
  subjectLabel,
  theirLogView,
  verdictRows
} from '@/features/capsules/lib/their-history-view'

const [CLEAN, ALARMED] = HARNESS_PANE_B_PAYLOAD.rows

/** The freeze-candidate capture's peer (two-node mesh, 2026-09-26): nothing
 *  fetched, nothing asked (`asked` state `absent`), no inbound log. */
const NEVER_ASKED: PaneBRow = {
  ...CLEAN,
  peer_id: 'key:71eb777777777777',
  identity: { signing_key_id: '71eb777777777777'.padEnd(64, '0'), endpoint_id: 'e5ba9d1001', node_id: null },
  history: { state: 'NOT_CHECKED' },
  verdicts: { state: 'NOT_CHECKED' },
  asked: { state: 'absent', count: 0 },
  asked_of_you: undefined
}

const SEGMENT_ROW: PaneBRow = {
  ...CLEAN,
  history: {
    state: 'verified',
    segment: {
      links: [
        {
          checkpoint: { mmr_size: 7, timestamp: '2026-09-26T05:06:00Z', witnesses: [] },
          leaf_counts: { exchange: 3, adjudication: 1 },
          leaf_digests: ['a'.repeat(64)]
        },
        {
          checkpoint: { mmr_size: 11, timestamp: '2026-09-26T05:07:00Z', witnesses: [{}] },
          leaf_counts: { exchange: 2 }
        }
      ]
    },
    adjudications: {
      state: 'enriched',
      delivered: { contradicted: { delivered: 2, acknowledged: 1, disputed: 1 } }
    }
  },
  verdicts: {
    state: 'present',
    tally: { corroborated: 1, contradicted: 0, inconclusive: 0 },
    references_asked: 3,
    references_answered: 2,
    references_tally: { corroborated: 2, contradicted: 0, inconclusive: 0 }
  }
}

function entry(overrides: Partial<AskedOfYouEntry>): AskedOfYouEntry {
  return {
    ts: '2026-09-26T05:00:00Z',
    path: 'evidence-request',
    requester_id: 'key:71eb777777777777',
    subject_kind: 'chain_segment',
    status: 'answered',
    reason: null,
    ...overrides
  }
}

describe('their history -- "never asked" is never a zero', () => {
  it('each part of an un-asked peer says so in words', () => {
    expect(theirLogView(NEVER_ASKED)).toEqual({
      kind: 'not_asked',
      text: 'Never asked — your node hasn’t asked for their log.'
    })
    expect(othersSayView(NEVER_ASKED).kind).toBe('never_asked')
    expect(othersSayView(NEVER_ASKED).text).not.toMatch(/\d/)
    expect(askedOfYouView(NEVER_ASKED).kind).toBe('not_shown')
    expect(askedOfYouView(NEVER_ASKED).text).not.toMatch(/\d/)
  })

  it('a request sent to them with no checked answer is not "never asked"', () => {
    const asked: PaneBRow = { ...NEVER_ASKED, asked: { state: 'present', count: 2 } }
    expect(theirLogView(asked)).toEqual({
      kind: 'no_answer',
      text: 'Asked — no answer from them has been checked here yet.'
    })
    // An `absent` count is "not counted", not a real zero or a real two.
    expect(theirLogView({ ...NEVER_ASKED, asked: { state: 'absent', count: 2 } }).kind).toBe('not_asked')
  })

  it('verdict columns with no data are unknown in every row, not a column of zeros', () => {
    for (const row of verdictRows(NEVER_ASKED)) {
      expect(row.byYou).toEqual({ known: false, text: 'not counted' })
      expect(row.deliveredToThem).toEqual({ known: false, text: 'not shown' })
      expect(row.byOthers).toEqual({ known: false, text: 'never asked' })
    }
  })

  it('asked zero references is a real zero, distinct from never asked', () => {
    const askedNone: PaneBRow = { ...NEVER_ASKED, verdicts: { state: 'present', references_asked: 0 } }
    expect(othersSayView(askedNone)).toMatchObject({ kind: 'asked', asked: 0, answered: 0 })
    expect(verdictRows(askedNone)[0].byOthers).toEqual({ known: true, text: '0', count: 0 })
  })

  it('a carried inbound log with nothing from this peer is a real zero, not "not shown"', () => {
    const view = askedOfYouView({ ...NEVER_ASKED, asked_of_you: { entries: [] } })
    expect(view).toMatchObject({ kind: 'shown', asked: 0, answered: 0, text: '0 requests that named you · 0 answered' })
  })

  it('a never-enriched card reads "not shown", not zero delivered', () => {
    const row: PaneBRow = {
      ...SEGMENT_ROW,
      history: { ...SEGMENT_ROW.history, adjudications: { state: 'never_enriched' } }
    }
    expect(verdictRows(row).map((r) => r.deliveredToThem.known)).toEqual([false, false, false])
  })
})

describe('their log, as shown to you', () => {
  it('lists each checkpoint with its own counts and witness entries, never summed across checkpoints', () => {
    const view = theirLogView(SEGMENT_ROW)
    expect(view.kind).toBe('shown')
    if (view.kind !== 'shown') return
    expect(view.text).toBe('2 checkpoints from their log, as your node fetched and checked it.')
    expect(view.checkpoints.map((c) => [c.size, c.witnessEntries, c.counts.map((k) => k.text)])).toEqual([
      [7, 0, ['1 verdict', '3 exchanges']],
      [11, 1, ['2 exchanges']]
    ])
  })

  it('says one of a kind in the singular, and never echoes a kind it has no words for', () => {
    expect(kindCountText('exchange_twin', 1)).toBe('1 twin exchange')
    expect(kindCountText('exchange', 2)).toBe('2 exchanges')
    expect(kindCountText('best_score', 2)).toBe('2 entries of another kind')
  })

  it('never carries a leaf digest into the view, even when the peer sent one', () => {
    expect(JSON.stringify(theirLogView(SEGMENT_ROW))).not.toContain('a'.repeat(64))
  })

  it('refused and failed are named, with the registry reason in words and an unknown reason not echoed', () => {
    expect(theirLogView({ ...CLEAN, history: { state: 'refused', refusal_reason: 'not_authorized' } }).text).toBe(
      'They refused to show their log (not shared under the sharing setting).'
    )
    expect(reasonLabel('top_rated_only')).toBe('another reason')
    expect(theirLogView({ ...CLEAN, history: { state: 'failed' } }).kind).toBe('failed')
  })

  it('a verified fetch with no segment claims only what the producer said', () => {
    const view = theirLogView(CLEAN)
    expect(view).toMatchObject({ kind: 'shown', checkpoints: [] })
    expect(view.text).toBe(
      'Your node fetched their log and it passed its checks (41 checkpoints seen). Counts per checkpoint weren’t included.'
    )
    expect(view.text).not.toMatch(/unbroken/i)
  })
})

describe('verdicts on their exchanges, by provenance', () => {
  it('keeps the three provenances in separate cells per verdict', () => {
    const [corroborated, contradicted] = verdictRows(SEGMENT_ROW)
    expect(corroborated.byYou).toMatchObject({ known: true, count: 1 })
    expect(corroborated.deliveredToThem).toMatchObject({ known: true, count: 0 })
    expect(corroborated.byOthers).toMatchObject({ known: true, count: 2 })
    expect(contradicted.deliveredToThem).toEqual({
      known: true,
      text: '2 · 1 acknowledged · 1 disputed',
      count: 2
    })
  })

  it('ignores delivered verdicts on a log that did not check out', () => {
    const row: PaneBRow = { ...SEGMENT_ROW, history: { ...SEGMENT_ROW.history, state: 'failed' } }
    expect(verdictRows(row)[1].deliveredToThem.known).toBe(false)
  })

  it('reads the Python sidecar shape (a zero tally is always present) as known zeros', () => {
    const sidecar: PaneBRow = {
      ...NEVER_ASKED,
      verdicts: { state: 'pending', tally: { corroborated: 0, contradicted: 0, inconclusive: 0 } }
    }
    expect(verdictRows(sidecar).map((r) => r.byYou)).toEqual(
      [0, 0, 0].map((n) => ({ known: true, text: '0', count: n }))
    )
  })
})

describe('what others say, dealings', () => {
  it('two counts, never a ratio', () => {
    expect(othersSayView(SEGMENT_ROW).text).toBe('Asked 3 references · 2 answered')
    expect(dealingsText(CLEAN)).toBe('24 exchanges · 16 confirmed by the other side')
  })

  it('names refusal records as records of THEM refusing, not references declining', () => {
    const row: PaneBRow = { ...SEGMENT_ROW, verdicts: { ...SEGMENT_ROW.verdicts, ack_refusals: 1 } }
    expect(othersSayView(row).text).toBe(
      'Asked 3 references · 2 answered · 1 record of them refusing a verdict delivered to them'
    )
  })

  it('dealings carry the same "differ" note as the confirmed column', () => {
    expect(dealingsText(ALARMED)).toMatch(/ · \d+ differ$/)
  })
})

describe('asked of you', () => {
  const peer = NEVER_ASKED

  it('lists this peer’s requests newest first with outcome and reason; a refusal is not answered', () => {
    const view = askedOfYouView({
      ...peer,
      asked_of_you: {
        entries: [
          entry({}),
          entry({ ts: '2026-09-26T06:00:00Z', subject_kind: 'record', status: 'refused', reason: 'not_authorized' })
        ]
      }
    })
    expect(view.kind).toBe('shown')
    if (view.kind !== 'shown') return
    expect(view.text).toBe('2 requests that named you · 1 answered')
    expect(view.lines).toEqual([
      {
        when: '26 Sep 06:00 UTC',
        who: 'key:71eb777777777777',
        subject: 'one record',
        outcome: 'refused',
        reason: 'not shared under the sharing setting'
      },
      {
        when: '26 Sep 05:00 UTC',
        who: 'key:71eb777777777777',
        subject: 'a piece of your log',
        outcome: 'answered',
        reason: null
      }
    ])
  })

  it('matches a request by any id the row is known by (endpoint alias here)', () => {
    const view = askedOfYouView({ ...peer, asked_of_you: { entries: [entry({ requester_id: 'e5ba9d1001' })] } })
    expect(view).toMatchObject({ kind: 'shown', asked: 1 })
  })

  it('leaves out lines naming no one, naming another node, and record pushes', () => {
    const view = askedOfYouView({
      ...peer,
      asked_of_you: {
        entries: [
          entry({ requester_id: null }),
          entry({ requester_id: 'node:a70d3967bea3b22f' }),
          entry({ path: 'evidence/record-push', requester_id: 'key:71eb777777777777', status: 'received' })
        ]
      }
    })
    expect(view).toMatchObject({ kind: 'shown', asked: 0, answered: 0, lines: [] })
  })

  it('never echoes a subject kind it has no words for', () => {
    expect(subjectLabel('reputation_dump')).toBe('something else')
    expect(subjectLabel(null)).toBe('not stated')
  })
})
