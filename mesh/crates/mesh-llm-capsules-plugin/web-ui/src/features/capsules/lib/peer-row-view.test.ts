// Honesty-backbone invariants for the Peers tab (
// ): with-you/their-chain facts are never
// summed, adjudications always carry a denominator, NOT_CHECKED never
// renders as a silent corroborated pass, and the new accountability columns
// (confirmed-by-other-side, match, period) never invent a fraction the
// sidecar doesn't back. These are the properties an adversarial reviewer
// would try to break, so they're asserted directly against the pure
// view-model functions rather than only through component snapshots.
import { describe, expect, it } from 'vitest'
import type { PaneBConfirmedSibling, PaneBRow } from '@/features/capsules/api/sidecarTypes'
import { LatencySource } from '@/lib/api/types'
import type { PeerMeshStatus } from '@/features/capsules/lib/peer-mesh-status'
import {
  adjudicationCompactText,
  adjudicationSummary,
  adjudicationSummaryText,
  alarmSignal,
  advertisedOnlyRowView,
  confirmedByOtherSide,
  confirmedByOtherSideText,
  dealtWithRowView,
  isUnattributedPeerRow,
  matchTally,
  matchTallyText,
  meshMetaLine,
  peerAliasLine,
  peerDisplayId,
  periodRangeText,
  peerSortKey,
  PEER_COLUMN_INFO,
  SELF_REPORTED_DETAIL,
  sortPeerRows,
  theirChainSummary,
  unattributedExchangesLine,
  withYouCounts,
  withYouCountsText,
  WITNESS_COVERAGE_COMPACT_TEXT
} from '@/features/capsules/lib/peer-row-view'
import { fixtureMineCell, fixtureTheirsCell } from '@/features/capsules/lib/pushed-half-fixtures'
import { STANDING_WORD } from '@/features/capsules/lib/banned-terms'

function baseRow(overrides: Partial<PaneBRow> = {}): PaneBRow {
  return {
    peer_id: 'node:test-peer',
    node: { state: 'present', text: 'node:test-peer' },
    rung: { state: 'present', text: 'full_bilateral', rung: 'full_bilateral' },
    role: { state: 'present', text: '', role: 'both', you_to_them_count: 0, them_to_you_count: 0, exchange_count: 0 },
    history: { state: 'NOT_CHECKED', text: null },
    served: { state: 'NOT_CHECKED', text: null },
    pair: { state: 'absent', text: null, verified: 0, failed: 0, missing: 0, details: [] },
    verdicts: { state: 'NOT_CHECKED', text: null, tally: { corroborated: 0, contradicted: 0, inconclusive: 0 } },
    asked: { state: 'absent', text: null, count: 0 },
    exchange_count: 0,
    first_seen: null,
    last_seen: null,
    ...overrides
  }
}

/** A pushed counterparty half as the Rust pane supplies it: both bodies, the
 *  door's verdict and the browser's id recompute. `verified` -> the bodies
 *  agree (closes); `failed` -> a digest differs (contradicts); `absent` -> the
 *  pushed body carries no digests to compare (not confirmed). Same builders as
 *  the Pane C fixtures so the two panes are pinned against the SAME gate. */
function confirmedSibling(
  overrides: Partial<PaneBConfirmedSibling['theirs']> & { matchState?: 'verified' | 'failed' | 'absent' } = {}
): PaneBConfirmedSibling {
  const { matchState = 'verified', ...theirs } = overrides
  const base = fixtureTheirsCell(matchState === 'failed' ? 'disagrees' : 'agrees', { receivedFrom: 'node:node-a' })
  if (matchState === 'absent' && base.record) {
    const { effect: _noDigests, ...record } = base.record
    base.record = record
  }
  return {
    mine: fixtureMineCell(),
    theirs: { ...base, ...theirs },
    digest_match: { state: matchState }
  }
}

describe('withYouCounts — fold-in finding 1: confirmed comes from the ONE gate', () => {
  it('reads requested/served from role_and_count_cell and confirmed from the gate — NEVER pair.verified', () => {
    const row = baseRow({
      role: {
        state: 'present',
        text: '',
        role: 'both',
        you_to_them_count: 24,
        them_to_you_count: 12,
        exchange_count: 36
      },
      // ADVERSARIAL: pair.verified says 20, but no native payload populates
      // it (always 0 live) -- the inspector Overview must not read it. The
      // gate closes exactly the two siblings below.
      pair: { state: 'verified', text: '', verified: 20, failed: 0, missing: 0, details: [] },
      confirmed_siblings: [confirmedSibling(), confirmedSibling()]
    })
    expect(withYouCounts(row)).toEqual({ requested: 24, served: 12, confirmed: 2 })
    expect(withYouCountsText(withYouCounts(row))).toBe('24 requested · 12 served · 2 confirmed')
  })

  it('PINNED: the inspector Overview count equals the Peers table count — one predicate, one number, two surfaces', () => {
    const rows = [
      baseRow({ exchange_count: 6, confirmed_siblings: [confirmedSibling(), confirmedSibling(), confirmedSibling()] }),
      baseRow({ exchange_count: 2, confirmed_siblings: [] }),
      baseRow({
        exchange_count: 4,
        confirmed_siblings: [confirmedSibling(), confirmedSibling({ matchState: 'failed' })]
      })
    ]
    for (const row of rows) {
      expect(withYouCounts(row).confirmed).toBe(confirmedByOtherSide(row).confirmed)
    }
  })
})

describe('adjudicationSummary honesty invariants', () => {
  it('never summarizes a NOT_CHECKED verdicts cell as corroborated, even with a nonzero exchange_count', () => {
    const row = baseRow({
      exchange_count: 24,
      verdicts: { state: 'NOT_CHECKED', text: null, tally: { corroborated: 0, contradicted: 0, inconclusive: 0 } }
    })
    const summary = adjudicationSummary(row)
    expect(summary.notChecked).toBe(true)
    const text = adjudicationSummaryText(summary)
    expect(text).not.toMatch(/corroborated/)
    expect(text.toLowerCase()).toContain('not yet checked')
    expect(adjudicationCompactText(summary)).toBe('none')
  })

  it('always carries the exchange_count denominator alongside a real tally, compact form matches the full form', () => {
    const row = baseRow({
      exchange_count: 24,
      verdicts: { state: 'present', text: '', tally: { corroborated: 8, contradicted: 0, inconclusive: 0 } }
    })
    const summary = adjudicationSummary(row)
    expect(summary).toEqual({
      checked: 8,
      denominator: 24,
      corroborated: 8,
      contradicted: 0,
      inconclusive: 0,
      notChecked: false
    })
    expect(adjudicationSummaryText(summary)).toBe('8 of 24 adjudicated · 8 corroborated')
    expect(adjudicationCompactText(summary)).toBe('8 of 24 · 8 corroborated')
  })

  it('never drops a contradicted/inconclusive count out of either the full or compact summary text', () => {
    const row = baseRow({
      exchange_count: 14,
      verdicts: { state: 'contradicted', text: '', tally: { corroborated: 6, contradicted: 1, inconclusive: 2 } }
    })
    const summary = adjudicationSummary(row)
    expect(adjudicationSummaryText(summary)).toBe(
      '9 of 14 adjudicated · 6 corroborated · 1 contradicted · 2 inconclusive'
    )
    expect(adjudicationCompactText(summary)).toBe('9 of 14 · 6 corroborated · 1 contradicted · 2 inconclusive')
  })

  it('treats an all-zero tally as not-checked even if the cell state claims "present"', () => {
    // Defends against a future backend regression where a present-but-empty
    // tally could otherwise read as "0 corroborated" (which looks like a
    // clean bill, not "nothing was checked").
    const row = baseRow({
      exchange_count: 5,
      verdicts: { state: 'present', text: '', tally: { corroborated: 0, contradicted: 0, inconclusive: 0 } }
    })
    expect(adjudicationSummary(row).notChecked).toBe(true)
  })
})

describe('theirChainSummary — never presents your own chain as theirs', () => {
  it('labels the common NOT_CHECKED case honestly, without the word "pending"', () => {
    const row = baseRow({
      history: {
        state: 'NOT_CHECKED',
        text: null,
        mine_for_reference: { history: { state: 'verified', text: '', history_depth: 9, checkpoint_count: 9 } }
      }
    })
    const chain = theirChainSummary(row)
    expect(chain.state).toBe('NOT_CHECKED')
    expect(chain.ownChainForReferenceOnly).toBe(true)
    expect(chain.text.toLowerCase()).not.toMatch(/\bpending\b/)
  })

  it('only claims "unbroken" when a real peer-fetch verified history_summary backs it', () => {
    const row = baseRow({
      history: {
        state: 'verified',
        text: '',
        history_summary: { verified_bundles: 5, checkpoint_count: 41 }
      }
    })
    const chain = theirChainSummary(row)
    expect(chain.text).toContain('41')
    expect(chain.text).toMatch(/^Unbroken/)
    expect(chain.ownChainForReferenceOnly).toBe(false)
  })

  it('reports a failed verification honestly, never as unbroken', () => {
    const row = baseRow({ history: { state: 'failed', text: 'fetch verification failed: chain diverged' } })
    const chain = theirChainSummary(row)
    expect(chain.text).not.toMatch(/^Unbroken/)
    expect(chain.text).toMatch(/failed/i)
  })
})

describe('confirmedByOtherSide — routed through the ONE gate, the same predicate Pane C uses', () => {
  it('states "none received yet" as a fact (never "pending" work) when no half has arrived from the other side', () => {
    const row = baseRow({ exchange_count: 11, confirmed_siblings: [] })
    const summary = confirmedByOtherSide(row)
    expect(summary).toEqual({ confirmed: 0, total: 11, note: 'none confirmed yet' })
    expect(confirmedByOtherSideText(summary)).toBe('0 of 11')
    // ADVERSARIAL: never resurrect the retired browser-peer-fetch framing.
    expect(confirmedByOtherSideText(summary)).not.toMatch(/pending|fetch/i)
  })

  it('counts a half confirmed only when the ONE gate closes it (signature_ok + verified digest_match) -- never a fetch predicate', () => {
    const row = baseRow({
      exchange_count: 3,
      confirmed_siblings: [confirmedSibling(), confirmedSibling(), confirmedSibling()]
    })
    const summary = confirmedByOtherSide(row)
    expect(summary).toEqual({ confirmed: 3, total: 3, note: null })
    expect(confirmedByOtherSideText(summary)).toBe('3 of 3')
  })

  it('denominator is the DISTINCT-exchange count, so 3 confirmed exchanges read "3 of 3", never "3 of 6"', () => {
    // The host now sends exchange_count as the distinct-exchange count
    // (`distinct_exchange_count` in capsule_panes_native.rs), NOT the record
    // count -- 3 exchanges, whose 6 halves would have read "3 of 6" off a
    // record count. The peer-row view reads that field verbatim for both the
    // Exchanges cell and the Confirmed denominator, so both track the fix.
    const row = baseRow({
      exchange_count: 3,
      confirmed_siblings: [confirmedSibling(), confirmedSibling(), confirmedSibling()]
    })
    expect(confirmedByOtherSideText(confirmedByOtherSide(row))).toBe('3 of 3')
    expect(dealtWithRowView(row).exchangeCount).toBe(3)
    expect(dealtWithRowView(row).confirmedByOtherSide).toBe('3 of 3')
  })

  it('never closes a sibling the gate does not close (no signature, or no digest match)', () => {
    const row = baseRow({
      exchange_count: 2,
      confirmed_siblings: [confirmedSibling({ signature_ok: false }), confirmedSibling({ matchState: 'absent' })]
    })
    expect(confirmedByOtherSide(row)).toEqual({ confirmed: 0, total: 2, note: 'none confirmed yet' })
  })

  it('names a contradicted half distinctly, so the count and the alarm can never diverge', () => {
    const row = baseRow({
      exchange_count: 4,
      confirmed_siblings: [confirmedSibling(), confirmedSibling({ matchState: 'failed' })]
    })
    const summary = confirmedByOtherSide(row)
    expect(summary).toEqual({ confirmed: 1, total: 4, note: '1 contradicted' })
    expect(confirmedByOtherSideText(summary)).toBe('1 of 4 · 1 differ')
  })
})

describe('matchTally — clean/mismatch of the halves the other side sent, via the SAME ONE gate', () => {
  it('always shows clean/mismatch even at zero, omits contradicted when zero', () => {
    const row = baseRow({ confirmed_siblings: Array.from({ length: 11 }, () => confirmedSibling()) })
    expect(matchTally(row)).toEqual({ clean: 11, mismatch: 0, contradicted: 0 })
    expect(matchTallyText(matchTally(row))).toBe('11 · 0 differ')
  })

  it('counts a gate-contradicted sibling as a mismatch, and appends the adjudication contradicted count when nonzero', () => {
    const row = baseRow({
      confirmed_siblings: [
        ...Array.from({ length: 9 }, () => confirmedSibling()),
        confirmedSibling({ matchState: 'failed' })
      ],
      verdicts: { state: 'contradicted', text: '', tally: { corroborated: 6, contradicted: 1, inconclusive: 0 } }
    })
    expect(matchTallyText(matchTally(row))).toBe('9 · 1 differ')
  })

  it('counts halves the door refused on their claims as differing, so Peers matches the hero', () => {
    const row = baseRow({
      confirmed_siblings: Array.from({ length: 3 }, () => confirmedSibling()),
      claims_refused: 2
    })
    expect(matchTallyText(matchTally(row))).toBe('3 · 2 differ')
  })
})

describe('periodRangeText', () => {
  it('is "—" with no exchange history to bound', () => {
    expect(periodRangeText(null, null)).toBe('—')
  })

  it('compresses a same-month range to "D–D Mon"', () => {
    expect(periodRangeText('2026-09-22T00:00:00Z', '2026-09-23T00:00:00Z')).toBe('22–23 Sep')
  })

  it('spells out both months when the range crosses a month boundary', () => {
    expect(periodRangeText('2026-08-20T09:12:00Z', '2026-09-08T16:58:05Z')).toBe('20 Aug – 8 Sep')
  })

  it('spells out both years when the range crosses a year boundary', () => {
    expect(periodRangeText('2025-12-30T00:00:00Z', '2026-01-02T00:00:00Z')).toBe('30 Dec 2025 – 2 Jan 2026')
  })
})

describe('alarmSignal', () => {
  it('surfaces a contradiction with a resolved date when the ledger has it', () => {
    const row = baseRow({
      verdicts: {
        state: 'contradicted',
        text: '',
        tally: { corroborated: 6, contradicted: 1, inconclusive: 0 },
        adjudication_capsule_id: 'cap-0007'
      }
    })
    const alarm = alarmSignal(row, (id) => (id === 'cap-0007' ? '2026-09-01' : null))
    expect(alarm).toEqual({ present: true, text: 'Contradiction found 2026-09-01', tone: 'bad' })
  })

  it('never fabricates a date when the local ledger has no matching record', () => {
    const row = baseRow({
      verdicts: {
        state: 'contradicted',
        text: '',
        tally: { corroborated: 0, contradicted: 1, inconclusive: 0 },
        adjudication_capsule_id: 'cap-unknown'
      }
    })
    const alarm = alarmSignal(row, () => null)
    expect(alarm.text).toBe('Contradiction found')
  })

  it('is absent for a clean row', () => {
    expect(alarmSignal(baseRow()).present).toBe(false)
  })
})

describe('sortPeerRows', () => {
  it('floats an alarmed row above a lower-latency clean row', () => {
    const clean = baseRow({ peer_id: 'clean' })
    const alarmed = baseRow({
      peer_id: 'alarmed',
      verdicts: { state: 'contradicted', text: '', tally: { corroborated: 0, contradicted: 1, inconclusive: 0 } }
    })
    const sorted = sortPeerRows(
      [
        { row: clean, latencyMs: 10 },
        { row: alarmed, latencyMs: 500 }
      ],
      ({ row, latencyMs }) => peerSortKey(row, latencyMs)
    )
    expect(sorted.map((entry) => entry.row.peer_id)).toEqual(['alarmed', 'clean'])
  })

  it('sorts two clean rows closest-first by latency', () => {
    const near = baseRow({ peer_id: 'near' })
    const far = baseRow({ peer_id: 'far' })
    const sorted = sortPeerRows(
      [
        { row: far, latencyMs: 200 },
        { row: near, latencyMs: 5 }
      ],
      ({ row, latencyMs }) => peerSortKey(row, latencyMs)
    )
    expect(sorted.map((entry) => entry.row.peer_id)).toEqual(['near', 'far'])
  })
})

describe('peerDisplayId — no synthetic peer', () => {
  it('ADVERSARIAL: never returns the string "unknown peer" — a row with no identity resolves to null', () => {
    const row = baseRow({ peer_id: null, node: { state: 'present', text: null } })
    expect(peerDisplayId(row)).toBeNull()
    expect(peerDisplayId(row)).not.toBe('unknown peer')
    expect(isUnattributedPeerRow(row)).toBe(true)
  })

  it('falls back to the node cell id when the row-level peer_id is absent', () => {
    const row = baseRow({ peer_id: null, node: { state: 'present', text: null, peer_id: 'node:aa11bb22cc33' } })
    expect(peerDisplayId(row)).toBe('node:aa11bb22cc33')
    expect(isUnattributedPeerRow(row)).toBe(false)
  })

  it('prefers the row-level peer_id over the node cell id when both are present', () => {
    const row = baseRow({ peer_id: 'peer-verified', node: { state: 'present', text: null, peer_id: 'node:other' } })
    expect(peerDisplayId(row)).toBe('peer-verified')
  })
})

describe('peerAliasLine — D3: aliases on ONE row, never extra peers', () => {
  it('renders "signed by <key16> · endpoint <id>" for a key-joined peer with no node evidence', () => {
    const row = baseRow({
      peer_id: 'key:71eb777777777777',
      identity: {
        signing_key_id: '71eb777777777777777777777777777777777777777777777777777777777777',
        endpoint_id: 'e5ba9d1001',
        node_id: null
      }
    })
    expect(peerAliasLine(row)).toBe('signed by 71eb777777777777 · endpoint e5ba9d1001')
  })

  it('never puts a 64-hex endpoint id on the row face: 16 characters and an ellipsis', () => {
    const endpoint = 'c6839699559fd005'.repeat(4)
    const row = baseRow({
      peer_id: 'key:71eb777777777777',
      identity: {
        signing_key_id: '71eb777777777777777777777777777777777777777777777777777777777777',
        endpoint_id: endpoint,
        node_id: null
      }
    })
    expect(peerAliasLine(row)).toBe('signed by 71eb777777777777 · endpoint c6839699559fd005…')
    expect(peerAliasLine(row)).not.toMatch(/[0-9a-f]{64}/)
  })

  it('adds the node alias only when the evidence carries one (the bridged case)', () => {
    const row = baseRow({
      peer_id: 'key:71eb777777777777',
      identity: {
        signing_key_id: '71eb777777777777777777777777777777777777777777777777777777777777',
        endpoint_id: 'e5ba9d1001',
        node_id: `a70d3967bea3b22f${'a'.repeat(48)}`
      }
    })
    expect(peerAliasLine(row)).toBe('signed by 71eb777777777777 · node a70d3967bea3b22f… · endpoint e5ba9d1001')
  })

  it('labels the UNLINKED node-id row honestly — the id is from our own records, no signing key linked yet', () => {
    const row = baseRow({
      peer_id: 'node:a70d3967bea3b22f',
      identity: { signing_key_id: null, endpoint_id: null, node_id: `a70d3967bea3b22f${'a'.repeat(48)}` }
    })
    expect(peerAliasLine(row)).toBe('node id from your own records — no signing key linked yet')
  })

  it('degrades to null (no line, never a fabricated identity) when the row carries no identity evidence', () => {
    expect(peerAliasLine(baseRow())).toBeNull()
    expect(peerAliasLine(baseRow({ identity: null }))).toBeNull()
  })

  it('the dealt-with view carries the alias line; an advertised-only view never does', () => {
    const row = baseRow({
      identity: { signing_key_id: 'ab'.repeat(32), endpoint_id: 'e5ba9d1001', node_id: null }
    })
    expect(dealtWithRowView(row).aliasLine).toBe(`signed by ${'ab'.repeat(8)} · endpoint e5ba9d1001`)
    expect(advertisedOnlyRowView('node:unused').aliasLine).toBeNull()
  })
})

describe('unattributedExchangesLine', () => {
  it('states the count as a fact, never implying pending work', () => {
    expect(unattributedExchangesLine(3)).toBe(
      '3 exchanges have no counterparty recorded yet. They appear under Exchanges.'
    )
    expect(unattributedExchangesLine(3)).not.toMatch(/not resolved yet/i)
  })

  it('uses singular grammar for a count of one', () => {
    expect(unattributedExchangesLine(1)).toBe(
      '1 exchange has no counterparty recorded yet. They appear under Exchanges.'
    )
  })
})

describe('meshMetaLine — still used by the PeerInspector modal', () => {
  function meshStatus(overrides: Partial<PeerMeshStatus> = {}): PeerMeshStatus {
    return {
      modelName: 'Qwen3.6-27B-UD',
      quant: 'Q4_K_XL',
      contextLengthK: 256,
      latencyMs: 38,
      latencySource: LatencySource.DIRECT,
      online: true,
      ...overrides
    }
  }

  it('joins model/quant/context', () => {
    expect(meshMetaLine(meshStatus())).toBe('Qwen3.6-27B-UD · Q4_K_XL · 256k ctx')
  })

  it('degrades to null (never a fabricated line) when nothing is known', () => {
    expect(meshMetaLine(null)).toBeNull()
  })
})

describe('dealtWithRowView / advertisedOnlyRowView — one shape, honest degradation', () => {
  it('a dealt-with row carries the real Pane B row, exchange count, and every accountability column', () => {
    const row = baseRow({
      peer_id: 'node:abc',
      confirmed_siblings: Array.from({ length: 16 }, () => confirmedSibling()),
      exchange_count: 24,
      first_seen: '2026-09-01T00:00:00Z',
      last_seen: '2026-09-03T00:00:00Z',
      verdicts: { state: 'present', text: '', tally: { corroborated: 8, contradicted: 0, inconclusive: 0 } }
    })
    const view = dealtWithRowView(row)
    expect(view.hasDealings).toBe(true)
    expect(view.row).toBe(row)
    expect(view.identityNote).toBe('')
    expect(view.exchangeCount).toBe(24)
    expect(view.match).toBe('16 · 0 differ')
    expect(view.adjudicationCompact).toBe('8 of 24 · 8 corroborated')
    expect(view.witnessCompact).toBe(WITNESS_COVERAGE_COMPACT_TEXT)
    expect(view.period).toBe('3 Sep')
    // 16 halves closed through the gate, 24 exchanges total.
    expect(view.confirmedByOtherSide).toBe('16 of 24')
  })

  it('a zero-dealings advertised-only peer shows "no exchanges yet" and "—" for every accountability column, never a fabricated zero-of-zero', () => {
    const view = advertisedOnlyRowView('node:unused')
    expect(view.hasDealings).toBe(false)
    expect(view.row).toBeNull()
    expect(view.identityNote).toBe('no exchanges yet')
    expect(view.exchangeCount).toBe(0)
    for (const field of [
      view.confirmedByOtherSide,
      view.match,
      view.adjudicationCompact,
      view.witnessCompact,
      view.period
    ]) {
      expect(field).toBe('—')
    }
  })

  it('no accountability figure on either view ever renders a percentage or ratio ramp (R-D)', () => {
    const row = baseRow({
      exchange_count: 24,
      verdicts: { state: 'present', text: '', tally: { corroborated: 8, contradicted: 0, inconclusive: 0 } }
    })
    const dealtWith = dealtWithRowView(row)
    const advertised = advertisedOnlyRowView('node:unused')
    for (const view of [dealtWith, advertised]) {
      for (const text of [
        view.confirmedByOtherSide,
        view.match,
        view.adjudicationCompact,
        view.witnessCompact,
        view.period
      ]) {
        expect(text).not.toContain('%')
      }
    }
  })
})

describe('Item 4 — Peers column (i) copy + self-reported detail: evidence, never a score', () => {
  it('Confirmed carries the UX §8 sentence verbatim', () => {
    expect(PEER_COLUMN_INFO.confirmed).toBe(
      'How many of your exchanges with them are confirmed by their own signed record, checked on this machine.'
    )
  })

  it('every column line names what it means / what backs it and refuses the score reading', () => {
    for (const key of ['exchanges', 'confirmed', 'match', 'adjudication', 'witness', 'period'] as const) {
      const info = PEER_COLUMN_INFO[key]
      expect(info.length).toBeGreaterThan(0)
      expect(info).not.toMatch(new RegExp(`\\b(score|rating|proven|${STANDING_WORD}|judgement|ranking)\\b`, 'i'))
    }
  })

  it('the self-reported detail states the identity is self-reported and not independently attested', () => {
    expect(SELF_REPORTED_DETAIL).toMatch(/self-reported/i)
    expect(SELF_REPORTED_DETAIL).toMatch(/only the records they signed, checked on this machine, count as evidence/i)
    expect(SELF_REPORTED_DETAIL).not.toMatch(/\b(score|rating|proven|halves|recomputed)\b/i)
  })
})
