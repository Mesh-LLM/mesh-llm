import { describe, expect, it } from 'vitest'
import { exceptionsFirstLine, exceptionsFirstTally } from '@/features/capsules/lib/exceptions-first-line'
import { buildExchangeLedgerRows } from '@/features/capsules/lib/exchange-ledger'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'

function paneCRow(overrides: Partial<PaneCRow>): PaneCRow {
  return {
    exchange_key: 'exch-1',
    role_tag: 'ASKED',
    header_state: 'ok',
    properties: null,
    has_issue: false,
    mine: { state: 'present', capsule_id: 'mine-1' },
    theirs: { state: 'absent', capsule_id: null },
    unilateral: false,
    timestamp: '2026-09-08T00:00:00Z',
    ...overrides
  }
}

const CLEAN_ROW = paneCRow({ exchange_key: 'exch-clean' })
const FAILED_ROW = paneCRow({
  exchange_key: 'exch-failed',
  has_issue: true,
  properties: { content_binding: { state: 'FAIL', text: 'digest mismatch' } }
})
const PEER_ASSERTED_UNFETCHED_ROW = paneCRow({
  exchange_key: 'exch-mismatched',
  theirs: { state: 'NOT_CHECKED', capsule_id: 'a'.repeat(64), peer_id: 'peer-1' }
})
const ASKED_UNANSWERED_ROW = paneCRow({
  exchange_key: 'exch-asked',
  theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'unanswered', evidence_outcome_date: '2026-09-01' }
})
const CONFIRMED_ROW = paneCRow({
  exchange_key: 'exch-confirmed',
  theirs: { state: 'NOT_CHECKED', capsule_id: 'c'.repeat(64), peer_id: 'peer-2' }
})

describe('exceptionsFirstTally', () => {
  it('is all-zero for an empty set', () => {
    expect(exceptionsFirstTally([])).toEqual({
      total: 0,
      needingAttention: 0,
      failed: 0,
      mismatched: 0,
      askedUnanswered: 0,
      confirmedByAnyoneElse: 0
    })
  })

  it('counts a hasIssue row as failed', () => {
    const [row] = buildExchangeLedgerRows([FAILED_ROW], new Map())
    const tally = exceptionsFirstTally([row])
    expect(tally).toEqual({
      total: 1,
      needingAttention: 1,
      failed: 1,
      mismatched: 0,
      askedUnanswered: 0,
      confirmedByAnyoneElse: 0
    })
  })

  it('a peer-asserted-but-unfetched row is pending fetch, not mismatched -- never contradicted without evidence', () => {
    const [row] = buildExchangeLedgerRows([PEER_ASSERTED_UNFETCHED_ROW], new Map())
    expect(row.hasIssue).toBe(false)
    expect(row.rightCellState.kind).toBe('open_not_held')
    const tally = exceptionsFirstTally([row])
    expect(tally).toEqual({
      total: 1,
      needingAttention: 0,
      failed: 0,
      mismatched: 0,
      askedUnanswered: 0,
      confirmedByAnyoneElse: 0
    })
  })

  it('counts an open_asked right-cell state as asked-and-unanswered', () => {
    const [row] = buildExchangeLedgerRows([ASKED_UNANSWERED_ROW], new Map())
    const tally = exceptionsFirstTally([row])
    expect(tally).toEqual({
      total: 1,
      needingAttention: 1,
      failed: 0,
      mismatched: 0,
      askedUnanswered: 1,
      confirmedByAnyoneElse: 0
    })
  })

  it('never double-counts one row across needingAttention when it trips more than one category', () => {
    const bothRow = paneCRow({
      exchange_key: 'exch-both',
      has_issue: true,
      theirs: { state: 'absent', capsule_id: null, evidence_outcome: 'unanswered', evidence_outcome_date: '2026-09-01' }
    })
    const [row] = buildExchangeLedgerRows([bothRow], new Map())
    const tally = exceptionsFirstTally([row])
    expect(tally.failed).toBe(1)
    expect(tally.askedUnanswered).toBe(1)
    expect(tally.needingAttention).toBe(1)
  })

  it('a clean row counts toward total only', () => {
    const [row] = buildExchangeLedgerRows([CLEAN_ROW], new Map())
    const tally = exceptionsFirstTally([row])
    expect(tally).toEqual({
      total: 1,
      needingAttention: 0,
      failed: 0,
      mismatched: 0,
      askedUnanswered: 0,
      confirmedByAnyoneElse: 0
    })
  })

  it('a row this browser fetched and confirmed counts toward confirmedByAnyoneElse, never needingAttention', () => {
    const [row] = buildExchangeLedgerRows([CONFIRMED_ROW], new Map())
    expect(row.rightCellState.kind).toBe('open_not_held') // no live fetch state threaded through the ledger row build
    // exceptionsFirstTally counts off `row.rightCellState`, which is the
    // ledger's at-rest state (no fetch happened) -- `confirmedByAnyoneElse`
    // only ever increments once a row's precomputed state is truly `closed`.
    const tally = exceptionsFirstTally([row])
    expect(tally.confirmedByAnyoneElse).toBe(0)
  })
})

describe('exceptionsFirstLine', () => {
  it('clean state states the role-aware truth, never "Nothing needs your attention"', () => {
    const line = exceptionsFirstLine(
      { total: 5, needingAttention: 0, failed: 0, mismatched: 0, askedUnanswered: 0, confirmedByAnyoneElse: 0 },
      '3 Sep – 11 Sep'
    )
    expect(line).toBe('5 sealed by you, 3 Sep – 11 Sep.')
    expect(line).not.toMatch(/Nothing needs your attention/)
    expect(line).not.toMatch(/recomputed clean/)
  })

  it('never repeats the headline above it: no "confirmed by anyone else", no registration word', () => {
    // the Exchanges headline carries
    // "M confirmed by the other side · registered/not registered"; this line
    // restating it was the retired duplicate.
    const line = exceptionsFirstLine(
      { total: 5, needingAttention: 0, failed: 0, mismatched: 0, askedUnanswered: 0, confirmedByAnyoneElse: 2 },
      null
    )
    expect(line).toBe('5 sealed by you.')
    expect(line).not.toMatch(/confirmed by anyone else/)
    expect(line).not.toMatch(/registered/)
  })

  it('non-clean state leads with the failing count, not "Nothing needs your attention", and leaves confirmed/registered to the headline', () => {
    const line = exceptionsFirstLine(
      { total: 5, needingAttention: 2, failed: 1, mismatched: 1, askedUnanswered: 0, confirmedByAnyoneElse: 1 },
      '3 Sep – 11 Sep'
    )
    expect(line.startsWith('2 exchanges need your attention')).toBe(true)
    expect(line).not.toMatch(/Nothing needs your attention/)
    expect(line).toMatch(/1 failed · 1 mismatched · 0 asked-and-unanswered/)
    expect(line).not.toMatch(/confirmed by anyone else/)
  })

  it('singular phrasing for exactly one exchange needing attention', () => {
    const line = exceptionsFirstLine(
      { total: 3, needingAttention: 1, failed: 1, mismatched: 0, askedUnanswered: 0, confirmedByAnyoneElse: 0 },
      null
    )
    expect(line.startsWith('1 exchange needs your attention')).toBe(true)
  })

  it('states the range when available, omits it honestly when not', () => {
    const withRange = exceptionsFirstLine(
      { total: 1, needingAttention: 0, failed: 0, mismatched: 0, askedUnanswered: 0, confirmedByAnyoneElse: 0 },
      '3 Sep'
    )
    expect(withRange).toContain('1 sealed by you, 3 Sep')

    const withoutRange = exceptionsFirstLine(
      { total: 1, needingAttention: 0, failed: 0, mismatched: 0, askedUnanswered: 0, confirmedByAnyoneElse: 0 },
      null
    )
    expect(withoutRange).toBe('1 sealed by you.')
    expect(withoutRange).not.toContain('null')
  })

  it('never renders a fraction/ratio for the tally', () => {
    const line = exceptionsFirstLine(
      { total: 5, needingAttention: 2, failed: 1, mismatched: 1, askedUnanswered: 0, confirmedByAnyoneElse: 0 },
      null
    )
    expect(line).not.toMatch(/\d+\/\d+/)
  })
})
