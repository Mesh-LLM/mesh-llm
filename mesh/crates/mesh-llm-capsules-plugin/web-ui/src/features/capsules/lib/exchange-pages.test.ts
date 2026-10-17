import { describe, expect, it } from 'vitest'
import {
  flattenPage,
  fullRangeLabel,
  groupStreamRows,
  indexGroupsWithinPage,
  isTwinBracketGroup,
  pageBoundaryContinuity,
  pageIndexForGroupKey,
  pageIndexForRowKey,
  pageRowRange,
  paginateGroups,
  windowBannerHeadline,
  type ExchangeRowGroup
} from '@/features/capsules/lib/exchange-pages'
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import type { PaneCRow } from '@/features/capsules/api/sidecarTypes'

function streamRow(overrides: Partial<ExchangeLedgerRow> = {}): ExchangeLedgerRow {
  return {
    exchangeKey: 'exch-1',
    timestamp: '2026-09-08T00:00:00Z',
    roleTag: 'ASKED',
    counterparty: null,
    confirmed: false,
    localOnly: false,
    hasIssue: false,
    checksText: '—',
    rightCellState: { kind: 'open_not_asked', date: null },
    contentToggleState: { your: { kind: 'populated', date: null }, their: { kind: 'not_asked', date: null } },
    sessionId: null,
    twinBracketId: null,
    raw: {} as PaneCRow,
    ...overrides
  }
}

function singletonGroups(n: number): ExchangeRowGroup[] {
  return Array.from({ length: n }, (_, i) => streamRow({ exchangeKey: `exch-${i}` })).map((row) => ({
    groupKey: row.exchangeKey,
    rows: [row]
  }))
}

describe('groupStreamRows', () => {
  it('makes one atomic group per row, in order', () => {
    const rows = [streamRow({ exchangeKey: 'a' }), streamRow({ exchangeKey: 'b' })]
    expect(groupStreamRows(rows)).toEqual([
      { groupKey: 'a', rows: [rows[0]] },
      { groupKey: 'b', rows: [rows[1]] }
    ])
  })

  it('merges two ADJACENT rows sharing a non-null twinBracketId into one bracket group', () => {
    const rows = [
      streamRow({ exchangeKey: 'twin-a', twinBracketId: 'twin-xyz' }),
      streamRow({ exchangeKey: 'twin-b', twinBracketId: 'twin-xyz' })
    ]
    const groups = groupStreamRows(rows)
    expect(groups).toEqual([{ groupKey: 'twin-bracket:twin-xyz', rows }])
    expect(isTwinBracketGroup(groups[0])).toBe(true)
  })

  it('never renders a half-bracket: a lone row carrying a bracket id whose twin is missing renders as an ordinary singleton', () => {
    const rows = [
      streamRow({ exchangeKey: 'unmatched', twinBracketId: 'twin-orphan' }),
      streamRow({ exchangeKey: 'plain', twinBracketId: null })
    ]
    const groups = groupStreamRows(rows)
    expect(groups).toEqual([
      { groupKey: 'unmatched', rows: [rows[0]] },
      { groupKey: 'plain', rows: [rows[1]] }
    ])
    expect(groups.every((g) => !isTwinBracketGroup(g))).toBe(true)
  })

  it('never merges rows sharing a bracket id when a DIFFERENT exchange sits between them -- the time-ordered stream is never reordered', () => {
    const rows = [
      streamRow({ exchangeKey: 'twin-a', twinBracketId: 'twin-xyz' }),
      streamRow({ exchangeKey: 'unrelated', twinBracketId: null }),
      streamRow({ exchangeKey: 'twin-b', twinBracketId: 'twin-xyz' })
    ]
    const groups = groupStreamRows(rows)
    expect(groups.map((g) => g.groupKey)).toEqual(['twin-a', 'unrelated', 'twin-b'])
    expect(groups.every((g) => !isTwinBracketGroup(g))).toBe(true)
  })

  it('a real bracket of two never splits pagination-wise from a following unrelated row on the same page', () => {
    const rows = [
      streamRow({ exchangeKey: 'twin-a', twinBracketId: 'twin-xyz' }),
      streamRow({ exchangeKey: 'twin-b', twinBracketId: 'twin-xyz' }),
      streamRow({ exchangeKey: 'plain', twinBracketId: null })
    ]
    const groups = groupStreamRows(rows)
    expect(flattenPage(groups).map((r) => r.exchangeKey)).toEqual(['twin-a', 'twin-b', 'plain'])
  })
})

describe('paginateGroups', () => {
  it('packs groups into pages of at most pageSize rows', () => {
    const pages = paginateGroups(singletonGroups(120), 50)
    expect(pages.map((page) => page.reduce((sum, g) => sum + g.rows.length, 0))).toEqual([50, 50, 20])
  })

  it('LOAD-BEARING — v3 §2a: a twin bracket (a group of >1 row) never splits across a page boundary', () => {
    // 48 singleton groups (48 rows) then a synthetic 3-row bracket. At
    // pageSize 50, 48 + 3 = 51 overflows -- the bracket must move whole to
    // the next page rather than filling the first page to exactly 50 by
    // splitting it.
    const singles = singletonGroups(48)
    const bracket: ExchangeRowGroup = {
      groupKey: 'twin-bracket-1',
      rows: [
        streamRow({ exchangeKey: 'twin-a' }),
        streamRow({ exchangeKey: 'twin-b' }),
        streamRow({ exchangeKey: 'twin-c' })
      ]
    }
    const trailing = singletonGroups(10).map((g) => ({ ...g, groupKey: `trailing-${g.groupKey}` }))

    const pages = paginateGroups([...singles, bracket, ...trailing], 50)

    expect(pages[0].reduce((sum, g) => sum + g.rows.length, 0)).toBe(48)
    expect(pages[0].some((g) => g.groupKey === 'twin-bracket-1')).toBe(false)

    const bracketPage = pages.find((page) => page.some((g) => g.groupKey === 'twin-bracket-1'))
    expect(bracketPage).toBeDefined()
    const bracketGroup = bracketPage?.find((g) => g.groupKey === 'twin-bracket-1')
    expect(bracketGroup?.rows.map((r) => r.exchangeKey)).toEqual(['twin-a', 'twin-b', 'twin-c'])
  })

  it('gives an oversized group (bigger than pageSize) a whole page to itself rather than splitting it', () => {
    const hugeBracket: ExchangeRowGroup = {
      groupKey: 'huge',
      rows: Array.from({ length: 60 }, (_, i) => streamRow({ exchangeKey: `huge-${i}` }))
    }
    const pages = paginateGroups([hugeBracket], 50)
    expect(pages).toHaveLength(1)
    expect(pages[0][0].rows).toHaveLength(60)
  })

  it('returns no pages for an empty group list', () => {
    expect(paginateGroups([], 50)).toEqual([])
  })
})

describe('indexGroupsWithinPage', () => {
  it('gives each group its starting flat-row index, accounting for a multi-row bracket', () => {
    const page: ExchangeRowGroup[] = [
      { groupKey: 'plain-1', rows: [streamRow({ exchangeKey: 'plain-1' })] },
      {
        groupKey: 'twin-bracket:twin-xyz',
        rows: [streamRow({ exchangeKey: 'twin-a' }), streamRow({ exchangeKey: 'twin-b' })]
      },
      { groupKey: 'plain-2', rows: [streamRow({ exchangeKey: 'plain-2' })] }
    ]
    expect(indexGroupsWithinPage(page)).toEqual([
      { group: page[0], startIndex: 0 },
      { group: page[1], startIndex: 1 },
      { group: page[2], startIndex: 3 }
    ])
  })

  it('is empty for an empty page', () => {
    expect(indexGroupsWithinPage([])).toEqual([])
  })
})

describe('flattenPage', () => {
  it('flattens groups back into row order', () => {
    const groups = singletonGroups(3)
    expect(flattenPage(groups).map((r) => r.exchangeKey)).toEqual(['exch-0', 'exch-1', 'exch-2'])
  })
})

describe('pageIndexForGroupKey', () => {
  it('finds the page containing a group', () => {
    const pages = paginateGroups(singletonGroups(120), 50)
    expect(pageIndexForGroupKey(pages, 'exch-0')).toBe(0)
    expect(pageIndexForGroupKey(pages, 'exch-50')).toBe(1)
    expect(pageIndexForGroupKey(pages, 'exch-119')).toBe(2)
  })

  it('returns null when the key is not present in any page (e.g. filtered out)', () => {
    const pages = paginateGroups(singletonGroups(10), 50)
    expect(pageIndexForGroupKey(pages, 'not-there')).toBeNull()
  })
})

describe('pageIndexForRowKey', () => {
  it('finds a row inside a multi-row group, whose group key is not the row key', () => {
    const pages = paginateGroups(
      [
        ...singletonGroups(50),
        { groupKey: 'twin:t1', rows: [streamRow({ exchangeKey: 'twin-a' }), streamRow({ exchangeKey: 'twin-b' })] }
      ],
      50
    )
    expect(pageIndexForGroupKey(pages, 'twin-b')).toBeNull()
    expect(pageIndexForRowKey(pages, 'twin-b')).toBe(1)
    expect(pageIndexForRowKey(pages, 'exch-3')).toBe(0)
    expect(pageIndexForRowKey(pages, 'not-there')).toBeNull()
  })
})

describe('pageRowRange', () => {
  it('computes the [start, end) row range for each page', () => {
    const pages = paginateGroups(singletonGroups(120), 50)
    expect(pageRowRange(pages, 0)).toEqual({ start: 0, end: 50 })
    expect(pageRowRange(pages, 1)).toEqual({ start: 50, end: 100 })
    expect(pageRowRange(pages, 2)).toEqual({ start: 100, end: 120 })
  })
})

describe('pageBoundaryContinuity', () => {
  it('flags a session rail that started on the previous page and continues onto this one', () => {
    // Page break falls mid-session: rows 0..4 = page 0, rows 5.. = page 1.
    const segments = [
      { hasRail: true, isSegmentStart: true },
      { hasRail: true, isSegmentStart: false },
      { hasRail: true, isSegmentStart: false },
      { hasRail: true, isSegmentStart: false },
      { hasRail: true, isSegmentStart: false },
      { hasRail: true, isSegmentStart: false },
      { hasRail: false, isSegmentStart: false }
    ]
    expect(pageBoundaryContinuity(segments, 0, 5)).toEqual({
      continuesFromPreviousPage: false,
      continuesToNextPage: true
    })
    expect(pageBoundaryContinuity(segments, 5, 7)).toEqual({
      continuesFromPreviousPage: true,
      continuesToNextPage: false
    })
  })

  it('reports no continuity when a session rail fits entirely within one page', () => {
    const segments = [
      { hasRail: true, isSegmentStart: true },
      { hasRail: true, isSegmentStart: false },
      { hasRail: false, isSegmentStart: false }
    ]
    expect(pageBoundaryContinuity(segments, 0, 3)).toEqual({
      continuesFromPreviousPage: false,
      continuesToNextPage: false
    })
  })

  it('the very first page never continues from a previous one, even if its first row is not a segment start', () => {
    const segments = [{ hasRail: false, isSegmentStart: false }]
    expect(pageBoundaryContinuity(segments, 0, 1).continuesFromPreviousPage).toBe(false)
  })
})

describe('fullRangeLabel — L-K', () => {
  it('formats the oldest–newest span across the FULL set, not a page', () => {
    const rows = [
      streamRow({ timestamp: '2026-09-11T16:58:05Z' }),
      streamRow({ timestamp: '2026-09-03T09:00:00Z' }),
      streamRow({ timestamp: '2026-09-07T00:00:00Z' })
    ]
    expect(fullRangeLabel(rows)).toBe('3 Sep – 11 Sep')
  })

  it('collapses to a single date when every row falls on the same day', () => {
    const rows = [streamRow({ timestamp: '2026-09-11T01:00:00Z' }), streamRow({ timestamp: '2026-09-11T23:00:00Z' })]
    expect(fullRangeLabel(rows)).toBe('11 Sep')
  })

  it('returns null when no row carries a timestamp -- never fabricates a range', () => {
    expect(fullRangeLabel([streamRow({ timestamp: null })])).toBeNull()
  })

  it('returns null for an empty set', () => {
    expect(fullRangeLabel([])).toBeNull()
  })
})

describe('windowBannerHeadline — L-K', () => {
  it('states the exact v3 §2a example verbatim', () => {
    expect(windowBannerHeadline(50, 1284, '3 Sep – 11 Sep')).toBe('Showing 50 of 1,284 exchanges · 3 Sep – 11 Sep')
  })

  it('LOAD-BEARING: never states a full-range count without the range label beside it, when one is available', () => {
    const headline = windowBannerHeadline(50, 1284, '3 Sep – 11 Sep')
    expect(headline).toContain('1,284')
    expect(headline).toContain('3 Sep – 11 Sep')
  })

  it('omits the range suffix honestly (never fabricates one) when no row carries a timestamp', () => {
    expect(windowBannerHeadline(3, 3, null)).toBe('Showing 3 of 3 exchanges')
  })

  it('singular "exchange" when there is exactly one', () => {
    expect(windowBannerHeadline(1, 1, null)).toBe('Showing 1 of 1 exchange')
  })
})
