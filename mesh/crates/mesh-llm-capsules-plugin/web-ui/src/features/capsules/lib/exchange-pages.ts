// Windowed paging over the two-sided Ledger stream (
// v3 §2a). Pure, so the "twin brackets never split" invariant is
// unit-testable today even though nothing in the current payload produces a
// bracket bigger than one row yet (that lands in a later batch).
import type { ExchangeLedgerRow } from '@/features/capsules/lib/exchange-ledger'
import type { RailSegment } from '@/features/capsules/lib/exchange-stream'

export const LEDGER_PAGE_SIZE = 50

/** Stable DOM id for a row -- used by the keyboard/deep-link handlers in
 *  `LedgerPage.tsx` to scroll a row into view without ref plumbing. */
export function exchangeRowDomId(exchangeKey: string): string {
  return `exchange-row-${exchangeKey}`
}

/** Stable DOM id for one `▸ checks` panel property row -- the entry row's
 *  chip strip scrolls to this once
 *  the panel is expanded, keyed by the same property key
 *  (`security-checks-view.ts`'s `ChecksRow.key`) the chip names. */
export function checkRowDomId(exchangeKey: string, propertyKey: string): string {
  return `check-row-${exchangeKey}-${propertyKey}`
}

export type ExchangeRowGroup = {
  /** Stable identity for the group -- a lone row's own exchangeKey, or
   *  `twin-bracket:<id>` for a real multi-row twin bracket.
   *  The packing logic below doesn't care what the key
   *  means, only that a group is atomic. */
  groupKey: string
  rows: readonly ExchangeLedgerRow[]
}

/**
 * Groups ADJACENT rows sharing the same non-null `twinBracketId` into one
 * atomic bracket (v3 §2a "twin brackets never split"); every other row is
 * its own singleton group, as it was before twin brackets existed.
 *
 * Adjacency, not just a shared id, is required to merge: `streamRows` is
 * already time-ordered (L-N) and nothing here may reorder it, so if a
 * bracket's twin isn't next to it (interleaved by an unrelated exchange, or
 * simply missing) this renders BOTH as ordinary singleton rows rather than
 * forcing a reorder or a malformed one-row "bracket." This is also what
 * keeps a lone/unmatched bracket id from ever rendering as a half-bracket:
 * a bracket only exists once two (or more) adjacent rows actually agree on
 * one id.
 */
export function groupStreamRows(rows: readonly ExchangeLedgerRow[]): ExchangeRowGroup[] {
  const groups: ExchangeRowGroup[] = []
  let i = 0
  while (i < rows.length) {
    const row = rows[i]
    const bracketId = row.twinBracketId
    if (bracketId !== null) {
      const bracketRows: ExchangeLedgerRow[] = [row]
      let j = i + 1
      while (j < rows.length && rows[j].twinBracketId === bracketId) {
        bracketRows.push(rows[j])
        j++
      }
      if (bracketRows.length > 1) {
        groups.push({ groupKey: `twin-bracket:${bracketId}`, rows: bracketRows })
        i = j
        continue
      }
      // Only one row carries this bracket id here (its twin is missing or
      // not adjacent) -- never a half-bracket; falls through to render as
      // an ordinary singleton, same as a row with no bracket id at all.
    }
    groups.push({ groupKey: row.exchangeKey, rows: [row] })
    i++
  }
  return groups
}

/** Whether a group is a real, renderable twin bracket (>1 row). Mirrors the
 *  `groupKey` prefix `groupStreamRows` uses, kept as one predicate so
 *  callers never re-derive "is this a bracket" by string-matching the key
 *  themselves. */
export function isTwinBracketGroup(group: ExchangeRowGroup): boolean {
  return group.rows.length > 1
}

/**
 * Packs groups into pages of at most `pageSize` rows, never splitting a
 * group across a page boundary. A group larger than `pageSize` still gets a
 * whole page to itself (oversized) rather than being split -- splitting a
 * twin bracket would let a reader see one half and draw the wrong
 * conclusion (v3 §2a "twin brackets never split").
 */
export function paginateGroups(groups: readonly ExchangeRowGroup[], pageSize: number): ExchangeRowGroup[][] {
  const pages: ExchangeRowGroup[][] = []
  let currentPage: ExchangeRowGroup[] = []
  let currentCount = 0
  for (const group of groups) {
    if (currentCount > 0 && currentCount + group.rows.length > pageSize) {
      pages.push(currentPage)
      currentPage = []
      currentCount = 0
    }
    currentPage.push(group)
    currentCount += group.rows.length
  }
  if (currentPage.length > 0) pages.push(currentPage)
  return pages
}

export function flattenPage(page: readonly ExchangeRowGroup[]): ExchangeLedgerRow[] {
  return page.flatMap((group) => group.rows)
}

export type IndexedGroup = {
  group: ExchangeRowGroup
  /** This group's first row's 0-based index within the page's flattened
   *  (`flattenPage`) row order -- what a per-row prop like `rail` or
   *  keyboard-cursor `focused` needs, since those still key off the FLAT
   *  row position, not the group. */
  startIndex: number
}

/** Pairs each group in a page with its starting flat-row index, so a
 *  renderer can thread per-row props (rail segment, keyboard focus) to
 *  rows inside a multi-row bracket exactly like it does for a singleton
 *  row, without re-deriving the running offset itself. */
export function indexGroupsWithinPage(page: readonly ExchangeRowGroup[]): IndexedGroup[] {
  const indexed: IndexedGroup[] = []
  let index = 0
  for (const group of page) {
    indexed.push({ group, startIndex: index })
    index += group.rows.length
  }
  return indexed
}

/** 0-based index of the page containing `groupKey`, or null when it isn't
 *  in any page (e.g. filtered out of the current view). */
export function pageIndexForGroupKey(pages: readonly ExchangeRowGroup[][], groupKey: string): number | null {
  const index = pages.findIndex((page) => page.some((group) => group.groupKey === groupKey))
  return index === -1 ? null : index
}

/** The page holding the row with this key, whether it stands alone or sits
 *  inside a group (a twin pair's group key is not either row's key). */
export function pageIndexForRowKey(pages: readonly ExchangeRowGroup[][], rowKey: string): number | null {
  const index = pages.findIndex((page) => page.some((group) => group.rows.some((row) => row.exchangeKey === rowKey)))
  return index === -1 ? null : index
}

/** The row-index range `[start, end)` a page occupies in the flattened
 *  (filtered, time-ordered) stream -- used to slice the full-stream rail
 *  segments so cross-page continuity (L-N/L-O) can be detected per page. */
export function pageRowRange(pages: readonly ExchangeRowGroup[][], pageIndex: number): { start: number; end: number } {
  let start = 0
  for (let i = 0; i < pageIndex; i++) {
    start += pages[i].reduce((sum, group) => sum + group.rows.length, 0)
  }
  const pageSize = (pages[pageIndex] ?? []).reduce((sum, group) => sum + group.rows.length, 0)
  return { start, end: start + pageSize }
}

export type PageBoundaryContinuity = {
  /** The first row of this page carries a rail that started on the
   *  PREVIOUS page -- render "…session continues" instead of a fresh
   *  "session <id>" label (v3 §2a: "a rail crossing a boundary shows
   *  '…session continues' at the break on both pages"). */
  continuesFromPreviousPage: boolean
  /** The last row of this page carries a rail that continues onto the
   *  NEXT page. */
  continuesToNextPage: boolean
}

/**
 * `railSegments` must be built over the FULL (filtered, time-ordered)
 * stream, not just the current page -- otherwise every page's first row
 * looks like a segment start and the continuation can never be detected.
 */
export function pageBoundaryContinuity(
  railSegments: readonly RailSegment[],
  pageStart: number,
  pageEnd: number
): PageBoundaryContinuity {
  const firstOnPage = railSegments[pageStart]
  const continuesFromPreviousPage = pageStart > 0 && firstOnPage?.hasRail === true && !firstOnPage.isSegmentStart

  const lastOnPage = railSegments[pageEnd - 1]
  const firstOnNextPage = railSegments[pageEnd]
  const continuesToNextPage =
    lastOnPage?.hasRail === true && firstOnNextPage?.hasRail === true && !firstOnNextPage.isSegmentStart

  return { continuesFromPreviousPage, continuesToNextPage }
}

const MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] as const

function formatShortDate(isoTimestamp: string): string | null {
  const date = new Date(isoTimestamp)
  if (Number.isNaN(date.getTime())) return null
  return `${date.getUTCDate()} ${MONTH_ABBR[date.getUTCMonth()]}`
}

/**
 * NORMATIVE — L-K. "The view states its own boundary, and every count above
 * the table states the range it covers." `allRows` must be every row the
 * banner's total count was computed from (the full filtered set), never
 * just the current page -- passing a page here would silently mislabel a
 * page's span as the full range.
 */
export function fullRangeLabel(allRows: readonly ExchangeLedgerRow[]): string | null {
  const timestamps = allRows.map((row) => row.timestamp).filter((value): value is string => value !== null)
  if (timestamps.length === 0) return null
  const sorted = [...timestamps].sort()
  const oldest = formatShortDate(sorted[0])
  const newest = formatShortDate(sorted[sorted.length - 1])
  if (!oldest || !newest) return null
  return oldest === newest ? oldest : `${oldest} – ${newest}`
}

/**
 * NORMATIVE — L-K. The bounded-window banner headline: "Showing 50 of 1,284
 * exchanges · 3 Sep – 11 Sep". `rangeLabel` is appended whenever available
 * rather than only sometimes, so the full-range count is never shown
 * without stating the range it covers; when no row carries a timestamp
 * (`rangeLabel` is null) the count still renders honestly, just without a
 * fabricated range suffix.
 */
export function windowBannerHeadline(rowsOnPage: number, totalRows: number, rangeLabel: string | null): string {
  const noun = totalRows === 1 ? 'exchange' : 'exchanges'
  const base = `Showing ${rowsOnPage} of ${totalRows.toLocaleString('en-US')} ${noun}`
  return rangeLabel ? `${base} · ${rangeLabel}` : base
}
