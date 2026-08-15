import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  advanceLogsPage,
  closeLogInspector,
  formatRelativeTime,
  legacyRequestInspectorSearch,
  openLogInspector,
  parseLogsLedgerSearch,
  resetLogsSearch,
  resolveRelativeTime,
  toLogsRequestQuery,
  updateLogCategories,
  updateLogsFilter,
  updateLogsTimeRange
} from './log-search'
import { parseLogRequestDetailsSearch } from './log-request-details'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'
const NOW_MS = Date.parse('2026-08-04T12:00:00.000Z')

afterEach(() => vi.useRealTimers())

describe('logs ledger URL search', () => {
  it('restores supported filters and an opaque cursor from the route search (no time bounds without preset)', () => {
    const search = parseLogsLedgerSearch({
      model: 'Qwen3',
      provider: 'reserve-a',
      engine: 'skippy',
      route: 'reserve',
      source: 'durable',
      outcome: 'failed',
      cursor: 'next-page',
      trail: ['previous-page']
    })

    expect(toLogsRequestQuery(search)).toMatchObject({
      from: undefined,
      to: undefined,
      model: 'Qwen3',
      provider: 'reserve-a',
      engine: 'skippy',
      route: 'reserve',
      source: 'durable',
      outcome: 'failed'
    })
    expect(toLogsRequestQuery(search).cursor?.toString()).toBe('next-page')
    expect(search.trail).toEqual(['previous-page'])
  })

  it('resolves timeRange preset to from/to bounds at query time', () => {
    const search = parseLogsLedgerSearch({
      model: 'Qwen3',
      timeRange: '24h'
    })

    expect(search.timeRange).toBe('24h')
    const query = toLogsRequestQuery(search, NOW_MS)
    expect(query.from).toBeDefined()
    expect(query.to).toBeDefined()

    if (query.from && query.to) {
      const diffHours = (new Date(query.to).getTime() - new Date(query.from).getTime()) / 3_600_000
      expect(diffHours).toBeCloseTo(24, 1)
    }

    const bounds7d = resolveRelativeTime('7d', NOW_MS)
    if (bounds7d?.from && bounds7d?.to) {
      const diffDays = (new Date(bounds7d.to).getTime() - new Date(bounds7d.from).getTime()) / 86_400_000
      expect(diffDays).toBeCloseTo(7, 1)
    }

    const bounds12h = resolveRelativeTime('12h', NOW_MS)
    expect(bounds12h?.from).toBe(new Date(NOW_MS - 12 * 3_600_000).toISOString())

    expect(resolveRelativeTime('', NOW_MS)).toBeUndefined()
  })

  it('retains explicit legacy from/to bounds for API calls and reconnect recovery', () => {
    const search = parseLogsLedgerSearch({
      from: '2026-08-01T00:00:00Z',
      to: '2026-08-02T00:00:00Z',
      model: 'Qwen3'
    })

    expect(search).toMatchObject({ from: '2026-08-01T00:00:00Z', to: '2026-08-02T00:00:00Z' })
    expect(toLogsRequestQuery(search)).toMatchObject({
      from: '2026-08-01T00:00:00Z',
      to: '2026-08-02T00:00:00Z',
      model: 'Qwen3'
    })
    expect(parseLogsLedgerSearch({ from: 'not-a-date', to: 'also-not-a-date' })).not.toHaveProperty('from')
  })

  it('keeps opaque cursor history for next and previous pages without inventing a server limit', () => {
    const first = parseLogsLedgerSearch({ model: 'Qwen3' })
    const second = advanceLogsPage(first, 'cursor-2')
    const third = advanceLogsPage(second, 'cursor-3')

    expect(second).toMatchObject({ cursor: 'cursor-2', trail: [] })
    expect(third).toMatchObject({ cursor: 'cursor-3', trail: ['cursor-2'] })
    expect(advanceLogsPage(third, undefined)).toMatchObject({ cursor: 'cursor-2', trail: [] })
  })

  it('clears filters and pagination together', () => {
    const reset = resetLogsSearch(
      parseLogsLedgerSearch({ model: 'Qwen3', source: 'active', cursor: 'next-page', trail: ['previous-page'] })
    )

    expect(reset).toEqual({})
  })

  it('update helpers clear pagination on filter change and preserve other filters', () => {
    const base = parseLogsLedgerSearch({ model: 'Qwen3', source: 'active', cursor: 'next' })
    const updatedFilter = updateLogsFilter(base, 'engine', 'skippy')
    expect(updatedFilter.engine).toBe('skippy')
    expect(updatedFilter.cursor).toBeUndefined()

    const updatedTimeRange = updateLogsTimeRange(
      { ...base, from: '2026-08-01T00:00:00Z', to: '2026-08-02T00:00:00Z' },
      '6h'
    )
    expect(updatedTimeRange.timeRange).toBe('6h')
    expect(updatedTimeRange.cursor).toBeUndefined()
    expect(updatedTimeRange.from).toBeUndefined()
    expect(updatedTimeRange.to).toBeUndefined()

    expect(updateLogsTimeRange(updatedTimeRange, '')).not.toMatchObject({
      from: expect.anything(),
      to: expect.anything()
    })
  })

  it('parses typed inspector and multi-select category state while dropping invalid values', () => {
    expect(
      parseLogsLedgerSearch({
        inspectType: 'request',
        inspectId: REQUEST_ID,
        tab: 'routing',
        categories: ['requests', 'gossip', 'gossip', 'private']
      })
    ).toMatchObject({
      inspectType: 'request',
      inspectId: REQUEST_ID,
      tab: 'timeline',
      categories: ['requests', 'gossip']
    })
    expect(parseLogsLedgerSearch({ inspectType: 'request', inspectId: 'not-a-request-id' })).not.toHaveProperty(
      'inspectType'
    )
    expect(parseLogsLedgerSearch({ inspectType: 'audit', inspectId: 'audit-1', tab: 'errors' })).toMatchObject({
      inspectType: 'audit',
      inspectId: 'audit-1'
    })
  })

  it.each([
    ['summary', 'overview'],
    ['request', 'payloads'],
    ['response', 'payloads'],
    ['routing', 'timeline'],
    ['stream', 'timeline'],
    ['errors', 'diagnostics']
  ])('normalizes the legacy %s request tab to %s at both public parse seams', (legacyTab, canonicalTab) => {
    const detailsSearch = parseLogRequestDetailsSearch({ tab: legacyTab })
    const ledgerSearch = parseLogsLedgerSearch({ inspectType: 'request', inspectId: REQUEST_ID, tab: legacyTab })

    expect(detailsSearch.tab).toBe(canonicalTab)
    expect(ledgerSearch.tab).toBe(canonicalTab)
  })

  it.each(['overview', 'payloads', 'timeline', 'diagnostics'])(
    'keeps the canonical %s request tab unchanged at both public parse seams',
    (canonicalTab) => {
      const detailsSearch = parseLogRequestDetailsSearch({ tab: canonicalTab })
      const ledgerSearch = parseLogsLedgerSearch({
        inspectType: 'request',
        inspectId: REQUEST_ID,
        tab: canonicalTab
      })

      expect(detailsSearch.tab).toBe(canonicalTab)
      expect(ledgerSearch.tab).toBe(canonicalTab)
    }
  )

  it('falls back invalid request tabs to Overview without assigning a request tab to audit inspectors', () => {
    expect(parseLogRequestDetailsSearch({ tab: 'not-a-tab' }).tab).toBe('overview')
    expect(parseLogsLedgerSearch({ inspectType: 'request', inspectId: REQUEST_ID, tab: 'not-a-tab' }).tab).toBe(
      'overview'
    )
    expect(parseLogsLedgerSearch({ inspectType: 'audit', inspectId: 'audit-1', tab: 'not-a-tab' })).not.toHaveProperty(
      'tab'
    )
  })

  it('opens requests on Overview without dropping filters or opaque cursor history', () => {
    const search = parseLogsLedgerSearch({
      provider: 'reserve-a',
      cursor: 'next-page',
      trail: ['previous-page'],
      categories: ['requests', 'system']
    })
    const opened = openLogInspector(search, {
      type: 'request',
      id: REQUEST_ID
    })

    expect(opened).toMatchObject({
      provider: 'reserve-a',
      cursor: 'next-page',
      trail: ['previous-page'],
      categories: ['requests', 'system'],
      inspectType: 'request',
      inspectId: REQUEST_ID,
      tab: 'overview'
    })
    expect(closeLogInspector({ ...opened, tab: 'stream' })).toEqual(search)
  })

  it('encodes an explicit empty category selection without disturbing the loaded request cursor', () => {
    const search = parseLogsLedgerSearch({ cursor: 'next-page', provider: 'reserve-a' })

    expect(updateLogCategories(search, [])).toMatchObject({
      cursor: 'next-page',
      provider: 'reserve-a',
      categories: 'none'
    })
  })

  it('bridges a legacy request deep link to the canonical request inspector search', () => {
    const legacy = parseLogsLedgerSearch({ provider: 'reserve-a', cursor: 'next-page', trail: ['previous-page'] })

    expect(legacyRequestInspectorSearch(REQUEST_ID, { ...legacy, tab: 'errors' })).toMatchObject({
      provider: 'reserve-a',
      cursor: 'next-page',
      trail: ['previous-page'],
      inspectType: 'request',
      inspectId: REQUEST_ID,
      tab: 'diagnostics'
    })
  })

  it('formatRelativeTime produces readable labels for various age ranges', () => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW_MS)
    const now = new Date(NOW_MS).toISOString()
    expect(formatRelativeTime(now)).toContain('just now')

    const thirtyMinAgo = new Date(NOW_MS - 30 * 60_000).toISOString()
    expect(formatRelativeTime(thirtyMinAgo)).toMatch(/\d+m ago/)

    const twoHoursAgo = new Date(NOW_MS - 2 * 3_600_000).toISOString()
    expect(formatRelativeTime(twoHoursAgo)).toContain('2h')

    const threeDaysAgo = new Date(NOW_MS - 3 * 86_400_000).toISOString()
    expect(formatRelativeTime(threeDaysAgo)).toContain('3d')

    const thirtyDaysAgo = new Date(NOW_MS - 30 * 86_400_000).toISOString()
    const oldLabel = formatRelativeTime(thirtyDaysAgo)
    expect(oldLabel.length).toBeGreaterThan(5)
  })
})
