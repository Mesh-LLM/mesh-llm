import { afterEach, describe, expect, it, vi } from 'vitest'
import { LogsApiClient } from '@/features/logs/api/client'
import { LogPageCursor } from '@/features/logs/api/ids'
import {
  LEDGER_MAX_RECORDS,
  LEDGER_PAGE_SIZE,
  loadCompleteLedger,
  logsKeys
} from '@/features/logs/api/use-logs-ledger-query'
import type { LogsRequestQuery } from '@/features/logs/api/client'

const REQUEST_QUERY = {
  cursor: LogPageCursor.parse('next-page'),
  model: 'Qwen3',
  provider: 'reserve-a',
  engine: 'skippy',
  route: 'reserve',
  source: 'durable',
  outcome: 'completed',
  sort: 'desc'
} as const satisfies LogsRequestQuery

describe('logsKeys.ledger', () => {
  it('serializes opaque cursors deterministically in the request cache key', () => {
    const equivalentCursor = { toString: () => 'next-page' } as LogPageCursor

    expect(logsKeys.ledger({ ...REQUEST_QUERY, cursor: equivalentCursor }, 'live')).toEqual(
      logsKeys.ledger(REQUEST_QUERY, 'live')
    )
  })

  it('changes the request cache key when a server filter changes', () => {
    const filteredQuery = { ...REQUEST_QUERY, provider: 'reserve-b' } satisfies LogsRequestQuery

    expect(logsKeys.ledger(filteredQuery, 'live')).not.toEqual(logsKeys.ledger(REQUEST_QUERY, 'live'))
  })

  it('includes the ledger route exclusions in the stable request cache key', () => {
    expect(logsKeys.ledger(REQUEST_QUERY, 'live')).toContainEqual(
      expect.objectContaining({ excludeRoute: 'models', excludeRoutePrefix: 'management_' })
    )
  })
})

describe('loadCompleteLedger', () => {
  afterEach(() => vi.restoreAllMocks())

  it('aggregates server pages with the original filters instead of treating one page as complete', async () => {
    const listRequests = vi
      .spyOn(LogsApiClient.prototype, 'listRequests')
      .mockResolvedValueOnce({
        state: 'supported',
        value: { items: ['first'] as never[], nextCursor: LogPageCursor.parse('page-2') }
      })
      .mockResolvedValueOnce({ state: 'supported', value: { items: ['second'] as never[], nextCursor: undefined } })

    const result = await loadCompleteLedger(
      { from: '2026-08-01T00:00:00Z', to: '2026-08-02T00:00:00Z', model: 'Qwen3' },
      'live'
    )

    expect(result).toEqual({ state: 'supported', value: { items: ['first', 'second'], nextCursor: undefined } })
    expect(listRequests).toHaveBeenNthCalledWith(
      1,
      {
        from: '2026-08-01T00:00:00Z',
        to: '2026-08-02T00:00:00Z',
        model: 'Qwen3',
        excludeRoute: 'models',
        excludeRoutePrefix: 'management_',
        cursor: undefined,
        limit: LEDGER_PAGE_SIZE
      },
      'live'
    )
    expect(listRequests).toHaveBeenNthCalledWith(
      2,
      {
        from: '2026-08-01T00:00:00Z',
        to: '2026-08-02T00:00:00Z',
        model: 'Qwen3',
        excludeRoute: 'models',
        excludeRoutePrefix: 'management_',
        cursor: LogPageCursor.parse('page-2'),
        limit: LEDGER_PAGE_SIZE
      },
      'live'
    )
  })

  it('retains an explicit cursor and incomplete state at the safety cap', async () => {
    const listRequests = vi.spyOn(LogsApiClient.prototype, 'listRequests').mockImplementation(async (query = {}) => {
      const page = Number(query.cursor?.toString() ?? 0)
      return {
        state: 'supported',
        value: {
          items: Array.from({ length: LEDGER_PAGE_SIZE }, (_, index) => page * LEDGER_PAGE_SIZE + index) as never[],
          nextCursor: LogPageCursor.parse(String(page + 1))
        }
      }
    })

    const result = await loadCompleteLedger({}, 'live')

    expect(result).toMatchObject({
      state: 'supported',
      value: { items: Array.from({ length: LEDGER_MAX_RECORDS }, (_, index) => index), incomplete: true }
    })
    if (result.state === 'supported') expect(result.value.nextCursor?.toString()).toBe('10')
    expect(listRequests).toHaveBeenCalledTimes(10)
    for (const [query] of listRequests.mock.calls) {
      expect(query).toMatchObject({ excludeRoute: 'models', excludeRoutePrefix: 'management_' })
    }
  })

  it('stops safely when an empty page advertises a continuation cursor', async () => {
    const listRequests = vi.spyOn(LogsApiClient.prototype, 'listRequests').mockResolvedValue({
      state: 'supported',
      value: { items: [], nextCursor: LogPageCursor.parse('stale-next-page') }
    })

    const result = await loadCompleteLedger({}, 'live')

    expect(result).toEqual({ state: 'supported', value: { items: [], nextCursor: undefined } })
    expect(listRequests).toHaveBeenCalledOnce()
  })
})
