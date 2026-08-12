import { describe, expect, it, vi } from 'vitest'
import { LogPageCursor } from '@/features/logs/api/ids'
import { DETAIL_ITEM_LIMIT, loadBoundedDetailPages } from '@/features/logs/api/use-log-request-details-query'

describe('loadBoundedDetailPages', () => {
  it('loads diagnostics beyond the server default page size', async () => {
    const records = Array.from({ length: 75 }, (_, index) => index)
    const fetchPage = vi.fn(async (cursor: LogPageCursor | undefined, limit: number) => {
      const start = Number(cursor?.toString() ?? 0)
      const items = records.slice(start, start + limit)
      const next = start + items.length
      return {
        items,
        nextCursor: next < records.length ? LogPageCursor.parse(String(next)) : undefined
      }
    })

    const result = await loadBoundedDetailPages(fetchPage)

    expect(result.items).toEqual(records)
    expect(result.nextCursor).toBeUndefined()
    expect(fetchPage).toHaveBeenCalledTimes(2)
  })

  it('preserves an incomplete cursor after the bounded diagnostic cap', async () => {
    const fetchPage = vi.fn(async (cursor: LogPageCursor | undefined, limit: number) => {
      const start = Number(cursor?.toString() ?? 0)
      return {
        items: Array.from({ length: limit }, (_, index) => start + index),
        nextCursor: LogPageCursor.parse(String(start + limit))
      }
    })

    const result = await loadBoundedDetailPages(fetchPage)

    expect(result.items).toHaveLength(DETAIL_ITEM_LIMIT)
    expect(result.nextCursor?.toString()).toBe(String(DETAIL_ITEM_LIMIT))
    expect(fetchPage).toHaveBeenCalledTimes(DETAIL_ITEM_LIMIT / 50)
  })

  it('fails closed when the server cannot advance an empty page', async () => {
    await expect(
      loadBoundedDetailPages(async () => ({ items: [], nextCursor: LogPageCursor.parse('50') }))
    ).rejects.toThrow('empty page')
  })
})
