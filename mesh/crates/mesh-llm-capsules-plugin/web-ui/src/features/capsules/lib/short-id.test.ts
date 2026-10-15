import { describe, expect, it } from 'vitest'
import { shortId } from '@/features/capsules/lib/short-id'

describe('shortId', () => {
  it('leaves an already-short id unchanged -- no first4…last4 collapse on a short string', () => {
    expect(shortId('mine-clean')).toBe('mine-clean')
    expect(shortId('exch-known-peer')).toBe('exch-known-peer')
  })

  it('truncates a long (real digest-shaped) id to first4…last4', () => {
    const digest = 'a'.repeat(64)
    expect(shortId(digest)).toBe('aaaa…aaaa')
  })
})
