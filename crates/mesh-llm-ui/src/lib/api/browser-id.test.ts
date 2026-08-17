import { afterEach, describe, expect, it, vi } from 'vitest'

import { createBrowserId } from '@/lib/api/browser-id'

describe('createBrowserId', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('uses crypto.randomUUID when it is available', () => {
    vi.stubGlobal('crypto', { randomUUID: () => 'uuid-123' })

    expect(createBrowserId('request')).toBe('uuid-123')
  })

  it('falls back when crypto.randomUUID is unavailable', () => {
    vi.stubGlobal('crypto', {})
    vi.spyOn(Date, 'now').mockReturnValue(1_718_000_000_000)
    vi.spyOn(Math, 'random').mockReturnValue(0.25)

    expect(createBrowserId('request')).toBe('request-lx8kuby8-9')
  })
})
