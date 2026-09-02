import { describe, expect, it } from 'vitest'

import { cleanupScopeFromQuery } from './LogCleanupScope'

describe('cleanupScopeFromQuery', () => {
  it('preserves the ledger exact-route exclusion without forwarding its prefix exclusion', () => {
    const scope = cleanupScopeFromQuery({
      excludeRoutePrefix: 'management_'
    })

    expect(scope).toHaveProperty('excludeRoute', 'models')
    expect(scope).not.toHaveProperty('excludeRoutePrefix')
  })
})
