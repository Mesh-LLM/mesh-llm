import { describe, expect, it } from 'vitest'
import { pluginScopedApiUrl } from '@/features/plugins/web-ui/host-surface'

describe('plugin web UI host network scope', () => {
  it('keeps paths and query strings under the mounted plugin namespace', () => {
    expect(pluginScopedApiUrl('demo plugin', '/http/items?limit=2')).toContain(
      '/api/plugins/demo%20plugin/http/items?limit=2'
    )
    expect(pluginScopedApiUrl('demo', 'http/items?redirect=/one?next=two')).toContain(
      '/api/plugins/demo/http/items?redirect=/one?next=two'
    )
  })

  it.each([
    '../config',
    './config',
    'https://example.test/data',
    'http\\evil',
    'items#secret',
    'items\r\nInjected: true'
  ])('rejects an escaping or ambiguous path: %s', (path) => {
    expect(() => pluginScopedApiUrl('demo', path)).toThrow(TypeError)
  })
})
