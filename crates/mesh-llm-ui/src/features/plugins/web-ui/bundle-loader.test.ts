import { describe, expect, it } from 'vitest'
import { assertPluginUiMountHandle, assertPluginUiRegistration } from '@/features/plugins/web-ui/bundle-loader'

describe('plugin UI bundle runtime contract', () => {
  it('accepts registration maps and mount handles', () => {
    expect(() => assertPluginUiRegistration({ pages: {}, configSections: {} })).not.toThrow()
    expect(() => assertPluginUiMountHandle({ unmount() {} })).not.toThrow()
  })

  it.each([
    [{}, 'pages object'],
    [{ pages: [], configSections: {} }, 'pages object'],
    [{ pages: {}, configSections: [] }, 'configSections'],
    [{ pages: { overview: true } }, "pages entry 'overview'"],
    [{ pages: {}, configSections: { settings: null } }, "configSections entry 'settings'"]
  ])('rejects malformed registration %#', (registration, message) => {
    expect(() => assertPluginUiRegistration(registration)).toThrow(message)
  })

  it.each([undefined, {}, { unmount: true }])('rejects malformed mount handle %#', (handle) => {
    expect(() => assertPluginUiMountHandle(handle)).toThrow('unmount() handle')
  })
})
