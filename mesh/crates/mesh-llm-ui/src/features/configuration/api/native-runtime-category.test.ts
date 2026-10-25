import { describe, expect, it } from 'vitest'
import { createConfigurationRuntimeSettingsFromSchema } from './config-adapter-schema'
import { createRuntimePolicySettingsFromSchema } from './runtime-settings'
import { schemaSetting } from './config-adapter-test-support'
import type { RuntimeConfigSchemaEntry } from './config-adapter-types'

const paths = [
  'runtime.native_runtime.selection',
  'runtime.native_runtime.mesh_version',
  'runtime.native_runtime.skippy_abi'
]

function nativeEntries(withPresentation: boolean): RuntimeConfigSchemaEntry[] {
  return paths.map((path, index) => ({
    ...schemaSetting(path, path.split('.').at(-1)!, { kind: 'string' }),
    presentation: withPresentation
      ? { category_id: 'runtime', category_label: 'Runtime', setting_order: 100 + index * 10 }
      : undefined
  }))
}

describe('native runtime settings category', () => {
  it.each([true, false])('groups all backend pins under Runtime, presentation present: %s', (withPresentation) => {
    const schema = { settings: nativeEntries(withPresentation) }
    for (const adapt of [createConfigurationRuntimeSettingsFromSchema, createRuntimePolicySettingsFromSchema]) {
      const result = adapt(schema)
      expect(result.settings.map((setting) => setting.categoryId)).toEqual(['runtime', 'runtime', 'runtime'])
      expect(result.categories).toEqual([expect.objectContaining({ id: 'runtime', label: 'Runtime' })])
      expect(result.settings.map((setting) => setting.canonicalPath).sort()).toEqual([...paths].sort())
      expect(result.settings.map((setting) => setting.tomlSection)).toEqual([
        'runtime.native_runtime',
        'runtime.native_runtime',
        'runtime.native_runtime'
      ])
      if (withPresentation) expect(result.settings.map((setting) => setting.settingOrder)).toEqual([100, 110, 120])
    }
  })

  it('keeps other runtime settings in Runtime Policy when metadata is absent', () => {
    const entry = schemaSetting('runtime.activity.enabled', 'enabled', { kind: 'boolean' })
    entry.presentation = undefined
    const result = createConfigurationRuntimeSettingsFromSchema({ settings: [entry, ...nativeEntries(false)] })
    expect(result.settings.find((setting) => setting.canonicalPath === 'runtime.activity.enabled')?.categoryId).toBe(
      'runtime-policy'
    )
  })
})
