import { describe, expect, it } from 'vitest'
import {
  isPluginModelAllowlist,
  modelAllowlistEditorText,
  modelAllowlistFromEditorText,
  parsePluginModelAllowlist,
  serializePluginModelAllowlist
} from '@/features/configuration/lib/plugin-model-allowlist'

const LISTS = [['sold', 'also-sold'], ['a,b', 'c'], ['[]', 'sold'], [' padded '], ['["x"]'], ['[open'], []]

describe('plugin model allowlist', () => {
  it('recognizes only the plugin entry models key', () => {
    expect(isPluginModelAllowlist({ canonicalPath: 'plugin.demo.models' })).toBe(true)
    expect(
      isPluginModelAllowlist({
        canonicalPath: 'plugin.com.example.models',
        tomlSection: 'plugin.com.example',
        tomlKey: 'models'
      })
    ).toBe(true)
    expect(isPluginModelAllowlist({ canonicalPath: 'plugin.demo.args' })).toBe(false)
    expect(isPluginModelAllowlist({ canonicalPath: 'plugin.demo.settings.models' })).toBe(false)
  })

  it('round-trips every list through the form value', () => {
    for (const models of LISTS) {
      expect(parsePluginModelAllowlist(serializePluginModelAllowlist(models)), JSON.stringify(models)).toEqual(models)
    }
    expect(serializePluginModelAllowlist(['sold', 'also-sold'])).toBe('sold,also-sold')
    expect(serializePluginModelAllowlist([])).toBe('[]')
  })

  it('round-trips every list through the editor text', () => {
    for (const models of LISTS) {
      const value = serializePluginModelAllowlist(models)
      expect(modelAllowlistFromEditorText(modelAllowlistEditorText(value)), JSON.stringify(models)).toBe(value)
    }
    expect(modelAllowlistEditorText('')).toBe('')
    expect(modelAllowlistFromEditorText('')).toBe('')
  })

  it('reads one model ID per line, commas and spaces included', () => {
    expect(modelAllowlistFromEditorText('a,b\n\nc\r\n')).toBe(serializePluginModelAllowlist(['a,b', 'c']))
    expect(modelAllowlistFromEditorText(' padded ')).toBe(serializePluginModelAllowlist([' padded ']))
    expect(modelAllowlistFromEditorText('[]')).toBe('[]')
    expect(modelAllowlistFromEditorText('  \n ')).toBe('')
  })
})
