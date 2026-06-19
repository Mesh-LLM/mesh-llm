import { expect, it } from 'vitest'
import { evaluateSettingControlState } from '@/features/configuration/lib/settings-utils'
import {
  choiceSetting,
  condition,
  textSetting,
  typedValues
} from '@/features/configuration/lib/settings-utils.test-helpers'

export function defineSettingsControlRuleTests() {
  it('uses disable_when rules and conflict rules with deterministic sources and policies', () => {
    const assignmentSetting = choiceSetting({ id: 'gpu.assignment', value: 'auto', options: ['auto', 'pinned'] })
    const deviceSetting = textSetting({
      id: 'defaults.hardware.device',
      value: 'cuda:0',
      controlBehavior: {
        disable_when: [
          {
            condition: condition('gpu.assignment', 'equals', [{ kind: 'string', value: 'auto' }]),
            reason: 'Pinned device selection is unavailable while assignment is auto.',
            note: 'Switch assignment to pinned to edit this field.',
            write_policy: 'preserve_existing'
          }
        ]
      }
    })
    const pinnedOnlySetting = textSetting({
      id: 'defaults.hardware.gpu_id',
      value: '0',
      controlBehavior: {
        conflicts: [
          {
            group: 'gpu-assignment',
            condition: condition('gpu.assignment', 'equals', [{ kind: 'string', value: 'auto' }]),
            reason: 'Manual GPU targeting conflicts with automatic assignment.',
            preferred_path: { segments: ['gpu', 'assignment'] }
          }
        ]
      }
    })
    const settings = [assignmentSetting, deviceSetting, pinnedOnlySetting] as const
    const values = { 'gpu.assignment': 'auto' }

    expect(evaluateSettingControlState(deviceSetting, settings, values)).toMatchObject({
      enabled: false,
      source: 'dependency',
      reason: 'Pinned device selection is unavailable while assignment is auto.',
      note: 'Switch assignment to pinned to edit this field.',
      write_policy: 'preserve_existing'
    })
    expect(evaluateSettingControlState(pinnedOnlySetting, settings, values)).toMatchObject({
      enabled: false,
      source: 'conflict',
      reason: 'Manual GPU targeting conflicts with automatic assignment.',
      note: 'Prefer gpu.assignment',
      write_policy: 'preserve_existing'
    })
  })

  it('resolves plugin-local control paths relative to the current plugin settings namespace', () => {
    const pluginMode = choiceSetting({
      id: 'plugin.blackboard.settings.mode',
      canonicalPath: 'plugin.blackboard.settings.mode',
      value: 'remote',
      options: ['local', 'remote']
    })
    const pluginEndpoint = textSetting({
      id: 'plugin.blackboard.settings.endpoint',
      canonicalPath: 'plugin.blackboard.settings.endpoint',
      controlBehavior: {
        enable_when: [condition('mode', 'equals', [{ kind: 'string', value: 'remote' }])]
      }
    })
    const pluginLocalCache = textSetting({
      id: 'plugin.blackboard.settings.local_cache_path',
      canonicalPath: 'plugin.blackboard.settings.local_cache_path',
      controlBehavior: {
        conflicts: [
          {
            group: 'plugin-mode',
            condition: condition('mode', 'equals', [{ kind: 'string', value: 'remote' }]),
            reason: 'Local cache path conflicts with remote mode.',
            preferred_path: { segments: ['mode'] }
          }
        ]
      }
    })
    const settings = [pluginMode, pluginEndpoint, pluginLocalCache] as const
    const values = typedValues({ 'plugin.blackboard.settings.mode': 'remote' })

    expect(evaluateSettingControlState(pluginEndpoint, settings, values)).toMatchObject({
      enabled: true,
      source: 'static'
    })
    expect(evaluateSettingControlState(pluginLocalCache, settings, values)).toMatchObject({
      enabled: false,
      source: 'conflict',
      reason: 'Local cache path conflicts with remote mode.',
      note: 'Prefer plugin.blackboard.settings.mode'
    })
  })

  it('resolves plugin-local control paths for dotted plugin names via the settings namespace marker', () => {
    const pluginMode = choiceSetting({
      id: 'plugin.com.example.tool.settings.mode',
      canonicalPath: 'plugin.com.example.tool.settings.mode',
      value: 'remote',
      options: ['local', 'remote']
    })
    const pluginEndpoint = textSetting({
      id: 'plugin.com.example.tool.settings.endpoint',
      canonicalPath: 'plugin.com.example.tool.settings.endpoint',
      controlBehavior: {
        enable_when: [condition('mode', 'equals', [{ kind: 'string', value: 'remote' }])]
      }
    })
    const pluginLocalCache = textSetting({
      id: 'plugin.com.example.tool.settings.local_cache_path',
      canonicalPath: 'plugin.com.example.tool.settings.local_cache_path',
      controlBehavior: {
        conflicts: [
          {
            group: 'plugin-mode',
            condition: condition('mode', 'equals', [{ kind: 'string', value: 'remote' }]),
            reason: 'Local cache path conflicts with remote mode.',
            preferred_path: { segments: ['mode'] }
          }
        ]
      }
    })
    const settings = [pluginMode, pluginEndpoint, pluginLocalCache] as const
    const values = typedValues({ 'plugin.com.example.tool.settings.mode': 'remote' })

    expect(evaluateSettingControlState(pluginEndpoint, settings, values)).toMatchObject({
      enabled: true,
      source: 'static'
    })
    expect(evaluateSettingControlState(pluginLocalCache, settings, values)).toMatchObject({
      enabled: false,
      source: 'conflict',
      reason: 'Local cache path conflicts with remote mode.',
      note: 'Prefer plugin.com.example.tool.settings.mode'
    })
  })
}
