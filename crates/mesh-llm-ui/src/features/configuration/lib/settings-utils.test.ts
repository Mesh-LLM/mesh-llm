import { describe, expect, it } from 'vitest'
import {
  evaluateSettingControlState,
  getSettingDependencyStatus,
  getSettingDisabledReason,
  getSettingValue,
  isSettingDisabled
} from '@/features/configuration/lib/settings-utils'
import type { ConfigurationDefaultsValues } from '@/features/app-tabs/types'
import {
  choiceSetting,
  condition,
  rangeSetting,
  textSetting,
  typedValues
} from '@/features/configuration/lib/settings-utils.test-helpers'
import { defineSettingsControlRuleTests } from '@/features/configuration/lib/settings-utils.control-rules-test-cases'

describe('settings-utils', () => {
  it('preserves legacy dependsOn reasons for direct and multi-option dependencies', () => {
    const modeSetting = choiceSetting({ id: 'mirostat-mode', value: 'disabled', options: ['disabled', '1', '2'] })
    const dependentSetting = rangeSetting({
      id: 'mirostat-entropy',
      value: '5',
      dependsOn: { settingId: 'mirostat-mode', condition: (value: string) => value !== 'disabled' }
    })
    const settings = [modeSetting, dependentSetting] as const
    const values: ConfigurationDefaultsValues = {}

    expect(getSettingValue(modeSetting, { 'mirostat-mode': '1' })).toBe('1')
    expect(isSettingDisabled(dependentSetting, settings, values)).toBe(true)
    expect(getSettingDependencyStatus(dependentSetting, settings, values)).toEqual({
      disabled: true,
      reason: 'Requires mirostat-mode = 1 or 2'
    })
    expect(getSettingDisabledReason(dependentSetting, settings, values)).toBe('Requires mirostat-mode = 1 or 2')
    expect(isSettingDisabled(dependentSetting, settings, { 'mirostat-mode': '2' })).toBe(false)
  })

  it.each([
    {
      label: 'equals',
      parent: choiceSetting({ id: 'rule.parent.equals', value: 'draft', options: ['draft', 'ngram'] }),
      dependent: textSetting({
        id: 'rule.child.equals',
        controlBehavior: {
          enable_when: [condition('rule.parent.equals', 'equals', [{ kind: 'string', value: 'draft' }])]
        }
      }),
      values: typedValues({ 'rule.parent.equals': 'draft' }),
      expectedEnabled: true
    },
    {
      label: 'not_equals',
      parent: choiceSetting({ id: 'rule.parent.not_equals', value: 'draft', options: ['draft', 'ngram'] }),
      dependent: textSetting({
        id: 'rule.child.not_equals',
        controlBehavior: {
          enable_when: [condition('rule.parent.not_equals', 'not_equals', [{ kind: 'string', value: 'ngram' }])]
        }
      }),
      values: typedValues({ 'rule.parent.not_equals': 'draft' }),
      expectedEnabled: true
    },
    {
      label: 'in',
      parent: choiceSetting({ id: 'rule.parent.in', value: '2', options: ['0', '1', '2'] }),
      dependent: textSetting({
        id: 'rule.child.in',
        controlBehavior: {
          enable_when: [
            condition('rule.parent.in', 'in', [
              { kind: 'string', value: '1' },
              { kind: 'string', value: '2' }
            ])
          ]
        }
      }),
      values: typedValues({ 'rule.parent.in': '2' }),
      expectedEnabled: true
    },
    {
      label: 'not_in',
      parent: choiceSetting({ id: 'rule.parent.not_in', value: 'draft', options: ['draft', 'ngram'] }),
      dependent: textSetting({
        id: 'rule.child.not_in',
        controlBehavior: {
          enable_when: [condition('rule.parent.not_in', 'not_in', [{ kind: 'string', value: 'ngram' }])]
        }
      }),
      values: typedValues({ 'rule.parent.not_in': 'draft' }),
      expectedEnabled: true
    },
    {
      label: 'present',
      parent: textSetting({ id: 'rule.parent.present', value: '' }),
      dependent: textSetting({
        id: 'rule.child.present',
        controlBehavior: { enable_when: [condition('rule.parent.present', 'present')] }
      }),
      values: typedValues({ 'rule.parent.present': 'configured' }),
      expectedEnabled: true
    },
    {
      label: 'absent',
      parent: textSetting({ id: 'rule.parent.absent', value: 'configured' }),
      dependent: textSetting({
        id: 'rule.child.absent',
        controlBehavior: { enable_when: [condition('rule.parent.absent', 'absent')] }
      }),
      values: typedValues({ 'rule.parent.absent': '' }),
      expectedEnabled: true
    },
    {
      label: 'truthy',
      parent: choiceSetting({
        id: 'rule.parent.truthy',
        value: 'off',
        options: ['off', 'on'],
        valueSchema: { kind: 'boolean' }
      }),
      dependent: textSetting({
        id: 'rule.child.truthy',
        controlBehavior: { enable_when: [condition('rule.parent.truthy', 'truthy')] }
      }),
      values: typedValues({ 'rule.parent.truthy': 'on' }),
      expectedEnabled: true
    },
    {
      label: 'falsy',
      parent: choiceSetting({
        id: 'rule.parent.falsy',
        value: 'on',
        options: ['off', 'on'],
        valueSchema: { kind: 'boolean' }
      }),
      dependent: textSetting({
        id: 'rule.child.falsy',
        controlBehavior: { enable_when: [condition('rule.parent.falsy', 'falsy')] }
      }),
      values: typedValues({ 'rule.parent.falsy': 'off' }),
      expectedEnabled: true
    },
    {
      label: 'range',
      parent: rangeSetting({ id: 'rule.parent.range', value: '4', valueSchema: { kind: 'integer' } }),
      dependent: textSetting({
        id: 'rule.child.range',
        controlBehavior: {
          enable_when: [
            condition('rule.parent.range', 'range', [
              { kind: 'integer', value: 2 },
              { kind: 'integer', value: 8 }
            ])
          ]
        }
      }),
      values: typedValues({ 'rule.parent.range': '4' }),
      expectedEnabled: true
    }
  ])('supports the $label operator', ({ parent, dependent, values, expectedEnabled }) => {
    const evaluation = evaluateSettingControlState(dependent, [parent, dependent], values)

    expect(evaluation.enabled).toBe(expectedEnabled)
  })

  it('uses static and runtime availability before dependency rules', () => {
    const modeSetting = choiceSetting({ id: 'defaults.speculative.mode', value: 'draft', options: ['draft', 'ngram'] })
    const runtimeDisabled = textSetting({
      id: 'defaults.hardware.device',
      controlBehavior: {
        enable_when: [condition('defaults.speculative.mode', 'equals', [{ kind: 'string', value: 'draft' }])]
      },
      controlState: {
        enabled: false,
        reason: 'No compatible GPU was detected.',
        note: 'The current value will be preserved but cannot be edited.',
        source: 'runtime',
        write_policy: 'preserve_existing'
      }
    })
    const staticallyRejected = textSetting({
      id: 'defaults.rejected.setting',
      controlBehavior: {
        availability: { enabled: false, reason: 'Rejected by schema.', source: 'static' },
        write_policy: 'reject_when_disabled'
      }
    })

    expect(
      evaluateSettingControlState(runtimeDisabled, [modeSetting, runtimeDisabled], {
        'defaults.speculative.mode': 'draft'
      })
    ).toMatchObject({
      enabled: false,
      reason: 'No compatible GPU was detected.',
      source: 'runtime',
      write_policy: 'preserve_existing'
    })
    expect(evaluateSettingControlState(staticallyRejected, [staticallyRejected], {})).toMatchObject({
      enabled: false,
      reason: 'Rejected by schema.',
      source: 'static',
      write_policy: 'reject_when_disabled'
    })
  })

  defineSettingsControlRuleTests()

  it('enables speculative draft fields and disables ngram fields from the draft values map', () => {
    const modeSetting = choiceSetting({ id: 'defaults.speculative.mode', value: 'ngram', options: ['draft', 'ngram'] })
    const draftField = rangeSetting({
      id: 'defaults.speculative.draft_max_tokens',
      value: '16',
      valueSchema: { kind: 'integer' },
      controlBehavior: {
        enable_when: [condition('defaults.speculative.mode', 'equals', [{ kind: 'string', value: 'draft' }])]
      }
    })
    const ngramField = rangeSetting({
      id: 'defaults.speculative.ngram_max',
      value: '8',
      valueSchema: { kind: 'integer' },
      controlBehavior: {
        enable_when: [condition('defaults.speculative.mode', 'equals', [{ kind: 'string', value: 'ngram' }])]
      }
    })
    const settings = [modeSetting, draftField, ngramField] as const
    const values = { 'defaults.speculative.mode': 'draft' } satisfies ConfigurationDefaultsValues

    expect(evaluateSettingControlState(draftField, settings, values)).toMatchObject({
      enabled: true,
      source: 'static',
      write_policy: 'preserve_existing'
    })
    expect(evaluateSettingControlState(ngramField, settings, values)).toMatchObject({
      enabled: false,
      source: 'dependency',
      reason: 'Requires defaults.speculative.mode = ngram',
      write_policy: 'omit_when_disabled'
    })
  })
})
