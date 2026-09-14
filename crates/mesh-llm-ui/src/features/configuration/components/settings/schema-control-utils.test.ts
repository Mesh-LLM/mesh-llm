import { describe, expect, it } from 'vitest'
import type { ConfigurationRuntimeControlStateEntry } from '@/features/app-tabs/types'
import { choiceSetting } from '@/features/configuration/lib/settings-utils.test-helpers'
import { resolvedChoiceOptions } from './schema-control-utils'

const nativeBackendControlState: ConfigurationRuntimeControlStateEntry = {
  enabled: true,
  source: 'runtime',
  write_policy: 'preserve_existing',
  options: [
    { value: { kind: 'string', value: 'cpu' }, label: 'CPU', disabled: false, source: 'runtime_native_backends' },
    { value: { kind: 'string', value: 'metal' }, label: 'Metal', disabled: false, source: 'runtime_native_backends' }
  ]
}

function selectionSetting(value: string) {
  return choiceSetting({
    id: 'runtime.native_runtime.selection',
    canonicalPath: 'runtime.native_runtime.selection',
    value,
    options: ['', 'recommended', 'cpu', 'metal'],
    controlState: nativeBackendControlState
  })
}

describe('resolvedChoiceOptions', () => {
  it('keeps the runtime backend list when the stored value already matches an option', () => {
    const options = resolvedChoiceOptions(selectionSetting('metal'), 'metal')
    expect(options.map((option) => option.value)).toEqual(['', 'recommended', 'cpu', 'metal'])
  })

  it('surfaces an unmatched exact: pin as a synthetic option so it is not silently dropped', () => {
    const pin = 'exact:meshllm-native-runtime-linux-x86_64-cuda12'
    const options = resolvedChoiceOptions(selectionSetting(pin), pin)

    expect(options.map((option) => option.value)).toEqual(['', 'recommended', 'cpu', 'metal', pin])
    expect(options.at(-1)).toEqual({ value: pin, label: pin, description: 'Set in config file' })
  })
})
