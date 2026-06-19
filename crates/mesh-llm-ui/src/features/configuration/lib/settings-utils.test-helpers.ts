import type {
  ConfigurationControlCondition,
  ConfigurationDefaultsControl,
  ConfigurationDefaultsSetting,
  ConfigurationDefaultsValues,
  ConfigurationSettingControlBehavior,
  ConfigurationSettingValueSchema
} from '@/features/app-tabs/types'

type TestSettingInput = {
  id: string
  canonicalPath?: string
  control: ConfigurationDefaultsControl
  valueSchema?: ConfigurationSettingValueSchema
  controlBehavior?: ConfigurationSettingControlBehavior
  dependsOn?: ConfigurationDefaultsSetting['dependsOn']
  controlState?: ConfigurationDefaultsSetting['controlState']
}

function createSetting(input: TestSettingInput): ConfigurationDefaultsSetting {
  return {
    id: input.id,
    categoryId: 'runtime',
    icon: 'cog',
    label: input.id,
    description: input.id,
    inheritedLabel: input.id,
    canonicalPath: input.canonicalPath ?? input.id,
    control: input.control,
    valueSchema: input.valueSchema,
    controlBehavior: input.controlBehavior,
    dependsOn: input.dependsOn,
    controlState: input.controlState
  }
}

export function choiceSetting(
  input: Omit<TestSettingInput, 'control'> & { value?: string; options?: readonly string[] }
) {
  const options = input.options ?? ['off', 'on']
  return createSetting({
    ...input,
    control: {
      kind: 'choice',
      name: input.id,
      value: input.value ?? options[0],
      options: options.map((value) => ({ value, label: value }))
    }
  })
}

export function textSetting(input: Omit<TestSettingInput, 'control'> & { value?: string }) {
  return createSetting({
    ...input,
    control: { kind: 'text', name: input.id, value: input.value ?? '', placeholder: 'value' }
  })
}

export function rangeSetting(input: Omit<TestSettingInput, 'control'> & { value?: string }) {
  return createSetting({
    ...input,
    control: { kind: 'range', name: input.id, value: input.value ?? '0', min: 0, max: 64, step: 1 }
  })
}

export function condition(
  path: string,
  operator: ConfigurationControlCondition['operator'],
  values?: ConfigurationControlCondition['values']
) {
  return { path: { segments: path.split('.') }, operator, values } satisfies ConfigurationControlCondition
}

export function typedValues(values: ConfigurationDefaultsValues) {
  return values
}
