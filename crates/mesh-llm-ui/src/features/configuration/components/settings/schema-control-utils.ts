import type {
  ConfigurationControlAvailabilitySource,
  ConfigurationControlCondition,
  ConfigurationControlConditionValue,
  ConfigurationDefaultsChoice,
  ConfigurationDefaultsSetting,
  ConfigurationDisabledWritePolicy,
  ConfigurationRuntimeControlOption,
  ConfigurationSettingValueSchema
} from '@/features/app-tabs/types'

export type SchemaSettingControlProps = {
  readonly ariaDescribedBy?: string
  readonly disabled?: boolean
  readonly invalid?: boolean
  readonly onChange: (value: string) => void
  readonly setting: ConfigurationDefaultsSetting
  readonly value: string
}

export type ResolvedChoiceOption = ConfigurationDefaultsChoice & {
  readonly disabled?: boolean
}

export type SettingAvailabilityState = {
  readonly disabled: boolean
  readonly note?: string
  readonly reason?: string
  readonly source?: ConfigurationControlAvailabilitySource
  readonly writePolicy?: ConfigurationDisabledWritePolicy
}

export type NumericControlMetadata = {
  readonly max?: number
  readonly min?: number
  readonly step?: number
  readonly unit?: string
}

function normalizedChoiceValue(value: string) {
  if (value === 'true') return 'on'
  if (value === 'false') return 'off'
  return value
}

function controlConditionValueString(
  value: ConfigurationControlConditionValue | ConfigurationRuntimeControlOption['value']
) {
  switch (value.kind) {
    case 'bool':
      return value.value ? 'on' : 'off'
    case 'integer':
    case 'float':
      return String(value.value)
    case 'string':
      return value.value
  }
}

function enumValues(schema: ConfigurationSettingValueSchema | undefined): string[] {
  if (!schema) return []
  if (schema.kind === 'enum') return schema.values
  if (schema.kind === 'one_of') return schema.variants.flatMap((variant) => enumValues(variant))
  return []
}

function booleanValues(schema: ConfigurationSettingValueSchema | undefined): string[] {
  return hasSchemaKind(schema, 'boolean') ? ['on', 'off'] : []
}

function pathSegmentsLabel(segments: readonly unknown[]) {
  const labels = segments
    .map((segment) => {
      if (typeof segment === 'string' || typeof segment === 'number') return String(segment)
      return null
    })
    .filter((segment): segment is string => segment !== null)
  return labels.length > 0 ? labels.join('.') : 'related setting'
}

function describeConditionValues(values: readonly ConfigurationControlConditionValue[]) {
  return values.map((value) => controlConditionValueString(value)).join(' or ')
}

export function hasSchemaKind(
  schema: ConfigurationSettingValueSchema | undefined,
  kind: ConfigurationSettingValueSchema['kind']
): boolean {
  if (!schema) return false
  if (schema.kind === kind) return true
  if (schema.kind === 'one_of') return schema.variants.some((variant) => hasSchemaKind(variant, kind))
  return false
}

export function effectiveRendererId(setting: ConfigurationDefaultsSetting) {
  if (setting.rendererId) return setting.rendererId
  if (setting.id === 'parallel-slots') return 'slot-meter'
  if (setting.id === 'kv-cache') return 'kv-cache-policy'
  if (setting.id === 'ctx-size') return 'context-slider'
  return undefined
}

export function isBooleanToggleChoice(setting: ConfigurationDefaultsSetting) {
  return (
    setting.control.kind === 'choice' &&
    setting.control.presentation === 'toggle' &&
    setting.control.options.length === 2 &&
    setting.control.options.every((option) => option.value === 'on' || option.value === 'off')
  )
}

export function textFormatForSetting(setting: ConfigurationDefaultsSetting) {
  const controlTextFormat = setting.controlBehavior?.text_format
  if (controlTextFormat) return controlTextFormat
  if (hasSchemaKind(setting.valueSchema, 'path')) return 'path'
  if (hasSchemaKind(setting.valueSchema, 'url')) return 'url'
  if (hasSchemaKind(setting.valueSchema, 'socket_addr')) return 'socket_addr'
  return 'plain'
}

export function numericMetadataForSetting(setting: ConfigurationDefaultsSetting): NumericControlMetadata {
  if (setting.control.kind === 'range') {
    return {
      min: setting.control.min,
      max: setting.control.max,
      step: setting.control.step,
      unit: setting.control.unit ?? setting.controlBehavior?.numeric?.unit
    }
  }

  const numeric = setting.controlBehavior?.numeric
  return {
    min: numeric?.min,
    max: numeric?.max,
    step: numeric?.step,
    unit: numeric?.unit
  }
}

export function acceptedValuesForSetting(setting: ConfigurationDefaultsSetting) {
  const fromSchema = [
    ...enumValues(setting.valueSchema).map(normalizedChoiceValue),
    ...booleanValues(setting.valueSchema)
  ]
  const fromControl = setting.control.kind === 'choice' ? setting.control.options.map((option) => option.value) : []
  return Array.from(new Set([...fromSchema, ...fromControl]))
}

export function resolvedChoiceOptions(setting: ConfigurationDefaultsSetting): readonly ResolvedChoiceOption[] {
  const fromControl: ResolvedChoiceOption[] =
    setting.control.kind === 'choice'
      ? setting.control.options.map((option) => ({
          value: option.value,
          label: option.label,
          description: option.description
        }))
      : acceptedValuesForSetting(setting).map((value) => ({ value, label: value, description: undefined }))

  const runtimeOptions = setting.controlState?.options ?? []
  if (runtimeOptions.length === 0) return fromControl

  const runtimeByValue = new Map(
    runtimeOptions.map((option) => [normalizedChoiceValue(controlConditionValueString(option.value)), option] as const)
  )
  const mergedOptions: ResolvedChoiceOption[] = []

  for (const option of fromControl) {
    const runtimeOption = runtimeByValue.get(option.value)
    mergedOptions.push({
      value: option.value,
      label: runtimeOption?.label ?? option.label,
      description: runtimeOption?.note ?? runtimeOption?.reason ?? option.description,
      disabled: runtimeOption?.disabled
    })
    runtimeByValue.delete(option.value)
  }

  for (const runtimeOption of runtimeByValue.values()) {
    const value = normalizedChoiceValue(controlConditionValueString(runtimeOption.value))
    mergedOptions.push({
      value,
      label: runtimeOption.label ?? value,
      description: runtimeOption.note ?? runtimeOption.reason,
      disabled: runtimeOption.disabled
    })
  }

  return mergedOptions
}

export function getSettingAvailability(setting: ConfigurationDefaultsSetting): SettingAvailabilityState {
  if (setting.controlState) {
    return {
      disabled: !setting.controlState.enabled,
      reason: setting.controlState.reason,
      note: setting.controlState.note,
      source: setting.controlState.source,
      writePolicy: setting.controlState.write_policy
    }
  }

  if (setting.controlBehavior?.availability) {
    return {
      disabled: !setting.controlBehavior.availability.enabled,
      reason: setting.controlBehavior.availability.reason,
      note: setting.controlBehavior.availability.note,
      source: setting.controlBehavior.availability.source,
      writePolicy: setting.controlBehavior.write_policy
    }
  }

  return {
    disabled: false,
    writePolicy: setting.controlBehavior?.write_policy
  }
}

export function describeControlCondition(condition: ConfigurationControlCondition) {
  const pathLabel = pathSegmentsLabel(condition.path.segments)
  const values = condition.values ?? []

  switch (condition.operator) {
    case 'equals':
      return values.length > 0 ? `${pathLabel} = ${describeConditionValues(values)}` : pathLabel
    case 'not_equals':
      return values.length > 0 ? `${pathLabel} ≠ ${describeConditionValues(values)}` : pathLabel
    case 'in':
      return values.length > 0 ? `${pathLabel} in ${describeConditionValues(values)}` : pathLabel
    case 'not_in':
      return values.length > 0 ? `${pathLabel} not in ${describeConditionValues(values)}` : pathLabel
    case 'present':
      return `${pathLabel} is present`
    case 'absent':
      return `${pathLabel} is absent`
    case 'truthy':
      return `${pathLabel} is enabled`
    case 'falsy':
      return `${pathLabel} is disabled`
    case 'range':
      return values.length > 0 ? `${pathLabel} within ${describeConditionValues(values)}` : `${pathLabel} within range`
  }
}
