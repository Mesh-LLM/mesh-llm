import type {
  ConfigurationDefaultsChoice,
  ConfigurationDefaultsControl,
  ConfigurationRuntimeControlOption,
  ConfigurationRuntimeControlStateEntry,
  ConfigurationSettingControlBehavior,
  ConfigurationSettingValueSchema
} from '@/features/app-tabs/types'

type SchemaControlConstraint =
  | { readonly kind: 'non_empty' }
  | { readonly kind: 'positive' }
  | { readonly kind: 'range'; readonly min?: string; readonly max?: string }
  | { readonly kind: 'requires'; readonly path: unknown }
  | { readonly kind: 'allowed_values'; readonly values: readonly string[] }
  | { readonly kind: 'allowed_pattern'; readonly pattern: string }

type SchemaControlPresentation = {
  readonly placeholder?: string
  readonly unit?: string
  readonly control_hint?: string
}

export type SchemaControlFactoryEntry = {
  readonly canonical_path: string
  readonly value_schema: ConfigurationSettingValueSchema
  readonly constraints?: readonly SchemaControlConstraint[]
  readonly presentation?: SchemaControlPresentation
  readonly control_behavior?: ConfigurationSettingControlBehavior
}

type CreateSchemaControlInput = {
  readonly entry: SchemaControlFactoryEntry
  readonly name: string
  readonly bespoke?: ConfigurationDefaultsControl
  readonly runtimeControlState?: ConfigurationRuntimeControlStateEntry
}

function enumValues(schema: ConfigurationSettingValueSchema): readonly string[] {
  if (schema.kind === 'enum') return schema.values
  if (schema.kind === 'one_of') return schema.variants.flatMap(enumValues)
  return []
}

function hasSchemaKind(
  schema: ConfigurationSettingValueSchema,
  kind: ConfigurationSettingValueSchema['kind']
): boolean {
  if (schema.kind === kind) return true
  if (schema.kind === 'one_of') return schema.variants.some((variant) => hasSchemaKind(variant, kind))
  return false
}

function normalizedChoiceValue(value: string): string {
  if (value === 'true') return 'on'
  if (value === 'false') return 'off'
  return value
}

function choiceOption(value: string): ConfigurationDefaultsChoice {
  const normalized = normalizedChoiceValue(value)
  return { value: normalized, label: normalized }
}

function controlConditionValueString(value: ConfigurationRuntimeControlOption['value']): string {
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

function runtimeChoiceOptions(
  controlState?: ConfigurationRuntimeControlStateEntry
): readonly ConfigurationDefaultsChoice[] {
  return (controlState?.options ?? []).map((option) => {
    const normalizedValue = normalizedChoiceValue(controlConditionValueString(option.value))
    return {
      value: normalizedValue,
      label: option.label ?? normalizedValue,
      description: option.note ?? option.reason
    }
  })
}

function schemaRange(entry: SchemaControlFactoryEntry): { readonly min?: number; readonly max?: number } {
  if (entry.control_behavior?.numeric) {
    return {
      min: entry.control_behavior.numeric.min,
      max: entry.control_behavior.numeric.max
    }
  }

  const range = entry.constraints?.find(
    (constraint): constraint is Extract<SchemaControlConstraint, { kind: 'range' }> => constraint.kind === 'range'
  )
  const min = range?.min == null ? undefined : Number(range.min)
  const max = range?.max == null ? undefined : Number(range.max)
  return {
    min: Number.isFinite(min) ? min : undefined,
    max: Number.isFinite(max) ? max : undefined
  }
}

function choicePresentation(
  controlHint: SchemaControlPresentation['control_hint'],
  options: readonly ConfigurationDefaultsChoice[]
): Extract<ConfigurationDefaultsControl, { kind: 'choice' }>['presentation'] {
  const hasAuto = options.some((option) => option.value === 'auto')
  const hasOnOff = options.some((option) => option.value === 'on') && options.some((option) => option.value === 'off')

  if (controlHint === 'toggle') return 'toggle'
  if (controlHint === 'select') return 'select'
  if (options.length <= 4) return hasOnOff && !hasAuto ? 'toggle' : 'segmented'
  return 'select'
}

function choiceDefaultValue(options: readonly ConfigurationDefaultsChoice[]): string {
  const hasAuto = options.some((option) => option.value === 'auto')
  const hasOnOff = options.some((option) => option.value === 'on') && options.some((option) => option.value === 'off')

  if (hasAuto) return 'auto'
  if (hasOnOff) return 'off'
  return options[0]?.value ?? ''
}

function runtimeChoicePlaceholder(entry: SchemaControlFactoryEntry): ConfigurationDefaultsChoice {
  if (entry.control_behavior?.options_source === 'runtime_gpus') return { value: '', label: 'Select GPU' }
  if (entry.control_behavior?.options_source === 'runtime_native_backends')
    return { value: '', label: 'Select backend' }
  if (entry.control_behavior?.options_source === 'runtime_local_models') return { value: '', label: 'Select model' }
  if (entry.control_behavior?.options_source === 'runtime_installed_plugins')
    return { value: '', label: 'Select plugin' }
  if (entry.control_behavior?.options_source === 'runtime_mesh_peers') return { value: '', label: 'Select peer' }
  return { value: '', label: 'Select value' }
}

function rangeControl(entry: SchemaControlFactoryEntry, name: string): ConfigurationDefaultsControl | undefined {
  const bounds = schemaRange(entry)
  if (bounds.min === undefined || bounds.max === undefined) return undefined

  return {
    kind: 'range',
    name,
    value: String(bounds.min),
    min: bounds.min,
    max: bounds.max,
    step: entry.control_behavior?.numeric?.step ?? (hasSchemaKind(entry.value_schema, 'float') ? 0.01 : 1),
    unit: entry.presentation?.unit ?? entry.control_behavior?.numeric?.unit
  }
}

function textControl(entry: SchemaControlFactoryEntry, name: string): ConfigurationDefaultsControl {
  return {
    kind: 'text',
    name,
    value: '',
    placeholder: entry.presentation?.placeholder ?? (entry.value_schema.kind === 'object' ? 'JSON object' : undefined)
  }
}

export function createSchemaControl(input: CreateSchemaControlInput): ConfigurationDefaultsControl {
  const { entry, name, bespoke, runtimeControlState } = input
  if (bespoke) return bespoke

  const runtimeOptions = runtimeChoiceOptions(runtimeControlState)
  if (runtimeOptions.length > 0) {
    return {
      kind: 'choice',
      name,
      value: '',
      presentation: 'select',
      options: [runtimeChoicePlaceholder(entry), ...runtimeOptions]
    }
  }

  const values = Array.from(new Set(enumValues(entry.value_schema).map(normalizedChoiceValue)))
  const boolLike = hasSchemaKind(entry.value_schema, 'boolean')
  const numericLike = hasSchemaKind(entry.value_schema, 'integer') || hasSchemaKind(entry.value_schema, 'float')
  const controlHint = entry.presentation?.control_hint

  if (values.length > 0 && numericLike && !boolLike) {
    return {
      kind: 'text',
      name,
      value: values[0] ?? '',
      placeholder: `${values.join(' or ')} or number`
    }
  }

  if (values.length > 0 || boolLike) {
    const options = values.length > 0 ? values.map(choiceOption) : ['on', 'off'].map(choiceOption)
    return {
      kind: 'choice',
      name,
      value: choiceDefaultValue(options),
      presentation: choicePresentation(controlHint, options),
      options
    }
  }

  if (entry.value_schema.kind === 'integer' || entry.value_schema.kind === 'float' || controlHint === 'range') {
    const control = rangeControl(entry, name)
    if (control) return control
  }

  return textControl(entry, name)
}
