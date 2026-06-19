import { SchemaArrayControl } from '@/features/configuration/components/settings/SchemaArrayControl'
import { SchemaBooleanControl } from '@/features/configuration/components/settings/SchemaBooleanControl'
import { SchemaChoiceControl } from '@/features/configuration/components/settings/SchemaChoiceControl'
import { SchemaNumberControl } from '@/features/configuration/components/settings/SchemaNumberControl'
import { SchemaObjectControl } from '@/features/configuration/components/settings/SchemaObjectControl'
import { SchemaPathControl } from '@/features/configuration/components/settings/SchemaPathControl'
import { SchemaRuntimeChoiceControl } from '@/features/configuration/components/settings/SchemaRuntimeChoiceControl'
import { SchemaUrlControl } from '@/features/configuration/components/settings/SchemaUrlControl'
import {
  acceptedValuesForSetting,
  describeControlCondition,
  hasSchemaKind,
  numericMetadataForSetting,
  textFormatForSetting,
  type NumericControlMetadata,
  type SchemaSettingControlProps,
  type SettingAvailabilityState
} from '@/features/configuration/components/settings/schema-control-utils'
import { cn } from '@/lib/cn'

type ConfigurationDefaultsControlProps = SchemaSettingControlProps & {
  readonly availability?: SettingAvailabilityState
}

type SchemaControlKind =
  | 'array'
  | 'boolean'
  | 'choice'
  | 'metric'
  | 'number'
  | 'object'
  | 'path'
  | 'runtime-choice'
  | 'text'
  | 'url'

type ControlDetailBuckets = {
  readonly disabledDetails: readonly string[]
  readonly visibleDetails: readonly string[]
}

type NumericControlPresentation = {
  readonly lowerBound?: {
    readonly inclusive: boolean
    readonly value: string
  }
  readonly upperBound?: {
    readonly inclusive: boolean
    readonly value: string
  }
}

function formatMetricValue(setting: ConfigurationDefaultsControlProps['setting']) {
  if (setting.control.kind === 'metric' && setting.control.unit)
    return `${setting.control.value} ${setting.control.unit}`
  if (setting.control.kind === 'metric') return setting.control.value
  return ''
}

function openStringControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  onChange,
  setting,
  value
}: SchemaSettingControlProps) {
  return (
    <input
      aria-describedby={ariaDescribedBy}
      aria-invalid={invalid ? 'true' : undefined}
      aria-label={setting.label}
      className={cn(
        'ui-control h-[32px] w-full min-w-[280px] rounded-[var(--radius)] border bg-surface px-2.5 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
        disabled && 'cursor-not-allowed opacity-60',
        invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
      )}
      disabled={disabled}
      name={'name' in setting.control ? setting.control.name : setting.id}
      onChange={(event) => onChange(event.currentTarget.value)}
      placeholder={'placeholder' in setting.control ? setting.control.placeholder : undefined}
      value={value}
    />
  )
}

// eslint-disable-next-line react-refresh/only-export-components -- utility tightly coupled to this component
export function configurationControlDetailBuckets(
  setting: ConfigurationDefaultsControlProps['setting'],
  value: string,
  availability: SettingAvailabilityState
): ControlDetailBuckets {
  const visibleDetails: string[] = []
  const disabledDetails: string[] = []

  const pushUnique = (target: string[], detail: string) => {
    const normalizedDetail = detail.trim()
    if (normalizedDetail.length === 0) return
    if (target.some((existingDetail) => existingDetail.trim() === normalizedDetail)) return
    target.push(detail)
  }

  const textFormat = textFormatForSetting(setting)
  if (textFormat === 'path') {
    pushUnique(visibleDetails, 'Path hint: enter a local filesystem path. No file picker is available here.')
  }
  if (textFormat === 'url') pushUnique(visibleDetails, 'URL hint: enter a full URL including protocol.')
  if (hasSchemaKind(setting.valueSchema, 'array')) {
    pushUnique(visibleDetails, 'List input: enter one item per line. Saved as a TOML string array.')
  }
  if (hasSchemaKind(setting.valueSchema, 'object')) {
    pushUnique(visibleDetails, 'Object input: enter a JSON object.')
    if (value.trim().length > 0) {
      try {
        const parsed: unknown = JSON.parse(value)
        if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
          pushUnique(visibleDetails, 'Object input expects valid JSON object syntax.')
        }
      } catch (error) {
        if (!(error instanceof SyntaxError)) throw error
        pushUnique(visibleDetails, 'Object input expects valid JSON object syntax.')
      }
    }
  }

  for (const condition of setting.controlBehavior?.enable_when ?? []) {
    pushUnique(disabledDetails, `Requires ${describeControlCondition(condition)}`)
  }
  for (const condition of setting.controlBehavior?.disable_when ?? []) {
    pushUnique(disabledDetails, condition.reason)
  }
  if (availability.reason) pushUnique(disabledDetails, availability.reason)
  if (availability.note) pushUnique(disabledDetails, availability.note)

  for (const conflict of setting.controlBehavior?.conflicts ?? []) {
    pushUnique(visibleDetails, `Conflict: ${conflict.reason}`)
  }

  return {
    disabledDetails,
    visibleDetails
  }
}

function formatSliderBoundLabel(metadata: NumericControlMetadata, label: 'Min' | 'Max', value: number | undefined) {
  if (value === undefined) return undefined

  const segments = [`${label} ${value}`]
  if (metadata.unit) segments.push(metadata.unit)
  return segments.join(' ')
}

function numericPresentation(setting: ConfigurationDefaultsControlProps['setting']): NumericControlPresentation {
  const numeric = numericMetadataForSetting(setting)

  return {
    lowerBound:
      numeric.min !== undefined
        ? {
            inclusive: true,
            value: formatSliderBoundLabel(numeric, 'Min', numeric.min) ?? String(numeric.min)
          }
        : undefined,
    upperBound:
      numeric.max !== undefined
        ? {
            inclusive: true,
            value: formatSliderBoundLabel(numeric, 'Max', numeric.max) ?? String(numeric.max)
          }
        : undefined
  }
}

function controlKind(setting: ConfigurationDefaultsControlProps['setting']): SchemaControlKind {
  if (setting.control.kind === 'metric') return 'metric'
  if (hasSchemaKind(setting.valueSchema, 'array')) return 'array'
  if (hasSchemaKind(setting.valueSchema, 'object')) return 'object'

  const textFormat = textFormatForSetting(setting)
  if (textFormat === 'path') return 'path'
  if (textFormat === 'url') return 'url'

  if (
    setting.controlState?.options?.length ||
    (setting.controlBehavior?.options_source?.startsWith('runtime_') ?? false)
  ) {
    return 'runtime-choice'
  }

  const acceptedValues = acceptedValuesForSetting(setting)
  const booleanLike =
    hasSchemaKind(setting.valueSchema, 'boolean') ||
    (acceptedValues.length > 0 &&
      acceptedValues.every((acceptedValue) => ['on', 'off', 'auto'].includes(acceptedValue)))

  if (booleanLike && setting.control.kind === 'choice') return 'boolean'
  if (acceptedValues.length > 0 && setting.control.kind === 'choice') return 'choice'

  const numeric = numericMetadataForSetting(setting)
  const hasNumericBounds = numeric.min !== undefined || numeric.max !== undefined || numeric.step !== undefined
  if (
    setting.control.kind === 'range' ||
    hasNumericBounds ||
    hasSchemaKind(setting.valueSchema, 'integer') ||
    hasSchemaKind(setting.valueSchema, 'float')
  ) {
    return 'number'
  }

  if (setting.control.kind === 'choice') return 'choice'
  return 'text'
}

export function ConfigurationDefaultsControl(props: ConfigurationDefaultsControlProps) {
  const resolvedKind = controlKind(props.setting)
  const presentation = numericPresentation(props.setting)

  const control = (() => {
    switch (resolvedKind) {
      case 'boolean':
        return <SchemaBooleanControl {...props} />
      case 'choice':
        return <SchemaChoiceControl {...props} />
      case 'number':
        return (
          <SchemaNumberControl {...props} lowerBound={presentation.lowerBound} upperBound={presentation.upperBound} />
        )
      case 'array':
        return <SchemaArrayControl {...props} />
      case 'object':
        return <SchemaObjectControl {...props} />
      case 'path':
        return <SchemaPathControl {...props} />
      case 'url':
        return <SchemaUrlControl {...props} />
      case 'runtime-choice':
        return <SchemaRuntimeChoiceControl {...props} />
      case 'metric':
        return (
          <span className="rounded-[var(--radius)] border border-border-soft bg-surface px-2.5 py-1.5 font-mono text-[length:var(--density-type-control)] text-fg-dim">
            {formatMetricValue(props.setting)}
          </span>
        )
      case 'text':
        return openStringControl(props)
    }
  })()

  return <div className="flex min-w-[280px] justify-end">{control}</div>
}
