import { cn } from '@/lib/cn'
import {
  TEXT_FIELD_BASE_CLASS,
  type SchemaSettingControlProps
} from '@/features/configuration/components/settings/schema-control-utils'

function validUrl(value: string) {
  if (value.trim().length === 0) return true
  try {
    new URL(value)
    return true
  } catch {
    return false
  }
}

export function SchemaUrlControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  onChange,
  setting,
  value
}: SchemaSettingControlProps) {
  const isValid = validUrl(value)
  const invalidUrl = invalid || !isValid

  return (
    <input
      aria-describedby={ariaDescribedBy}
      aria-invalid={invalidUrl ? 'true' : undefined}
      aria-label={setting.label}
      autoCapitalize="off"
      autoCorrect="off"
      className={cn(
        TEXT_FIELD_BASE_CLASS,
        'w-full min-w-[280px]',
        disabled && 'cursor-not-allowed opacity-60',
        invalidUrl && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
      )}
      disabled={disabled}
      name={'name' in setting.control ? setting.control.name : setting.id}
      onChange={(event) => onChange(event.currentTarget.value)}
      placeholder={'placeholder' in setting.control ? setting.control.placeholder : 'https://example.com/resource.gguf'}
      spellCheck={false}
      type="url"
      value={value}
    />
  )
}
