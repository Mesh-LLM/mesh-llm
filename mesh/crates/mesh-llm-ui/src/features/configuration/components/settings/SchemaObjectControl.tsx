import { cn } from '@/lib/cn'
import type { SchemaSettingControlProps } from '@/features/configuration/components/settings/schema-control-utils'

function isJsonObject(value: string) {
  if (value.trim().length === 0) return true

  try {
    const parsed: unknown = JSON.parse(value)
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
  } catch {
    return false
  }
}

export function SchemaObjectControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  onChange,
  setting,
  value
}: SchemaSettingControlProps) {
  const validObject = isJsonObject(value)
  const invalidObject = invalid || !validObject

  return (
    <textarea
      aria-describedby={ariaDescribedBy}
      aria-invalid={invalidObject ? 'true' : undefined}
      aria-label={setting.label}
      className={cn(
        'ui-control min-h-[96px] w-full min-w-[280px] rounded-[var(--radius)] border bg-surface px-2.5 py-2 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
        disabled && 'cursor-not-allowed opacity-60',
        invalidObject && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
      )}
      disabled={disabled}
      name={'name' in setting.control ? setting.control.name : setting.id}
      onChange={(event) => onChange(event.currentTarget.value)}
      placeholder={'placeholder' in setting.control ? setting.control.placeholder : '{\n  "key": "value"\n}'}
      value={value}
    />
  )
}
