import { cn } from '@/lib/cn'
import type { SchemaSettingControlProps } from '@/features/configuration/components/settings/schema-control-utils'

export function SchemaPathControl({
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
      autoCapitalize="off"
      autoCorrect="off"
      className={cn(
        'ui-control h-[32px] w-full min-w-[280px] rounded-[var(--radius)] border bg-surface px-2.5 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
        disabled && 'cursor-not-allowed opacity-60',
        invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
      )}
      disabled={disabled}
      name={'name' in setting.control ? setting.control.name : setting.id}
      onChange={(event) => onChange(event.currentTarget.value)}
      placeholder={'placeholder' in setting.control ? setting.control.placeholder : './path/to/file'}
      spellCheck={false}
      type="text"
      value={value}
    />
  )
}
