import { cn } from '@/lib/cn'
import type { SchemaSettingControlProps } from '@/features/configuration/components/settings/schema-control-utils'

function arrayItems(value: string) {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean)
}

export function SchemaArrayControl({
  ariaDescribedBy,
  disabled = false,
  invalid = false,
  onChange,
  setting,
  value
}: SchemaSettingControlProps) {
  const items = arrayItems(value)

  return (
    <div className="grid min-w-[280px] gap-2">
      <textarea
        aria-describedby={ariaDescribedBy}
        aria-invalid={invalid ? 'true' : undefined}
        aria-label={setting.label}
        className={cn(
          'ui-control min-h-[88px] w-full rounded-[var(--radius)] border bg-surface px-2.5 py-2 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
          disabled && 'cursor-not-allowed opacity-60',
          invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]'
        )}
        disabled={disabled}
        name={'name' in setting.control ? setting.control.name : setting.id}
        onChange={(event) => onChange(arrayItems(event.currentTarget.value).join(', '))}
        placeholder={'placeholder' in setting.control ? setting.control.placeholder : 'One item per line'}
        value={items.join('\n')}
      />
      {items.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {items.map((item) => (
            <span
              className="inline-flex items-center rounded-full border border-border-soft bg-panel-strong px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] text-fg-dim"
              key={item}
            >
              {item}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}
