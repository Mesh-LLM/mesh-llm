import type { ChangeEventHandler } from 'react'

import { cn } from '@/lib/cn'

export type NativeSelectOption = {
  value: string
  label: string
  disabled?: boolean
}

type NativeSelectProps = {
  ariaDescribedBy?: string
  ariaLabel: string
  className?: string
  disabled?: boolean
  invalid?: boolean
  name: string
  onValueChange: (value: string) => void
  options: readonly NativeSelectOption[]
  value: string
}

export function NativeSelect({
  ariaDescribedBy,
  ariaLabel,
  className,
  disabled = false,
  invalid = false,
  name,
  onValueChange,
  options,
  value
}: NativeSelectProps) {
  const handleChange: ChangeEventHandler<HTMLSelectElement> = (event) => {
    onValueChange(event.currentTarget.value)
  }

  return (
    <select
      aria-describedby={ariaDescribedBy}
      aria-invalid={invalid ? 'true' : undefined}
      aria-label={ariaLabel}
      className={cn(
        'ui-control h-[32px] min-w-[240px] rounded-[var(--radius)] border bg-surface px-2.5 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
        disabled && 'cursor-not-allowed opacity-60',
        invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]',
        className
      )}
      disabled={disabled}
      name={name}
      onChange={handleChange}
      value={value}
    >
      {options.map((option) => (
        <option disabled={option.disabled} key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
}
