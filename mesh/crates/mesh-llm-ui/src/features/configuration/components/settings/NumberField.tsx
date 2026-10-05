import type { InputHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/cn'
import { TEXT_FIELD_BASE_CLASS } from '@/features/configuration/components/settings/schema-control-utils'

type NumberFieldProps = InputHTMLAttributes<HTMLInputElement> & {
  readonly inputClassName?: string
  readonly invalid?: boolean
  readonly unit?: ReactNode
}

export function NumberField({
  className,
  disabled,
  inputClassName,
  invalid = false,
  unit,
  ...props
}: NumberFieldProps) {
  return (
    <div className={cn('grid min-w-[108px] justify-items-end gap-1', className)}>
      <input
        {...props}
        aria-invalid={invalid ? 'true' : props['aria-invalid']}
        className={cn(
          TEXT_FIELD_BASE_CLASS,
          'w-[108px]',
          disabled && 'cursor-not-allowed opacity-60',
          invalid && 'border-bad shadow-[var(--shadow-surface-error-inset)]',
          inputClassName
        )}
        disabled={disabled}
      />
      {unit ? (
        <span className="block max-w-full text-right font-mono text-[length:var(--density-type-caption)] leading-none text-fg-dim">
          {unit}
        </span>
      ) : null}
    </div>
  )
}
