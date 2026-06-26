import type { InputHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/cn'

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
          'ui-control h-[32px] w-[108px] rounded-[var(--radius)] border bg-surface px-2.5 font-mono text-[length:var(--density-type-control)] text-foreground outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
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
