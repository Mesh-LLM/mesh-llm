import { cn } from '@/lib/cn'
import { Minus, Plus } from 'lucide-react'
import { useCallback, type ChangeEvent, type FocusEvent, type KeyboardEvent } from 'react'

type StepperProps = {
  value: number
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  className?: string
  inputClassName?: string
  onChange: (value: number) => void
  onBlur?: (event: FocusEvent<HTMLInputElement>) => void
  'aria-label'?: string
}

function clamp(value: number, min: number | undefined, max: number | undefined) {
  let clamped = value
  if (min !== undefined) clamped = Math.max(clamped, min)
  if (max !== undefined) clamped = Math.min(clamped, max)
  return clamped
}

function Stepper({
  value,
  min,
  max,
  step = 1,
  disabled = false,
  className,
  inputClassName,
  onChange,
  onBlur,
  'aria-label': ariaLabel
}: StepperProps) {
  step = Math.max(1, Math.abs(step))

  const canDecrement = min === undefined || value > min
  const canIncrement = max === undefined || value < max

  const decrement = useCallback(() => {
    onChange(clamp(value - step, min, max))
  }, [value, step, min, max, onChange])

  const increment = useCallback(() => {
    onChange(clamp(value + step, min, max))
  }, [value, step, min, max, onChange])

  const handleInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const parsed = parseFloat(e.target.value)
      if (!Number.isNaN(parsed)) {
        onChange(clamp(parsed, min, max))
      }
    },
    [onChange, min, max]
  )

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        increment()
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        decrement()
      }
    },
    [increment, decrement]
  )

  return (
    <div
      className={cn(
        'inline-flex items-stretch overflow-hidden rounded-md border border-border bg-background',
        disabled && 'cursor-not-allowed opacity-60',
        className
      )}
      role="group"
      aria-label={ariaLabel}
    >
      <button
        type="button"
        disabled={disabled || !canDecrement}
        onClick={decrement}
        aria-label={ariaLabel ? `Decrease ${ariaLabel}` : 'Decrease value'}
        className={cn(
          'inline-flex h-8 w-8 items-center justify-center text-fg-dim transition-colors duration-150 ease-out',
          'hover:bg-secondary active:bg-secondary/80',
          'disabled:cursor-not-allowed disabled:opacity-40',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring'
        )}
      >
        <Minus className="h-4 w-4" />
      </button>
      <input
        type="text"
        inputMode="numeric"
        value={value}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onBlur={onBlur}
        disabled={disabled}
        className={cn(
          'h-8 w-14 border-x border-border bg-transparent text-center text-sm tabular-nums outline-none',
          'disabled:cursor-not-allowed',
          inputClassName
        )}
        aria-label={ariaLabel ? `${ariaLabel} value` : undefined}
      />
      <button
        type="button"
        disabled={disabled || !canIncrement}
        onClick={increment}
        aria-label={ariaLabel ? `Increase ${ariaLabel}` : 'Increase value'}
        className={cn(
          'inline-flex h-8 w-8 items-center justify-center text-fg-dim transition-colors duration-150 ease-out',
          'hover:bg-secondary active:bg-secondary/80',
          'disabled:cursor-not-allowed disabled:opacity-40',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring'
        )}
      >
        <Plus className="h-4 w-4" />
      </button>
    </div>
  )
}

export { Stepper }
export type { StepperProps }
