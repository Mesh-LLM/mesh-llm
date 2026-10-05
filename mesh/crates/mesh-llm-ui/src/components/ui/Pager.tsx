import * as RadioGroup from '@radix-ui/react-radio-group'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Fragment } from 'react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/cn'

type PagerProps = {
  readonly ariaLabel: string
  readonly className?: string
  readonly count: number
  readonly nextLabel?: string
  readonly pageLabel?: (index: number) => string
  readonly previousLabel?: string
  readonly statusLabel?: (index: number, count: number) => string
  readonly value: number
  readonly variant?: 'dots' | 'numbered'
  readonly onValueChange: (value: number) => void
}

const dotStepClassName = 'size-6 rounded-full text-fg-dim hover:text-foreground disabled:opacity-40'

const dotRadioClassName =
  'inline-grid size-6 shrink-0 place-items-center rounded-full p-0 outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'

const dotClassName =
  'rounded-full transition-[width,background-color] duration-150 ease-out motion-reduce:transition-none'

const numberedStepClassName =
  'size-10 rounded-[var(--radius-control)] border border-border bg-panel text-fg-dim transition-colors hover:border-border-strong hover:bg-panel-strong hover:text-foreground disabled:pointer-events-none disabled:border-border-soft disabled:bg-panel disabled:text-fg-faint disabled:opacity-50'

const numberedRadioClassName =
  'inline-grid size-10 shrink-0 place-items-center rounded-[var(--radius-control)] border p-0 font-mono type-caption tabular-nums outline-none transition-colors duration-150 ease-out focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent motion-reduce:transition-none'

const numberClassName =
  'grid size-full place-items-center rounded-[var(--radius-control)] font-mono type-caption tabular-nums transition-colors duration-150 ease-out motion-reduce:transition-none'

const MAX_VISIBLE_PAGE_ITEMS = 7

const defaultStatusLabel = (index: number, count: number) => `Page ${index + 1} of ${count}`

function visiblePageIndexes(count: number, current: number): readonly number[] {
  if (count <= MAX_VISIBLE_PAGE_ITEMS) return Array.from({ length: count }, (_, index) => index)

  const windowSize = MAX_VISIBLE_PAGE_ITEMS - 2
  const windowStart = Math.max(1, Math.min(current - Math.floor(windowSize / 2), count - windowSize - 1))
  return [0, ...Array.from({ length: windowSize }, (_, offset) => windowStart + offset), count - 1]
}

export function Pager({
  ariaLabel,
  className,
  count,
  nextLabel = 'Next page',
  pageLabel,
  previousLabel = 'Previous page',
  statusLabel = defaultStatusLabel,
  value,
  variant = 'dots',
  onValueChange
}: PagerProps) {
  if (count < 2) return null
  const clamped = (next: number) => Math.min(count - 1, Math.max(0, next))
  const currentPage = clamped(value)
  const pageIndexes = visiblePageIndexes(count, currentPage)
  const numbered = variant === 'numbered'

  return (
    <div
      className={cn(
        numbered
          ? 'grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2'
          : 'flex items-center justify-center gap-0',
        className
      )}
    >
      <Button
        aria-label={previousLabel}
        className={numbered ? numberedStepClassName : dotStepClassName}
        disabled={currentPage <= 0}
        onClick={() => onValueChange(clamped(currentPage - 1))}
        size="icon"
        type="button"
        variant={numbered ? 'outline' : 'ghost'}
      >
        <ChevronLeft className="size-4" />
      </Button>
      <RadioGroup.Root
        aria-label={ariaLabel}
        className={numbered ? 'flex min-w-0 flex-wrap items-center justify-center gap-1.5' : 'flex items-center gap-0'}
        loop={false}
        onKeyDown={(event) => {
          const direction =
            event.key === 'ArrowLeft' || event.key === 'ArrowUp'
              ? -1
              : event.key === 'ArrowRight' || event.key === 'ArrowDown'
                ? 1
                : 0
          if (direction === 0) return
          event.preventDefault()
          onValueChange(clamped(currentPage + direction))
        }}
        onValueChange={(next) => onValueChange(clamped(Number(next)))}
        orientation="horizontal"
        value={String(currentPage)}
      >
        {pageIndexes.map((index, position) => {
          const hasGap = position < pageIndexes.length - 1 && pageIndexes[position + 1] !== index + 1
          return (
            <Fragment key={index}>
              <RadioGroup.Item
                aria-label={pageLabel?.(index) ?? `Page ${index + 1} of ${count}`}
                className={cn(
                  numbered ? numberedRadioClassName : dotRadioClassName,
                  numbered &&
                    (index === currentPage
                      ? 'border-accent bg-accent text-accent-ink'
                      : 'border-border bg-panel text-fg-dim hover:border-border-strong hover:bg-panel-strong hover:text-foreground')
                )}
                value={String(index)}
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    numbered ? numberClassName : dotClassName,
                    numbered
                      ? index === currentPage
                        ? 'bg-accent text-accent-ink'
                        : 'text-fg-dim hover:bg-panel-strong hover:text-foreground'
                      : index === currentPage
                        ? 'h-1.5 w-4 bg-accent'
                        : 'size-1.5 bg-border hover:bg-fg-faint'
                  )}
                >
                  {numbered ? index + 1 : null}
                </span>
              </RadioGroup.Item>
              {hasGap ? (
                <span
                  aria-hidden="true"
                  className={cn(
                    'shrink-0 overflow-visible text-center text-xs leading-none',
                    numbered ? 'w-3 text-fg-faint' : 'w-0 text-foreground'
                  )}
                  data-testid="pager-gap"
                >
                  …
                </span>
              ) : null}
            </Fragment>
          )
        })}
      </RadioGroup.Root>
      <span aria-atomic="true" aria-live="polite" className="sr-only" role="status">
        {statusLabel(currentPage, count)}
      </span>
      <Button
        aria-label={nextLabel}
        className={numbered ? numberedStepClassName : dotStepClassName}
        disabled={currentPage >= count - 1}
        onClick={() => onValueChange(clamped(currentPage + 1))}
        size="icon"
        type="button"
        variant={numbered ? 'outline' : 'ghost'}
      >
        <ChevronRight className="size-4" />
      </Button>
    </div>
  )
}
