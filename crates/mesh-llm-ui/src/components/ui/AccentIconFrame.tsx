import type { CSSProperties, ReactNode } from 'react'
import { cn } from '@/lib/cn'

type AccentIconFrameTone = 'accent' | 'subtle' | 'bad'

type AccentIconFrameProps = {
  readonly children: ReactNode
  readonly className?: string
  readonly style?: CSSProperties
  readonly tone?: AccentIconFrameTone
}

const frameStyleByTone: Record<AccentIconFrameTone, CSSProperties> = {
  accent: {
    background: 'color-mix(in oklab, var(--color-accent) 25%, transparent)',
    border: '1px solid color-mix(in oklab, var(--color-accent) 40%, var(--color-border))'
  },
  subtle: {
    background: 'color-mix(in oklab, var(--color-accent-soft) 42%, var(--color-panel-strong))',
    border: '1px solid color-mix(in oklab, var(--color-accent) 18%, var(--color-border))',
    color: 'color-mix(in oklab, var(--color-accent) 48%, var(--color-fg-dim))'
  },
  bad: {
    background: 'color-mix(in oklab, var(--color-bad) 18%, transparent)',
    border: '1px solid color-mix(in oklab, var(--color-bad) 30%, var(--color-border))',
    color: 'var(--color-bad-text)'
  }
}

export function AccentIconFrame({ children, className, style, tone = 'accent' }: AccentIconFrameProps) {
  return (
    <span
      className={cn(
        'flex size-[34px] shrink-0 items-center justify-center rounded-[var(--radius)]',
        tone === 'accent' && 'text-accent',
        className
      )}
      style={{ ...frameStyleByTone[tone], ...style }}
    >
      {children}
    </span>
  )
}
