import type { CSSProperties } from 'react'
import {
  LOG_EVENT_CATEGORY_COLORS,
  LOG_EVENT_CATEGORY_LABELS,
  LOG_EVENT_CATEGORY_MARKER_CLASS
} from '@/features/logs/lib/log-event-category-style'
import type { LogEventCategory } from '@/features/logs/lib/log-event-ledger'
import { cn } from '@/lib/cn'

type LogEventCategoryBadgeProps = {
  readonly category: LogEventCategory
  readonly className?: string
}

function categoryBadgeStyle(category: LogEventCategory): CSSProperties {
  const color = LOG_EVENT_CATEGORY_COLORS[category]

  return {
    background: `color-mix(in oklab, ${color} 16%, var(--color-background))`,
    border: `1px solid color-mix(in oklab, ${color} 32%, var(--color-background))`
  }
}

export function LogEventCategoryBadge({ category, className }: LogEventCategoryBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-[5px] rounded-full px-2.5 py-0.5 font-medium text-[length:var(--density-type-caption)] text-fg-dim',
        className
      )}
      data-log-category={category}
      style={categoryBadgeStyle(category)}
    >
      <span
        aria-hidden="true"
        className={cn('size-2 shrink-0', LOG_EVENT_CATEGORY_MARKER_CLASS[category])}
        style={{ backgroundColor: LOG_EVENT_CATEGORY_COLORS[category] }}
      />
      {LOG_EVENT_CATEGORY_LABELS[category]}
    </span>
  )
}
