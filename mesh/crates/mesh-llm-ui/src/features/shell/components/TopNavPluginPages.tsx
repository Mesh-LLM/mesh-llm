import type { MouseEvent } from 'react'
import { Plug } from 'lucide-react'
import { HeaderHoverCard } from '@/features/shell/components/HeaderHoverCard'
import { cn } from '@/lib/cn'

export type TopNavPluginPageItem = {
  readonly pluginName: string
  readonly pageId: string
  readonly label: string
  readonly href: string
  readonly active?: boolean
}

type TopNavPluginPagesProps = {
  readonly items?: readonly TopNavPluginPageItem[]
  readonly onNavigate?: (item: TopNavPluginPageItem) => void
}

function isPlainLeftClick(event: MouseEvent<HTMLAnchorElement>) {
  return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey
}

export function TopNavPluginPages({ items = [], onNavigate }: TopNavPluginPagesProps) {
  if (items.length === 0) return null

  if (items.length === 1) {
    const [item] = items
    if (!item) return null

    return (
      <nav aria-label="Plugin pages" className="flex min-w-0 items-center">
        <a
          aria-current={item.active ? 'page' : undefined}
          aria-label={item.label}
          className={cn(
            'inline-flex h-[var(--nav-action-size)] min-w-0 items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-caption)] font-medium',
            item.active ? 'ui-control-primary' : 'ui-control-ghost'
          )}
          href={item.href}
          onClick={(event) => {
            if (!isPlainLeftClick(event)) return
            if (onNavigate) {
              event.preventDefault()
              onNavigate(item)
            }
          }}
        >
          <Plug className="size-[var(--nav-icon-size)] shrink-0" aria-hidden="true" />
          <span className="hidden max-w-48 truncate md:inline">{item.label}</span>
        </a>
      </nav>
    )
  }

  return (
    <HeaderHoverCard
      trigger={(triggerProps) => (
        <button
          {...triggerProps}
          aria-label="Plugin pages"
          className="ui-control inline-flex h-[var(--nav-action-size)] items-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-caption)] font-medium text-fg-dim"
          type="button"
        >
          <Plug className="size-[var(--nav-icon-size)]" aria-hidden="true" />
          <span className="hidden sm:inline">Plugins</span>
        </button>
      )}
      align="start"
      eyebrow="Plugin surfaces"
      title="Plugin pages"
      description="Host-projected plugin pages that are ready, enabled, and declared by the running plugin."
      triggerMode="click"
    >
      <nav aria-label="Plugin pages" className="space-y-1">
        {items.map((item) => (
          <a
            key={`${item.pluginName}:${item.pageId}`}
            href={item.href}
            className={cn(
              'ui-row-action grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 rounded-[var(--radius)] border border-border-soft px-3 py-2 text-left',
              'text-[length:var(--density-type-caption-lg)] text-foreground'
            )}
            onClick={(event) => {
              if (!isPlainLeftClick(event)) return
              if (onNavigate) {
                event.preventDefault()
                onNavigate(item)
              }
            }}
          >
            <span className="font-medium">{item.label}</span>
            <span className="type-label text-good">Ready</span>
            <span className="min-w-0 font-mono text-[length:var(--density-type-caption)] text-fg-faint">
              {item.pluginName}
            </span>
            <span className="font-mono text-[length:var(--density-type-caption)] text-fg-faint">{item.pageId}</span>
          </a>
        ))}
      </nav>
    </HeaderHoverCard>
  )
}
