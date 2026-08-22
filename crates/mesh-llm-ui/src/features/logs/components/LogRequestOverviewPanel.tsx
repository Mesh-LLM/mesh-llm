import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

type LogRequestOverviewPanelProps = {
  readonly ariaLabel: string
  readonly children: ReactNode
  readonly description?: string
  readonly icon: LucideIcon
  readonly title: string
}

export function LogRequestOverviewPanel({
  ariaLabel,
  children,
  description,
  icon: Icon,
  title
}: LogRequestOverviewPanelProps) {
  return (
    <section aria-label={ariaLabel} className="overflow-hidden rounded-[var(--radius)] border border-border bg-panel">
      <header className="flex items-center gap-3 border-b border-border-soft bg-panel-strong/55 px-[var(--panel-x)] py-[var(--panel-y)]">
        <span className="grid size-8 shrink-0 place-items-center rounded-[var(--radius-sm)] border border-[color:color-mix(in_oklab,var(--color-accent)_28%,var(--color-border-soft))] bg-[color:color-mix(in_oklab,var(--color-accent)_8%,var(--color-panel))] text-accent">
          <Icon aria-hidden="true" className="size-4" />
        </span>
        <div className="min-w-0">
          <h2 className="type-panel-title text-foreground">{title}</h2>
          {description === undefined ? null : <p className="type-caption mt-1 text-fg-dim">{description}</p>}
        </div>
      </header>
      {children}
    </section>
  )
}
