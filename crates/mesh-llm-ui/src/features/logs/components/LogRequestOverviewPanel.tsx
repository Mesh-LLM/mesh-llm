import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

type LogRequestOverviewPanelProps = {
  readonly ariaLabel: string
  readonly children: ReactNode
  readonly description: string
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
      <header className="flex items-start gap-2 border-b border-border-soft px-[var(--panel-x)] py-[var(--panel-y)]">
        <Icon aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-accent" />
        <div className="min-w-0">
          <h2 className="type-panel-title text-foreground">{title}</h2>
          <p className="type-caption mt-1 text-fg-dim">{description}</p>
        </div>
      </header>
      {children}
    </section>
  )
}
