import { useState, type ReactNode } from 'react'
import { AccentIconFrame } from '@/components/ui/AccentIconFrame'
import { InfoBanner } from '@/components/ui/InfoBanner'
import { SidebarNavigation } from '@/components/ui/SidebarNavigation'
import { HighlightedTomlLines } from '@/features/configuration/components/toml-highlight'
import { cn } from '@/lib/cn'

export type SettingsCategoryItem = {
  id: string
  label: string
  summary: string
  count: number
  icon?: ReactNode
}

type SettingsSummaryBannerProps = {
  eyebrow?: string
  titleId?: string
  title: string
  description: ReactNode
  status: string
  action?: ReactNode
}

type SettingsCategoryRailProps = {
  categories: readonly SettingsCategoryItem[]
  activeId: string
  footer: ReactNode
  onSelect: (id: string) => void
}

type SettingsSectionProps = {
  id: string
  icon: ReactNode
  title: string
  subtitle: string
  children: ReactNode
}

type SettingsRowProps = {
  className?: string
  disabled?: boolean
  disabledReason?: string
  dirty?: boolean
  errorMessage?: string
  errorMessageId?: string
  label: ReactNode
  labelAccessory?: ReactNode
  hint: string
  hintId?: string
  children: ReactNode
  showDisabledReason?: boolean
}

type SettingsPreviewRailProps = {
  title: string
  code: string
  tip: ReactNode
}

export function SettingsSummaryBanner({
  eyebrow,
  titleId = 'defaults-summary-heading',
  title,
  description,
  status,
  action
}: SettingsSummaryBannerProps) {
  return (
    <InfoBanner
      action={action}
      descriptionClassName="max-w-none whitespace-normal"
      description={
        <>
          {eyebrow ? <span className="type-label mb-1 block text-fg-faint">{eyebrow}</span> : null}
          <span className="text-[length:var(--density-type-control)] leading-relaxed">{description}</span>
        </>
      }
      status={
        <span className="inline-flex rounded-full border border-border-soft bg-transparent px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] leading-none text-fg-dim">
          {status}
        </span>
      }
      title={title}
      titleId={titleId}
    />
  )
}

export function SettingsCategoryRail({ categories, activeId, footer, onSelect }: SettingsCategoryRailProps) {
  return (
    <SidebarNavigation
      activeId={activeId}
      ariaLabel="Defaults sections"
      className="static self-start lg:sticky lg:top-[72px]"
      eyebrow="Categories"
      footer={footer}
      items={categories.map((category) => ({
        id: category.id,
        label: category.label,
        count: category.count,
        icon: category.icon
      }))}
      onSelect={onSelect}
    />
  )
}

export function SettingsSection({ id, icon, title, subtitle, children }: SettingsSectionProps) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-heading`}
      className="panel-shell scroll-mt-20 rounded-[var(--radius-lg)] border border-border bg-panel px-5 pb-5 pt-4 shadow-surface-panel"
      data-panel-soft-elevation="none"
    >
      <header className="mb-2 flex items-start gap-3">
        <AccentIconFrame className="size-9">{icon}</AccentIconFrame>
        <div>
          <h3 id={`${id}-heading`} className="type-panel-title text-foreground">
            {title}
          </h3>
          <p className="mt-1 type-caption text-fg-dim">{subtitle}</p>
        </div>
      </header>
      <div>{children}</div>
    </section>
  )
}

export function SettingsRow({
  className,
  disabled = false,
  disabledReason,
  dirty = false,
  errorMessage,
  errorMessageId,
  label,
  labelAccessory,
  hint,
  hintId,
  children,
  showDisabledReason = true
}: SettingsRowProps) {
  return (
    <div
      className={cn(
        'grid min-h-[68px] gap-3 border-t border-border-soft py-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-start',
        disabled && 'opacity-55',
        className
      )}
      data-settings-row="true"
      data-settings-row-disabled={disabled ? 'true' : undefined}
      data-settings-row-dirty={dirty ? 'true' : undefined}
      aria-disabled={disabled ? 'true' : undefined}
    >
      <div className="min-w-0">
        <div className="flex min-h-6 flex-wrap items-start gap-2">
          <p
            className={cn(
              'text-[length:var(--density-type-control)] font-medium leading-tight text-foreground',
              dirty && 'text-warn'
            )}
          >
            {label}
          </p>
          {labelAccessory ? <div className="-mt-0.5 shrink-0">{labelAccessory}</div> : null}
        </div>
        <p className="mt-1 type-caption text-fg-dim" id={hintId}>
          {hint}
        </p>
        {errorMessage ? (
          <p className="mt-1 type-caption font-medium text-bad" id={errorMessageId}>
            {errorMessage}
          </p>
        ) : null}
        {disabledReason && showDisabledReason ? (
          <p className="mt-1 type-caption text-fg-dim">{disabledReason}</p>
        ) : null}
      </div>
      <div className="flex min-w-0 justify-end md:justify-self-stretch md:pt-0.5">{children}</div>
    </div>
  )
}

export function SettingsPreviewRail({ title, code, tip }: SettingsPreviewRailProps) {
  const [scrollOffset, setScrollOffset] = useState({ left: 0, top: 0 })

  return (
    <aside aria-label={title} className="sticky top-[72px] space-y-3">
      <section
        className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-3 shadow-surface-panel"
        data-panel-soft-elevation="none"
      >
        <h3 className="flex items-center gap-2 text-[length:var(--density-type-control)] font-semibold text-foreground">
          <span>Preview</span>
          <span className="font-mono text-fg-dim">{title}</span>
        </h3>
        <div className="relative mt-2.5 max-h-[320px] min-h-[210px] overflow-hidden rounded-[var(--radius)] border border-border-soft bg-background font-mono text-[length:var(--density-type-caption)] leading-relaxed">
          <pre
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 m-0 whitespace-pre p-3 text-fg-dim"
            style={{ transform: `translate(${-scrollOffset.left}px, ${-scrollOffset.top}px)` }}
          >
            <HighlightedTomlLines toml={code} />
          </pre>
          <textarea
            aria-label={`${title} preview code`}
            className="absolute inset-0 block h-full w-full resize-none overflow-auto bg-transparent p-3 font-mono leading-relaxed text-transparent caret-transparent outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            onScroll={(event) =>
              setScrollOffset({ left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop })
            }
            readOnly
            value={code}
            wrap="off"
          />
        </div>
      </section>
      <section className="panel-shell rounded-[var(--radius-lg)] border border-dashed border-border bg-panel p-3 type-caption text-fg-dim">
        <div className="mb-1.5 type-label text-fg-faint">TIP</div>
        {tip}
      </section>
    </aside>
  )
}
