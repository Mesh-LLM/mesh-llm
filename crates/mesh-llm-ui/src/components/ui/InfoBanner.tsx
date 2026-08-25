import type { ReactNode } from 'react'
import { CircleAlert } from 'lucide-react'
import { AccentIconFrame } from '@/components/ui/AccentIconFrame'
import { cn } from '@/lib/cn'

type InfoBannerVariant = 'default' | 'error'
type InfoBannerAnnotationTone = 'warn' | 'bad'

export type InfoBannerProps = {
  readonly title: ReactNode
  readonly description: ReactNode
  readonly action?: ReactNode
  readonly actionClassName?: string
  readonly annotation?: ReactNode
  readonly className?: string
  readonly contentClassName?: string
  readonly descriptionClassName?: string
  readonly leadingIcon?: ReactNode
  readonly leadingIconClassName?: string
  readonly status?: ReactNode
  readonly titleClassName?: string
  readonly titleId?: string
  readonly titleLevel?: 'h1' | 'h2' | 'h3'
  readonly variant?: InfoBannerVariant
}

type InfoBannerAnnotationProps = {
  readonly ariaLabel: string
  readonly children: ReactNode
  readonly className?: string
  readonly tone?: InfoBannerAnnotationTone
}

const bannerBackground: Record<InfoBannerVariant, string> = {
  default:
    'linear-gradient(90deg, color-mix(in oklab, var(--color-accent) 10%, var(--color-panel)) 0%, var(--color-panel) 60%)',
  error:
    'linear-gradient(90deg, color-mix(in oklab, var(--color-bad) 10%, var(--color-panel)) 0%, var(--color-panel) 60%)'
}

const annotationToneClass: Record<InfoBannerAnnotationTone, string> = {
  warn: 'text-warn',
  bad: 'text-bad'
}

export function InfoBannerAnnotation({ ariaLabel, children, className, tone = 'warn' }: InfoBannerAnnotationProps) {
  return (
    <fieldset
      aria-label={ariaLabel}
      className={cn(
        'm-0 mt-3 flex min-w-0 items-start gap-2 border-x-0 border-b-0 border-t border-border-soft p-0 pt-3 type-caption',
        className
      )}
    >
      <CircleAlert aria-hidden="true" className={cn('mt-0.5 size-3.5 shrink-0', annotationToneClass[tone])} />
      <div className="min-w-0 flex-1 text-fg-dim">{children}</div>
    </fieldset>
  )
}

export function InfoBanner({
  title,
  description,
  action,
  actionClassName,
  className,
  contentClassName,
  descriptionClassName,
  leadingIcon,
  leadingIconClassName,
  status,
  titleClassName,
  titleId,
  titleLevel = 'h2',
  variant = 'default',
  annotation
}: InfoBannerProps) {
  const Heading = titleLevel

  return (
    <section
      aria-labelledby={titleId}
      className={cn(
        'panel-shell flex items-center gap-5 rounded-[var(--radius-lg)] border border-border px-5 py-4',
        variant === 'error' && 'border-bad/40',
        className
      )}
      role={variant === 'error' ? 'alert' : undefined}
      style={{ background: bannerBackground[variant] }}
    >
      {leadingIcon ? (
        <AccentIconFrame className={leadingIconClassName} tone={variant === 'error' ? 'bad' : 'accent'}>
          {leadingIcon}
        </AccentIconFrame>
      ) : null}
      <div className={cn('min-w-0 flex-1', contentClassName)}>
        <div className="flex flex-wrap items-center gap-2">
          <Heading
            id={titleId}
            className={cn(
              titleLevel === 'h1'
                ? 'type-headline'
                : 'text-[length:var(--density-type-title)] font-semibold leading-tight text-foreground',
              titleClassName
            )}
          >
            {title}
          </Heading>
          {status ? <div>{status}</div> : null}
        </div>
        <div className={cn('type-caption mt-1.5 text-fg-dim', descriptionClassName)}>{description}</div>
        {annotation}
      </div>
      {action ? (
        <div className={cn('flex shrink-0 items-center justify-end self-center', actionClassName)}>{action}</div>
      ) : null}
    </section>
  )
}
