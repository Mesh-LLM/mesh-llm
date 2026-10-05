import { CircleHelp } from 'lucide-react'
import type { ReactNode } from 'react'
import { TooltipContent, TooltipProvider, TooltipRoot, TooltipTrigger } from '@/components/ui/tooltip'

type DisabledControlFrameProps = {
  readonly children: ReactNode
  readonly details: readonly string[]
  readonly detailsId: string
  readonly disabled?: boolean
  readonly disabledDetails?: readonly string[]
}

function tooltipContent(details: readonly string[]) {
  const [firstDetail, ...remainingDetails] = details

  return (
    <div className="grid gap-1.5">
      <div>{firstDetail}</div>
      {remainingDetails.map((detail) => (
        <div className="border-t border-border-soft pt-1.5" key={detail}>
          {detail}
        </div>
      ))}
    </div>
  )
}

type InfoTriggerProps = {
  readonly details: readonly string[]
  readonly detailsId: string
  readonly label: string
}

export function SettingInfoTrigger({ details, detailsId, label }: InfoTriggerProps) {
  return (
    <TooltipProvider delayDuration={250} skipDelayDuration={120}>
      <TooltipRoot>
        <TooltipTrigger asChild>
          <button
            aria-label={label}
            type="button"
            className="ui-control inline-grid size-5 shrink-0 place-items-center rounded-[var(--radius)] border border-border-soft bg-panel-strong p-0 text-fg-faint transition-[border-color,background-color,color,box-shadow] hover:border-accent/45 hover:bg-accent/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <CircleHelp aria-hidden="true" className="size-3" strokeWidth={1.9} />
          </button>
        </TooltipTrigger>
        <TooltipContent
          aria-label={details.join(' ')}
          className="max-w-[320px] border border-border bg-panel px-3 py-2 font-sans text-[length:var(--density-type-caption)] leading-relaxed text-fg shadow-[var(--shadow-surface-popover)]"
          id={detailsId}
          side="top"
        >
          {tooltipContent(details)}
        </TooltipContent>
      </TooltipRoot>
    </TooltipProvider>
  )
}

export function DisabledControlFrame({
  children,
  details,
  detailsId,
  disabled = false,
  disabledDetails = []
}: DisabledControlFrameProps) {
  const hasDetails = details.length > 0
  const hasDisabledDetails = disabled && disabledDetails.length > 0

  if (!hasDetails && !hasDisabledDetails) return <>{children}</>

  return (
    <div className="flex min-w-0 items-start gap-2">
      <div className="min-w-0 flex-1">{children}</div>
      <div className="flex shrink-0 items-center gap-1.5">
        {hasDetails ? <SettingInfoTrigger details={details} detailsId={detailsId} label="Setting information" /> : null}
        {hasDisabledDetails ? (
          <SettingInfoTrigger details={disabledDetails} detailsId={`${detailsId}-disabled`} label="Why unavailable" />
        ) : null}
      </div>
    </div>
  )
}
