import { RotateCcw } from 'lucide-react'
import { Tooltip } from '@/components/ui/tooltip'
import { cn } from '@/lib/cn'

export const RESTART_REQUIRED_TOOLTIP = 'This setting requires a restart to take effect'

type RestartRequiredIndicatorProps = {
  className?: string
}

export function RestartRequiredIndicator({ className }: RestartRequiredIndicatorProps) {
  return (
    <Tooltip content={RESTART_REQUIRED_TOOLTIP} side="bottom">
      <button
        aria-label="Restart required"
        type="button"
        className={cn(
          'ui-control inline-grid size-5 shrink-0 place-items-center rounded-[var(--radius)] border p-0 text-fg-faint transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
          className
        )}
      >
        <RotateCcw aria-hidden="true" className="size-3" strokeWidth={1.9} />
      </button>
    </Tooltip>
  )
}
