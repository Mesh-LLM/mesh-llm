import { RotateCcw } from 'lucide-react'
import { Tooltip } from '@/components/ui/tooltip'
import { cn } from '@/lib/cn'

export const SETTING_RESET_TOOLTIP = 'Reset this setting to its default value'

type SettingResetButtonProps = {
  className?: string
  label: string
  onClick: () => void
}

export function SettingResetButton({ className, label, onClick }: SettingResetButtonProps) {
  return (
    <Tooltip content={SETTING_RESET_TOOLTIP} side="bottom">
      <button
        aria-label={label}
        type="button"
        className={cn(
          'ui-control inline-grid size-6 shrink-0 place-items-center rounded-[var(--radius)] border p-0 text-fg-faint transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
          className
        )}
        onClick={onClick}
      >
        <RotateCcw aria-hidden="true" className="size-3.5" strokeWidth={1.9} />
      </button>
    </Tooltip>
  )
}
