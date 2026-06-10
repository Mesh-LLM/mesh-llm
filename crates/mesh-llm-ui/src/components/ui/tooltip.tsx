import * as RadixTooltip from '@radix-ui/react-tooltip'
import { useSyncExternalStore } from 'react'
import type { ReactElement, ReactNode } from 'react'

// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipProvider = RadixTooltip.Provider
// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipRoot = RadixTooltip.Root
// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipTrigger = RadixTooltip.Trigger
// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipPortal = RadixTooltip.Portal
// eslint-disable-next-line react-refresh/only-export-components -- re-exports of Radix UI components for convenience
export const TooltipArrow = RadixTooltip.Arrow

export function TooltipContent({
  className,
  sideOffset = 6,
  ...props
}: React.ComponentPropsWithoutRef<typeof RadixTooltip.Content>) {
  return (
    <RadixTooltip.Portal>
      <RadixTooltip.Content
        sideOffset={sideOffset}
        className={`surface-menu-panel z-50 max-w-[240px] rounded-[var(--radius)] px-2.5 py-1.5 font-mono text-[length:var(--density-type-annotation)] leading-snug text-fg outline-none ${className ?? ''}`}
        collisionPadding={8}
        {...props}
      >
        {props.children}
        <RadixTooltip.Arrow className="fill-panel-strong stroke-border" height={5} width={9} />
      </RadixTooltip.Content>
    </RadixTooltip.Portal>
  )
}

type TooltipProps = {
  children: ReactElement
  content: ReactNode
  side?: RadixTooltip.TooltipContentProps['side']
}

const TOUCH_TOOLTIP_QUERY = '(hover: none), (pointer: coarse)'

function tooltipShouldBeDisabledForTouch() {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false
  return window.matchMedia(TOUCH_TOOLTIP_QUERY).matches
}

function subscribeToTouchTooltipChanges(onStoreChange: () => void) {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return () => {}

  const mediaQuery = window.matchMedia(TOUCH_TOOLTIP_QUERY)
  const handleChange = () => onStoreChange()

  if (typeof mediaQuery.addEventListener === 'function') {
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }

  mediaQuery.addListener(handleChange)
  return () => mediaQuery.removeListener(handleChange)
}

function useTooltipDisabledForTouch() {
  return useSyncExternalStore(subscribeToTouchTooltipChanges, tooltipShouldBeDisabledForTouch, () => false)
}

export function Tooltip({ children, content, side = 'top' }: TooltipProps) {
  const disabledForTouch = useTooltipDisabledForTouch()
  if (disabledForTouch) return children

  return (
    <RadixTooltip.Provider delayDuration={250} skipDelayDuration={120}>
      <RadixTooltip.Root>
        <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
        <TooltipContent side={side}>{content}</TooltipContent>
      </RadixTooltip.Root>
    </RadixTooltip.Provider>
  )
}
