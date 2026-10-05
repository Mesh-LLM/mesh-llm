import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

type LogRequestTabBodyProps = {
  readonly active: boolean
  readonly children: ReactNode
}

const ACTIVE_BODY_CLASS =
  'overflow-y-auto overscroll-y-contain pb-[var(--shell-normal)] pt-[var(--panel-y)] outline-none focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent focus-visible:outline-solid'
const INACTIVE_BODY_CLASS = 'overflow-hidden py-[var(--panel-y)]'

export function LogRequestTabBody({ active, children }: LogRequestTabBodyProps) {
  return (
    <div
      className={cn('min-h-0 flex-1 px-[var(--panel-x)]', active ? ACTIVE_BODY_CLASS : INACTIVE_BODY_CLASS)}
      data-request-inspector-scroll={active ? 'body' : undefined}
      tabIndex={active ? 0 : undefined}
    >
      {children}
    </div>
  )
}
