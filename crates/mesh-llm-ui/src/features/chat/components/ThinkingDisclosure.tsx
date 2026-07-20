import * as Collapsible from '@radix-ui/react-collapsible'
import { ChevronRight, Loader2, Network } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { cn } from '@/lib/cn'
import { MOA_PROGRESS_LABEL } from '@/features/chat/lib/moa-progress'

type ThinkingDisclosureProps = {
  active: boolean
  children: ReactNode
}

export function ThinkingDisclosure({ active, children }: ThinkingDisclosureProps) {
  const [open, setOpen] = useState(false)
  const label = active ? MOA_PROGRESS_LABEL : 'Peer consultation'

  return (
    <Collapsible.Root
      className="block overflow-hidden rounded-[var(--radius)] border text-[length:var(--density-type-body)] leading-[1.5] text-fg-dim"
      data-thinking-state={active ? 'active' : 'complete'}
      onOpenChange={setOpen}
      open={open}
      style={{
        background: 'color-mix(in oklab, var(--color-accent) 5%, var(--color-panel))',
        borderColor: 'color-mix(in oklab, var(--color-accent) 22%, var(--color-border-soft))'
      }}
    >
      <Collapsible.Trigger asChild>
        <button
          aria-label={`${label} ${open ? 'Hide' : 'Show'} details`}
          className="flex w-full select-none items-center gap-2 px-3 py-2.5 text-left font-mono text-[length:var(--density-type-label)] uppercase tracking-[0.07em] text-fg-faint outline-none transition-colors hover:text-fg-dim focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent"
          type="button"
        >
          {active ? (
            <Loader2 className="size-3 shrink-0 animate-spin" aria-hidden={true} />
          ) : (
            <Network className="size-3 shrink-0" aria-hidden={true} strokeWidth={1.7} />
          )}
          <span aria-live="polite" className="min-w-0 flex-1 truncate">
            {label}
          </span>
          <span className="hidden normal-case tracking-normal text-fg-faint sm:inline">
            {open ? 'Hide details' : 'Show details'}
          </span>
          <ChevronRight
            aria-hidden={true}
            className={cn('size-3 shrink-0 transition-transform', open && 'rotate-90')}
          />
        </button>
      </Collapsible.Trigger>
      <Collapsible.Content
        aria-label={`${label} details`}
        className="max-h-64 overflow-y-auto border-t border-border-soft px-3 py-2.5 outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent"
        forceMount
        hidden={!open}
        role="region"
        tabIndex={open ? 0 : -1}
      >
        {children}
      </Collapsible.Content>
    </Collapsible.Root>
  )
}
