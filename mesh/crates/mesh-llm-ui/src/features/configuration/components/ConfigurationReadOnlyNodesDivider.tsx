import { LockKeyhole } from 'lucide-react'

export function ConfigurationReadOnlyNodesDivider() {
  return (
    <div className="mt-4 mb-2 flex items-center gap-2.5">
      <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-fg-faint">Peers</span>
      <span aria-hidden="true" className="h-px flex-1 bg-border-soft" />
      <span className="inline-flex shrink-0 items-center gap-1.5 font-mono text-[11px] font-medium uppercase leading-none tracking-[0.14em] text-fg-faint">
        read-only
        <LockKeyhole aria-hidden="true" className="size-3.5" strokeWidth={1.7} />
      </span>
    </div>
  )
}
