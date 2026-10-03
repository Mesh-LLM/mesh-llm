import { Crosshair } from 'lucide-react'

type ChatTargetNoticeProps = {
  readonly target: string
  readonly onClear?: () => void
}

/** Shown above the composer while the chat is pointed at one node. */
export function ChatTargetNotice({ target, onClear }: ChatTargetNoticeProps) {
  return (
    <div
      className="mb-2 flex min-w-0 items-center gap-2 text-[length:var(--density-type-caption)] text-fg-dim"
      role="status"
    >
      <Crosshair aria-hidden="true" className="size-3.5 shrink-0" />
      <span className="min-w-0 truncate">
        Sending to{' '}
        <span className="font-mono text-foreground" title={target}>
          {target.slice(0, 12)}
        </span>
      </span>
      {onClear ? (
        <>
          <span aria-hidden="true">·</span>
          <button className="shrink-0 font-medium text-accent hover:underline" onClick={onClear} type="button">
            clear
          </button>
        </>
      ) : null}
    </div>
  )
}
