import { AlertTriangle, BookOpen, Copy, LockKeyhole } from 'lucide-react'
import type { RuntimeControlBootstrapPayload } from '@/features/configuration/api/config-adapter'
import {
  OWNER_CONTROL_DOCS_URL,
  formatRuntimeControlDisabledMessage,
  formatRuntimeControlDisabledReason
} from '@/features/configuration/components/runtime-control-copy'
import { copyStateLabel } from '@/lib/copyStateLabel'
import { useClipboardCopy } from '@/lib/useClipboardCopy'

function RuntimeCommandRow({ command, hint, index }: { command: string; hint: string; index: number }) {
  const { copyState, copyText } = useClipboardCopy()
  const [binary, action, ...args] = command.split(' ')

  return (
    <div className="grid grid-cols-[1.5rem_minmax(0,1fr)_4.5rem] items-start gap-2 px-5 py-2.5 sm:grid-cols-[1.75rem_minmax(0,1fr)_4.5rem]">
      <div className="pt-1.5 text-right font-mono text-[length:var(--density-type-annotation)] font-medium leading-none text-fg-faint">
        {index}
      </div>
      <div className="min-w-0">
        <div className="flex min-w-0 items-center gap-1.5 overflow-x-auto whitespace-nowrap font-mono text-[length:var(--density-type-caption-lg)] font-semibold leading-6">
          <span className="text-accent">$</span>
          <span className="text-accent">{binary}</span>
          <span className="text-warn">{action}</span>
          {args.map((arg) => (
            <span key={arg} className="text-accent-contrast">
              {arg}
            </span>
          ))}
        </div>
        <div className="mt-1 flex items-center gap-2 type-caption text-fg-dim">
          <span aria-hidden="true" className="font-mono text-warn">
            →
          </span>
          <span>{hint}</span>
        </div>
      </div>
      <button
        aria-label={`Copy ${command}`}
        className="ui-control inline-flex h-[30px] items-center justify-center gap-1.5 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-control)] font-medium leading-none"
        onClick={() => void copyText(command)}
        type="button"
      >
        <Copy aria-hidden="true" className="size-3.5 shrink-0" />
        {copyStateLabel(copyState)}
      </button>
    </div>
  )
}

export function ConfigurationRuntimeControlBanner({ bootstrap }: { bootstrap: RuntimeControlBootstrapPayload }) {
  const suggestedCommands = bootstrap.suggested_commands?.length
    ? bootstrap.suggested_commands
    : ['mesh-llm auth init --no-passphrase', 'mesh-llm serve --owner-required']
  const authInitCommand = suggestedCommands.includes('mesh-llm auth init --no-passphrase')
    ? 'mesh-llm auth init --no-passphrase'
    : suggestedCommands[0]
  const restartCommand = suggestedCommands.includes('mesh-llm serve --owner-required')
    ? 'mesh-llm serve --owner-required'
    : (suggestedCommands[1] ?? 'mesh-llm serve --owner-required')
  const commandPairs = [
    {
      command: authInitCommand,
      hint: 'Initialize owner identity (creates a local keypair)'
    },
    {
      command: restartCommand,
      hint: 'Restart the daemon so the new identity takes effect'
    }
  ]

  return (
    <section
      aria-labelledby="configuration-runtime-control-read-only"
      className="panel-shell overflow-hidden rounded-[var(--radius-lg)] border border-border bg-panel text-foreground"
    >
      <div className="panel-divider flex flex-col gap-3 border-b border-border-soft px-[14px] py-[10px] sm:flex-row sm:items-center sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-warn)_42%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-warn)_10%,var(--color-panel))] text-warn shadow-surface-inset">
            <LockKeyhole aria-hidden="true" className="size-4" strokeWidth={1.8} />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 id="configuration-runtime-control-read-only" className="type-panel-title text-foreground">
                Configuration UI is read-only
              </h2>
              <span className="inline-flex items-center gap-1.5 rounded-[var(--radius)] border border-[color:color-mix(in_oklab,var(--color-warn)_48%,var(--color-border))] bg-[color:color-mix(in_oklab,var(--color-warn)_8%,var(--color-panel))] px-2 py-0.5 font-mono text-[length:var(--density-type-annotation)] font-semibold uppercase leading-none tracking-[0.14em] text-warn">
                <AlertTriangle aria-hidden="true" className="size-3" strokeWidth={1.9} />
                {formatRuntimeControlDisabledReason(bootstrap)}
              </span>
            </div>
            <p className="mt-1 max-w-[72ch] type-caption text-fg-dim">
              {formatRuntimeControlDisabledMessage(bootstrap)}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2 sm:self-start">
          <a
            className="ui-control inline-flex h-[30px] items-center justify-center gap-1.5 rounded-[var(--radius)] border px-3 text-[length:var(--density-type-control)] font-medium leading-none"
            href={OWNER_CONTROL_DOCS_URL}
            rel="noreferrer"
            target="_blank"
          >
            <BookOpen aria-hidden="true" className="size-3.5" strokeWidth={1.8} />
            Docs
          </a>
        </div>
      </div>
      <div className="bg-background py-3">
        {commandPairs.map((commandPair, index) => (
          <RuntimeCommandRow key={commandPair.command} index={index + 1} {...commandPair} />
        ))}
      </div>
    </section>
  )
}
