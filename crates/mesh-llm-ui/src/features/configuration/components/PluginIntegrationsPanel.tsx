import { useMemo } from 'react'
import { PlugZap, ShieldAlert } from 'lucide-react'
import { StatusBadge, type StatusBadgeTone } from '@/components/ui/StatusBadge'
import {
  adaptPluginSummariesToWebUiEntries,
  usePluginSummariesQuery,
  useSetPluginWebUiEnabledMutation,
  type PluginWebUiEntry
} from '@/features/plugins/api/plugin-web-ui'
import { PluginConfigSectionMount } from '@/features/configuration/components/PluginConfigSectionMount'
import type { PluginSummaryRaw, PluginWebUiConfigSectionRaw } from '@/lib/api/plugin-types'
import { cn } from '@/lib/cn'

type PluginIntegrationsPanelProps = {
  readonly metadataEnabled: boolean
}

function assertNever(value: never): never {
  throw new TypeError(`Unhandled plugin web UI state: ${String(value)}`)
}

function webUiTone(entry: PluginWebUiEntry): StatusBadgeTone {
  switch (entry.state) {
    case 'ready':
      return 'good'
    case 'disabled':
    case 'plugin_not_running':
      return 'warn'
    case 'invalid':
      return 'bad'
    case 'none':
      return 'muted'
    default:
      return assertNever(entry.state)
  }
}

function webUiLabel(entry: PluginWebUiEntry): string {
  switch (entry.state) {
    case 'ready':
      return 'Web UI ready'
    case 'disabled':
      return 'Web UI disabled'
    case 'invalid':
      return 'Web UI invalid'
    case 'plugin_not_running':
      return 'Plugin not running'
    case 'none':
      return 'Web UI not declared'
    default:
      return assertNever(entry.state)
  }
}

function processTone(summary: PluginSummaryRaw): StatusBadgeTone {
  if (!summary.enabled) return 'muted'
  if (summary.status === 'running') return 'good'
  if (summary.status === 'error' || summary.status === 'failed') return 'bad'
  return 'warn'
}

function pluginDescription(summary: PluginSummaryRaw): string | undefined {
  const directDescription = summary.description?.trim()
  if (directDescription) return directDescription

  const manifestDescription = summary.manifest?.description
  return typeof manifestDescription === 'string' && manifestDescription.trim() ? manifestDescription : undefined
}

function projectedConfigSections(entry: PluginWebUiEntry): readonly PluginWebUiConfigSectionRaw[] {
  if (entry.state !== 'ready' || !entry.declared || !entry.enabled || !entry.available) return []
  return entry.configSections.filter(
    (section) => section.parent_tab === undefined || section.parent_tab === 'integrations'
  )
}

function PluginWebUiToggle({ entry }: { readonly entry: PluginWebUiEntry }) {
  const mutation = useSetPluginWebUiEnabledMutation(entry.pluginName)
  const checked = entry.enabled && entry.state !== 'disabled'

  return (
    <button
      aria-checked={checked}
      aria-label={`${entry.pluginName} web UI projection`}
      className={cn(
        'ui-control inline-flex h-[30px] items-center gap-2 rounded-[var(--radius)] border px-2.5 text-[length:var(--density-type-control)] font-semibold',
        checked && 'border-accent bg-accent/10 text-accent'
      )}
      disabled={mutation.isPending}
      onClick={() => mutation.mutate(!checked)}
      role="switch"
      type="button"
    >
      <span className="relative inline-flex h-3.5 w-6 rounded-full border border-current/40 bg-current/10">
        <span
          className={cn(
            'absolute top-1/2 size-2 -translate-y-1/2 rounded-full bg-current transition-transform',
            checked ? 'translate-x-[13px]' : 'translate-x-[3px]'
          )}
        />
      </span>
      {checked ? 'Web UI on' : 'Web UI off'}
    </button>
  )
}

function PluginIntegrationCard({
  entry,
  summary
}: {
  readonly entry: PluginWebUiEntry
  readonly summary: PluginSummaryRaw
}) {
  const sections = projectedConfigSections(entry)
  const description = pluginDescription(summary)

  return (
    <article
      className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-4"
      aria-labelledby={`${entry.pluginName}-integration-heading`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="type-label text-fg-faint">Plugin</div>
          <h3
            className="type-panel-title mt-1 font-mono text-foreground"
            id={`${entry.pluginName}-integration-heading`}
          >
            {entry.pluginName}
          </h3>
          {description ? <p className="type-caption mt-1 text-fg-dim">{description}</p> : null}
        </div>
        {entry.declared ? <PluginWebUiToggle entry={entry} /> : null}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <StatusBadge tone={processTone(summary)}>
          {summary.enabled ? 'Process enabled' : 'Process disabled'}
        </StatusBadge>
        <StatusBadge tone={processTone(summary)}>{summary.status}</StatusBadge>
        <StatusBadge tone={webUiTone(entry)}>{webUiLabel(entry)}</StatusBadge>
        <StatusBadge tone={entry.available ? 'good' : 'muted'}>
          {entry.available ? 'Assets available' : 'Assets unavailable'}
        </StatusBadge>
      </div>

      {entry.unavailableReason ? <p className="type-caption mt-3 text-fg-dim">{entry.unavailableReason}</p> : null}
      {entry.configSections.length > 0 ? (
        <p className="type-caption mt-3 text-fg-dim">
          Config sections:{' '}
          <span className="font-mono text-foreground">
            {entry.configSections.map((section) => section.title).join(', ')}
          </span>
        </p>
      ) : null}
      {sections.length > 0 ? (
        <div className="mt-4 space-y-3">
          {sections.map((section) => (
            <PluginConfigSectionMount
              key={section.id}
              pluginName={entry.pluginName}
              section={section}
              webUi={summary.web_ui}
            />
          ))}
        </div>
      ) : null}
    </article>
  )
}

export function PluginIntegrationsPanel({ metadataEnabled }: PluginIntegrationsPanelProps) {
  const summariesQuery = usePluginSummariesQuery({ enabled: metadataEnabled })
  const summaries = useMemo(
    () => (metadataEnabled ? (summariesQuery.data ?? []) : []),
    [metadataEnabled, summariesQuery.data]
  )
  const entries = useMemo(() => adaptPluginSummariesToWebUiEntries(summaries), [summaries])

  return (
    <div className="space-y-[14px]" data-plugin-integration-metadata="true">
      {metadataEnabled && summariesQuery.isPending ? (
        <section className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-4">
          <div className="type-label text-fg-faint">Installed plugins</div>
          <p className="type-caption mt-1 text-fg-dim">Loading plugin integration metadata...</p>
        </section>
      ) : null}
      {metadataEnabled && summariesQuery.isError ? (
        <section className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel p-4" role="alert">
          <div className="flex items-center gap-2 text-bad">
            <ShieldAlert aria-hidden="true" className="size-4" />
            <h2 className="type-panel-title">Plugin metadata unavailable</h2>
          </div>
          <p className="type-caption mt-1 text-fg-dim">
            Existing plugin settings remain visible, but web UI state could not be loaded.
          </p>
        </section>
      ) : null}
      {summaries.length > 0 ? (
        <section aria-labelledby="plugin-integrations-heading" className="space-y-3">
          <div className="flex items-center gap-2">
            <PlugZap aria-hidden="true" className="size-4 text-accent" />
            <h2 className="type-headline text-foreground" id="plugin-integrations-heading">
              Installed plugins
            </h2>
          </div>
          <div className="grid gap-3 xl:grid-cols-2">
            {summaries.map((summary, index) => {
              const entry = entries[index]
              return entry ? <PluginIntegrationCard key={summary.name} summary={summary} entry={entry} /> : null
            })}
          </div>
        </section>
      ) : null}
    </div>
  )
}
