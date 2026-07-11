import { useEffect, useRef, useState } from 'react'
import { pluginWebUiAssetUrl } from '@/features/plugins/api/plugin-web-ui'
import { importPluginUiBundle } from '@/features/plugins/web-ui/bundle-loader'
import type { MeshPluginUiConfigMountContext, MeshPluginUiMountHandle } from '@/features/plugins/web-ui/host-contract'
import { createMeshPluginUiHost } from '@/features/plugins/web-ui/host-surface'
import type { PluginSummaryRaw, PluginWebUiConfigSectionRaw, PluginWebUiPageRaw } from '@/lib/api/plugin-types'

type ConfigMountStatus =
  { readonly kind: 'loading' } | { readonly kind: 'mounted' } | { readonly kind: 'error'; readonly message: string }

type PluginConfigSectionMountProps = {
  readonly pluginName: string
  readonly section: PluginWebUiConfigSectionRaw
  readonly webUi: PluginSummaryRaw['web_ui']
}

function sameOriginAssetUrl(pluginName: string, entryScript: string): string {
  const assetUrl = new URL(pluginWebUiAssetUrl(pluginName, entryScript), window.location.origin)
  if (assetUrl.origin !== window.location.origin) throw new TypeError('Plugin web UI asset URL must be same-origin')
  return assetUrl.href
}

function unmountOnce(handle: MeshPluginUiMountHandle): () => void {
  let mounted = true
  return () => {
    if (!mounted) return
    mounted = false
    handle.unmount()
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Plugin config section mount failed'
}

function sectionPage(section: PluginWebUiConfigSectionRaw): PluginWebUiPageRaw {
  return {
    id: `config:${section.id}`,
    label: section.title,
    route: 'integrations',
    bundle_id: section.bundle_id,
    entry_script: section.entry_script
  }
}

export function PluginConfigSectionMount({ pluginName, section, webUi }: PluginConfigSectionMountProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const [mountStatus, setMountStatus] = useState<ConfigMountStatus>({ kind: 'loading' })

  useEffect(() => {
    let cancelled = false
    let cleanup: (() => void) | undefined

    const mountSection = async () => {
      const page = sectionPage(section)
      const module = await importPluginUiBundle(sameOriginAssetUrl(pluginName, section.entry_script))
      const host = createMeshPluginUiHost({
        pluginName,
        page,
        webUi,
        navigateTo: (path) => window.location.assign(path),
        openPluginPage: (pageId) =>
          window.location.assign(`/plugins/${encodeURIComponent(pluginName)}/${encodeURIComponent(pageId)}`)
      })
      const registration = await module.registerMeshPluginUi(host)
      const mount = registration.configSections?.[section.id]
      const element = mountRef.current

      if (!mount || !element) {
        setMountStatus({ kind: 'error', message: 'Plugin bundle did not register this config section.' })
        return
      }

      const context: MeshPluginUiConfigMountContext = { element, host, section }
      cleanup = unmountOnce(await mount(context))

      if (cancelled) {
        cleanup()
        return
      }

      setMountStatus({ kind: 'mounted' })
    }

    void mountSection().catch((error: unknown) => {
      if (!cancelled) setMountStatus({ kind: 'error', message: errorMessage(error) })
    })

    return () => {
      cancelled = true
      cleanup?.()
    }
  }, [pluginName, section, webUi])

  return (
    <section
      aria-labelledby={`${pluginName}-${section.id}-config-heading`}
      className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel"
    >
      <header className="border-b border-border-soft px-4 py-3">
        <div className="type-label text-fg-faint">Plugin config section</div>
        <h4 className="type-panel-title mt-1 text-foreground" id={`${pluginName}-${section.id}-config-heading`}>
          {section.title}
        </h4>
      </header>
      {mountStatus.kind === 'loading' ? (
        <div className="type-caption border-b border-border-soft px-4 py-2 text-fg-faint">
          Loading plugin config section...
        </div>
      ) : null}
      {mountStatus.kind === 'error' ? (
        <div className="type-caption border-b border-border-soft px-4 py-2 text-bad" role="alert">
          {mountStatus.message}
        </div>
      ) : null}
      <section ref={mountRef} className="min-h-[72px] p-4" aria-label={`${section.title} plugin config host`} />
      {mountStatus.kind === 'mounted' ? (
        <div className="sr-only" aria-live="polite">
          Plugin config section mounted
        </div>
      ) : null}
    </section>
  )
}
