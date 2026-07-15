import { useEffect, useRef, useState } from 'react'
import { StatusBanner } from '@/components/ui/StatusBanner'
import {
  resolvePluginWebUiAssetUrl,
  usePluginWebUiConfigMutation,
  usePluginWebUiConfigQuery
} from '@/features/plugins/api/plugin-web-ui'
import {
  assertPluginUiMountHandle,
  assertPluginUiRegistration,
  importPluginUiBundle
} from '@/features/plugins/web-ui/bundle-loader'
import type {
  MeshPluginUiConfigMountContext,
  MeshPluginUiMountHandle,
  MeshPluginUiToast
} from '@/features/plugins/web-ui/host-contract'
import { createMeshPluginUiHost } from '@/features/plugins/web-ui/host-surface'
import type { PluginSummaryRaw, PluginWebUiConfigSectionRaw, PluginWebUiPageRaw } from '@/lib/api/plugin-types'

type ConfigMountStatus =
  { readonly kind: 'loading' } | { readonly kind: 'mounted' } | { readonly kind: 'error'; readonly message: string }

type MutationStatus =
  | { readonly kind: 'idle' }
  | { readonly kind: 'pending' }
  | { readonly kind: 'success'; readonly message: string }
  | { readonly kind: 'error'; readonly message: string }

type PluginConfigSectionMountProps = {
  readonly pluginName: string
  readonly section: PluginWebUiConfigSectionRaw
  readonly webUi: PluginSummaryRaw['web_ui']
}

function sameOriginAssetUrl(pluginName: string, webUi: PluginSummaryRaw['web_ui'], entryScript: string): string {
  if (!webUi.asset_base_url) throw new TypeError('Plugin web UI asset base URL is unavailable')
  const assetUrl = new URL(resolvePluginWebUiAssetUrl(pluginName, webUi, entryScript), window.location.origin)
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
  const [mutationStatus, setMutationStatus] = useState<MutationStatus>({ kind: 'idle' })
  const visibleConfigQuery = usePluginWebUiConfigQuery(pluginName)
  const configMutation = usePluginWebUiConfigMutation(pluginName)
  const mutateConfigRef = useRef(configMutation.mutateAsync)
  const visibleConfigRef = useRef(visibleConfigQuery.data)
  const visibleConfigReady = Boolean(visibleConfigQuery.data)

  useEffect(() => {
    mutateConfigRef.current = configMutation.mutateAsync
  }, [configMutation.mutateAsync])

  useEffect(() => {
    visibleConfigRef.current = visibleConfigQuery.data
  }, [visibleConfigQuery.data])

  useEffect(() => {
    if (!visibleConfigReady) {
      const timeout = window.setTimeout(() => setMountStatus({ kind: 'loading' }), 0)
      return () => window.clearTimeout(timeout)
    }

    let cancelled = false
    let cleanup: (() => void) | undefined

    const loadingTimeout = window.setTimeout(() => {
      if (!cancelled) setMountStatus({ kind: 'loading' })
    }, 0)

    const showToast = (toast: MeshPluginUiToast) => {
      setMutationStatus({
        kind: toast.tone === 'error' ? 'error' : 'success',
        message: toast.description ? `${toast.title}: ${toast.description}` : toast.title
      })
    }

    const mountSection = async () => {
      const visibleConfig = visibleConfigRef.current
      if (!visibleConfig) return
      const page = sectionPage(section)
      const module = await importPluginUiBundle(sameOriginAssetUrl(pluginName, webUi, section.entry_script))
      const host = createMeshPluginUiHost({
        pluginName,
        page,
        webUi,
        visibleConfig,
        navigateTo: (path) => window.location.assign(path),
        openPluginPage: (pageId) =>
          window.location.assign(`/plugins/${encodeURIComponent(pluginName)}/${encodeURIComponent(pageId)}`),
        requestConfigMutation: async (request) => {
          setMutationStatus({ kind: 'pending' })
          try {
            const result = await mutateConfigRef.current(request)
            setMutationStatus({ kind: 'success', message: 'Plugin settings saved.' })
            return result
          } catch (error) {
            const message = errorMessage(error)
            setMutationStatus({ kind: 'error', message })
            throw error
          }
        },
        showToast
      })
      const registration = await module.registerMeshPluginUi(host)
      assertPluginUiRegistration(registration)
      const mount = registration.configSections?.[section.id]
      const element = mountRef.current

      if (!mount || !element) {
        setMountStatus({ kind: 'error', message: 'Plugin bundle did not register this config section.' })
        return
      }

      const context: MeshPluginUiConfigMountContext = { element, host, section }
      const handle = await mount(context)
      assertPluginUiMountHandle(handle)
      cleanup = unmountOnce(handle)

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
      window.clearTimeout(loadingTimeout)
      cleanup?.()
    }
  }, [pluginName, section, visibleConfigReady, webUi])

  return (
    <section
      aria-labelledby={`${pluginName}-${section.id}-config-heading`}
      className="panel-shell rounded-[var(--radius-lg)] border border-border bg-panel"
    >
      {mountStatus.kind === 'loading' ? <StatusBanner>Loading plugin config section...</StatusBanner> : null}
      {mountStatus.kind === 'error' ? (
        <StatusBanner role="alert" tone="bad">
          {mountStatus.message}
        </StatusBanner>
      ) : null}
      {mutationStatus.kind !== 'idle' ? (
        <StatusBanner
          role={mutationStatus.kind === 'error' ? 'alert' : 'status'}
          tone={mutationStatus.kind === 'error' ? 'bad' : 'muted'}
        >
          {mutationStatus.kind === 'pending' ? 'Saving plugin settings...' : mutationStatus.message}
        </StatusBanner>
      ) : null}
      <header className="border-b border-border-soft px-4 py-3">
        <div className="type-label text-fg-faint">Plugin config section</div>
        <h4 className="type-panel-title mt-1 text-foreground" id={`${pluginName}-${section.id}-config-heading`}>
          {section.title}
        </h4>
      </header>
      <section ref={mountRef} className="min-h-[72px] p-4" aria-label={`${section.title} plugin config host`} />
      {mountStatus.kind === 'mounted' ? (
        <div className="sr-only" aria-live="polite">
          Plugin config section mounted
        </div>
      ) : null}
    </section>
  )
}
