import { useEffect, useMemo, useRef, useState } from 'react'
import { useParams, useRouter } from '@tanstack/react-router'
import { Boxes, ShieldAlert } from 'lucide-react'
import { InfoBanner } from '@/components/ui/InfoBanner'
import {
  resolvePluginWebUiAssetUrl,
  usePluginWebUiConfigMutation,
  usePluginWebUiConfigQuery,
  usePluginWebUiQuery
} from '@/features/plugins/api/plugin-web-ui'
import {
  assertPluginUiMountHandle,
  assertPluginUiRegistration,
  importPluginUiBundle
} from '@/features/plugins/web-ui/bundle-loader'
import { createMeshPluginUiHost } from '@/features/plugins/web-ui/host-surface'
import type { PluginWebUiPageRaw, PluginWebUiStateRaw } from '@/lib/api/plugin-types'
import type { MeshPluginUiMountHandle } from '@/features/plugins/web-ui/host-contract'

type PluginRouteEligibility =
  | { readonly kind: 'ready'; readonly page: PluginWebUiPageRaw }
  | {
      readonly kind: 'fallback'
      readonly title: string
      readonly detail: string
      readonly tone: 'warn' | 'bad' | 'muted'
    }

type MountStatus =
  | { readonly kind: 'idle' }
  | { readonly kind: 'loading' }
  | { readonly kind: 'mounted' }
  | { readonly kind: 'error'; readonly message: string }

function fallbackForState(webUi: PluginWebUiStateRaw): PluginRouteEligibility {
  switch (webUi.state) {
    case 'none':
      return {
        kind: 'fallback',
        title: 'Plugin web UI is not declared',
        detail: 'This plugin does not publish host-projected console pages.',
        tone: 'muted'
      }
    case 'disabled':
      return {
        kind: 'fallback',
        title: 'Plugin web UI is disabled',
        detail: webUi.unavailable_reason ?? 'The plugin remains available, but its console projection is turned off.',
        tone: 'warn'
      }
    case 'invalid':
      return {
        kind: 'fallback',
        title: 'Plugin web UI bundle is invalid',
        detail: webUi.unavailable_reason ?? 'The host rejected the declared bundle metadata or assets.',
        tone: 'bad'
      }
    case 'plugin_not_running':
      return {
        kind: 'fallback',
        title: 'Plugin is not running',
        detail: webUi.unavailable_reason ?? 'The UI declaration is known, but the plugin process is unavailable.',
        tone: 'warn'
      }
    case 'ready':
      return {
        kind: 'fallback',
        title: 'Plugin page is not declared',
        detail: 'The requested page id is not part of the ready plugin web UI declaration.',
        tone: 'muted'
      }
    default:
      return assertNever(webUi.state)
  }
}

function assertNever(value: never): never {
  throw new TypeError(`Unhandled plugin web UI state: ${String(value)}`)
}

function resolvePluginRouteEligibility(
  webUi: PluginWebUiStateRaw | undefined,
  pageId: string
): PluginRouteEligibility | null {
  if (!webUi) return null
  if (webUi.state !== 'ready') return fallbackForState(webUi)
  if (!webUi.declared || !webUi.enabled || !webUi.available) return fallbackForState(webUi)
  if (!webUi.asset_base_url) {
    return {
      kind: 'fallback',
      title: 'Plugin web UI asset route is unavailable',
      detail: 'The host marked this plugin ready without a bundle asset base URL.',
      tone: 'bad'
    }
  }
  const page = webUi.pages?.find((candidate) => candidate.id === pageId)
  return page ? { kind: 'ready', page } : fallbackForState(webUi)
}

function sameOriginAssetUrl(pluginName: string, webUi: PluginWebUiStateRaw, entryScript: string): string {
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
  return error instanceof Error ? error.message : 'Plugin bundle import failed'
}

function PluginRouteFallback({ detail, title, tone }: Extract<PluginRouteEligibility, { kind: 'fallback' }>) {
  const Icon = tone === 'bad' ? ShieldAlert : Boxes
  const toneClass = tone === 'bad' ? 'text-bad' : tone === 'warn' ? 'text-warn' : 'text-fg-faint'

  return (
    <InfoBanner
      className="mt-0"
      description={detail}
      leadingIcon={<Icon className="size-4" aria-hidden="true" />}
      leadingIconClassName={toneClass}
      title={title}
      titleId="plugin-web-ui-fallback-title"
      titleLevel="h1"
    />
  )
}

export function PluginWebUiRoutePage() {
  const { pluginName, pageId } = useParams({ from: '/plugins/$pluginName/$pageId' })
  const router = useRouter()
  const metadataQuery = usePluginWebUiQuery(pluginName)
  const visibleConfigQuery = usePluginWebUiConfigQuery(pluginName, { enabled: metadataQuery.data?.state === 'ready' })
  const configMutation = usePluginWebUiConfigMutation(pluginName)
  const mutateConfigRef = useRef(configMutation.mutateAsync)
  const mountRef = useRef<HTMLDivElement | null>(null)
  const [mountStatus, setMountStatus] = useState<MountStatus>({ kind: 'idle' })
  const [notification, setNotification] = useState<string | null>(null)
  const eligibility = useMemo(
    () => resolvePluginRouteEligibility(metadataQuery.data, pageId),
    [metadataQuery.data, pageId]
  )

  useEffect(() => {
    mutateConfigRef.current = configMutation.mutateAsync
  }, [configMutation.mutateAsync])

  useEffect(() => {
    if (eligibility?.kind !== 'ready' || !metadataQuery.data || !visibleConfigQuery.data) {
      const timeout = window.setTimeout(() => setMountStatus({ kind: 'idle' }), 0)
      return () => window.clearTimeout(timeout)
    }

    let cancelled = false
    let cleanup: (() => void) | undefined

    const loadingTimeout = window.setTimeout(() => {
      if (!cancelled) setMountStatus({ kind: 'loading' })
    }, 0)

    const mountPluginPage = async () => {
      const assetUrl = sameOriginAssetUrl(pluginName, metadataQuery.data, eligibility.page.entry_script)
      const module = await importPluginUiBundle(assetUrl)
      const host = createMeshPluginUiHost({
        pluginName,
        page: eligibility.page,
        webUi: metadataQuery.data,
        visibleConfig: visibleConfigQuery.data,
        navigateTo: (path) => void router.navigate({ to: path }),
        openPluginPage: (nextPageId) =>
          void router.navigate({ to: '/plugins/$pluginName/$pageId', params: { pluginName, pageId: nextPageId } }),
        requestConfigMutation: (request) => mutateConfigRef.current(request),
        showToast: (toast) => setNotification(toast.description ? `${toast.title}: ${toast.description}` : toast.title)
      })
      const registration = await module.registerMeshPluginUi(host)
      assertPluginUiRegistration(registration)
      const mountPage = registration.pages[eligibility.page.id]
      const element = mountRef.current

      if (!mountPage || !element) {
        setMountStatus({ kind: 'error', message: 'Plugin bundle did not register the requested page.' })
        return
      }

      const handle = await mountPage({ element, host, page: eligibility.page })
      assertPluginUiMountHandle(handle)
      cleanup = unmountOnce(handle)

      if (cancelled) {
        cleanup()
        return
      }

      setMountStatus({ kind: 'mounted' })
    }

    void mountPluginPage().catch((error: unknown) => {
      if (!cancelled) setMountStatus({ kind: 'error', message: errorMessage(error) })
    })

    return () => {
      cancelled = true
      window.clearTimeout(loadingTimeout)
      cleanup?.()
    }
  }, [eligibility, metadataQuery.data, pluginName, router, visibleConfigQuery.data])

  if (metadataQuery.isPending) {
    return (
      <PluginRouteFallback
        kind="fallback"
        title="Loading plugin web UI"
        detail="Fetching plugin UI metadata."
        tone="muted"
      />
    )
  }

  if (metadataQuery.isError) {
    return (
      <PluginRouteFallback
        kind="fallback"
        title="Plugin web UI metadata unavailable"
        detail="The host could not fetch this plugin web UI declaration."
        tone="bad"
      />
    )
  }

  if (!eligibility) return null
  if (eligibility.kind === 'fallback') return <PluginRouteFallback {...eligibility} />

  return (
    <section className="panel-shell flex min-h-[24rem] flex-col rounded-[var(--radius-lg)] border border-border bg-panel">
      <header className="border-b border-border-soft px-5 py-4">
        <div className="type-label text-fg-faint">Plugin page</div>
        <h1 className="type-headline mt-1 text-foreground">{eligibility.page.label}</h1>
        <p className="type-caption mt-1 text-fg-dim">
          Mounted from <span className="font-mono text-foreground">{pluginName}</span> page{' '}
          <span className="font-mono text-foreground">{eligibility.page.id}</span>.
        </p>
      </header>
      {mountStatus.kind === 'loading' ? (
        <div className="border-b border-border-soft px-5 py-2 text-[length:var(--density-type-caption)] text-fg-faint">
          Loading plugin bundle...
        </div>
      ) : null}
      {mountStatus.kind === 'error' ? (
        <div
          className="border-b border-border-soft px-5 py-2 text-[length:var(--density-type-caption)] text-bad"
          role="alert"
        >
          {mountStatus.message}
        </div>
      ) : null}
      {notification ? (
        <div
          className="border-b border-border-soft px-5 py-2 text-[length:var(--density-type-caption)] text-foreground"
          role="status"
        >
          {notification}
        </div>
      ) : null}
      <section ref={mountRef} className="min-h-0 flex-1 p-5" aria-label={`${eligibility.page.label} plugin host`} />
      {mountStatus.kind === 'mounted' ? (
        <div className="sr-only" aria-live="polite">
          Plugin page mounted
        </div>
      ) : null}
    </section>
  )
}
