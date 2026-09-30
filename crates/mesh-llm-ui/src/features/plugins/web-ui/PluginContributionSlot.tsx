import { createContext, useContext, useEffect, useMemo, useRef, type ReactNode } from 'react'
import {
  buildPluginWebUiContributionItems,
  resolvePluginWebUiAssetUrl,
  usePluginWebUiConfigMutation,
  usePluginWebUiConfigQuery,
  type PluginWebUiContributionItem
} from '@/features/plugins/api/plugin-web-ui'
import {
  assertPluginUiMountHandle,
  assertPluginUiRegistration,
  importPluginUiBundle
} from '@/features/plugins/web-ui/bundle-loader'
import type { MeshPluginUiContributionSubject } from '@/features/plugins/web-ui/host-contract'
import { createMeshPluginUiHost } from '@/features/plugins/web-ui/host-surface'
import type { PluginSummaryRaw, PluginWebUiContributionRaw, PluginWebUiPageRaw } from '@/lib/api/plugin-types'

const PluginContributionsContext = createContext<readonly PluginSummaryRaw[]>([])

/** Makes the plugin summaries the shell already loaded available to the
 *  contribution slots below it. Without this provider every slot is empty. */
export function PluginContributionsProvider({
  summaries,
  children
}: {
  readonly summaries: readonly PluginSummaryRaw[]
  readonly children: ReactNode
}) {
  return <PluginContributionsContext.Provider value={summaries}>{children}</PluginContributionsContext.Provider>
}

function sameOriginAssetUrl(item: PluginWebUiContributionItem): string {
  if (!item.webUi.asset_base_url) throw new TypeError('Plugin web UI asset base URL is unavailable')
  const assetUrl = new URL(
    resolvePluginWebUiAssetUrl(item.pluginName, item.webUi, item.contribution.entry_script),
    window.location.origin
  )
  if (assetUrl.origin !== window.location.origin) throw new TypeError('Plugin web UI asset URL must be same-origin')
  return assetUrl.href
}

function contributionPage(contribution: PluginWebUiContributionRaw): PluginWebUiPageRaw {
  return {
    id: `contribution:${contribution.id}`,
    label: contribution.label,
    route: contribution.slot,
    bundle_id: contribution.bundle_id,
    entry_script: contribution.entry_script
  }
}

function PluginContributionMount({
  item,
  subject
}: {
  readonly item: PluginWebUiContributionItem
  readonly subject: MeshPluginUiContributionSubject
}) {
  const { pluginName } = item
  const itemRef = useRef(item)
  const mountRef = useRef<HTMLDivElement | null>(null)
  const visibleConfigQuery = usePluginWebUiConfigQuery(pluginName)
  const configMutation = usePluginWebUiConfigMutation(pluginName)
  const mutateConfigRef = useRef(configMutation.mutateAsync)
  const visibleConfigRef = useRef(visibleConfigQuery.data)
  const subjectRef = useRef(subject)
  const visibleConfigReady = Boolean(visibleConfigQuery.data)
  // Remount only when an id changes, not on every render's new object.
  const subjectKey = JSON.stringify(subject)
  // The summaries refetch in the background; remount only when what is
  // loaded changes.
  const itemKey = JSON.stringify([item.contribution, item.webUi.asset_base_url])

  useEffect(() => {
    itemRef.current = item
    mutateConfigRef.current = configMutation.mutateAsync
    visibleConfigRef.current = visibleConfigQuery.data
    subjectRef.current = subject
  })

  useEffect(() => {
    if (!visibleConfigReady) return
    let cancelled = false
    let cleanup: (() => void) | undefined

    const mountContribution = async () => {
      const current = itemRef.current
      const { contribution } = current
      const visibleConfig = visibleConfigRef.current
      const element = mountRef.current
      if (!visibleConfig || !element) return
      const module = await importPluginUiBundle(sameOriginAssetUrl(current))
      const host = createMeshPluginUiHost({
        pluginName,
        page: contributionPage(contribution),
        webUi: current.webUi,
        visibleConfig,
        navigateTo: (path) => window.location.assign(path),
        openPluginPage: (pageId) =>
          window.location.assign(`/plugins/${encodeURIComponent(pluginName)}/${encodeURIComponent(pageId)}`),
        requestConfigMutation: (request) => mutateConfigRef.current(request)
      })
      const registration = await module.registerMeshPluginUi(host)
      assertPluginUiRegistration(registration)
      const mount = registration.contributions?.[contribution.id]
      if (!mount || cancelled) return
      const handle = await mount({ element, host, contribution, subject: subjectRef.current })
      assertPluginUiMountHandle(handle)
      let mounted = true
      cleanup = () => {
        if (!mounted) return
        mounted = false
        handle.unmount()
      }
      if (cancelled) cleanup()
    }

    // A contribution that fails to load leaves its slot empty; the host's own
    // row stays as it was.
    void mountContribution().catch(() => undefined)

    return () => {
      cancelled = true
      cleanup?.()
    }
  }, [itemKey, pluginName, subjectKey, visibleConfigReady])

  return (
    <div
      ref={mountRef}
      aria-label={item.contribution.label}
      className="min-w-0"
      data-plugin-contribution={`${pluginName}:${item.contribution.id}`}
      role="group"
    />
  )
}

/** Mounts every ready plugin's contribution for this slot, passing the host
 *  ids in `subject`. Renders nothing when no plugin contributes to it. */
export function PluginContributionSlot({
  subject,
  className
}: {
  readonly subject: MeshPluginUiContributionSubject
  readonly className?: string
}) {
  const summaries = useContext(PluginContributionsContext)
  const items = useMemo(() => buildPluginWebUiContributionItems(summaries, subject.slot), [summaries, subject.slot])
  if (items.length === 0) return null

  return (
    <div className={className ?? 'flex min-w-0 flex-wrap items-center gap-2'}>
      {items.map((item) => (
        <PluginContributionMount key={`${item.pluginName}:${item.contribution.id}`} item={item} subject={subject} />
      ))}
    </div>
  )
}
