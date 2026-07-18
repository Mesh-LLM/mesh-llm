import { env } from '@/lib/env'
import type {
  PluginWebUiConfigMutationRequest,
  PluginWebUiPageRaw,
  PluginWebUiStateRaw,
  PluginWebUiVisibleConfigRaw
} from '@/lib/api/plugin-types'
import type {
  MeshPluginUiAppearance,
  MeshPluginUiHost,
  MeshPluginUiStateSnapshot,
  MeshPluginUiStateSubscriber,
  MeshPluginUiToast
} from '@/features/plugins/web-ui/host-contract'

type CreateMeshPluginUiHostInput = {
  readonly pluginName: string
  readonly page: PluginWebUiPageRaw
  readonly webUi: PluginWebUiStateRaw
  readonly visibleConfig: PluginWebUiVisibleConfigRaw
  readonly navigateTo: (path: string) => void
  readonly openPluginPage: (pageId: string) => void
  readonly requestConfigMutation: (request: PluginWebUiConfigMutationRequest) => Promise<PluginWebUiVisibleConfigRaw>
  readonly showToast?: (toast: MeshPluginUiToast) => void
}

function documentAppearance(): MeshPluginUiAppearance {
  const root = document.documentElement
  const style = window.getComputedStyle(root)

  return {
    theme: root.dataset.theme ?? 'dark',
    accent: root.dataset.accent ?? 'blue',
    density: root.dataset.density ?? 'normal',
    panelStyle: root.dataset.panelStyle ?? 'soft',
    tokens: {
      background: style.getPropertyValue('--color-background').trim(),
      foreground: style.getPropertyValue('--color-foreground').trim(),
      panel: style.getPropertyValue('--color-panel').trim(),
      panelStrong: style.getPropertyValue('--color-panel-strong').trim(),
      border: style.getPropertyValue('--color-border').trim(),
      borderSoft: style.getPropertyValue('--color-border-soft').trim(),
      accent: style.getPropertyValue('--color-accent').trim(),
      accentInk: style.getPropertyValue('--color-accent-ink').trim(),
      good: style.getPropertyValue('--color-good').trim(),
      warn: style.getPropertyValue('--color-warn').trim(),
      bad: style.getPropertyValue('--color-bad').trim(),
      radius: style.getPropertyValue('--radius').trim(),
      radiusLarge: style.getPropertyValue('--radius-lg').trim()
    }
  }
}

function hasControlCharacter(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index)
    if (code <= 0x1f || code === 0x7f) return true
  }
  return false
}

export function pluginScopedApiUrl(pluginName: string, path: string): string {
  if (path.includes('\\') || path.includes('#') || path.includes('://') || hasControlCharacter(path)) {
    throw new TypeError('Plugin API path must be a relative path without a fragment or origin')
  }
  const queryIndex = path.indexOf('?')
  const pathname = queryIndex === -1 ? path : path.slice(0, queryIndex)
  const query = queryIndex === -1 ? '' : path.slice(queryIndex + 1)
  const rawSegments = pathname.split('/').filter(Boolean)
  if (rawSegments.some((segment) => segment === '.' || segment === '..')) {
    throw new TypeError('Plugin API path cannot contain dot segments')
  }
  const pathSegments = rawSegments.map(encodeURIComponent).join('/')
  const suffix = pathSegments ? `/${pathSegments}` : ''
  const search = query ? `?${query}` : ''
  return `${env.managementApiUrl}/api/plugins/${encodeURIComponent(pluginName)}${suffix}${search}`
}

function createPluginUiStateStore() {
  let snapshot: MeshPluginUiStateSnapshot = {}
  const subscribers = new Set<MeshPluginUiStateSubscriber>()

  return {
    getSnapshot: () => snapshot,
    update: (patch: MeshPluginUiStateSnapshot) => {
      snapshot = { ...snapshot, ...patch }
      subscribers.forEach((subscriber) => {
        subscriber(snapshot)
      })
      return snapshot
    },
    subscribe: (subscriber: MeshPluginUiStateSubscriber) => {
      subscribers.add(subscriber)
      return () => subscribers.delete(subscriber)
    }
  }
}

export function createMeshPluginUiHost({
  pluginName,
  page,
  webUi,
  visibleConfig,
  navigateTo,
  openPluginPage,
  requestConfigMutation,
  showToast
}: CreateMeshPluginUiHostInput): MeshPluginUiHost {
  const state = createPluginUiStateStore()

  return {
    plugin: { name: pluginName },
    page: { id: page.id, label: page.label, route: page.route },
    webUi,
    appearance: documentAppearance(),
    network: {
      fetchPlugin: async (path, init) => fetch(pluginScopedApiUrl(pluginName, path), init),
      json: async (path, init) => {
        const response = await fetch(pluginScopedApiUrl(pluginName, path), init)
        if (!response.ok) {
          const detail = await response.text()
          throw new Error(detail || `Plugin API request failed with HTTP ${response.status}`)
        }
        return response.json()
      }
    },
    config: {
      visible: visibleConfig,
      requestMutation: requestConfigMutation
    },
    navigation: {
      navigateTo,
      openPluginPage
    },
    notifications: {
      show: showToast ?? (() => undefined)
    },
    state
  }
}
