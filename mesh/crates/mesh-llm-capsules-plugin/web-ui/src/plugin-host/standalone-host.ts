// A `MeshPluginUiHost` for running the page OUTSIDE the mesh-llm console: the
// unit tests and the fixture-mode dev harness (`dev/`). It is never part of the
// built bundle; inside the console the host object comes from mesh-llm.
//
// `fetchPlugin` resolves a plugin-relative path to
// `/api/plugins/<plugin>/<path>` under the same rules as the console's own
// `pluginScopedApiUrl` (mesh-llm-ui `features/plugins/web-ui/host-surface.ts`):
// no origin, fragment, backslash, control character, or dot segment. So a
// request that works here is one the real host would also send.
import type { MeshPluginUiHost } from '@/plugin-host/host-contract'

export const PLUGIN_NAME = 'capsules'

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
  return `/api/plugins/${encodeURIComponent(pluginName)}${suffix}${search}`
}

// The console's default `@theme` token values, in the `host.appearance.tokens` shape.
const DEFAULT_TOKENS: MeshPluginUiHost['appearance']['tokens'] = {
  background: 'oklch(0.985 0.003 80)',
  foreground: 'oklch(0.22 0.02 250)',
  panel: 'oklch(0.975 0.004 80)',
  panelStrong: 'oklch(0.955 0.005 80)',
  border: 'oklch(0.88 0.005 80)',
  borderSoft: 'oklch(0.93 0.005 80)',
  accent: 'oklch(0.62 0.14 220)',
  accentInk: 'oklch(0.98 0.01 220)',
  good: 'oklch(0.58 0.13 150)',
  warn: 'oklch(0.62 0.14 55)',
  bad: 'oklch(0.62 0.16 28)',
  radius: '6px',
  radiusLarge: '10px'
}

export type StandaloneHostOptions = {
  /** Called for every host navigation (`/chat`, `/logs?...`). */
  navigate?: (path: string) => void
  settings?: Record<string, unknown>
}

export function createStandaloneHost(options: StandaloneHostOptions = {}): MeshPluginUiHost {
  let snapshot: Readonly<Record<string, unknown>> = {}
  const subscribers = new Set<(next: Readonly<Record<string, unknown>>) => void>()
  const navigate = options.navigate ?? (() => undefined)
  const page = { id: 'evidence', label: 'Evidence', route: 'evidence' }
  const visible = { plugin: PLUGIN_NAME, settings: options.settings ?? {} }
  return {
    plugin: { name: PLUGIN_NAME },
    page,
    webUi: {
      state: 'ready',
      declared: true,
      enabled: true,
      available: true,
      pages: [{ ...page, bundle_id: 'main', entry_script: 'register-mesh-plugin-ui.js' }],
      asset_base_url: `/api/plugins/${PLUGIN_NAME}/web-ui/assets/`
    },
    appearance: { theme: 'default', accent: 'default', density: 'default', panelStyle: 'default', tokens: DEFAULT_TOKENS },
    network: {
      // Resolve `fetch` at call time so tests that stub the global see it.
      fetchPlugin: (path, init) => fetch(pluginScopedApiUrl(PLUGIN_NAME, path), init),
      json: async (path, init) => {
        const response = await fetch(pluginScopedApiUrl(PLUGIN_NAME, path), init)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json() as Promise<unknown>
      }
    },
    config: {
      visible,
      requestMutation: async () => visible
    },
    navigation: {
      navigateTo: navigate,
      openPluginPage: (pageId) => navigate(`/plugins/${PLUGIN_NAME}/${pageId}`)
    },
    notifications: { show: () => undefined },
    state: {
      getSnapshot: () => snapshot,
      update: (patch) => {
        snapshot = { ...snapshot, ...patch }
        for (const subscriber of subscribers) subscriber(snapshot)
        return snapshot
      },
      subscribe: (subscriber) => {
        subscribers.add(subscriber)
        return () => subscribers.delete(subscriber)
      }
    }
  }
}
