// The mesh-llm console host this page is mounted into.
//
// The page runs inside mesh-llm's plugin web UI projection (docs/plugins/README.md
// "Plugin Web UI Projection Contract" in Mesh-LLM/mesh-llm): the console imports
// `register-mesh-plugin-ui.js` from the installed package and hands every mount a
// `MeshPluginUiHost`. The data clients in `features/capsules/api/` are plain
// functions, not hooks, so the mounted host is held here rather than in React
// context. One page mounts at a time; `mountEvidencePage` sets it and clears it
// on unmount.
import type { MeshPluginUiHost } from '@/plugin-host/host-contract'

let mountedHost: MeshPluginUiHost | null = null

export function setPluginHost(host: MeshPluginUiHost | null): void {
  mountedHost = host
}

/** The host the page is mounted into. Throws when read outside a mount: a data
 *  call with no host has nowhere honest to go, and must not fall back to a
 *  guessed URL. */
export function pluginHost(): MeshPluginUiHost {
  if (!mountedHost) throw new Error('capsules page: no mesh-llm host is mounted')
  return mountedHost
}

/** GET a plugin-relative JSON route (`host.network.fetchPlugin`), keeping the
 *  HTTP status on failure so callers can tell "route not served" (404 / 503)
 *  from any other error. */
export class PluginRouteError extends Error {
  status: number
  constructor(status: number, path: string) {
    super(`plugin route failed: HTTP ${status} (${path})`)
    this.status = status
  }
}

export async function getPluginJson<T>(path: string): Promise<T> {
  const response = await pluginHost().network.fetchPlugin(path)
  if (!response.ok) throw new PluginRouteError(response.status, path)
  return (await response.json()) as T
}

/** Navigate the console (host-owned routes such as `/chat` or `/logs`). */
export function navigateHost(path: string): void {
  pluginHost().navigation.navigateTo(path)
}
