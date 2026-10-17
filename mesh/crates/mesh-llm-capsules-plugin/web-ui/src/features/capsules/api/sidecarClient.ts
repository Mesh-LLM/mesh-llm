// Fetches the accountability pane data from this plugin's own `panes/*`
// routes, through the console host's plugin-scoped fetch
// (`host.network.fetchPlugin`).
//
// Ported from the fork tab, which read the same JSON through a fork-only host
// route (`/api/capsules/panes/*`) that forwarded to the Python sidecar. A
// plugin page may only reach its own plugin's routes, so the pane reader is
// the plugin's to serve (`web-ui/DATA-ROUTES.md`); the response shapes are
// unchanged.
import type { PaneAJson, PaneBJson, PaneCDrilldownJson, PaneCListJson } from '@/features/capsules/api/sidecarTypes'
import {
  attachPushedHalfRecomputeToPaneB,
  attachPushedHalfRecomputeToPaneC
} from '@/features/capsules/lib/pushed-half-recompute'
import { PluginRouteError, getPluginJson } from '@/plugin-host/host'

/** A non-2xx from a pane route, carrying the HTTP status so callers can tell
 *  "this plugin build serves no panes" (404) and "its pane service isn't
 *  running" (503) apart from any other failure. */
export { PluginRouteError as PaneFetchError }

export const PANE_ROUTES = {
  paneA: 'http/panes/pane-a',
  paneB: 'http/panes/pane-b',
  paneC: 'http/panes/pane-c'
} as const

export function fetchPaneA(): Promise<PaneAJson> {
  return getPluginJson<PaneAJson>(PANE_ROUTES.paneA)
}

/** Resolves only after every pushed half's `capsule_id` has been recomputed
 *  in this browser (`pushed-half-recompute.ts`), so no reader of this query
 *  ever sees a pushed half without its verdict. */
export async function fetchPaneB(): Promise<PaneBJson> {
  return attachPushedHalfRecomputeToPaneB(await getPluginJson<PaneBJson>(PANE_ROUTES.paneB))
}

export async function fetchPaneCList(opts?: { limit?: number; afterSeq?: number }): Promise<PaneCListJson> {
  const params = new URLSearchParams()
  if (opts?.limit != null) params.set('limit', String(opts.limit))
  if (opts?.afterSeq != null) params.set('after_seq', String(opts.afterSeq))
  const query = params.toString()
  // Same recompute-before-resolve rule as `fetchPaneB`.
  return attachPushedHalfRecomputeToPaneC(
    await getPluginJson<PaneCListJson>(`${PANE_ROUTES.paneC}${query ? `?${query}` : ''}`)
  )
}

export function fetchPaneCDrilldown(exchangeId: string): Promise<PaneCDrilldownJson> {
  return getPluginJson<PaneCDrilldownJson>(`${PANE_ROUTES.paneC}?exchange_id=${encodeURIComponent(exchangeId)}`)
}

