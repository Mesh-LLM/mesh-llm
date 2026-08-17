import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type {
  PluginSummaryRaw,
  PluginWebUiConfigMutationRequest,
  PluginWebUiPageRaw,
  PluginWebUiStateRaw,
  PluginWebUiVisibleConfigRaw
} from '@/lib/api/plugin-types'
import { ApiError } from '@/lib/api/errors'
import { env } from '@/lib/env'
import { pluginKeys, statusKeys } from '@/lib/query/query-keys'

export type PluginWebUiEntry = {
  readonly pluginName: string
  readonly state: PluginWebUiStateRaw['state']
  readonly declared: boolean
  readonly enabled: boolean
  readonly available: boolean
  readonly navigationEligible: boolean
  readonly unavailableReason?: string
  readonly pages: readonly PluginWebUiPageRaw[]
  readonly configSections: NonNullable<PluginWebUiStateRaw['config_sections']>
  readonly assetBaseUrl?: string
}

export type PluginWebUiNavItem = {
  readonly pluginName: string
  readonly pageId: string
  readonly label: string
  readonly route: string
}

function assertNever(value: never): never {
  throw new TypeError(`Unhandled plugin web UI state: ${String(value)}`)
}

function pluginWebUiEndpoint(pluginName: string): string {
  return `${env.managementApiUrl}/api/plugins/${encodeURIComponent(pluginName)}/web-ui`
}

function pluginWebUiEnabledEndpoint(pluginName: string): string {
  return `${pluginWebUiEndpoint(pluginName)}/enabled`
}

function pluginWebUiConfigEndpoint(pluginName: string): string {
  return `${pluginWebUiEndpoint(pluginName)}/config`
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text()
    throw new ApiError(response.status, body, `HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function pluginWebUiAssetUrl(pluginName: string, assetPath: string): string {
  const encodedAssetPath = assetPath.split('/').filter(Boolean).map(encodeURIComponent).join('/')
  return `${pluginWebUiEndpoint(pluginName)}/assets/${encodedAssetPath}`
}

export function resolvePluginWebUiAssetUrl(
  pluginName: string,
  webUi: Pick<PluginWebUiStateRaw, 'asset_base_url'>,
  assetPath: string
): string {
  const encodedAssetPath = assetPath.split('/').filter(Boolean).map(encodeURIComponent).join('/')
  if (webUi.asset_base_url) {
    const baseUrl = webUi.asset_base_url.endsWith('/') ? webUi.asset_base_url : `${webUi.asset_base_url}/`
    return new URL(encodedAssetPath, new URL(baseUrl, window.location.origin)).toString()
  }
  return new URL(pluginWebUiAssetUrl(pluginName, assetPath), window.location.origin).toString()
}

export async function fetchPluginSummaries(): Promise<readonly PluginSummaryRaw[]> {
  const response = await fetch(`${env.managementApiUrl}/api/plugins`)
  return parseJsonResponse<readonly PluginSummaryRaw[]>(response)
}

export async function fetchPluginWebUi(pluginName: string): Promise<PluginWebUiStateRaw> {
  const response = await fetch(pluginWebUiEndpoint(pluginName))
  return parseJsonResponse<PluginWebUiStateRaw>(response)
}

export async function setPluginWebUiEnabled(pluginName: string, enabled: boolean): Promise<PluginWebUiStateRaw> {
  const response = await fetch(pluginWebUiEnabledEndpoint(pluginName), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled })
  })
  return parseJsonResponse<PluginWebUiStateRaw>(response)
}

export async function fetchPluginWebUiConfig(pluginName: string): Promise<PluginWebUiVisibleConfigRaw> {
  const response = await fetch(pluginWebUiConfigEndpoint(pluginName))
  return parseJsonResponse<PluginWebUiVisibleConfigRaw>(response)
}

export async function requestPluginWebUiConfigMutation(
  pluginName: string,
  request: PluginWebUiConfigMutationRequest
): Promise<PluginWebUiVisibleConfigRaw> {
  const response = await fetch(pluginWebUiConfigEndpoint(pluginName), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, plugin: request.plugin ?? pluginName })
  })
  return parseJsonResponse<PluginWebUiVisibleConfigRaw>(response)
}

export function isPluginWebUiNavigationEligible(webUi: PluginWebUiStateRaw): boolean {
  switch (webUi.state) {
    case 'ready':
      return webUi.declared && webUi.enabled && webUi.available && (webUi.pages?.length ?? 0) > 0
    case 'none':
    case 'disabled':
    case 'invalid':
    case 'plugin_not_running':
      return false
    default:
      return assertNever(webUi.state)
  }
}

export function adaptPluginSummaryToWebUiEntry(summary: PluginSummaryRaw): PluginWebUiEntry {
  return {
    pluginName: summary.name,
    state: summary.web_ui.state,
    declared: summary.web_ui.declared,
    enabled: summary.web_ui.enabled,
    available: summary.web_ui.available,
    navigationEligible: isPluginWebUiNavigationEligible(summary.web_ui),
    unavailableReason: summary.web_ui.unavailable_reason,
    pages: summary.web_ui.pages ?? [],
    configSections: summary.web_ui.config_sections ?? [],
    assetBaseUrl: summary.web_ui.asset_base_url
  }
}

export function adaptPluginSummariesToWebUiEntries(
  summaries: readonly PluginSummaryRaw[]
): readonly PluginWebUiEntry[] {
  return summaries.map(adaptPluginSummaryToWebUiEntry)
}

export function buildPluginWebUiNavItems(entries: readonly PluginWebUiEntry[]): readonly PluginWebUiNavItem[] {
  return entries.flatMap((entry) => {
    if (!entry.navigationEligible) return []
    return entry.pages.map((page) => ({
      pluginName: entry.pluginName,
      pageId: page.id,
      label: page.label,
      route: page.route
    }))
  })
}

export function usePluginSummariesQuery(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: pluginKeys.list(),
    queryFn: fetchPluginSummaries,
    staleTime: 30_000,
    enabled: options?.enabled ?? true
  })
}

export function usePluginWebUiQuery(pluginName: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: pluginKeys.webUi(pluginName),
    queryFn: () => fetchPluginWebUi(pluginName),
    staleTime: 30_000,
    enabled: options?.enabled ?? true
  })
}

export function usePluginWebUiConfigQuery(pluginName: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: pluginKeys.webUiConfig(pluginName),
    queryFn: () => fetchPluginWebUiConfig(pluginName),
    staleTime: 30_000,
    enabled: options?.enabled ?? true
  })
}

export function usePluginWebUiConfigMutation(pluginName: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: PluginWebUiConfigMutationRequest) => requestPluginWebUiConfigMutation(pluginName, request),
    onSuccess: (config) => {
      queryClient.setQueryData(pluginKeys.webUiConfig(pluginName), config)
      void queryClient.invalidateQueries({ queryKey: pluginKeys.webUiConfig(pluginName) })
      void queryClient.invalidateQueries({ queryKey: pluginKeys.webUi(pluginName) })
      void queryClient.invalidateQueries({ queryKey: pluginKeys.list() })
      void queryClient.invalidateQueries({ queryKey: statusKeys.detail() })
    }
  })
}

export function useSetPluginWebUiEnabledMutation(pluginName: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (enabled: boolean) => setPluginWebUiEnabled(pluginName, enabled),
    onSuccess: (webUi) => {
      queryClient.setQueryData(pluginKeys.webUi(pluginName), webUi)
      void queryClient.invalidateQueries({ queryKey: pluginKeys.webUi(pluginName) })
      void queryClient.invalidateQueries({ queryKey: pluginKeys.list() })
      void queryClient.invalidateQueries({ queryKey: statusKeys.detail() })
    }
  })
}
