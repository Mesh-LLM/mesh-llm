export type PluginWebUiStateKind = 'none' | 'ready' | 'disabled' | 'invalid' | 'plugin_not_running'

export type PluginWebUiPlacementRaw = 'primary' | 'auxiliary'

export type PluginWebUiPageRaw = {
  readonly id: string
  readonly label: string
  readonly icon?: string
  readonly route: string
  readonly bundle_id: string
  readonly entry_script: string
  readonly placement?: PluginWebUiPlacementRaw
}

export type PluginWebUiConfigSectionRaw = {
  readonly id: string
  readonly title: string
  readonly entry_script: string
  readonly parent_tab?: string
  readonly bundle_id: string
}

/** Where the host mounts a contribution: under a finished assistant chat
 *  message, or in a Logs request's inspector header. */
export type PluginWebUiContributionSlot = 'chat_message' | 'logs_request'

export type PluginWebUiContributionRaw = {
  readonly id: string
  readonly slot: PluginWebUiContributionSlot
  readonly label: string
  readonly bundle_id: string
  readonly entry_script: string
}

export type PluginWebUiManifestOverviewRaw = {
  readonly pages?: readonly PluginWebUiPageRaw[]
  readonly config_sections?: readonly PluginWebUiConfigSectionRaw[]
  readonly contributions?: readonly PluginWebUiContributionRaw[]
}

export type PluginWebUiStateRaw = {
  readonly state: PluginWebUiStateKind
  readonly declared: boolean
  readonly enabled: boolean
  readonly available: boolean
  readonly unavailable_reason?: string
  readonly pages?: readonly PluginWebUiPageRaw[]
  readonly config_sections?: readonly PluginWebUiConfigSectionRaw[]
  readonly contributions?: readonly PluginWebUiContributionRaw[]
  readonly asset_base_url?: string
  readonly primary_tab_enabled: boolean
}

export type PluginWebUiVisibleConfigRaw = {
  readonly plugin: string
  readonly settings: Readonly<Record<string, unknown>>
  readonly schema?: unknown
}

export type PluginWebUiConfigMutationRequest = {
  readonly plugin?: string
  readonly settings?: Readonly<Record<string, unknown>>
  readonly unset?: readonly string[]
}

export type PluginToolSummaryRaw = {
  readonly name: string
  readonly description?: string
  readonly input_schema?: unknown
}

export type PluginStartupSummaryRaw = {
  readonly phase?: string
  readonly message?: string
  readonly started_at_unix_ms?: number
  readonly last_error?: string
}

export type PluginManifestOverviewRaw = {
  readonly capabilities?: readonly string[]
  readonly web_ui?: PluginWebUiManifestOverviewRaw
  readonly [key: string]: unknown
}

export type PluginSummaryRaw = {
  readonly name: string
  readonly kind: string
  readonly enabled: boolean
  readonly status: string
  readonly description?: string
  readonly pid?: number
  readonly version?: string
  readonly capabilities?: readonly string[]
  readonly command?: string
  readonly args?: readonly string[]
  readonly tools?: readonly PluginToolSummaryRaw[]
  readonly manifest?: PluginManifestOverviewRaw
  readonly web_ui: PluginWebUiStateRaw
  readonly startup?: PluginStartupSummaryRaw
  readonly error?: string
}
