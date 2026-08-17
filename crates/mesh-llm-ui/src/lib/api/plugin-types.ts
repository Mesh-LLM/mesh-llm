export type PluginWebUiStateKind = 'none' | 'ready' | 'disabled' | 'invalid' | 'plugin_not_running'

export type PluginWebUiPageRaw = {
  readonly id: string
  readonly label: string
  readonly icon?: string
  readonly route: string
  readonly bundle_id: string
  readonly entry_script: string
}

export type PluginWebUiConfigSectionRaw = {
  readonly id: string
  readonly title: string
  readonly entry_script: string
  readonly parent_tab?: string
  readonly bundle_id: string
}

export type PluginWebUiManifestOverviewRaw = {
  readonly pages?: readonly PluginWebUiPageRaw[]
  readonly config_sections?: readonly PluginWebUiConfigSectionRaw[]
}

export type PluginWebUiStateRaw = {
  readonly state: PluginWebUiStateKind
  readonly declared: boolean
  readonly enabled: boolean
  readonly available: boolean
  readonly unavailable_reason?: string
  readonly pages?: readonly PluginWebUiPageRaw[]
  readonly config_sections?: readonly PluginWebUiConfigSectionRaw[]
  readonly asset_base_url?: string
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
