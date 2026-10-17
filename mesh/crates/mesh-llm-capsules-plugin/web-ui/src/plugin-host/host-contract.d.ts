export type PluginWebUiState = 'none' | 'ready' | 'disabled' | 'invalid' | 'plugin_not_running'

export type PluginWebUiPage = {
  readonly id: string
  readonly label: string
  readonly route: string
  readonly bundle_id: string
  readonly entry_script: string
  readonly icon?: string
}

export type PluginWebUiConfigSection = {
  readonly id: string
  readonly title: string
  readonly bundle_id: string
  readonly entry_script: string
  readonly parent_tab?: 'integrations'
}

export type PluginWebUiVisibleConfig = {
  readonly plugin: string
  readonly settings: Readonly<Record<string, unknown>>
  readonly schema?: unknown
}

export type MeshPluginUiHost = {
  readonly plugin: { readonly name: string }
  readonly page: { readonly id: string; readonly label: string; readonly route: string }
  readonly webUi: {
    readonly state: PluginWebUiState
    readonly declared: boolean
    readonly enabled: boolean
    readonly available: boolean
    readonly unavailable_reason?: string
    readonly pages?: readonly PluginWebUiPage[]
    readonly config_sections?: readonly PluginWebUiConfigSection[]
    readonly asset_base_url?: string
  }
  readonly appearance: {
    readonly theme: string
    readonly accent: string
    readonly density: string
    readonly panelStyle: string
    readonly tokens: {
      readonly background: string
      readonly foreground: string
      readonly panel: string
      readonly panelStrong: string
      readonly border: string
      readonly borderSoft: string
      readonly accent: string
      readonly accentInk: string
      readonly good: string
      readonly warn: string
      readonly bad: string
      readonly radius: string
      readonly radiusLarge: string
    }
  }
  readonly network: {
    readonly fetchPlugin: (path: string, init?: RequestInit) => Promise<Response>
    readonly json: (path: string, init?: RequestInit) => Promise<unknown>
  }
  readonly config: {
    readonly visible: PluginWebUiVisibleConfig
    readonly requestMutation: (request: {
      readonly plugin?: string
      readonly settings?: Readonly<Record<string, unknown>>
      readonly unset?: readonly string[]
    }) => Promise<PluginWebUiVisibleConfig>
  }
  readonly navigation: {
    readonly navigateTo: (path: string) => void
    readonly openPluginPage: (pageId: string) => void
  }
  readonly notifications: {
    readonly show: (notice: {
      readonly title: string
      readonly description?: string
      readonly tone?: 'info' | 'success' | 'warning' | 'error'
    }) => void
  }
  readonly state: {
    readonly getSnapshot: () => Readonly<Record<string, unknown>>
    readonly update: (patch: Readonly<Record<string, unknown>>) => Readonly<Record<string, unknown>>
    readonly subscribe: (subscriber: (snapshot: Readonly<Record<string, unknown>>) => void) => () => void
  }
}

export type MeshPluginUiMountHandle = { readonly unmount: () => void }

export type MeshPluginUiMountContext = {
  readonly element: HTMLElement
  readonly host: MeshPluginUiHost
  readonly page: PluginWebUiPage
}

export type MeshPluginUiConfigMountContext = {
  readonly element: HTMLElement
  readonly host: MeshPluginUiHost
  readonly section: PluginWebUiConfigSection
}

/** A contribution: a small piece of this plugin's UI the console places in
 *  one of its own views (mesh-llm 0.78+, "contribution slots"). */
export type PluginWebUiContribution = {
  readonly id: string
  readonly slot: 'chat_message' | 'logs_request'
  readonly label: string
  readonly bundle_id: string
  readonly entry_script: string
}

/** Host ids for one finished assistant chat message. Optional fields are
 *  absent when the host does not have them. */
export type MeshPluginUiChatMessageSubject = {
  readonly slot: 'chat_message'
  readonly messageId: string
  readonly clientNonce?: string
  readonly model?: string
  readonly servedBy?: string
}

/** Host ids for one Logs request. */
export type MeshPluginUiLogsRequestSubject = {
  readonly slot: 'logs_request'
  readonly requestId: string
  readonly exchangeId?: string
}

export type MeshPluginUiContributionSubject = MeshPluginUiChatMessageSubject | MeshPluginUiLogsRequestSubject

export type MeshPluginUiContributionMountContext = {
  readonly element: HTMLElement
  readonly host: MeshPluginUiHost
  readonly contribution: PluginWebUiContribution
  readonly subject: MeshPluginUiContributionSubject
}

export type MeshPluginUiRegistration = {
    readonly pages: Readonly<Record<string, (context: {
      readonly element: HTMLElement
      readonly host: MeshPluginUiHost
      readonly page: PluginWebUiPage
    }) => MeshPluginUiMountHandle | Promise<MeshPluginUiMountHandle>>>
    readonly configSections?: Readonly<Record<string, (context: {
      readonly element: HTMLElement
      readonly host: MeshPluginUiHost
      readonly section: PluginWebUiConfigSection
    }) => MeshPluginUiMountHandle | Promise<MeshPluginUiMountHandle>>>
    readonly contributions?: Readonly<Record<string, (context: MeshPluginUiContributionMountContext) =>
      MeshPluginUiMountHandle | Promise<MeshPluginUiMountHandle>>>
}

export type MeshPluginUiBundleModule = {
  readonly registerMeshPluginUi: (host: MeshPluginUiHost) =>
    MeshPluginUiRegistration | Promise<MeshPluginUiRegistration>
}
