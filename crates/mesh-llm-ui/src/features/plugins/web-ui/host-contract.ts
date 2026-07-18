import type {
  PluginWebUiConfigMutationRequest,
  PluginWebUiConfigSectionRaw,
  PluginWebUiPageRaw,
  PluginWebUiStateRaw,
  PluginWebUiVisibleConfigRaw
} from '@/lib/api/plugin-types'

export type MeshPluginUiAppearance = {
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

export type MeshPluginUiToast = {
  readonly title: string
  readonly description?: string
  readonly tone?: 'info' | 'success' | 'warning' | 'error'
}

export type MeshPluginUiStateSnapshot = Readonly<Record<string, unknown>>
export type MeshPluginUiStateSubscriber = (snapshot: MeshPluginUiStateSnapshot) => void

export type MeshPluginUiHost = {
  readonly plugin: {
    readonly name: string
  }
  readonly page: {
    readonly id: string
    readonly label: string
    readonly route: string
  }
  readonly webUi: PluginWebUiStateRaw
  readonly appearance: MeshPluginUiAppearance
  readonly network: {
    readonly fetchPlugin: (path: string, init?: RequestInit) => Promise<Response>
    readonly json: (path: string, init?: RequestInit) => Promise<unknown>
  }
  readonly config: {
    readonly visible: PluginWebUiVisibleConfigRaw
    readonly requestMutation: (request: PluginWebUiConfigMutationRequest) => Promise<PluginWebUiVisibleConfigRaw>
  }
  readonly navigation: {
    readonly navigateTo: (path: string) => void
    readonly openPluginPage: (pageId: string) => void
  }
  readonly notifications: {
    readonly show: (toast: MeshPluginUiToast) => void
  }
  readonly state: {
    readonly getSnapshot: () => MeshPluginUiStateSnapshot
    readonly update: (patch: MeshPluginUiStateSnapshot) => MeshPluginUiStateSnapshot
    readonly subscribe: (subscriber: MeshPluginUiStateSubscriber) => () => void
  }
}

export type MeshPluginUiMountContext = {
  readonly element: HTMLElement
  readonly host: MeshPluginUiHost
  readonly page: PluginWebUiPageRaw
}

export type MeshPluginUiConfigMountContext = {
  readonly element: HTMLElement
  readonly host: MeshPluginUiHost
  readonly section: PluginWebUiConfigSectionRaw
}

export type MeshPluginUiMountHandle = {
  readonly unmount: () => void
}

export type MeshPluginUiPageMount = (
  context: MeshPluginUiMountContext
) => MeshPluginUiMountHandle | Promise<MeshPluginUiMountHandle>

export type MeshPluginUiConfigSectionMount = (
  context: MeshPluginUiConfigMountContext
) => MeshPluginUiMountHandle | Promise<MeshPluginUiMountHandle>

export type MeshPluginUiRegistration = {
  readonly pages: Readonly<Record<string, MeshPluginUiPageMount>>
  readonly configSections?: Readonly<Record<string, MeshPluginUiConfigSectionMount>>
}

export type MeshPluginUiBundleModule = {
  readonly registerMeshPluginUi: (
    host: MeshPluginUiHost
  ) => MeshPluginUiRegistration | Promise<MeshPluginUiRegistration>
}
