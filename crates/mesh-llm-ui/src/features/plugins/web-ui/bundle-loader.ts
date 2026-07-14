import type {
  MeshPluginUiBundleModule,
  MeshPluginUiMountHandle,
  MeshPluginUiRegistration
} from '@/features/plugins/web-ui/host-contract'

class PluginUiBundleContractError extends Error {
  readonly name = 'PluginUiBundleContractError'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isPluginUiBundleModule(value: unknown): value is MeshPluginUiBundleModule {
  return isRecord(value) && typeof value.registerMeshPluginUi === 'function'
}

function assertMountMap(value: Record<string, unknown>, label: string): void {
  for (const [id, mount] of Object.entries(value)) {
    if (typeof mount !== 'function') {
      throw new PluginUiBundleContractError(`Plugin bundle ${label} entry '${id}' must be a mount function`)
    }
  }
}

export function assertPluginUiRegistration(value: unknown): asserts value is MeshPluginUiRegistration {
  if (!isRecord(value) || !isRecord(value.pages)) {
    throw new PluginUiBundleContractError('Plugin bundle registration must provide a pages object')
  }
  assertMountMap(value.pages, 'pages')
  if (value.configSections !== undefined && !isRecord(value.configSections)) {
    throw new PluginUiBundleContractError('Plugin bundle configSections must be an object when provided')
  }
  if (isRecord(value.configSections)) assertMountMap(value.configSections, 'configSections')
}

export function assertPluginUiMountHandle(value: unknown): asserts value is MeshPluginUiMountHandle {
  if (!isRecord(value) || typeof value.unmount !== 'function') {
    throw new PluginUiBundleContractError('Plugin UI mount must return an unmount() handle')
  }
}

export async function importPluginUiBundle(assetUrl: string): Promise<MeshPluginUiBundleModule> {
  const module: unknown = await import(/* @vite-ignore */ assetUrl)
  if (isPluginUiBundleModule(module)) return module
  throw new PluginUiBundleContractError('Plugin bundle did not export registerMeshPluginUi(host)')
}
