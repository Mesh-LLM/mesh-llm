import type { MeshPluginUiBundleModule } from '@/features/plugins/web-ui/host-contract'

class PluginUiBundleContractError extends Error {
  readonly name = 'PluginUiBundleContractError'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isPluginUiBundleModule(value: unknown): value is MeshPluginUiBundleModule {
  return isRecord(value) && typeof value.registerMeshPluginUi === 'function'
}

export async function importPluginUiBundle(assetUrl: string): Promise<MeshPluginUiBundleModule> {
  const module: unknown = await import(/* @vite-ignore */ assetUrl)
  if (isPluginUiBundleModule(module)) return module
  throw new PluginUiBundleContractError('Plugin bundle did not export registerMeshPluginUi(host)')
}
