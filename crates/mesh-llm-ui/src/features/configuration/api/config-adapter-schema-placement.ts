import type { ConfigurationModelPlacementOptions, ConfigurationModelPlacementPaths } from '@/features/app-tabs/types'
import type { RuntimeConfigSchemaEntry, RuntimeConfigSchemaReference } from './config-adapter-types'

const PATH_RENDERER_FALLBACKS: Record<string, string> = {
  'defaults.throughput.parallel': 'slot-meter',
  'defaults.model_fit.kv_cache_policy': 'kv-cache-policy',
  'defaults.model_fit.ctx_size': 'context-slider'
}

export function rendererIdForEntry(entry: RuntimeConfigSchemaEntry): string | undefined {
  return entry.presentation?.renderer_id ?? PATH_RENDERER_FALLBACKS[entry.canonical_path]
}

export const DEFAULT_MODEL_PLACEMENT_PATHS: ConfigurationModelPlacementPaths = {
  model: 'models.<model-ref>.model',
  ctxSize: 'models.<model-ref>.model_fit.ctx_size',
  device: 'models.<model-ref>.hardware.device',
  gpuLayers: 'models.<model-ref>.hardware.gpu_layers',
  cacheTypeK: 'models.<model-ref>.model_fit.cache_type_k',
  cacheTypeV: 'models.<model-ref>.model_fit.cache_type_v',
  kvCachePolicy: 'models.<model-ref>.model_fit.kv_cache_policy',
  flashAttention: 'models.<model-ref>.model_fit.flash_attention',
  mmproj: 'models.<model-ref>.multimodal.mmproj'
}

export function modelPlacementPathsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined
): ConfigurationModelPlacementPaths {
  if (!schema) return DEFAULT_MODEL_PLACEMENT_PATHS

  const pathByRenderer = new Map(
    schema.settings
      .filter((entry) => entry.canonical_path.startsWith('models.<model-ref>.'))
      .map((entry) => [rendererIdForEntry(entry), entry.canonical_path] as const)
      .filter((entry): entry is readonly [string, string] => Boolean(entry[0]))
  )

  return {
    model: pathByRenderer.get('model-placement-model') ?? DEFAULT_MODEL_PLACEMENT_PATHS.model,
    ctxSize: pathByRenderer.get('model-placement-context') ?? DEFAULT_MODEL_PLACEMENT_PATHS.ctxSize,
    device: pathByRenderer.get('model-placement-device') ?? DEFAULT_MODEL_PLACEMENT_PATHS.device,
    gpuLayers: pathByRenderer.get('model-placement-gpu-layers') ?? DEFAULT_MODEL_PLACEMENT_PATHS.gpuLayers,
    cacheTypeK: DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeK,
    cacheTypeV: DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeV,
    kvCachePolicy: DEFAULT_MODEL_PLACEMENT_PATHS.kvCachePolicy,
    flashAttention: DEFAULT_MODEL_PLACEMENT_PATHS.flashAttention,
    mmproj: DEFAULT_MODEL_PLACEMENT_PATHS.mmproj
  }
}

function schemaEnumValuesForPath(schema: RuntimeConfigSchemaReference | undefined, canonicalPath: string): string[] {
  const setting = schema?.settings.find((entry) => entry.canonical_path === canonicalPath)
  if (setting?.value_schema.kind === 'enum') return [...setting.value_schema.values]
  return []
}

export function modelPlacementOptionsFromSchema(
  schema: RuntimeConfigSchemaReference | undefined
): ConfigurationModelPlacementOptions {
  return {
    cacheTypeK: schemaEnumValuesForPath(schema, DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeK!),
    cacheTypeV: schemaEnumValuesForPath(schema, DEFAULT_MODEL_PLACEMENT_PATHS.cacheTypeV!)
  }
}
