export const MOA_PROGRESS_LABEL = 'Consulting peers and corroborating responses…'

const SYNTHETIC_MOA_PROGRESS = new Set([
  'routing through mesh…',
  'querying peer models…',
  'comparing responses…',
  'waiting on a slow peer…',
  'still gathering responses…',
  "hold on, this one's taking a moment…",
  'hold on, this is taking a moment…'
])

export function isMeshVirtualModel(model?: string) {
  return model?.trim().toLowerCase() === 'mesh'
}

export function syntheticMoaProgressKey(delta: string) {
  const normalized = delta.trim().replace(/\s+/g, ' ').toLowerCase()
  return SYNTHETIC_MOA_PROGRESS.has(normalized) ? normalized : undefined
}
