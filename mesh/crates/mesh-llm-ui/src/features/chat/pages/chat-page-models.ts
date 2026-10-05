import type { ModelSelectOption, ModelSummary } from '@/features/app-tabs/types'

// The dropdown shows a single "Mesh — automatic" entry. Selecting it keeps
// `model === AUTO_MODEL_VALUE` in UI state (so Radix Select can highlight it
// correctly) but sends `AUTO_BACKEND_MODEL` on the wire so requests fan out
// through the Mixture-of-Agents gateway.
export const AUTO_MODEL_VALUE = 'auto'
export const AUTO_BACKEND_MODEL = 'mesh'
export const AUTO_MODEL_LABEL = 'Mesh — automatic'
export const AUTO_MODEL_OPTION: ModelSelectOption = {
  value: AUTO_MODEL_VALUE,
  label: AUTO_MODEL_LABEL,
  status: { label: 'Auto', tone: 'accent' }
}

export function modelStatusBadge(model: ModelSummary): ModelSelectOption['status'] {
  if (model.status === 'offline') return { label: 'Offline', tone: 'bad' }
  if (model.status === 'warming') return { label: 'Warming', tone: 'warn' }
  if (model.status === 'ready') return { label: 'Ready', tone: 'good' }
  return { label: 'Warm', tone: 'good' }
}

export function isChatSelectableModel(model: ModelSummary): boolean {
  return model.status === 'ready' || model.status === 'warm'
}
