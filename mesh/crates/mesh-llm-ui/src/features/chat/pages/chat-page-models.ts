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

/**
 * The workload classes `/v1/responses` admits a chat turn to. This mirrors
 * `class_is_compatible(CausalGeneration, _)` in the host's workload routing so
 * the picker never offers a model the endpoint would refuse. An unadvertised
 * class is a peer that predates workload advertisement and stays offered, the
 * same way the host still admits it; every other class, including a
 * decision-only System One model such as Laya, is not a chat model.
 */
const CHAT_CAPABLE_WORKLOAD_CLASSES = new Set(['causal_generation', 'encoder_decoder'])

export function isChatSelectableModel(model: ModelSummary): boolean {
  if (model.status !== 'ready' && model.status !== 'warm') return false
  return model.workloadClass == null || CHAT_CAPABLE_WORKLOAD_CLASSES.has(model.workloadClass)
}
