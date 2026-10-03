import type { RuntimeControlBootstrapPayload } from '@/features/configuration/api/config-adapter'

export const OWNER_CONTROL_READ_ONLY_MESSAGE =
  'No owner-control identity on this node, run both commands to unlock saving.'
export const OWNER_CONTROL_SAVE_ERROR = 'Config was not saved. Runtime control is disabled: missing owner identity.'
export const OWNER_CONTROL_DOCS_URL = 'https://meshllm.cloud/'

export function formatRuntimeControlDisabledReason(bootstrap: RuntimeControlBootstrapPayload | undefined) {
  if (bootstrap?.disabled_reason === 'missing_owner_identity') return 'missing owner identity'
  return bootstrap?.disabled_reason?.replace(/_/g, ' ') ?? 'unavailable'
}

export function formatRuntimeControlDisabledMessage(bootstrap: RuntimeControlBootstrapPayload) {
  if (bootstrap.disabled_reason === 'missing_owner_identity') return OWNER_CONTROL_READ_ONLY_MESSAGE
  return (
    bootstrap.message ??
    `Configuration saving is unavailable because runtime control is disabled: ${formatRuntimeControlDisabledReason(bootstrap)}.`
  )
}

export function formatRuntimeControlDisabledSaveError(bootstrap: RuntimeControlBootstrapPayload | undefined) {
  if (bootstrap?.disabled_reason === 'missing_owner_identity') return OWNER_CONTROL_SAVE_ERROR
  return `Config was not saved. Runtime control is disabled: ${formatRuntimeControlDisabledReason(bootstrap)}.`
}
