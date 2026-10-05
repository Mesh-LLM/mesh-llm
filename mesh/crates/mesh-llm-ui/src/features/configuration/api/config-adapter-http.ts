import { ApiError, parseApiErrorBody } from '@/lib/api/errors'
import { env } from '@/lib/env'
import type {
  RuntimeConfigControlStatePayload,
  RuntimeConfigSchemaReference,
  RuntimeConfigValidateResponse,
  RuntimeControlApplyInput,
  RuntimeControlApplyResponse,
  RuntimeControlBootstrapPayload,
  RuntimeControlConfigResult,
  RuntimeControlConfigSnapshot
} from './config-adapter-types'
import { mergeConfigurationIntoMeshConfig } from './config-adapter-merge'

type RuntimeControlConfigResponse = {
  snapshot: RuntimeControlConfigSnapshot
}

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await parseApiErrorBody(response)
    throw new ApiError(response.status, message, message)
  }

  return response.json() as Promise<T>
}

export async function fetchRuntimeControlBootstrap(): Promise<RuntimeControlBootstrapPayload> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control-bootstrap`)
  return expectJson<RuntimeControlBootstrapPayload>(response)
}

export async function fetchRuntimeConfigSchema(): Promise<RuntimeConfigSchemaReference> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/config-schema`)
  return expectJson<RuntimeConfigSchemaReference>(response)
}

export async function fetchRuntimeConfigControlState(): Promise<RuntimeConfigControlStatePayload> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/config-control-state`)
  try {
    const payload = await expectJson<RuntimeConfigControlStatePayload>(response)
    return { settings: payload.settings ?? {} }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return { settings: {} }
    throw error
  }
}

export async function fetchRuntimeControlConfigSnapshot(endpoint: string): Promise<RuntimeControlConfigSnapshot> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control/get-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ endpoint })
  })
  const payload = await expectJson<RuntimeControlConfigResponse>(response)
  return payload.snapshot
}

export async function fetchRuntimeControlConfig(): Promise<RuntimeControlConfigResult> {
  const [bootstrap, schema, controlState] = await Promise.all([
    fetchRuntimeControlBootstrap(),
    fetchRuntimeConfigSchema(),
    fetchRuntimeConfigControlState()
  ])
  const endpoint = bootstrap.endpoint?.trim()

  if (!bootstrap.enabled || !endpoint) return { bootstrap, schema, controlState }

  const snapshot = await fetchRuntimeControlConfigSnapshot(endpoint)
  return { bootstrap: { ...bootstrap, endpoint }, schema, snapshot, controlState }
}

export async function applyRuntimeControlConfig(
  endpoint: string,
  snapshot: RuntimeControlConfigSnapshot,
  input: RuntimeControlApplyInput,
  schema?: RuntimeConfigSchemaReference,
  controlState?: RuntimeConfigControlStatePayload
): Promise<{ response: RuntimeControlApplyResponse; snapshot: RuntimeControlConfigSnapshot }> {
  const config = mergeConfigurationIntoMeshConfig(snapshot.config, input, schema, {
    includeModelAssignments: true,
    controlState
  })
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control/apply-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      endpoint,
      expected_revision: snapshot.revision,
      config
    })
  })
  const payload = await expectJson<RuntimeControlApplyResponse>(response)

  return {
    response: payload,
    snapshot: {
      ...snapshot,
      revision: payload.current_revision,
      config
    }
  }
}

export async function validateRuntimeConfigToml(toml: string, path?: string): Promise<RuntimeConfigValidateResponse> {
  const response = await fetch(`${env.managementApiUrl}/api/runtime/config/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ toml, path })
  })
  return expectJson<RuntimeConfigValidateResponse>(response)
}
