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
import { getOwnerControlCredential, setOwnerControlCredential } from './owner-control-credential'

type RuntimeControlConfigResponse = {
  snapshot: RuntimeControlConfigSnapshot
}

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const { body: message } = await parseApiErrorBody(response)
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
  const token = await getOwnerControlCredential()
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control/get-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ endpoint })
  })
  if (response.status === 401) setOwnerControlCredential(null)
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

type PluginCommandSource =
  | {
      plugin?: ReadonlyArray<{ name?: string; command?: string; args?: unknown[] }>
    }
  | null
  | undefined

export type PluginCommandApproval = { plugin: string; command: string; invocation: { args: string[] } }

// Plugins whose command executable this apply adds or changes; the host
// rejects an apply whose declared approvals do not cover this diff.
export function changedPluginCommandApprovals(
  previous: PluginCommandSource,
  next: PluginCommandSource
): PluginCommandApproval[] {
  const previousCommands = new Map<string, { command?: string; args?: unknown[] }>()
  for (const plugin of previous?.plugin ?? []) {
    if (plugin.name) previousCommands.set(plugin.name, plugin)
  }
  const approvals: PluginCommandApproval[] = []
  for (const plugin of next?.plugin ?? []) {
    const command = plugin.command ?? null
    if (!plugin.name || !command) continue
    const previousPlugin = previousCommands.get(plugin.name)
    if (
      previousPlugin?.command !== command ||
      JSON.stringify(previousPlugin.args ?? []) !== JSON.stringify(plugin.args ?? [])
    ) {
      const args = plugin.args ?? []
      if (!args.every((arg): arg is string => typeof arg === 'string')) {
        throw new Error(`Plugin ${plugin.name} has invalid command arguments.`)
      }
      approvals.push({ plugin: plugin.name, command, invocation: { args } })
    }
  }
  return approvals
}

export function describePluginCommandApprovals(
  config: PluginCommandSource,
  approvals: PluginCommandApproval[]
): string[] {
  return (config?.plugin ?? [])
    .filter((entry) =>
      approvals.some(
        (approval) =>
          approval.plugin === entry.name &&
          approval.command === entry.command &&
          JSON.stringify(approval.invocation.args) === JSON.stringify(entry.args ?? [])
      )
    )
    .map((entry) => `${entry.name}: ${entry.command} ${JSON.stringify(entry.args ?? [])}`)
}

export async function applyRuntimeControlConfig(
  endpoint: string,
  snapshot: RuntimeControlConfigSnapshot,
  input: RuntimeControlApplyInput,
  schema?: RuntimeConfigSchemaReference,
  controlState?: RuntimeConfigControlStatePayload,
  confirmedApprovals: PluginCommandApproval[] = []
): Promise<{ response: RuntimeControlApplyResponse; snapshot: RuntimeControlConfigSnapshot }> {
  const config = mergeConfigurationIntoMeshConfig(snapshot.config, input, schema, {
    includeModelAssignments: true,
    controlState
  })
  const token = await getOwnerControlCredential()
  const response = await fetch(`${env.managementApiUrl}/api/runtime/control/apply-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      endpoint,
      expected_revision: snapshot.revision,
      config,
      ...(confirmedApprovals.length > 0 ? { plugin_command_approvals: confirmedApprovals } : {})
    })
  })
  if (response.status === 401) setOwnerControlCredential(null)
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
