import { describe, expect, it } from 'vitest'
import { parse } from 'smol-toml'
import type {
  RuntimeConfigControlStatePayload,
  RuntimeConfigSchemaEntry,
  RuntimeConfigSchemaReference
} from '@/features/configuration/api/config-adapter'
import { createRuntimePolicySettingsFromSchema } from '@/features/configuration/api/runtime-settings'
import { buildTOML } from '@/features/configuration/lib/build-toml'

function runtimeSetting(
  canonicalPath: string,
  valueSchema: RuntimeConfigSchemaEntry['value_schema'],
  constraints?: RuntimeConfigSchemaEntry['constraints']
): RuntimeConfigSchemaEntry {
  const name = canonicalPath.split('.').pop() ?? canonicalPath
  return {
    canonical_path: canonicalPath,
    owner: 'built_in',
    source: { kind: 'built_in' },
    value_schema: valueSchema,
    support: 'supported',
    control_surfaces: ['config_file'],
    apply_mode: 'static_on_load',
    restart_scope: 'process_restart',
    visibility: 'user',
    constraints,
    presentation: { label: name.replaceAll('_', ' '), help: `${name} setting.`, category_id: 'runtime-policy' }
  }
}

describe('createRuntimePolicySettingsFromSchema', () => {
  it('renders and saves runtime mode and activity policy from schema entries', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        // Core runtime settings (from Task 1 backend fields)
        runtimeSetting('runtime.mode', { kind: 'enum', values: ['serve', 'on_demand', 'client'] }),
        runtimeSetting('runtime.activity.response', {
          kind: 'enum',
          values: ['pause_remote', 'pause_all', 'reduce_priority']
        }),
        runtimeSetting('runtime.activity.enabled', { kind: 'boolean' }),

        // Non-runtime entry that should be excluded from policy settings
        runtimeSetting('defaults.throughput.parallel', { kind: 'integer' }, [{ kind: 'range', min: '1', max: '16' }])
      ]
    }

    const result = createRuntimePolicySettingsFromSchema(schema)

    // Only runtime.* entries (not defaults.*) appear in policy settings
    expect(result.settings.length).toBe(3)

    // Mode enum renders as a choice control with expected values
    const modeSetting = result.settings.find((s) => s.id === 'runtime.mode')!
    expect(modeSetting.control.kind).toBe('choice')
    if (modeSetting.control.kind !== 'choice') throw new Error()
    const modeValues = modeSetting.control.options.map((o) => o.value)
    expect(modeValues).toContain('serve')
    expect(modeValues).toContain('on_demand')

    // Activity response enum serializes correctly
    const activityResponse = result.settings.find((s) => s.id === 'runtime.activity.response')!
    if (activityResponse.control.kind !== 'choice') throw new Error()
    const arValues = activityResponse.control.options.map((o) => o.value)
    expect(arValues).toContain('pause_remote')

    // Activity enabled renders as a toggle choice
    const activityEnabled = result.settings.find((s) => s.id === 'runtime.activity.enabled')!
    if (activityEnabled.control.kind !== 'choice') throw new Error()
    expect(activityEnabled.control.presentation).toBe('toggle')
  })

  it('tolerates old server status and unsupported activity detector', () => {
    // Old servers return no runtime settings — the function should not crash,
    // just return an empty policy harness.
    const schema: RuntimeConfigSchemaReference = { settings: [] }
    const result = createRuntimePolicySettingsFromSchema(schema)

    expect(result.settings).toEqual([])
    expect(result.categories).toEqual([])

    // undefined schema also returns a safe fallback (the default harness)
    const emptyResult = createRuntimePolicySettingsFromSchema(undefined)
    expect(Array.isArray(emptyResult.settings)).toBe(true)
  })

  it('excludes runtime.debug and runtime.listen_all from policy settings', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        runtimeSetting('runtime.mode', { kind: 'enum', values: ['serve'] }),
        runtimeSetting('runtime.debug', { kind: 'boolean' }),
        runtimeSetting('runtime.listen_all', { kind: 'boolean' })
      ]
    }

    const result = createRuntimePolicySettingsFromSchema(schema)
    expect(result.settings.length).toBe(1)
    expect(result.settings[0]?.id).toBe('runtime.mode')
  })

  it('does not read or write reserve preview state', () => {
    // Reserve wake-policy preview is a separate feature — runtime activity controls
    // must never touch it. The schema-driven console only reads from the config-schema endpoint.
    const schema: RuntimeConfigSchemaReference = {
      settings: [runtimeSetting('runtime.activity.enabled', { kind: 'boolean' })]
    }

    const result = createRuntimePolicySettingsFromSchema(schema)

    // No reserve-related paths in output
    for (const setting of result.settings) {
      expect(setting.id).not.toMatch(/reserve/i)
      expect(setting.tomlSection ?? '').not.toMatch(/reserve/i)
    }

    // Nested activity settings keep their own TOML sub-table (issue #1784),
    // never a reserved section and never flattened into `[runtime]`.
    const activitySetting = result.settings.find((s) => s.id === 'runtime.activity.enabled')!
    expect(activitySetting.tomlSection).toBe('runtime.activity')
  })

  it('marks restart-required mutability from schema entry', () => {
    // Runtime settings that require process_restart show as restart-required
    const schema: RuntimeConfigSchemaReference = {
      settings: [runtimeSetting('runtime.mode', { kind: 'enum', values: ['serve'] })]
    }

    const result = createRuntimePolicySettingsFromSchema(schema)
    expect(result.settings[0]?.mutability).toBe('restart-required')

    // A dynamic_apply + none restart_scope entry would be runtime mutability
    const schema2: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...runtimeSetting('runtime.activity.poll_interval_secs', { kind: 'integer' }),
          apply_mode: 'dynamic_apply',
          restart_scope: 'none'
        }
      ]
    }

    const result2 = createRuntimePolicySettingsFromSchema(schema2)
    expect(result2.settings[0]?.mutability).toBe('runtime')
  })

  it('uses runtime control-state options for runtime policy select controls', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        {
          ...runtimeSetting('runtime.native_backend', { kind: 'enum', values: ['metal', 'vulkan'] }),
          control_behavior: { options_source: 'runtime_native_backends' }
        }
      ]
    }
    const nativeBackendControlState: NonNullable<RuntimeConfigControlStatePayload['settings']>[string] = {
      enabled: true,
      source: 'runtime',
      write_policy: 'preserve_existing',
      options: [
        {
          value: { kind: 'string', value: 'metal' },
          label: 'Metal',
          note: 'Available on this host',
          disabled: false,
          source: 'runtime_native_backends'
        },
        {
          value: { kind: 'string', value: 'vulkan' },
          label: 'Vulkan',
          reason: 'No Vulkan runtime was detected',
          disabled: true,
          source: 'runtime_native_backends'
        }
      ]
    }
    const controlState: RuntimeConfigControlStatePayload = {
      settings: {
        'runtime.native_backend': nativeBackendControlState
      }
    }

    const result = createRuntimePolicySettingsFromSchema(schema, controlState)

    const nativeBackend = result.settings.find((setting) => setting.id === 'runtime.native_backend')
    if (!nativeBackend) throw new Error('runtime.native_backend setting should be present')
    expect(nativeBackend.controlState).toBe(nativeBackendControlState)
    expect(nativeBackend.controlState?.options?.[1]?.disabled).toBe(true)
    expect(nativeBackend.control.kind).toBe('choice')
    if (nativeBackend.control.kind !== 'choice') throw new Error('runtime.native_backend should be a choice')
    expect(nativeBackend.control.presentation).toBe('select')
    expect(nativeBackend.control.options).toEqual([
      { value: '', label: 'Select backend' },
      { value: 'metal', label: 'Metal', description: 'Available on this host' },
      { value: 'vulkan', label: 'Vulkan', description: 'No Vulkan runtime was detected' }
    ])
  })
})

const NESTED_RUNTIME_SCHEMA: RuntimeConfigSchemaReference = {
  settings: [
    runtimeSetting('runtime.mode', { kind: 'enum', values: ['client', 'serve', 'on_demand'] }),
    runtimeSetting('runtime.drain_timeout_max_secs', { kind: 'integer' }),
    runtimeSetting('runtime.activity.enabled', { kind: 'boolean' }),
    runtimeSetting('runtime.activity.idle_after_secs', { kind: 'integer' }),
    runtimeSetting('runtime.activity.advertisement', {
      kind: 'enum',
      values: ['none', 'availability_only', 'coarse_state', 'private_coarse_state']
    }),
    runtimeSetting('runtime.native_runtime.selection', { kind: 'string' }),
    runtimeSetting('runtime.kv_cache.disk.mode', { kind: 'enum', values: ['off', 'auto', 'fixed'] })
  ]
}

describe('runtime policy TOML placement', () => {
  it('derives each nested runtime sub-table from the canonical path', () => {
    const byPath = new Map(
      createRuntimePolicySettingsFromSchema(NESTED_RUNTIME_SCHEMA).settings.map((setting) => [setting.id, setting])
    )

    expect(byPath.get('runtime.mode')?.tomlSection).toBe('runtime')
    expect(byPath.get('runtime.drain_timeout_max_secs')?.tomlSection).toBe('runtime')
    expect(byPath.get('runtime.activity.enabled')?.tomlSection).toBe('runtime.activity')
    expect(byPath.get('runtime.activity.idle_after_secs')?.tomlSection).toBe('runtime.activity')
    expect(byPath.get('runtime.activity.advertisement')?.tomlSection).toBe('runtime.activity')
    expect(byPath.get('runtime.native_runtime.selection')?.tomlSection).toBe('runtime.native_runtime')
    expect(byPath.get('runtime.kv_cache.disk.mode')?.tomlSection).toBe('runtime.kv_cache.disk')
  })

  it('keeps the last canonical path segment as the TOML key', () => {
    const settings = createRuntimePolicySettingsFromSchema(NESTED_RUNTIME_SCHEMA).settings

    for (const setting of settings) {
      expect(setting.tomlKey).toBe(setting.id.split('.').at(-1))
    }
  })
})

describe('buildTOML runtime policy round-trip', () => {
  // Issue #1784: nested runtime sub-tables must not collapse into `[runtime]`,
  // where mesh-llm silently ignores the misplaced keys.
  const sourceValues: Record<string, string> = {
    'runtime.mode': 'client',
    'runtime.drain_timeout_max_secs': '120',
    'runtime.activity.enabled': 'on',
    'runtime.activity.idle_after_secs': '600',
    'runtime.activity.advertisement': 'coarse_state',
    'runtime.native_runtime.selection': 'metal',
    'runtime.kv_cache.disk.mode': 'fixed'
  }

  function renderedToml(): string {
    const defaults = createRuntimePolicySettingsFromSchema(NESTED_RUNTIME_SCHEMA)
    return buildTOML([], [], [], { defaults, defaultsValues: sourceValues })
  }

  it('places nested runtime settings in their own tables', () => {
    const toml = renderedToml()

    expect(toml).toContain('[runtime]')
    expect(toml).toContain('[runtime.activity]')
    expect(toml).toContain('[runtime.native_runtime]')
    expect(toml).toContain('[runtime.kv_cache.disk]')
  })

  it('parses back to the nested runtime config without flattening', () => {
    const parsed = parse(renderedToml()) as unknown as { runtime: Record<string, unknown> }

    expect(parsed.runtime.mode).toBe('client')
    expect(parsed.runtime.drain_timeout_max_secs).toBe(120)
    expect(parsed.runtime.activity).toEqual({
      enabled: true,
      idle_after_secs: 600,
      advertisement: 'coarse_state'
    })
    expect(parsed.runtime.native_runtime).toEqual({ selection: 'metal' })
    expect(parsed.runtime.kv_cache).toEqual({ disk: { mode: 'fixed' } })

    // Regression guard: the flattened preview used to emit these keys directly
    // under `[runtime]`, which mesh-llm accepts as unknown fields.
    expect(parsed.runtime.selection).toBeUndefined()
    expect(parsed.runtime.advertisement).toBeUndefined()
    expect(parsed.runtime.enabled).toBeUndefined()
    expect(parsed.runtime.idle_after_secs).toBeUndefined()
  })
})
