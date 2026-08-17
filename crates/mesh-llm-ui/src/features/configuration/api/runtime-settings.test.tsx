import { describe, expect, it } from 'vitest'
import type {
  RuntimeConfigControlStatePayload,
  RuntimeConfigSchemaEntry,
  RuntimeConfigSchemaReference
} from '@/features/configuration/api/config-adapter'
import { createRuntimePolicySettingsFromSchema } from '@/features/configuration/api/runtime-settings'

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

    // Setting uses the runtime TOML section, not a reserved or separate one
    const activitySetting = result.settings.find((s) => s.id === 'runtime.activity.enabled')!
    expect(activitySetting.tomlSection).toBe('runtime')
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
