import { describe, expect, it } from 'vitest'
import {
  adaptStatusToConfiguration,
  createConfigurationAuditSettingsFromSchema,
  createConfigurationDefaultsValuesFromMeshConfig,
  createConfigurationIntegrationsFromSchema,
  createConfigurationMeshLLMSettingsFromSchema,
  createConfigurationModelSettingsFromSchema,
  createConfigurationNetworkSettingsFromSchema,
  formatConfigDiagnostics,
  mergeConfigurationDefaultsIntoMeshConfig,
  mergeConfigurationIntoMeshConfig
} from '@/features/configuration/api/config-adapter'
import { AUDIT_SCHEMA, loggingSchemaSetting, SCHEMA_REFERENCE, STATUS_PAYLOAD } from './config-adapter-test-support'
import type {
  RuntimeConfigControlStatePayload,
  RuntimeConfigSchemaReference,
  RuntimeControlDiagnostic
} from './config-adapter-types'

describe('configuration logging and runtime-control metadata', () => {
  it('preserves current editability when runtime control-state is empty', () => {
    const modelSettings = createConfigurationModelSettingsFromSchema(SCHEMA_REFERENCE, { settings: {} })
    const assignment = modelSettings.settings.find((setting) => setting.id === 'gpu.assignment')

    expect(assignment).toMatchObject({
      label: 'GPU assignment',
      control: expect.objectContaining({
        kind: 'choice',
        value: 'auto'
      })
    })
    expect(assignment?.controlState).toBeUndefined()
  })

  it('instantiates integration controls from plugin instances and plugin-owned schema settings', () => {
    const integrations = createConfigurationIntegrationsFromSchema(SCHEMA_REFERENCE)

    expect(integrations?.categories[0]).toMatchObject({
      id: 'plugin:blackboard',
      label: 'Blackboard'
    })
    expect(integrations?.settings.find((setting) => setting.id === 'plugin.blackboard.enabled')).toMatchObject({
      id: 'plugin.blackboard.enabled',
      canonicalPath: 'plugin.blackboard.enabled',
      label: 'Enabled',
      control: expect.objectContaining({
        kind: 'choice',
        value: 'on'
      })
    })
    expect(
      integrations?.settings.find((setting) => setting.id === 'plugin.blackboard.settings.retention_days')
    ).toMatchObject({
      id: 'plugin.blackboard.settings.retention_days',
      canonicalPath: 'plugin.blackboard.settings.retention_days',
      label: 'Retention days',
      control: expect.objectContaining({
        kind: 'range',
        min: 1,
        max: 365
      })
    })
  })

  it('places debug and listen-all settings on their requested tabs and writes runtime config', () => {
    const meshllm = createConfigurationMeshLLMSettingsFromSchema(SCHEMA_REFERENCE)
    const network = createConfigurationNetworkSettingsFromSchema(SCHEMA_REFERENCE)

    expect(meshllm.settings.find((setting) => setting.id === 'runtime.debug')).toMatchObject({
      label: 'Debug output',
      categoryId: 'meshllm',
      tomlSection: 'runtime',
      tomlKey: 'debug',
      control: expect.objectContaining({ kind: 'choice', value: 'off' })
    })
    expect(network.settings.find((setting) => setting.id === 'runtime.listen_all')).toMatchObject({
      label: 'Listen on all interfaces',
      categoryId: 'network',
      tomlSection: 'runtime',
      tomlKey: 'listen_all',
      control: expect.objectContaining({ kind: 'choice', value: 'off' })
    })

    const merged = mergeConfigurationDefaultsIntoMeshConfig(
      { version: 1 },
      {
        'runtime.debug': 'on',
        'runtime.listen_all': 'on'
      },
      SCHEMA_REFERENCE
    )

    expect(merged.runtime).toMatchObject({
      debug: true,
      listen_all: true
    })
  })

  it('projects logging settings from the live schema with server-owned controls and apply metadata', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        loggingSchemaSetting(
          'logging.retention_ttl_secs',
          { kind: 'integer' },
          {
            apply_mode: 'dynamic_apply',
            restart_scope: 'none',
            constraints: [{ kind: 'range', min: '60', max: '604800' }],
            presentation: {
              label: 'Retention period',
              help: 'Server-provided retention copy.',
              category_id: 'logs-retention',
              category_label: 'Event history',
              category_summary: 'Server-provided logging category.',
              category_order: 35,
              setting_order: 10,
              unit: 'seconds'
            }
          }
        ),
        loggingSchemaSetting(
          'logging.replay_capacity',
          { kind: 'integer' },
          {
            apply_mode: 'dynamic_apply',
            restart_scope: 'none',
            constraints: [{ kind: 'range', min: '1', max: '5000' }],
            presentation: {
              label: 'Replay buffer capacity',
              help: 'Server-provided replay copy.',
              category_id: 'logs-buffers',
              setting_order: 20
            }
          }
        ),
        loggingSchemaSetting(
          'logging.artifact.capture_mode',
          {
            kind: 'enum',
            values: ['metadata_only', 'redacted_artifacts']
          },
          {
            presentation: {
              label: 'Artifact retention mode',
              help: 'Server-provided artifact mode copy.',
              category_id: 'logs-artifacts',
              setting_order: 30
            }
          }
        ),
        loggingSchemaSetting(
          'logging.artifact.byte_limit_bytes',
          { kind: 'integer' },
          {
            constraints: [{ kind: 'range', min: '1024', max: '1048576' }],
            presentation: {
              label: 'Per-artifact byte cap',
              help: 'Server-provided artifact cap copy.',
              category_id: 'logs-artifacts',
              setting_order: 40
            }
          }
        ),
        loggingSchemaSetting('logging.legacy_payload_access', { kind: 'string' }, { support: 'unsupported' })
      ]
    }
    const controlState = {
      settings: {
        'logging.artifact.byte_limit_bytes': {
          enabled: false,
          reason: 'Artifact capture is unavailable while capture mode is metadata_only.',
          source: 'runtime',
          write_policy: 'reject_when_disabled'
        }
      }
    } satisfies RuntimeConfigControlStatePayload

    const settings = createConfigurationAuditSettingsFromSchema(schema, controlState)
    const retention = settings.settings.find((setting) => setting.id === 'logging.retention_ttl_secs')
    const replay = settings.settings.find((setting) => setting.id === 'logging.replay_capacity')
    const artifactLimit = settings.settings.find((setting) => setting.id === 'logging.artifact.byte_limit_bytes')

    expect(settings.categories).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: 'logs-retention', label: 'Event history', order: 35 })])
    )
    expect(settings.settings.map((setting) => setting.id).sort()).toEqual([
      'logging.artifact.byte_limit_bytes',
      'logging.artifact.capture_mode',
      'logging.replay_capacity',
      'logging.retention_ttl_secs'
    ])
    expect(retention).toMatchObject({
      label: 'Retention period',
      description: 'Server-provided retention copy.',
      validationConstraints: [{ kind: 'range', min: '60', max: '604800' }],
      control: { kind: 'range', min: 60, max: 604800, step: 1, unit: 'seconds' },
      mutability: 'runtime',
      applyMode: 'dynamic_apply',
      restartScope: 'none'
    })
    expect(replay).toMatchObject({
      label: 'Replay buffer capacity',
      description: 'Server-provided replay copy.',
      mutability: 'runtime',
      applyMode: 'dynamic_apply',
      restartScope: 'none'
    })
    expect(artifactLimit).toMatchObject({
      label: 'Per-artifact byte cap',
      description: 'Server-provided artifact cap copy.',
      mutability: 'restart-required',
      applyMode: 'static_on_load',
      restartScope: 'process_restart',
      controlState: controlState.settings['logging.artifact.byte_limit_bytes']
    })
    expect(settings.settings.find((setting) => setting.id === 'logging.legacy_payload_access')).toBeUndefined()

    expect(
      createConfigurationDefaultsValuesFromMeshConfig(
        { logging: { retention_ttl_secs: 120, replay_capacity: 25 } },
        schema,
        controlState
      )
    ).toMatchObject({
      'logging.retention_ttl_secs': '120',
      'logging.replay_capacity': '25'
    })

    expect(retention?.visibility).toBe('standard')
    expect(replay?.visibility).toBe('standard')
  })

  it('sorts enablement toggles first within their category group', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        loggingSchemaSetting('logging.audit.log_format', { kind: 'enum', values: ['json', 'json_lines'] }, {}),
        loggingSchemaSetting('logging.audit.enabled', { kind: 'boolean' }, {}),
        loggingSchemaSetting('logging.audit.log_level', { kind: 'enum', values: ['info', 'warn'] }, {})
      ]
    }

    const settings = createConfigurationAuditSettingsFromSchema(schema, undefined)

    const auditGroup = settings.settings.filter((setting) => setting.categoryId === 'logs-audit')
    expect(auditGroup.map((setting) => setting.id)).toEqual([
      'logging.audit.enabled',
      'logging.audit.log_format',
      'logging.audit.log_level'
    ])
    expect(auditGroup[0]).toMatchObject({ id: 'logging.audit.enabled', settingOrder: 10 })
  })

  it('keeps core logging visible and security audit under advanced settings', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        loggingSchemaSetting(
          'logging.retention_ttl_secs',
          { kind: 'integer' },
          {
            apply_mode: 'dynamic_apply',
            restart_scope: 'none',
            visibility: 'advanced'
          }
        ),
        {
          canonical_path: 'logging.audit.enabled',
          owner: 'built_in' as const,
          source: { kind: 'built_in' as const },
          value_schema: { kind: 'boolean' },
          support: 'supported' as const,
          control_surfaces: ['config_file'],
          apply_mode: 'static_on_load' as const,
          restart_scope: 'process_restart' as const,
          visibility: 'advanced' as const
        }
      ]
    }

    const auditSettings = createConfigurationAuditSettingsFromSchema(schema)
    const loggingSetting = auditSettings.settings.find((s) => s.id === 'logging.retention_ttl_secs')
    const auditSetting = auditSettings.settings.find((s) => s.id === 'logging.audit.enabled')

    expect(loggingSetting?.visibility).toBe('standard')
    expect(auditSetting?.visibility).toBe('advanced')
  })

  it('keeps metadata-only artifact controls from authorizing payload-related writes', () => {
    const schema: RuntimeConfigSchemaReference = {
      settings: [
        loggingSchemaSetting('logging.artifact.capture_mode', {
          kind: 'enum',
          values: ['metadata_only', 'redacted_artifacts']
        }),
        loggingSchemaSetting('logging.artifact.byte_limit_bytes', { kind: 'integer' })
      ]
    }
    const controlState = {
      settings: {
        'logging.artifact.byte_limit_bytes': {
          enabled: false,
          reason: 'Artifact capture is unavailable while capture mode is metadata_only.',
          source: 'runtime',
          write_policy: 'reject_when_disabled'
        }
      }
    } satisfies RuntimeConfigControlStatePayload

    expect(() =>
      mergeConfigurationIntoMeshConfig(
        { logging: { artifact: { capture_mode: 'metadata_only' } } },
        {
          values: {
            'logging.artifact.byte_limit_bytes': '4096',
            'logging.artifact.payload': 'not-a-schema-setting'
          },
          nodes: [],
          assigns: [],
          catalog: []
        },
        schema,
        { controlState }
      )
    ).toThrow(/logging\.artifact\.byte_limit_bytes/)
  })

  it('formats invalid and unsupported logging diagnostics without losing typed fields', () => {
    const diagnostics = [
      {
        code: 'invalid_value',
        severity: 'error',
        source: 'config',
        schema_source: 'built_in',
        canonical_path: 'logging.retention_ttl_secs',
        message: 'Retention must be at least 60 seconds.',
        help: 'Increase logging.retention_ttl_secs.'
      },
      {
        code: 'unsupported_setting',
        severity: 'warning',
        source: 'config',
        schema_source: 'built_in',
        canonical_path: 'logging.legacy_payload_access',
        message: 'This logging setting is unsupported.'
      }
    ] satisfies readonly RuntimeControlDiagnostic[]

    expect(formatConfigDiagnostics(diagnostics)).toContain('logging.retention_ttl_secs')
    expect(formatConfigDiagnostics(diagnostics)).toContain('unsupported')
    expect(diagnostics[1]?.schema_source).toBe('built_in')
  })
  it('places audit settings in a dedicated category and config section', () => {
    const audit = createConfigurationAuditSettingsFromSchema(AUDIT_SCHEMA)
    const configuration = adaptStatusToConfiguration(STATUS_PAYLOAD, [], undefined, AUDIT_SCHEMA)

    expect(audit.categories).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: 'logs-audit',
          label: 'Security Audit',
          tomlSection: 'logging.audit'
        })
      ])
    )
    expect(audit.settings).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: 'logging.audit.enabled',
          categoryId: 'logs-audit',
          tomlSection: 'logging.audit',
          icon: 'shield'
        })
      ])
    )
    expect(configuration.audit?.settings.map((setting) => setting.id)).toEqual([
      'logging.audit.enabled',
      'logging.audit.log_format'
    ])
    expect(configuration.defaults.settings.map((setting) => setting.id)).toEqual(
      expect.arrayContaining(['logging.audit.enabled', 'logging.audit.log_format'])
    )
  })
})
