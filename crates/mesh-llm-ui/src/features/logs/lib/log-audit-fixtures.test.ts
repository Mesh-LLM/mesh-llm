import { describe, expect, it } from 'vitest'
import { parseLogAuditPage } from '@/features/logs/api/schemas'
import { HARNESS_LOG_AUDIT_FIXTURES } from '@/features/logs/lib/log-fixtures'

const EXPECTED_AUDIT_CODES = [
  'runtime_startup_started',
  'runtime_startup_failed',
  'runtime_ready',
  'runtime_shutdown_started',
  'runtime_model_load_started',
  'runtime_model_ready',
  'runtime_model_load_failed',
  'runtime_model_unloaded',
  'skippy_native_runtime_startup_started',
  'skippy_native_runtime_ready',
  'skippy_native_runtime_startup_failed',
  'skippy_native_runtime_shutdown_started',
  'skippy_native_model_open_started',
  'skippy_native_model_open_finished',
  'skippy_native_model_open_failed',
  'runtime_config_apply_started',
  'runtime_config_apply_accepted',
  'runtime_config_apply_rejected',
  'runtime_config_diagnostics_clean',
  'runtime_config_diagnostics_info',
  'runtime_config_diagnostics_warning',
  'runtime_config_diagnostics_error',
  'runtime_discovery_decision_join',
  'runtime_discovery_decision_start_new',
  'runtime_discovery_join_started',
  'runtime_discovery_join_succeeded',
  'runtime_discovery_join_failed',
  'runtime_discovery_failed',
  'runtime_local_target_added',
  'runtime_local_target_removed',
  'runtime_local_serving_ready',
  'runtime_local_serving_unavailable',
  'mesh_quic_handler_failed',
  'mesh_quic_inbound_accepted',
  'mesh_control_handler_failed',
  'mesh_control_alpn_rejected',
  'mesh_control_connection_accepted',
  'gossip_policy_rejected',
  'gossip_direct_peer_promoted',
  'gossip_incompatible_version_rejected',
  'gossip_peer_removed',
  'mesh_auto_join_succeeded',
  'mesh_auto_join_failed',
  'cli_command_started',
  'cli_command_completed',
  'cli_command_failed',
  'cli_command_rejected',
  'logging_cleanup_completed',
  'logging_cleanup_failed',
  'audit_error',
  'uncategorized_bus_record'
] as const

describe('operational audit harness fixtures', () => {
  it('provides the exact schema-valid operational audit vocabularies', () => {
    expect(new Set(HARNESS_LOG_AUDIT_FIXTURES.map((entry) => entry.source))).toEqual(
      new Set(['logging_service', 'logs_api', 'runtime', 'mesh', 'cli'])
    )
    expect(new Set(HARNESS_LOG_AUDIT_FIXTURES.map((entry) => entry.severity))).toEqual(
      new Set(['info', 'warning', 'error'])
    )
    expect(new Set(HARNESS_LOG_AUDIT_FIXTURES.map((entry) => entry.code))).toEqual(new Set(EXPECTED_AUDIT_CODES))
    expect(new Set(HARNESS_LOG_AUDIT_FIXTURES.map((entry) => entry.entryId)).size).toBe(
      HARNESS_LOG_AUDIT_FIXTURES.length
    )
    expect(HARNESS_LOG_AUDIT_FIXTURES.every((entry) => entry.sequence > 0 && entry.code.length > 0)).toBe(true)
    expect(parseLogAuditPage({ items: HARNESS_LOG_AUDIT_FIXTURES, nextCursor: null }).items).toEqual(
      HARNESS_LOG_AUDIT_FIXTURES
    )
  })

  it('keeps operational audits bounded to public DTO fields and private-content free', () => {
    const serialized = JSON.stringify(HARNESS_LOG_AUDIT_FIXTURES).toLowerCase()

    for (const entry of HARNESS_LOG_AUDIT_FIXTURES) {
      expect(Object.keys(entry).sort()).toEqual(['code', 'entryId', 'occurredAt', 'sequence', 'severity', 'source'])
      expect(entry.code).toMatch(/^[a-z][a-z0-9_]{0,47}$/)
    }
    for (const privateFragment of [
      'http://',
      'https://',
      '/private',
      'token=',
      'bearer ',
      'prompt=',
      'peer=',
      'node=',
      'config.toml',
      'secret'
    ]) {
      expect(serialized).not.toContain(privateFragment)
    }
  })
})
