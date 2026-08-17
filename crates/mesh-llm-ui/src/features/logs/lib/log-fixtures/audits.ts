import type { LogAuditEntry, LogAuditSeverity, LogAuditSource } from '@/features/logs/api/schemas'
import { harnessTimestamp } from './support'

type AuditDefinition = {
  readonly source: LogAuditSource
  readonly code: string
  readonly severity: LogAuditSeverity
}

const VISIBLE_AUDIT_DEFINITIONS = [
  { source: 'runtime', code: 'runtime_startup_started', severity: 'info' },
  { source: 'runtime', code: 'skippy_native_runtime_ready', severity: 'info' },
  { source: 'runtime', code: 'runtime_config_diagnostics_warning', severity: 'warning' },
  { source: 'runtime', code: 'runtime_discovery_join_succeeded', severity: 'info' },
  { source: 'runtime', code: 'runtime_local_serving_ready', severity: 'info' },
  { source: 'mesh', code: 'mesh_quic_inbound_accepted', severity: 'info' },
  { source: 'mesh', code: 'mesh_control_alpn_rejected', severity: 'warning' },
  { source: 'mesh', code: 'gossip_direct_peer_promoted', severity: 'info' },
  { source: 'mesh', code: 'mesh_auto_join_failed', severity: 'warning' },
  { source: 'cli', code: 'cli_command_started', severity: 'info' },
  { source: 'cli', code: 'cli_command_completed', severity: 'info' },
  { source: 'cli', code: 'cli_command_failed', severity: 'warning' },
  { source: 'cli', code: 'cli_command_rejected', severity: 'warning' },
  { source: 'logs_api', code: 'logging_cleanup_completed', severity: 'info' },
  { source: 'logs_api', code: 'logging_cleanup_failed', severity: 'error' },
  { source: 'logging_service', code: 'audit_error', severity: 'error' }
] satisfies readonly AuditDefinition[]

const ADDITIONAL_AUDIT_DEFINITIONS = [
  { source: 'runtime', code: 'runtime_startup_failed', severity: 'warning' },
  { source: 'runtime', code: 'runtime_ready', severity: 'info' },
  { source: 'runtime', code: 'runtime_shutdown_started', severity: 'info' },
  { source: 'runtime', code: 'runtime_model_load_started', severity: 'info' },
  { source: 'runtime', code: 'runtime_model_ready', severity: 'info' },
  { source: 'runtime', code: 'runtime_model_load_failed', severity: 'warning' },
  { source: 'runtime', code: 'runtime_model_unloaded', severity: 'info' },
  { source: 'runtime', code: 'skippy_native_runtime_startup_started', severity: 'info' },
  { source: 'runtime', code: 'skippy_native_runtime_startup_failed', severity: 'warning' },
  { source: 'runtime', code: 'skippy_native_runtime_shutdown_started', severity: 'info' },
  { source: 'runtime', code: 'skippy_native_model_open_started', severity: 'info' },
  { source: 'runtime', code: 'skippy_native_model_open_finished', severity: 'info' },
  { source: 'runtime', code: 'skippy_native_model_open_failed', severity: 'warning' },
  { source: 'runtime', code: 'runtime_config_apply_started', severity: 'info' },
  { source: 'runtime', code: 'runtime_config_apply_accepted', severity: 'info' },
  { source: 'runtime', code: 'runtime_config_apply_rejected', severity: 'warning' },
  { source: 'runtime', code: 'runtime_config_diagnostics_clean', severity: 'info' },
  { source: 'runtime', code: 'runtime_config_diagnostics_info', severity: 'info' },
  { source: 'runtime', code: 'runtime_config_diagnostics_error', severity: 'warning' },
  { source: 'runtime', code: 'runtime_discovery_decision_join', severity: 'info' },
  { source: 'runtime', code: 'runtime_discovery_decision_start_new', severity: 'info' },
  { source: 'runtime', code: 'runtime_discovery_join_started', severity: 'info' },
  { source: 'runtime', code: 'runtime_discovery_join_failed', severity: 'warning' },
  { source: 'runtime', code: 'runtime_discovery_failed', severity: 'warning' },
  { source: 'runtime', code: 'runtime_local_target_added', severity: 'info' },
  { source: 'runtime', code: 'runtime_local_target_removed', severity: 'info' },
  { source: 'runtime', code: 'runtime_local_serving_unavailable', severity: 'info' },
  { source: 'mesh', code: 'mesh_quic_handler_failed', severity: 'warning' },
  { source: 'mesh', code: 'mesh_control_handler_failed', severity: 'warning' },
  { source: 'mesh', code: 'mesh_control_connection_accepted', severity: 'info' },
  { source: 'mesh', code: 'gossip_policy_rejected', severity: 'warning' },
  { source: 'mesh', code: 'gossip_incompatible_version_rejected', severity: 'warning' },
  { source: 'mesh', code: 'gossip_peer_removed', severity: 'info' },
  { source: 'mesh', code: 'mesh_auto_join_succeeded', severity: 'info' },
  { source: 'logging_service', code: 'uncategorized_bus_record', severity: 'warning' }
] satisfies readonly AuditDefinition[]

const AUDIT_DEFINITIONS = [...VISIBLE_AUDIT_DEFINITIONS, ...ADDITIONAL_AUDIT_DEFINITIONS]

export const HARNESS_LOG_AUDIT_FIXTURES: readonly LogAuditEntry[] = AUDIT_DEFINITIONS.map((definition, index) => {
  const sequence = AUDIT_DEFINITIONS.length - index
  return {
    entryId: `audit-${sequence.toString().padStart(4, '0')}`,
    occurredAt: harnessTimestamp(index * 2.5 + 0.5),
    ...definition,
    sequence
  }
})
