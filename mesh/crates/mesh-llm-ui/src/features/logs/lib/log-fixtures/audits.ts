import type { LogAuditEntry, LogAuditSeverity, LogAuditSource } from '@/features/logs/api/schemas'
import { harnessTimestamp } from './support'

type AuditDefinition = {
  readonly source: LogAuditSource
  readonly code: string
  readonly severity: LogAuditSeverity
  readonly subjectKind?: LogAuditEntry['subjectKind']
  readonly subjectId?: string
  readonly remoteAddr?: string
  readonly pathType?: LogAuditEntry['pathType']
  readonly reasonCode?: string
  readonly outcome?: string
  readonly durationMs?: number
  readonly numericSummaries?: Readonly<Record<string, number>>
}

const DIRECT_PEER_ID = '9f0c4cbe8cb7a8d5d577c20e50ef03fd2f63a2e7fd9897c155823bcbb281bb04'
const RELAY_PEER_ID = 'de2f01895ab34c2c8f5d97a703311f5c7279082eef191644c397d2175210aa9b'

const VISIBLE_AUDIT_DEFINITIONS: readonly AuditDefinition[] = [
  { source: 'runtime', code: 'runtime_startup_started', severity: 'info' },
  { source: 'runtime', code: 'skippy_native_runtime_ready', severity: 'info' },
  { source: 'runtime', code: 'runtime_config_diagnostics_warning', severity: 'warning' },
  { source: 'runtime', code: 'runtime_discovery_join_succeeded', severity: 'info' },
  { source: 'runtime', code: 'runtime_local_serving_ready', severity: 'info' },
  {
    source: 'mesh',
    code: 'mesh_quic_inbound_accepted',
    severity: 'info',
    subjectKind: 'mesh_peer',
    subjectId: DIRECT_PEER_ID,
    remoteAddr: '203.0.113.24:48712',
    pathType: 'direct',
    outcome: 'accepted',
    numericSummaries: { protocol_gen: 1 }
  },
  {
    source: 'mesh',
    code: 'mesh_control_alpn_rejected',
    severity: 'warning',
    subjectKind: 'mesh_peer',
    subjectId: RELAY_PEER_ID,
    pathType: 'relay',
    outcome: 'rejected',
    reasonCode: 'alpn_unsupported'
  },
  {
    source: 'mesh',
    code: 'gossip_direct_peer_promoted',
    severity: 'info',
    subjectKind: 'mesh_peer',
    subjectId: DIRECT_PEER_ID,
    remoteAddr: '203.0.113.24:48712',
    pathType: 'direct',
    outcome: 'promoted',
    numericSummaries: { direct_peers: 4 }
  },
  { source: 'mesh', code: 'mesh_auto_join_failed', severity: 'warning' },
  { source: 'cli', code: 'cli_command_started', severity: 'info' },
  { source: 'cli', code: 'cli_command_completed', severity: 'info' },
  { source: 'cli', code: 'cli_command_failed', severity: 'warning' },
  { source: 'cli', code: 'cli_command_rejected', severity: 'warning' },
  { source: 'logs_api', code: 'logging_cleanup_completed', severity: 'info' },
  { source: 'logs_api', code: 'logging_cleanup_failed', severity: 'error' },
  { source: 'logging_service', code: 'audit_error', severity: 'error' }
]

const ADDITIONAL_AUDIT_DEFINITIONS: readonly AuditDefinition[] = [
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
  {
    source: 'mesh',
    code: 'mesh_quic_handler_failed',
    severity: 'warning',
    subjectKind: 'mesh_peer',
    subjectId: RELAY_PEER_ID,
    pathType: 'relay',
    outcome: 'failed',
    reasonCode: 'internal',
    durationMs: 812
  },
  {
    source: 'mesh',
    code: 'mesh_control_handler_failed',
    severity: 'warning',
    subjectKind: 'mesh_peer',
    subjectId: DIRECT_PEER_ID,
    remoteAddr: '203.0.113.24:48712',
    pathType: 'direct',
    outcome: 'failed',
    reasonCode: 'internal',
    durationMs: 94
  },
  {
    source: 'mesh',
    code: 'mesh_control_connection_accepted',
    severity: 'info',
    subjectKind: 'mesh_peer',
    subjectId: DIRECT_PEER_ID,
    remoteAddr: '203.0.113.24:48712',
    pathType: 'direct',
    outcome: 'accepted',
    numericSummaries: { protocol_gen: 1 }
  },
  {
    source: 'mesh',
    code: 'gossip_policy_rejected',
    severity: 'warning',
    subjectKind: 'mesh_peer',
    subjectId: DIRECT_PEER_ID,
    remoteAddr: '203.0.113.24:48712',
    pathType: 'direct',
    outcome: 'rejected',
    reasonCode: 'owner_attestation_required'
  },
  {
    source: 'mesh',
    code: 'gossip_incompatible_version_rejected',
    severity: 'warning',
    subjectKind: 'mesh_peer',
    subjectId: RELAY_PEER_ID,
    pathType: 'relay',
    outcome: 'rejected',
    reasonCode: 'protocol_version_unsupported',
    numericSummaries: { peer_gen: 0, local_gen: 1 }
  },
  {
    source: 'mesh',
    code: 'gossip_peer_removed',
    severity: 'info',
    subjectKind: 'mesh_peer',
    subjectId: DIRECT_PEER_ID,
    remoteAddr: '203.0.113.24:48712',
    pathType: 'direct',
    outcome: 'removed',
    reasonCode: 'reconnect_failed',
    numericSummaries: { direct_peers: 3 }
  },
  { source: 'mesh', code: 'mesh_auto_join_succeeded', severity: 'info' },
  { source: 'logging_service', code: 'uncategorized_bus_record', severity: 'warning' }
]

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
