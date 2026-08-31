import { describe, expect, it } from 'vitest'
import { parseLogAuditPage } from '@/features/logs/api/schemas'
import type { LogAuditEntry } from '@/features/logs/api/schemas'
import { HARNESS_LOG_AUDIT_FIXTURES } from '@/features/logs/lib/log-fixtures'

const TYPED_AUDIT_FIXTURE: LogAuditEntry = {
  entryId: 'audit-typed-context',
  occurredAt: '2026-08-08T12:02:00Z',
  source: 'cli',
  code: 'cli_command_completed',
  severity: 'info',
  sequence: 100,
  contextVersion: 1,
  operationId: 'runtime-instance-7',
  requestId: '00000000-0000-4000-8000-000000000001',
  commandSummary: 'mesh-llm load name [REDACTED] --root-relay [REDACTED]'
}

const AUDIT_FIXTURES: readonly LogAuditEntry[] = [...HARNESS_LOG_AUDIT_FIXTURES, TYPED_AUDIT_FIXTURE]

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

  it('uses producer-backed mesh reasons while accepting bounded historical reason strings', () => {
    const handlerReasons = HARNESS_LOG_AUDIT_FIXTURES.filter(
      (entry) => entry.code === 'mesh_quic_handler_failed' || entry.code === 'mesh_control_handler_failed'
    ).map((entry) => entry.reasonCode)
    const peerRemoval = HARNESS_LOG_AUDIT_FIXTURES.find((entry) => entry.code === 'gossip_peer_removed')

    expect(handlerReasons).toEqual(['internal', 'internal'])
    expect(peerRemoval?.reasonCode).toBe('reconnect_failed')
    expect(
      parseLogAuditPage({
        items: [
          {
            entryId: 'audit-historical-reason',
            occurredAt: '2026-08-08T12:01:00Z',
            source: 'mesh',
            code: 'mesh_quic_handler_failed',
            severity: 'warning',
            sequence: 1,
            reasonCode: 'historical_unknown_reason'
          }
        ],
        nextCursor: null
      }).items[0]?.reasonCode
    ).toBe('historical_unknown_reason')
  })

  it('keeps operational audits bounded to public DTO fields and private-content free', () => {
    expect(parseLogAuditPage({ items: AUDIT_FIXTURES, nextCursor: null }).items).toEqual(AUDIT_FIXTURES)
    const serialized = JSON.stringify(AUDIT_FIXTURES).toLowerCase()
    const allowedFields = new Set([
      'code',
      'commandSummary',
      'contextVersion',
      'durationMs',
      'entryId',
      'numericSummaries',
      'occurredAt',
      'operationId',
      'outcome',
      'pathType',
      'reasonCode',
      'remoteAddr',
      'requestId',
      'sequence',
      'severity',
      'source',
      'subjectId',
      'subjectKind'
    ])

    for (const entry of AUDIT_FIXTURES) {
      expect(Object.keys(entry).every((field) => allowedFields.has(field))).toBe(true)
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

  it('covers direct, relay, repeated, and legacy mesh audit identities', () => {
    const peerEntries = HARNESS_LOG_AUDIT_FIXTURES.filter((entry) => entry.subjectKind === 'mesh_peer')
    const directEntries = peerEntries.filter((entry) => entry.pathType === 'direct')
    const relayEntries = peerEntries.filter((entry) => entry.pathType === 'relay')
    const peerCounts = new Map<string, number>()
    for (const entry of peerEntries) {
      if (entry.subjectId) peerCounts.set(entry.subjectId, (peerCounts.get(entry.subjectId) ?? 0) + 1)
    }

    expect(peerEntries.every((entry) => /^[a-f0-9]{64}$/.test(entry.subjectId ?? ''))).toBe(true)
    expect(directEntries.some((entry) => entry.remoteAddr !== undefined)).toBe(true)
    expect(relayEntries.every((entry) => entry.remoteAddr === undefined)).toBe(true)
    expect([...peerCounts.values()].some((count) => count > 1)).toBe(true)
    expect(HARNESS_LOG_AUDIT_FIXTURES.some((entry) => entry.code === 'mesh_auto_join_succeeded')).toBe(true)
    expect(
      HARNESS_LOG_AUDIT_FIXTURES.find((entry) => entry.code === 'mesh_auto_join_succeeded')?.subjectKind
    ).toBeUndefined()
  })
})
