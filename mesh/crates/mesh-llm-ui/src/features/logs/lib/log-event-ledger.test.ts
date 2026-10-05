import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogAuditEntry, LogRequest } from '@/features/logs/api/schemas'
import {
  classifyAuditCategory,
  filterLogEventRows,
  logEventCategoryOptions,
  mergeLogEventWindow
} from './log-event-ledger'

const REQUEST_ID = '00000000-0000-4000-8000-000000000001'

function request(createdAt: string, outcome: LogRequest['outcome'] = 'completed'): LogRequest {
  return {
    requestId: LogRequestId.parse(REQUEST_ID),
    outcome,
    createdAt,
    terminalAt: outcome === 'active' ? undefined : createdAt,
    route: 'chat_completions',
    model: 'Qwen3',
    provider: 'mesh',
    engine: 'skippy',
    statusCode: outcome === 'completed' ? 200 : undefined,
    source: outcome === 'active' ? 'active' : 'durable'
  }
}

function audit(code: string, occurredAt: string, sequence = 1): LogAuditEntry {
  return {
    entryId: `audit-${sequence}`,
    occurredAt,
    source: 'mesh',
    code,
    severity: 'info',
    sequence
  }
}

describe('log event ledger model', () => {
  it.each([
    ['runtime_ready', 'system'],
    ['runtime_discovery_join_succeeded', 'system'],
    ['mesh_auto_join_succeeded', 'system'],
    ['mesh_quic_inbound_accepted', 'quic'],
    ['mesh_control_alpn_rejected', 'quic'],
    ['gossip_direct_peer_promoted', 'gossip'],
    ['iroh_test_event', 'system'],
    ['mesh_iroh_test_event', 'system'],
    ['runtime_not_iroh_transport', 'system']
  ] as const)('classifies %s as %s without inferring Iroh from audit-code text', (code, category) => {
    expect(classifyAuditCategory(code)).toBe(category)
  })

  it('merges independent windows newest-first, prefers active duplicates, breaks ties deterministically, and applies the bound', () => {
    const rows = mergeLogEventWindow(
      [
        request('2026-08-08T12:00:00Z'),
        request('2026-08-08T12:00:00Z', 'active'),
        { ...request('2026-08-08T11:58:00Z'), requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000002') }
      ],
      [audit('runtime_ready', '2026-08-08T12:01:00Z', 2), audit('gossip_peer_removed', '2026-08-08T12:00:00Z', 1)],
      3
    )

    expect(rows.map((row) => `${row.type}:${row.category}:${row.id}`)).toEqual([
      'audit:system:audit:audit-2',
      `request:requests:request:${REQUEST_ID}`,
      'audit:gossip:audit:audit-1'
    ])
    expect(rows[1]?.type).toBe('request')
    if (rows[1]?.type === 'request') expect(rows[1].request.outcome).toBe('active')
  })

  it('orders lexically misleading offsets newest-first by instant', () => {
    // Given
    const requests = [request('2026-08-04T12:00:00Z')]
    const audits = [audit('runtime_ready', '2026-08-04T10:30:00-02:00')]

    // When
    const rows = mergeLogEventWindow(requests, audits)

    // Then
    expect(rows.map((row) => row.id)).toEqual(['audit:audit-1', `request:${REQUEST_ID}`])
  })

  it('preserves type and ID tie-breakers for equal instants across offsets', () => {
    // Given
    const requests = [request('2026-08-04T10:00:00-02:00')]
    const audits = [
      audit('runtime_ready', '2026-08-04T13:00:00+01:00', 2),
      audit('gossip_peer_removed', '2026-08-04T12:00:00Z', 1)
    ]

    // When
    const rows = mergeLogEventWindow(requests, audits)

    // Then
    expect(rows.map((row) => row.id)).toEqual([`request:${REQUEST_ID}`, 'audit:audit-1', 'audit:audit-2'])
  })

  it('omits Iroh from filters when current audit rows have no authoritative category', () => {
    const ordinaryRows = mergeLogEventWindow(
      [request('2026-08-08T12:00:00Z')],
      [audit('runtime_discovery_join_succeeded', '2026-08-08T11:59:00Z')]
    )
    const irohNamedRows = mergeLogEventWindow([], [audit('iroh_test_event', '2026-08-08T12:01:00Z')])

    const ordinaryOptions = logEventCategoryOptions(ordinaryRows)
    expect(ordinaryOptions.map((option) => option.value)).toEqual(['requests', 'system', 'quic', 'gossip'])
    expect(logEventCategoryOptions(irohNamedRows).map((option) => option.value)).toEqual([
      'requests',
      'system',
      'quic',
      'gossip'
    ])
    expect(irohNamedRows[0]?.category).toBe('system')
  })

  it('filters the merged window with multi-select union semantics', () => {
    const rows = mergeLogEventWindow(
      [request('2026-08-08T12:00:00Z')],
      [audit('runtime_ready', '2026-08-08T11:59:00Z'), audit('gossip_peer_removed', '2026-08-08T11:58:00Z', 2)]
    )

    expect(filterLogEventRows(rows, new Set(['requests', 'gossip'])).map((row) => row.category)).toEqual([
      'requests',
      'gossip'
    ])
  })
})
