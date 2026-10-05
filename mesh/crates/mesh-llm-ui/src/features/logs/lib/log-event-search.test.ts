import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogAuditEntry, LogRequest } from '@/features/logs/api/schemas'
import { formatEndpointId } from '@/features/logs/lib/log-client-info'
import type { LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'
import { logEventSearchText } from '@/features/logs/lib/log-event-search'

const ENDPOINT_ID = '9f0c4cbe8cb7a8d5d577c20e50ef03fd2f63a2e7fd9897c155823bcbb281bb04'
const OCCURRED_AT = '2026-08-08T12:00:00Z'

const REQUEST: LogRequest = {
  requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000001'),
  outcome: 'completed',
  createdAt: OCCURRED_AT,
  terminalAt: '2026-08-08T12:00:01Z',
  route: 'chat_completions',
  model: 'Qwen3',
  provider: 'mesh',
  engine: 'skippy',
  statusCode: 200,
  source: 'durable',
  callerEndpointId: ENDPOINT_ID,
  callerAddr: '203.0.113.24:48712',
  callerPathType: 'remote_quic_http'
}

const AUDIT: LogAuditEntry = {
  entryId: 'audit-peer-1',
  occurredAt: OCCURRED_AT,
  source: 'mesh',
  code: 'gossip_policy_rejected',
  severity: 'warning',
  sequence: 7,
  subjectKind: 'mesh_peer',
  subjectId: ENDPOINT_ID,
  remoteAddr: '203.0.113.24:48712',
  pathType: 'direct'
}

const REQUEST_ROW: LogEventLedgerRow = {
  type: 'request',
  id: `request:${REQUEST.requestId.toString()}`,
  occurredAt: REQUEST.createdAt,
  category: 'requests',
  request: REQUEST
}

const AUDIT_ROW: LogEventLedgerRow = {
  type: 'audit',
  id: `audit:${AUDIT.entryId}`,
  occurredAt: AUDIT.occurredAt,
  category: 'gossip',
  audit: AUDIT
}

describe('log event search text', () => {
  it('indexes the full and displayed endpoint IDs for request callers', () => {
    const searchText = logEventSearchText(REQUEST_ROW)

    expect(searchText).toContain(ENDPOINT_ID)
    expect(searchText).toContain(formatEndpointId(ENDPOINT_ID))
  })

  it('indexes the full and displayed endpoint IDs for peer audit subjects', () => {
    const searchText = logEventSearchText(AUDIT_ROW)

    expect(searchText).toContain(ENDPOINT_ID)
    expect(searchText).toContain(formatEndpointId(ENDPOINT_ID))
  })
})
