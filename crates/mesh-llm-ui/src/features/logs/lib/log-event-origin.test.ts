import { describe, expect, it } from 'vitest'
import { LogRequestId } from '@/features/logs/api/ids'
import type { LogRequest } from '@/features/logs/api/schemas'
import type { LogEventLedgerRow } from '@/features/logs/lib/log-event-ledger'
import { logEventOriginLabel, logEventOriginLines } from '@/features/logs/lib/log-event-origin'

const CREATED_AT = '2026-08-08T12:00:00Z'

const BASE_REQUEST: LogRequest = {
  requestId: LogRequestId.parse('00000000-0000-4000-8000-000000000001'),
  outcome: 'completed',
  createdAt: CREATED_AT,
  terminalAt: undefined,
  route: undefined,
  model: undefined,
  provider: undefined,
  engine: undefined,
  statusCode: undefined,
  source: 'durable'
}

function requestRow(request: LogRequest): LogEventLedgerRow {
  return {
    type: 'request',
    id: `request:${request.requestId.toString()}`,
    occurredAt: request.createdAt,
    category: 'requests',
    request
  }
}

describe('logEventOriginLabel', () => {
  it('returns a placeholder for a request without user-facing origin metadata', () => {
    const row = requestRow(BASE_REQUEST)

    const origin = logEventOriginLabel(row)

    expect(origin).toBe('—')
  })

  it('formats a shortened endpoint and remote QUIC path in a request origin', () => {
    const row = requestRow({
      ...BASE_REQUEST,
      callerEndpointId: 'endpoint-1234567890',
      callerPathType: 'remote_quic_http'
    })

    const origin = logEventOriginLabel(row)

    expect(origin).toBe('endp…7890 · Remote QUIC HTTP')
  })

  it('separates request identity from its path while preserving the flat origin label', () => {
    // Given
    const row = requestRow({
      ...BASE_REQUEST,
      provider: 'mesh',
      callerAddr: '127.0.0.1:65251',
      callerPathType: 'local_http'
    })

    // When
    const lines = logEventOriginLines(row)

    // Then
    expect(lines).toEqual({ identity: 'mesh · 127.0.0.1:65251', path: 'Local HTTP' })
    expect(logEventOriginLabel(row)).toBe('mesh · 127.0.0.1:65251 · Local HTTP')
  })

  it.each<[string, LogRequest, string]>([
    ['mesh-only origin', { ...BASE_REQUEST, provider: 'mesh' }, 'mesh'],
    ['internal-engine-only origin', { ...BASE_REQUEST, engine: 'skippy' }, '—'],
    ['caller-only origin', { ...BASE_REQUEST, callerAddr: '203.0.113.24:48712' }, '203.0.113.24:48712'],
    [
      'fully populated origin',
      {
        ...BASE_REQUEST,
        provider: 'mesh',
        engine: 'raw_ingress',
        callerAddr: '203.0.113.24:48712'
      },
      'mesh · 203.0.113.24:48712'
    ]
  ])('formats a %s with single separators', (_caseName, request, expected) => {
    const row = requestRow(request)

    const origin = logEventOriginLabel(row)

    expect(origin).toBe(expected)
  })

  it('omits internal provider and lifecycle fields from a request origin', () => {
    const row = requestRow({
      ...BASE_REQUEST,
      provider: 'openai_frontend',
      engine: 'raw_ingress'
    })

    const origin = logEventOriginLabel(row)

    expect(origin).toBe('—')
  })

  it('returns the audit source for an audit ledger row', () => {
    const row: LogEventLedgerRow = {
      type: 'audit',
      id: 'audit:audit-entry-1',
      occurredAt: CREATED_AT,
      category: 'system',
      audit: {
        entryId: 'audit-entry-1',
        occurredAt: CREATED_AT,
        source: 'runtime',
        code: 'runtime_started',
        sequence: 1
      }
    }

    const origin = logEventOriginLabel(row)
    const lines = logEventOriginLines(row)

    expect(origin).toBe('runtime')
    expect(lines).toEqual({ identity: 'runtime' })
  })
})
