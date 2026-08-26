import { describe, expect, it } from 'vitest'
import { parseAuditEntry, parseLogRequest } from '@/features/logs/api/schemas'

const ENDPOINT_ID = '9f0c4cbe8cb7a8d5d577c20e50ef03fd2f63a2e7fd9897c155823bcbb281bb04'
const TIMESTAMP = '2026-02-20T14:22:12.944Z'

function requestDto(requestId: string) {
  return {
    requestId,
    outcome: 'completed',
    createdAt: TIMESTAMP,
    terminalAt: TIMESTAMP,
    route: 'chat_completions',
    model: 'Qwen3-30B-A3B-Q4_K_M.gguf',
    provider: 'openai_frontend',
    engine: 'skippy',
    statusCode: 200,
    source: 'durable'
  }
}

describe('client information schemas', () => {
  it('retains optional peer fields on mesh audit entries', () => {
    const parsed = parseAuditEntry({
      entryId: 'audit-gossip-1',
      occurredAt: '2026-02-20T14:22:08.301Z',
      sequence: 18,
      severity: 'info',
      code: 'gossip_peer_discovered',
      source: 'mesh',
      subjectKind: 'mesh_peer',
      subjectId: ENDPOINT_ID,
      remoteAddr: '203.0.113.24:48712',
      pathType: 'direct'
    })

    expect(parsed).toMatchObject({
      subjectKind: 'mesh_peer',
      subjectId: ENDPOINT_ID,
      remoteAddr: '203.0.113.24:48712',
      pathType: 'direct'
    })
  })

  it('accepts relay peer entries and legacy audit entries without client fields', () => {
    const relay = parseAuditEntry({
      entryId: 'audit-quic-1',
      occurredAt: '2026-02-20T14:22:10.944Z',
      sequence: 19,
      severity: 'warning',
      code: 'quic_path_degraded',
      source: 'mesh',
      subjectKind: 'mesh_peer',
      subjectId: ENDPOINT_ID,
      pathType: 'relay'
    })
    const legacy = parseAuditEntry({
      entryId: 'audit-legacy-1',
      occurredAt: '2026-02-20T14:22:11.944Z',
      sequence: 20,
      severity: 'info',
      code: 'auto_join_started',
      source: 'mesh'
    })

    expect(relay).toMatchObject({ subjectKind: 'mesh_peer', subjectId: ENDPOINT_ID, pathType: 'relay' })
    expect(legacy).not.toHaveProperty('subjectKind')
    expect(legacy).not.toHaveProperty('remoteAddr')
    expect(legacy).not.toHaveProperty('pathType')
  })

  it('retains optional caller fields while accepting legacy requests', () => {
    const direct = parseLogRequest({
      ...requestDto('00000000-0000-4000-8000-000000000101'),
      callerEndpointId: ENDPOINT_ID,
      callerAddr: '203.0.113.24:48712',
      callerPathType: 'remote_quic_http'
    })
    const relay = parseLogRequest({
      ...requestDto('00000000-0000-4000-8000-000000000102'),
      callerEndpointId: ENDPOINT_ID,
      callerPathType: 'relay'
    })
    const legacy = parseLogRequest(requestDto('00000000-0000-4000-8000-000000000103'))

    expect(direct).toMatchObject({
      callerEndpointId: ENDPOINT_ID,
      callerAddr: '203.0.113.24:48712',
      callerPathType: 'remote_quic_http'
    })
    expect(relay).toMatchObject({ callerEndpointId: ENDPOINT_ID, callerPathType: 'relay' })
    expect(legacy).not.toHaveProperty('callerEndpointId')
    expect(legacy).not.toHaveProperty('callerAddr')
    expect(legacy).not.toHaveProperty('callerPathType')
  })

  it('rejects stage transport as a top-level request caller', () => {
    expect(() =>
      parseLogRequest({
        ...requestDto('00000000-0000-4000-8000-000000000104'),
        callerEndpointId: ENDPOINT_ID,
        callerAddr: '203.0.113.24:48712',
        callerPathType: 'remote_quic_stage'
      })
    ).toThrow()
  })
})
