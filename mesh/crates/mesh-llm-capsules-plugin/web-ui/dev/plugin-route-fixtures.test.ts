import { describe, expect, it } from 'vitest'
import { answerFromFixtures, type Captured } from './plugin-route-fixtures'

const P = '/api/plugins/capsules'

function reader(entries: Record<string, { status?: number; body: string | Buffer; contentType?: string }>) {
  return (key: string): Captured | null => {
    const entry = entries[key]
    if (!entry) return null
    return {
      status: entry.status ?? 200,
      contentType: entry.contentType ?? 'application/json',
      body: Buffer.isBuffer(entry.body) ? entry.body : Buffer.from(entry.body)
    }
  }
}

function bodyOf(answer: ReturnType<typeof answerFromFixtures>) {
  return JSON.parse(String(answer?.body))
}

describe('fixture translation (plugin route -> captured host route)', () => {
  it('serves a pane route from the fork capture of the same pane, query included', () => {
    const read = reader({ 'GET /api/capsules/panes/pane-c?exchange_id=digest%3Aab': { body: '{"rows":[1]}' } })
    const answer = answerFromFixtures('GET', `${P}/http/panes/pane-c?exchange_id=digest%3Aab`, read)
    expect(answer?.status).toBe(200)
    expect(bodyOf(answer)).toEqual({ rows: [1] })
  })

  it('wraps the jsonl ledger and the PEM key into the ledger JSON', () => {
    const read = reader({
      'GET /api/capsules/ledger/capsules.jsonl': { body: '{"capsule_id":"a"}\nnot json\n\n{"capsule_id":"b"}\n' },
      'GET /api/capsules/ledger/node-key.pub.pem': { body: 'PEM' }
    })
    expect(bodyOf(answerFromFixtures('GET', `${P}/http/ledger`, read))).toEqual({
      records: [{ capsule_id: 'a' }, { capsule_id: 'b' }],
      node_pub_key_pem: 'PEM'
    })
  })

  it('base64-wraps a captured signed statement, and nulls a captured 404', () => {
    const read = reader({
      'GET /api/capsules/ledger/signed-statements/c1.cose': { body: Buffer.from([1, 2, 3]) },
      'GET /api/capsules/ledger/signed-statements/c2.cose': { status: 404, body: '' }
    })
    expect(bodyOf(answerFromFixtures('GET', `${P}/http/ledger/signed-statement?capsule_id=c1`, read))).toEqual({
      signed_statement_b64: 'AQID'
    })
    expect(bodyOf(answerFromFixtures('GET', `${P}/http/ledger/signed-statement?capsule_id=c2`, read))).toEqual({
      signed_statement_b64: null
    })
  })

  it('never synthesises: an uncaptured route is a 404 naming the miss', () => {
    const answer = answerFromFixtures('POST', `${P}/tools/mesh_ledger_fetch`, reader({}))
    expect(answer?.status).toBe(404)
    expect(bodyOf(answer).error).toBe('evidence fixture mode: not captured')
    expect(answerFromFixtures('GET', `${P}/http/panes/pane-a`, reader({}))?.status).toBe(404)
  })

  it('passes the console status/models captures through, and ignores other paths', () => {
    const read = reader({ 'GET /api/status': { body: '{"node_id":"n"}' } })
    expect(bodyOf(answerFromFixtures('GET', '/api/status', read))).toEqual({ node_id: 'n' })
    expect(answerFromFixtures('GET', '/api/other', read)).toBeNull()
  })
})
