import { afterEach, describe, expect, it, vi } from 'vitest'
import { decodeSignedStatementB64, fetchPeerLedgerCapsule } from '@/features/capsules/api/peerLedgerFetchClient'

function jsonResponse(body: unknown, ok = true, status = 200) {
  return {
    ok,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  } as Response
}

describe('fetchPeerLedgerCapsule', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("posts {peer_id, capsule_id} to this plugin's mesh_ledger_fetch tool route", async () => {
    const fetchSpy = vi.fn().mockResolvedValue(jsonResponse({ status: 'not_found', capsule_id: 'cap-1' }))
    vi.stubGlobal('fetch', fetchSpy)

    await fetchPeerLedgerCapsule('peer-a', 'cap-1')

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const [url, init] = fetchSpy.mock.calls[0]
    expect(String(url)).toBe('/api/plugins/capsules/tools/mesh_ledger_fetch')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ peer_id: 'peer-a', capsule_id: 'cap-1' })
  })

  it('returns kind: found with the real capsule/statement/pubkey on a hit', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: 'found',
          capsule: { capsule_id: 'cap-1' },
          signed_statement_b64: 'AAA=',
          node_pub_key_pem: '-----BEGIN PUBLIC KEY-----\nAAA=\n-----END PUBLIC KEY-----'
        })
      )
    )

    const outcome = await fetchPeerLedgerCapsule('peer-a', 'cap-1')
    expect(outcome).toEqual({
      kind: 'found',
      capsule: { capsule_id: 'cap-1' },
      signedStatementB64: 'AAA=',
      nodePubKeyPem: '-----BEGIN PUBLIC KEY-----\nAAA=\n-----END PUBLIC KEY-----'
    })
  })

  // (negative, R4 other half) an honest "no such entry" must render as
  // not_found, never silently coerced into a found/pass shape.
  it('returns kind: not_found honestly, never fabricating a found result', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'not_found', capsule_id: 'cap-1' })))

    const outcome = await fetchPeerLedgerCapsule('peer-a', 'cap-1')
    expect(outcome).toEqual({ kind: 'not_found' })
  })

  it('reports an archived record as set aside, never as not_found', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'archived', capsule_id: 'cap-1' })))
    const outcome = await fetchPeerLedgerCapsule('peer-a', 'cap-1')
    expect(outcome).toEqual({ kind: 'error', message: 'the peer set this record aside when it started a new history' })
  })

  it('reports a declined fetch as the peer declining, never as found or not_found', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ status: 'not_authorized', capsule_id: 'cap-1' })))

    const outcome = await fetchPeerLedgerCapsule('peer-a', 'cap-1')
    expect(outcome).toEqual({ kind: 'error', message: 'the peer declined to share this record' })
  })

  it('returns kind: error for a responder-side error, carrying the real message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ status: 'error', message: 'ledger lookup failed: io error' }))
    )

    const outcome = await fetchPeerLedgerCapsule('peer-a', 'cap-1')
    expect(outcome).toEqual({ kind: 'error', message: 'ledger lookup failed: io error' })
  })

  it('returns kind: transport_error on a non-2xx host response (e.g. an unroutable peer 502), never a found', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse('could not reach peer: timed out', false, 502)))

    const outcome = await fetchPeerLedgerCapsule('peer-a', 'cap-1')
    expect(outcome.kind).toBe('transport_error')
  })

  it('returns kind: transport_error when the network call itself throws', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('network unreachable')))

    const outcome = await fetchPeerLedgerCapsule('peer-a', 'cap-1')
    expect(outcome).toEqual({ kind: 'transport_error', message: 'network unreachable' })
  })

  // MUTANT check (R4): if a `found`-shaped body missing a required field
  // (here: no `node_pub_key_pem`) were accepted anyway, a recompute
  // downstream would silently verify against `undefined`. Confirmed this
  // guard bites: removing the `typeof tagged.node_pub_key_pem === 'string'`
  // condition in peerLedgerFetchClient.ts makes this assertion fail (the
  // outcome becomes `kind: 'found'` with `nodePubKeyPem: undefined`) --
  // reverted after confirming.
  it('does not accept a found-shaped body missing node_pub_key_pem as a real found result', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ status: 'found', capsule: { capsule_id: 'cap-1' }, signed_statement_b64: 'AAA=' })
        )
    )

    const outcome = await fetchPeerLedgerCapsule('peer-a', 'cap-1')
    expect(outcome.kind).not.toBe('found')
  })
})

describe('decodeSignedStatementB64', () => {
  it('round-trips standard base64 to the original bytes', () => {
    // "hello" in base64
    const bytes = decodeSignedStatementB64('aGVsbG8=')
    expect(new TextDecoder().decode(bytes)).toBe('hello')
  })
})
