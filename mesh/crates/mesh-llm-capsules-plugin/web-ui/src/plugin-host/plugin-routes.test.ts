// The page reaches data ONLY through the host's plugin-scoped fetch: every
// request resolves under `/api/plugins/capsules/`, never a
// host-private route such as the fork's `/api/capsules/*`.
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  PLUGIN_ROUTES,
  fetchCapsuleLedger,
  fetchDisclosurePreimage,
  fetchSignedStatement
} from '@/features/capsules/api/client'
import {
  PaneFetchError,
  fetchPaneA,
  fetchPaneB,
  fetchPaneCDrilldown,
  fetchPaneCList
} from '@/features/capsules/api/sidecarClient'
import { pluginScopedApiUrl } from '@/plugin-host/standalone-host'

function respond(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

function stubFetch(body: unknown, status = 200) {
  const spy = vi.fn(async () => respond(body, status))
  vi.stubGlobal('fetch', spy)
  return spy
}

function requestedUrls(spy: ReturnType<typeof vi.fn>): string[] {
  return spy.mock.calls.map(([url]) => String(url))
}

afterEach(() => vi.unstubAllGlobals())

describe('ledger client', () => {
  it('reads the wrapped ledger from the plugin route and skips a malformed entry', async () => {
    const spy = stubFetch({ records: [{ capsule_id: 'a' }, 'not-a-record', null, { capsule_id: 'b' }], node_pub_key_pem: 'PEM' })
    const ledger = await fetchCapsuleLedger()
    expect(requestedUrls(spy)).toEqual(['/api/plugins/capsules/http/ledger'])
    expect(ledger).toEqual({ records: [{ capsule_id: 'a' }, { capsule_id: 'b' }], nodePubKeyPem: 'PEM' })
  })

  it('reads a 404 (nothing sealed yet) as an empty ledger, and throws on any other failure', async () => {
    stubFetch({}, 404)
    await expect(fetchCapsuleLedger()).resolves.toEqual({ records: [], nodePubKeyPem: null })
    stubFetch({}, 500)
    await expect(fetchCapsuleLedger()).rejects.toThrow(/HTTP 500/)
  })

  it('decodes the signed statement from base64, and reads an absent one as null', async () => {
    const spy = stubFetch({ signed_statement_b64: 'AQID' })
    await expect(fetchSignedStatement('cap/1')).resolves.toEqual(new Uint8Array([1, 2, 3]))
    expect(requestedUrls(spy)).toEqual([
      '/api/plugins/capsules/http/ledger/signed-statement?capsule_id=cap%2F1'
    ])
    stubFetch({ signed_statement_b64: null })
    await expect(fetchSignedStatement('cap-1')).resolves.toBeNull()
    stubFetch({}, 404)
    await expect(fetchSignedStatement('cap-1')).resolves.toBeNull()
  })

  it('reads the disclosure preimage, and an absent one as null', async () => {
    stubFetch({ disclosure: { request_text: 'q' } })
    await expect(fetchDisclosurePreimage('cap-1')).resolves.toEqual({ request_text: 'q' })
    stubFetch({ disclosure: null })
    await expect(fetchDisclosurePreimage('cap-1')).resolves.toBeNull()
    expect(PLUGIN_ROUTES.disclosure('x y')).toBe('http/ledger/disclosure?capsule_id=x%20y')
  })
})

describe('pane client', () => {
  it('reads every pane from the plugin panes routes', async () => {
    const spy = stubFetch({ rows: [] })
    await fetchPaneA()
    await fetchPaneB()
    await fetchPaneCList({ limit: 50, afterSeq: 7 })
    await fetchPaneCDrilldown('digest:ab')
    expect(requestedUrls(spy)).toEqual([
      '/api/plugins/capsules/http/panes/pane-a',
      '/api/plugins/capsules/http/panes/pane-b',
      '/api/plugins/capsules/http/panes/pane-c?limit=50&after_seq=7',
      '/api/plugins/capsules/http/panes/pane-c?exchange_id=digest%3Aab'
    ])
  })

  it('keeps the HTTP status on failure so the page can say which failure it is', async () => {
    stubFetch({}, 404)
    const error = await fetchPaneA().catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(PaneFetchError)
    expect((error as PaneFetchError).status).toBe(404)
  })
})

describe('plugin-scoped paths (the console host rules)', () => {
  it('encodes segments and keeps the query', () => {
    expect(pluginScopedApiUrl('capsules', 'panes/pane-c?limit=1')).toBe(
      '/api/plugins/capsules/panes/pane-c?limit=1'
    )
  })

  it.each(['../status', 'panes/./a', 'http://evil/x', 'panes#frag', 'a\\b', 'a\u0000b'])(
    'rejects %j',
    (path) => {
      expect(() => pluginScopedApiUrl('capsules', path)).toThrow(TypeError)
    }
  )
})
