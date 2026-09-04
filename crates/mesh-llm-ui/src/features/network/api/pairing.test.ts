import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  connectWithPairingOffer,
  createPairingOffer,
  decidePairingSession,
  discoverNearbyPairingDevices,
  fetchPairingSessions
} from '@/features/network/api/pairing'

const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

describe('pairing API', () => {
  afterEach(() => vi.restoreAllMocks())

  it('uses the loopback management API for the complete pairing ceremony', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(response({ sessions: [] }))
      .mockResolvedValueOnce(response({ offer: 'opaque', url: 'mesh-llm://pair/opaque', expires_at: 42 }, 201))
      .mockResolvedValueOnce(response({ id: 'session-1', status: 'connecting' }, 202))
      .mockResolvedValueOnce(response({ id: 'session-1', status: 'waiting_for_peer' }))
      .mockResolvedValueOnce(response([{ instance_name: 'studio' }]))

    await fetchPairingSessions()
    await createPairingOffer()
    await connectWithPairingOffer('opaque')
    await decidePairingSession('session-1', 'approve')
    await discoverNearbyPairingDevices()

    expect(fetchMock).toHaveBeenNthCalledWith(1, expect.stringMatching(/\/api\/pairing\/sessions$/), undefined)
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      expect.stringMatching(/\/api\/pairing\/connect$/),
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ offer: 'opaque' }) })
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      expect.stringMatching(/\/api\/pairing\/sessions\/session-1\/approve$/),
      { method: 'POST' }
    )
    expect(fetchMock).toHaveBeenNthCalledWith(5, expect.stringMatching(/\/api\/discover$/), undefined)
  })

  it('does not place an invite in a discovery query string', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response({ id: 'session-1' }, 202))

    await connectWithPairingOffer('secret-looking-opaque-offer')

    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).not.toContain('secret-looking-opaque-offer')
    expect(init?.body).toBe(JSON.stringify({ offer: 'secret-looking-opaque-offer' }))
  })
})
