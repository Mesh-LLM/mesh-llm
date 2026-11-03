import { JsonRpcProvider, toQuantity } from 'ethers'
import { ensure, SafeError } from './errors.mjs'
import { CHAIN_ID, USDT, quantity, hashHex, height } from './validation.mjs'

export const RPC_TIMEOUT_MS = 8000
export const CACHE_MAX_ENTRIES = 128
const MAX_RESPONSE_BYTES = 8 * 1024 * 1024

// A real ethers provider lets the pinned WDK use its native estimateGas/send
// APIs while sharing this worker's bounded, no-redirect JSON-RPC transport.
class LocalProvider extends JsonRpcProvider {
  constructor(rpc) { super(undefined, CHAIN_ID, { staticNetwork: true, batchMaxCount: 1, cacheTimeout: -1 }); this.rpc = rpc }
  async _send(payload) {
    const requests = Array.isArray(payload) ? payload : [payload]
    return await Promise.all(requests.map(async value => ({ id: value.id, jsonrpc: '2.0',
      result: await this.rpc.call(value.method, value.params) })))
  }
}
export class EthereumRpc {
  constructor(settings) {
    this.url = settings.rpc_url; this.cache = new Map(); this.nextId = 1
    this.provider = new LocalProvider(this)
  }
  static async create(settings) { return new EthereumRpc(settings) }
  dispose() { this.provider.destroy(); this.cache.clear() }
  async call(method, params = []) {
    try {
      const id = this.nextId++
      const response = await fetch(this.url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jsonrpc: '2.0', id, method, params }), signal: AbortSignal.timeout(RPC_TIMEOUT_MS), redirect: 'error' })
      ensure(response.ok, `Ethereum RPC returned HTTP ${response.status}; check endpoint access and rate limits.`)
      ensure(Number(response.headers.get('content-length') || 0) <= MAX_RESPONSE_BYTES, 'Ethereum RPC response is too large.')
      const reader = response.body.getReader(), chunks = []; let total = 0
      try {
        while (true) {
          const { done, value } = await reader.read(); if (done) break
          total += value.byteLength
          ensure(total <= MAX_RESPONSE_BYTES, 'Ethereum RPC response is too large.')
          chunks.push(value)
        }
      } finally { await reader.cancel().catch(() => {}) }
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8'))
      ensure(body && body.jsonrpc === '2.0' && body.id === id, 'Ethereum RPC returned an inconsistent response.')
      ensure(!body.error && Object.hasOwn(body, 'result'), 'Ethereum RPC rejected the operation; reconcile the original transaction before retrying a payment.')
      return body.result
    } catch (error) { if (error instanceof SafeError) throw error; throw new SafeError('Ethereum RPC is unavailable or returned invalid data.') }
  }
  async cached(key, action, milliseconds = 3000) {
    const now = Date.now()
    for (const [name, entry] of this.cache) if (!entry.pending && entry.expires <= now) this.cache.delete(name)
    const previous = this.cache.get(key)
    if (previous && (previous.pending || previous.expires > now)) return structuredClone(await previous.value)
    while (this.cache.size >= CACHE_MAX_ENTRIES) this.cache.delete(this.cache.keys().next().value)
    const entry = { value: Promise.resolve().then(action), pending: true, expires: now + milliseconds }
    this.cache.set(key, entry)
    try {
      const value = await entry.value
      entry.pending = false; entry.expires = Date.now() + milliseconds
      return structuredClone(value)
    } catch (error) { if (this.cache.get(key) === entry) this.cache.delete(key); throw error }
  }
  async checkNetwork() {
    return await this.cached('network', async () => {
      ensure(quantity(await this.call('eth_chainId')) === BigInt(CHAIN_ID), 'RPC is not Ethereum mainnet.')
      const finalized = await this.finalized(), tag = toQuantity(height(finalized.number))
      const [code, decimals] = await Promise.all([ this.call('eth_getCode', [USDT, tag]),
        this.call('eth_call', [{ to: USDT, data: '0x313ce567' }, tag]) ])
      ensure(typeof code === 'string' && /^0x(?:[0-9a-fA-F]{2})+$/.test(code), 'Canonical USDT contract is absent on this RPC.')
      ensure(typeof decimals === 'string' && /^0x[0-9a-fA-F]{64}$/.test(decimals) && BigInt(decimals) === 6n, 'USDT precision is not six decimals.')
      // Genesis queries are intentionally omitted: ordinary pruned mainnet
      // providers need not retain block zero. This remains a trusted RPC check.
      return true
    }, 15000)
  }
  async finalized() {
    return await this.cached('finalized', async () => {
      const block = await this.call('eth_getBlockByNumber', ['finalized', false])
      ensure(block && height(block.number) >= 0, 'RPC does not provide Ethereum finalized blocks.')
      hashHex(block.hash)
      return block
    })
  }
}
