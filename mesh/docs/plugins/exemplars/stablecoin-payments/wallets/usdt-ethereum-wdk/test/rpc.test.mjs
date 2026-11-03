import assert from 'node:assert/strict'
import test from 'node:test'
import { createServer } from 'node:http'
import { EthereumRpc, CACHE_MAX_ENTRIES, RPC_TIMEOUT_MS } from '../rpc.mjs'
import { startEthereumFixture } from './ethereum-fixture.mjs'

test('RPC pins mainnet and finalized canonical USDT without requiring an archive/genesis query', async t => {
  const fixture = await startEthereumFixture(t)
  const rpc = new EthereumRpc({ rpc_url: fixture.url }); t.after(() => rpc.dispose())
  assert.equal(await rpc.checkNetwork(), true)
  assert.deepEqual(new Set(fixture.requests.map(call => call.method)),
    new Set(['eth_chainId', 'eth_getBlockByNumber', 'eth_getCode', 'eth_call']))
  assert.ok(fixture.requests.filter(call => call.method === 'eth_getBlockByNumber').every(call => call.params[0] === 'finalized'))
  assert.equal((await rpc.finalized()).number, '0x64')
  assert.deepEqual(fixture.broadcasts, [])
})

test('wrong network, absent token, wrong precision or unavailable finalized state fail closed', async t => {
  const fixture = await startEthereumFixture(t)
  async function check() {
    const rpc = new EthereumRpc({ rpc_url: fixture.url })
    try { return await rpc.checkNetwork() } finally { rpc.dispose() }
  }
  fixture.chainId = 11155111
  await assert.rejects(check(), /mainnet/)
  fixture.chainId = 1
  fixture.overrides.set('eth_getCode', '0x')
  await assert.rejects(check(), /absent/)
  fixture.overrides.clear(); fixture.tokenDecimals = 18
  await assert.rejects(check(), /six decimals/)
  fixture.tokenDecimals = 6
  fixture.overrides.set('eth_getBlockByNumber', null)
  await assert.rejects(check(), /finalized blocks/)
  assert.deepEqual(fixture.broadcasts, [])
})

test('bounded RPC cache prunes old observations and prevents callers mutating shared evidence', async t => {
  let now = 1000
  t.mock.method(Date, 'now', () => now)
  const rpc = new EthereumRpc({ rpc_url: 'http://127.0.0.1:1' }); t.after(() => rpc.dispose())
  for (let index = 0; index < CACHE_MAX_ENTRIES + 2; index++) {
    await rpc.cached(`receipt-${index}`, () => ({ index }), 10)
    assert.ok(rpc.cache.size <= 128)
  }
  assert.equal(rpc.cache.size, 128)
  assert.equal(rpc.cache.has('receipt-0'), false)
  let calls = 0
  const action = () => { calls++; return { amount: '27' } }
  const first = await rpc.cached('current', action, 10); first.amount = '1000'
  assert.deepEqual(await rpc.cached('current', action, 10), { amount: '27' })
  assert.equal(calls, 1)
  now += 11
  await rpc.cached('fresh', () => true)
  assert.deepEqual([...rpc.cache.keys()], ['fresh'])
})

test('pending RPC observations coalesce past TTL and failed lookups remain recoverable', async t => {
  let now = 1000, calls = 0, release
  t.mock.method(Date, 'now', () => now)
  const rpc = new EthereumRpc({ rpc_url: 'http://127.0.0.1:1' }); t.after(() => rpc.dispose())
  const deferred = new Promise(resolve => { release = resolve })
  const action = () => { calls++; return deferred }
  const first = rpc.cached('pending', action, 10)
  await Promise.resolve(); now += 20
  const second = rpc.cached('pending', action, 10)
  assert.equal(calls, 1)
  release({ finalized: false })
  assert.deepEqual(await Promise.all([first, second]), [{ finalized: false }, { finalized: false }])
  assert.equal(calls, 1)
  await assert.rejects(rpc.cached('failed', () => { throw new Error('offline fixture failure') }), /fixture failure/)
  assert.equal(rpc.cache.has('failed'), false)
  assert.equal(await rpc.cached('failed', () => 'recovered'), 'recovered')
})

test('WDK ethers bridge shares no-redirect transport and enforces streamed response size bounds', async t => {
  let targetCalls = 0
  async function listen(handler) {
    const server = createServer(handler)
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    t.after(() => new Promise(resolve => server.close(resolve)))
    return `http://127.0.0.1:${server.address().port}`
  }
  const target = await listen((_request, reply) => { targetCalls++; reply.end('{}') })
  const redirect = await listen((_request, reply) => { reply.writeHead(302, { location: target }); reply.end() })
  const redirected = new EthereumRpc({ rpc_url: redirect }); t.after(() => redirected.dispose())
  assert.equal(RPC_TIMEOUT_MS, 8000)
  await assert.rejects(redirected.provider.send('eth_chainId', []))
  assert.equal(targetCalls, 0)
  const oversized = await listen((_request, reply) => {
    reply.writeHead(200, { 'content-type': 'application/json', 'transfer-encoding': 'chunked' })
    reply.end(Buffer.alloc(8 * 1024 * 1024 + 1, ' '))
  })
  const bounded = new EthereumRpc({ rpc_url: oversized }); t.after(() => bounded.dispose())
  await assert.rejects(bounded.call('eth_chainId'), /too large/)
})
