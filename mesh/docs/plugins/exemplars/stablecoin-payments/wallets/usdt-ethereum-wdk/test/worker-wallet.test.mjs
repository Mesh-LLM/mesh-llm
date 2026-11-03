import assert from 'node:assert/strict'
import test from 'node:test'
import * as fs from 'node:fs/promises'
import { readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Transaction, keccak256 } from 'ethers'
import { initializeProfile } from '../keys.mjs'
import { EthereumWallet } from '../worker-wallet.mjs'
import { startEthereumFixture, paymentRequest, transferReceipt, transactionObject,
  RECIPIENT, USDT, FIXTURE_SEED } from './ethereum-fixture.mjs'

async function walletFixture(t) {
  const rpc = await startEthereumFixture(t)
  const directory = await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()), 'mesh-ethereum-worker-test-'))
  await fs.chmod(directory, 0o700)
  t.after(() => fs.rm(directory, { recursive: true, force: true }))
  const settings = { data_dir: path.join(directory, 'wallet'), password_file: path.join(directory, 'credential'), rpc_url: rpc.url }
  await fs.writeFile(settings.password_file, 'offline-fixture-password-not-for-funding', { mode: 0o600 })
  const profile = await initializeProfile(settings, { seed: FIXTURE_SEED, wallet_id: 'offline-worker' })
  const wallet = await EthereumWallet.open(settings); t.after(() => wallet.dispose())
  const request = paymentRequest({ payer: profile.address })
  const payment = { payment_id: request.obligation.id, request, max_fee: '0', max_total: '27' }
  return { rpc, directory, settings, profile, wallet, request, payment }
}

test('prepare uses actual WDK signing, empty USDT return data and separate ETH fee accounting at 18 decimals', async t => {
  const { rpc, wallet, profile, request, payment } = await walletFixture(t)
  const info = await wallet.info()
  assert.equal(info.address, profile.address.toLowerCase())
  assert.equal(info.can_send, true)
  assert.deepEqual(info.asset, { chain_id: 1, token: USDT.toLowerCase(), decimals: 6 })
  assert.deepEqual(info.fee_asset, { chain_id: 1, token: 'native:ETH', decimals: 18 })
  const estimate = await wallet.estimate(request)
  assert.deepEqual(estimate.fee_asset, info.fee_asset)
  assert.equal(estimate.gas_limit, 84000)
  assert.equal(estimate.fee, '252000000000000')
  const prepared = await wallet.prepare(payment)
  const signed = Transaction.from(prepared.signed_transaction)
  assert.equal(signed.from.toLowerCase(), profile.address.toLowerCase())
  assert.equal(prepared.tx_hash, signed.hash)
  assert.equal(prepared.nonce, '0')
  assert.equal(signed.data.length, 202)
  assert.equal(signed.data.slice(-64), request.memo.slice(2))
  assert.equal(signed.value, 0n)
  assert.equal(prepared.fee, estimate.fee)
  assert.deepEqual(rpc.broadcasts, [])
  const executions = rpc.requests.filter(call => call.method === 'eth_estimateGas'
    || call.method === 'eth_call' && call.params[0].data.startsWith('0xa9059cbb'))
  assert.ok(executions.length >= 4)
  assert.ok(executions.every(call => call.params[0].data.length === 202 && call.params[0].data.endsWith(request.memo.slice(2))))
})

test('a different payment ID cannot alias an immutable obligation before any signing work', async t => {
  const { rpc, wallet, payment } = await walletFixture(t)
  await assert.rejects(wallet.prepare({ ...payment, payment_id: 'different-payment-id' }), /immutable obligation/)
  assert.deepEqual(rpc.requests, [])
  assert.deepEqual(rpc.broadcasts, [])
})

test('reopening and rebroadcast preserve exact signed bytes, nonce and transaction hash', async t => {
  const { rpc, directory, settings, wallet, request, payment } = await walletFixture(t)
  const prepared = await wallet.prepare(payment)
  const persisted = { request, signed_transaction: prepared.signed_transaction, tx_hash: prepared.tx_hash }
  const journal = path.join(directory, 'signed-fixture-journal.json')
  const handle = await fs.open(journal, 'wx', 0o600)
  await handle.writeFile(JSON.stringify(persisted)); await handle.sync(); await handle.close()
  wallet.dispose()
  const reopened = await EthereumWallet.open(settings); t.after(() => reopened.dispose())
  rpc.beforeBroadcast = raw => assert.equal(raw, JSON.parse(readFileSync(journal, 'utf8')).signed_transaction)
  const recovered = JSON.parse(await fs.readFile(journal, 'utf8'))
  assert.deepEqual(await reopened.broadcast(recovered), { accepted: true })
  assert.deepEqual(await reopened.broadcast(recovered), { accepted: true })
  assert.deepEqual(rpc.broadcasts, [prepared.signed_transaction, prepared.signed_transaction])
  assert.equal(keccak256(rpc.broadcasts[0]), prepared.tx_hash)
  assert.equal(Transaction.from(rpc.broadcasts[1]).nonce, 0)
  rpc.transactions.set(prepared.tx_hash, transactionObject(prepared.signed_transaction))
  rpc.receipts.set(prepared.tx_hash, transferReceipt(prepared.signed_transaction, request))
  reopened.rpc.cache.clear()
  const result = await reopened.verify({ request, tx_hash: prepared.tx_hash })
  assert.equal(result.state, 'finalized')
  assert.equal(result.receipt.amount, '27')
  assert.equal(result.receipt.memo, request.memo)
})

test('read-only wallets verify exact finalized delivery and cannot prepare or broadcast', async t => {
  const { rpc, settings, wallet, request, payment } = await walletFixture(t)
  const prepared = await wallet.prepare(payment)
  rpc.transactions.set(prepared.tx_hash, transactionObject(prepared.signed_transaction))
  rpc.receipts.set(prepared.tx_hash, transferReceipt(prepared.signed_transaction, request))
  const readOnly = await EthereumWallet.open({ data_dir: settings.data_dir, rpc_url: rpc.url }); t.after(() => readOnly.dispose())
  assert.equal((await readOnly.info()).can_send, false)
  assert.equal((await readOnly.verify({ request, tx_hash: prepared.tx_hash })).state, 'finalized')
  await assert.rejects(readOnly.prepare(payment), /receive-only/)
  await assert.rejects(readOnly.broadcast({ request, ...prepared }), /receive-only/)
  assert.deepEqual(rpc.broadcasts, [])
})

test('missing balances, delegated sender or failed USDT simulation prevent signing and broadcast', async t => {
  const { rpc, wallet, payment } = await walletFixture(t)
  rpc.ethBalance = 0n
  await assert.rejects(wallet.prepare(payment), /Insufficient ETH/)
  rpc.ethBalance = 1000000000000000000n; rpc.tokenBalance = 0n
  wallet.rpc.cache.delete('balance')
  await assert.rejects(wallet.prepare(payment), /Insufficient finalized USDT/)
  rpc.tokenBalance = 1000000000n; rpc.simulationResult = '0x' + '0'.repeat(64)
  await assert.rejects(wallet.prepare(payment), /simulation failed/)
  rpc.simulationResult = '0x'
  rpc.overrides.set('eth_getCode', () => '0x6001600055')
  await assert.rejects(wallet.prepare(payment), /undelegated/)
  assert.deepEqual(rpc.broadcasts, [])
})

test('missing receipts stay pending after local deadline without clearing or replacing signed bytes', async t => {
  const { rpc, wallet, request, payment } = await walletFixture(t)
  const prepared = await wallet.prepare(payment)
  const future = (request.obligation.expires_at + 3600) * 1000
  t.mock.method(Date, 'now', () => future)
  const result = await wallet.lookup({ request, tx_hash: prepared.tx_hash })
  assert.equal(result.state, 'submitted')
  assert.equal(result.receipt, null)
  await assert.rejects(wallet.broadcast({ request, ...prepared }), /deadline/)
  assert.deepEqual(rpc.broadcasts, [])
})

test('persisted public identity must match the decrypted WDK seed on reopening', async t => {
  const { settings, profile } = await walletFixture(t)
  await fs.writeFile(path.join(settings.data_dir, 'wallet.json'), JSON.stringify({ ...profile, address: RECIPIENT }))
  await assert.rejects(EthereumWallet.open(settings), /seed does not match/)
})
