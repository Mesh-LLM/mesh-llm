import assert from 'node:assert/strict'
import test from 'node:test'
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import * as fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { initializeProfile } from '../keys.mjs'
import { Transaction, keccak256 } from 'ethers'
import { paymentRequest, startEthereumFixture } from './ethereum-fixture.mjs'

function startWorker(t) {
  const child = spawn(process.execPath, [fileURLToPath(new URL('../worker.mjs', import.meta.url))], {
    stdio: ['pipe', 'pipe', 'pipe']
  })
  const lines = createInterface({ input: child.stdout })
  const pending = new Map(), output = [], errors = []
  let nextId = 0, stopped = false
  child.stdin.on('error', () => {})
  child.stderr.on('data', data => errors.push(data.toString()))
  lines.on('line', line => {
    output.push(line)
    let message
    try { message = JSON.parse(line) } catch { return }
    const waiter = pending.get(message.id)
    if (waiter) { pending.delete(message.id); waiter.resolve(message) }
  })
  const closed = new Promise((resolve, reject) => {
    child.once('error', reject)
    child.once('close', code => {
      stopped = true
      for (const waiter of pending.values()) waiter.reject(new Error('Fixture worker closed before replying'))
      pending.clear()
      resolve(code)
    })
  })
  async function stop() {
    if (!stopped) child.stdin.end()
    const timer = setTimeout(() => child.kill('SIGKILL'), 3000)
    try { return await closed } finally { clearTimeout(timer) }
  }
  t.after(stop)
  return {
    output, errors, stop,
    async call(operation, input) {
      const id = ++nextId
      const result = new Promise((resolve, reject) => pending.set(id, { resolve, reject }))
      child.stdin.write(JSON.stringify({ id, operation, input }) + '\n')
      let timer
      try {
        return await Promise.race([result, new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error(`Fixture worker ${operation} did not reply`)), 5000)
        })])
      } finally { clearTimeout(timer); pending.delete(id) }
    }
  }
}

test('actual private worker protocol preserves prepared transaction identity across process restart', async t => {
  const rpc = await startEthereumFixture(t)
  const directory = await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()), 'mesh-ethereum-ipc-test-'))
  t.after(() => fs.rm(directory, { recursive: true, force: true }))
  const settings = { data_dir: path.join(directory, 'wallet'), password_file: path.join(directory, 'credential'), rpc_url: rpc.url }
  const fakePassword = 'offline-ipc-fixture-password-not-for-funding'
  await fs.writeFile(settings.password_file, fakePassword, { mode: 0o600 })
  const profile = await initializeProfile(settings, { seed: Buffer.alloc(64, 1), wallet_id: 'offline-ipc' })
  const request = paymentRequest({ payer: profile.address })
  const first = startWorker(t)
  const opened = await first.call('open', settings)
  assert.equal(opened.error, undefined)
  assert.equal(opened.result.address, profile.address.toLowerCase())
  assert.equal(opened.result.can_send, true)
  const alias = await first.call('prepare', { payment_id: 'different-ipc-payment-id', request, max_fee: '0', max_total: '27' })
  assert.match(alias.error, /immutable obligation/)
  assert.equal(alias.result, undefined)
  assert.equal(rpc.broadcasts.length, 0)
  const prepared = await first.call('prepare', { payment_id: request.obligation.id, request, max_fee: '0', max_total: '27' })
  assert.equal(prepared.error, undefined)
  assert.equal(prepared.result.tx_hash, Transaction.from(prepared.result.signed_transaction).hash)
  assert.equal(rpc.broadcasts.length, 0)
  const persisted = { request, tx_hash: prepared.result.tx_hash, signed_transaction: prepared.result.signed_transaction }
  const journal = path.join(directory, 'signed-ipc-fixture.json')
  const handle = await fs.open(journal, 'wx', 0o600)
  await handle.writeFile(JSON.stringify(persisted)); await handle.sync(); await handle.close()
  assert.equal(await first.stop(), 0)

  const second = startWorker(t)
  assert.equal((await second.call('open', settings)).result.wallet_id, profile.wallet_id)
  const recovered = JSON.parse(await fs.readFile(journal, 'utf8'))
  assert.deepEqual((await second.call('broadcast', recovered)).result, { accepted: true })
  assert.deepEqual((await second.call('broadcast', recovered)).result, { accepted: true })
  assert.equal(rpc.broadcasts.length, 2)
  assert.deepEqual(rpc.broadcasts[0], rpc.broadcasts[1])
  assert.deepEqual(rpc.broadcasts[0], persisted.signed_transaction)
  assert.equal(keccak256(rpc.broadcasts[0]), prepared.result.tx_hash)
  const invalid = await second.call('broadcast', { ...recovered, tx_hash: '0x' + 'cc'.repeat(32) })
  assert.ok(invalid.error)
  assert.equal(rpc.broadcasts.length, 2)
  assert.equal(await second.stop(), 0)
  for (const worker of [first, second]) {
    assert.ok(worker.output.every(line => { try { JSON.parse(line); return true } catch { return false } }))
    const publicOutput = worker.output.join('\n') + worker.errors.join('\n')
    assert.ok(!publicOutput.includes(fakePassword))
    assert.ok(!publicOutput.includes(Buffer.alloc(64, 1).toString('hex')))
  }
})
