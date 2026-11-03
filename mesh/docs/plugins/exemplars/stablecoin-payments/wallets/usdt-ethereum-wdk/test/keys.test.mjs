import assert from 'node:assert/strict'
import test from 'node:test'
import * as fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { HDNodeWallet } from 'ethers'
import { initializeProfile, openProfile, readProfile, validateSettings } from '../keys.mjs'

// These bytes are a published fixture, not a wallet backup. Never fund them.
const FIXTURE_SEED = Buffer.alloc(64, 1)
const PASSWORD = 'offline-fixture-password-not-for-funding'

async function temporarySettings(t) {
  const directory = await fs.mkdtemp(path.join(await fs.realpath(os.tmpdir()), 'mesh-ethereum-keys-test-'))
  await fs.chmod(directory, 0o700)
  t.after(() => fs.rm(directory, { recursive: true, force: true }))
  const settings = { data_dir: path.join(directory, 'wallet'), password_file: path.join(directory, 'password') }
  await fs.writeFile(settings.password_file, PASSWORD + '\r\n', { mode: 0o600 })
  return { directory, settings }
}

test('settings require explicit private local paths and an HTTPS or loopback RPC', () => {
  assert.equal(validateSettings({ data_dir: '/fixture' }).rpc_url, 'https://ethereum-rpc.publicnode.com')
  assert.equal(validateSettings({ data_dir: '/fixture', rpc_url: 'http://127.0.0.1:10000' }).rpc_url, 'http://127.0.0.1:10000')
  for (const settings of [
    { data_dir: 'relative' },
    { data_dir: '/fixture', password_file: 'relative' },
    { data_dir: '/fixture', rpc_url: 'http://remote.example' },
    { data_dir: '/fixture', rpc_url: 'https://user:secret@example.invalid' },
    { data_dir: '/fixture', rpc_url: 'https://example.invalid?key=secret' },
    { data_dir: '/fixture', seed: 'caller supplied secret' }
  ]) assert.throws(() => validateSettings(settings))
})

test('explicit initialization encrypts a deterministic WDK seed and unlocks it with the private password', async t => {
  const { settings } = await temporarySettings(t)
  const expectedAccount = HDNodeWallet.fromSeed(FIXTURE_SEED).derivePath("m/44'/60'/0'/0/0")
  const profile = await initializeProfile(settings, { seed: FIXTURE_SEED, wallet_id: 'offline-fixture' })
  assert.equal(profile.address.toLowerCase(), expectedAccount.address.toLowerCase())
  assert.equal(profile.wallet_id, 'offline-fixture')
  assert.equal(profile.chain_id, 1)
  const envelope = await fs.readFile(path.join(settings.data_dir, 'seed.json'), 'utf8')
  assert.ok(!envelope.includes(FIXTURE_SEED.toString('hex')))
  assert.ok(!envelope.includes(PASSWORD))
  for (const name of ['wallet.json', 'seed.json']) {
    assert.equal((await fs.stat(path.join(settings.data_dir, name))).mode & 0o777, 0o600)
  }
  assert.equal((await fs.stat(settings.data_dir)).mode & 0o777, 0o700)
  const opened = await openProfile(settings)
  assert.deepEqual(opened.profile, profile)
  assert.deepEqual(opened.seed, FIXTURE_SEED)
  opened.seed.fill(0)
  assert.ok(FIXTURE_SEED.every(byte => byte === 1), 'initialization must not erase the caller fixture buffer')
  assert.deepEqual(await readProfile(settings), profile)
})

test('wrong password and authenticated ciphertext corruption cannot unlock a wallet', async t => {
  const { settings } = await temporarySettings(t)
  await initializeProfile(settings, { seed: FIXTURE_SEED })
  await fs.writeFile(settings.password_file, 'wrong-fixture-password-for-tests')
  await assert.rejects(openProfile(settings), /Unable to unlock/)
  await fs.writeFile(settings.password_file, PASSWORD)
  const seedPath = path.join(settings.data_dir, 'seed.json')
  const envelope = JSON.parse(await fs.readFile(seedPath, 'utf8'))
  envelope.tag = (envelope.tag.startsWith('00') ? '01' : '00') + envelope.tag.slice(2)
  await fs.writeFile(seedPath, JSON.stringify(envelope))
  await assert.rejects(openProfile(settings), /Unable to unlock/)
})

test('read-only opening requires an existing profile and never creates a wallet', async t => {
  const { settings } = await temporarySettings(t)
  await assert.rejects(openProfile({ data_dir: settings.data_dir }))
  await assert.rejects(fs.stat(settings.data_dir), { code: 'ENOENT' })
  const profile = await initializeProfile(settings, { seed: FIXTURE_SEED })
  await fs.unlink(path.join(settings.data_dir, 'seed.json'))
  assert.deepEqual(await openProfile({ data_dir: settings.data_dir }), { profile, seed: null })
  const altered = { ...profile, chain_id: 11155111 }
  await fs.writeFile(path.join(settings.data_dir, 'wallet.json'), JSON.stringify(altered))
  await assert.rejects(readProfile(settings), /mainnet/)
})

test('initialization refuses to overwrite existing keys or identity', async t => {
  const { settings } = await temporarySettings(t)
  await initializeProfile(settings, { seed: FIXTURE_SEED })
  const names = ['wallet.json', 'seed.json']
  const before = await Promise.all(names.map(name => fs.readFile(path.join(settings.data_dir, name))))
  await assert.rejects(initializeProfile(settings, { seed: Buffer.alloc(64, 2) }), /already exists/)
  const after = await Promise.all(names.map(name => fs.readFile(path.join(settings.data_dir, name))))
  assert.deepEqual(after, before)
})

test('insecure permissions and symlink credential paths are refused', { skip: process.platform === 'win32' }, async t => {
  const { directory, settings } = await temporarySettings(t)
  await initializeProfile(settings, { seed: FIXTURE_SEED })
  await fs.chmod(settings.password_file, 0o644)
  await assert.rejects(openProfile(settings), /owner-only/)
  await fs.chmod(settings.password_file, 0o600)
  await fs.chmod(settings.data_dir, 0o755)
  await assert.rejects(openProfile(settings), /owner-only/)
  await fs.chmod(settings.data_dir, 0o700)
  await fs.chmod(path.join(settings.data_dir, 'seed.json'), 0o644)
  await assert.rejects(openProfile(settings), /unlock|owner-only/)
  await fs.chmod(path.join(settings.data_dir, 'seed.json'), 0o600)
  const symlink = path.join(directory, 'password-link')
  await fs.symlink(settings.password_file, symlink)
  await assert.rejects(openProfile({ ...settings, password_file: symlink }), /Symlink/)
})
