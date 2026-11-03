import { constants } from 'node:fs'
import * as fs from 'node:fs/promises'
import path from 'node:path'
import { randomBytes, randomUUID, scrypt, createCipheriv, createDecipheriv } from 'node:crypto'
import { promisify } from 'node:util'
import WDK from '@tetherto/wdk'
import WalletManagerEvm from '@tetherto/wdk-wallet-evm'
import { addressHex, CHAIN_ID, USDT } from './validation.mjs'
import { ensure, SafeError } from './errors.mjs'

const deriveKey = promisify(scrypt)
const KDF = Object.freeze({ N: 32768, r: 8, p: 1, maxmem: 64 * 1024 * 1024 })
const FILE_FLAGS = constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0)

export function validateSettings(input) {
  ensure(input && typeof input === 'object' && !Array.isArray(input), 'Wallet settings are required.')
  ensure(Object.keys(input).every(k => ['data_dir','rpc_url','password_file'].includes(k)), 'Unknown wallet setting.')
  ensure(typeof input.data_dir === 'string' && path.isAbsolute(input.data_dir), 'data_dir must be absolute.')
  const settings = { rpc_url: 'https://ethereum-rpc.publicnode.com', ...input }
  const url = new URL(settings.rpc_url)
  ensure(url.protocol === 'https:' || (url.protocol === 'http:' && ['localhost','127.0.0.1','[::1]'].includes(url.hostname)), 'RPC must use HTTPS or loopback HTTP.')
  ensure(!url.username && !url.password && !url.search && !url.hash, 'RPC URL must not contain credentials, query or fragment.')
  settings.rpc_url = url.href.replace(/\/$/, '')
  if (settings.password_file !== undefined) ensure(typeof settings.password_file === 'string' && path.isAbsolute(settings.password_file), 'Credential file paths must be absolute.')
  return settings
}
async function noSymlinks(filename) {
  let current = path.parse(filename).root
  for (const component of filename.slice(current.length).split(path.sep).filter(Boolean)) {
    current = path.join(current, component)
    try { ensure(!(await fs.lstat(current)).isSymbolicLink(), 'Symlink wallet and credential paths are not allowed.') }
    catch (error) { if (error.code === 'ENOENT') return; throw error }
  }
}
export async function privateDirectory(directory, create = false) {
  await noSymlinks(directory)
  if (create) await fs.mkdir(directory, { recursive: true, mode: 0o700 })
  const stat = await fs.lstat(directory)
  ensure(stat.isDirectory() && !stat.isSymbolicLink() && (stat.mode & 0o077) === 0, 'Wallet directory must be real and owner-only (0700).')
  if (process.getuid) ensure(stat.uid === process.getuid(), 'Wallet directory belongs to another OS account.')
}
export async function readPrivateFile(filename, maximum = 65536) {
  await noSymlinks(filename)
  const file = await fs.open(filename, FILE_FLAGS)
  try {
    const stat = await file.stat()
    ensure(stat.isFile() && (stat.mode & 0o077) === 0 && stat.size <= maximum, 'Wallet or credential file must be owner-only and bounded.')
    if (process.getuid) ensure(stat.uid === process.getuid(), 'Wallet file belongs to another OS account.')
    return await file.readFile()
  } finally { await file.close() }
}
export async function credential(filename, minimum = 1) {
  const bytes = await readPrivateFile(filename, 4096)
  let end = bytes.length
  while (end && (bytes[end - 1] === 10 || bytes[end - 1] === 13)) end--
  const value = Buffer.from(bytes.subarray(0, end)); bytes.fill(0)
  ensure(value.length >= minimum, 'Credential file is empty or too short.')
  return value
}
async function writeNew(filename, value) {
  const file = await fs.open(filename, 'wx', 0o600)
  try { await file.writeFile(JSON.stringify(value) + '\n'); await file.sync() }
  finally { await file.close() }
}
function validateProfile(profile) {
  ensure(profile?.version === 1 && profile.chain_id === CHAIN_ID && addressHex(profile.token) === addressHex(USDT), 'Wallet profile is not mainnet Ethereum USDT.')
  ensure(typeof profile.wallet_id === 'string' && profile.wallet_id.length > 0 && profile.wallet_id.length <= 128, 'Invalid wallet profile identity.')
  addressHex(profile.address)
  return profile
}
export async function readProfile(input) {
  const settings = validateSettings(input)
  await privateDirectory(settings.data_dir)
  return validateProfile(JSON.parse((await readPrivateFile(path.join(settings.data_dir, 'wallet.json'))).toString('utf8')))
}
export async function initializeProfile(input, fixture = {}) {
  const settings = validateSettings(input)
  ensure(settings.password_file, 'Initialization requires a local password file.')
  await privateDirectory(settings.data_dir, true)
  for (const name of ['wallet.json','seed.json']) {
    try { await fs.lstat(path.join(settings.data_dir, name)); throw new SafeError('Wallet already exists; never replace its profile or encrypted seed.') }
    catch (error) { if (error.code !== 'ENOENT') throw error }
  }
  const password = await credential(settings.password_file, 16)
  const seed = fixture.seed ? Buffer.from(fixture.seed) : randomBytes(64)
  ensure(seed.length >= 16 && seed.length <= 64, 'Invalid seed length.')
  let key, wdk
  try {
    wdk = new WDK(seed).registerWallet('ethereum', WalletManagerEvm, {})
    const account = await wdk.getAccount('ethereum', 0)
    const profile = { version: 1, wallet_id: fixture.wallet_id ?? randomUUID(), address: addressHex(await account.getAddress()), chain_id: CHAIN_ID, token: USDT }
    const salt = randomBytes(32), iv = randomBytes(12)
    key = await deriveKey(password, salt, 32, KDF)
    const cipher = createCipheriv('aes-256-gcm', key, iv)
    const encrypted = Buffer.concat([cipher.update(seed), cipher.final()])
    await writeNew(path.join(settings.data_dir, 'seed.json'), { version: 1, kdf: 'scrypt', N: KDF.N, r: KDF.r, p: KDF.p,
      salt: salt.toString('hex'), iv: iv.toString('hex'), tag: cipher.getAuthTag().toString('hex'), ciphertext: encrypted.toString('hex') })
    await writeNew(path.join(settings.data_dir, 'wallet.json'), validateProfile(profile))
    const directory = await fs.open(settings.data_dir, 'r')
    try { await directory.sync() } finally { await directory.close() }
    return profile
  } finally { wdk?.dispose(); seed.fill(0); password.fill(0); key?.fill(0) }
}
export async function openProfile(input) {
  const settings = validateSettings(input), profile = await readProfile(settings)
  if (!settings.password_file) return { profile, seed: null }
  const password = await credential(settings.password_file, 16)
  let key
  try {
    const envelope = JSON.parse((await readPrivateFile(path.join(settings.data_dir, 'seed.json'))).toString('utf8'))
    ensure(envelope.version === 1 && envelope.kdf === 'scrypt' && envelope.N === KDF.N && envelope.r === KDF.r && envelope.p === KDF.p, 'Unsupported encrypted wallet format.')
    for (const [name, size] of [['salt',64],['iv',24],['tag',32]]) ensure(typeof envelope[name] === 'string' && new RegExp(`^[0-9a-f]{${size}}$`).test(envelope[name]), 'Invalid encrypted wallet metadata.')
    ensure(typeof envelope.ciphertext === 'string' && /^(?:[0-9a-f]{2}){16,64}$/.test(envelope.ciphertext), 'Invalid encrypted wallet payload.')
    key = await deriveKey(password, Buffer.from(envelope.salt, 'hex'), 32, KDF)
    const decipher = createDecipheriv('aes-256-gcm', key, Buffer.from(envelope.iv, 'hex'))
    decipher.setAuthTag(Buffer.from(envelope.tag, 'hex'))
    const seed = Buffer.concat([decipher.update(Buffer.from(envelope.ciphertext, 'hex')), decipher.final()])
    return { profile, seed }
  } catch { throw new SafeError('Unable to unlock the encrypted wallet with its local credential.') }
  finally { password.fill(0); key?.fill(0) }
}
