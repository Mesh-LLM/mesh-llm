import { initializeProfile, readProfile } from './keys.mjs'
import { EthereumWallet } from './worker-wallet.mjs'
import { ensure, safeFailure } from './errors.mjs'

let wallet
// Dependencies and future diagnostic code must not corrupt the private protocol.
for (const name of ['log','info','warn','error','debug']) console[name] = () => {}
async function dispatch(operation, input) {
  if (operation === 'initialize') return await initializeProfile(input)
  if (operation === 'profile') return await readProfile(input)
  if (operation === 'open') {
    ensure(!wallet, 'Worker wallet is already open.')
    wallet = await EthereumWallet.open(input)
    return await wallet.info()
  }
  ensure(wallet, 'Open the configured wallet before wallet operations.')
  ensure(['info','balance','estimate','prepare','broadcast','verify','lookup'].includes(operation), 'Unknown private wallet operation.')
  return await wallet[operation](input)
}
async function line(value) {
  let id = null
  try {
    const message = JSON.parse(value)
    ensure((typeof message.id === 'string' && message.id.length <= 128) || Number.isSafeInteger(message.id), 'Invalid private request identifier.')
    id = message.id
    const result = await dispatch(message.operation, message.input ?? {})
    process.stdout.write(JSON.stringify({ id, result }) + '\n')
  } catch (error) { process.stdout.write(JSON.stringify({ id, error: safeFailure(error) }) + '\n') }
}
try {
  process.stdin.setEncoding('utf8')
  let pending = ''
  for await (const chunk of process.stdin) {
    pending += chunk
    ensure(Buffer.byteLength(pending) <= 256 * 1024, 'Private worker request exceeds its size limit.')
    let index
    while ((index = pending.indexOf('\n')) >= 0) {
      const next = pending.slice(0, index); pending = pending.slice(index + 1)
      if (next.trim()) await line(next)
    }
  }
  if (pending.trim()) await line(pending)
} catch { process.stdout.write(JSON.stringify({ id: null, error: 'Private wallet input stream failed.' }) + '\n'); process.exitCode = 1 }
finally { wallet?.dispose() }
