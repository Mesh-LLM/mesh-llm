import assert from 'node:assert/strict'
import test from 'node:test'
import { WalletAccountEvm } from '@tetherto/wdk-wallet-evm'
import { Transaction, keccak256 } from 'ethers'
import { addressHex, units, validatePaymentRequest, validateSignedTransaction, verifyEvidence } from '../validation.mjs'
import { OWNER, RECIPIENT, USDT, FIXTURE_SEED, block, paymentRequest, transferData,
  transactionObject, transferReceipt, word } from './ethereum-fixture.mjs'

async function signedFixture(t, request = paymentRequest()) {
  const account = new WalletAccountEvm(FIXTURE_SEED, "0'/0/0")
  t.after(() => account.dispose())
  const transaction = { type: 2, chainId: 1, nonce: 0, to: USDT, value: 0n,
    data: transferData(request), gasLimit: 84000n, maxFeePerGas: 3000000000n, maxPriorityFeePerGas: 1000000000n }
  const raw = await account.signTransaction(transaction)
  const tx = Transaction.from(raw)
  return { account, transaction, raw, tx, request, evidence: transactionObject(raw), receipt: transferReceipt(raw, request),
    canonical: block(100, [tx.hash]), finalized: block(101) }
}

test('Ethereum addresses accept canonical/lowercase forms and reject invalid mixed-case checksums', () => {
  assert.equal(addressHex(OWNER), OWNER.toLowerCase())
  assert.equal(addressHex(OWNER.toLowerCase()), OWNER.toLowerCase())
  const index = [...OWNER].findIndex((character, index) => index > 2 && /[a-f]/i.test(character))
  const changed = OWNER.slice(0, index) + (OWNER[index] === OWNER[index].toLowerCase()
    ? OWNER[index].toUpperCase() : OWNER[index].toLowerCase()) + OWNER.slice(index + 1)
  for (const invalid of [changed, '0x' + '00'.repeat(20), OWNER.slice(2), ` ${OWNER}`, 'not-an-address']) {
    assert.throws(() => addressHex(invalid), /address|checksum/)
  }
})

test('immutable payment terms reject wrong chain/token/decimals, unsafe amounts and memo changes', () => {
  const request = paymentRequest()
  assert.equal(validatePaymentRequest(request, OWNER), request)
  for (const asset of [
    { chain_id: 11155111, token: USDT, decimals: 6 },
    { chain_id: 1, token: RECIPIENT, decimals: 6 },
    { chain_id: 1, token: USDT, decimals: 18 }
  ]) assert.throws(() => validatePaymentRequest(paymentRequest({ asset })), /mainnet/)
  for (const amount of [27, '027', '-1', '1.0', '18446744073709551616']) assert.throws(() => units(amount))
  assert.equal(units('18446744073709551615'), (1n << 64n) - 1n)
  assert.throws(() => validatePaymentRequest({ ...request, memo: '0x' + '00'.repeat(32) }), /Memo/)
  assert.throws(() => validatePaymentRequest(request, RECIPIENT), /payer/)
  assert.throws(() => validatePaymentRequest(paymentRequest({ recipient: OWNER })), /Self-payment/)
})

test('actual WDK detached signing produces exact mainnet type2 bytes with a 32-byte obligation suffix', async t => {
  const { raw, tx, request } = await signedFixture(t)
  assert.equal(validateSignedTransaction(request, raw, { payer: OWNER }), tx.hash)
  assert.equal(tx.from.toLowerCase(), OWNER.toLowerCase())
  assert.equal(tx.chainId, 1n)
  assert.equal(tx.type, 2)
  assert.equal(tx.data.length, 2 + 100 * 2)
  assert.equal(tx.data.slice(-64), request.memo.slice(2))
  assert.equal(keccak256(raw), tx.hash)
})

test('valid signatures cannot authorize altered memo, token, value, recipient, amount or chain', async t => {
  const { account, transaction, request, raw } = await signedFixture(t)
  const mutations = [
    { chainId: 11155111 }, { to: RECIPIENT }, { value: 1n },
    { data: transaction.data.slice(0, -64) + '00'.repeat(32) },
    { data: transferData(paymentRequest({ recipient: OWNER, payer: RECIPIENT })) },
    { data: transferData(paymentRequest({ amount: '28' })) }
  ]
  for (const mutation of mutations) {
    const signed = await account.signTransaction({ ...transaction, ...mutation })
    assert.throws(() => validateSignedTransaction(request, signed))
  }
  const legacy = await account.signTransaction({ ...transaction, type: 0, gasPrice: 1000000000n,
    maxFeePerGas: undefined, maxPriorityFeePerGas: undefined })
  assert.throws(() => validateSignedTransaction(request, legacy), /EIP-1559/)
  assert.throws(() => validateSignedTransaction(request, raw.slice(0, -2)))
})

test('seller independently recovers the signer and accepts only exact canonical finalized delivery', async t => {
  const { request, tx, evidence, receipt, canonical, finalized } = await signedFixture(t)
  receipt.logs[0].logIndex = '0x2a'
  receipt.logs.unshift({ ...receipt.logs[0], data: '0x' + word(1), logIndex: '0x29' })
  const result = verifyEvidence(request, tx.hash, evidence, receipt, canonical, finalized)
  assert.equal(result.state, 'finalized')
  assert.deepEqual(result.receipt, { tx_hash: tx.hash, block_hash: canonical.hash, block_number: 100,
    transfer_log_index: 42, payer: OWNER, recipient: RECIPIENT, amount: '27', memo: request.memo, finalized: true })
})

test('wrong logs, reverted execution, unfinalized blocks and inconsistent RPC evidence do not settle', async t => {
  const { request, tx, evidence, receipt, canonical, finalized } = await signedFixture(t)
  const verify = (transaction = evidence, info = receipt, chainBlock = canonical, final = finalized) =>
    verifyEvidence(request, tx.hash, transaction, info, chainBlock, final)
  assert.equal(verify(null, null).state, 'submitted')
  assert.equal(verify(evidence, receipt, canonical, block(99)).state, 'submitted')
  assert.equal(verify(evidence, { ...receipt, status: '0x0' }).state, 'reverted')
  for (const mutate of [
    value => { value.logs[0].address = RECIPIENT },
    value => { value.logs[0].topics[0] = '0x' + '00'.repeat(32) },
    value => { value.logs[0].topics[1] = '0x' + '00'.repeat(32) },
    value => { value.logs[0].topics[2] = '0x' + '00'.repeat(32) },
    value => { value.logs[0].data = '0x' + word(28) },
    value => { value.logs[0].removed = true },
    value => { value.logs = [] }
  ]) {
    const changed = structuredClone(receipt)
    mutate(changed)
    const result = verify(evidence, changed)
    assert.equal(result.state, 'held')
    assert.equal(result.receipt, null)
  }
  assert.throws(() => verify({ ...evidence, from: RECIPIENT }), /signer/)
  assert.throws(() => verify({ ...evidence, input: '0x' }), /hash|inconsistent/)
  assert.throws(() => verify(evidence, { ...receipt, transactionHash: '0x' + 'cc'.repeat(32) }), /hash/)
  assert.throws(() => verify(evidence, receipt, block(100)), /absent/)
  assert.throws(() => verify(evidence, receipt, { ...canonical, hash: '0x' + 'cc'.repeat(32) }), /Canonical/)
  assert.throws(() => verifyEvidence(paymentRequest({ id: 'other-obligation' }), tx.hash, evidence, receipt, canonical, finalized), /memo/)
})

test('an obligation deadline is not native Ethereum transaction expiry or proof of nonpayment', async t => {
  const request = paymentRequest({ expires_at: Math.floor(Date.now() / 1000) - 3600 })
  const { raw, tx, evidence, receipt, canonical, finalized } = await signedFixture(t, request)
  assert.equal(validateSignedTransaction(request, raw), tx.hash)
  assert.equal(verifyEvidence(request, tx.hash, null, null, canonical, finalized).state, 'submitted')
  assert.equal(verifyEvidence(request, tx.hash, evidence, receipt, canonical, finalized).state, 'finalized')
})
