import { createHash } from 'node:crypto'
import { getAddress, Transaction, keccak256 } from 'ethers'
import { ensure } from './errors.mjs'

export const CHAIN_ID = 1
export const MAINNET_GENESIS = '0xd4e56740f876aef8c010b86a40d5f56745a118d0906a34e69aec8c0db1cb8fa3'
export const USDT = '0xdac17f958d2ee523a2206206994597c13d831ec7'
export const ASSET = Object.freeze({ chain_id: CHAIN_ID, token: USDT, decimals: 6 })
export const FEE_ASSET = Object.freeze({ chain_id: CHAIN_ID, token: 'native:ETH', decimals: 18 })
export const METHOD = 'ethereum.erc20-memo.v1'
export const TRANSFER_TOPIC = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
const U64_MAX = (1n << 64n) - 1n

export function units(value) {
  ensure(typeof value === 'string' && /^(0|[1-9][0-9]{0,19})$/.test(value), 'Amounts must be unsigned decimal strings.')
  const result = BigInt(value)
  ensure(result <= U64_MAX, 'Amount exceeds supported range.')
  return result
}
export function wei(value) {
  ensure(typeof value === 'string' && /^(0|[1-9][0-9]{0,77})$/.test(value), 'Native fees must be unsigned decimal strings.')
  const result = BigInt(value)
  ensure(result < (1n << 256n), 'Native amount exceeds supported range.')
  return result
}
export function addressHex(value) {
  ensure(typeof value === 'string' && /^0x[0-9a-fA-F]{40}$/.test(value) && !/^0x0{40}$/i.test(value), 'Invalid Ethereum account address.')
  // getAddress also rejects mixed-case addresses with a bad EIP-55 checksum.
  try { return getAddress(value).toLowerCase() } catch { ensure(false, 'Invalid Ethereum address checksum.') }
}
export function hashHex(value) {
  ensure(typeof value === 'string' && /^0x[0-9a-fA-F]{64}$/.test(value), 'Invalid Ethereum transaction or block hash.')
  return value.toLowerCase()
}
export function quantity(value) {
  ensure(typeof value === 'string' && /^0x(?:0|[1-9a-fA-F][0-9a-fA-F]*)$/.test(value) && value.length <= 66, 'Invalid Ethereum RPC quantity.')
  return BigInt(value)
}
export function height(value) {
  const number = quantity(value)
  ensure(number <= BigInt(Number.MAX_SAFE_INTEGER), 'Ethereum block or nonce exceeds supported range.')
  return Number(number)
}
function text(value, maximum = 256) {
  ensure(typeof value === 'string' && value.length > 0 && Buffer.byteLength(value) <= maximum, 'Invalid obligation identifier.')
  return value
}
export function canonicalObligation(value) {
  ensure(value && value.version === 1, 'Unsupported payment obligation.')
  ensure(['admission', 'usage'].includes(value.purpose), 'Unsupported payment purpose.')
  ensure(value.asset?.chain_id === CHAIN_ID && addressHex(value.asset?.token) === addressHex(USDT) && value.asset?.decimals === 6, 'Only mainnet Ethereum USDT is supported.')
  const payer = addressHex(value.payer), recipient = addressHex(value.recipient)
  ensure(payer !== recipient, 'Self-payment is not supported.')
  ensure(units(value.amount) > 0n, 'Zero obligation needs no payment.')
  ensure(Number.isSafeInteger(value.expires_at) && value.expires_at > 0 && value.expires_at <= Math.floor(Number.MAX_SAFE_INTEGER / 1000), 'Invalid obligation expiry.')
  return {
    version: 1, id: text(value.id, 128), purpose: value.purpose,
    buyer_peer: text(value.buyer_peer), seller_peer: text(value.seller_peer),
    request_id: text(value.request_id), model: text(value.model, 1024), offer_id: text(value.offer_id),
    asset: { chain_id: CHAIN_ID, token: value.asset.token, decimals: 6 }, payer: value.payer, recipient: value.recipient,
    amount: value.amount, expires_at: value.expires_at
  }
}
export function obligationMemo(obligation) {
  return '0x' + createHash('sha256').update('mesh-llm/stablecoin/obligation/v1\0')
    .update(JSON.stringify(canonicalObligation(obligation))).digest('hex')
}
export function validatePaymentRequest(request, payer) {
  ensure(request && typeof request === 'object', 'Missing payment request.')
  const obligation = canonicalObligation(request.obligation)
  ensure(request.memo === obligationMemo(obligation), 'Memo does not match payment obligation.')
  if (payer !== undefined) ensure(addressHex(obligation.payer) === addressHex(payer), 'Payment payer differs from this wallet.')
  return request
}
export function transferData(request) {
  return '0xa9059cbb' + addressHex(request.obligation.recipient).slice(2).padStart(64, '0')
    + units(request.obligation.amount).toString(16).padStart(64, '0') + request.memo.slice(2)
}
export function validateSignedTransaction(request, raw, { payer } = {}) {
  validatePaymentRequest(request, payer)
  ensure(typeof raw === 'string' && /^0x(?:[0-9a-fA-F]{2})+$/.test(raw) && raw.length <= 4096, 'Invalid signed Ethereum transaction bytes.')
  let transaction
  try { transaction = Transaction.from(raw) } catch { ensure(false, 'Unable to decode the signed Ethereum transaction.') }
  ensure(transaction.isSigned() && transaction.type === 2 && transaction.chainId === 1n, 'Expected a signed mainnet EIP-1559 transaction.')
  ensure(addressHex(transaction.from) === addressHex(request.obligation.payer), 'Transaction signer differs from payment payer.')
  ensure(addressHex(transaction.to) === addressHex(USDT) && transaction.value === 0n && transaction.data.toLowerCase() === transferData(request), 'Transaction changes token, native value, recipient, amount or memo.')
  ensure(transaction.accessList?.length === 0 && transaction.authorizationList === null && transaction.blobVersionedHashes === null, 'Unexpected Ethereum transaction permissions or extra fields.')
  ensure(transaction.gasLimit > 0n && transaction.maxFeePerGas > 0n && transaction.maxPriorityFeePerGas >= 0n && transaction.maxFeePerGas >= transaction.maxPriorityFeePerGas, 'Invalid signed network fee fields.')
  ensure(Number.isSafeInteger(transaction.nonce) && transaction.nonce >= 0, 'Invalid Ethereum transaction nonce.')
  ensure(hashHex(transaction.hash) === keccak256(raw), 'Signed transaction hash differs from its exact bytes.')
  return hashHex(transaction.hash)
}
// Reconstruct the exact type-2 signed envelope from standard JSON-RPC fields.
// This independently checks the transaction hash and recovers the actual signer.
export function rpcTransactionRaw(value) {
  ensure(value && height(value.type) === 2 && Array.isArray(value.accessList) && value.accessList.length === 0, 'Unexpected on-chain Ethereum transaction type.')
  ensure(value.authorizationList === undefined || value.authorizationList === null, 'Unexpected on-chain delegation.')
  let transaction
  try {
    transaction = Transaction.from({ type: 2, chainId: quantity(value.chainId), nonce: height(value.nonce),
      to: value.to, data: value.input, value: quantity(value.value), gasLimit: quantity(value.gas),
      maxFeePerGas: quantity(value.maxFeePerGas), maxPriorityFeePerGas: quantity(value.maxPriorityFeePerGas),
      accessList: [], signature: { r: value.r, s: value.s, yParity: height(value.yParity ?? value.v) } })
  } catch { ensure(false, 'Invalid on-chain signed transaction evidence.') }
  ensure(addressHex(transaction.from) === addressHex(value.from) && hashHex(transaction.hash) === hashHex(value.hash), 'On-chain transaction hash or signer is inconsistent.')
  return transaction.serialized
}
function result(state, reason) { return { state, receipt: null, reason } }
export function verifyEvidence(request, txHash, transaction, receipt, block, finalizedBlock) {
  validatePaymentRequest(request); txHash = hashHex(txHash)
  if (!transaction || !receipt) return result('submitted', 'Waiting for Ethereum transaction evidence; its original nonce remains reserved.')
  ensure(validateSignedTransaction(request, rpcTransactionRaw(transaction)) === txHash, 'Chain transaction differs from requested payment hash.')
  ensure(hashHex(receipt.transactionHash) === txHash, 'Receipt differs from requested transaction hash.')
  const number = height(receipt.blockNumber), finalizedHeight = height(finalizedBlock.number)
  if (number > finalizedHeight) return result('submitted', 'Waiting for Ethereum consensus finality; the original nonce remains reserved.')
  const blockHash = hashHex(block.hash)
  ensure(height(block.number) === number && hashHex(receipt.blockHash) === blockHash
    && hashHex(transaction.blockHash) === blockHash && height(transaction.blockNumber) === number, 'Canonical block differs from transaction or receipt.')
  ensure(block.transactions?.some(tx => hashHex(typeof tx === 'string' ? tx : tx.hash) === txHash), 'Payment is absent from its canonical finalized block.')
  if (quantity(receipt.status) === 0n) return result('reverted', 'Finalized token transaction reverted; network fees may have been charged.')
  ensure(quantity(receipt.status) === 1n && Array.isArray(receipt.logs), 'Invalid Ethereum execution receipt.')
  const payer = addressHex(request.obligation.payer).slice(2).padStart(64, '0')
  const recipient = addressHex(request.obligation.recipient).slice(2).padStart(64, '0')
  const log = receipt.logs.find(value => !value.removed && typeof value.address === 'string'
    && value.address.toLowerCase() === USDT.toLowerCase() && value.topics?.length === 3
    && value.topics[0]?.toLowerCase() === TRANSFER_TOPIC
    && value.topics[1]?.toLowerCase() === '0x' + payer && value.topics[2]?.toLowerCase() === '0x' + recipient
    && typeof value.data === 'string' && /^0x[0-9a-fA-F]{64}$/.test(value.data) && BigInt(value.data) === units(request.obligation.amount))
  if (!log) return result('held', 'No exact USDT Transfer delivery exists; do not automatically repay.')
  ensure(hashHex(log.transactionHash) === txHash && hashHex(log.blockHash) === blockHash && height(log.blockNumber) === number, 'Transfer log differs from finalized receipt.')
  return { state: 'finalized', reason: null, receipt: { tx_hash: txHash, block_hash: blockHash,
    block_number: number, transfer_log_index: height(log.logIndex), payer: request.obligation.payer,
    recipient: request.obligation.recipient, amount: request.obligation.amount, memo: request.memo, finalized: true } }
}
