// Public deterministic fixtures. Never fund these accounts.
import { createHash } from 'node:crypto'
import { createServer } from 'node:http'
import { HDNodeWallet, Transaction, id } from 'ethers'

export const USDT = '0xdAC17F958D2ee523a2206206994597C13D831ec7'
export const FIXTURE_SEED = Buffer.alloc(64, 1)
export const OWNER = HDNodeWallet.fromSeed(FIXTURE_SEED).derivePath("m/44'/60'/0'/0/0").address
export const RECIPIENT = HDNodeWallet.fromSeed(Buffer.alloc(64, 2)).derivePath("m/44'/60'/0'/0/0").address
export const TRANSFER_TOPIC = id('Transfer(address,address,uint256)')
export const quantity = value => '0x' + BigInt(value).toString(16)
export const word = value => BigInt(value).toString(16).padStart(64, '0')
export const addressWord = value => value.slice(2).toLowerCase().padStart(64, '0')

export function paymentRequest(overrides = {}) {
  const obligation = {
    version: 1,
    id: 'offline-obligation',
    purpose: 'usage',
    buyer_peer: 'aa'.repeat(32),
    seller_peer: 'bb'.repeat(32),
    request_id: 'offline-request',
    model: 'offline-model',
    offer_id: 'offline-offer',
    asset: { chain_id: 1, token: USDT, decimals: 6 },
    payer: OWNER,
    recipient: RECIPIENT,
    amount: '27',
    expires_at: Math.floor(Date.now() / 1000) + 3600,
    ...overrides
  }
  const memo = '0x' + createHash('sha256').update('mesh-llm/stablecoin/obligation/v1\0')
    .update(JSON.stringify(obligation)).digest('hex')
  return { obligation, memo }
}

export function transferData(request) {
  return '0xa9059cbb' + addressWord(request.obligation.recipient) + word(request.obligation.amount) + request.memo.slice(2)
}

export function block(number, transactions = [], timestamp = Math.floor(Date.now() / 1000)) {
  return {
    number: quantity(number), hash: '0x' + createHash('sha256').update(`offline-ethereum-block-${number}`).digest('hex'),
    parentHash: '0x' + '00'.repeat(32), timestamp: quantity(timestamp),
    nonce: '0x0000000000000000', difficulty: '0x0', totalDifficulty: '0x0',
    gasLimit: quantity(30000000), gasUsed: '0x0', extraData: '0x',
    miner: '0x' + '00'.repeat(20), baseFeePerGas: quantity(1000000000),
    mixHash: '0x' + '00'.repeat(32), logsBloom: '0x' + '00'.repeat(256),
    stateRoot: '0x' + '00'.repeat(32), transactionsRoot: '0x' + '00'.repeat(32),
    receiptsRoot: '0x' + '00'.repeat(32), sha3Uncles: '0x' + '00'.repeat(32),
    uncles: [], transactions
  }
}

export function transactionObject(raw, height = 100) {
  const tx = Transaction.from(raw)
  return {
    hash: tx.hash, from: tx.from, to: tx.to, input: tx.data,
    nonce: quantity(tx.nonce), chainId: quantity(tx.chainId), type: quantity(tx.type),
    value: quantity(tx.value), gas: quantity(tx.gasLimit),
    gasPrice: quantity(tx.gasPrice ?? tx.maxFeePerGas ?? 0),
    maxFeePerGas: tx.maxFeePerGas === null ? undefined : quantity(tx.maxFeePerGas),
    maxPriorityFeePerGas: tx.maxPriorityFeePerGas === null ? undefined : quantity(tx.maxPriorityFeePerGas),
    accessList: tx.accessList ?? [], r: tx.signature.r, s: tx.signature.s,
    v: quantity(tx.signature.yParity), yParity: quantity(tx.signature.yParity),
    blockNumber: quantity(height), blockHash: block(height).hash, transactionIndex: '0x0'
  }
}

export function transferReceipt(raw, request, height = 100) {
  const tx = Transaction.from(raw)
  return {
    transactionHash: tx.hash, blockNumber: quantity(height), blockHash: block(height).hash,
    transactionIndex: '0x0', from: tx.from, to: USDT, contractAddress: null,
    gasUsed: quantity(70000), cumulativeGasUsed: quantity(70000),
    effectiveGasPrice: quantity(1000000000), status: '0x1', type: '0x2',
    logsBloom: '0x' + '00'.repeat(256),
    logs: [{
      address: USDT, topics: [TRANSFER_TOPIC, '0x' + addressWord(request.obligation.payer), '0x' + addressWord(request.obligation.recipient)],
      data: '0x' + word(request.obligation.amount), logIndex: '0x0', transactionIndex: '0x0',
      transactionHash: tx.hash, blockHash: block(height).hash, blockNumber: quantity(height), removed: false
    }]
  }
}

export async function startEthereumFixture(test) {
  const state = {
    chainId: 1, latestHeight: 101, finalizedHeight: 100,
    tokenDecimals: 6, tokenBalance: 1000000000n, ethBalance: 1000000000000000000n,
    nonce: 0, gasEstimate: 70000, simulationResult: '0x', timestamp: Math.floor(Date.now() / 1000),
    broadcasts: [], requests: [], unexpected: [], overrides: new Map(),
    transactions: new Map(), receipts: new Map(), beforeBroadcast: undefined
  }
  function response(method, params) {
    if (state.overrides.has(method)) {
      const value = state.overrides.get(method)
      return typeof value === 'function' ? value(params) : value
    }
    switch (method) {
      case 'eth_chainId': return quantity(state.chainId)
      case 'eth_getCode': return params[0].toLowerCase() === USDT.toLowerCase() ? '0x6001600055' : '0x'
      case 'eth_getBalance': return quantity(state.ethBalance)
      case 'eth_getTransactionCount': return quantity(state.nonce)
      case 'eth_gasPrice': return quantity(1000000000)
      case 'eth_maxPriorityFeePerGas': return quantity(1000000000)
      case 'eth_blockNumber': return quantity(state.latestHeight)
      case 'eth_getBlockByNumber': {
        const height = params[0] === 'finalized' || params[0] === 'safe' ? state.finalizedHeight
          : params[0] === 'latest' || params[0] === 'pending' ? state.latestHeight : Number(BigInt(params[0]))
        const transactions = [...state.transactions.values()].filter(tx => tx.blockNumber === quantity(height))
          .map(tx => params[1] ? tx : tx.hash)
        return block(height, transactions, state.timestamp)
      }
      case 'eth_call': {
        const data = (params[0].data ?? params[0].input ?? '').toLowerCase()
        if (data.startsWith('0x313ce567')) return '0x' + word(state.tokenDecimals)
        if (data.startsWith('0x70a08231')) return '0x' + word(state.tokenBalance)
        if (data.startsWith('0xa9059cbb')) return state.simulationResult
        throw new Error('Unexpected eth_call selector in offline fixture')
      }
      case 'eth_estimateGas': return quantity(state.gasEstimate)
      case 'eth_sendRawTransaction': {
        const raw = params[0]
        state.beforeBroadcast?.(raw)
        state.broadcasts.push(raw)
        return Transaction.from(raw).hash
      }
      case 'eth_getTransactionByHash': return state.transactions.get(params[0].toLowerCase()) ?? null
      case 'eth_getTransactionReceipt': return state.receipts.get(params[0].toLowerCase()) ?? null
    }
    throw new Error(`Unexpected offline Ethereum RPC method: ${method}`)
  }
  function handle(input) {
    try {
      state.requests.push({ method: input.method, params: structuredClone(input.params ?? []) })
      return { jsonrpc: '2.0', id: input.id, result: response(input.method, input.params ?? []) }
    } catch (error) {
      if (!error.rpcError) state.unexpected.push(error.message)
      return { jsonrpc: '2.0', id: input.id, error: error.rpcError ?? { code: -32601, message: error.message } }
    }
  }
  const server = createServer(async (request, reply) => {
    let body = ''
    for await (const chunk of request) body += chunk
    try {
      const input = JSON.parse(body)
      const result = Array.isArray(input) ? input.map(handle) : handle(input)
      reply.writeHead(200, { 'content-type': 'application/json' }); reply.end(JSON.stringify(result))
    } catch (error) {
      state.unexpected.push(error.message)
      reply.writeHead(500); reply.end('{}')
    }
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  test.after(async () => {
    await new Promise(resolve => server.close(resolve))
    if (state.unexpected.length) throw new Error(state.unexpected.join('\n'))
  })
  state.url = `http://127.0.0.1:${server.address().port}`
  return state
}
