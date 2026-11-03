import WDK from '@tetherto/wdk'
import WalletManagerEvm, { WalletAccountReadOnlyEvm } from '@tetherto/wdk-wallet-evm'
import { toQuantity } from 'ethers'
import { openProfile, validateSettings } from './keys.mjs'
import { EthereumRpc } from './rpc.mjs'
import { ensure, SafeError } from './errors.mjs'
import { ASSET, FEE_ASSET, METHOD, addressHex, hashHex, units, wei, quantity, height,
  validatePaymentRequest, validateSignedTransaction, transferData, verifyEvidence } from './validation.mjs'

export class EthereumWallet {
  static async open(input) {
    const settings = validateSettings(input), { profile, seed } = await openProfile(settings)
    let wdk, rpc
    try {
      rpc = await EthereumRpc.create(settings)
      let account
      if (seed) {
        wdk = new WDK(seed).registerWallet('ethereum', WalletManagerEvm, { provider: rpc.provider, chainId: 1 })
        account = await wdk.getAccount('ethereum', 0)
        ensure(addressHex(await account.getAddress()) === addressHex(profile.address), 'Encrypted seed does not match persisted wallet address.')
      } else account = new WalletAccountReadOnlyEvm(profile.address, { provider: rpc.provider, chainId: 1 })
      return new EthereumWallet(settings, profile, account, rpc, wdk, seed)
    } catch (error) { wdk?.dispose(); rpc?.dispose(); seed?.fill(0); throw error }
  }
  constructor(settings, profile, account, rpc, wdk, seed) {
    Object.assign(this, { settings, profile, account, rpc, wdk, seed })
  }
  dispose() { this.wdk?.dispose(); this.rpc?.dispose(); this.seed?.fill(0); this.seed = null; this.wdk = null }
  async info() {
    await this.rpc.checkNetwork()
    return { wallet_id: this.profile.wallet_id, address: addressHex(this.profile.address), asset: { ...ASSET },
      method: METHOD, can_send: !!this.wdk, fee_asset: { ...FEE_ASSET } }
  }
  async balance() {
    await this.rpc.checkNetwork()
    return await this.rpc.cached('balance', async () => {
      const block = await this.rpc.finalized(), tag = toQuantity(height(block.number))
      const data = '0x70a08231' + addressHex(this.profile.address).slice(2).padStart(64, '0')
      const result = await this.rpc.call('eth_call', [{ to: ASSET.token, data }, tag])
      ensure(typeof result === 'string' && /^0x[0-9a-fA-F]{64}$/.test(result), 'RPC omitted the finalized USDT balance.')
      const balance = units(BigInt(result).toString()).toString()
      return { balance, reserved: '0', spendable: balance, block_number: height(block.number) }
    })
  }
  async buildAndQuote(request) {
    validatePaymentRequest(request, this.profile.address)
    ensure(request.obligation.expires_at * 1000 > Date.now() + 5000, 'Obligation expired or too close to expiry.')
    await this.rpc.checkNetwork()
    const from = addressHex(this.profile.address), data = transferData(request)
    const call = { from, to: ASSET.token, value: '0x0', data }
    const [block, nonce, priority, simulation, estimated, senderCode] = await Promise.all([
      this.rpc.call('eth_getBlockByNumber', ['latest', false]), this.rpc.call('eth_getTransactionCount', [from, 'pending']),
      this.rpc.call('eth_maxPriorityFeePerGas'), this.rpc.call('eth_call', [call, 'latest']),
      this.rpc.call('eth_estimateGas', [call]), this.rpc.call('eth_getCode', [from, 'latest']) ])
    // Canonical USDT's old ERC-20 implementation deliberately returns no bool.
    // Appended memo bytes are included in BOTH simulation and gas estimation.
    ensure(simulation === '0x' || simulation === '0x' + '1'.padStart(64, '0'), 'USDT transfer simulation failed; no signature was created.')
    ensure(senderCode === '0x', 'Only an undelegated externally owned sender account is supported.')
    const baseFee = quantity(block.baseFeePerGas), tip = quantity(priority)
    const gas = quantity(estimated), gasLimit = (gas * 120n + 99n) / 100n
    ensure(gas > 0n && gasLimit <= BigInt(Number.MAX_SAFE_INTEGER), 'Invalid simulated execution gas.')
    const maxFeePerGas = baseFee * 2n + tip
    ensure(maxFeePerGas > 0n, 'Invalid Ethereum fee quote.')
    const unsigned = { type: 2, chainId: 1, nonce: height(nonce), to: ASSET.token, value: 0n, data,
      gasLimit, maxFeePerGas, maxPriorityFeePerGas: tip, accessList: [] }
    const fee = wei((gasLimit * maxFeePerGas).toString()).toString()
    return { unsigned, estimate: { gas_limit: Number(gasLimit), fee, fee_asset: { ...FEE_ASSET }, block_number: height(block.number) } }
  }
  async estimate(request) { return (await this.buildAndQuote(request)).estimate }
  async prepare(input) {
    ensure(this.wdk, 'Wallet is receive-only; configure its local password file to send.')
    ensure(typeof input.payment_id === 'string' && input.payment_id.length > 0 && input.payment_id.length <= 128, 'Invalid payment identifier.')
    ensure(input.payment_id === input.request?.obligation?.id, 'Payment identifier must equal its immutable obligation identifier.')
    validatePaymentRequest(input.request, this.profile.address)
    const maximumFee = wei(input.max_fee), maximumTotal = units(input.max_total)
    ensure(maximumTotal >= units(input.request.obligation.amount), 'Payment exceeds its USDT authorization.')
    const { unsigned, estimate } = await this.buildAndQuote(input.request)
    if (maximumFee > 0n) ensure(BigInt(estimate.fee) <= maximumFee, 'Estimated ETH fee exceeds its explicit authorization.')
    const [balance, ether] = await Promise.all([this.balance(), this.rpc.call('eth_getBalance', [this.profile.address, 'latest'])])
    ensure(BigInt(balance.balance) >= units(input.request.obligation.amount), 'Insufficient finalized USDT for the inference payment.')
    ensure(quantity(ether) >= BigInt(estimate.fee), 'Insufficient ETH for the signed maximum network fee.')
    const signed = await this.account.signTransaction(unsigned)
    const txHash = validateSignedTransaction(input.request, signed, { payer: this.profile.address })
    // expires_at is LOCAL authorization metadata. It is NOT an Ethereum expiry:
    // the exact signed transaction can remain valid indefinitely on chain.
    return { signed_transaction: signed, tx_hash: txHash, nonce: String(unsigned.nonce),
      expires_at: input.request.obligation.expires_at * 1000, fee: estimate.fee }
  }
  async broadcast(input) {
    ensure(this.wdk, 'Wallet is receive-only; reopen its local signer before broadcasting.')
    const txHash = validateSignedTransaction(input.request, input.signed_transaction, { payer: this.profile.address })
    ensure(txHash === hashHex(input.tx_hash), 'Broadcast hash differs from signed transaction.')
    ensure(input.request.obligation.expires_at * 1000 > Date.now(), 'Local authorization deadline passed; reconcile the original hash without replacing its nonce.')
    await this.rpc.checkNetwork()
    try {
      // WDK beta24 simulates again, then submits these exact bytes unchanged.
      // Any failure here is uncertain even if simulation fails: a prior attempt
      // may already have reached the network. Never prepare a replacement.
      const response = await this.account.sendTransaction(input.signed_transaction)
      ensure(hashHex(response.hash) === txHash, 'Node acknowledged a different transaction hash.')
      return { accepted: true }
    } catch (error) { if (error instanceof SafeError) throw error; throw new SafeError('Broadcast outcome is uncertain; reconcile the original transaction hash and nonce.') }
  }
  async verify(input) {
    validatePaymentRequest(input.request)
    const txHash = hashHex(input.tx_hash)
    return await this.rpc.cached(`verify:${txHash}:${input.request.memo}`, async () => {
      await this.rpc.checkNetwork()
      const [transaction, receipt] = await Promise.all([
        this.rpc.call('eth_getTransactionByHash', [txHash]), this.rpc.call('eth_getTransactionReceipt', [txHash]) ])
      if (!transaction || !receipt) return { state: 'submitted', receipt: null,
        reason: 'Waiting for Ethereum evidence; a missing transaction never releases its reserved nonce.' }
      const [finalized, block] = await Promise.all([this.rpc.finalized(),
        this.rpc.call('eth_getBlockByNumber', [receipt.blockNumber, false])])
      return verifyEvidence(input.request, txHash, transaction, receipt, block, finalized)
    })
  }
  async lookup(input) { return await this.verify(input) }
}
