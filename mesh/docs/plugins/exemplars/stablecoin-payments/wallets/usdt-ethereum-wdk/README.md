# Private Ethereum USDT WDK worker

This private Node.js 22.13+ process implements the wallet provider's local signing
and RPC adapter. It has no HTTP listener. Rust owns the exclusive profile lock,
SQLite payment journal and the single unresolved outgoing payment. Mesh core
contains no token, key or settlement logic.

The supported asset is Ethereum mainnet (chain 1) USDT at
`0xdac17f958d2ee523a2206206994597c13d831ec7`, with six decimals. Network fees are
ETH, with 18 decimals, and are never added to USDT quantities. The method is
`ethereum.erc20-memo.v1`. No accounts, hosted signing, paymaster, allowance,
escrow or prepaid balance is involved. A buyer needs both USDT and ETH.

## Install and local storage

Run `npm ci` in this directory. The lockfile pins WDK Core `1.0.0-beta.22`, EVM
`1.0.0-beta.24`, compatible base wallet `1.0.0-beta.21`, and ethers `6.17.0`.
Dependency code and lockfile changes require review; this is a beta SDK.

Settings are `{data_dir, rpc_url?, password_file?}`. The default endpoint is
`https://ethereum-rpc.publicnode.com`; a trusted Ethereum JSON-RPC endpoint can
be substituted. URLs cannot contain credentials, query parameters or fragments.
HTTPS is required except loopback HTTP for deterministic fixtures. RPC calls
have an eight second deadline, bounded responses and no redirects or automatic
application retries. No hosted RPC account is required by this implementation.

Only explicit `initialize` creates a wallet. It derives WDK account zero at
`m/44'/60'/0'/0/0` from a fresh 64-byte local BIP-32 seed. `wallet.json` contains
only the public address/identity; `seed.json` holds the seed encrypted with
AES-256-GCM and a scrypt-derived key (`N=32768,r=8,p=1`, random salt and nonce).
The password comes from an owner-only local file and is never passed in protocol
responses or logs. Directories must be 0700 and files 0600, owned by the current
OS account, with no symlinks in wallet/credential paths. Back up the encrypted
seed and credential separately; no mnemonic/export API is exposed here.

`profile` reads public information offline. `open` without `password_file` is
receive-only and can inspect balances and verify other peers' payments without
unlocking the seed. With a credential, the derived address must match the public
profile. WDK disposal and best-effort buffer wiping occur on worker shutdown;
JavaScript does not guarantee removal of every intermediate memory copy.

## Private protocol

Each input line is `{id,operation,input}`; the matching output is `{id,result}`
or `{id,error}` with a safe error string. Requests run sequentially. Stdout is
reserved for this protocol. The operations are:

- `initialize(settings)` and `profile(settings)` return the persisted public
  profile without any network request.
- `open(settings)` / `info({})` return wallet identity, supported asset/method,
  `can_send`, and the separate `fee_asset` (`native:ETH`, 18 decimals).
- `balance({})` returns finalized USDT units, block number and zero worker-side
  reservation. Rust subtracts its durable reservation from spendable funds.
- `estimate(PaymentRequest)` simulates the full token call and returns a signed
  gas ceiling estimate in wei, plus `gas_limit` and its ETH fee asset.
- `prepare(PayRequest)` validates the immutable obligation and identifier,
  simulates and estimates the exact call, checks USDT/ETH funds, and uses WDK
  `signTransaction` without broadcasting. It returns `signed_transaction` as raw
  hex, `tx_hash`, decimal `nonce`, wei `fee`, and local `expires_at` metadata.
- `broadcast({request,signed_transaction,tx_hash})` independently decodes and
  validates the signature, hash, chain, nonce, fees and exact token calldata,
  then passes the identical bytes to WDK `sendTransaction(rawHex)`. WDK simulates
  again before `eth_sendRawTransaction`; an error never proves non-submission.
- `verify({request,tx_hash})` / `lookup(...)` return the common settlement result.

All token and fee amounts cross the protocol as decimal strings. A zero
`max_fee` means no extra application fee cap; `max_total` is a USDT authorization
and is not mixed with wei. EIP-1559 `gasLimit`, `maxFeePerGas` and priority fee
remain necessary transaction fields. The wallet uses the pending nonce and a
20% gas margin with a fee rate of twice the latest base fee plus the provider's
priority fee quote. Fee estimates are ceilings, not the eventual charged fee.

## Delivery proof and recovery

The 100-byte calldata consists of ERC-20 `transfer(address,uint256)` plus the
32-byte obligation memo. Canonical USDT permits trailing bytes and returns no
boolean; the worker accepts an empty successful `eth_call` return. The complete
100 bytes are simulated and gas-estimated before signing. There is no custom
contract. This binds the payment to its inference obligation and prevents one
transfer receipt from satisfying another obligation.

Before broadcast, Rust must durably persist the exact signed bytes and hash.
Ethereum transactions have **no native expiry**. `expires_at` only limits local
signing/rebroadcast authorization. Once signed, an unresolved nonce remains
reserved even after that deadline or a provider returns no transaction. Do not
create a replacement payment, increment a nonce to escape uncertainty, or treat
RPC absence as proof of failure. Reconcile the original hash indefinitely or
perform explicit operator reconciliation. An external broadcaster can still
submit the same signed bytes after the local deadline; verification therefore
recognizes a valid finalized payment even if mined late. Use this key only in
this provider; concurrent external use can consume its nonce.

Settlement requires the exact signed type-2 mainnet transaction (signature and
hash reconstructed from RPC fields), zero ETH value, canonical USDT destination,
exact transfer arguments/memo, a successful receipt in its canonical block,
Ethereum's `finalized` block tag, and an exact USDT `Transfer` log including its
transaction/block identity. A success receipt without delivery stays held. A
reverted finalized transaction can consume ETH. Ethereum finality commonly takes
many minutes; callers must keep asynchronous pending state instead of retrying
inference or repayment. The implementation trusts its RPC endpoint; it is not a
light client. Pruned RPCs need not retain genesis, so genesis queries are not
required. Unsupported `finalized` queries fail closed.

## Validation

`npm test` runs deterministic local fixtures with real WDK signing. No test
creates a mainnet wallet, reads real credentials, funds an account or broadcasts
to a public chain.

Primary references: [WDK transaction API](https://docs.wdk.tether.io/sdk/wallet-modules/wallet-evm/guides/send-transactions/),
[pinned EVM package](https://github.com/tetherto/wdk-wallet-evm/blob/908c6d9b56a4880bc2166969e23f4f491b6acf7d/package.json),
[Tether supported protocols](https://tether.to/en/supported-protocols/),
[canonical USDT source](https://etherscan.io/address/0xdAC17F958D2ee523a2206206994597C13D831ec7#code),
[Ethereum JSON-RPC](https://ethereum.org/en/developers/apis/json-rpc/).
