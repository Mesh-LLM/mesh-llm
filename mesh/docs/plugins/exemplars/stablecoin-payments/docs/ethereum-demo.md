# Direct USDT/Ethereum demo with Tether WDK

`wallet-usdt-ethereum` implements the existing `wallet.stablecoin.v1` binding for
**Ethereum mainnet (chain 1)** and canonical USDT
`0xdac17f958d2ee523a2206206994597c13d831ec7` (six decimals).
The payment engine remains `pay-stablecoin`; Mesh core needs no changes.
The earlier Tron experiment is not part of this active implementation.

The buyer needs **USDT and ETH on Ethereum mainnet**. USDT pays the inference
seller; ETH pays gas. This is a standard locally controlled account, with no
relay, smart account, escrow, deposit or token allowance.

## Build and package

Install Rust 1.95+, Node.js 22.13+ (or Node 24), npm, `just` and a C compiler.
Use a complete Mesh product/native runtime at the workspace's pinned #2654 commit.

```sh
just wdk-install
just check
just test
just build
just package-debug aarch64-apple-darwin # actual platform; debug demo packages
```

For optimized packages use `just release` followed by `just package TARGET`.
Build and install the locked Node dependencies separately on each target platform.
The Ethereum archive includes the Rust executable, private WDK worker and its
dependencies. Node.js must be installed on the target. Copying only the Rust
executable is insufficient. Keep both lockfiles.

Pinned WDK Core is `1.0.0-beta.22`, WDK EVM wallet `1.0.0-beta.24`, ethers `6.17.0`,
with shared WDK base wallet `1.0.0-beta.21`. All signing is local. Runtime startup,
installation and builds never create a wallet.

## Explicit local wallet setup

Prepare an owner-only password file (at least 16 bytes) through your normal
secret-management process. Use real absolute paths without symlinks. A separate
wallet and engine state directory is required on each demo node; do not reuse
the existing Tempo wallet or payment ledger.

```sh
wallet-usdt-ethereum --state-dir /ABS/private/ethereum-wallet \
  --password-file /ABS/private/ethereum-password init
wallet-usdt-ethereum --state-dir /ABS/private/ethereum-wallet address
```

Initialization generates a random local seed, derives WDK account zero
(`m/44'/60'/0'/0/0`), and stores `seed.json` encrypted with scrypt/AES-256-GCM.
The public identity is in `wallet.json`. Wallet directories use `0700` and files
`0600`. Back up the profile, encrypted seed and `payments.sqlite3` together; keep
the password separately. No private key or mnemonic is printed or passed as an
argument. Unlocked signing material exists in local process memory.

Fund the **buyer's displayed address** with USDT and ETH on Ethereum mainnet.
The seller can receive and verify while locked and does not need ETH to receive.
Local address creation needs no service signup. The default public RPC needs no
account; an operator may configure another HTTPS mainnet RPC or their own node.

## Install, grant and bind

Install the `pay-stablecoin` and `wallet-usdt-ethereum` archives on both nodes:

```sh
mesh-llm plugins install --archive /ABS/pay-stablecoin-v0.1.0-TARGET.tar.gz
mesh-llm plugins install --archive /ABS/wallet-usdt-ethereum-v0.1.0-TARGET.tar.gz
```

Copy [the Mesh configuration template](../demo/mesh.toml.example). Change the
wallet entry and the engine binding's `provider` to `wallet-usdt-ethereum`, keeping
the same capability and operations. Use this provider entry:

```toml
[[plugin]]
name = "wallet-usdt-ethereum"

[plugin.settings]
data_dir = "/ABS/private/ethereum-wallet"
rpc_url = "https://ethereum-rpc.publicnode.com"
# password_file = "/ABS/private/ethereum-password" # buyer only
node_binary = "node" # or an absolute Node.js 22.13+ executable
max_payment = "1000000"
max_fee = "0" # no extra application fee cap in this first demo
```

Use the installed plugin name without a `command` override. Keep the template's
explicit operation binding and inference-hook/identity grants. Automatic parent
installation and a public catalog entry are not implemented.

Use the buyer/seller JSON templates with new engine `data_dir` paths. Set the
buyer's exact seller-peer allowlist and `buying.max_fee` to `"0"`. Prices,
admission, `max_payment`, `max_total` and daily token budgets are **micro-USDT**.
Fee metadata is **ETH in wei** and is never added to micro-USDT. No separate
elaborate fee-budget system is introduced for this cut.

Keep the persistent Mesh identity setup from [the original demo](demo.md):
model-free buyers use `serve` with `[runtime] mode="on_demand"`, a persistent node
key and matching native runtime. Use matching updated payment engines on both
nodes. Do not use this host revision's ephemeral `client` identity for billing.

## Inference and settlement

Use serial bounded `/v1/chat/completions` requests, including SSE with usage.
The built-in console's Responses API is not supported by this demo.

The initial anti-griefing payment must finalize before inference. Ethereum
finality normally takes about 15 minutes, so the first request can return
**402: admission pending** after the short hook deadline. No inference runs for
that rejected request. Inspect status while the original payment reconciles:

```sh
mesh-llm plugins call pay-stablecoin control --input '{"command":"status"}'
```

Once admission is finalized/ready, **explicitly retry the same peer/model**.
The engine reuses the original exchange and paid admission, preserving its
first-request credit without charging again. It does not automatically run
inference. Admission quotes allow an hour; an unresolved expired quote requires
inspection instead of silently creating another charge.

After decode, actual token usage creates the post-hoc payment. Output can finish
before that payment finalizes; wait for both peers' cleared debt before the next
request. The first total is `max(admission, usage cost)`, and later requests pay
full usage. Background recovery checks each pending payment once per pass so
one slow peer cannot prevent other peers' reconciliation.

## Transaction identity and recovery

The worker builds one USDT `transfer(address,uint256)` call and appends the
32-byte obligation hash to its calldata. This is a signed transaction commitment,
not a new USDT event or a custom settlement contract. The deployed USDT contract
accepts trailing calldata; the provider simulates and estimates the complete call
before signing. USDT's valid empty return data is accepted; receipt evidence,
not a Boolean return value, determines delivery.

WDK signs without broadcasting. Rust then commits the immutable signed raw bytes,
hash, nonce and intent before the exact raw bytes can be submitted. Receipts must
match the successful canonical **finalized** block, original transaction and
calldata, signer, asset, exact recipient/amount and Transfer event. Each delivery
is consumed once. RPCs are trusted chain-data sources, not a local light client.

Ethereum transactions have **no native expiration**. A local obligation deadline,
empty receipt, changed nonce or dropped mempool entry does not prove nonpayment.
Keep the original attempt reserved and reconcile its hash. Do not delete history,
refresh the nonce, fee-bump or replace an uncertain transaction. Explicit operator
`control resubmit` may rebroadcast only the same signed bytes; automatic recovery
does not submit a fresh payment. Use a dedicated key, not a concurrently used
wallet or active restored clone.

See [the wallet implementation](../crates/wallet-usdt-ethereum/README.md) and
[validation record](validation.md). Live funded Ethereum settlement is an operator
step after these offline checks; historical OUSD settlement is separate evidence.

Primary references: [WDK EVM signing](https://docs.wdk.tether.io/sdk/wallet-modules/wallet-evm/guides/send-transactions/),
[Tether contracts](https://tether.to/en/supported-protocols/),
[Tether token source](https://github.com/tethercoin/USDT/blob/main/TetherToken.sol),
[Ethereum finality](https://ethereum.org/roadmap/single-slot-finality/).
