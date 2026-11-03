# OUSD wallet on Tempo mainnet

`wallet-ousd-tempo` implements `wallet.stablecoin.v1` for chain **4217** and native
OUSD `0x20c0000000000000000000006a37da5c996874be` (six decimals). It sends native
Tempo type `0x76` transactions and pays gas in the same OUSD balance. No ETH,
paymaster, bundler, account signup, smart-account deployment, allowance or escrow
is used. This is an operator-owned local hot wallet, not a custody service.

The shared Rust contract is the `mesh-llm-stablecoin-wallet` crate. It is linked
into this executable and the separate payment engine; it is not installed as
another runtime plugin. Install both executables explicitly using the workspace
instructions. No automatic parent-plugin installation is assumed.

## Local setup

Build with the workspace `just build`. All commands below are for **mainnet**.
Creating a wallet is an explicit operator action; startup never generates a key.
Prepare a strong password in an owner-only credential file using your existing
secret-management process; neither the password nor a private key is a CLI argument.
The wallet requires at least 16 password bytes. Trailing CR/LF is removed.

```sh
wallet-ousd-tempo --state-dir /absolute/private/wallet --password-file /absolute/private/wallet-password init
wallet-ousd-tempo --state-dir /absolute/private/wallet address
wallet-ousd-tempo --state-dir /absolute/private/wallet balance
```

The state directory must be owner-only (`0700` on Unix); credential and keystore
files must be owner-only (`0600`). New directories/files are created privately.
The key is generated from OS randomness using Alloy, encrypted into a standard
Ethereum JSON keystore, reloaded to verify the address, and stored locally.
Back up the encrypted keystore, `wallet.json`, and `payments.sqlite3` together;
keep its password separately. An encrypted key backup restores funds but cannot
recreate lost obligation/transaction history.

Fund the displayed address with **OUSD on Tempo mainnet** from an existing holder.
No signup is needed to generate an address or receive a crypto transfer. A fiat
on-ramp can have its own requirements. Sending requires enough OUSD for the
inference amount **plus** a maximum gas reserve. Receiving requires no separate
gas asset. Wallet initialization/funding is not performed during development.

Without `password_file`, the provider is locked/read-only and can report the
address/balance or verify incoming payments. For unattended spending, the wallet
process reads the protected password file on startup and holds its key in memory.
Keep that file outside the wallet backup and configure access for only the service
account. The Mesh host receives the pathname in settings, not the secret contents.
Local code with the same OS account can still access a hot wallet; use a dedicated
service account and a small funded balance for this demo. There is no generic
arbitrary-signing operation.

## Mesh configuration

```toml
version = 1

[[plugin]]
name = "wallet-ousd-tempo"

[plugin.settings]
data_dir = "/absolute/private/wallet"
rpc_url = "https://rpc.tempo.xyz"
password_file = "/absolute/private/wallet-password"
max_payment = "1000000" # 1 OUSD per payment, local safety limit
max_fee = "10000"      # 0.01 OUSD per transaction, local safety limit
```

No command arguments are needed when Mesh launches it. Host initialization
settings are authoritative. Install the plugin and resolve it by name; omit a
`command` override so the host recognizes its installed executable for grants.
`wallet_open` accepts `{}` and never selects a caller-
controlled directory or silently initializes a missing keystore.

`--manifest` prints the packaged capability and settings without touching wallet
data or making RPC requests. The wallet's operations are private to explicitly
granted host bindings; they are not published as HTTP/MCP tools. Local management
uses the wallet CLI, and the payment engine applies its policy before sending.
RPC must use HTTPS, except loopback HTTP for
local test fixtures. Every wallet operation verifies mainnet identity and native
OUSD precision. Changing a URL cannot silently switch to a test network or token.

## Direct payment and recovery

`wallet_estimate_payment` accepts a `PaymentRequest` from the shared contract.
`wallet_pay` accepts `PayRequest { payment_id, request, max_fee, max_total }`.
All US-dollar/token amounts are raw six-decimal **strings** on the wire. The
provider constructs `transferWithMemo(recipient, amount, memo)`, sets explicit
OUSD `feeToken` and native `validBefore` equal to the obligation expiry, reserves
sequential nonce lane zero, estimates gas with a 20%
gas-limit margin and twice the current gas price, and refuses an excessive fee,
total or local payment limit. Fees convert attodollars to micro-OUSD with upward
rounding. The operator's limits apply independently of the payment engine.

The journal commits the request, exact signed bytes, nonce and locally computed
hash before the first broadcast. Only one process can own the state directory,
and only one outgoing payment can remain unresolved. Never use this same key in
another wallet or active restored clone. The same payment ID and economic request
returns/reconciles the existing attempt; a different economic request under that
ID is rejected. Unknown RPC outcomes never trigger a fresh payment or nonce.
An explicit retry can rebroadcast only the original identical bytes. This demo
does not automatically fee-bump or cancel transfers.

The provider stops rebroadcasting once the local validity deadline passes. It
releases an expired reservation only after finalized chain time has passed the
signed deadline and the nonce at that same finalized block proves the transaction
never executed. A consumed nonce with missing evidence stays uncertain. Completed
or provably expired attempts remain in the journal; never delete history to reuse
a nonce. PAY and LOOKUP share a lock so concurrent reconciliation cannot overwrite
a completed state with an older pending snapshot.

`wallet_lookup` reconciles by payment ID, `wallet_list_attempts` lists safe public
metadata, and `wallet_verify_payment` independently checks finalized onchain
evidence. RPC timeout is uncertain, not proof of failure. Never delete the journal
or create a replacement payment ID just to clear a timeout.

Manual mainnet spending uses a typed JSON `PayRequest` file and an explicit flag:

```sh
wallet-ousd-tempo --state-dir /absolute/private/wallet --password-file /absolute/private/wallet-password pay --request /absolute/payment.json --confirm-mainnet
wallet-ousd-tempo --state-dir /absolute/private/wallet lookup --payment-id EXISTING_ID
```

The live demo's inference engine calls the same contract through a host operation
binding, with operator-configured budgets. The CLI spending flag is a local manual
command guard, not an extra prompt during authorized engine settlement.

## What counts as paid

Only a canonical successful finalized receipt with both the exact native-OUSD
`Transfer` delivery to the seller and matching `TransferWithMemo` reference counts.
Payer, recipient, exact amount and memo must match the obligation; a balance
increase or a successful transaction alone is insufficient. This catches receiver
policy holds. The returned canonical transfer log index lets the payment engine
atomically prevent receipt reuse. A reorg is uncertain until reconciled; a
canonical revert consumed gas but did not pay inference. Held evidence keeps
future requests blocked for manual investigation instead of automatically paying
again. Zero-value obligations require no transaction.

The engine owns pricing, admission, usage debts, request identity and one-time
receipt consumption; the wallet owns signing, native token mechanics and evidence
verification. Mesh core owns none of these monetary semantics.

SDK: official `tempo-alloy@1.10.1`, Alloy `2.1.1`, Rust `1.95+`. Unit tests use
deterministic fixtures and local temporary data; they send no transactions.
The full transport test uses a loopback JSON-RPC fixture and a public deterministic
test key to exercise encrypted-key loading, signing, journaling, submission,
receipt verification, restart and duplicate-call handling. It makes no external
RPC calls and never funds the fixture key.
