# wallet-usdt-ethereum

A separately installed `wallet.stablecoin.v1` provider for **USDT on Ethereum
mainnet**. Rust owns the Mesh operation protocol and durable payment journal.
A private Node.js 22.13+ child process uses pinned **Tether WDK Core and its standard
EVM wallet module** to derive and sign with a local externally owned account.
There is no custodial signer, relayer, escrow, deposit or token allowance.

Canonical USDT is `0xdac17f958d2ee523a2206206994597c13d831ec7`, six decimals,
on chain `1`. Native **ETH** pays gas, measured in wei (18 decimals). ETH fees
are separate from micro-USDT payment amounts. Both peers must use this exact
asset/method pair; other EVM tokens and networks are rejected.

## Build and install

From the standalone workspace, run `just wdk-install`, then `just build` or
`just release`. Packaging includes the executable, worker and locked production
WDK dependencies. Node.js 22.13+ must exist on the target machine; `node_binary`
can select its executable. Install dependencies for the target platform.

Install this provider and `pay-stablecoin` explicitly. Bind the engine's `wallet`
capability to `wallet-usdt-ethereum`. Use **new wallet and engine state directories**;
existing OUSD wallets and ledgers must not be repurposed. Use the installed
executable rather than a command override when running through Mesh.

## Local wallet ownership

Creation happens only through an explicit `init` command. The worker generates
64 random seed bytes locally, derives WDK account zero at the standard Ethereum
path, and encrypts the seed using scrypt and AES-256-GCM. The public profile is
`wallet.json`; the encrypted seed is `seed.json`. These are this plugin's files,
not Ethereum JSON keystore files. Back up both and `payments.sqlite3`, keeping
the password separately. The CLI never prints a mnemonic or private key.

```sh
wallet-usdt-ethereum --state-dir /absolute/private/ethereum-wallet \
  --password-file /absolute/private/ethereum-password init
wallet-usdt-ethereum --state-dir /absolute/private/ethereum-wallet address
```

Supply an owner-only `0600` password file containing at least 16 bytes. Wallet
state directories must be owner-only `0700`; use real paths without symlinks.
Files are created with `0600`. Omitting `password_file` opens a receive-only
wallet that cannot sign. Unlocked seed material lives in the local worker's
memory; the same OS account can access this hot wallet.

Fund the sending address with **USDT on Ethereum mainnet** and **ETH on Ethereum
mainnet**. A receiving-only seller does not need ETH to receive or verify. Wallet
creation needs no signup. The configurable HTTPS RPC is a trusted source of
chain data; its access policy and reliability depend on its operator.

## Settlement and recovery

The worker constructs a direct canonical USDT `transfer(address,uint256)` call
with the exact 32-byte Mesh obligation hash appended to the calldata. It signs
without broadcasting. Rust durably journals the raw signed transaction, its
hash, immutable intent and local policy deadline using SQLite FULL synchronization
before any submission. The nonce is committed inside the signed transaction.
No private key crosses the worker pipe.

Only one unresolved outgoing intent is allowed per wallet. Retrying the same
payment ID can only rebroadcast the exact journaled bytes and hash. LOOKUP only
reconciles. Ethereum transactions have **no native expiry**: the local obligation
deadline can stop rebroadcast, but never releases a reservation or authorizes a
new signature. An absent receipt or RPC error stays unresolved. No automatic
nonce replacement, cancellation or fee bump is implemented. Use a dedicated
wallet and do not send unrelated transactions from another application.

Only a successful transaction in the canonical **finalized** chain, with matching
USDT Transfer log, payer, recipient, amount and signed calldata memo, counts as
paid. Sender and receiver verify independently. A head block or one confirmation
is insufficient. If finalized data is unavailable, the attempt remains pending;
there is no silent downgrade. Mismatched delivery never causes automatic payment.

An initial request may return HTTP 402 while admission finalizes. Let background
reconciliation finish and explicitly retry the inference request once the engine
reports ready; receipt arrival alone never starts inference. Quotes allow one
hour for this first demo. Usage payments also wait for finality before the next
paid request to that peer.

Wallet operations are capability-only and require an operator-granted local
binding. They are not public HTTP/MCP tools. Manual spending also requires
`pay --request /absolute/authorized.json --confirm-mainnet`.

## Fees

`max_total` and the engine's daily budget are **USDT units**. A nonzero `max_fee`
is an optional estimate check in **wei**; zero disables this application check
for the early demo. The signed Ethereum transaction still commits its gas fields.
The sender must have sufficient ETH. Ethereum gas can cost much more than a tiny
inference payment; this implementation demonstrates direct settlement without
claiming economical micropayments. No batching or sponsorship is automated.
