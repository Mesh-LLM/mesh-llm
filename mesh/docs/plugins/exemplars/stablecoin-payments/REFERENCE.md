# Reference: direct stablecoin payments for Mesh inference

This demonstration adds a generic stablecoin engine and independently installed
wallet providers to the payment-agnostic host introduced by
[#2654](https://github.com/Mesh-LLM/mesh-llm/pull/2654).
OUSD on Tempo was tested with funded inference between a local buyer and a
Vast.ai seller. The new USDT/Ethereum provider uses Tether WDK and has passed
local signing, recovery, receipt and packaging tests; funded Ethereum inference
remains the next validation step.

This snapshot is for architectural discussion and inspection, not a proposal to
merge payment implementations into Mesh core. It preserves an independent Cargo
workspace, lockfiles and an immutable Mesh SDK dependency. There are no root
workspace, host, console or Lightning plugin changes.

## Pieces and ownership

```text
Mesh core: inference lifecycle, usage reports, authenticated peers, plugin hooks
    |
    +-- pay-stablecoin
            |
            | explicit operation binding: wallet.stablecoin.v1
            |
            +-- wallet-ousd-tempo     OR     wallet-usdt-ethereum
                local key + Tempo RPC      local WDK worker + Ethereum RPC
```

| Piece | Responsibility |
| --- | --- |
| [`pay-stablecoin`](crates/pay-stablecoin/README.md) | Authenticated peer negotiation, quoted prices, admission, usage billing, spending limits, durable debt and one-time receipt consumption. |
| [`wallet-ousd-tempo`](crates/wallet-ousd-tempo/README.md) | Local encrypted key, native Tempo transactions, OUSD gas, submission recovery and finalized receipt verification. |
| [`wallet-usdt-ethereum`](crates/wallet-usdt-ethereum/README.md) | Local encrypted WDK seed, direct mainnet USDT transfers, ETH fees, durable signed-transaction recovery and finalized receipt verification. |
| [`mesh-llm-stablecoin-wallet`](crates/wallet-contract/README.md) | Shared Rust types and operation schemas for `wallet.stablecoin.v1`; compiled into the engine and wallet providers, with no separately installed process. |
| [`mesh-stablecoin-demo-fixture`](crates/demo-fixture/README.md) | Offline fake-wallet/canned-backend fixtures and a two-real-host smoke harness; excluded from production archives. |

The stablecoin engine is parallel to `pay-lightning`. Each payment family owns
its wallet contract; there is no shared universal wallet coordinator. Operators
install the engine and one selected wallet provider on each dedicated demo node,
grant capabilities and bind operations explicitly. Automatic parent installation
and simultaneous selection between rails are not implemented.

The wallet contract now describes an optional separate fee asset. Missing fee
metadata preserves Tempo's existing same-asset accounting. Ethereum declares
18-decimal ETH fees separately from six-decimal USDT payment budgets; these
quantities are never added together. The first cut allows a zero application fee
cap, without introducing an elaborate fee-budget subsystem.

## Payment flow

1. The buyer negotiates over the authenticated `pay.stablecoin.v1` peer channel
   and checks its configured peer allowlist, prices and token budgets.
2. A new buyer pays the seller's nonrefundable admission minimum `A`. The seller
   independently verifies finalized delivery before admitting inference.
3. Mesh reports terminal token usage. If the first request costs `C`, its
   post-decode bill is `max(C - A, 0)`; later requests pay their full cost `C`.
   Admission surplus never carries to another request.
4. The buyer sends a direct transfer. Both peers verify its exact payer,
   recipient, asset, amount and obligation memo against finalized chain evidence.
   Unpaid or uncertain obligations block subsequent work.

There is no escrow, deposit account, reusable balance, allowance or batching.
Durable local state records obligations, admission standing, budgets and consumed
receipts; direct settlement does not mean discarding that state.

Ethereum settlement is asynchronous. Admission quotes allow one hour. The short
host hook can return 402 pending while the original payment reconciles. An
explicit same-peer/model retry attaches to the original admission rather than
charging again or automatically starting inference. Background reconciliation
checks each pending payment once per pass so one peer's slow finality or error
does not starve the rest. An expired unresolved quote requires inspection,
without silently creating a replacement payment.

## Noncustodial wallets

**OUSD/Tempo:** Open USD, not Origin Dollar, chain 4217, token
`0x20c0000000000000000000006a37da5c996874be`, six decimals. Explicit initialization
creates an encrypted local key. Native type `0x76` transactions call
`transferWithMemo` and specify OUSD as the fee token. Payment and gas both use
OUSD; a locked receiving seller needs no funded gas balance.

**USDT/Ethereum:** canonical mainnet USDT, chain 1, token
`0xdac17f958d2ee523a2206206994597c13d831ec7`, six decimals. An explicit local
initialization creates a random seed, derives WDK account zero at
`m/44'/60'/0'/0/0`, and encrypts the seed with scrypt/AES-256-GCM. WDK Core
`1.0.0-beta.22`, EVM wallet `1.0.0-beta.24`, ethers `6.17.0` and the compatible
base wallet `1.0.0-beta.21` are lockfile-pinned. Signing runs in a private local
Node.js 22.13+ child process. No wallet-provider account, relay or hosted signer
is needed. The buyer funds its address with **USDT plus ETH**; the seller can
remain locked and needs no ETH to receive.

The Ethereum worker signs one EIP-1559 transfer with the 32-byte obligation memo
appended to its calldata. It simulates and estimates the full 100 bytes and
accepts USDT's valid empty return data. No custom contract or allowance is used.
Receipt verification independently reconstructs the signed transaction and
requires its canonical finalized block and exact successful USDT Transfer event.

Both providers journal the exact signed bytes before broadcast and verify
receipts through their own configured RPC, which remains a trust dependency.
Ethereum has no native transaction expiry: a local deadline or missing receipt
does not release the unresolved nonce or authorize another payment. Restart
reconciliation only looks up the original attempt; explicit operator resubmission
can rebroadcast the same bytes. Wallet spending remains binding-only and is not
exposed as a public HTTP/MCP tool. Installation never creates or funds a wallet.

## Validation and checkpoints

The host/SDK remains pinned to
`8d03dea4555cbb1ac7cdfce7b9fe1a284f55fb76`, the #2654 checkpoint used by the
original demonstration. The current standalone source checkpoint is
`cd79fb1fa818fbc7ee36fca1b30101e599e0ed32`. All tracked source and build inputs
match that checkpoint; this reference note and the README banner are added here.

Current Ethereum implementation validation on Apple Silicon passed:

- Formatting, all-target checking, Clippy with warnings denied and JS syntax.
- **50 Rust tests and 26 Node tests**, including actual WDK detached signing,
  process restart/raw-byte recovery, separate fee units, finalized receipt
  rejection cases and asynchronous admission without a duplicate payment.
- Local builds and all three plugin archives. The extracted Ethereum archive
  loaded its bundled WDK dependencies and read an offline public fixture profile.
- A fresh two-real-host smoke using the production engine with offline wallet
  and inference fixtures: admission, buffered/SSE billing, buyer restart,
  exact settlement, cleared debts and public-wallet-spending denial.
- Read-only mainnet RPC metadata and zero-value memo-bearing USDT simulation.

**No live Ethereum wallet or USDT transfer was created during implementation.**
Ethereum finality takes many minutes; funded testing must allow admission and
post-decode settlement to finish. The current OUSD services were not restarted.

The original OUSD checkpoint `a285e0afbc6469a8da77431a24fd2234589aaec8` passed
34 tests on Mac and Linux, release packaging on Vast.ai, and funded inference:
a local Mac buyer made buffered and SSE requests to a Vast.ai CUDA seller running
`Qwen3.6-27B-Q6_K`, restarting between requests. Each returned `OK`, with 17 input
tokens and one output token. Three finalized transfers were 0.000010 OUSD
admission, 0.000017 OUSD first remainder and 0.000027 OUSD second usage:
**0.000054 OUSD total payment**, plus **0.000219 OUSD total gas**. Both debt
ledgers cleared; the seller stayed locked with no outgoing attempt. These are
historical observations at that checkpoint, not current fee guarantees or a
claim of funded validation for the new engine changes.

Private profiles, seeds, credentials, invites, balances, machine connection
details, transaction identifiers, dependencies and build artifacts are excluded
from this source snapshot.

## Reproducing and remaining scope

Run `just wdk-install`, then `just check`, `just test` and `just build` serially.
Keep both lockfiles. The host also requires a complete matching native runtime,
even for a model-free buyer. See [OUSD setup](docs/demo.md),
[USDT/Ethereum setup](docs/ethereum-demo.md) and [validation](docs/validation.md)
for installation, grants, bindings, wallet initialization and funding.

The demo uses serial `/v1/chat/completions` requests, including SSE with usage.
The bundled console's Responses API, concurrent exchanges, multi-rail
advertisement/selection, catalog publication and automatic parent installation
remain follow-on work. Post-decode billing blocks future unpaid work; it cannot
compel payment for output already delivered. A host crash before terminal usage
can require operator reconciliation. Backend token usage remains a seller claim,
not proof of computation. This is an early functional demonstration.
