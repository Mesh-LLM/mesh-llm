# Reference: direct OUSD payments for Mesh inference

This working demonstration adds **two external runtime plugins** to the
payment-agnostic host introduced by [#2654](https://github.com/Mesh-LLM/mesh-llm/pull/2654).
A local buyer paid a remote GPU seller on Tempo mainnet for real inference,
including admission and post-decode settlement. No additional Mesh core changes
were needed.

This snapshot is for architectural discussion and inspection, not a proposal to
merge payment implementations into Mesh core. It preserves an independent Cargo
workspace, its lockfile, and its immutable Mesh SDK dependency. There are no root
workspace, host, console, or Lightning plugin changes.

## Pieces and ownership

```text
Mesh core: inference lifecycle, usage reports, authenticated peers, plugin hooks
    |
    +-- pay-stablecoin
            |
            | explicit operation binding: wallet.stablecoin.v1
            |
            +-- wallet-ousd-tempo
                    |
                    +-- local encrypted key + Tempo mainnet RPC
```

| Piece | Responsibility |
| --- | --- |
| [`pay-stablecoin`](crates/pay-stablecoin/README.md) | Authenticated peer negotiation, quoted token pricing, admission, usage billing, spending limits, durable debt and one-time receipt consumption. |
| [`wallet-ousd-tempo`](crates/wallet-ousd-tempo/README.md) | Noncustodial key storage, transaction construction/signing, OUSD gas, durable submission recovery, and independent finalized receipt verification. |
| [`mesh-llm-stablecoin-wallet`](crates/wallet-contract/README.md) | Shared Rust types and operation schemas for `wallet.stablecoin.v1`; compiled into both plugins, with no separately installed process. |
| [`mesh-stablecoin-demo-fixture`](crates/demo-fixture/README.md) | Offline fake-wallet/canned-backend fixtures and a two-real-host smoke harness; excluded from production plugin packages. |

The stablecoin engine is parallel to `pay-lightning`. Each payment family owns
its own wallet contract; there is no shared universal wallet coordinator. Mesh
hosts supply generic lifecycle hooks, authenticated peer channels, identity
access and operation bindings. The operator explicitly installs both stablecoin
plugins, grants capabilities, and binds the engine to its wallet provider.
Automatic dependency installation is not implemented.

The contract has generic asset identity fields, but is a first-rail draft exercised
only with OUSD on Tempo. Its receipt and fee models need review for further rails;
this demo does not establish a final contract for every stablecoin chain. The
current budget calculation adds payment and fees in the same asset, which works
for OUSD gas. USDC on Base with ETH gas would need separate asset accounting.

## Payment flow

1. The buyer negotiates with the authenticated seller over `pay.stablecoin.v1`
   and checks its configured peer allowlist, price, payment, fee and daily limits.
2. A new buyer pays the seller's nonrefundable admission minimum `A`. The seller
   verifies finalized payment before admitting the request.
3. Mesh reports actual terminal token usage. If the first request costs `C`, its
   post-decode bill is `max(C - A, 0)`; later requests pay their full cost `C`.
   Admission surplus never carries to another request.
4. The buyer signs and sends a direct transfer. Both sides verify the exact
   payer, recipient, asset, amount and obligation memo against finalized chain
   evidence. The seller clears the debt and permits subsequent work.

There is no escrow, deposit account, reusable credit, allowance or settlement
batching. Durable local state records obligations, admission standing, budgets
and receipts; direct settlement does not mean discarding that state. Unpaid or
uncertain obligations block further requests from the peer.

## Noncustodial wallet implementation

The asset is **Open USD (OUSD), not Origin Dollar**, on Tempo mainnet, chain
**4217**, token `0x20c0000000000000000000006a37da5c996874be`, six decimals.
An explicit local `init` command creates a fresh key from OS randomness and
encrypts it into an Ethereum-compatible JSON keystore. A protected local password
file unlocks unattended sending. Installation and builds never create wallets.
There is no wallet-provider signup, API key or custody account.

The provider locally signs native Tempo **type `0x76`** transactions containing
`transferWithMemo` and an explicit **OUSD fee token**. Payment and gas both come
from OUSD; ETH and a fee sponsor are unnecessary. A receiving seller can remain
locked and needs no funded gas balance.

Before broadcast, SQLite records the exact signed bytes, hash, nonce and
obligation. Unknown outcomes are reconciled by lookup or rebroadcast of those
same bytes, without automatically signing a replacement payment. Finalized,
canonical successful delivery and its matching memo are required; policy-held
transfers are not treated as settled. Each peer checks receipts through its own
configured RPC, which remains a trust dependency; this is not a chain light client.
Wallet operations are binding-only rather than publicly callable HTTP/MCP tools.

## Demonstrated results

The tested host/SDK checkpoint was
`8d03dea4555cbb1ac7cdfce7b9fe1a284f55fb76`, the #2654 head used for the demo.
The standalone source checkpoint was
`a285e0afbc6469a8da77431a24fd2234589aaec8`. All Rust source, manifests, scripts
and the lockfile in this snapshot match that checkpoint; reference documentation
adds the final results and clarifies installed-plugin configuration.

- Formatting, all-target checking, warning-denying Clippy and **34 tests** passed;
  builds completed on Apple Silicon and Linux on Vast.ai.
- The offline smoke exercised real Mesh hosts, plugin IPC and peer transport,
  buffered and SSE Chat Completions, durable buyer restart, exact settlement and
  denial of public HTTP access to wallet spending. It sent no blockchain transfers.
- The live demo used a local Mac buyer and a Vast.ai CUDA seller running
  `Qwen3.6-27B-Q6_K`. A buffered request and a streamed request each returned `OK`,
  with **17 input tokens and 1 output token** per request. The buyer restarted
  between requests and retained its identity and admission standing.
- At **1 OUSD/million input tokens** and **10 OUSD/million output tokens**, the
  three finalized transfers were **0.000010 OUSD** admission,
  **0.000017 OUSD** first-request remainder, and **0.000027 OUSD** second usage:
  **0.000054 OUSD total inference payment**, plus **0.000219 OUSD total gas**.
- Both sides independently verified the receipts and cleared their debt ledgers.
  The seller wallet stayed locked and created no outgoing payment attempt.

These are completed demo observations, not guarantees of current network fees
or a performance benchmark. Private profiles, keys, invites, balances, machine
connection details, transaction identifiers and runtime artifacts are excluded.

## Reproducing and remaining scope

From this directory, use `just build`, `just check` and `just test` serially.
Keep `Cargo.lock`; the dependency set is deliberately pinned. The host also
requires a complete matching native runtime, even for the model-free buyer.
See [the walkthrough](docs/demo.md) and [validation details](docs/validation.md)
for packaging, explicit grants/bindings, persistent buyer identity, funding and
bounded spending policy. Running unit tests does not spend mainnet funds.

The current demo accepts **serial, bounded `/v1/chat/completions` requests**,
including SSE with usage. There is one outstanding exchange per authenticated
peer; a subsequent request may be rejected while settlement is pending. The
bundled console's Responses API path is not supported by this plugin. A temporary
manual Chat Completions client used during testing is outside this snapshot.

Simultaneous Lightning/stablecoin alternatives, method advertisement/selection,
additional assets, concurrent exchanges, console integration and catalog
publication are follow-on work. Post-decode billing cannot compel a malicious
buyer to pay for output already delivered; it blocks future work. A host crash
before terminal usage is delivered can require operator reconciliation. Token
usage reported by the backend remains a seller claim, not proof of computation.
This is a functional mainnet demo, not an audited production payment system.
