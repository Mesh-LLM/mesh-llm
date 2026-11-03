# Mesh direct stablecoin payments

**Reference demo only.** This is a source snapshot of separately developed
plugins, built against #2654 with no additional Mesh core changes. Start with
[the architecture and demonstrated results](REFERENCE.md). This placement makes
the work reviewable; it does not propose shipping these plugins in Mesh core.

A standalone payment engine and two wallet providers implement direct stablecoin
payments for inference. OUSD on Tempo is the live-tested first rail; USDT on Ethereum
uses Tether's standard Wallet Development Kit (WDK).

- `pay-stablecoin`: peer admission, quoted prices, actual token-usage bills and
  durable debt enforcement.
- `wallet-ousd-tempo`: an operator-owned encrypted local wallet, native Tempo
  transactions, OUSD gas payment and independently verified finalized receipts.
- `wallet-usdt-ethereum`: a locally encrypted WDK seed, direct ERC20 USDT transfers,
  ETH network fees, immutable signed-transaction recovery and finalized receipts.

They communicate through `wallet.stablecoin.v1`, defined in the shared Rust
`mesh-llm-stablecoin-wallet` crate. That crate is linked into the engine and wallet providers;
it is not another installed plugin. Mesh core remains payment agnostic. The
Lightning plugin and its wallet contract remain separate.

The Tempo token is Open USD, **not Origin Dollar**:
`0x20c0000000000000000000006a37da5c996874be`, six decimals. Both payment and
network fees come from the sender's OUSD balance. There is no ETH, paymaster,
custodial service, inference escrow, allowance or reusable customer balance.
Keys are generated and held locally, with no wallet-provider account or API key.
Funding from an existing OUSD holder is a normal crypto transfer; fiat funding
services can impose their own requirements.

The Ethereum wallet accepts only mainnet USDT
`0xdac17f958d2ee523a2206206994597c13d831ec7`, six decimals, chain `1`.
Its sender needs **USDT plus ETH**, with no relay or wallet-provider account. See the
[USDT/Ethereum walkthrough](docs/ethereum-demo.md). Each selected rail uses separate wallet
and engine state; automatic selection between multiple rails is not implemented.

## Build and validate

Install Rust 1.95+, `just` and a C compiler. The Mesh SDK supplies `protoc` unless
the builder sets `PROTOC`. The workspace pins the Mesh
SDK to the reviewed #2654 head and Tempo's Rust SDK to its documented release.

Keep `Cargo.lock` and use the locked build recipes: Tempo's pinned Reth revision
requires the SDK's compatible Alloy dependency set. A fresh unconstrained update
currently selects an incompatible newer trie dependency.

For the Ethereum wallet, install **Node.js 22.13+** and the pinned WDK dependencies first:

```sh
just wdk-install
```

```sh
just build
just check
just test
```

Run Cargo commands serially. For an actual-host offline smoke, use a complete
Mesh product from the same pinned source, including its compatible native
runtime (`just build` in Mesh builds both). `just mesh` alone builds only the
host and is insufficient unless a verified matching native bundle is supplied.
Even the model-free on-demand buyer initializes a native runtime at this pinned
revision. No model weights are needed for the offline fixture.

With a complete product and adjacent `native-runtimes/`, run:

```sh
target/debug/host-smoke --mesh-bin /absolute/path/to/mesh-llm --output-dir /absolute/private/fresh-smoke
```

For a host-only binary plus a separately verified matching bundle, set
`MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR=/absolute/matching/native-runtimes` on the
same command. Both spawned nodes inherit it. The harness disables automatic
runtime downloading, so a missing/incompatible bundle fails closed.

The smoke uses two real Mesh hosts, actual plugin IPC and peer transport, a canned
inference backend and an explicitly fake local wallet. It creates no payment wallet key,
uses no blockchain RPC and sends no transaction. These fixture binaries are
excluded from production plugin archives. A passing smoke is **not** a live
mainnet settlement claim.

## Mainnet demonstration

Follow [the complete local-client and Vast-server walkthrough](docs/demo.md).
It covers packaging, explicit grants, creating a protected wallet, funding it
with mainnet OUSD, enabling spending for one allowlisted peer and checking both
admission and post-decode settlement. Start with a small balance and bounded
per-payment, fee and daily limits. Installation never creates or funds a wallet.

Configuration templates are in [demo/](demo/); every `/ABS/...` and peer/model
placeholder must be replaced. Both runtime plugins are installed explicitly.
Configure them by name and omit `command` overrides: Mesh's identity grant
requires the actual installed executable path, even for the same built binary.
No automatic parent-plugin installation or published catalog entry is assumed.

## Current scope

This is a first-rail demonstration for dedicated nodes running only
`pay-stablecoin` as their payment engine. Simultaneous Lightning/stablecoin
alternatives and cross-engine admission/debt coordination remain follow-on work.
Use serial, bounded chat-completion requests. Completed streamed output cannot
be clawed back; unpaid work blocks later requests rather than guaranteeing that
a malicious buyer will pay. Host death before terminal usage reaches the plugin
can still leave an unbillable exchange; do not erase reservations to bypass it.

The wallet journals exact signed bytes before broadcasting. Unknown outcomes
remain pending and are reconciled; the demo never automatically signs a second
transfer, fee-bumps or cancels an unresolved payment. Receiver-policy holds are
not settlement and never trigger automatic repayment.

See the [wallet implementation and command reference](crates/wallet-ousd-tempo/README.md),
[payment engine](crates/pay-stablecoin/README.md) and [contract](crates/wallet-contract/README.md).
