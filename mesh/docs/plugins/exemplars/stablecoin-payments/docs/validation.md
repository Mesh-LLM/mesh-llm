# Demo validation — 2026-11-03

The implementation passed local validation against unmodified Mesh commit
`8d03dea4555cbb1ac7cdfce7b9fe1a284f55fb76` (#2654).

- `just check`: formatting, all-target checking, and Clippy with warnings denied.
- `just test`: 34 passing tests, including native Tempo transaction signing,
  encrypted keystores, durable replay/restart recovery, finalized receipt checks,
  authenticated admission, alias resolution, and bounded request-body receipts.
- `just build`: all production and fixture executables built on Apple Silicon.
- Actual-host smoke: two real Mesh processes, real plugin IPC and peer transport,
  production payment engines, offline wallet and inference fixtures. One buffered
  request and one SSE request settled with atomic amounts `[3, 2, 5]`: admission,
  first remainder, then full second usage. Both debt ledgers cleared. The buyer
  restarted between requests, preserved its full node identity and standing, and
  required no second admission payment. Wallet spending was unavailable through
  the HTTP tools API (404).

The smoke used the locally built Mesh host `0.78.0+g8D03DE` (SHA-256
`8e32790b965756da9d9797588bf71ee056d22a22f8e8031a4ec6a9f6344b0193`)
and a hash-verified matching Metal native bundle, Skippy 0.76.1 / ABI 0.1.75.
No model weights were needed. The fixture test sent zero blockchain transactions.

Read-only Tempo mainnet checks confirmed chain 4217 and the canonical OUSD
contract's presence and six decimals.

## Subsequent Linux and live-mainnet validation

The same standalone source checkpoint
`a285e0afbc6469a8da77431a24fd2234589aaec8` also passed `just check` and all
34 tests on Linux, and produced release plugin packages on Vast.ai. An offline
two-host smoke passed there with a complete matching CUDA Mesh product.

A funded local Mac buyer then made two real inference requests to the Vast.ai
seller's `Qwen3.6-27B-Q6_K`: buffered Chat Completions followed by SSE, with a buyer
restart between requests. Each request returned `OK` and reported 17 input tokens
and 1 output token. Three native Tempo transactions finalized: admission 10,
first usage remainder 17, second usage 27 atomic OUSD units. Total inference
payment was 54 units (0.000054 OUSD); total gas was 219 units (0.000219 OUSD).
Both peers independently verified finalized delivery; the seller consumed each
receipt once, and both debt ledgers cleared. The seller remained locked/read-only
with no outgoing attempts.

This validates the Chat Completions payment flow at the pinned checkpoint.
Responses API / bundled-console compatibility, concurrent exchanges and combined
Lightning/stablecoin selection were not implemented or validated. No new live
transactions are needed to inspect or run unit tests for this reference snapshot.
See [the reference overview](../REFERENCE.md) and [demo.md](demo.md) for setup and
explicit spending bounds.
