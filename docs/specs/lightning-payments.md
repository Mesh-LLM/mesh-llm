# Lightning payments PoC

This branch implements two-payment inference over authenticated mesh QUIC
connections, a provider-neutral wallet API, durable settlement, CLI controls, and
an initial Lexe mainnet adapter. Mainnet settlement was exercised on September
18, 2026 with two isolated Lexe wallets and a real Skippy CUDA model. The
validation results and remaining gaps are recorded below.

## Payment flow

1. Select a provider and send the request with its advertised prices. The caller
   may supply the ordinary OpenAI output limit. There is no separate quote round trip.
2. The provider prefills the complete model-tokenized input, then sends an input
   BOLT11 invoice with frozen prices, input count, the backend's resolved output
   allowance, and total cap.
3. The payer validates the invoice and obtains per-request manual approval or
   reserves against its automatic spending budget.
4. After successful prefill, decode runs concurrently with invoice creation and
   payment observation. The provider buffers backend HTTP output in a bounded
   256 KiB queue, applying backpressure when full. No response headers or body
   are released until its own wallet observes payment arrival (`claiming` or
   terminal fallback). Failed or expired authorization discards buffered output.
5. Authorized output streams through the ordinary OpenAI response path. The
   payer reads concurrently with its terminal payment reconciliation; neither
   ledger records settlement until terminal success is observed.
6. The provider invoices actual output transmitted, and the payer settles under
   the original authorization. No second manual approval is needed.

Prices and counts are the provider's claims. Proof of prefill, proof of correct
inference, refunds, and encrypted output are not implemented. Prefill before
payment avoids requiring blind input prepayment; it still exposes providers to
unpaid work. A paid provider must run a single-node text model. Distributed
inference, multimodal paid inference, and fan-out are out of scope.

The supported underlying endpoints are `/v1/chat/completions` and
`/v1/completions`, with one completion per request. Existing OpenAI callers keep
using the local proxy. Payments impose no additional output-token default or
ceiling. Omitted limits use the backend's normal generation settings (by default,
the remaining context window); explicit limits are preserved and may be clamped
to the remaining context by the backend. The former 256-token fallback and
4,096-token payment ceiling have been removed. Input remains limited to 131,072
tokens and requests to 1 MiB. Rejected paid attempts do not silently retry on
another paid provider.

The backend reports its resolved output allowance after tokenization and prefill.
That allowance is persisted and included in the input invoice terms so the payer
can reserve a maximum cost without truncating generation for payment reasons.
It must be positive and no greater than an explicit caller limit, if present.
Longer context allowances can require larger wallet/budget reservations; a
request whose maximum cost does not fit is declined before input payment rather
than silently shortening its answer. Only actual delivered output is charged.
The wire fields are unchanged, and existing persisted requests keep their
original terms. Older payment implementations still apply their own limits.

## Wallet and persistence

`mesh-llm-payments::wallet::WalletProvider` exposes balance, recent transactions,
invoice creation, bounded payment, lookup by payment hash, and asynchronous
`wait_for_payment(payment_hash)` completion. Lexe types stay
inside its private adapter. The default feature selects Lexe 0.1.23 on mainnet.
NWC, BOLT12, and operator-facing provider selection are deferred. Embedders can
inject a `WalletFactory` through `PaymentService::with_factory`; discovery is
side-effect-free and opening remains lazy/single-flight. The default factory
keeps the existing Lexe directory and wallet identity. Routing and settlement
no longer inspect Lexe seed paths themselves.

The payment service awaits provider completion for incoming and pending outgoing
payments. A second method awaits the earliest receiver-side evidence that an
incoming payment has arrived, used only to open the output-delivery gate; it defaults to
the completion wait, so an adapter without such a signal is simply slower, never
wrong. Event-capable adapters can use native subscriptions; Lexe implements both
by polling on a bounded lookup-start cadence, and reports the arrival signal from
its detailed per-payment status. The method returns an
authoritative succeeded/failed transaction and must handle settlement before or
during subscription, support multiple waiters, and tolerate cancellation of an
observer without cancelling or resubmitting the payment. Subscribe before reading
current state, or use a replayable subscription, to avoid missing an update.

Incoming unpaid waits are bounded by invoice expiry; an already-expired invoice
gets a single authoritative lookup to recognize an existing receipt. Outgoing HTLC waits have no
invoice-expiry deadline: an expired invoice does not establish payment failure.
The independent 15-second recovery scan still reconciles durable state after
crashes or observation errors. Local manual-approval polling is separate from
wallet settlement notifications.

Each config directory owns a `payments/` directory (normally
`~/.mesh-llm/payments`). Wallet operations provision the wallet lazily; merely
seeing a paid provider does not provision one. The directory contains:

- `payments.sqlite3` and its WAL: policy, seller prices, frozen request terms,
  approvals, reservations, invoices, payment outcomes, receivables and output
  delivery counts. SQLite uses WAL and synchronous FULL.
- `lexe/seedphrase.txt`: recovery material persisted before wallet provisioning,
  with the SDK's exclusive creation and private file permissions. Unix payment
  and wallet directories are mode 0700. Protect and back up this directory; no
  seed export UI or encrypted-at-rest application keystore is added by this PoC.
- Process locks: one service and wallet writer per directory. CLI commands use
  the running node's API, falling back to direct access only on connect failure.

Payment intent is committed as `prepared` before wallet I/O and changes durably
to `pending` immediately before submission. Only prepared intents may be
submitted during recovery. Unknown outcomes retain their reservation and are
observed by the same payment hash, including after restart; an absent wallet
record never triggers resubmission of a pending attempt. A crash between marking
submission started and making the wallet call can therefore retain a reservation.
The wallet contract distinguishes `NotSubmitted` from `Uncertain` errors.
Definite preflight rejection and terminal failure close unused authorization,
while uncertain sibling charges remain reserved. Successful standalone sends
also release their unused fee allowance during recovery without needing the
original CLI or GUI caller to return.
Recovery never reruns inference. Delivered-output debt survives provider restart;
KV state does not. No payment retry can replace a recorded segment invoice.
Receiving-wallet identity is pinned by the input invoice's signed payee key.

An uncertain payment is not released just because its invoice expired. If the
provider remains unreachable or a payment cannot be conclusively reconciled, its
reservation remains held; the PoC has no force-release command. Recovery of
zero-output completion after a provider changes its endpoint identity also
remains unresolved rather than trusting an unrelated peer's completion claim.

Providers reject a peer with unpaid recorded invoices or finished, delivered
output debt awaiting invoice creation. Background recovery creates missing
output invoices in bounded batches after a crash or temporary wallet failure.
Admission waits up to 30 seconds for the requesting peer's recorded debt to
settle, refreshing only that peer's unpaid invoices. No backend starts during
this wait. The transactional admission check still rejects outstanding debt
after the deadline; claiming alone does not clear it. This covers the window
where the previous HTTP response finished but its trailing invoice is settling. Peer identities
can be replaced, so this blacklist is only a PoC deterrent. Rejecting an input
invoice leaves that peer blocked too; the PoC has no operator unblock command or
automatic removal of expired unpaid receivables.

Definitively failed inference payments stop automatic recovery attempts and
release unused payer authorization; provider debt is not forgiven. The PoC does
not retry a terminally failed invoice from the same wallet or replace an invoice
already bound to a segment. An unpaid invoice may still be settled from another
wallet if it remains payable. Never-submitted and failed attempts do not produce
an automatic refund. Sending an already-settled invoice reports its existing
payment without creating a new approval or debit.

## Prices, fees, and token accounting

The seller explicitly configures per-model input/output msat per million tokens
and a minimum invoice quantum. Free serving is the default. Charges use wide
integer arithmetic, ceiling division and quantum rounding at the invoice
boundary. Zero delivered output produces no output invoice.

Enabling `wallet pricing MODEL` without explicit rates uses 500 input and 1500
output msat per million tokens;
small requests at those rates usually round to 1 msat. A 1 msat accounting unit
is not evidence of economical routing: channel minima, routing fees and liquidity
can dominate. Validate amounts and receiving liquidity with Lexe on mainnet.

The payer allows 1000 msat for each inference payment's additional debit, for a
maximum of 2000 msat per request beyond the capped inference charge. Lexe
preflights a route and submits that same route only when its total debit fits.
Route minimums can increase the sent amount; that increase also counts against
the cap. Actual outgoing amount and fees are recorded. Operators can choose a
different cap for an explicit `wallet send`.

Input includes templates, system messages and tools, including cached input at
the ordinary rate. Output counts canonical accepted tokens, excluding EOS and
rejected speculative candidates. Streaming responses carry ordered usage
watermarks; billing advances only after those bytes are accepted by the peer
transport. Socket acceptance does not prove receipt by the end application.
Partial frames or a crash between transmission and accounting can conservatively
undercharge. Non-streaming responses become billable when their complete JSON
usage is transmitted.

Unpaid output delivery waits until the invoice's actual expiry (Lexe's default:
24 hours), or cancellation. Decode can run ahead within the backend and transport
buffers; the 256 KiB queue bounds this adapter's buffered bytes, not all native
compute or socket buffering. Settlement continues independently of the application's HTTP
connection. The provider stops further generation when it receives cancellation.
Late input payment after state release is recorded but does not regenerate output
or trigger an automatic refund.

## Approval and routing

Manual mode is the default. Each approval covers both invoices and their fee
allowances. Automatic mode requires a positive daily budget. The budget uses UTC
calendar days and actual settlement timestamps, and includes fees. Outstanding
reservations carry across midnight. Approval atomically checks both unreserved
wallet balance and remaining budget so concurrent requests cannot reserve the
same funds.

Client intent defaults to `free_only`, independently of wallet funding or automatic
approval. Configure deliberate paid use through the trusted-local wallet API:

```json
{"command":"payment_intent","value":{"mode":"allow_paid","max_input_msat_per_million":1000000,"max_output_msat_per_million":1000000,"max_total_msat":10000}}
```

Read it with `{"command":"payment_intent"}`; reset with
`{"command":"payment_intent","value":{"mode":"free_only"}}`. This profile-level
setting works for ordinary OpenAI clients without custom request fields. Paid
intent retains free providers as candidates. Rate caps and total debit (including
both fee allowances and invoice rounding) are enforced during ranking and against
exact input-invoice terms before approval, then rechecked after approval. Existing
remote-ingress restrictions remain authoritative. Once submission starts,
changing intent does not cancel existing settlement obligations. Per-request
intent overrides and a dedicated CLI command are not implemented yet.

Routing prefers local inference, eligible paid peers, then free peers. Paid peers
are ranked by estimated input plus maximum output cost for the exact model.
Existing capability, health and context checks still apply. Equal-price choices
retain cache/observed-performance preferences. No universal throughput floor is
introduced. The invoice supplies the actual input charge before authorization.

Only loopback-originated requests can spend. QUIC ingress does not gain wallet
access. Paid routing and management wallet routes both require trusted-local
Host/Origin checks, including for browser simple POSTs with `text/plain` JSON
bodies. Native clients without an Origin header remain supported. Wallet-enabled
nodes refuse the provenance-losing legacy TCP bridge;
normal direct QUIC ingress still supports free inference. A process-owned socket
registry also preserves remote origin for existing legacy connections if wallet
or seller configuration changes while a connection is open. Non-loopback TCP
callers likewise cannot bypass seller charges. Wallet API requests can
bind an expected runtime PID and config directory to reject stale destinations.

Pricing gossip is additive (protobuf field 51 and optional JSON metadata). Older
nodes ignore it, while paid providers reject unpaid ordinary remote inference
with HTTP 402. Free nodes retain the existing mesh ALPNs. Live interoperability
with released v0.76.1 was verified in both directions; see the follow-up results
below. This does not certify every older release.

## CLI and application API

```sh
mesh-llm wallet get-balance
mesh-llm wallet get-transactions --limit 20
mesh-llm wallet fund-wallet
mesh-llm wallet fund-wallet --amount-sats 10000
mesh-llm wallet send lnbc... --max-fee-msat 1000
mesh-llm wallet send lnbc... --amount-msat 10000 --max-fee-msat 1000
mesh-llm wallet pending
mesh-llm wallet approve REQUEST_UUID
mesh-llm wallet reject REQUEST_UUID
mesh-llm wallet policy --mode automatic --daily-budget-sats 100
mesh-llm wallet policy --mode manual
mesh-llm wallet pricing MODEL --input-msat-per-million 500 --output-msat-per-million 1500
mesh-llm wallet pricing MODEL --free
```

`balance`, `transactions`, and `fund` are aliases. `wallet --port PORT` selects
the management port. An explicit global `--config PATH` binds the CLI to that
config directory. Wallet policy lives in the payment ledger, not config.toml;
it persists across engine restarts and mesh switches.

Applications POST JSON to `/api/wallet` on the local management port:

```json
{"command":"balance","expected_pid":12345}
```

Commands are `balance`, `transactions` (`limit`), `fund`, `inspect_invoice`
(`invoice`), `send` (`invoice`, optional `amount_msat`, `max_fee_msat`), `pending`,
`approve`/`reject` (`id`), `policy` (optional `value`), `pricing`, and `set_pricing`
(`model`, nullable `value`). `expected_pid` and `expected_directory` are optional
local destination checks. The balance response exposes `spendable_msat` and
`available_for_inference_msat` after policy and reservations. `pending` returns
durable request records, including completed history; filter `state="pending"`
for approvals. Transactions and invoices use provider-neutral JSON types.

The [companion mesh-app fork](https://github.com/benthecarman/mesh-app/tree/lightning-wallet)
adds native Wallet menu controls for balance,
funding, sending, history, request approvals and manual/automatic policy. It uses
this API and the retained engine PID, with network calls off the UI thread. It
requires this branch's engine; the previously pinned released engine lacks these
routes. It does not embed a second wallet or SDK.

## Validation and operator mainnet runbook

After removing payment-specific output limits, validation passed 34 wallet tests,
16 focused host payment tests, all-target Clippy, and `just build cpu`. The host
tests use real QUIC peers with simulated wallets/backends: omitted limits and an
explicit 65,536-token ceiling both resolve to a 6,000-token backend allowance and
settle 5,000 actual output tokens. Ledger coverage verifies the resolved allowance
survives restart and rejects excess delivery accounting. No mainnet payments were
made for this change.

Prior baseline validation completed: `just build cpu`, 3,581 host unit tests (11 ignored),
24 wallet tests, ten focused host payment tests, the opt-in native model
test, all-target Clippy, and the publish/crate-list/console-print consistency
checks. Desktop `just verify` and native fixture interactions also passed.

Local checks cover invoice substitution, pricing/overflow, concurrent
reservations, restart recovery after an uncertain send, duplicate segment/hash
rejection, two actual QUIC peers with simulated wallets (automatic, manual approval and
cancellation settlement), and native-model prefill
suspension before decode. These do not prove mainnet routing or Lexe availability.

### Mainnet results, September 18, 2026

Two isolated nodes on Ubuntu 26.04 used Lexe mainnet and a private QUIC mesh.
The provider served `Qwen3.8-27B-Lightning` from the Qwen3.8-27B Q4_K_M GGUF on
an RTX 5090 through embedded Skippy. The payer had a 100-sat test spending cap.
Input/output prices were 10,000,000/30,000,000 msat per million tokens, with a
1,000-msat invoice quantum. These are test rates, not new defaults.

- Manual approval withheld application output until authorization. Input and
  output invoices both settled and matched the provider's receipts.
- Automatic non-streaming inference settled both charges and returned correct
  text and usage.
- Disconnecting a stream after five displayed tokens stopped delivery at six
  transmitted tokens, settled the final invoice, and released the reservation.
- Two requests with an exhausted daily budget created no invoices or payments.
  The current error is a misleading capacity-related HTTP 503; it needs a
  payment-specific explanation.
- Restarting the payer preserved policy, invoices and settled records, and a
  subsequent inference succeeded. Restarting the provider preserved its seller
  prices and financial records.
- Killing the payer after its first output token left a durable output debt.
  Recovery after restart settled the original output invoice without a duplicate
  input debit. QUIC buffered the then-current 256-token cap before detecting the
  dead client, so this crash was billed for 256 transmitted tokens. That test
  predates removal of the payment-specific cap; normal context/caller limits now
  bound that exposure. Neither promises billing only for displayed tokens.
- Explicit wallet sending and replaying the same invoice produced one debit.
- Rejecting manual approval returned HTTP 402 without a debit. The provider then
  denied that peer's next request. Explicitly settling the rejected 1-sat test
  invoice cleared the block, and another paid inference succeeded.
- A free offer remained usable with an exhausted automatic budget. Price changes
  propagated through the normal approximately 60-second heartbeat; the wallet
  control API does not currently trigger immediate price gossip.
- Untrusted Host and Origin headers were refused by the wallet API.

The initial 10,000-sat deposit credited 9,950 sats after a 50-sat receiving fee.
Provider receipts also deducted 0.5%: each 1-sat invoice credited 0.995 sats.
These observations demonstrate the tested Lexe-to-Lexe routes only, not arbitrary
Lightning routing or economical sub-satoshi invoices. The successful test sends
used no outgoing routing fees.

Total outgoing test spending was **21 sats across 14 successful payments**,
including the explicit send and rejected-invoice settlement. All recorded
provider invoices were paid at the end. The payer was returned to manual mode;
both isolated wallet directories were retained, and no wallet credentials or
transaction files are included in the downloadable application bundle.

Lexe's default invoice expiry and recovery from an ambiguous in-flight Lightning
HTLC have not been exercised on mainnet. The crash
test interrupted inference after input settlement, rather than interrupting the
wallet's payment submission. Simulated tests cover uncertain-send recovery and
concurrent reservation enforcement. Refunds and proof of computation remain
outside this PoC.

### Follow-up failure and compatibility checks

The following passed without funding new wallets or spending additional bitcoin:

- **Mixed versions:** a released v0.76.1 client received real model output from
  this branch's free provider, and HTTP 402 after that provider enabled paid
  serving. This branch's client also received real model output from the released
  provider. Both directions used separate processes, private meshes, isolated
  profiles, and each binary's matching CPU runtime. Pin
  `runtime.native_runtime.selection = "cpu"`; `--device CPU` alone does not
  prevent the released binary from trying to install a GPU runtime at startup.
- **Concurrent authorization:** 16 simultaneous service requests competed for a
  1,000-msat budget with 700-msat per-request reservations. Exactly one reached
  the simulated wallet. Its 600-msat charge plus 10-msat fee left 300 msat
  available while reserved, then 390 msat after completion.
- **Uncertain HTLC:** simulated payment submission lost its response while the
  wallet retained a pending HTLC. Restart, a status-query outage and invoice
  expiry did not release its reservation or resubmit payment. Separate success
  and failure cases reconciled the original hash, including replay afterward.
- **Expiry:** a real QUIC exchange with a signed, two-second BOLT11 invoice and
  simulated wallets released the waiting backend with zero decoded tokens and
  no output invoice. Approval after expiry never called the wallet. The existing
  policy retains the expired unpaid input invoice in the peer blacklist.
- **Native resource release:** the real SmolLM2 model prefills before approval,
  emits no tokens when the gate reports expiry, and then successfully handles a
  new request on the same single-lane backend.
- **Three-node forwarding:** a remote caller targeting a paid third node could
  not spend an automatically enabled intermediary wallet. Both direct QUIC
  ingress and the legacy loopback bridge returned HTTP 402, with no payer
  records or wallet payment calls. Spoofed loopback `X-Forwarded-For` headers did
  not confer local spending authority.

These faults use controlled wallet responses, not a disrupted mainnet HTLC.
The released CPU archive used for compatibility was
`mesh-llm-v0.76.1-x86_64-unknown-linux-gnu.tar.gz`, SHA256
`ea0dbdc83bb85abe31acf7786b84837a17cf238650c2f27636fba4df8c1e7dd2`,
verified against its published checksum. To repeat with extracted product
bundles and the model fixture below:

```sh
python3 scripts/qa-lightning-compatibility.py \
  --current-binary /absolute/current/mesh-bundle/mesh-llm \
  --released-binary /absolute/released/mesh-bundle/mesh-llm \
  --model /absolute/SmolLM2-135M-Instruct-Q8_0.gguf \
  --output /absolute/new-evidence-directory

just with-lld cargo test -p mesh-llm-payments --lib
just with-lld cargo test -p mesh-llm-host-runtime --lib payment
```

The script writes results and process logs, stops its own nodes, and leaves its
isolated profiles for inspection. It never calls wallet funding or sending.

Notification coverage also exercises already-settled payments, settlement during
initial lookup, multiple subscribers, unrelated and duplicate events, successful
and failed outgoing payments, incoming expiry, and cancellation followed by
restart. A paused Tokio clock verifies that event-backed waiters perform no
periodic status queries. The QUIC payment fixtures now use event-backed wallets.

Review follow-up coverage adds definite preflight rejection, terminal send
failure, recovery of successful/failed sends without manual request finalization,
prepared-intent recovery, and an uncertain payment temporarily absent from wallet
lookup. It also checks that a failed charge cannot release an uncertain sibling,
already-paid invoices create no phantom approvals (and repair older ones),
invalid/reused send invoices cannot reserve funds, and uninvoiced output debt
blocks the debtor through restart and invoice-creation failures.

Transport regressions reject cross-site loopback requests before wallet or peer
access and cancel a payer only after it has consumed a frame prefix, then verify
the remaining frame and final invoice settle without the recovery loop. The
follow-up wallet suite passes 33 tests and the host payment suite passes 13 tests,
including the existing two-/three-node QUIC fixtures. All-target Clippy passes.
The local CPU product builds and the released/current compatibility smoke cases
pass again. These checks spend no mainnet funds; the mainnet results above are
from the earlier build.

For the opt-in CPU generation test, build this checkout's runtime and set:

```sh
MESH_PAYMENT_TEST_MODEL=/absolute/SmolLM2-135M-Instruct-Q8_0.gguf \
MESH_PAYMENT_TEST_RUNTIME="$PWD/target/debug/native-runtimes/meshllm-native-runtime-linux-x86_64-cpu" \
just with-lld cargo test -p skippy-server --features dynamic-native-runtime \
  payments_real_model -- --ignored
```

The tested fixture is `unsloth/SmolLM2-135M-Instruct-GGUF` revision
`9e6855bc4be717fca1ef21360a1db4b29d5c559a`, file
`SmolLM2-135M-Instruct-Q8_0.gguf`, SHA256
`c4a3dd037301b6ecea31d6da37f5cd793ead920dd5ddfe6d589294628d6ce66a`.
Use only a native bundle built from the same checkout.

For operator-run mainnet validation:

1. Build a current release product using the repository's `just release-build`
   or composed release-bundle instructions. Use two separate config directories
   and distinct management/inference ports (or two machines), each with one
   running node. Join a private mesh using the usual invite flow.
2. Run `wallet --port PORT fund-wallet` on each node. Fund a deliberately small
   allowance and confirm spendable balance and receiving liquidity. Keep both
   wallets' recovery material. Do not infer success from an invoice alone.
3. Serve a small text model on the provider. Get its exact ID from `/v1/models`,
   enable prices, and verify the peer advertisement before requesting inference.
   Use prices large enough for practical mainnet routing during this test.
4. On the payer (client-only), select manual policy and request that exact model
   with a small output cap. Confirm the pending request appears after prefill,
   approve it, then verify two settled payments on the payer and two receipts on
   the provider. Compare their amounts, token usage and routing fees.
5. Repeat in automatic mode with a small budget. Launch concurrent requests that
   together exceed the budget and verify rejected reservations do not spend.
6. Cancel a streaming request after output begins. Confirm generation stops and
   only transmitted output is invoiced. Restart after an uncertain payment and
   verify the same hash is reconciled without another debit.
7. Verify rejection, invoice expiry, unpaid-peer denial, free-provider fallback,
   remote requests never spending the relay wallet, and a released peer's free
   inference/gossip interoperability. Record hashes/amounts privately as evidence.

## Future encrypted chunks

Later, the inferencer sends an invoice and N tokens encrypted using a key derived
from that invoice's preimage in the same payload. Payment reveals the preimage,
allowing decryption. Bind ciphertext to the request, sequence and invoice, with a
fresh preimage per independently sold chunk. The ledger already identifies
numbered charge segments; token ranges and the cryptographic construction remain
TODO. Preimage-based access does not prove valid inference or honest prefill.

Proof of prefill remains an open [TODO](../../crates/mesh-llm/TODO.md).

## Advertised price visibility

`GET /v1/models` includes an additive `payment` object for concrete model IDs:
`free_available`, `paid_available`, `binding_quote: false`, and `offers` keyed by
`provider_id`. Each offer includes `paid`, nullable `pricing` (input/output
msat-per-million rates and minimum invoice msat), and peer last-seen age. The
age describes peer contact, not a guaranteed quote timestamp. Mixed free/paid
providers remain separate offers. Local advertised seller prices describe remote
service; ordinary local inference does not pay itself. Unknown external-plugin
pricing is not inferred from these offers. Invoice terms remain authoritative.

## Selected-provider sanity check (initial implementation)

A trusted-local profile can require a recent inference sanity check before its
selected remote provider receives the user's request:

```json
{"command":"vetting_policy","value":{"required":true,"serve_probes":false,"ttl_ms":86400000}}
```

Compatibility default is `required: false`: old peers remain usable, but are not
recorded as verified. Required mode rejects unsupported, busy, timed-out or wrong
answers as unverified/unavailable; it does not accuse a peer of fraud. The
separate `/mesh/vetting/v1` tunnel upgrade leaves payment-v1 frames unchanged.
The provider first returns a versioned supported/unsupported capability response
before accepting a challenge. Providers opt in independently with `serve_probes:
true` (default false). This is selected-peer negotiation, not a gossip-driven
probe. Bounded alternative-provider retry remains follow-up work.

On a cache miss the client sends a randomized, fixed arithmetic challenge, never
a user-authored free prompt. The provider uses an already-local backend with a
32-token output ceiling. Global concurrency is one, request starts are limited
to one per second, and the deadline is ten seconds. The client verifies the
nonce, version and integer answer, and persists the authenticated endpoint ID,
tested model, challenge version and observation time. Cache is bounded to 1024
records; TTL is configurable up to seven days, with 24 hours as the default.
Clock rollback makes a record stale. Concurrent misses serialize and recheck.
Failures have a 30-second bounded in-memory cooldown; per-peer provider probes
are limited to one per minute alongside the global limits. Operators can reset
all local observations with `{"command":"reset_vetting"}`.
The cache does not authorize spending or attest a provider's entire catalogue.
The simple challenge can be scripted or forwarded: this is service sanity, not
cryptographic proof of inference.

Tests cover cache persistence, expiry/rollback, frame bounds, cache hits, and a
real two-node gossip/QUIC/tunnel exchange through a simulated local HTTP backend.
No real model or mainnet wallet is used by that test. Live model validation and
latency evidence, negative probe cohorts and bounded reselection remain required
before calling the complete feature ready.
