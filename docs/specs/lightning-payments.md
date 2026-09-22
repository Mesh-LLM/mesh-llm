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
3. The payer validates the invoice and reserves its maximum debit against
   the automatic spending budget.
4. After successful prefill, decode runs concurrently with invoice creation and
   payment observation. The provider buffers backend HTTP output in a bounded
   256 KiB queue, applying backpressure when full. No response headers or body
   are released until its own wallet observes payment arrival (`claiming` or
   terminal fallback). Failed or expired authorization discards buffered output.
5. Authorized output streams through the ordinary OpenAI response path. The
   payer reads concurrently with its terminal payment reconciliation; neither
   ledger records settlement until terminal success is observed.
6. The provider invoices actual output transmitted, and the payer settles under
   the original authorization. No human approval is needed.

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

The wallet is a plugin. `mesh-llm-wallet::provider::WalletProvider` exposes
balance, recent transactions, invoice creation, bounded payment, lookup by
payment hash, and asynchronous `wait_for_payment(payment_hash)` completion.
`mesh-llm-payments` re-exports it under `mesh_llm_payments::wallet` and drives
it from the ledger; neither crate links a wallet SDK.

Concrete wallets are plugin processes that advertise the `wallet.v1`
capability (`mesh-llm-wallet::contract`). The host resolves the provider by
capability, never by plugin name. `mesh-wallet-lexe` is the shipped
implementation: Lexe 0.1.23 on mainnet, built into the `mesh-llm` executable
behind the `wallet-lexe` cargo feature and served blobstore-style as
`mesh-llm --plugin wallet-lexe`, auto-registered as the optional built-in
plugin `wallet-lexe`. No second binary ships. The process starts with the host
but is idle until the first wallet operation; starting it never provisions or
contacts a wallet. `[[plugin]] name = "wallet-lexe" enabled = false` turns it
off at runtime (only `enabled` may be set on a built-in); a different
`wallet.v1` implementation is configured as an ordinary external plugin under
its own name, with the built-in disabled. A build without the `wallet-lexe`
feature (SDK consumers) accepts the same stanza and registers nothing.
NWC, BOLT12 and multi-provider selection are deferred.

Feature layering, so embedding applications never link a wallet SDK:
`payments` (host-runtime, `mesh-llm`, `mesh-llm-embedded-runtime`,
`mesh-llm-sdk`) is the ledger, gates and the `wallet.v1` adapter; `wallet-lexe`
(host-runtime, `mesh-llm`) is the built-in Lexe implementation and the only
feature that links Lexe. The shipped CLI enables both. `mesh-llm-sdk` with
`serving` compiles neither; with `serving,payments` it compiles the ledger and
adapter and expects an external `wallet.v1` plugin.

What stays in the host: token metering, output gating, budgets, the ledger,
settlement bookkeeping, recovery, invoice lifetimes and fee policy. What the
plugin does: turn wallet intents into wallet facts. Response bytes never cross
the plugin boundary, and no per-token IPC exists.

The host supplies every invoice's expiry (`wallet_create_invoice.expiry_secs`);
a plugin must not substitute a provider default. `mesh-llm-payments::lifetimes`
owns the values: input inference invoices expire after 5 minutes, output
invoices after 60 minutes, `fund-wallet` invoices after 24 hours. The seller
waits for the input payment to arrive for exactly the input invoice lifetime.
A payer cannot recall an in-flight HTLC, and the payee's node claims any HTLC
that lands before expiry, so a shorter wait would leave a window in which the
seller has discarded its buffered output but still gets paid for it. Aligning
the two means "the seller gave up" and "the payment can no longer land" are the
same moment; the price is that an unpaid request can hold a backend slot for
up to five minutes. This also bounds how long an unpaid input invoice keeps a
peer blocked.

Receiver-side arrival is a normalized field, `Transaction.claiming`, set by the
plugin only for a pending inbound payment whose HTLC is irrevocably committed.
`status_msg` is display text and nothing in the payment path branches on it.
A plugin that cannot observe claiming leaves the flag false and the receiver
waits for completion, which the contract permits.

`wallet.v1` operations return structured errors with a kind:
`not_open`, `invalid_request`, `not_submitted`, `uncertain`, `failed`. The
host adapter (`network/payments/wallet_plugin.rs`) maps these back onto
`PayError`: only `not_submitted` and `invalid_request` become `NotSubmitted`;
IPC loss, timeouts, `failed` and unstructured errors are `Uncertain` and are
never re-sent. A `not_open` (plugin restarted and lost its open wallet) is
answered with exactly one re-open and one retry. `pay` and the `wait_for_*`
long-polls carry no IPC deadline; the caller owns cancellation by dropping the
future.

The host pins the wallet identity. After the first successful open it writes
`payments/wallet-provider.json` (`plugin`, `wallet_id`, `provider`, `network`)
and refuses to open a plugin or wallet that does not match, because outstanding
reservations and receivables are only meaningful against the wallet that created
them. `has_persisted_wallet` reads this pin; it is side-effect-free and never
starts the plugin. Embedders can still inject their own `WalletFactory` through
`PaymentService::with_factory`; without a plugin manager the service is
ledger-only and every wallet operation fails with a clear error.

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
crashes or observation errors. Wallet settlement notifications do not require operator polling.

Each config directory owns a `payments/` directory (normally
`~/.mesh-llm/payments`). Wallet operations provision the wallet lazily; merely
seeing a paid provider does not provision one. The directory contains:

- `payments.sqlite3` and its WAL: policy, seller prices, frozen request terms,
  approvals, reservations, invoices, payment outcomes, receivables and output
  delivery counts. SQLite uses WAL and synchronous FULL.
- `wallet-provider.json`: the host-owned wallet pin described above.
- `lexe/`: handed to the wallet plugin as its data directory. For
  `mesh-wallet-lexe` it holds `seedphrase.txt`, recovery material persisted
  before wallet provisioning with the SDK's exclusive creation and private file
  permissions; the plugin re-asserts mode 0600 on the seed at every open. Unix
  payment and wallet directories are mode 0700. Protect and back up this
  directory; no seed export UI or encrypted-at-rest application keystore is
  added by this PoC.
- Process locks: one service per directory in the host, one wallet writer per
  directory in the plugin. CLI commands use the running node's API. When the
  node is not running, ledger-only commands (policy, pricing, pending) fall back
  to direct access; wallet commands (balance, fund, send, transactions) need the
  node running because only it owns the wallet plugin.

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
Startup releases abandoned approvals only when no charge was ever recorded.
Live cancellation atomically closes pending or approved requests only before
charge preparation; periodic reconciliation does not sweep live approvals.
Prepared and uncertain charges retain their reservations.

Recovery refreshes input receipts from authoritative wallet status, including
zero-output requests. Output invoices require terminal input success; claiming,
unknown, failed, or unavailable input status preserves debt without invoicing it.
Recovery never reruns inference. Delivered-output debt survives provider restart;
KV state does not. No payment retry can replace a recorded segment invoice.
Receiving-wallet identity is pinned by the input invoice's signed payee key.

An uncertain payment is not released just because its invoice expired. If the
provider remains unreachable or a payment cannot be conclusively reconciled, its
reservation remains held; the PoC has no force-release command. Recovery contacts only the original authenticated peer. Replacing its endpoint
identity while retaining its wallet/database is deliberately unsupported; unrelated
same-model peers are never asked to settle that debt.

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

The payer reserves a routing-fee allowance for each inference payment of
`max(3000 msat, 1% of the amount)` (`pricing::fee_allowance_msat`), so a
request's cap is both inference charges plus both allowances. Seller and payer
compute the cap from the same function and the payer rejects terms that
disagree. The wallet is told the resulting cap per payment and must not submit
a payment whose amount plus fees exceeds it; Lexe preflights a route and
submits that same route only when its total debit fits. Route minimums can
increase the sent amount; that increase also counts against the cap. Actual
outgoing amount and fees are recorded. Operators can choose a different cap for
an explicit `wallet send`.

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

Free-only is the default. Automatic policy enables paid providers when available
and pays without per-inference approval, subject to a positive daily budget and
available wallet funds. There is no separate saved payment opt-in.

For a funded wallet, configure the client once:

```sh
mesh-llm client --auto
# In another terminal, create and externally pay a funding invoice if needed:
mesh-llm wallet fund-wallet --amount-sats 1000
mesh-llm wallet policy --mode automatic --daily-budget-sats 100
# Use an ordinary OpenAI client at http://127.0.0.1:9337/v1.
mesh-llm wallet policy
mesh-llm wallet policy --mode free-only
```

`client --auto` selects mesh discovery, not payment authorization. Funding alone
does not enable spending. Automatic policy retains free providers as candidates;
it neither requires nor guarantees a paid provider. A request whose maximum cost
does not fit the remaining budget is not authorized. There are no mandatory
per-token price caps or extra setup calls. Seller pricing is independent.

Policy persists across restarts and mesh switches. `wallet policy` without flags
reads it without changing it, returning `mode`, `daily_budget_msat`,
`spent_today_msat`, `reserved_msat`, and `remaining_daily_budget_msat`. The latter
is the budget allowance, not a guarantee of wallet funds. Reading policy does not
provision a wallet. Applications use the same `policy` command through
`POST /api/wallet`; no second setting is required.

The budget uses UTC calendar days and actual settlement timestamps, including
fees. Outstanding reservations carry across midnight. Authorization atomically
checks unreserved wallet balance and remaining budget. Switching to free-only
stops new paid inference, but does not cancel already-submitted settlement.
Manual per-inference approval is not supported. Provider/model approvals and
price browsing are separate future features.

Requests may still include `mesh_payment` to **restrict** the profile for that
request (for example `{"mode":"free_only"}`). This is not an opt-in or a stored
setting: requests cannot enable paid use under free-only policy or expand the
budget. The field is stripped before backend or remote forwarding. Existing
remote-ingress restrictions remain authoritative.

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
mesh-llm wallet policy --mode automatic --daily-budget-sats 100
mesh-llm wallet policy --mode free-only
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
`policy` (optional `value`), `pricing`, and `set_pricing`
(`model`, nullable `value`). `expected_pid` and `expected_directory` are optional
local destination checks. The balance response exposes `spendable_msat` and
`available_for_inference_msat` after policy and reservations. `pending` returns
durable request records, including completed history; filter `state="pending"`
when inspecting unfinished work. Transactions and invoices use provider-neutral JSON types.

The [companion mesh-app fork](https://github.com/benthecarman/mesh-app/tree/lightning-wallet)
adds native Wallet menu controls for balance,
funding, sending and history. Its earlier manual-approval controls need updating
to this branch’s free-only/automatic policy. It uses
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
4. On the payer (client-only), configure automatic policy with an explicit small
   daily budget and request that exact model through the local OpenAI endpoint.
   Verify two settled payments on the payer and two receipts on the provider.
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

## Payer-side evidence hooks

`RequestTerms.exchange_id` optionally retains the host's existing OpenAI evidence
exchange ID in the payer's existing JSON terms record. It is not the private
payment recovery ID. Older records omit it; there is no SQL schema migration.
The provider protocol and provider evidence emission are unchanged.

A trusted local plugin declaring `payment.lifecycle.v1` can observe the active
payer exchange: `terms_accepted`, `input_invoice_issued`,
`input_settlement_observed`, `output_invoice_issued`,
`output_settlement_observed`, and `final_accounted`. Subscribe to
`openai.exchange.v1` as well to obtain the host exchange and join on `exchange_id`.
Events contain `exchange_id`, stable `event_ref`, `terms_digest`, `phase`, `source`,
nullable `segment` (0=input, 1=output), nullable `payment_hash`, nullable
`settlement` (`terminal` for wallet success), and `amount_msat`.
Terms acceptance is `payer_asserted` (amount is the approved cap); invoice issuance
is `provider_asserted` as observed by the payer, not independently verified
issuance. Settlement is `wallet_reported`, amount excluding fees. Final accounting
is `payer_asserted`, summing successful debits including fees. No provider-side
claiming observation is implied by this payer-only stream.

The terms digest is lowercase SHA-256 over checked JCS JSON using the existing
host `request_body_digest` helper, applied to the object containing exactly
`exchange_id`, `payee`, `model`, `pricing`, `input_tokens`, `max_output_tokens`,
`max_total_msat`, and `expires_at_ms` (including nulls). Pricing retains its three
named rate/minimum fields. Recovery ID and local peer are excluded. Event refs
use the same construction over the event object with `event_ref` set to `""`.
Integers outside the helper's safe range suppress evidence, not payment.

An eight-event per-exchange queue and one-second publication timeout bound
best-effort delivery. No plugin delivery is awaited by settlement; terms hashing
is skipped without a subscriber. No per-token observation is added. Missing
correlation, disconnects, queue drops, process exits and restart recovery can
leave incomplete evidence; there is no replay or complete audit-log guarantee.
The persisted correlation remains available to recovery tooling, but recovery
currently emits no events. No raw invoice, preimage, wallet transaction ID,
prompt or response text is published. Payment hashes are linkable metadata.


### Failure and recovery boundaries

Connection/write failure or a transport drop while awaiting the initial input
invoice can return to ordinary mesh routing before any payment-capable task is
spawned. Provider-reported prefill failure is also retryable at that boundary.
Malformed invoices, invalid terms and policy refusal remain terminal. After the
validated invoice is handed to the payer task, failures remain terminal even if
its payment is still pending: no fallback may create another bill. Client
disconnection during foreground setup drops that setup without spawning a payer.

Background unpaid-input observation reads at most 32 records per batch and
rotates past unpaid/error records, wrapping at the end. This bounds lookup count,
not wallet-call duration. The cursor is in memory; restart begins at the first
record without deleting or forgiving debt.
