# Lightning payments PoC

> [!WARNING]
> Wallet implementations, including the Lexe wallet, are currently for example
> purposes only and are still being explored.

Paid inference runs over authenticated mesh QUIC connections using a
backend-neutral, token-metered exchange (`/mesh/payment/v3`), a
provider-neutral wallet API, durable settlement and CLI controls. The earlier
fixed-allowance protocols (`/mesh/payment/v1` and `/v2`: prefill before an
input invoice, a resolved output allowance, a pre-payment token cap, a
generation gate and a trailing output invoice for delivered tokens) are no
longer served; a seller answers them with an `Error` frame that asks the payer
to upgrade. Wallets, keys, balances and the ledger file are unchanged by that
cutover. Mainnet settlement was last exercised on September 18, 2026 against
the previous fixed-allowance protocol; those results are kept below for the
wallet, routing and operational findings that still apply.

## Payment flow

Settled protocol rules (not open design choices):

- Cold peers pay a flat **1 sat deposit**, deducted from the final bill.
- After deduction, a remainder below **500 msat has no invoice**; 500–999
  msat produces a 1-sat invoice; larger remainders preserve exact msat.
- Invoice creation, like payment, has no host IPC deadline and is not
  automatically replayed after an uncertain transport failure. The buyer's
  initial admission wait is 300 seconds; client disconnect still cancels it.
  Invoice expiry bounds payment observation, not the creation RPC itself.
- Warm-peer deposit skipping depends on the provider's settled-request cache,
  independently of outstanding finals. Cache loss can require a new deposit.
- Three distinct issued, unpaid final invoices block further admissions;
  paying one clears that strike. Pending finals below that limit have grace.
- Recovery by request ID returns the existing outcome/invoice or `Pending`;
  it never runs inference twice.

Token-metered, backend-neutral, deposit before work, settled from the
backend's own usage report. There are no prefill-time invoices, output
allowance terms, delivery watermarks, receivables or refunds.

1. The payer selects a provider and sends a `Request` frame with its request
   ID, the exact model, the advertised pricing and the ordinary OpenAI HTTP
   request. The payer's own spending cap is local policy and never leaves the
   payer. There is no quote round trip.
2. The provider resolves the backend that serves the model (there is no
   prompt-token estimate; only backend-reported usage is billed), and asks the payments engine to admit the request at its current
   prices. Admission is refused only when the peer already has three distinct
   issued and unpaid final invoices.
3. The provider replies `Admission` with frozen `AdmissionTerms` (`id`,
   `model`, `pricing`, `deposit_msat`). Unless the peer is
   in the provider's trust cache, the terms carry a deposit invoice for a
   flat 1 sat (1000 msat). It only proves the buyer can pay before backend
   work starts. The deposit is applied against the final bill; it is not an
   extra fee, a reservation or an input-token claim.
4. The payer validates the terms against its request and its local cap,
   durably authorizes them, sends `Accepted` and pays any deposit. The
   provider waits for the deposit to settle in its own wallet **before any
   backend work starts**; a peer admitted without a deposit is dispatched at
   once.
5. The provider forwards the request to the local model or plugin inference
   endpoint unchanged and streams the raw HTTP response back as `Output`
   frames. Usage is read from the backend's own response (the JSON `usage`
   object, or the SSE usage chunk; `stream_options.include_usage` is forced
   on). Stream chunks are never counted as tokens, so any OpenAI-compatible
   engine supplies the count with no engine-specific code. When the caller did
   not ask for usage, the payer hides the usage chunk from the application on
   a best-effort basis; all other bytes pass through unchanged.
6. On completion the provider sends `Settlement`: the reported usage, the bill
   (`prompt_tokens` at the input rate + `completion_tokens` at the output
   rate), the deposit applied and, unless `bill - deposit` is under 500 msat,
   one final invoice for that remainder (`final_invoice_msat`: exact to the
   msat, a 500..999 msat remainder is raised to the 1 sat floor). The payer recomputes the bill from the stored
   admission, checks it against its local cap, pays the final invoice under the
   original authorization, and the provider replies `Complete` once that
   invoice has settled. No human approval is needed.

Outcomes:

- **Usage reported:** billed exactly once. A deposit covering the bill means
  no final invoice; the excess is kept, not refunded or carried forward.
- **Abandoned** (backend connection or non-2xx failure, buyer `Cancel`,
  response without a usage report, meter overflow, provider crash before
  settlement): no bill, no invoice, no debt, no strike. A received deposit
  stays received. A missing usage report is never treated as a zero bill.
  `Settlement` carries `usage: null`, `bill_msat: null`,
  `deposit_applied_msat: 0`, no invoice, and is followed by `Complete`.

Billing uses backend-generated tokens, not proof that every token reached the
buyer. The backend is observed independently of buyer transport: a buyer that
goes away mid-response owes nothing (the request is abandoned), while a backend
response that completes after the last frame failed to land is still billed.
Recording that bill does not collect payment from a disconnected buyer; it is
an unpaid final invoice (below). On a `Cancel` or transport failure before the
backend's EOF, only a complete SSE usage event followed by `[DONE]` is retained
as evidence; a provisional usage object, a truncated event or an ordinary JSON
prefix is not. Backends without `[DONE]` still settle on successful EOF but do
not qualify for abort-path billing.

Strikes and trust: a buyer is refused (`Blocked`) at three distinct issued and
unpaid final invoices. Deposits, retries, abandoned requests and bills the
deposit covered are not strikes. Paying a final invoice clears it, including a
payment observed late by the provider's periodic refresh. Trust (deposit
skipping) is an optimization: a peer is trusted only when it has settled a
request in this provider process's lifetime. Outstanding final invoices do not
make a warm peer cold; only the separate three-final admission limit blocks it.
The cache is in memory, bounded to 1,024 peers and cleared when full,
so losing it means the next request asks for a deposit again.

`Recover { id }` lets a payer refetch the settlement of a request after a
disconnect or restart. It never re-dispatches: a request still in progress is
answered `Pending`, an abandoned one with a null-usage `Settlement` and
`Complete`, and a billed one with the stored bill and the same final invoice.
A payer that replays a `Request` this seller already holds past admission is
answered the same way. The request ID is a fresh UUID minted by the payer; it
doubles as the bearer recovery capability and is never published as evidence.

Prices and counts are the provider's claims. Proof of correct inference,
refunds and encrypted output are not implemented. The supported endpoints are
`/v1/chat/completions` and `/v1/completions`, one completion per request
(`n` must be 1), POST, at most 1 MiB. A paid
request without `max_tokens` / `max_completion_tokens` is given the default
output allowance of 16,384 tokens, which the payer inserts into the request it
forwards so the backend is bounded by the same limit the cap was computed
from; an explicit caller limit is preserved. Rejected paid attempts do not
silently retry on another paid provider.

Frames (`mesh-llm-payments-types::metered_wire`) are JSON with a 4-byte length
prefix, at most 4 MiB, `deny_unknown_fields`: `request`, `admission`,
`accepted`, `output`, `settlement`, `complete`, `recover`, `pending`, `cancel`,
`error`. Frames of the retired protocols do not parse. The seller reads the
first frame within 10 seconds and the payer's `Accepted` within 60; the payer
reads the admission within 30 seconds. The backend may go 300 seconds without
producing a byte before the request is abandoned. Usage evidence is collected
from at most 2 MiB of response; beyond that the meter invalidates and the
request is abandoned.

## Wallet and persistence

The wallet is a plugin. `mesh-llm-wallet::provider::WalletProvider` exposes
balance, recent transactions, invoice creation, bounded payment, lookup by
payment hash, and asynchronous `wait_for_payment(payment_hash)` completion.
`mesh-llm-payments` re-exports it under `mesh_llm_payments::wallet` and drives
it from the ledger; neither crate links a wallet SDK.

### Build defaults are not spending policy

Default CLI builds, including releases, include **payment infrastructure**
(`payments`): the ledger, budgets, the metered exchange, recovery and the
generic `wallet.v1` adapter. No wallet implementation or wallet SDK is
compiled into mesh-llm; wallets are external plugins. Compiling the
infrastructure does not enable spending or supply an operational wallet.

Without an available `wallet.v1` provider, wallet-dependent operations cannot
create invoices or settle payments. Free inference does not require a wallet.
A fresh profile uses `free_only` spending policy; installing or funding a wallet
does not enable automatic paid inference. Paying requires a usable wallet and
explicit spending authorization/budget. Charging for serving is configured
separately through model pricing. Existing policy, pricing, ledger and wallet
files are not reset by changing build features; pending settlement still needs
the original wallet to become available again.

Concrete wallets are plugin processes that advertise `wallet.v1`
(`mesh-llm-wallet::contract`). The host finds wallet plugins by that capability
and chooses among them as described below. An installed external wallet uses
the ordinary plugin loader; no Lexe-specific client API is required. Existing
wallet pins bind both plugin name and wallet identity: replacing a provider with
a differently named plugin is not an automatic migration, even when it uses the
same seed. Switch providers with `mesh-llm wallet unpin` (below), which refuses
while anything outstanding depends on the pinned wallet; do not delete the pin
by hand.

### Using the Lexe wallet

Lexe is distributed as the external `lexe-wallet` plugin
([Mesh-LLM/lexe-wallet](https://github.com/Mesh-LLM/lexe-wallet)), Lexe 0.1.24
on mainnet. Build or install its binary, then register it like any external
plugin in `config.toml`:

```toml
[[plugin]]
name = "lexe-wallet"
command = "/path/to/lexe-wallet"

[payments]
wallet = "lexe-wallet"   # optional when it is the only wallet plugin
```

The host hands it `payments/wallets/lexe-wallet/` as its data directory. The
process stays idle until the first wallet operation; startup alone never
provisions or contacts a wallet. NWC and BOLT12 are deferred.

The former built-in `wallet-lexe` is no longer compiled into mesh-llm; a
`[[plugin]] name = "wallet-lexe"` stanza is ignored with a warning.

When more than one `wallet.v1` plugin runs, the host picks one in this order:

1. The plugin named in the ledger's pin (below). A pin is never silently
   replaced; a pinned plugin that is not running is an error.
2. `[payments] wallet = "<plugin>"` in `config.toml`. It must be running.
3. The only running wallet plugin. Two or more need an explicit choice. This
   automatic choice is refused while any enabled plugin that might be a wallet
   is not running, because the choice is pinned: a plugin that crashed has lost
   its capability list, and choosing among the running plugins could pin the
   ledger to the wrong wallet for good. Start the plugin or set `[payments] wallet`.

`wallet_open` also returns the wallet's `features`: whether it can create
amount-less invoices. The host refuses amount-less `fund-wallet` invoices on a
wallet that cannot make them. Plugins written before the field existed get the
original contract's behavior: amount-less invoices.

Every `wallet_pay` carries the fee headroom the host authorized
(`max_total_msat`). A wallet that can bound routing fees must refuse a payment
that would exceed it; one that cannot still pays. Either way the fee the wallet
reports is recorded as spend. With a wallet that cannot bound fees, that fee can
exceed the headroom, and the overrun is recorded after the fact, not prevented:

- The day's spend can go past the daily budget. Later payments are refused
  until the budget recovers.
- The request's spend can go past its cap. The final invoice is still checked
  against the bill, not against what the deposit actually cost, so a deposit
  fee overrun does not by itself leave the seller unpaid.

Feature layering: `payments` (host-runtime, `mesh-llm`,
`mesh-llm-embedded-runtime`, `mesh-llm-sdk`) supplies the ledger, metered
exchange and `wallet.v1` adapter. `mesh-llm-sdk` with `serving` compiles none
of it; with `serving,payments` it compiles payment infrastructure and expects
an external wallet provider. Apps can keep using the generic wallet operator
API while distributing/installing a wallet plugin separately.

Ownership: the host keeps the transport, the backend connection, response
bytes and usage observation, and talks to the payments engine only through
coarse per-request `payments.v1` operations (`mesh-llm-payments-types::contract`).
Buyer side: `prefetch`, `accept_admission`, `pay_deposit`, `settle_final`,
`finish`, `cancel`, `payment_intent`, `routing_budget`, `reconcile`. Seller
side: `serve_admit`, `serve_deposit_confirmed`, `serve_settle`,
`serve_abandon`, `serve_recover`, `settle_received`, `await_arrival`. The
engine (`mesh-llm-payments`, installed in-process by the binary or the SDK's
`payments` feature) owns budgets, the ledger, admission and billing
bookkeeping, recovery, invoice lifetimes and fee policy; host-runtime links
only `mesh-llm-payments-types`. With no engine installed the node is free-only.
Ledger-only operations (`pricing`, `finish`, `payment_intent`, `prefetch`,
`cancel`, `serve_admit`, `serve_settle`, `serve_abandon`, `serve_recover`) must
answer within 10 seconds; operations that wait on a wallet or a peer are
bounded by invoice expiry and the caller's cancellation instead. The provider
is resolved by capability, never by name, for every host-side payment read: the
advertised seller prices used by gossip, `GET /v1/models` and the paid
remote-HTTP check all project through `payments.v1`, the periodic recovery
pass runs against whichever provider serves the capability, and reading prices
never provisions a ledger. An external `payments.v1` provider is therefore a
supported configuration for the operation surface the builtin serves. What the
wallet plugin does: turn wallet intents into wallet facts. Response bytes never
cross the plugin boundary, and no per-token IPC exists.

The host supplies every invoice's expiry (`wallet_create_invoice.expiry_secs`);
a plugin must not substitute a provider default. `mesh-llm-payments::lifetimes`
owns the values: deposit invoices expire after 60 seconds, final invoices after
60 minutes, `fund-wallet` invoices after 24 hours. The seller's wait for the
deposit is bounded by the deposit invoice's lifetime, so an unpaid admission
can hold the exchange (but no backend slot: nothing has been dispatched) for
up to 60 seconds. The final invoice's hour bounds how long a payer that
crashed mid-exchange has to recover and pay this exact invoice.

Receiver-side arrival is a normalized field, `Transaction.claiming`, set by the
plugin only for a pending inbound payment whose HTLC is irrevocably committed.
`status_msg` is display text and nothing in the payment path branches on it.
The metered serving path waits for completion, not arrival, before dispatch;
`await_arrival` remains in the contract for providers that want the earlier
signal.

`wallet.v1` operations return structured errors with a kind:
`not_open`, `invalid_request`, `not_submitted`, `uncertain`, `failed`. The
host adapter (`network/payments/wallet_plugin.rs`) maps these back onto
`PayError`: only `not_submitted` and `invalid_request` become `NotSubmitted`;
IPC loss, timeouts, `failed` and unstructured errors are `Uncertain`. The
plugin transport never automatically replays `wallet_pay` after a lost reply,
including across a plugin restart; the ledger owns subsequent hash-based
reconciliation and any deliberate idempotent resubmission described below.
Other operations retain their existing transport retry behavior.
A `not_open` (plugin restarted and lost its open wallet) is
answered with exactly one re-open and one retry. `pay` and the `wait_for_*`
long-polls carry no IPC deadline; the caller owns cancellation by dropping the
future.

The host pins the wallet identity. After the first successful open it writes
`payments/wallet-provider.json` (`plugin`, `wallet_id`, `provider`, `network`)
and refuses to open a plugin or wallet that does not match, because outstanding
charges and issued invoices are only meaningful against the wallet that created
them. `has_persisted_wallet` reads this pin; it is side-effect-free and never
starts the plugin. `mesh-llm wallet unpin` removes the pin so the next wallet
operation opens and pins another wallet. It only runs while the node is stopped,
because a running node may be opening its wallet against the pin; through the
running node's API it is refused. It is also refused while any outgoing payment
is prepared or pending, or while the ledger records any issued invoice as unpaid,
expired or not: the old wallet may still settle those, or may already have
received a payment the ledger has not recorded. Run the node on the old wallet
to let recovery resolve them, or forgive a peer's unpaid invoices with
`mesh-llm wallet unblock`. Unpin neither moves funds nor cancels invoices: the
old wallet keeps its balance, and a forgiven invoice that has not expired can
still be paid into it after the switch, where the ledger will not see it. The
old wallet's data directory is left in place, so switching back re-adopts the
same identity. Embedders can still inject their own `WalletFactory` through
`PaymentService::with_factory`; without a plugin manager the service is
ledger-only and every wallet operation fails with a clear error.

The payment service awaits provider completion for incoming and pending outgoing
payments. Event-capable adapters can use native subscriptions. To stay within
Lexe's rate limits, the Lexe adapter implements this with one shared watcher
that tails the wallet's payment-update feed. Each poll asks Lexe's gateway for
changes first and only contacts the user node when a payment changed. Lookups
by payment hash, including the recovery scan's, read the SDK's gateway-synced
cache using each payment's created index, recorded from the same feed, so an
idle node stays asleep. The method returns an authoritative succeeded/failed
transaction and must handle settlement before or during subscription, support
multiple waiters, and tolerate cancellation of an observer without cancelling
or resubmitting the payment. Subscribe before reading current state, or use a
replayable subscription, to avoid missing an update.

Incoming unpaid waits are bounded by invoice expiry; an already-expired invoice
gets a single authoritative lookup to recognize an existing receipt. Outgoing
HTLC waits have no invoice-expiry deadline: an expired invoice does not
establish payment failure. The independent 15-second recovery scan still
reconciles durable state after crashes or observation errors. Wallet settlement
notifications do not require operator polling.

Each config directory owns a `payments/` directory (normally
`~/.mesh-llm/payments`). Wallet operations provision the wallet lazily; merely
seeing a paid provider does not provision one. The directory contains:

- `payments.sqlite3` and its WAL: policy, seller prices, buyer request terms
  and pinned admissions, approvals, charges and payment outcomes, and the
  seller's metered requests (`admitted` / `billed` / `abandoned`) with their
  deposit and final invoices. SQLite uses WAL and synchronous FULL.
- `payee_strikes.json`: the buyer's local blocklist of providers that took a
  deposit and delivered nothing (see routing below).
- `wallet-provider.json`: the host-owned wallet pin described above.
- `wallets/<plugin>/`: handed to each wallet plugin as its data directory, so
  switching wallets never exposes one backend's credentials to another. For
  the `lexe-wallet` plugin (`wallets/lexe-wallet/`) it holds `seedphrase.txt`,
  recovery material persisted before wallet provisioning with the SDK's
  exclusive creation and private file permissions; the plugin re-asserts mode
  0600 on the seed at every open. Unix payment and wallet directories are mode
  0700. Protect and back up this directory; no seed export UI or
  encrypted-at-rest application keystore is added by this PoC.
- Process locks: one service per directory in the host, one wallet writer per
  directory in the plugin. CLI commands use the running node's API. When the
  node is not running, ledger-only commands (policy, pricing, pending, blocked,
  unblock, unpin) fall back to direct access; wallet commands (balance, fund,
  send, transactions) need the node running because only it owns the wallet
  plugin.

Payment intent is committed as `prepared` before wallet I/O and changes durably
to `pending` immediately before submission. Unknown outcomes retain their
reservation and are observed by the same payment hash, including after restart.
Metered buyer charges (deposit, segment 0; final, segment 1) are **observed
only** by recovery: the seller's settlement decides what is owed, so a deposit
or final that was never submitted is never resubmitted without that settlement
in hand, and a never-submitted deposit is retired when the exchange finishes.
Explicit `wallet send` charges keep the earlier resubmission rule: when the
wallet has no record of a pending attempt, recovery resubmits the same invoice,
amount and `max_total_msat` through the normal preflight, fee check and pay
path (providers must treat a repeat `pay` as a lookup of the existing payment,
which Lexe 0.1.24 and later does), and once the invoice has expired with
nothing found the charge is marked `failed` and its reservation released.
The wallet contract distinguishes `NotSubmitted` from `Uncertain` errors.
Definite preflight rejection and terminal failure close unused authorization,
while uncertain sibling charges remain reserved. Startup releases abandoned
approvals only when no charge was ever recorded. Live cancellation atomically
closes pending or approved requests only before charge preparation; periodic
reconciliation does not sweep live approvals. Prepared and uncertain charges
retain their reservations.

An uncertain payment the wallet has recorded is not released just because its
invoice expired: an in-flight HTLC can outlive the invoice. If the wallet remains
unreachable or a recorded payment cannot be conclusively reconciled, its
reservation remains held; the PoC has no force-release command.

Buyer-side recovery (`payment_recovery.rs`) runs every 15 seconds: the engine
reconciles its charges, then returns the accepted admissions still open with
their sellers, and the host sends each seller `Recover { id }` (five seconds
per peer, one unavailable seller never blocks the rest). A `Settlement` is
handed to `settle_final` exactly as in the live exchange, so recovery pays a
final invoice it had not yet paid and finishes the request; `Pending` leaves
it for the next pass. Recovery never reruns inference and never replays
application output. It contacts only the original authenticated peer;
replacing a seller's endpoint identity while retaining its wallet/database is
deliberately unsupported, and unrelated same-model peers are never asked to
settle that debt.

Seller-side recovery is the same reconcile pass: every unpaid deposit and
final invoice is looked up in the wallet, and a payment observed late is
recorded, clearing the strike. Billed state survives provider restart; KV
state does not. A request interrupted before `serve_settle` is abandoned on
the error path and stays abandoned; a crash that skipped even that leaves it
`admitted`, which `serve_recover` reports as `Pending` and which a replayed
`Request` cannot dispatch again. Invoice creation is serialized and reuses
the persisted invoice, so a request never carries two invoices of one kind
and no retry can replace a recorded invoice. Expiry is never treated as
settlement or failure on the receiving side.

Debt and forgiveness: an unpaid final invoice is the only recorded debt. The
peer is refused at three; `mesh-llm wallet blocked` lists every peer with any
unpaid final invoice (request ID, amount, expiry), not only those at the
limit, and `mesh-llm wallet unblock PEER` (the full identifier or a unique
prefix of at least eight characters) marks all of that peer's unpaid invoices
forgiven. Both are ledger-only commands and work while the node is stopped.
Forgiveness is an operator override, not a refund; a forgiven invoice that is
paid later is still recorded as received. An unpaid deposit is an admission
that never dispatched, not debt, and never blocks. Peer identities can be
replaced, so the strike limit is only a PoC deterrent.

## Prices, fees, and token accounting

The seller explicitly configures per-model input/output msat per million
tokens. Free serving is the default. Charges use wide integer arithmetic and
ceiling division to the msat. Bills are recorded to the msat, but every
Lightning payment has a 1 sat floor (`PAYMENT_FLOOR_MSAT`): live Lexe
payments of 60 and 90 msat found no route while 1000 msat and above settled.
The deposit is exactly the floor. The final remainder is invoiced exact to the
msat; only a remainder below the floor is rounded: under 500 msat is free (no
final invoice) and 500 to 999 msat is raised to 1 sat. Both sides derive it from `final_invoice_msat`, so this
rounding is part of the `/mesh/payment/v3` wire contract. The deprecated
`LightningOffer.minimum_invoice_msat` gossip field is sent as 1 and ignored
on read. The seller
refuses admission when the pricing in the `Request` differs from its current
configuration for that model, so a price change takes effect at the next
request a payer makes after it has seen the new advertisement.

Enabling `wallet pricing MODEL` without explicit rates uses 500 input and 1500
output msat per million tokens;
small requests at those rates bill a few msat, which rounds down to free.

The payer reserves a routing-fee allowance for each inference payment of
`max(3000 msat, 1% of the amount)` (`pricing::fee_allowance_msat`). Its local
cap for a request is the effective policy's per-request `max_total_msat`;
there is no prompt estimate and no wallet-balance check before the request.
The seller's rates must fit the policy's rate caps. Nothing is reserved when
the admission is accepted: the daily budget is checked against each payment as
it is prepared (today's settled spend plus payments in flight, fee caps
included). The seller never sees the cap. At settlement the recomputed bill must
not exceed the cap. The wallet is told `amount + fee_allowance` as the cap for
each payment. A wallet that can bound fees must not submit a payment whose
amount plus fees exceeds it; Lexe preflights a route and submits that same
route only when its total debit fits. A wallet that cannot bound fees may
exceed it (see the wallet contract above). Route minimums can increase the
sent amount; that increase also counts against the cap. Actual outgoing amount
and fees are recorded. Operators can choose a different cap for an explicit
`wallet send`.

Billing uses the `prompt_tokens` and `completion_tokens` the backend reports
in its response, at the rates frozen in the admission. Reasoning and cached
token breakdowns are not added to those counters again. Socket acceptance does
not prove receipt by the end application. Settlement continues independently
of the application's HTTP connection, and the provider stops the backend when
it receives `Cancel`. Late deposit payment after the seller gave up is
recorded as received but does not dispatch anything or trigger a refund.

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
it neither requires nor guarantees a paid provider. A request whose cap does
not fit the remaining budget is not authorized. There are no mandatory
per-token price caps or extra setup calls. Seller pricing is independent.

Policy persists across restarts and mesh switches. `wallet policy` without flags
reads it without changing it, returning `mode`, `daily_budget_msat`,
`spent_today_msat`, `in_flight_msat`, and `remaining_daily_budget_msat`. The latter
is the budget allowance, not a guarantee of wallet funds. Reading policy does not
provision a wallet. Applications use the same `policy` command through
`POST /api/wallet`; no second setting is required.

The budget uses UTC calendar days and actual settlement timestamps, including
fees. Payments still in flight count against the budget, including across
midnight. Each payment atomically checks the remaining budget. The wallet balance is not checked up front: a
payment the wallet cannot make fails at payment time and the request is
dropped. Switching to free-only stops new paid
inference, but does not cancel already-submitted settlement. Manual
per-inference approval is not supported. Provider/model approvals and price
browsing are separate future features.

Requests may still include `mesh_payment` to **restrict** the profile for that
request (for example `{"mode":"free_only"}`). This is not an opt-in or a stored
setting: requests cannot enable paid use under free-only policy or expand the
budget. The field is stripped before backend or remote forwarding. Existing
remote-ingress restrictions remain authoritative.

Routing prefers local inference, eligible paid peers, then free peers. Paid peers
are ranked by maximum output cost for the exact model, input rate breaking ties.
Existing capability, health and context checks still apply. Equal-price choices
retain cache/observed-performance preferences. No universal throughput floor is
introduced. Paid providers on the local payee blocklist are dropped before
ranking: a strike is recorded when this node's deposit settled, the provider
then ended the exchange without delivering any output, and the payer did not
cancel. Three strikes within 24 hours block the payee from paid routing for
seven days. The list (`payments/payee_strikes.json`) is never gossiped; an
operator clears a false positive by removing the entry or the file. Free and
local targets are never affected. A seller-side `Error` before admission is
not a strike and lets ordinary routing continue to another candidate.

Only loopback-originated requests can spend. QUIC ingress does not gain wallet
access. Paid routing and management wallet routes both require trusted-local
Host/Origin checks, including for browser simple POSTs with `text/plain` JSON
bodies. Native clients without an Origin header remain supported. Wallet-enabled
nodes refuse the provenance-losing legacy TCP bridge;
normal direct QUIC ingress still supports free inference. A process-owned socket
registry also preserves remote origin for existing legacy connections if wallet
or seller configuration changes while a connection is open. Non-loopback TCP
callers likewise cannot bypass seller charges.

The legacy TCP bridge is the older inbound path for embedders that register a
loopback port instead of installing direct ingress (`tunnel.rs`). On a node
with seller prices, a persisted wallet or a live wallet, requests arriving over
it get HTTP 402 (`inbound_http.rs`, `legacy_bridge_requires_payment_ingress`);
free nodes and builds without the `payments` feature are unchanged. Embedders
that need paid serving must install direct ingress. Wallet API requests can
bind an expected runtime PID and config directory to reject stale destinations.

Pricing gossip is additive (protobuf field 51 and optional JSON metadata). Older
nodes ignore it, while paid providers reject unpaid ordinary remote inference
with HTTP 402. Free nodes retain the existing mesh ALPNs. A released payer that
still speaks `/mesh/payment/v1` or `/v2` receives an `Error` frame in the
frame format it parses and makes no payment. Free inference and gossip
interoperability with released v0.76.1 was verified in both directions before
the metered cutover; see the follow-up results below. This does not certify
every older release.

## CLI and application API

```sh
mesh-llm wallet get-balance
mesh-llm wallet get-transactions --limit 20
mesh-llm wallet fund-wallet
mesh-llm wallet fund-wallet --amount-sats 10000
mesh-llm wallet send lnbc... --max-fee-msat 1000
mesh-llm wallet send lnbc... --amount-msat 10000 --max-fee-msat 1000
mesh-llm wallet pending
mesh-llm wallet blocked
mesh-llm wallet unblock PEER_ID
mesh-llm wallet unpin
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
`blocked`, `unblock` (`peer`), `unpin`, `policy` (optional `value`), `pricing`, and
`set_pricing`
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

Local coverage for the metered protocol lives in `mesh-llm-payments`
(`src/tests/metered_service.rs`, `metered_settlement.rs`, `buying.rs`,
`blocking.rs`, `recovery_boundaries.rs`, `failure_boundaries.rs`) and the
host payment suite (`mesh-llm-host-runtime` `network/payments/tests`,
including real QUIC peers with simulated wallets and backends, the strike
blocklist and usage-filter cases). These cover admission, deposit computation
on both sides, trust and the strike limit, abandoned requests, replayed
requests, recovery of pending/abandoned/billed requests, cancellation,
transport loss during output, and the payee blocklist. They do not prove
mainnet routing or Lexe availability. No mainnet payments have been made with
the metered protocol.

The earlier wallet, provisioning, resubmission, notification, uncertain-HTLC
and unpin coverage (`resubmission.rs`, `payment_notifications.rs`,
`provisioning.rs`, `unpin.rs`, `review_regressions.rs`) is unchanged by the
cutover, as is the mixed-version compatibility harness
(`cargo xtool automation lightning-compatibility`) described below.

### Mainnet results, September 18, 2026

> These results were produced against the previous fixed-allowance protocol
> (input invoice after prefill, output invoice for delivered tokens, manual
> approval mode). The wallet, funding, fee, budget, restart and security
> observations still describe the shipped wallet and ledger; the invoice
> shapes and delivery gating they mention no longer exist.

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
  The error at the time was a misleading capacity-related HTTP 503; the
  metered path answers HTTP 402 with a payment-specific message.
- Restarting the payer preserved policy, invoices and settled records, and a
  subsequent inference succeeded. Restarting the provider preserved its seller
  prices and financial records.
- Killing the payer after its first output token left a durable output debt.
  Recovery after restart settled the original output invoice without a duplicate
  input debit. (Under the metered protocol the same crash abandons the
  request: nothing is billed, and the deposit stays with the provider.)
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
provider invoices were paid at the end. Both isolated wallet directories were
retained, and no wallet credentials or transaction files are included in the
downloadable application bundle.

Lexe's default invoice expiry and recovery from an ambiguous in-flight Lightning
HTLC have not been exercised on mainnet. Simulated tests cover uncertain-send
recovery and concurrent reservation enforcement, including resubmission of an
unrecorded `wallet send`, reconciliation without a second `pay`, the expiry
give-up, and a resubmission racing the first landing
(`mesh-llm-payments` `src/tests/resubmission.rs`). Refunds and proof of
computation remain outside this PoC.

### Follow-up failure and compatibility checks

The following passed without funding new wallets or spending additional
bitcoin. Checks that depended on the fixed-allowance protocol are marked.

- **Mixed versions:** a released v0.76.1 client received real model output from
  this branch's free provider, and HTTP 402 after that provider enabled paid
  serving. This branch's client also received real model output from the released
  provider. Both directions used separate processes, private meshes, isolated
  profiles, and each binary's matching CPU runtime. Pin
  `runtime.native_runtime.selection = "cpu"`; `--device CPU` alone does not
  prevent the released binary from trying to install a GPU runtime at startup.
- **Concurrent authorization:** 16 simultaneous service requests competed for a
  1,000-msat budget with 700-msat per-request reservations. Exactly one reached
  the simulated wallet.
- **Uncertain HTLC:** simulated payment submission lost its response while the
  wallet retained a pending HTLC. Restart, a status-query outage and invoice
  expiry did not release its reservation or submit a second payment. Separate
  success and failure cases reconciled the original hash, including replay
  afterward.
- **Expiry** (fixed-allowance protocol): a signed two-second input invoice
  released the waiting backend with zero decoded tokens and no output invoice.
  Under the metered protocol the equivalent case is an unpaid deposit: the
  seller never dispatches and the request is abandoned.
- **Native resource release** (fixed-allowance protocol): the real SmolLM2
  model prefilled before approval and emitted no tokens when the gate reported
  expiry. The metered path runs nothing on the backend before the deposit
  settles, so there is no gate to release.
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
just with-lld cargo xtool automation lightning-compatibility \
  --current-binary /absolute/current/mesh-bundle/mesh-llm \
  --released-binary /absolute/released/mesh-bundle/mesh-llm \
  --released-dialect serve-client-bind-port \
  --model /absolute/SmolLM2-135M-Instruct-Q8_0.gguf \
  --output /absolute/new-evidence-directory

just with-lld cargo test -p mesh-llm-payments --lib
just with-lld cargo test -p mesh-llm-host-runtime --lib payment
```

The native command writes `results.json` and process logs, stops its own nodes,
and leaves isolated profiles for inspection. The declared released dialect must
match the selected bundle: `serve`, `client`, `--bind-port`, and `--bind-ip`.
The output directory must be new; the default total budget is 600 seconds
(`--timeout-secs` accepts 30 through 3600). It never funds or sends wallet payments.
Its local fixtures cover orchestration; actual product bundles and the model
remain separate mixed-version runtime qualification.

Notification coverage also exercises already-settled payments, settlement during
initial lookup, multiple subscribers, unrelated and duplicate events, successful
and failed outgoing payments, incoming expiry, and cancellation followed by
restart. A paused Tokio clock verifies that event-backed waiters perform no
periodic status queries. The QUIC payment fixtures use event-backed wallets.

The tested model fixture is `unsloth/SmolLM2-135M-Instruct-GGUF` revision
`9e6855bc4be717fca1ef21360a1db4b29d5c559a`, file
`SmolLM2-135M-Instruct-Q8_0.gguf`, SHA256
`c4a3dd037301b6ecea31d6da37f5cd793ead920dd5ddfe6d589294628d6ce66a`.
Use only a native bundle built from the same checkout.

For operator-run mainnet validation of the metered protocol:

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
   Verify a deposit payment and a final payment on the payer and two receipts
   on the provider, and that the bill equals the backend's reported usage at
   the advertised rates.
5. Repeat immediately: the provider should admit the second request without a
   deposit (one final payment only). Restart the provider and repeat once more
   to confirm the deposit returns.
6. Launch concurrent requests that together exceed the budget and verify
   rejected reservations do not spend.
7. Cancel a streaming request after output begins. Confirm generation stops
   and the request is abandoned (no final invoice). Kill the payer during a
   response, restart it, and confirm recovery fetches the settlement and pays
   the final invoice without a second deposit.
8. Verify the strike limit (three unpaid finals refuse the peer; `wallet
   blocked` / `unblock`), free-provider fallback, remote requests never
   spending the relay wallet, and a released peer's free inference/gossip
   interoperability. Record hashes/amounts privately as evidence.

## Future encrypted chunks

Later, the inferencer sends an invoice and N tokens encrypted using a key derived
from that invoice's preimage in the same payload. Payment reveals the preimage,
allowing decryption. Bind ciphertext to the request, sequence and invoice, with a
fresh preimage per independently sold chunk. The ledger already identifies
numbered charge segments; token ranges and the cryptographic construction remain
TODO. Preimage-based access does not prove valid inference or honest counts.

Proof of computation remains an open [TODO](../../crates/mesh-llm/TODO.md).

## Advertised price visibility

`GET /v1/models` includes an additive `payment` object for concrete model IDs:
`free_available`, `paid_available`, `binding_quote: false`, and `offers` keyed by
`provider_id`. Each offer includes `paid`, nullable `pricing` (input/output
msat-per-million rates), and peer last-seen age. The
age describes peer contact, not a guaranteed quote timestamp. Mixed free/paid
providers remain separate offers. Local advertised seller prices describe remote
service; ordinary local inference does not pay itself. Unknown external-plugin
pricing is not inferred from these offers. Admission terms remain authoritative.

## Payer-side evidence hooks

`RequestTerms.exchange_id` optionally retains the host's existing OpenAI evidence
exchange ID in the payer's existing JSON terms record. It is not the private
payment recovery ID. Older records omit it; there is no SQL schema migration.

A trusted local plugin declaring `payment.lifecycle.v1` can observe the active
payer exchange: `terms_accepted`, `input_invoice_issued`,
`input_settlement_observed`, `output_invoice_issued`,
`output_settlement_observed`, and `final_accounted`. Subscribe to
`openai.exchange.v1` as well to obtain the host exchange and join on `exchange_id`.
Events contain `exchange_id`, stable `event_ref`, `terms_digest`, `role` (`payer`
or `provider`), `phase`, `source`, nullable `segment` (0 = deposit, 1 = final),
nullable `payment_hash`, nullable `settlement` (`terminal` for wallet success),
`amount_msat`, and nullable `tokens`. The phase names are kept from the earlier
protocol: `input_*` now describes the deposit and `output_*` the final invoice.
A settlement event also carries the wallet's own numbers when it reported them:
`fee_msat` (payer: the fee the wallet paid on top of `amount_msat`; provider:
the fee the receiving wallet deducted) and, on the provider side,
`credited_msat` (what the receiving wallet reports it credited; Lexe reports
it net of the fee). Both are omitted when the wallet or the `payments.v1`
provider did not report them, and both are covered by `event_ref`.
Terms acceptance is `payer_asserted` (amount is the payer's local cap);
invoice issuance is `provider_asserted` as observed by the payer, not
independently verified issuance. Settlement is `wallet_reported`, amount
excluding fees (the fee is in `fee_msat`). Final accounting is
`payer_asserted` and its amount is the bill the payer verified against the
admission (zero for an abandoned request), not the sum of debits. The deposit
settlement event can arrive after output or even after `final_accounted`,
because the deposit payment runs concurrently with the exchange. Recovery
emits no events.

The terms digest is lowercase SHA-256 over checked JCS JSON using the existing
host `request_body_digest` helper, applied to the object containing exactly
`exchange_id`, `model`, `pricing`, and `deposit_msat`: the
public admission terms. The recovery ID, the payer's cap, the output allowance
and the local peer are excluded. Event refs use the same construction over the
event object with `event_ref` set to `""`. Integers outside the helper's safe
range suppress evidence, not payment.

An eight-event per-exchange queue and one-second publication timeout bound
best-effort delivery. No plugin delivery is awaited by settlement; terms hashing
is skipped without a subscriber. No per-token observation is added. Missing
correlation, disconnects, queue drops, process exits and restart recovery can
leave incomplete evidence; there is no replay or complete audit-log guarantee.
No raw invoice, preimage, wallet transaction ID, prompt or response text is
published. Payment hashes are linkable metadata, and fee amounts can hint at
which wallet a node uses.

## Provider-side evidence hooks

The serving host emits the same `payment.lifecycle.v1` events, with `role:
provider`, at the `payments.v1` seller operations it already calls. Emission is
in the host's serving path, not in the payments engine, so any `payments.v1`
provider gets it unchanged.

| seller operation | phase | source |
|---|---|---|
| `serve_deposit_confirmed`, before the wait | `input_invoice_issued` (segment 0, the deposit) | `provider_asserted` |
| `serve_deposit_confirmed`, after the wait | `input_settlement_observed` (segment 0) | `wallet_reported` |
| `serve_settle` | `delivered` (`tokens` is the backend's `completion_tokens`, `amount_msat` 0) | `provider_asserted` |
| `serve_settle`, bill sent | `final_accounted` (`amount_msat` is the bill) | labeled `payer_asserted` |
| `settle_received`, final | `output_invoice_issued` (segment 1) | `provider_asserted` |
| `settle_received`, final | `output_settlement_observed` (segment 1) | `wallet_reported` |

Observation starts at admission, so a request admitted without a deposit emits
nothing until `delivered`, and a request that fails before admission (an
invalid request, a `Blocked` buyer, no backend for the model: `Frame::Error`,
nothing billed) emits no phases; treat that as never priced. The serving path
does not emit `terms_accepted` for the provider. `delivered` is emitted once,
when `serve_settle` succeeds, whatever happened to the transport; an abandoned
request emits no `delivered` or `final_accounted`. Settlement is the receiving
wallet's report that `settle_received` recorded; `amount_msat` is the invoice
amount, and `credited_msat` / `fee_msat` are what the receiving wallet reported
(`settle_received` answers them; a provider that answers `{}` leaves them out).
Since the payments engine is a plugin, nothing here claims that payment
preceded wallet access: `source` says who asserted, which is all it can say.

The request ID is the private recovery ID, so the serving host names each paid
serving request with a fresh `exchange_id` of its own, minted once when a plugin
subscribes to either channel. Its `payment.lifecycle.v1` events and the paid
serving path's `openai.exchange.v1` events both carry it, so a plugin subscribed
to both joins the provider's two channels on `exchange_id`, as on the payer side.
`terms_digest` uses the payer's construction over the same admission terms with
that ID, so the two sides' digests differ only by `exchange_id`. Join the
payer's and the provider's streams on `payment_hash`, which both sides report
for each invoice and settlement.

The payer section's bounds apply unchanged: an eight-event queue per request,
a one-second publication timeout, nothing awaited by serving or settlement, no
per-token events, and no ID or hashing without a subscriber. Evidence is
best-effort and can be incomplete. Unpaid finals, operator unblocking and
recovery emit no events.

### Evidence interpretation and acceptance expectations

Neither payment nor matching payer/provider records prove correct inference or
honest token counts. Wallet settlement is an observation reported by the
configured wallet, not independent verification of inference. The usage a
provider bills is what its backend reported, not what the payer measured.

Use synthetic identities and payment hashes in committed fixtures. These are
acceptance expectations, not a claim that every fixture is implemented:

| Scenario | What the evidence should distinguish |
| --- | --- |
| Paid request | Payer and provider observations correlate by payment hash; invoice claims remain distinct from wallet settlement reports. |
| Deposit-free request | Only segment 1 events exist; a trusted peer paid nothing before output. |
| Abandoned request | A deposit may have settled with no `delivered` or `final_accounted`; nothing is owed. |
| Exhausted automatic budget | Paid request is rejected with HTTP 402, not represented as a successful paid exchange. |
| Free-only policy against a paid-only route | HTTP 402 policy refusal, not a wallet outage or settlement failure. |
| Restart with a pending payment | Reconcile the existing payment without a second debit; successful terminal reconciliation releases its reservation. |
| Harness or observation failure | Preserve the failed attempt and its limitation separately from any corrected rerun; do not silently turn it into a pass. |

Uncertain payments must remain distinguishable from terminal outcomes. Do not
discard unresolved charges merely to report zero outstanding reservations.

### Failure and recovery boundaries

A transport failure, timeout or seller `Error` frame while awaiting the
`Admission` returns to ordinary mesh routing before any payment-capable task is
spawned: nothing has been authorized and the seller cannot have charged this
buyer. An admission that does not restate the request, the frozen price or the
single defined deposit, a malformed invoice, and policy refusal are terminal
for the HTTP request (`cancel`, stage `unstarted`, releases the prefetched
balance). Client disconnection during foreground setup drops that setup
without spawning a payer. Once the admission is accepted and `Accepted` has
been sent, failures are terminal even if the deposit is still pending: the
application receives HTTP 402 (`paid inference interrupted; settlement remains
recoverable`), no fallback may create another bill, and the detached exchange
or the 15-second recovery pass completes settlement from durable state.

Application disconnect after `Accepted` sends `Cancel`; the seller stops the
backend and abandons the request unless a complete usage report was already
observed. A seller that closes the exchange having sent no output after the
deposit settled is a strike on the buyer's local blocklist. On the seller, any
exit from the admitted exchange without a settlement abandons the request
(idempotent), and the buyer is told `metered exchange interrupted; recover by
request ID` rather than the underlying error.
