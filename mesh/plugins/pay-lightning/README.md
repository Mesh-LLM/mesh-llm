# Lightning payment plugin spike

`mesh-pay-lightning` is an **out-of-process** `openai_exchange.v1` admission hook and authenticated `pay.ln.v1` channel consumer. It owns the ledger, prices, purchasing policy, tickets, wallet child, and settlement. Core only sees hook verdicts, HTTP 402, channel messages, and identities. There are no payment MCP tools or promoted host payment APIs.

The preserved libraries live in [`lib/`](lib/): [`mesh-llm-payments`](lib/mesh-llm-payments/README.md), [`mesh-llm-payments-types`](lib/mesh-llm-payments-types/README.md), and [`mesh-llm-wallet`](lib/mesh-llm-wallet/README.md). The plugin reuses their deposit/final/recovery, strike and policy logic; it does not implement a second balance or pricing algorithm.

## Build and offline verification

From this directory:

```sh
just check
just test
just build
just package
```

`just test` runs this package and the three preserved library packages in full. Tests use signed **offline fixture invoices** and mock wallets; they do not load Lexe, provision wallets, contact payment services, access Keychain, or move funds. One ignored unit-test entrypoint is invoked automatically as a mock subprocess by its parent test (not a skipped live test).

`just package` creates a local native plugin directory under `dist/`. The runtime manifest carries the channel and PEER_UP/PEER_DOWN subscriptions; `manifest.json` is the permission declaration snapshot only. Configure the native executable directly for this spike. No release publication or automatic installation is provided.

## Operator configuration

Use the same plugin name on both ends: `pay-lightning`. The required metadata/admission grant must exactly cover the declaration. Example Mesh config (replace the executable path):

```toml
[[plugin]]
name = "pay-lightning"
command = "/absolute/path/to/mesh-pay-lightning"
enabled = true
allow_peer_blocks = true

[plugin.openai_exchange_grant]
endpoints = ["chat_completions", "completions", "responses"]
phases = ["request_received", "backend_selected", "exchange_finished"]
headers = ["x-plugin-7061792d6c696768746e696e67-pay-terms"]
admission = true
metadata = true
deadline_ms = 30000
max_body_bytes = 1048576
max_queue_bytes = 4194304
max_in_flight = 32
failure_policy = "required"
```

Environment inherited by the plugin (all paths absolute):

| Variable | Meaning |
| --- | --- |
| `PAY_LIGHTNING_DATA` | Dedicated plugin-owned ledger/ticket directory; never the old host payments directory |
| `PAY_LIGHTNING_WALLET` | Absolute native `lexe-wallet` child executable; opened at plugin startup after the host handshake, for both roles (`--manifest` never opens it) |
| `PAY_LIGHTNING_PRICES` | Seller JSON map model → `{input_msat_per_million,output_msat_per_million,cache_discount_percent}`; default `{}` |
| `PAY_LIGHTNING_BUY_PRICES` | Buyer JSON map of authorized models and maximum input/output rates; discount is the minimum acceptable discount; default `{}` |
| `PAY_LIGHTNING_BUY_PEERS` | Comma-separated exact authenticated seller endpoint IDs, explicitly authorized for automatic preflight deposits; default none |
| `PAY_LIGHTNING_CAP_MSAT` | Positive buyer admission authorization value; default 0 disables buying. The inherited engine enforces its **daily** policy, not a hard per-request cap |
| `PAY_LIGHTNING_POLICY` | JSON policy, e.g. `{"mode":"automatic","daily_budget_msat":100000}`; default free-only |
| `PAY_LIGHTNING_ERROR_LOG` | Optional dedicated plugin operational-error JSONL file |

The wallet state lives at `$PAY_LIGHTNING_DATA/wallet`. Its wallet identity is durably pinned and mainnet is required by the preserved contract. A corrupt pin fails before launching the child; a replacement wallet identity fails before any send. Child IPC is multiplexed so an incoming settlement wait does not block another payment. A disconnect during `wallet_pay` is **uncertain**, never automatically retried; reconciliation observes by hash. Child process lifetime belongs to the plugin and it is killed when dropped.

**Live funding/provisioning is separate authorization and validation work.** These commands do not copy/adopt existing wallet state. Opening a real unprovisioned Lexe child may provision a wallet. Do not run live on unattended macOS unless its credential path has been verified noninteractive. Do not change or copy existing wallet state to make the spike run.

## Actual protocol and experience

1. Buyer PEER_UP sends `quote` only to an explicitly allowed seller. Before quoting it reconciles uncertain local charges and sends `recover{id}` for outstanding buyer ledger requests.
2. Seller `quote` pre-creates **one single-use admission ticket per buyer/model** through the existing `serve_admit`. Cold ticket carries a deposit invoice; warm ticket omits it. The original library deposit is per-admission, not transferable credit and not a refund promise.
3. Buyer verifies seller identity, authorized model/rate ceiling, deposit/minimum limits, and daily policy through `accept_admission`; pays through the preserved `pay_deposit`; sends `paid{id}`. Seller verifies receipt against its wallet, then makes that ticket ready. A claimed proof alone is not credit.
4. Seller `request_received` at `serving_host_ingress` accepts only a ready ticket for the QUIC-authenticated `requested_by_node_id` and model. It durably, atomically consumes the ticket and binds it to the host `exchange_id`. It never waits for a wallet. No ticket → deny **402** + namespaced terms. Gateway/local-user and unpriced model events abstain.
5. Seller terminal callback schedules settlement outside the hook deadline. Reported complete usage is billed at admission rates via `serve_settle`; absent/malformed usage → `serve_abandon`, no bill/strike. Cached usage is normalized from either supported host shape. The settlement travels over the authenticated channel.
6. Buyer `settle_final` recomputes the bill/deposit/remainder and pins invoice payee before paying. Seller refreshes wallet facts, establishes warm trust, and preflights the next ticket. A buyer recovery probe returns stored Pending/Abandoned/Billed only; it cannot dispatch backend work.

### Deliberate spike limitations

- **Preflight must finish before inference.** Requests that arrive too soon get 402; host negative cache can delay reuse for five minutes (sole-candidate fallback still probes). No cache-clear or routing-verdict seam is claimed. Terms on peer 402 are not surfaced as gateway event headers; the channel avoids this dependency.
- One outstanding ticket per buyer/model; parallel requests need a future bounded ticket pool. Payer authorizes one preflight deposit on peer discovery, even if it never makes a request. This is why exact peer and model/rate allowlists are mandatory. Neither a general public-mesh spending policy nor transparent inline deposit is claimed.
- Seller restart abandons admitted ledger work according to the existing engine; unconsumed tickets are discarded, consumed correlations persist for audit. A deposit on work abandoned at crash stays received, as in the preserved engine.
- Recovery runs on PEER_UP. There is no general periodic reconnect/retry coordinator yet; malformed/failed messages need operator inspection, not blind money-moving retries.
- This spike has no plugin UI/config schema, generic wallet operations, portable Windows wallet-child transport, release archive, or migration. It is not production ready.
- Unit/channel tests and a real **mock** wallet subprocess validate C. Real Lexe money movement, host-installed paid inference, two-node cooldown/failover, and crash/disconnect timing remain slice D validation.

### Opt-in latency diagnostics

Set `PAY_LIGHTNING_TIMING_LOG` to an absolute, plugin-owned JSONL file path
on each role. Records contain wall-clock microseconds, process id, phase,
operation, begin/end boundary and monotonic elapsed microseconds. They cover
channel send calls, channel handler execution and wallet operation/startup calls.
No invoice, payment hash, peer identity, response or error payload is recorded.
An unset variable produces no output; write failures do not change payments.
Handler begin is **not** transport receipt: time queued in the host or plugin
runtime precedes it. Send end acknowledges the local send API, not remote receipt.
End records also occur on error/cancellation and do not imply success. These
boundaries narrow transport-versus-handler delays but do not measure wire transit
or identify a specific exchange when same-kind operations overlap.
