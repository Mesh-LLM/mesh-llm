# mesh-llm-pay-lightning

Standalone Lightning payments utility and its admission/settlement engine. Mesh
owns generic authenticated transport, admission hooks and authoritative terminal
usage; this utility owns pricing, policy, durable ledger, tickets, invoices and
reconciliation. A separately installed `wallet.v1` plugin owns provider-specific
wallet access. The host does not link this crate, the ledger, payment types or
wallet abstraction. Payment operations are not MCP tools.

Both peers install `pay-lightning`, communicating on `pay.ln.v1`. Their wallet
plugins may have different names; existing compatible wallet binaries need not
be rebuilt. `run(name, version, factory, config)` remains for in-process fixtures;
the installed executable uses `run_bound` and the host-owned wallet connection.

## Explicit operator binding

Install and enable the wallet and utility separately. The host passes the three
declared utility settings in `MESH_LLM_PLUGIN_SETTINGS_JSON`. Example host config:

```toml
[[plugin]]
name = "pay-lightning"
settings = { config_path = "/absolute/payment/config.json", wallet_binding = "wallet", allow_wallet_provisioning = false }
allow_peer_blocks = true

[[plugin.operation_bindings]]
name = "wallet"
provider = "lexe-wallet"
profile = "default"
setup_operation = "wallet_open"
setup_input_json = '{"directory":"/absolute/existing/wallet-state"}'
operations = ["wallet_balance", "wallet_transactions", "wallet_create_invoice", "wallet_pay", "wallet_lookup", "wallet_wait_for_payment", "wallet_wait_for_arrival"]
```

The host also needs the explicit required exchange grant described below. The
binding exclusively selects one provider and fixes the open payload. The utility
cannot choose a wallet executable or supply an open directory. The host alone
launches and supervises the wallet; killing the utility does not kill that wallet.

`config_path` points to a separate payment JSON file:

```json
{
  "data_dir": "/absolute/payment/ledger",
  "prices": {}, "buy_prices": {}, "buy_peers": [], "cap_msat": 0,
  "policy": {"mode": "free_only"}
}
```

The utility pins `wallet_id`, `provider` and `network` in the ledger directory
before starting its engine. A changed identity or non-mainnet wallet is refused.
The pin's historical `plugin` field records the **provider binding name**. First
adoption without a ledger pin requires `allow_wallet_provisioning = true`, before
any setup is attempted; this also applies when adopting an already provisioned
wallet into a fresh ledger. Installing a package or inspecting `--manifest`
never opens a wallet. Do not copy or replace funded wallet state to change a pin.

Setup tolerates provider-startup ordering by retrying only requests that the host
reports were not dispatched, for a bounded 90-second window. A provider error or
uncertain setup fails closed. Repeated successful setup must be idempotent. A
provider restart invalidates the utility's handle; the utility then re-runs
setup once (re-checking the pinned wallet identity) and retries the refused
operation, including `wallet_pay`, because the host refused it before anything
reached the provider. A second refusal fails closed.

`prices` maps model IDs to `input_msat_per_million`, `output_msat_per_million` and
`cache_discount_percent`. `buy_prices` uses the same shape for authorized ceilings;
`buy_peers` contains exact authenticated peer IDs. Buying requires positive
`cap_msat` and explicit automatic policy with `daily_budget_msat`. The budget is
daily, not a per-request cap. Defaults never authorize spending. Optional absolute
`timing_log` and `error_log` paths belong inside isolated process-owned state.

The host must grant required metadata/admission access to `request_received`,
`backend_selected`, `exchange_finished`, header
`x-plugin-7061792d6c696768746e696e67-pay-terms`, a 30000 ms deadline and peer block
permission. See [exchange contract](../../docs/plugins/openai-exchange-lifecycle.md).

## Admission, settlement and lifecycle

Cold clients pay a deposit. Warm tickets refill concurrently with inference,
without waiting for the final wallet payment. The three-unpaid-final limit still
bounds credit. An independent receipt sweep wakes admission even when a buyer's
`Paid` message is delayed. Persisted invoice receipts reconcile after restart.
Background receipt observation reserves at most eight invoices per sweep, with
one-second lookup timeouts and per-invoice exponential backoff from 250 ms to
60 seconds. Attempts are reserved before I/O so cancellation cannot hot-loop the
same first invoice. Expired unpaid invoices remain debts and continue low-rate
observation; expiry is never payment or forgiveness. This bounds each sweep's
wallet I/O, not the ledger scan or total scheduler memory.

Uncertain payments are reconciled by hash, never blindly replayed by the adapter.
A request known not to have been dispatched is classified as not submitted;
transport ambiguity remains uncertain. Long-poll observations can be reissued
after the host's binding deadline; that does not retry a payment. One outstanding
ticket per buyer/model is supported; restart abandons interrupted work and drops
unconsumed tickets rather than replaying inference.

## Offline development and evidence boundaries

Run `just check`, `just test`, `just build`, `just package` here. `package` stages
the separate utility; `package-mock` stages the combined offline fixture. Raise
the macOS file limit with `ulimit -n 8192` before the full suite. The opt-in
`offline-mock` binary uses signed synthetic invoices and an explicit shared file
network. It never contacts wallet services or moves funds. Its
`MESH_LLM_OFFLINE_WALLET_CHILD` mode serves only `wallet.v1` for legacy subprocess
adapter tests; the installed utility does not use that child adapter, and the
`wallet_child` module is compiled only with that feature.

Tests cover synthetic payment settlement/recovery and legacy child ownership,
plus the installed utility's host RPC setup, startup delay, SIGKILL and identity
pin reuse. Synthetic children are one backend under different names, not two
independently implemented real providers. These tests do not establish live
QUIC/inference performance, automatic install/binding, wallet setup UX, migration,
removal UX or cross-platform installed behavior. Run `just clean` after validation.
