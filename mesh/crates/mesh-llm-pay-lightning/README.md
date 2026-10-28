# mesh-llm-pay-lightning

Standalone Lightning payments utility and its admission/settlement engine. Mesh
owns generic authenticated transport, admission hooks and authoritative terminal
usage; this utility owns pricing, policy, durable ledger, tickets, invoices and
reconciliation. A selected `wallet.v1` child owns provider-specific wallet access.
No payment operations are exposed as MCP tools. The host does not link this
crate, the ledger, payment types or wallet abstraction.

The common installed utility identity is `pay-lightning` on both peers and its
peer channel is `pay.ln.v1`. Wallet identities are local only: a `wallet-alpha`
child and a `wallet-beta` child can use the same payment protocol. Compatible
wallet binaries need not be rebuilt for payment engine or protocol fixes.
The library `run(name, version, factory, config)` remains for comparison/fixtures;
it is not necessary for independent wallet adapters to link the payment engine.

## Explicit spike binding (not finished installation UX)

On Unix, `pay-lightning` reads `MESH_LLM_PLUGIN_CONFIG`. The current binding is
explicit operator configuration, **not dependency installation or store resolution**:

```json
{
  "payment": {
    "data_dir": "/absolute/isolated/payment-state",
    "prices": {}, "buy_prices": {}, "buy_peers": [], "cap_msat": 0,
    "policy": {"mode": "free_only"}
  },
  "wallet_plugin": "lexe-wallet",
  "wallet_executable": "/absolute/approved/wallet-executable",
  "wallet_directory": "/absolute/isolated/wallet-state",
  "allow_wallet_provisioning": false
}
```

First open requires explicit `allow_wallet_provisioning: true` if there is no
wallet pin. Installing a package or inspecting `--manifest` never opens a wallet.
Do not enable this against an existing funded host profile; no migration or state
adoption is implemented. A pin is persisted in both ledger and wallet state, so
changing provider, wallet identity or wallet directory cannot silently orphan
payment obligations. A profile lock rejects concurrent child owners using this
utility; this is not a lock honored by unrelated wallet launchers.

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
observation; expiry is never treated as payment or forgiveness. This bounds each
sweep's wallet I/O, not the ledger scan or total scheduler memory.
Uncertain submissions are observed by hash, never blindly retried. One outstanding
ticket per buyer/model is supported; restart abandons interrupted work and drops
unconsumed tickets rather than replaying inference.

The utility owns the wallet child and multiplexes wallet operations over one IPC
connection. Normal drop kills the child; an abrupt utility exit closes IPC and a
wallet using `PluginRuntime` exits on connection loss. Tests exercise SIGKILL and
restart with a synthetic wallet. Arbitrary third-party executables that ignore
IPC closure are not proven orphan-safe; installed ownership/supervision and
Windows transport remain open work. Child failure is fail-closed, not transparent
reconnect/replay. No existing wallet is auto-launched as a second owner.

## Offline development and evidence boundaries

Run `just check`, `just test`, `just build`, `just package` here. `package` stages
the separate utility; `package-mock` separately stages the combined offline
fixture. The opt-in `offline-mock` binary uses signed synthetic invoices and a
shared file network selected explicitly by `MESH_LLM_OFFLINE_NETWORK_DIR`. It never
contacts wallet services or moves funds. Its `MESH_LLM_OFFLINE_WALLET_CHILD` mode
serves only `wallet.v1`, for subprocess integration tests.

Tests cover two distinctly named wallet children through real IPC, cold deposit
and final settlement; delayed-Paid admission recovery; receipt recovery after
restart; identity pins; ownership exclusion; uncertain-send disconnect; and
utility SIGKILL/child exit/restart. The two wallet children are the **same synthetic
backend under different names**, not two independently implemented real wallet
providers. These tests do not establish QUIC/live inference performance parity,
released-main paid compatibility, automatic install/binding, Windows support,
wallet setup/controls, migration or removal UX. Run `just clean` after verification.
