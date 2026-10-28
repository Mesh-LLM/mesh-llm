# mesh-llm-pay-lightning

Reusable Lightning admission and settlement runtime linked by external wallet plugins.
The host does not link this crate, the ledger, payment types or wallet abstraction.
One plugin process owns policy, durable ledger, tickets and its in-process wallet;
there is no wallet subprocess. No payment operations are exposed as MCP tools.

Wallet authors implement `mesh_llm_wallet::provisioning::WalletFactory` and call
`run(name, version, factory, config).await`. `manifest_json(name)` is side-effect
free and must be handled before loading config. Use the same installed identity
`pay-lightning` on both peers; the protocol channel remains `pay.ln.v1`.
The external Lexe build is distributed by the catalog under that identity.

## Configuration

Supply `MESH_LLM_PLUGIN_CONFIG` through the generic plugin environment map, or set
`MESH_LLM_PLUGIN_DATA_DIR` to a directory containing `config.json`. These are
explicit operator values, not automatically supplied host variables. JSON fields:

```json
{"data_dir":"/absolute/isolated/payment-state","prices":{},"buy_prices":{},"buy_peers":[],"cap_msat":0,"policy":{"mode":"free_only"}}
```

`prices` maps model IDs to `input_msat_per_million`, `output_msat_per_million` and
`cache_discount_percent`. `buy_prices` uses the same shape for authorized ceilings;
`buy_peers` contains exact authenticated peer IDs. Buying requires positive
`cap_msat` and explicit policy (e.g. `mode: automatic`, `daily_budget_msat`). The
budget is daily, not a per-request cap. Defaults never authorize spending.
Optional `timing_log` and `error_log` paths must be absolute, process-owned files.
Timing records contain no payment payload; error logs are operational diagnostics.
No legacy environment policy or host wallet directory is adopted.

The host configuration must grant required metadata/admission access to
`request_received`, `backend_selected`, `exchange_finished` and header
`x-plugin-7061792d6c696768746e696e67-pay-terms`, with a 30000 ms deadline and peer
block permission. See [exchange contract](../../docs/plugins/openai-exchange-lifecycle.md).

## Offline development

Run `just check`, `just test`, `just build`, `just package` here. The opt-in
`offline-mock` executable uses signed synthetic invoices and a shared file network
selected explicitly by `MESH_LLM_OFFLINE_NETWORK_DIR`. It never contacts wallet
services or moves funds. `just package` copies it to `dist/pay-lightning` and writes
`plugin-manifest.json`; it is a test fixture, not a real wallet release.

Admission creation is serialized; tickets are single-use and owner-bound. Wallet
settlement/recovery work is detached from the ordered handler on both roles.
Discovery/reconnect reconciles once rather than sending hello/recover per request.
Uncertain submissions are observed by hash, never blindly retried. Restart drops
unconsumed tickets and abandons interrupted metered work. One outstanding ticket
per buyer/model is supported; concurrent throughput, periodic recovery and live
money validation remain separate work. Live provisioning requires explicit
permission and fresh isolated state. Run `just clean` after verification.
