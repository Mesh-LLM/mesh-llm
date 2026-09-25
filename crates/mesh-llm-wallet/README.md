# mesh-llm-wallet

Provider-neutral Lightning wallet abstraction for mesh-llm.

- `provider::WalletProvider` — the trait the payment ledger drives.
- `invoice::Invoice` — BOLT11 parsing/validation at trust boundaries.
- `contract` — the versioned `wallet.v1` plugin capability: operation names and JSON shapes.
- `backend::WalletBackend` + `plugin_server` (feature `plugin-server`) — implement one trait, get a mesh plugin.

No wallet SDK is linked here. Concrete wallets are plugin processes; the shipped one is
`crates/mesh-wallet-lexe`, served from the mesh-llm executable as `--plugin wallet-lexe`.

The host owns invoice lifetime (`create_invoice(amount, expiry_secs)`) and fee caps
(`pay(.., max_total_msat)`); a plugin never substitutes provider defaults for either.
Receiver-side arrival is the normalized `Transaction.claiming` flag; `status_msg` is display only.
