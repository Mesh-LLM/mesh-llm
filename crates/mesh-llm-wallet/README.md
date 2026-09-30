# mesh-llm-wallet

Provider-neutral Lightning wallet abstraction for mesh-llm.

- `provider::WalletProvider` — the trait the payment ledger drives.
- `invoice::Invoice` — BOLT11 parsing/validation at trust boundaries.
- `contract` — the versioned `wallet.v1` plugin capability: operation names and JSON shapes.
- `backend::WalletBackend` + `plugin_server` (feature `plugin-server`) — implement one trait, get a mesh plugin.

No wallet SDK is linked here. Concrete wallets are plugin processes. Default
Mesh builds retain the payment infrastructure but require an external `wallet.v1` provider for wallet
operations. The optional built-in `crates/mesh-wallet-lexe` is compiled out by
default; when explicitly enabled, it is served from the mesh-llm executable as
`--plugin wallet-lexe`. Wallet availability and spending authorization are
separate from compiling payment infrastructure.

The host owns invoice lifetime (`create_invoice(amount, expiry_secs)`) and fee caps
(`pay(.., max_total_msat)`); a plugin never substitutes provider defaults for either.
Receiver-side arrival is the normalized `Transaction.claiming` flag; `status_msg` is display only.
