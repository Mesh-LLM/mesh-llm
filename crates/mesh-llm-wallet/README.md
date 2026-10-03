# mesh-llm-wallet

Provider-neutral Lightning wallet abstraction for mesh-llm.

- `provider::WalletProvider` — the trait the payment ledger drives.
- `invoice::Invoice` — BOLT11 parsing/validation at trust boundaries.
- `contract` — the versioned `wallet.v1` plugin capability: operation names and JSON shapes.
- `backend::WalletBackend` + `plugin_server` (feature `plugin-server`) — implement one trait, get a mesh plugin.

No wallet SDK is linked here. Concrete wallets are plugin processes. Default
Mesh builds retain the payment infrastructure but require an external `wallet.v1` provider for wallet
operations, such as the external `lexe-wallet` plugin. Wallet availability and spending authorization are
separate from compiling payment infrastructure.

The host owns invoice lifetime (`create_invoice(amount, expiry_secs)`) and fee caps
(`pay(.., max_total_msat)`); a plugin never substitutes provider defaults for either.
Receiver-side arrival is the normalized `Transaction.claiming` flag; `status_msg` is display only.
