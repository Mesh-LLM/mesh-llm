# mesh-llm-wallet

Provider-neutral Lightning wallet abstraction for mesh-llm.

- `provider::WalletProvider` — the trait the payment ledger drives.
- `invoice::Invoice` — BOLT11 parsing/validation at trust boundaries.
- `contract` — the versioned `wallet.v1` plugin capability: operation names and JSON shapes.
- `backend::WalletBackend` + `plugin_server` (feature `plugin-server`) — implement one trait, get a mesh plugin.

No wallet SDK is linked here. Concrete wallets are plugin executables; see `crates/mesh-wallet-lexe`.
