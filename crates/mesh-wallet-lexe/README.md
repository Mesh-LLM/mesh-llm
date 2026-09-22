# mesh-wallet-lexe

The Lexe Lightning wallet as a mesh-llm plugin. Serves the `wallet.v1` capability
(see `crates/mesh-llm-wallet`). The host resolves it by capability, never by name;
bundled next to `mesh-llm` it is auto-registered as the optional plugin `wallet-lexe`.

Disable at runtime with:

```toml
[[plugin]]
name = "wallet-lexe"
enabled = false
```

Wallet state lives under `<config-dir>/payments/lexe/`, the same layout the
in-process implementation used, so existing wallets keep working.
