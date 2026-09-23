# mesh-wallet-lexe

The Lexe Lightning wallet as a built-in mesh-llm plugin. Serves the `wallet.v1`
capability (see `crates/mesh-llm-wallet`). Like `blobstore`, it runs as a
separate process that the host launches from its own executable:
`mesh-llm --plugin wallet-lexe`. No second binary ships.

The host resolves the wallet by capability, never by name, so an external
`wallet.v1` plugin can replace it. Disable the built-in at runtime with:

```toml
[[plugin]]
name = "wallet-lexe"
enabled = false
```

This is the only crate that links the Lexe SDK. It is compiled into `mesh-llm`
through the host-runtime `wallet-lexe` cargo feature (default on for the shipped
binary). SDK consumers (`mesh-llm-sdk` with `serving` or `serving,payments`) do
not enable it and never compile Lexe.

Wallet state lives under `<config-dir>/payments/lexe/`, the same layout the
in-process implementation used, so existing wallets keep working.
