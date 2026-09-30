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
through the host-runtime `wallet-lexe` Cargo feature, which is **off by default**,
including release builds. Opt in with `MESH_LLM_WALLET_LEXE=1 just build` or
`MESH_LLM_WALLET_LEXE=1 just release-build`. The default host retains payment
infrastructure and can use an installed external `wallet.v1` plugin; that does
not itself enable spending. SDK consumers (`mesh-llm-sdk` with `serving` or
`serving,payments`) do not enable it unless they explicitly opt into the
host-runtime wallet feature.

Wallet state lives under `<config-dir>/payments/lexe/`, the same layout the
in-process implementation used. A build without a wallet provider cannot use
that wallet until a compatible provider is available. Host pins also bind the
plugin name: switching to a differently named external plugin requires explicit
identity-checked adoption, not deletion of the existing pin or wallet state.
