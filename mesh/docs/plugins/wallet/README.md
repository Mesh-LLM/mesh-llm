# Wallet engineering notes

> [!WARNING]
> Wallet implementations, including the Lexe wallet, are currently for example
> purposes only and are still being explored.

These maintainer references describe wallet boundaries and evidence expectations,
not a record of private test systems.

For operators, start with [external wallet setup and existing-state adoption](SETUP.md):
installation, profile-relative storage, funding, spending policy and sending.

- [Lightning payments specification](../../specs/lightning-payments.md) — payment
  behavior and current limitations, including
  [evidence expectations](../../specs/lightning-payments.md#evidence-interpretation-and-acceptance-expectations).
- [Wallet crate](../../../crates/mesh-llm-wallet/README.md) — provider-neutral API.
