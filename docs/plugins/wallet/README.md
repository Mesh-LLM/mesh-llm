# Wallet engineering notes

> [!WARNING]
> Work in progress: Lightning payments, paid inference, and wallet integrations
> (including Lexe) are experimental features still being explored. These notes
> describe ongoing development, not production-ready features or stable contracts.

Design decisions and evidence expectations for wallet plugins. These are
maintainer notes, not a wallet setup guide or a record of private test systems.

- [Wallet boundaries and evidence](DECISIONS_AND_EVIDENCE.md) — why settlement
  observations have explicit provenance, what hash lookup provides, and the
  scenarios evidence fixtures should cover.

The [Lightning payments specification](../../specs/lightning-payments.md) owns
the payment behavior and current limitations. The
[wallet crate](../../../crates/mesh-llm-wallet/README.md) owns the provider-neutral
API; these notes explain the reasoning without replacing that contract.
