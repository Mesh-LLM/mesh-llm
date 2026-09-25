# Wallet engineering notes

Design decisions and evidence expectations for wallet plugins. These are
maintainer notes, not a wallet setup guide or a record of private test systems.

- [Wallet boundaries and evidence](DECISIONS_AND_EVIDENCE.md) — why settlement
  observations have explicit provenance, what hash lookup provides, and the
  scenarios evidence fixtures should cover.

The [Lightning payments specification](../../specs/lightning-payments.md) owns
the payment behavior and current limitations. The
[wallet crate](../../../crates/mesh-llm-wallet/README.md) owns the provider-neutral
API; these notes explain the reasoning without replacing that contract.
