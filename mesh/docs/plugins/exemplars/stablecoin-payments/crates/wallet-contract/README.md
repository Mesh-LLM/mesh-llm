# Stablecoin wallet contract

Versioned `wallet.stablecoin.v1` operations consumed by `pay-stablecoin` and
implemented by concrete stablecoin wallet providers. This crate owns public operation types,
amount serialization and immutable obligation memo construction. It owns no
private keys, transport, RPC access, pricing policy or runtime process.

`WalletInfo.fee_asset` optionally declares a fee currency distinct from the
payment asset. Absent means the payment asset, preserving older OUSD profiles
and wire data. `PayRequest.max_fee` uses that fee currency; `max_total` caps
payment-asset debit and includes fees only when they use the same asset. Providers
define the meaning of a zero fee cap. The immutable obligation/memo format is
unchanged by this wallet metadata extension.
