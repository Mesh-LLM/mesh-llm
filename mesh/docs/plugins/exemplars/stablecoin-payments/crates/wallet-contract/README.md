# Stablecoin wallet contract

Versioned `wallet.stablecoin.v1` operations consumed by `pay-stablecoin` and
implemented by `wallet-ousd-tempo`. This crate owns public operation types,
amount serialization and immutable obligation memo construction. It owns no
private keys, transport, RPC access, pricing policy or runtime process.
