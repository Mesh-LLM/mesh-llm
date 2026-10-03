# mesh-llm-payments-types

Pure data types shared by mesh core and the payment engine: integer-msat
pricing (`pricing`), invoice lifetimes (`lifetimes`), the peer payment wire
format (`wire`) and `RequestTerms`. No storage, ledger or wallet SDK — only
serde and the wallet type crate — so mesh core can name payment types without
linking the engine. `mesh-llm-payments` re-exports these under their old paths.

See [Lightning payments](../../docs/specs/lightning-payments.md).
