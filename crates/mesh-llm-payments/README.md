# mesh-llm-payments

Integer-msat pricing, durable spending authorization, and two-payment inference
settlement over a provider-neutral wallet. The wallet trait, BOLT11 validation
and the `wallet.v1` plugin contract live in `mesh-llm-wallet`; the shipped
Lexe implementation is the `mesh-wallet-lexe` crate, served as the built-in
`mesh-llm --plugin wallet-lexe` process. This crate links no wallet SDK and owns
invoice lifetimes (`lifetimes`) and fee policy (`pricing`). Applications use the
local management API.

`WalletProvider::wait_for_payment` awaits an authoritative terminal payment
update. Adapters may use native events or poll behind this interface.
Subscriptions must also observe already-settled payments and close the
subscribe/lookup race. Cancelling observation never cancels the payment or
releases an uncertain reservation. Durable recovery continues independently.

See [Lightning payments](../../docs/specs/lightning-payments.md) for the protocol,
CLI, security boundaries, recovery behavior, and PoC limitations.
