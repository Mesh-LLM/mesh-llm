# mesh-llm-payments

Provider-neutral Lightning wallet operations, BOLT11 validation, integer-msat
pricing, durable spending authorization, and two-payment inference settlement.
The default private adapter uses Lexe on Bitcoin mainnet. Public contracts expose
only this crate's wallet types; applications use the local management API.

`WalletProvider::wait_for_payment` awaits an authoritative terminal payment
update. Adapters may use native events; Lexe polls behind this interface.
Subscriptions must also observe already-settled payments and close the
subscribe/lookup race. Cancelling observation never cancels the payment or
releases an uncertain reservation. Durable recovery continues independently.

See [Lightning payments](../../docs/specs/lightning-payments.md) for the protocol,
CLI, security boundaries, recovery behavior, and PoC limitations.
