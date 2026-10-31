# mesh-llm-payments

Integer-msat pricing, durable spending authorization, and token-metered
inference settlement over a provider-neutral wallet. The seller side
(`serving`, `ledger::metered`) admits a request with a flat deposit (default
1 sat, `mesh-llm wallet admission-deposit`, optionally per model; none for
buyers in good standing), bills the
backend-reported usage at the admitted rates, applies the deposit and issues
one final invoice for the remainder; a buyer with three issued, unpaid final
invoices is refused. The wallet trait, BOLT11 validation
and the `wallet.v1` plugin contract live in `mesh-llm-wallet`; concrete
wallets such as Lexe are external plugins (`lexe-wallet`). This crate links no wallet SDK and owns
invoice lifetimes (`lifetimes`) and fee policy (`pricing`). Applications use the
local management API.

`WalletProvider::wait_for_payment` awaits an authoritative terminal payment
update. Adapters may use native events or poll behind this interface.
Subscriptions must also observe already-settled payments and close the
subscribe/lookup race. Cancelling observation never cancels the payment or
discards an uncertain payment record. Durable recovery continues independently.

See [Lightning payments](../../docs/specs/lightning-payments.md) for the protocol,
CLI, security boundaries, recovery behavior, and PoC limitations.
