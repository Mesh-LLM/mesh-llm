# mesh-llm-payments

Provider-neutral integer-msat ledger, spending authorization, admission deposits,
token-metered settlement and recovery. Consumed by wallet-linked payment runtimes,
not the Mesh host. Wallet SDKs belong to external adapters.

Wallet observation is authoritative; cancellation does not cancel payments or
erase uncertainty. Never blindly retry a possibly submitted send.

See [payment types](../mesh-llm-payments-types/README.md) and
[Lightning runtime](../mesh-llm-pay-lightning/README.md).
