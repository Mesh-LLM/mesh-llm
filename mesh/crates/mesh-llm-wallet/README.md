# mesh-llm-wallet

Provider-neutral wallet traits, validated BOLT11 invoices and durable provider
pins for wallet-linked Lightning plugins. Primary consumers are the payment
engine and external wallet adapters. No wallet SDK or host ledger is linked here.

`provisioning::WalletFactory` opens isolated plugin-owned state; discovery must
be side-effect free. `provider::WalletProvider` supplies authoritative payment
facts and preserves uncertainty after submission. Receiver arrival uses
`Transaction.claiming`, never display-only status text.

The optional legacy `plugin-server` feature retains the wallet.v1 contract for
existing consumers; the Lightning runtime uses an in-process factory instead.
