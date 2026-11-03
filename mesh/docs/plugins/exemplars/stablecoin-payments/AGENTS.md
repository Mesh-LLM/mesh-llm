# Stablecoin demo development

This standalone workspace ships two separately installed plugins: `pay-stablecoin`
and `wallet-ousd-tempo`. Keep all token, pricing, wallet and settlement logic here;
the Mesh checkout is an SDK/host dependency and must remain payment agnostic.

The demonstration uses Tempo mainnet, chain 4217, and native OUSD. Never silently
switch to testnet, another token, a fee sponsor, escrow or accumulated credit.
Unit tests use deterministic local fixtures. Live spending is an explicit operator
action after funding, with configured amount and fee bounds.

Use `just build`, `just check` and `just test`. Run Cargo commands serially. Keep
runtime modules below 1,000 lines and group code by responsibility. Never log keys,
passwords or invite credentials. Do not generate or fund real wallets during code
development. Keep operation schemas in `crates/wallet-contract` in sync with both
consumers. Every crate needs a README and manifest description.
