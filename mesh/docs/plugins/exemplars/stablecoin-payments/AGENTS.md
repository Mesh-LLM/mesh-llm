# Stablecoin demo development

This standalone workspace ships a payment engine, `pay-stablecoin`, and separately
installed wallet providers, `wallet-ousd-tempo` and `wallet-usdt-ethereum`.
Keep all token, pricing, wallet and settlement logic here;
the Mesh checkout is an SDK/host dependency and must remain payment agnostic.

The demonstrations use Tempo mainnet (4217) OUSD, or canonical USDT on Ethereum
mainnet (1) with ETH fees through the standard Tether WDK. Each rail has
its own wallet and engine state. Never silently switch to testnet, another token,
a fee sponsor, escrow or accumulated credit.
Unit tests use deterministic local fixtures. Live spending is an explicit operator
action after funding, with configured amount and fee bounds.

Use `just build`, `just check` and `just test`. Run Cargo commands serially. Keep
runtime modules below 1,000 lines and group code by responsibility. Never log keys,
passwords or invite credentials. Do not generate or fund real wallets during code
development. Keep operation schemas in `crates/wallet-contract` in sync with both
consumers. Every crate needs a README and manifest description.

The Ethereum provider uses a private local Node.js 22.13+ WDK worker. Keep dependencies
and lockfile pinned; use `just wdk-install` before checks/builds on a new machine.
Never copy `node_modules`, wallet profiles or encrypted seeds into tracked source.
Tests use deterministic temporary wallets and local RPC fixtures, never mainnet.
