# External wallet setup and existing-state adoption

> [!WARNING]
> Wallet implementations, including Lexe, are examples under exploration.
> These commands can provision a mainnet wallet and move real money. Use small
> amounts, protect recovery material, and do not use a funded profile for tests.

This guide uses the external **`lexe-wallet`** plugin and the current
`payments/wallets/lexe-wallet/` directory, not the former built-in wallet.

## Select the profile before opening a wallet

The parent directory of the selected `--config` file owns `payments/`. For
example, `/absolute/path/payer/config.toml` uses
`/absolute/path/payer/payments/`; without a custom config, the normal location
is `~/.mesh-llm/payments/`. Changing HOME alone does not relocate payments when
an explicit config selects another directory. Two config files in the same
directory share payment state; putting a config in an evidence subdirectory
selects a different ledger and wallet.

Use the same absolute config path for the node and **every** wallet command.
The config must exist. Explicit `--config` lets wallet CLI requests check the
running node's payment directory; `wallet --port` selects its management port,
not its OpenAI port. Do not drop the config check to bypass a mismatch.

Before using an existing wallet, follow [adoption](#adopt-existing-state-safely)
below. Even `get-balance` can open/provision a wallet: it is not a safe way to
find out whether a directory contains the intended seed.

## Install and start

Install the released plugin for your platform:

```sh
mesh-llm plugins install Mesh-LLM/lexe-wallet
```

Installed, enabled plugins are discovered when the node starts. Restart an
existing node after installation. Do not also register another copy of the
same executable. If choosing explicitly, add this to the selected config:

```toml
[payments]
wallet = "lexe-wallet"
```

A manual executable registration is an alternative to installation, not an
additional step:

```toml
[[plugin]]
name = "lexe-wallet"
command = "/absolute/path/to/lexe-wallet"
```

For a fresh profile, create its config first. For an existing profile, keep its
config and payment state intact. In these examples replace the profile path:

```sh
CONFIG="/absolute/path/payer/config.toml"
mesh-llm --config "$CONFIG" client --console 3131
```

Keep the node running. In a second terminal, set `CONFIG` to the same path.
Plugin installation/startup alone does not provision the wallet. Wallet-dependent
commands need the node running; ledger-only policy, pricing and pending commands
can work offline.

## Balance, funding, policy and sending

Only after confirming the profile (and adopting existing state if necessary):

```sh
CONFIG="/absolute/path/payer/config.toml"
mesh-llm --config "$CONFIG" wallet --port 3131 policy
mesh-llm --config "$CONFIG" wallet --port 3131 get-balance
mesh-llm --config "$CONFIG" wallet --port 3131 get-transactions --limit 20
```

A fresh profile starts **free-only**. Installing/funding a wallet does not enable
automatic inference spending. If the intended wallet already has spendable
funds, no top-up is required. Otherwise create a funding invoice:

```sh
mesh-llm --config "$CONFIG" wallet --port 3131 fund-wallet --amount-sats 1000
```

Pay that invoice from an existing external Lightning wallet, then check balance
and transactions again. `fund-wallet` creates an invoice; it does not transfer
funds by itself. Amounts shown in millisatoshis use **1 sat = 1,000 msat**.

Enable paid inference only when wanted, with a deliberately small daily budget:

```sh
mesh-llm --config "$CONFIG" wallet --port 3131 policy --mode automatic --daily-budget-sats 100
mesh-llm --config "$CONFIG" wallet --port 3131 pending
mesh-llm --config "$CONFIG" wallet --port 3131 policy --mode free-only
```

Policy persists across restarts. Free-only prevents new automatic paid inference;
it does not cancel payments already submitted. The budget is not a wallet balance.

An explicit send is a separate authorization to pay the supplied mainnet BOLT11
invoice, not a way to enable automatic inference. Replace the placeholder only
when you intend to pay:

```sh
mesh-llm --config "$CONFIG" wallet --port 3131 send 'lnbc...' --max-fee-msat 1000
# For an amount-less invoice, also supply --amount-msat AMOUNT.
```

Inspect the destination, amount and fee limit before sending. After a timeout or
uncertain result, inspect transactions and pending records and let recovery
reconcile the same payment; do not create a replacement invoice/payment merely
because the response was lost. `pending` includes completed history, not only
unfinished payments. Do not erase reservations to make the balance look clear.

## Adopt existing state safely

This is preservation of an existing **`lexe-wallet`** identity, not a generic
provider migration. Do not start with `get-balance`, `fund-wallet` or `send`.

1. Stop the specific node using this profile and ensure no wallet process is
   writing its state. Preserve a private backup of the whole config/payment
   directory, including the ledger and any SQLite WAL/SHM files, pin and wallet
   recovery material. Never put these in logs, issues or repositories.
2. Confirm the intended config parent and its existing `payments/` directory.
   Inspect only the pin's non-secret `plugin`, `wallet_id`, `provider` and
   `network` fields. Preserve the pin and ledger unchanged.
3. The current destination is `payments/wallets/lexe-wallet/`. If this profile
   already has that directory, use it as-is. If a previously external
   `lexe-wallet` profile stores its state in `payments/lexe/`, move that **whole
   wallet directory** into the current destination while stopped, only after
   confirming the destination does not exist and the pin already names
   `lexe-wallet`. Retain its seed, associated files and private permissions;
   do not merge directories or copy only the seed into a newly opened wallet.
4. If both directories exist, the pin is missing or unexpected, or identity is
   uncertain, stop and resolve provenance before opening anything. In particular,
   a former **`wallet-lexe`** pin is not a `lexe-wallet` pin. Renaming/deleting the
   pin or using `wallet unpin` to force adoption is not this procedure. Unpin is
   a separate guarded provider-switch operation; it does not move funds.
5. Check the expected wallet state exists at the final path **before** restarting.
   Restart with the same config and explicit plugin selection, then check balance,
   transactions, policy and pending records through that profile-bound CLI.
   Require the existing pin/identity, known history and reservations to remain
   consistent. An unexpected zero balance or identity mismatch is a reason to
   stop and inspect the path, not to fund another wallet or remove its pin.

The host checks the returned wallet identity against the pin, but a missing seed
can allow the plugin to provision a different wallet before that check rejects
it. The directory preflight is therefore essential; the pin is not a substitute.

For the protocol, persistence and recovery contract, see the
[Lightning payments specification](../../specs/lightning-payments.md).
