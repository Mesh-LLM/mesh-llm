use clap::{Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum PaymentMode {
    FreeOnly,
    Automatic,
}

#[derive(Debug, Subcommand)]
pub enum WalletCommand {
    /// Show spendable balance in millisatoshis.
    #[command(alias = "balance")]
    GetBalance,
    /// List recent wallet transactions.
    #[command(alias = "transactions")]
    GetTransactions {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Create a mainnet BOLT11 invoice, amount-less unless --amount-sats is given.
    #[command(alias = "fund")]
    FundWallet {
        /// Fixed invoice amount in satoshis, for payers that reject amount-less invoices.
        #[arg(long)]
        amount_sats: Option<u64>,
    },
    /// Pay a mainnet BOLT11 invoice with a bounded routing fee.
    Send {
        invoice: String,
        #[arg(long)]
        amount_msat: Option<u64>,
        #[arg(long, default_value_t = 1000)]
        max_fee_msat: u64,
    },
    /// Inspect durable inference payment requests.
    Pending,
    /// List peers refused paid inference for unpaid debt, with the identifier `unblock` takes.
    Blocked,
    /// Forgive a peer's recorded debt so it may request paid inference again. Nothing is refunded.
    Unblock {
        /// Full peer ID from `wallet blocked`, or a unique prefix of at least 8 characters.
        peer: String,
    },
    /// Forget which wallet backs the payment ledger so another can be opened.
    /// Only works with the node stopped; refused while any payment still depends on the old wallet.
    Unpin,
    /// Use free providers only, or automatically pay for inference within a daily budget.
    Policy {
        #[arg(long, value_enum)]
        mode: Option<PaymentMode>,
        #[arg(long, requires = "mode")]
        daily_budget_sats: Option<u64>,
    },
    /// Show or set the minimum Lightning payment (default 1 sat). Sellers raise any
    /// smaller final remainder to it; buyers refuse sellers whose minimum is above theirs.
    MinPayment {
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=1000))]
        sats: Option<u64>,
    },
    /// Show or set the admission deposit new peers pay this node before it serves them
    /// (default: the minimum payment, and never below it).
    /// It is applied to the bill. Buyers refuse sellers whose deposit is above theirs.
    AdmissionDeposit {
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=1000))]
        sats: Option<u64>,
        /// Follow the minimum payment again instead of a set deposit.
        #[arg(long, conflicts_with = "sats")]
        default: bool,
    },
    /// List seller rates, or enable an exact model (defaults: 500 input / 1500 output msat per million).
    Pricing {
        model: Option<String>,
        #[arg(long, requires_all = ["model", "output_msat_per_million"], conflicts_with = "free")]
        input_msat_per_million: Option<u64>,
        #[arg(long, requires_all = ["model", "input_msat_per_million"], conflicts_with = "free")]
        output_msat_per_million: Option<u64>,
        /// Discount reported cached input by this percentage (0 disables it).
        /// Without rates, update the model's existing price without resetting it.
        #[arg(long, requires = "model", conflicts_with = "free", value_parser = clap::value_parser!(u8).range(0..=100))]
        cache_discount_percent: Option<u8>,
        #[arg(long, requires = "model")]
        free: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct WalletCli {
        #[command(subcommand)]
        command: WalletCommand,
    }

    #[test]
    fn policy_has_two_modes_and_no_approval_or_opt_in_commands() {
        for args in [
            vec!["wallet", "policy"],
            vec!["wallet", "blocked"],
            vec!["wallet", "unblock", "0123456789ab"],
            vec!["wallet", "policy", "--mode", "free-only"],
            vec![
                "wallet",
                "policy",
                "--mode",
                "automatic",
                "--daily-budget-sats",
                "100",
            ],
        ] {
            assert!(WalletCli::try_parse_from(args).is_ok());
        }
        for args in [
            vec!["wallet", "policy", "--mode", "manual"],
            vec!["wallet", "payment-intent"],
            vec!["wallet", "approve", "id"],
            vec!["wallet", "reject", "id"],
            vec!["wallet", "unblock"],
        ] {
            assert!(WalletCli::try_parse_from(args).is_err());
        }
    }

    #[test]
    fn admission_deposit_can_return_to_the_minimum() {
        assert!(WalletCli::try_parse_from(["wallet", "admission-deposit", "--default"]).is_ok());
        assert!(
            WalletCli::try_parse_from(["wallet", "admission-deposit", "--default", "--sats", "5"])
                .is_err()
        );
    }
}
