use clap::{Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum PaymentMode {
    Manual,
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
    /// Authorize one request up to its displayed total including fees.
    Approve {
        id: String,
    },
    Reject {
        id: String,
    },
    /// Inspect or change automatic payment policy.
    Policy {
        #[arg(long, value_enum)]
        mode: Option<PaymentMode>,
        #[arg(long, requires = "mode")]
        daily_budget_sats: Option<u64>,
    },
    /// List seller rates, or enable an exact model (defaults: 500 input / 1500 output msat per million).
    Pricing {
        model: Option<String>,
        #[arg(long, requires_all = ["model", "output_msat_per_million"], conflicts_with = "free")]
        input_msat_per_million: Option<u64>,
        #[arg(long, requires_all = ["model", "input_msat_per_million"], conflicts_with = "free")]
        output_msat_per_million: Option<u64>,
        #[arg(long, default_value_t = 1)]
        minimum_invoice_msat: u64,
        #[arg(long, requires = "model")]
        free: bool,
    },
}
