//! Mesh operations stay in Rust; local WDK owns Ethereum account derivation and signing.
mod bridge;
mod journal;
mod server;
#[cfg(test)]
mod tests;
mod wallet;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use wallet::{Settings, Wallet};
use wallet_contract::{PayRequest, PaymentRequest, VerifyRequest};

#[derive(Parser)]
#[command(
    version,
    about = "Noncustodial USDT on Ethereum mainnet using Tether WDK; fees in ETH"
)]
struct Cli {
    #[arg(long, global = true)]
    state_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    rpc_url: Option<String>,
    #[arg(long, global = true)]
    password_file: Option<PathBuf>,
    #[arg(long, global = true)]
    node_binary: Option<PathBuf>,
    #[arg(long, global = true)]
    worker_path: Option<PathBuf>,
    #[arg(long, global = true)]
    max_payment: Option<u64>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Explicitly create an encrypted local seed. No RPC or blockchain transaction.
    Init,
    /// Read the public receiving address without RPC access or unlocking.
    Address,
    Info,
    Balance,
    Estimate {
        #[arg(long)]
        request: PathBuf,
    },
    Pay {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        confirm_mainnet: bool,
    },
    Lookup {
        #[arg(long)]
        payment_id: String,
    },
    Verify {
        #[arg(long)]
        request: PathBuf,
    },
    Attempts,
    Serve,
}

fn output(value: impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn read<T: serde::de::DeserializeOwned>(path: PathBuf) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().any(|argument| argument == "--manifest") {
        println!(
            "{}",
            mesh_llm_plugin::package_manifest_json(&server::manifest())?
        );
        return Ok(());
    }
    let cli = Cli::parse();
    let mut settings = Settings::default();
    if let Some(value) = cli.state_dir {
        settings.data_dir = value;
    }
    if let Some(value) = cli.rpc_url {
        settings.rpc_url = value;
    }
    if let Some(value) = cli.node_binary {
        settings.node_binary = value;
    }
    if let Some(value) = cli.max_payment {
        settings.max_payment = value;
    }
    settings.password_file = cli.password_file;
    settings.worker_path = cli.worker_path;
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => server::run().await,
        Command::Init => {
            ensure!(
                settings.data_dir.is_absolute(),
                "--state-dir must be absolute"
            );
            settings
                .password_file
                .as_ref()
                .context("init requires --password-file")?;
            let worker = bridge::Bridge::spawn(&settings).await?;
            output(bridge::invoke(worker.as_ref(), "initialize", settings.worker_settings()).await?)
        }
        Command::Address => {
            let worker = bridge::Bridge::spawn(&settings).await?;
            output(bridge::invoke(worker.as_ref(), "profile", settings.worker_settings()).await?)
        }
        command => {
            let wallet = Wallet::load(settings).await?;
            match command {
                Command::Info => output(wallet.info().await?),
                Command::Balance => output(wallet.balance().await?),
                Command::Estimate { request } => {
                    output(wallet.estimate(read::<PaymentRequest>(request)?).await?)
                }
                Command::Pay {
                    request,
                    confirm_mainnet,
                } => {
                    ensure!(
                        confirm_mainnet,
                        "real mainnet spending requires --confirm-mainnet"
                    );
                    output(wallet.pay(read::<PayRequest>(request)?).await?)
                }
                Command::Lookup { payment_id } => output(wallet.lookup(&payment_id).await?),
                Command::Verify { request } => {
                    output(wallet.verify(read::<VerifyRequest>(request)?).await?)
                }
                Command::Attempts => output(wallet.journal.list()?),
                Command::Serve | Command::Init | Command::Address => unreachable!(),
            }
        }
    }
}
