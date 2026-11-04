//! The receipts surface of the Skippy CLI: the receipts flags of `serve`
//! and how they become a stage's receipts options, and the `audit-receipt`
//! and `recheck-receipt` commands.

use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use skippy_serving::receipts::ReceiptsOptions;

/// The receipts flags of `serve`.
#[derive(clap::Args)]
pub struct ServeReceiptsArgs {
    /// Issue verifiable inference receipts but serve no receipts API: a
    /// later stage, which the stage before it reaches over the stage
    /// connection. Opt-in: every weight matmul is recorded.
    #[arg(long, group = "receipts_on")]
    pub receipts: bool,
    /// Issue receipts and serve the receipts API on this address. Opt-in:
    /// every weight matmul is recorded.
    #[arg(long, group = "receipts_on")]
    pub receipts_listen: Option<SocketAddr>,
    /// Allow a non-loopback --receipts-listen address. Openings carry
    /// activations derived from prompts, so only expose them to auditors you
    /// trust.
    #[arg(long, requires = "receipts_on")]
    pub receipts_allow_remote: bool,
    /// Receipts API of the next stage, for example http://127.0.0.1:9502.
    /// Without it, receipts reach the next stage over the stage connection.
    #[arg(long, requires = "receipts_on")]
    pub receipts_downstream: Option<String>,
    /// Model named in receipts. Defaults to the stage model id.
    #[arg(long, requires = "receipts_on")]
    pub receipts_claimed_model: Option<String>,
    /// File holding the hex Ed25519 seed that signs receipts. Without it,
    /// the stage signs with a key made at startup.
    #[arg(long, requires = "receipts_on")]
    pub receipts_signing_key: Option<PathBuf>,
}

impl ServeReceiptsArgs {
    /// The receipts options of a binary stage that serves `model_id`, when
    /// the flags turn receipts on.
    pub fn stage_options(&self, model_id: &str) -> Result<Option<ReceiptsOptions>> {
        let signing_key = self
            .receipts_signing_key
            .as_deref()
            .map(read_signing_key)
            .transpose()?;
        Ok(
            (self.receipts || self.receipts_listen.is_some()).then(|| ReceiptsOptions {
                listen: self.receipts_listen,
                allow_remote: self.receipts_allow_remote,
                downstream: self.receipts_downstream.clone(),
                claimed_model: self
                    .receipts_claimed_model
                    .clone()
                    .unwrap_or_else(|| model_id.to_owned()),
                signing_key,
            }),
        )
    }
}

/// A receipts signing key file: 64 hex digits, the Ed25519 seed.
fn read_signing_key(path: &std::path::Path) -> Result<[u8; 32]> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read receipts signing key {}", path.display()))?;
    let bytes =
        hex::decode(text.trim()).with_context(|| format!("{} is not hex", path.display()))?;
    <[u8; 32]>::try_from(bytes.as_slice()).map_err(|_| {
        anyhow::anyhow!(
            "{} holds {} bytes; an Ed25519 seed has 32",
            path.display(),
            bytes.len()
        )
    })
}

/// A receipts setting of `serve`: `receipts` itself or one of its
/// `receipts-` options. They all need a binary stage.
pub fn is_serve_setting(name: &str) -> bool {
    name == "receipts" || name.starts_with("receipts-")
}

#[derive(Parser)]
pub struct AuditReceiptArgs {
    /// OpenAI base URL of stage 0.
    #[arg(long, default_value = "http://127.0.0.1:9337")]
    pub api: String,
    /// Receipts API of stage 0. Without it the audit opens the receipt on
    /// the OpenAI API, which a Mesh node serves.
    #[arg(long)]
    pub receipts: Option<String>,
    /// The GGUF the node claims to serve: a file here, or
    /// `repo@revision/file` on Hugging Face, which the audit reads by range
    /// without downloading the model.
    #[arg(long)]
    pub claimed_model: PathBuf,
    #[arg(long, default_value = "Explain in two sentences why the sky is blue.")]
    pub prompt: String,
    #[arg(long, default_value_t = 32)]
    pub max_tokens: u32,
    /// Challenge this layer alone instead of random ones.
    #[arg(long)]
    pub layer: Option<u32>,
    /// Times to compute each challenged layer for the client cost report.
    #[arg(long, default_value_t = 1)]
    pub timing_runs: u32,
    /// Write the audit's evidence to this file, for `recheck-receipt`.
    #[arg(long, conflicts_with = "layer")]
    pub evidence: Option<PathBuf>,
    /// Sample at this temperature. At 0 the node decodes greedily. Above 0
    /// the audit sets a seed and replays the draw of each checked token.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f64,
    /// Top-p of a sampled turn. Without it the node's default applies.
    #[arg(long, requires = "temperature")]
    pub top_p: Option<f64>,
    /// Top-k of a sampled turn. Without it the node's default applies.
    #[arg(long, requires = "temperature")]
    pub top_k: Option<i64>,
    /// Min-p of a sampled turn. Without it the node's default applies.
    #[arg(long, requires = "temperature")]
    pub min_p: Option<f64>,
    /// Seed of a sampled turn, above 0. Without it the audit picks one.
    #[arg(long, requires = "temperature")]
    pub seed: Option<u32>,
    /// Reasoning tokens to allow a thinking model. The node forces the end
    /// of the reasoning when they run out, and the audit replays which
    /// tokens it forced. Without it the model reasons without a limit.
    #[arg(long)]
    pub reasoning_budget: Option<u32>,
}

#[derive(Parser)]
pub struct RecheckReceiptArgs {
    /// Evidence that `audit-receipt --evidence` wrote.
    #[arg(long)]
    pub evidence: PathBuf,
    /// The GGUF the receipt claims: a file here, or `repo@revision/file`
    /// on Hugging Face. Defaults to the model the receipt names when that
    /// is a Hugging Face file.
    #[arg(long)]
    pub claimed_model: Option<PathBuf>,
}

impl From<RecheckReceiptArgs> for skippy_commands::receipts::RecheckReceiptCommand {
    fn from(args: RecheckReceiptArgs) -> Self {
        Self {
            evidence: args.evidence,
            claimed_model: args.claimed_model,
        }
    }
}

impl From<AuditReceiptArgs> for skippy_commands::receipts::AuditReceiptCommand {
    fn from(args: AuditReceiptArgs) -> Self {
        Self {
            api: args.api,
            receipts: args.receipts,
            claimed_model: args.claimed_model,
            prompt: args.prompt,
            max_tokens: args.max_tokens,
            layer: args.layer,
            timing_runs: args.timing_runs.max(1),
            evidence: args.evidence,
            sampling: skippy_commands::receipts::AuditSampling {
                temperature: args.temperature,
                top_p: args.top_p,
                top_k: args.top_k,
                min_p: args.min_p,
                seed: args.seed,
                reasoning_budget: args.reasoning_budget,
            },
        }
    }
}

/// Runs `audit-receipt`.
pub async fn audit(args: AuditReceiptArgs) -> Result<()> {
    let command: skippy_commands::receipts::AuditReceiptCommand = args.into();
    tokio::task::spawn_blocking(move || {
        skippy_commands::receipts::run(
            command,
            &skippy_receipt_audit::native_model::load_claimed_model,
        )
    })
    .await?
}

/// Runs `recheck-receipt`.
pub async fn recheck(args: RecheckReceiptArgs) -> Result<()> {
    let command: skippy_commands::receipts::RecheckReceiptCommand = args.into();
    tokio::task::spawn_blocking(move || {
        skippy_commands::receipts::recheck(
            command,
            &skippy_receipt_audit::native_model::load_claimed_model,
        )
    })
    .await?
}

/// The exit code a failed command names when it is a receipt audit that did
/// not pass: 1 failed, 2 not audited, 3 the node did not answer.
pub fn audit_exit_code(error: &anyhow::Error) -> Option<std::process::ExitCode> {
    skippy_commands::receipts::exit_code(error).map(std::process::ExitCode::from)
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::cli::{Cli, Command};

    #[test]
    fn a_later_stage_issues_receipts_without_an_api() {
        let parse = |flags: &[&str]| {
            let mut args = vec![
                "skippy",
                "serve",
                "--config",
                "stage.json",
                "--stage-transport",
                "binary",
            ];
            args.extend_from_slice(flags);
            Cli::try_parse_from(args)
        };

        let cli = parse(&["--receipts", "--receipts-claimed-model", "m"]).expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        let receipts = args
            .stage
            .receipts
            .stage_options("stage-model")
            .expect("resolve receipts options")
            .expect("receipts on");
        assert_eq!(receipts.listen, None);
        assert_eq!(receipts.downstream, None);
        assert_eq!(receipts.claimed_model, "m");

        assert!(parse(&["--receipts-claimed-model", "m"]).is_err());
        assert!(parse(&["--receipts", "--receipts-listen", "127.0.0.1:9501"]).is_err());
    }

    #[test]
    fn audit_receipt_samples_only_with_a_temperature() {
        let parse = |flags: &[&str]| {
            let mut args = vec!["skippy", "audit-receipt", "--claimed-model", "m.gguf"];
            args.extend_from_slice(flags);
            Cli::try_parse_from(args).map(|cli| {
                let Command::AuditReceipt(args) = cli.command else {
                    panic!("expected audit-receipt");
                };
                skippy_commands::receipts::AuditReceiptCommand::from(args).sampling
            })
        };
        // greedy by default
        let greedy = parse(&[]).expect("parse");
        assert_eq!(greedy, skippy_commands::receipts::AuditSampling::default());
        let sampled = parse(&[
            "--temperature",
            "0.7",
            "--top-p",
            "0.95",
            "--top-k",
            "40",
            "--min-p",
            "0.05",
            "--seed",
            "1234",
        ])
        .expect("parse");
        assert_eq!(
            sampled,
            skippy_commands::receipts::AuditSampling {
                temperature: 0.7,
                top_p: Some(0.95),
                top_k: Some(40),
                min_p: Some(0.05),
                seed: Some(1234),
                reasoning_budget: None,
            }
        );
        // a reasoning budget goes with a greedy turn as well
        let budgeted = parse(&["--reasoning-budget", "64"]).expect("parse");
        assert_eq!(budgeted.reasoning_budget, Some(64));
        assert!(parse(&["--reasoning-budget", "-1"]).is_err());
        // a filter or a seed means nothing to a greedy turn
        for flags in [
            ["--top-k", "40"],
            ["--top-p", "0.9"],
            ["--min-p", "0.1"],
            ["--seed", "7"],
        ] {
            assert!(parse(&flags).is_err(), "{flags:?}");
        }
        assert!(parse(&["--temperature", "0.7", "--seed", "-1"]).is_err());
    }
}
