//! `skippy recheck-receipt`: checks an audit's evidence again from the
//! evidence file and the claimed GGUF alone. A receipt that names its model
//! as `repo@revision/file` needs nothing else: the GGUF is read from Hugging
//! Face by range.
//!
//! The checks are those of [`skippy_receipt_audit::recheck`]. This module
//! reads the evidence file, opens the claimed model, and prints the report.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use skippy_receipt_audit::claimed_file::{self, HfFile};
use skippy_receipt_audit::claimed_model::LoadClaimedModel;
use skippy_receipt_audit::output::NoAuditOutput;
use skippy_receipt_audit::recheck::recheck_evidence;
use skippy_receipt_audit::turn_audit::{TurnAudit, claimed_layer_count};
use skippy_receipts::evidence::Evidence;

pub struct RecheckReceiptCommand {
    pub evidence: PathBuf,
    /// The GGUF the receipt claims: a file here or `repo@revision/file` on
    /// Hugging Face. Without it, the model the receipt names, when that is
    /// a Hugging Face file.
    pub claimed_model: Option<PathBuf>,
}

/// Prints the verdict the evidence's checks give, and ends with it: `Ok`
/// for a pass, else an error that carries the outcome's exit code
/// ([`super::exit_code`]).
pub fn recheck(command: RecheckReceiptCommand, load: LoadClaimedModel<'_>) -> Result<()> {
    let audit = rechecked(&command, load)
        .unwrap_or_else(|error| TurnAudit::NoAudit(NoAuditOutput::new("", format!("{error:#}"))));
    super::conclude(audit, None)
}

/// Opens the evidence and the claimed model. An error is this machine's:
/// the file or the GGUF cannot be read.
fn rechecked(command: &RecheckReceiptCommand, load: LoadClaimedModel<'_>) -> Result<TurnAudit> {
    let text = std::fs::read(&command.evidence)
        .with_context(|| format!("read evidence {}", command.evidence.display()))?;
    let evidence: Evidence = serde_json::from_slice(&text).context("decode the evidence")?;
    let cache_root = claimed_file::default_cache_root();
    let claimed = match &command.claimed_model {
        Some(path) => claimed_file::open(path, &cache_root)?,
        // the node wrote this name: it is a Hugging Face file at a commit,
        // and never a file here
        None if HfFile::parse(&evidence.receipt.model).is_some() => {
            claimed_file::open_named_by_receipt(&evidence.receipt.model, &cache_root)?
        }
        None => bail!(
            "the receipt names its model as {:?}, not repo@revision/file on Hugging Face; pass --claimed-model",
            evidence.receipt.model
        ),
    };
    let layer_count = claimed_layer_count(&claimed.path)?;
    let model = load(claimed.path);
    Ok(recheck_evidence(
        &evidence,
        &claimed.weights,
        model.as_ref(),
        layer_count,
    ))
}
