//! `skippy recheck-receipt`: checks an audit's evidence again from the
//! evidence file and the claimed GGUF alone. A receipt that names its model
//! as `repo@revision/file` needs nothing else: the GGUF is read from Hugging
//! Face by range.
//!
//! First it shows that the evidence is what it says: the node signed the
//! receipt, the opening and the attention rows, and the client's VRF drew
//! the challenge and the checked rows. Then it runs the audit's checks on the opening again.
//! Anyone with the claimed GGUF gets the same verdict, so a failure the
//! client shows is the node's.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::binding;
use skippy_receipts::coverage;
use skippy_receipts::evidence::Evidence;

use super::LoadClaimedModel;
use super::attention_audit::{self, OpenedRows};
use super::claimed_file::{self, HfFile};
use super::output::{self, AuditOutput};
use super::turn_audit::{Checks, check_opening, claimed_layer_count, executable_layer_bytes};

pub struct RecheckReceiptCommand {
    pub evidence: PathBuf,
    /// The GGUF the receipt claims: a file here or `repo@revision/file` on
    /// Hugging Face. Without it, the model the receipt names, when that is
    /// a Hugging Face file.
    pub claimed_model: Option<PathBuf>,
}

/// Fails when the file is not evidence. Valid evidence is reported with
/// the verdict its checks give, pass or fail.
pub fn recheck(command: RecheckReceiptCommand, load: LoadClaimedModel<'_>) -> Result<()> {
    let text = std::fs::read(&command.evidence)
        .with_context(|| format!("read evidence {}", command.evidence.display()))?;
    let evidence: Evidence = serde_json::from_slice(&text).context("decode the evidence")?;
    let claimed_model = match command.claimed_model {
        Some(path) => path,
        None if HfFile::parse(&evidence.receipt.model).is_some() => {
            evidence.receipt.model.clone().into()
        }
        None => bail!(
            "the receipt names its model as {:?}, not repo@revision/file on Hugging Face; pass --claimed-model",
            evidence.receipt.model
        ),
    };
    let claimed = claimed_file::open(&claimed_model, &claimed_file::default_cache_root())?;
    let layer_count = claimed_layer_count(&claimed.path)?;
    let weights = claimed.weights;
    let model = load(claimed.path);
    let model = model.as_ref();
    let provenance = evidence
        .verify(&executable_layer_bytes(&weights, layer_count))
        .context("not evidence of an audit")?;
    if let Some(reason) = coverage::unauditable(&evidence.receipt, &weights, &model.runtime()) {
        bail!("cannot recheck this evidence here: {reason}");
    }
    let checked = binding::verify_shape(&evidence.receipt, layer_count).and_then(|()| {
        binding::verify_paths(&evidence.receipt, &evidence.challenge, &provenance.opening)
    });
    let mut output = AuditOutput::new(
        String::new(),
        evidence.receipt.clone(),
        evidence.challenge.clone(),
        &checked,
    );
    output.rows_proof = Some(evidence.rows_proof.clone());
    output.tokens = None;
    if checked.is_ok() {
        let checks = Checks {
            timing_runs: 1,
            response: None,
            // the key is secret: a recheck computes the whole head
            head_key: None,
        };
        let layouts = check_opening(
            &checks,
            model,
            &weights,
            &provenance.opening,
            &provenance.rows_seed,
            &mut output,
        )?;
        // the rows the node sent answer the blocks the same seed draws again
        let open_rows = |_: &BlockChallenge| -> Result<OpenedRows> {
            let bytes = provenance
                .rows
                .clone()
                .context("the evidence holds no attention rows")?;
            Ok(OpenedRows {
                rows: serde_json::from_slice(&bytes).context("decode the attention rows")?,
                bytes,
                signature: evidence.rows_signature.clone(),
                seconds: 0.0,
            })
        };
        let (attention, _) = attention_audit::check_attention(
            &evidence.receipt,
            &evidence.challenge,
            &provenance.opening,
            &layouts,
            &provenance.rows_seed,
            &open_rows,
        );
        output.attention = Some(attention);
    }
    output.finish();
    crate::console::present(&output, |out| output::render(out, &output))
}
