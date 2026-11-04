//! Checks an audit's evidence again from the evidence and the claimed GGUF
//! alone, as `skippy recheck-receipt` does.
//!
//! First it shows that the evidence is what it says: the node signed the
//! receipt, the opening and the attention rows, and the client's VRF drew
//! the challenge and the checked rows. Then it runs the audit's checks on
//! the opening again, on the same path an audit takes: the opening's form
//! and Merkle paths, every value against the claimed model, and then the
//! checks. Anyone with the claimed GGUF gets the same verdict, so a
//! failure the client shows is the node's. The output head is checked
//! without the client's secret key, on every row of the output matrix,
//! which accepts the logits the key accepts.
//!
//! A file that is not evidence, a model this machine cannot check, and
//! evidence without the attention rows are no audit: none of them says
//! anything about the node.

use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::binding;
use skippy_receipts::coverage;
use skippy_receipts::evidence::{Evidence, Provenance};
use skippy_receipts::node_fault::NodeFault;
use skippy_receipts::opened_values;
use skippy_receipts::weights::WeightFile;

use crate::attention_audit::{self, OpenedRows, RowsAnswer};
use crate::claimed_model::ClaimedModel;
use crate::opening_checks::{Checks, check_opening};
use crate::output::{AuditOutput, NoAuditOutput};
use crate::turn_audit::{TurnAudit, executable_layer_bytes, fits_claimed_model};

/// Runs the audit's checks on `evidence` again.
pub fn recheck_evidence(
    evidence: &Evidence,
    weights: &WeightFile,
    model: &dyn ClaimedModel,
    layer_count: u32,
) -> TurnAudit {
    let id = evidence.receipt.receipt_id.as_str();
    let no_audit = |reason: String| TurnAudit::NoAudit(NoAuditOutput::new(id, reason));
    let provenance = match evidence.verify(&executable_layer_bytes(weights, layer_count)) {
        Ok(provenance) => provenance,
        // nothing in such a file is known to be the node's
        Err(error) => return no_audit(format!("not evidence of an audit: {error:#}")),
    };
    if let Some(reason) = coverage::unauditable(&evidence.receipt, weights, &model.runtime()) {
        return no_audit(format!("cannot recheck this evidence here: {reason}"));
    }
    let mut output = AuditOutput::new(
        String::new(),
        evidence.receipt.clone(),
        evidence.challenge.clone(),
    );
    output.rows_proof = Some(evidence.rows_proof.clone());
    // the same path an audit takes: the receipt against the claimed model,
    // the opening's form and paths, then every value against the model
    if let Err(error) = fits_claimed_model(weights, layer_count, &evidence.receipt) {
        return refused(output, &error, |output, reason| {
            output.receipt_error = Some(reason);
        });
    }
    let validated =
        binding::verify_paths(&evidence.receipt, &evidence.challenge, &provenance.opening)
            .and_then(|verified| opened_values::validate(verified, weights));
    let validated = match validated {
        Ok(validated) => validated,
        Err(error) => {
            return refused(output, &error, |output, reason| {
                output.opening_error = Some(reason);
            });
        }
    };
    let checks = Checks {
        timing_runs: 1,
        response: None,
        // the key is secret: a recheck checks the logits on every row
        head_key: None,
    };
    let layouts = check_opening(
        &checks,
        model,
        weights,
        &validated,
        &provenance.rows_seed,
        &mut output,
    );
    // the rows the node sent answer the blocks the same seed draws again
    let open_rows = |blocks: &BlockChallenge| evidence_rows(&provenance, blocks);
    let (attention, _) =
        attention_audit::check_attention(&validated, &layouts, &provenance.rows_seed, &open_rows);
    output.attention = Some(attention);
    output.finish();
    TurnAudit::Audited {
        output: Box::new(output),
        evidence: None,
    }
}

/// A receipt or an opening the checks refuse: a failure when the error
/// blames the node, else no audit.
fn refused(
    mut output: AuditOutput,
    error: &anyhow::Error,
    record: impl FnOnce(&mut AuditOutput, String),
) -> TurnAudit {
    if NodeFault::find(error).is_none() {
        let id = output.receipt.receipt_id;
        return TurnAudit::NoAudit(NoAuditOutput::new(id, format!("{error:#}")));
    }
    record(&mut output, format!("{error:#}"));
    output.finish();
    TurnAudit::Audited {
        output: Box::new(output),
        evidence: None,
    }
}

/// The attention rows of the evidence, for the blocks a recheck draws.
/// Evidence without rows, or with rows the node signed for other blocks,
/// is incomplete: the blocks cannot be checked, which is no failure. Rows
/// the node signed for these blocks that do not decode are its defect.
pub(crate) fn evidence_rows(provenance: &Provenance, blocks: &BlockChallenge) -> RowsAnswer {
    let Some(rows) = provenance.rows.as_ref() else {
        return RowsAnswer::Missing("the evidence holds no attention rows".into());
    };
    let bytes = match rows.answering(&provenance.binding, blocks) {
        Ok(bytes) => bytes.to_vec(),
        Err(error) => {
            return RowsAnswer::Missing(format!(
                "the evidence holds no attention rows the node signed for these blocks: {error:#}"
            ));
        }
    };
    match serde_json::from_slice(&bytes) {
        Ok(decoded) => RowsAnswer::Signed(OpenedRows {
            rows: decoded,
            bytes,
            signature: Some(rows.signature().to_owned()),
            seconds: 0.0,
        }),
        Err(error) => {
            RowsAnswer::Defective(format!("the signed attention rows do not decode: {error}"))
        }
    }
}
