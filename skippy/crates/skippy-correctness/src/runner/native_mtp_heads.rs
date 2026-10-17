use anyhow::{Context, Result, ensure};
use serde::Serialize;
use skippy_runtime::{MtpSource, RuntimeConfig, StageModel};

use crate::cli::{NativeMtpHeadsArgs, RuntimeArgs, StageLoadMode};

use super::{
    native_mtp::{emit_report, native_mtp_artifact_summary, normalize_runtime_layer_end},
    stage_execution::{runtime_flash_attn, runtime_model_identity, stage_runtime_plans},
};

#[derive(Debug, Serialize)]
struct HeadResult {
    head_offset: usize,
    layer: u32,
    proposed_token: i32,
    target_token: i32,
    baseline_token: i32,
    proposal_matches_target: bool,
}

#[derive(Debug, Serialize)]
struct HeadCoverage {
    mode: &'static str,
    status: &'static str,
    expected_heads: usize,
    proposed_heads: usize,
    verified_heads: usize,
    target_state_matches: bool,
    heads: Vec<HeadResult>,
}

fn coverage(
    expected: usize,
    trunk_layers: u32,
    drafts: &[i32],
    target: &[i32],
    baseline: &[i32],
) -> HeadCoverage {
    let complete = expected > 0
        && drafts.len() == expected
        && target.len() == expected + 1
        && baseline.len() == expected + 1;
    let target_state_matches = complete && target == baseline;
    let heads = drafts
        .iter()
        .zip(target.iter().skip(1))
        .zip(baseline.iter().skip(1))
        .enumerate()
        .map(|(offset, ((draft, target), baseline))| HeadResult {
            head_offset: offset,
            layer: trunk_layers + offset as u32,
            proposed_token: *draft,
            target_token: *target,
            baseline_token: *baseline,
            proposal_matches_target: draft == target,
        })
        .collect::<Vec<_>>();
    HeadCoverage {
        mode: "native-mtp-heads",
        status: if target_state_matches { "pass" } else { "fail" },
        expected_heads: expected,
        proposed_heads: drafts.len(),
        verified_heads: heads.len(),
        target_state_matches,
        heads,
    }
}

fn admitted_decoder_config(runtime: &RuntimeArgs) -> Result<RuntimeConfig> {
    let plan = stage_runtime_plans(
        StageLoadMode::RuntimeSlice,
        &runtime.model,
        &[&runtime.model],
        &[(0, runtime.layer_end)],
        runtime.ctx_size,
        1,
    )?
    .into_iter()
    .next()
    .context("missing MTP runtime plan")?;
    // A decoder-only context still needs admission for GGUFs containing MTP
    // tensors. Preserve their resident union without creating a draft sidecar.
    Ok(RuntimeConfig {
        layer_end: runtime.layer_end,
        ctx_size: runtime.ctx_size,
        n_batch: runtime.n_batch,
        n_ubatch: runtime.n_ubatch,
        n_gpu_layers: runtime.n_gpu_layers,
        flash_attn_type: runtime_flash_attn(runtime.flash_attn),
        mtp_source: MtpSource::Disabled,
        resident_tensor_names: plan.resident_tensor_names,
        execution_contract: plan.execution_contract,
        activation_import_identities: plan.activation_import_identities,
        activation_import_bindings: plan.activation_import_bindings,
        activation_export_identities: plan.activation_export_identities,
        activation_export_bindings: plan.activation_export_bindings,
        ..RuntimeConfig::default()
    })
}

/// A proposal's depth selects its native head. Requesting the metadata count
/// therefore exercises all heads, unlike the binary smoke's one-token budget.
/// Teacher-force the same proposal prefix into both target and clean baseline:
/// rejected proposals are valid, but corrupted target state is not.
pub fn native_mtp_heads(mut args: NativeMtpHeadsArgs) -> Result<()> {
    ensure!(
        args.runtime.stage_load_mode == StageLoadMode::RuntimeSlice,
        "native-mtp-heads requires runtime-slice source weights"
    );
    let summary = native_mtp_artifact_summary(&args.runtime.model)?;
    ensure!(
        summary.supports_native_mtp(),
        "fixture has no complete integrated MTP head"
    );
    let expected = usize::try_from(summary.nextn_predict_layers)?;
    normalize_runtime_layer_end(&mut args.runtime)?;
    let identity = runtime_model_identity(&args.runtime)?;
    let config = admitted_decoder_config(&args.runtime)?;
    let (inputs, drafts, target) = {
        let mut integrated = config.clone();
        integrated.mtp_source = MtpSource::Integrated;
        let model = StageModel::open(&args.runtime.model, &integrated)?;
        let token = *model
            .tokenize(&args.runtime.prompt, true)?
            .first()
            .context("empty prompt")?;
        let mut session = model.create_session()?;
        let (authoritative, proposal) = session.decode_step_sampled_mtp(token, None, expected)?;
        let drafts = proposal
            .context("native MTP returned no proposal")?
            .token_ids;
        let mut inputs = vec![token];
        let mut target = vec![authoritative];
        // These prefixes deliberately retain earlier draft tokens even after
        // rejection, so every head is compared at the prefix it predicted.
        for input in std::iter::once(authoritative)
            .chain(drafts.iter().copied())
            .take(drafts.len())
        {
            inputs.push(input);
            target.push(session.decode_step_sampled(input, None)?);
        }
        (inputs, drafts, target)
    };
    // Drop the proposal model and contexts before opening an independent
    // decoder-only baseline, so this lane never doubles model residency.
    let baseline = {
        let model = StageModel::open(&args.runtime.model, &config)?;
        let mut session = model.create_session()?;
        inputs
            .iter()
            .map(|token| session.decode_step_sampled(*token, None))
            .collect::<Result<Vec<_>>>()?
    };
    let report = coverage(
        expected,
        args.runtime.layer_end,
        &drafts,
        &target,
        &baseline,
    );
    emit_report(
        &serde_json::json!({"model_identity":identity,"coverage":report}),
        args.output.report_out.as_deref(),
    )?;
    ensure!(
        report.target_state_matches,
        "native MTP head coverage failed: expected {expected}, proposed {}, verified {}",
        report.proposed_heads,
        report.verified_heads
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_working_head_cannot_certify_a_three_head_model() {
        assert_eq!(coverage(3, 48, &[7], &[6, 8], &[6, 8]).status, "fail");
        assert_eq!(coverage(3, 48, &[7, 8, 9], &[6, 8], &[6, 8]).status, "fail");
    }

    #[test]
    fn rejected_proposals_pass_only_with_complete_unchanged_target_state() {
        let report = coverage(3, 48, &[7, 8, 9], &[6, 10, 11, 12], &[6, 10, 11, 12]);
        assert_eq!(report.status, "pass");
        assert_eq!(report.verified_heads, 3);
        assert_eq!(report.heads[2].layer, 50);
        assert!(!report.heads[0].proposal_matches_target);
        assert_eq!(
            coverage(3, 48, &[7, 8, 9], &[6, 10, 11, 99], &[6, 10, 11, 12]).status,
            "fail"
        );
    }
}
