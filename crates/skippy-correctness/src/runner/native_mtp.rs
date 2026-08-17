use std::path::Path;

use anyhow::{Context, Result, bail};
use model_artifact::gguf::scan_gguf_compact_meta;
use serde::Serialize;
use skippy_protocol::binary::StageReply;
use skippy_runtime::ModelInfo;

#[derive(Clone, Copy)]
pub(crate) struct NativeMtpRequirement {
    pub(crate) require_draft: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeMtpArtifactSummary {
    pub(crate) nextn_predict_layers: u32,
    pub(crate) has_eh_proj: bool,
    pub(crate) has_enorm: bool,
    pub(crate) has_hnorm: bool,
}

impl NativeMtpArtifactSummary {
    pub(crate) fn supports_native_mtp(&self) -> bool {
        self.nextn_predict_layers > 0 && self.has_eh_proj && self.has_enorm && self.has_hnorm
    }

    pub(crate) fn missing_reasons(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.nextn_predict_layers == 0 {
            missing.push("*.nextn_predict_layers > 0");
        }
        if !self.has_eh_proj {
            missing.push("*.nextn.eh_proj tensor");
        }
        if !self.has_enorm {
            missing.push("*.nextn.enorm tensor");
        }
        if !self.has_hnorm {
            missing.push("*.nextn.hnorm tensor");
        }
        missing
    }
}

use crate::{
    cli::{NativeMtpArgs, RuntimeArgs},
    report::{NativeMtpSidebandReport, NativeMtpVerificationReport},
};

pub(crate) fn native_mtp_verification_report(
    requested: bool,
    first: &NativeMtpSidebandReport,
    second_target_token: Option<i32>,
    second_baseline_token: Option<i32>,
    verification_compute_us: Option<i64>,
) -> Option<NativeMtpVerificationReport> {
    if !requested && first.draft_tokens.is_empty() {
        return None;
    }

    let drafted_tokens = first.draft_tokens.len() as u64;
    let verification_count =
        u64::from(!first.draft_tokens.is_empty() && second_target_token.is_some());
    let accepted_tokens = u64::from(matches!(
        (first.draft_tokens.first(), second_target_token),
        (Some(draft), Some(target)) if *draft == target
    ));
    let rejected_tokens = verification_count.saturating_sub(accepted_tokens);
    let pending_tokens = drafted_tokens.saturating_sub(verification_count);
    let byte_identical = matches!((second_target_token, second_baseline_token), (Some(target), Some(baseline)) if target == baseline);
    let accept_rate = if verification_count == 0 {
        0.0
    } else {
        accepted_tokens as f64 / verification_count as f64
    };

    Some(NativeMtpVerificationReport {
        drafted_tokens,
        accepted_tokens,
        rejected_tokens,
        pending_tokens,
        verification_count,
        accept_rate,
        byte_identical,
        draft_tokens: first.draft_tokens.clone(),
        second_target_token,
        second_baseline_token,
        proposal_compute_us: first.proposal_compute_us,
        verification_compute_us,
    })
}

pub(crate) fn native_mtp_verification_satisfies_requirement(
    report: &Option<NativeMtpVerificationReport>,
    requirement: NativeMtpRequirement,
) -> bool {
    if !requirement.require_draft {
        return true;
    }
    report.as_ref().is_some_and(|report| {
        report.drafted_tokens > 0 && report.verification_count == 1 && report.byte_identical
    })
}

pub(crate) fn native_mtp_sideband_report(reply: &StageReply) -> NativeMtpSidebandReport {
    let authoritative_token = reply.predicted_tokens.first().copied();
    let draft_tokens = reply
        .native_mtp_draft
        .as_ref()
        .map_or_else(Vec::new, |draft| draft.token_ids.clone());
    let proposal_compute_us = reply
        .native_mtp_draft
        .as_ref()
        .map(|draft| draft.proposal_compute_us.max(0));
    NativeMtpSidebandReport {
        sideband_present: reply.native_mtp_draft.is_some(),
        predicted_token_count: reply.predicted_tokens.len(),
        authoritative_matches_reply: authoritative_token
            .is_none_or(|token| token == reply.predicted),
        authoritative_token,
        draft_token_count: draft_tokens.len(),
        draft_tokens,
        proposal_compute_us,
    }
}

pub(crate) fn native_mtp_requirement(args: NativeMtpArgs) -> NativeMtpRequirement {
    NativeMtpRequirement {
        require_draft: args.require_native_mtp_draft,
    }
}

pub(crate) fn ensure_native_mtp_artifact_if_required(
    runtime: &RuntimeArgs,
    requirement: NativeMtpRequirement,
) -> Result<()> {
    if !requirement.require_draft {
        return Ok(());
    }

    let model_path = native_mtp_preflight_model_path(runtime);
    if !model_path.is_file() {
        return Ok(());
    }

    let summary = native_mtp_artifact_summary(model_path)?;
    if summary.supports_native_mtp() {
        return Ok(());
    }

    bail!(
        "native MTP draft was required, but {} does not advertise a usable native MTP head: missing {}",
        model_path.display(),
        summary.missing_reasons().join(", ")
    );
}

pub(crate) fn native_mtp_preflight_model_path(runtime: &RuntimeArgs) -> &Path {
    runtime
        .stage_model
        .as_deref()
        .filter(|path| path.is_file())
        .unwrap_or(runtime.model.as_path())
}

pub(crate) fn native_mtp_artifact_summary(model_path: &Path) -> Result<NativeMtpArtifactSummary> {
    let meta = scan_gguf_compact_meta(model_path)
        .with_context(|| format!("inspect GGUF metadata for {}", model_path.display()))?;
    let info = ModelInfo::open(model_path)
        .with_context(|| format!("inspect GGUF tensors for {}", model_path.display()))?;
    let tensors = info.tensors()?;
    Ok(native_mtp_artifact_summary_from_names(
        meta.nextn_predict_layers,
        tensors.iter().map(|tensor| tensor.name.as_str()),
    ))
}

fn native_mtp_artifact_summary_from_names<'a>(
    nextn_predict_layers: u32,
    names: impl IntoIterator<Item = &'a str>,
) -> NativeMtpArtifactSummary {
    let mut summary = NativeMtpArtifactSummary {
        nextn_predict_layers,
        ..NativeMtpArtifactSummary::default()
    };
    for name in names {
        let name = name.to_ascii_lowercase();
        summary.has_eh_proj |= native_mtp_name_matches(&name, "eh_proj");
        summary.has_enorm |= native_mtp_name_matches(&name, "enorm");
        summary.has_hnorm |= native_mtp_name_matches(&name, "hnorm");
    }
    summary
}

pub(crate) fn native_mtp_name_matches(name: &str, suffix: &str) -> bool {
    name.contains(&format!(".nextn.{suffix}"))
        || name.contains(&format!(".{suffix}."))
        || name.ends_with(&format!(".{suffix}"))
}

pub(crate) fn native_mtp_satisfies_requirement(
    report: &NativeMtpSidebandReport,
    requirement: NativeMtpRequirement,
) -> bool {
    report.authoritative_matches_reply && (!requirement.require_draft || report.sideband_present)
}

pub(crate) fn emit_report<T: Serialize>(report: &T, report_out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    println!("{json}");
    if let Some(path) = report_out {
        match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("create report directory {}", parent.display()))?;
            }
            _ => {}
        }
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write correctness report {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{StageNativeMtpDraft, StageReplyStats, WireReplyKind};

    fn predicted_reply(
        predicted: i32,
        predicted_tokens: Vec<i32>,
        native_mtp_draft: Option<StageNativeMtpDraft>,
    ) -> StageReply {
        StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted,
            predicted_tokens,
            native_mtp_draft,
            window: Default::default(),
            stats: StageReplyStats::default(),
        }
    }

    #[test]
    fn native_mtp_report_treats_plain_authoritative_token_as_no_draft() {
        let report = native_mtp_sideband_report(&predicted_reply(11, vec![11], None));

        assert!(!report.sideband_present);
        assert_eq!(report.predicted_token_count, 1);
        assert!(report.authoritative_matches_reply);
        assert_eq!(report.authoritative_token, Some(11));
        assert_eq!(report.draft_token_count, 0);
        assert!(report.draft_tokens.is_empty());
        assert_eq!(report.proposal_compute_us, None);
    }

    #[test]
    fn native_mtp_report_extracts_typed_draft() {
        let report = native_mtp_sideband_report(&predicted_reply(
            11,
            vec![11],
            Some(StageNativeMtpDraft {
                token_ids: vec![12, 13],
                proposal_compute_us: 34,
            }),
        ));

        assert!(report.sideband_present);
        assert_eq!(report.predicted_token_count, 1);
        assert!(report.authoritative_matches_reply);
        assert_eq!(report.authoritative_token, Some(11));
        assert_eq!(report.draft_token_count, 2);
        assert_eq!(report.draft_tokens, vec![12, 13]);
        assert_eq!(report.proposal_compute_us, Some(34));
    }

    #[test]
    fn native_mtp_report_flags_authoritative_prediction_mismatch() {
        let report = native_mtp_sideband_report(&predicted_reply(
            11,
            vec![10],
            Some(StageNativeMtpDraft {
                token_ids: vec![12],
                proposal_compute_us: 34,
            }),
        ));

        assert!(report.sideband_present);
        assert!(!report.authoritative_matches_reply);
        assert_eq!(report.authoritative_token, Some(10));
        assert_eq!(report.draft_tokens, vec![12]);
    }

    #[test]
    fn native_mtp_report_clamps_negative_proposal_time() {
        let report = native_mtp_sideband_report(&predicted_reply(
            11,
            vec![11],
            Some(StageNativeMtpDraft {
                token_ids: vec![12],
                proposal_compute_us: -34,
            }),
        ));

        assert_eq!(report.proposal_compute_us, Some(0));
    }

    #[test]
    fn native_mtp_requirement_can_require_draft_presence() {
        let no_draft = native_mtp_sideband_report(&predicted_reply(11, vec![11], None));
        let draft = native_mtp_sideband_report(&predicted_reply(
            11,
            vec![11],
            Some(StageNativeMtpDraft {
                token_ids: vec![12],
                proposal_compute_us: 34,
            }),
        ));
        let optional = NativeMtpRequirement {
            require_draft: false,
        };
        let required = NativeMtpRequirement {
            require_draft: true,
        };

        assert!(native_mtp_satisfies_requirement(&no_draft, optional));
        assert!(!native_mtp_satisfies_requirement(&no_draft, required));
        assert!(native_mtp_satisfies_requirement(&draft, required));
    }

    #[test]
    fn native_mtp_verification_report_accepts_matching_second_target() {
        let first = native_mtp_sideband_report(&predicted_reply(
            11,
            vec![11],
            Some(StageNativeMtpDraft {
                token_ids: vec![12, 13],
                proposal_compute_us: 34,
            }),
        ));
        let report = native_mtp_verification_report(true, &first, Some(12), Some(12), Some(9))
            .expect("verification report");

        assert_eq!(report.drafted_tokens, 2);
        assert_eq!(report.accepted_tokens, 1);
        assert_eq!(report.rejected_tokens, 0);
        assert_eq!(report.pending_tokens, 1);
        assert_eq!(report.verification_count, 1);
        assert_eq!(report.accept_rate, 1.0);
        assert!(report.byte_identical);
        assert_eq!(report.draft_tokens, vec![12, 13]);
        assert_eq!(report.proposal_compute_us, Some(34));
        assert_eq!(report.verification_compute_us, Some(9));
        assert!(native_mtp_verification_satisfies_requirement(
            &Some(report),
            NativeMtpRequirement {
                require_draft: true
            }
        ));
    }

    #[test]
    fn native_mtp_verification_report_rejects_mismatched_draft_without_failing_byte_identity() {
        let first = native_mtp_sideband_report(&predicted_reply(
            11,
            vec![11],
            Some(StageNativeMtpDraft {
                token_ids: vec![12],
                proposal_compute_us: 34,
            }),
        ));
        let report = native_mtp_verification_report(true, &first, Some(13), Some(13), Some(9))
            .expect("verification report");

        assert_eq!(report.drafted_tokens, 1);
        assert_eq!(report.accepted_tokens, 0);
        assert_eq!(report.rejected_tokens, 1);
        assert_eq!(report.pending_tokens, 0);
        assert_eq!(report.verification_count, 1);
        assert_eq!(report.accept_rate, 0.0);
        assert!(report.byte_identical);
        assert!(native_mtp_verification_satisfies_requirement(
            &Some(report),
            NativeMtpRequirement {
                require_draft: true
            }
        ));
    }

    #[test]
    fn native_mtp_verification_requirement_fails_when_required_draft_is_missing() {
        let first = native_mtp_sideband_report(&predicted_reply(11, vec![11], None));
        let report = native_mtp_verification_report(true, &first, Some(13), Some(13), Some(9))
            .expect("required verification report");

        assert_eq!(report.drafted_tokens, 0);
        assert_eq!(report.verification_count, 0);
        assert!(report.byte_identical);
        assert!(!native_mtp_verification_satisfies_requirement(
            &Some(report),
            NativeMtpRequirement {
                require_draft: true
            }
        ));
    }

    #[test]
    fn native_mtp_artifact_summary_requires_metadata_and_tensors() {
        let summary = native_mtp_artifact_summary_from_names(
            1,
            [
                "blk.47.nextn.eh_proj",
                "blk.47.nextn.enorm",
                "blk.47.nextn.hnorm",
            ],
        );

        assert!(summary.supports_native_mtp());
        assert!(summary.missing_reasons().is_empty());
    }

    #[test]
    fn native_mtp_artifact_summary_accepts_source_style_tensor_names() {
        let summary = native_mtp_artifact_summary_from_names(
            1,
            [
                "model.layers.47.eh_proj.weight",
                "model.layers.47.enorm.weight",
                "model.layers.47.hnorm.weight",
            ],
        );

        assert!(summary.supports_native_mtp());
    }

    #[test]
    fn native_mtp_artifact_summary_rejects_missing_nextn_metadata() {
        let summary = native_mtp_artifact_summary_from_names(
            0,
            [
                "blk.47.nextn.eh_proj",
                "blk.47.nextn.enorm",
                "blk.47.nextn.hnorm",
            ],
        );

        assert!(!summary.supports_native_mtp());
        assert_eq!(
            summary.missing_reasons(),
            vec!["*.nextn_predict_layers > 0"]
        );
    }
}
