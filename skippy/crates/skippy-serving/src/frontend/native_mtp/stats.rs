use std::collections::BTreeMap;

use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::frontend) enum NativeMtpVerification {
    #[default]
    NoPending,
    Accepted {
        draft: i32,
        target: i32,
    },
    Rejected {
        draft: i32,
        target: i32,
    },
}

impl NativeMtpVerification {
    pub(in crate::frontend) fn label(self) -> &'static str {
        match self {
            Self::NoPending => "none",
            Self::Accepted { .. } => "accepted",
            Self::Rejected { .. } => "rejected",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpStats {
    pub(in crate::frontend) drafted_tokens: u64,
    pub(in crate::frontend) accepted_tokens: u64,
    pub(in crate::frontend) rejected_tokens: u64,
    pub(in crate::frontend) pending_tokens: u64,
    pub(in crate::frontend) verification_count: u64,
    pub(in crate::frontend) proposal_compute_us: i64,
    pub(in crate::frontend) verification_compute_us: i64,
}

impl NativeMtpStats {
    pub(in crate::frontend) fn enabled(self) -> bool {
        self.drafted_tokens > 0 || self.verified_tokens() > 0
    }

    pub(in crate::frontend) fn verified_tokens(self) -> u64 {
        self.accepted_tokens + self.rejected_tokens
    }

    pub(in crate::frontend) fn accept_rate(self) -> f64 {
        let verified = self.verified_tokens();
        if verified == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / verified as f64
        }
    }

    pub(in crate::frontend) fn insert_attrs(self, attrs: &mut BTreeMap<String, Value>) {
        if !self.enabled() {
            attrs.insert("llama_stage.native_mtp.enabled".to_string(), json!(false));
            return;
        }

        attrs.insert("llama_stage.native_mtp.enabled".to_string(), json!(true));
        attrs.insert(
            "llama_stage.native_mtp.drafted".to_string(),
            json!(self.drafted_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.accepted".to_string(),
            json!(self.accepted_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.rejected".to_string(),
            json!(self.rejected_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.pending".to_string(),
            json!(self.pending_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.accept_rate".to_string(),
            json!(self.accept_rate()),
        );
        attrs.insert(
            "llama_stage.native_mtp.proposal_compute_us".to_string(),
            json!(self.proposal_compute_us),
        );
        attrs.insert(
            "llama_stage.native_mtp.verification_compute_us".to_string(),
            json!(self.verification_compute_us),
        );
        attrs.insert(
            "llama_stage.native_mtp.verifications".to_string(),
            json!(self.verification_count),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attrs_include_disabled_and_enabled_shapes() {
        let mut attrs = BTreeMap::new();
        NativeMtpStats::default().insert_attrs(&mut attrs);
        assert_eq!(
            attrs.get("llama_stage.native_mtp.enabled"),
            Some(&json!(false))
        );
        assert!(!NativeMtpStats::default().enabled());

        let stats = NativeMtpStats {
            drafted_tokens: 1,
            accepted_tokens: 1,
            verification_count: 1,
            proposal_compute_us: 7,
            verification_compute_us: 9,
            ..NativeMtpStats::default()
        };

        let mut attrs = BTreeMap::new();
        stats.insert_attrs(&mut attrs);
        assert_eq!(
            attrs.get("llama_stage.native_mtp.enabled"),
            Some(&json!(true))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.accept_rate"),
            Some(&json!(1.0))
        );
        assert!(stats.enabled());
    }

    #[test]
    fn verification_labels_match_telemetry_values() {
        assert_eq!(NativeMtpVerification::NoPending.label(), "none");
        assert_eq!(
            NativeMtpVerification::Accepted {
                draft: 1,
                target: 1
            }
            .label(),
            "accepted"
        );
        assert_eq!(
            NativeMtpVerification::Rejected {
                draft: 1,
                target: 2
            }
            .label(),
            "rejected"
        );
    }
}
