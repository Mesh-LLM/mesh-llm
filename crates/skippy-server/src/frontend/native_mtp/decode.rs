use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{
    NativeMtpDraftOrigin, native_mtp_batched_verify_enabled, native_mtp_defer_reject_trim_enabled,
    native_mtp_reject_cooldown_tokens, native_mtp_suppress_cooldown_draft_limit,
    native_mtp_suppress_cooldown_drafts_enabled,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::frontend) struct NativeMtpDecodeOptions {
    pub(in crate::frontend) batched_verify: bool,
    pub(in crate::frontend) max_draft_tokens: usize,
    pub(in crate::frontend) min_draft_tokens: usize,
    pub(in crate::frontend) reject_cooldown_tokens: usize,
    pub(in crate::frontend) defer_reject_trim: bool,
    pub(in crate::frontend) suppress_cooldown_drafts: bool,
    pub(in crate::frontend) suppress_cooldown_draft_limit: usize,
}

impl NativeMtpDecodeOptions {
    pub(in crate::frontend) fn from_env() -> Self {
        Self {
            batched_verify: native_mtp_batched_verify_enabled(),
            max_draft_tokens: 1,
            min_draft_tokens: 0,
            reject_cooldown_tokens: native_mtp_reject_cooldown_tokens(),
            defer_reject_trim: native_mtp_defer_reject_trim_enabled(),
            suppress_cooldown_drafts: native_mtp_suppress_cooldown_drafts_enabled(),
            suppress_cooldown_draft_limit: native_mtp_suppress_cooldown_draft_limit(),
        }
    }

    pub(in crate::frontend) fn with_window(
        mut self,
        max_draft_tokens: usize,
        min_draft_tokens: usize,
    ) -> Self {
        self.max_draft_tokens = max_draft_tokens.max(1);
        self.min_draft_tokens = min_draft_tokens.min(self.max_draft_tokens);
        self
    }
}

#[derive(Debug, Default)]
pub(in crate::frontend) struct NativeMtpDecodeCounters {
    suppressed_cooldown_draft_count: usize,
    batched_verification_count: usize,
    initial_serial_verification_count: usize,
    initial_serial_accepted_count: usize,
    serial_after_gap_verification_count: usize,
    serial_after_gap_accepted_count: usize,
    verify_next_verification_count: usize,
    verify_next_accepted_count: usize,
    verify_next_draft_available_count: usize,
    verify_next_draft_adopted_count: usize,
    deferred_reject_trim_count: usize,
    deferred_reject_trim_local_ms: f64,
}

impl NativeMtpDecodeCounters {
    pub(in crate::frontend) fn batched_verification_count(&self) -> usize {
        self.batched_verification_count
    }

    pub(in crate::frontend) fn observe_suppressed_cooldown_draft(&mut self) {
        self.suppressed_cooldown_draft_count += 1;
    }

    pub(in crate::frontend) fn observe_batched_verification(
        &mut self,
        origin: NativeMtpDraftOrigin,
        accepted: bool,
    ) {
        self.batched_verification_count += 1;
        match origin {
            NativeMtpDraftOrigin::InitialSerial => {
                self.initial_serial_verification_count += 1;
                if accepted {
                    self.initial_serial_accepted_count += 1;
                }
            }
            NativeMtpDraftOrigin::SerialAfterGap => {
                self.serial_after_gap_verification_count += 1;
                if accepted {
                    self.serial_after_gap_accepted_count += 1;
                }
            }
            NativeMtpDraftOrigin::VerifyNext => {
                self.verify_next_verification_count += 1;
                if accepted {
                    self.verify_next_accepted_count += 1;
                }
            }
        }
    }

    pub(in crate::frontend) fn observe_verify_next_draft(
        &mut self,
        available: bool,
        adopted: bool,
    ) {
        if available {
            self.verify_next_draft_available_count += 1;
        }
        if adopted {
            self.verify_next_draft_adopted_count += 1;
        }
    }

    pub(in crate::frontend) fn observe_deferred_reject_trim(&mut self, local_ms: f64) {
        self.deferred_reject_trim_count += 1;
        self.deferred_reject_trim_local_ms += local_ms;
    }

    pub(in crate::frontend) fn insert_summary_attrs(
        &self,
        attrs: &mut BTreeMap<String, Value>,
        options: NativeMtpDecodeOptions,
    ) {
        attrs.insert(
            "llama_stage.native_mtp.reject_cooldown_tokens".to_string(),
            json!(options.reject_cooldown_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.max_draft_tokens".to_string(),
            json!(options.max_draft_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.min_draft_tokens".to_string(),
            json!(options.min_draft_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.defer_reject_trim".to_string(),
            json!(options.defer_reject_trim),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppress_cooldown_drafts".to_string(),
            json!(options.suppress_cooldown_drafts),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppress_cooldown_draft_limit".to_string(),
            json!(options.suppress_cooldown_draft_limit),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppressed_cooldown_draft_count".to_string(),
            json!(self.suppressed_cooldown_draft_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.batched_verification_count".to_string(),
            json!(self.batched_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.initial_serial_verification_count".to_string(),
            json!(self.initial_serial_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.initial_serial_accepted_count".to_string(),
            json!(self.initial_serial_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.serial_after_gap_verification_count".to_string(),
            json!(self.serial_after_gap_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.serial_after_gap_accepted_count".to_string(),
            json!(self.serial_after_gap_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_verification_count".to_string(),
            json!(self.verify_next_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_accepted_count".to_string(),
            json!(self.verify_next_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_draft_available_count".to_string(),
            json!(self.verify_next_draft_available_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_draft_adopted_count".to_string(),
            json!(self.verify_next_draft_adopted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.deferred_reject_trim_count".to_string(),
            json!(self.deferred_reject_trim_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.deferred_reject_trim_local_ms".to_string(),
            json!(self.deferred_reject_trim_local_ms),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_track_batched_verification_by_origin() {
        let mut counters = NativeMtpDecodeCounters::default();
        counters.observe_batched_verification(NativeMtpDraftOrigin::InitialSerial, true);
        counters.observe_batched_verification(NativeMtpDraftOrigin::SerialAfterGap, false);
        counters.observe_batched_verification(NativeMtpDraftOrigin::VerifyNext, true);
        counters.observe_verify_next_draft(true, false);
        counters.observe_verify_next_draft(true, true);
        counters.observe_suppressed_cooldown_draft();
        counters.observe_deferred_reject_trim(1.25);

        let mut attrs = BTreeMap::new();
        counters.insert_summary_attrs(
            &mut attrs,
            NativeMtpDecodeOptions {
                batched_verify: true,
                max_draft_tokens: 3,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 6,
                defer_reject_trim: true,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 2,
            },
        );

        assert_eq!(
            attrs.get("llama_stage.native_mtp.batched_verification_count"),
            Some(&json!(3))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.initial_serial_accepted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.serial_after_gap_accepted_count"),
            Some(&json!(0))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_accepted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_draft_available_count"),
            Some(&json!(2))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_draft_adopted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.suppressed_cooldown_draft_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.reject_cooldown_tokens"),
            Some(&json!(6))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.defer_reject_trim"),
            Some(&json!(true))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.deferred_reject_trim_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.deferred_reject_trim_local_ms"),
            Some(&json!(1.25))
        );
    }
}
