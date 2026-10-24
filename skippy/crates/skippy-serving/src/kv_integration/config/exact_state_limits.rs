//! Exact-state snapshot retention limits, including indivisible payloads.
use super::super::{EXACT_STATE_RECORD_QUEUE_BYTES, ExactStateByteLimits, StagePrefixCachePayload};

const EXACT_STATE_RECORD_QUEUE_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Let a stage record its latest and shared hybrid boundaries together when
/// its cache budget can retain both. Other payload modes retain the existing
/// 4 GiB limit. The two-credit limit still bounds the recorder, and its byte
/// allowance never exceeds 8 GiB except for one indivisible record.
pub(super) fn exact_state_record_queue_limit(
    cache_bytes: u64,
    payload: StagePrefixCachePayload,
) -> u64 {
    if payload == StagePrefixCachePayload::KvRecurrent {
        cache_bytes.clamp(
            EXACT_STATE_RECORD_QUEUE_BYTES,
            EXACT_STATE_RECORD_QUEUE_MAX_BYTES,
        )
    } else {
        EXACT_STATE_RECORD_QUEUE_BYTES
    }
}

// That retention floor is not allowed to grow without bound. Once the catalog
// crosses this multiple of the soft cap it evicts again, down to the single
// entry that keeps exact prefix reuse alive for the stage. That indivisible
// entry may itself exceed the limit.
const EXACT_STATE_HARD_CAP_MULTIPLE: u64 = 8;

/// Resolves the exact-state byte budget from the stage cache cap.
///
/// `configured_max_bytes` is the attention-derived stage budget, which is used
/// as the soft cap. The hard limit defaults to a multiple of it so the working
/// set stays bounded without inheriting an estimate that structurally
/// undercounts exact-state payloads. One indivisible snapshot may remain above
/// that limit. `configured_exact_max_bytes` wins over the legacy environment
/// override when provided; zero means unbounded on either surface.
pub(super) fn exact_state_byte_limits(
    configured_max_bytes: u64,
    configured_exact_max_bytes: Option<u64>,
    override_max_bytes: Option<&str>,
) -> ExactStateByteLimits {
    let hard_bytes = configured_exact_max_bytes
        .or_else(|| override_max_bytes.and_then(|value| value.trim().parse::<u64>().ok()))
        .unwrap_or_else(|| configured_max_bytes.saturating_mul(EXACT_STATE_HARD_CAP_MULTIPLE));
    ExactStateByteLimits {
        soft_bytes: configured_max_bytes,
        hard_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_integration::exact_state::{CaptureAdmission, ExactStateAdmissionCredit};
    use std::sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    };
    fn limits(soft_bytes: u64, hard_bytes: u64) -> ExactStateByteLimits {
        ExactStateByteLimits {
            soft_bytes,
            hard_bytes,
        }
    }
    #[test]
    fn recorder_limit_admits_long_hybrid_pair_only_with_cache_room() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let latest_bytes = 3_619_618_816 + 52_691_548;
        let shared_bytes = 2_432_696_320 + 52_691_548;
        assert!(latest_bytes + shared_bytes > 4 * GIB);
        assert!(
            latest_bytes + shared_bytes
                < exact_state_record_queue_limit(8 * GIB, StagePrefixCachePayload::KvRecurrent)
        );
        let queued_bytes = Arc::new(AtomicU64::new(0));
        let total = Arc::new(AtomicUsize::new(0));
        let best_effort = Arc::new(AtomicUsize::new(0));
        let mut latest = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("latest capture credit");
        let mut shared = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("shared capture credit");
        let limit = exact_state_record_queue_limit(8 * GIB, StagePrefixCachePayload::KvRecurrent);
        assert!(latest.reserve_bytes(&queued_bytes, latest_bytes, limit));
        assert!(shared.reserve_bytes(&queued_bytes, shared_bytes, limit));
        assert_eq!(
            queued_bytes.load(Ordering::Acquire),
            latest_bytes + shared_bytes
        );
        drop(latest);
        drop(shared);
        assert_eq!(queued_bytes.load(Ordering::Acquire), 0);
        let mut latest = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("latest capture credit");
        let mut shared = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("shared capture credit");
        assert!(latest.reserve_bytes(&queued_bytes, latest_bytes, 4 * GIB));
        assert!(!shared.reserve_bytes(&queued_bytes, shared_bytes, 4 * GIB));
        drop(latest);
        drop(shared);
        assert_eq!(queued_bytes.load(Ordering::Acquire), 0);
        assert_eq!(
            exact_state_record_queue_limit(0, StagePrefixCachePayload::KvRecurrent),
            4 * GIB
        );
        assert_eq!(
            exact_state_record_queue_limit(2 * GIB, StagePrefixCachePayload::KvRecurrent),
            4 * GIB
        );
        assert_eq!(
            exact_state_record_queue_limit(16 * GIB, StagePrefixCachePayload::KvRecurrent),
            8 * GIB
        );
        for payload in [
            StagePrefixCachePayload::ResidentKv,
            StagePrefixCachePayload::FullState,
            StagePrefixCachePayload::Disabled,
        ] {
            assert_eq!(exact_state_record_queue_limit(16 * GIB, payload), 4 * GIB);
        }
    }
    #[test]
    fn exact_state_byte_limits_scale_the_ceiling_off_the_stage_budget() {
        assert_eq!(
            exact_state_byte_limits(1_024, None, None),
            limits(1_024, 1_024 * EXACT_STATE_HARD_CAP_MULTIPLE)
        );
        // An unset stage budget stays unbounded, as it was before.
        assert_eq!(exact_state_byte_limits(0, None, None), limits(0, 0));
    }

    #[test]
    fn exact_state_byte_limits_honour_the_operator_override() {
        assert_eq!(
            exact_state_byte_limits(1_024, None, Some(" 4096 ")),
            limits(1_024, 4_096)
        );
        assert_eq!(
            exact_state_byte_limits(1_024, None, Some("0")),
            limits(1_024, 0)
        );
        // A malformed override must not silently disable the ceiling.
        assert_eq!(
            exact_state_byte_limits(1_024, None, Some("not-a-number")),
            limits(1_024, 1_024 * EXACT_STATE_HARD_CAP_MULTIPLE)
        );
    }

    #[test]
    fn configured_limit_overrides_the_legacy_environment_including_unbounded() {
        assert_eq!(
            exact_state_byte_limits(1024, Some(2048), Some("4096")),
            limits(1024, 2048)
        );
        assert_eq!(
            exact_state_byte_limits(1024, Some(0), Some("4096")),
            limits(1024, 0)
        );
    }
}
