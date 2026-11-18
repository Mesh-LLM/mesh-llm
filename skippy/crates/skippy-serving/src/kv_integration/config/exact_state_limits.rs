//! Exact-state snapshot retention limits, including indivisible payloads.
use super::super::ExactStateByteLimits;

// That retention floor is not allowed to grow without bound. Once the catalog
// crosses this multiple of the soft cap it evicts again, down to the single
// entry that keeps exact prefix reuse alive for the stage. That indivisible
// entry may itself exceed the limit.
const EXACT_STATE_HARD_CAP_MULTIPLE: u64 = 8;

// The smallest working set the soft cap keeps, whatever the lane count.
pub(super) const EXACT_STATE_MIN_RETAINED_ENTRIES: usize = 4;

/// One snapshot per lane, at least the minimum, never past the entry limit.
pub(super) fn exact_state_working_set(lane_count: u32, max_entries: usize) -> usize {
    EXACT_STATE_MIN_RETAINED_ENTRIES
        .max(lane_count as usize)
        .min(max_entries)
}

/// Resolves the exact-state byte budget from the stage cache cap.
///
/// `configured_max_bytes` is the stage budget, used as the soft cap. Unless an
/// operator sets it, it covers half a context window of the stage's attention
/// KV and one recurrent snapshot per lane, so a long conversation's snapshot
/// can still exceed it. The hard limit defaults to a multiple of it so the
/// working set stays bounded. One indivisible snapshot may remain above that
/// limit. `configured_exact_max_bytes` wins over the legacy environment
/// override when provided; zero means unbounded on either surface.
pub(super) fn exact_state_byte_limits(
    configured_max_bytes: u64,
    configured_exact_max_bytes: Option<u64>,
    override_max_bytes: Option<&str>,
    working_set_entries: usize,
) -> ExactStateByteLimits {
    let hard_bytes = configured_exact_max_bytes
        .or_else(|| override_max_bytes.and_then(|value| value.trim().parse::<u64>().ok()))
        .unwrap_or_else(|| configured_max_bytes.saturating_mul(EXACT_STATE_HARD_CAP_MULTIPLE));
    ExactStateByteLimits {
        soft_bytes: configured_max_bytes,
        hard_bytes,
        working_set_entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits(soft_bytes: u64, hard_bytes: u64) -> ExactStateByteLimits {
        ExactStateByteLimits {
            soft_bytes,
            hard_bytes,
            working_set_entries: EXACT_STATE_MIN_RETAINED_ENTRIES,
        }
    }

    #[test]
    fn the_working_set_holds_one_snapshot_per_lane_within_the_entry_limit() {
        assert_eq!(
            exact_state_working_set(1, 64),
            EXACT_STATE_MIN_RETAINED_ENTRIES
        );
        assert_eq!(exact_state_working_set(16, 64), 16);
        assert_eq!(exact_state_working_set(16, 8), 8);
        assert_eq!(exact_state_working_set(1, 2), 2);
    }

    #[test]
    fn exact_state_byte_limits_scale_the_ceiling_off_the_stage_budget() {
        assert_eq!(
            exact_state_byte_limits(1_024, None, None, EXACT_STATE_MIN_RETAINED_ENTRIES),
            limits(1_024, 1_024 * EXACT_STATE_HARD_CAP_MULTIPLE)
        );
        // An unset stage budget stays unbounded, as it was before.
        assert_eq!(
            exact_state_byte_limits(0, None, None, EXACT_STATE_MIN_RETAINED_ENTRIES),
            limits(0, 0)
        );
    }

    #[test]
    fn exact_state_byte_limits_honour_the_operator_override() {
        assert_eq!(
            exact_state_byte_limits(
                1_024,
                None,
                Some(" 4096 "),
                EXACT_STATE_MIN_RETAINED_ENTRIES
            ),
            limits(1_024, 4_096)
        );
        assert_eq!(
            exact_state_byte_limits(1_024, None, Some("0"), EXACT_STATE_MIN_RETAINED_ENTRIES),
            limits(1_024, 0)
        );
        // A malformed override must not silently disable the ceiling.
        assert_eq!(
            exact_state_byte_limits(
                1_024,
                None,
                Some("not-a-number"),
                EXACT_STATE_MIN_RETAINED_ENTRIES
            ),
            limits(1_024, 1_024 * EXACT_STATE_HARD_CAP_MULTIPLE)
        );
    }

    #[test]
    fn configured_limit_overrides_the_legacy_environment_including_unbounded() {
        assert_eq!(
            exact_state_byte_limits(
                1024,
                Some(2048),
                Some("4096"),
                EXACT_STATE_MIN_RETAINED_ENTRIES
            ),
            limits(1024, 2048)
        );
        assert_eq!(
            exact_state_byte_limits(
                1024,
                Some(0),
                Some("4096"),
                EXACT_STATE_MIN_RETAINED_ENTRIES
            ),
            limits(1024, 0)
        );
    }
}
