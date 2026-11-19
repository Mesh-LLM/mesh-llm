//! Exact-state snapshot retention limits, including indivisible payloads.
use super::super::ExactStateByteLimits;

/// Resolves the exact-state byte budget from the stage cache cap.
///
/// `configured_max_bytes` is the attention-derived stage budget, which is used
/// as the soft cap. The hard limit defaults to a multiple of it so the working
/// set stays bounded without inheriting an estimate that structurally
/// undercounts exact-state payloads. `configured_exact_max_bytes` wins over
/// the legacy environment override when provided; zero means unbounded on
/// either surface. Whether the hard limit was explicitly provided is carried
/// alongside the value, because an explicit ceiling is operator truth: the
/// retention floor must keep only the documented single indivisible snapshot
/// above it, while an automatic estimate keeps the concurrent-session working
/// set.
pub(super) fn exact_state_byte_limits(
    configured_max_bytes: u64,
    configured_exact_max_bytes: Option<u64>,
    override_max_bytes: Option<&str>,
) -> ExactStateByteLimits {
    let explicit_hard_bytes = configured_exact_max_bytes
        .or_else(|| override_max_bytes.and_then(|value| value.trim().parse::<u64>().ok()));
    match explicit_hard_bytes {
        Some(hard_bytes) => ExactStateByteLimits {
            soft_bytes: configured_max_bytes,
            hard_bytes,
            hard_limit_explicit: true,
        },
        None => ExactStateByteLimits {
            soft_bytes: configured_max_bytes,
            hard_bytes: configured_max_bytes.saturating_mul(EXACT_STATE_HARD_CAP_MULTIPLE),
            hard_limit_explicit: false,
        },
    }
}

/// The hard limit derives from this multiple of the soft cap when no exact
/// budget is configured. The soft cap undercounts exact snapshots, so the
/// multiple keeps the automatic ceiling from binding inside one snapshot's
/// own footprint.
const EXACT_STATE_HARD_CAP_MULTIPLE: u64 = 8;

#[cfg(test)]
mod tests {
    use super::*;
    fn limits(soft_bytes: u64, hard_bytes: u64) -> ExactStateByteLimits {
        ExactStateByteLimits {
            soft_bytes,
            hard_bytes,
            hard_limit_explicit: false,
        }
    }
    fn explicit(soft_bytes: u64, hard_bytes: u64) -> ExactStateByteLimits {
        ExactStateByteLimits {
            soft_bytes,
            hard_bytes,
            hard_limit_explicit: true,
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
            explicit(1_024, 4_096)
        );
        assert_eq!(
            exact_state_byte_limits(1_024, None, Some("0")),
            explicit(1_024, 0)
        );
        // A malformed override must not silently disable the ceiling: it
        // falls back to the automatic estimate, which keeps its working set.
        assert_eq!(
            exact_state_byte_limits(1_024, None, Some("not-a-number")),
            limits(1_024, 1_024 * EXACT_STATE_HARD_CAP_MULTIPLE)
        );
    }

    #[test]
    fn configured_limit_overrides_the_legacy_environment_including_unbounded() {
        assert_eq!(
            exact_state_byte_limits(1024, Some(2048), Some("4096")),
            explicit(1024, 2048)
        );
        assert_eq!(
            exact_state_byte_limits(1024, Some(0), Some("4096")),
            explicit(1024, 0)
        );
    }
}
