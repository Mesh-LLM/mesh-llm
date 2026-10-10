//! Classification of one complete-set retry, never an execution loop.
use crate::command::DynResult;
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct Decision {
    pub attempt: u64,
    pub action: &'static str,
    pub reason: &'static str,
}

pub(super) fn classify(adverse: bool, attempt: u64) -> DynResult<Decision> {
    if attempt == 0 {
        return Err("benchmark attempt must be positive".into());
    }
    let (action, reason) = if !adverse {
        ("accept", "screen passed")
    } else if attempt == 1 {
        (
            "retry_permitted",
            "first adverse result: one full-set retry is permitted after recording and correcting a thermal/load/runtime mismatch",
        )
    } else {
        (
            "blocked_retry_exhausted",
            "second adverse result after the one predefined full-set retry: release is blocked",
        )
    };
    Ok(Decision {
        attempt,
        action,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_adverse_result_permits_only_the_predefined_complete_retry() {
        let decision = classify(true, 1).unwrap();
        assert_eq!(decision.action, "retry_permitted");
        assert!(decision.reason.contains("full-set"));
        assert!(decision.reason.contains("correcting"));
    }
    #[test]
    fn subsequent_adverse_results_block_without_creating_another_retry() {
        for attempt in [2, 3, u64::MAX] {
            assert_eq!(
                classify(true, attempt).unwrap().action,
                "blocked_retry_exhausted"
            );
        }
    }
    #[test]
    fn passing_screen_is_accepted_and_zero_attempt_is_malformed() {
        for attempt in [1, 2] {
            assert_eq!(classify(false, attempt).unwrap().action, "accept");
        }
        assert!(classify(false, 0).is_err());
    }
}
