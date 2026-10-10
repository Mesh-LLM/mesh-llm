//! Final local receipt admission; measurement observations survive terminal refusal.
use serde_json::{Value, json};
use std::time::Instant;

pub(super) fn admit(
    mut report: Value,
    restored: bool,
    cancelled: bool,
    deadline: Instant,
) -> Value {
    let within_deadline = Instant::now() < deadline;
    report["terminal"] = json!({
        "signal_scope_restored": restored,
        "cancelled": cancelled,
        "within_deadline": within_deadline,
        "scope": "local_final_check_before_receipt_write"
    });
    let refusal = if !restored {
        Some("signal_scope_finish_failed")
    } else if cancelled {
        Some("cancelled")
    } else if !within_deadline {
        Some("deadline_exhausted")
    } else {
        None
    };
    if let Some(reason) = refusal {
        report["passed"] = false.into();
        report["terminal_refusal"] = reason.into();
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn complete() -> Value {
        json!({
            "schema_version": 1, "passed": true,
            "cases": [{"status": 200}, {"status": 402}, {"status": 200}],
            "identity": {"model": {"sha256": "observed-model-digest"}},
            "source_unchanged": true,
            "members": [{"cleanup_complete": true, "cleanup_forced": false}]
        })
    }
    fn preserved(before: &Value, after: &Value) {
        for field in ["cases", "identity", "source_unchanged", "members"] {
            assert_eq!(before[field], after[field]);
        }
    }
    fn later() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    #[test]
    fn terminal_success_preserves_complete_and_failed_observations() {
        let before = complete();
        let after = admit(before.clone(), true, false, later());
        preserved(&before, &after);
        assert_eq!(after["passed"], true);
        assert!(after.get("terminal_refusal").is_none());
        assert_eq!(after["terminal"]["signal_scope_restored"], true);
        assert_eq!(after["terminal"]["cancelled"], false);
        assert_eq!(after["terminal"]["within_deadline"], true);
        let mut failed = before;
        failed["passed"] = false.into();
        assert_eq!(admit(failed, true, false, later())["passed"], false);
    }

    #[test]
    fn terminal_finish_error_downgrades_success_without_losing_observations() {
        let before = complete();
        let after = admit(before.clone(), false, false, later());
        preserved(&before, &after);
        assert_eq!(after["passed"], false);
        assert_eq!(after["terminal_refusal"], "signal_scope_finish_failed");
        assert_eq!(after["terminal"]["signal_scope_restored"], false);
    }

    #[test]
    fn terminal_cancellation_downgrades_success_without_losing_observations() {
        let before = complete();
        let after = admit(before.clone(), true, true, later());
        preserved(&before, &after);
        assert_eq!(after["passed"], false);
        assert_eq!(after["terminal_refusal"], "cancelled");
        assert_eq!(after["terminal"]["cancelled"], true);
    }

    #[test]
    fn terminal_deadline_downgrades_success_without_losing_observations() {
        let before = complete();
        // Equality is exhausted: no delay or scheduler-dependent deadline fixture.
        let after = admit(before.clone(), true, false, Instant::now());
        preserved(&before, &after);
        assert_eq!(after["passed"], false);
        assert_eq!(after["terminal_refusal"], "deadline_exhausted");
        assert_eq!(after["terminal"]["within_deadline"], false);
    }
}
