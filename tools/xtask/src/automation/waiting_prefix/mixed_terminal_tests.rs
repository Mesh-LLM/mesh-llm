use super::*;
use std::time::Duration;
fn observation(kind: Kind) -> Value {
    match kind {
        Kind::Worker => {
            json!({"error":null,"warmup":{"completion_tokens":2},"requests":[{"request_index":0,"completion_tokens":2}],"successful_requests":1,"summary":{"scheduler_available":false},"workload_sha256":"observed"})
        }
        Kind::Cell => {
            json!({"status":"mixed_cell_admitted","identity":{"binary_sha256":"observed"},"lifecycle":{"members":[{"cleanup_complete":true}]},"requests":{"requests":[{"completion_tokens":2}]},"summary":{"scheduler_phase_qualified":true},"request_sha256":"bound"})
        }
    }
}
fn retained(before: &Value, after: &Value) {
    for (key, value) in before.as_object().unwrap() {
        if !matches!(key.as_str(), "error" | "status") {
            assert_eq!(value, &after[key], "{key}");
        }
    }
}
#[test]
fn mixed_direct_terminal_success_preserves_worker_and_cell_observations() {
    for kind in [Kind::Worker, Kind::Cell] {
        let mut value = observation(kind);
        let before = value.clone();
        finalize(
            &mut value,
            Ok(()),
            &Cancellation::default(),
            Instant::now() + Duration::from_secs(60),
            kind,
        )
        .unwrap();
        assert_eq!(value["terminal_complete"], true);
        assert!(value["terminal_error"].is_null());
        assert!(value["error"].is_null());
        retained(&before, &value);
        assert_eq!(
            value["status"],
            match kind {
                Kind::Worker => "mixed_worker_completed",
                Kind::Cell => "mixed_cell_admitted",
            }
        );
    }
}
#[test]
fn mixed_direct_terminal_cancel_refuses_worker_and_cell_without_erasing_rows() {
    let cancel = Cancellation::default();
    cancel.cancel();
    for kind in [Kind::Worker, Kind::Cell] {
        let mut value = observation(kind);
        let before = value.clone();
        assert!(
            finalize(
                &mut value,
                Ok(()),
                &cancel,
                Instant::now() + Duration::from_secs(60),
                kind
            )
            .is_err()
        );
        assert_eq!(value["terminal_complete"], false);
        assert_eq!(value["error"], "mixed terminal cancellation");
        retained(&before, &value);
        assert_eq!(
            value["status"],
            match kind {
                Kind::Worker => "mixed_worker_failed",
                Kind::Cell => "mixed_cell_failed",
            }
        );
    }
}
#[test]
fn mixed_direct_terminal_deadline_refuses_worker_and_cell_without_erasing_rows() {
    for kind in [Kind::Worker, Kind::Cell] {
        let mut value = observation(kind);
        let before = value.clone();
        assert!(
            finalize(
                &mut value,
                Ok(()),
                &Cancellation::default(),
                Instant::now(),
                kind
            )
            .is_err()
        );
        assert_eq!(value["terminal_complete"], false);
        assert_eq!(value["error"], "mixed terminal deadline exhausted");
        retained(&before, &value);
    }
}
#[test]
fn mixed_direct_terminal_finish_failure_retains_prior_error_and_observed_rows() {
    for kind in [Kind::Worker, Kind::Cell] {
        for prior in [None, Some("earlier measured protocol refusal")] {
            let mut value = observation(kind);
            if let Some(error) = prior {
                value["error"] = json!(error);
            }
            let before = value.clone();
            assert!(
                finalize(
                    &mut value,
                    Err("private finalizer details".into()),
                    &Cancellation::default(),
                    Instant::now() + Duration::from_secs(60),
                    kind
                )
                .is_err()
            );
            assert_eq!(value["terminal_complete"], false);
            assert_eq!(
                value["terminal_error"],
                "mixed interrupt finalization failed"
            );
            assert_eq!(
                value["error"],
                prior.unwrap_or("mixed interrupt finalization failed")
            );
            retained(&before, &value);
            assert!(!value.to_string().contains("private finalizer details"));
        }
    }
}
