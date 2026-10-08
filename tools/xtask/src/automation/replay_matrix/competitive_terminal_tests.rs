use super::*;
use std::time::Duration;
fn observation() -> Value {
    json!({"completed":true,"passed":true,"cell":{"arm":"mesh"},"requests":1,"successful_requests":1,"completion_tokens":8,"requests_sha256":"observed","launch_provenance":{"model_sha256":"bound"},"parity":{"results":[{"completion_tokens":32}]},"measured_wall_seconds":0.1})
}
fn retained(before: &Value, after: &Value) {
    for (key, value) in before.as_object().unwrap() {
        if !["completed", "passed", "error"].contains(&key.as_str()) {
            assert_eq!(value, &after[key], "{key}");
        }
    }
}
#[test]
fn competitive_terminal_success_preserves_observed_worker_receipts() {
    let mut value = observation();
    let before = value.clone();
    finalize(
        &mut value,
        true,
        &Cancellation::default(),
        Instant::now() + Duration::from_secs(60),
    )
    .unwrap();
    assert_eq!(value["completed"], true);
    assert_eq!(value["passed"], true);
    assert_eq!(value["terminal_complete"], true);
    assert!(value["terminal_error"].is_null());
    retained(&before, &value);
}
#[test]
fn competitive_terminal_cancel_refuses_success_without_erasing_observations() {
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let mut value = observation();
    let before = value.clone();
    assert!(
        finalize(
            &mut value,
            true,
            &cancellation,
            Instant::now() + Duration::from_secs(60)
        )
        .is_err()
    );
    assert_eq!(value["completed"], false);
    assert_eq!(value["passed"], false);
    assert_eq!(value["terminal_complete"], false);
    assert_eq!(value["error"], "competitive terminal cancellation");
    retained(&before, &value);
}
#[test]
fn competitive_terminal_deadline_refuses_success_without_erasing_observations() {
    let mut value = observation();
    let before = value.clone();
    assert!(finalize(&mut value, true, &Cancellation::default(), Instant::now()).is_err());
    assert_eq!(value["completed"], false);
    assert_eq!(value["passed"], false);
    assert_eq!(
        value["terminal_error"],
        "competitive terminal deadline exhausted"
    );
    retained(&before, &value);
}
#[test]
fn competitive_terminal_finish_refusal_preserves_prior_error_and_measured_rows() {
    for prior in [None, Some("earlier request refusal")] {
        let mut value = observation();
        if let Some(error) = prior {
            value["error"] = json!(error);
        }
        let before = value.clone();
        assert!(
            finalize(
                &mut value,
                false,
                &Cancellation::default(),
                Instant::now() + Duration::from_secs(60)
            )
            .is_err()
        );
        assert_eq!(value["completed"], false);
        assert_eq!(value["terminal_complete"], false);
        assert_eq!(
            value["error"],
            prior.unwrap_or("competitive interrupt finalization failed")
        );
        retained(&before, &value);
    }
}
