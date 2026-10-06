use super::*;
use std::time::Duration;
fn observations() -> Value {
    json!({"schema_version":1,"cases":[{"cells":[{"requests":[{"completion_tokens":2,"content_sha256":"observed"}]}],"gate":{"passed":true}}],"requests":[{"request_id":0,"completion_tokens":2}],"error":null})
}
fn retained(given: &Value, observed: &Value) {
    assert_eq!(given["cases"], observed["cases"]);
    assert_eq!(given["requests"], observed["requests"]);
}
#[test]
fn radix_terminal_success_preserves_completed_observations() {
    let mut receipt = observations();
    let before = receipt.clone();
    finalize(
        &mut receipt,
        Ok(()),
        &Cancellation::default(),
        Instant::now() + Duration::from_secs(60),
    )
    .unwrap();
    assert_eq!(receipt["terminal_complete"], true);
    assert!(receipt["terminal_error"].is_null());
    retained(&receipt, &before);
}
#[test]
fn radix_terminal_cancel_refuses_admission_without_erasing_rows() {
    let mut receipt = observations();
    let before = receipt.clone();
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(
        finalize(
            &mut receipt,
            Ok(()),
            &cancel,
            Instant::now() + Duration::from_secs(60)
        )
        .is_err()
    );
    assert_eq!(receipt["terminal_complete"], false);
    assert_eq!(receipt["error"], "radix terminal cancellation");
    retained(&receipt, &before);
}
#[test]
fn radix_terminal_deadline_refuses_admission_without_erasing_rows() {
    let mut receipt = observations();
    let before = receipt.clone();
    assert!(
        finalize(
            &mut receipt,
            Ok(()),
            &Cancellation::default(),
            Instant::now()
        )
        .is_err()
    );
    assert_eq!(receipt["terminal_complete"], false);
    assert_eq!(receipt["error"], "radix terminal deadline expired");
    retained(&receipt, &before);
}
#[test]
fn radix_terminal_finish_failure_retains_prior_error_and_observations() {
    for prior in [None, Some("earlier cell custody failure")] {
        let mut receipt = observations();
        if let Some(reason) = prior {
            receipt["error"] = json!(reason);
        }
        let before = receipt.clone();
        assert!(
            finalize(
                &mut receipt,
                Err("private finalization detail".into()),
                &Cancellation::default(),
                Instant::now() + Duration::from_secs(60)
            )
            .is_err()
        );
        assert_eq!(receipt["terminal_complete"], false);
        assert_eq!(
            receipt["terminal_error"],
            "radix interrupt finalization failed"
        );
        assert_eq!(
            receipt["error"],
            prior.unwrap_or("radix interrupt finalization failed")
        );
        retained(&receipt, &before);
        assert!(
            !serde_json::to_string(&receipt)
                .unwrap()
                .contains("private finalization detail")
        );
    }
}
