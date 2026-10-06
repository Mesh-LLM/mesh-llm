use super::{
    contract::{Cohort, Input},
    measurement,
};
use crate::process::Cancellation;
use serde_json::json;
fn input() -> Input {
    serde_json::from_value(json!({"schema_version":1,"cohort":"native-serial","base_url":"http://127.0.0.1:12345/","prompt":"fixed",
        "model_id":null,"requests":3,"concurrency":1,"output_tokens":1,"request_timeout_ms":1000,"execution_timeout_ms":5000})).unwrap()
}
#[test]
fn cache_measurement_admits_distinct_serial_native_and_openai_cohorts() {
    let mut value = input();
    value.validate().unwrap();
    value.cohort = Cohort::NativeConcurrent;
    value.output_tokens = 128;
    value.validate().unwrap();
    value.cohort = Cohort::OpenaiConcurrent;
    value.base_url = "http://127.0.0.1:12345/v1".into();
    value.model_id = Some("declared-model".into());
    value.output_tokens = 32;
    value.validate().unwrap();
    value.base_url = "https://127.0.0.1:12345/v1".into();
    assert!(value.validate().is_err());
    value.base_url = "http://user:secret@127.0.0.1:12345/v1".into();
    assert!(value.validate().is_err());
    value.base_url = "http://localhost:12345/v1".into();
    assert!(value.validate().is_err());
}
#[test]
fn cache_measurement_refuses_cross_cohort_and_unbounded_workload_projection() {
    for (field, bad) in [
        ("concurrency", json!(2)),
        ("requests", json!(0)),
        ("output_tokens", json!(128)),
        ("model_id", json!("native-alias")),
        ("execution_timeout_ms", json!(0)),
        ("request_timeout_ms", json!(600001)),
        ("prompt", json!("")),
        ("base_url", json!("http://127.0.0.1:1/?credential=value")),
    ] {
        let mut value = serde_json::to_value(input()).unwrap();
        value[field] = bad;
        let value: Input = serde_json::from_value(value).unwrap();
        assert!(value.validate().is_err(), "{field}");
    }
}
#[tokio::test]
async fn cache_measurement_precancel_has_full_nonlaunched_roster_and_no_invented_summary() {
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let receipt = measurement::execute(&input(), "a".repeat(64), &cancellation).await;
    assert_eq!(receipt["status"], "incomplete");
    assert!(receipt["summary"].is_null());
    assert_eq!(receipt["rows"].as_array().unwrap().len(), 3);
    for (id, row) in receipt["rows"].as_array().unwrap().iter().enumerate() {
        assert_eq!(row["request_id"], id);
        assert_eq!(row["status"], "not-launched");
        assert!(row["evidence"].is_null());
    }
    assert_eq!(receipt["rows"][0]["excluded_warmup"], true);
}

#[test]
fn cache_measurement_terminal_admission_retains_sweep_rows_and_prior_errors() {
    for (cancelled, expired, finish_ok) in [
        (false, false, true),
        (true, false, true),
        (false, true, true),
        (false, false, false),
    ] {
        let mut receipt = json!({"status":"completed","sweep":[{"measurement":{"status":"completed","rows":[{"elapsed_ms":2.0}]}}],"error":"prior classified refusal"});
        super::finalize(&mut receipt, cancelled, expired, finish_ok);
        assert_eq!(
            receipt["status"],
            if cancelled || expired || !finish_ok {
                "incomplete"
            } else {
                "completed"
            }
        );
        assert_eq!(
            receipt["sweep"][0]["measurement"]["rows"][0]["elapsed_ms"],
            2.0
        );
        assert_eq!(receipt["error"], "prior classified refusal");
        if cancelled || expired || !finish_ok {
            assert_eq!(receipt["terminal_refusal"]["cancelled"], cancelled);
            assert_eq!(receipt["terminal_refusal"]["deadline_expired"], expired);
            assert_eq!(
                receipt["terminal_refusal"]["interrupt_finish_failed"],
                !finish_ok
            );
        }
    }
}
