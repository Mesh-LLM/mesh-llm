use super::*;
use serde_json::json;

fn input() -> serde_json::Value {
    json!({"schema_version":1,"round":1,"version":"old","base_url":"http://127.0.0.1:12345/v1",
        "model":"fixture","output_tokens":2,"request_timeout_secs":1.0,"stagger_ms":0.0,
        "prompts":[{"family":"one","prompt":"task"}]})
}

#[test]
fn request_admission_requires_local_endpoint_and_complete_positive_input() {
    serde_json::from_value::<Input>(input())
        .unwrap()
        .validate()
        .unwrap();
    for (field, value) in [
        ("base_url", json!("https://127.0.0.1:12345/v1")),
        ("base_url", json!("http://example.com:12345/v1")),
        ("base_url", json!("http://127.0.0.1/v1")),
        (
            "base_url",
            json!("http://127.0.0.1:12345/v1?unexpected=true"),
        ),
        ("schema_version", json!(2)),
        ("round", json!(0)),
        ("model", json!(" ")),
        ("output_tokens", json!(0)),
        ("request_timeout_secs", json!(0)),
        ("stagger_ms", json!(-1)),
        ("prompts", json!([])),
    ] {
        let mut document = input();
        document[field] = value;
        assert!(
            serde_json::from_value::<Input>(document)
                .unwrap()
                .validate()
                .is_err(),
            "{field}"
        );
    }
}

#[test]
fn stagger_budget_is_bounded_before_converting_it_to_a_duration() {
    let mut document = input();
    document["stagger_ms"] = json!(1e100);
    document["prompts"] =
        json!([{"family":"one","prompt":"task"},{"family":"two","prompt":"next"}]);
    assert!(
        serde_json::from_value::<Input>(document)
            .unwrap()
            .validate()
            .is_err()
    );
}

#[test]
fn cancellation_retains_every_request_identity_without_waiting_for_stagger() {
    let mut document = input();
    document["stagger_ms"] = json!(30000);
    document["prompts"] =
        json!([{"family":"one","prompt":"task"},{"family":"two","prompt":"next"}]);
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let phase = runtime
        .block_on(execute(
            serde_json::from_value(document).unwrap(),
            cancellation,
        ))
        .unwrap();
    assert_eq!(phase.requests.len(), 2);
    assert_eq!(
        phase
            .requests
            .iter()
            .map(|r| r.request_id)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    for row in phase.requests {
        let Outcome::Failed { error } = row.outcome else {
            panic!("cancelled phase succeeded")
        };
        assert!(error.contains("interrupted"));
    }
    assert!(phase.makespan_ms < 2000.0);
}
