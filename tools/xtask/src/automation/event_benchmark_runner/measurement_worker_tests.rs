use super::super::http_measurement::tests::{Fixture, Reply, body};
use super::*;
use serde_json::json;

fn input(port: u16) -> Input {
    let prompt = "Bounded paired prompt é".to_owned();
    Input {
        schema_version: 1,
        port,
        prompt_sha256: hex::encode(Sha256::digest(prompt.as_bytes())),
        prompt,
        max_tokens: 64,
        readiness_timeout_ms: 150,
        request_timeout_ms: 150,
        readiness_poll_ms: 1,
    }
}
async fn run(input: &Input, cancellation: &Cancellation) -> Evidence {
    tokio::time::timeout(Duration::from_secs(2), execute(input, cancellation))
        .await
        .unwrap()
        .unwrap()
}
#[tokio::test(flavor = "current_thread")]
async fn warmup_is_excluded_and_requests_bind_actual_model_prompt() {
    let mut fixture =
        Fixture::new(vec![Reply::models(), Reply::stream(999), Reply::stream(7)]).await;
    let input = input(fixture.port);
    let result = run(&input, &Cancellation::default()).await;
    assert!(result.error.is_none());
    assert_eq!(result.schema_version, 1);
    assert_eq!(result.model.as_deref(), Some("served-real-id"));
    assert_eq!(result.prompt_sha256, input.prompt_sha256);
    assert!(result.readiness_ms.is_some());
    assert!(result.warmup_ms.is_some());
    assert_eq!(result.measurement.unwrap().completion_tokens, Some(7));
    assert!(fixture.next().await.starts_with("GET /v1/models "));
    let warmup = body(&fixture.next().await);
    let measured = body(&fixture.next().await);
    assert_eq!(warmup, measured);
    assert_eq!(
        measured,
        json!({"model":"served-real-id", "messages":[{"role":"user","content":input.prompt}], "max_tokens":64, "temperature":0.0, "stream":true, "stream_options":{"include_usage":true}})
    );
}
#[tokio::test(flavor = "current_thread")]
async fn warmup_http_failure_does_not_discard_measurement() {
    let fixture = Fixture::new(vec![
        Reply::models(),
        Reply::Body(503, Vec::new()),
        Reply::stream(11),
    ])
    .await;
    let result = run(&input(fixture.port), &Cancellation::default()).await;
    assert!(result.error.is_none());
    assert!(result.warmup_error.unwrap().contains("503"));
    assert_eq!(result.measurement.unwrap().completion_tokens, Some(11));
}
#[tokio::test(flavor = "current_thread")]
async fn readiness_retries_missing_id_and_preserves_first_real_id() {
    let mut fixture = Fixture::new(vec![
        Reply::Body(200, br#"{"data":[]}"#.to_vec()),
        Reply::models(),
        Reply::stream(1),
        Reply::stream(2),
    ])
    .await;
    let result = run(&input(fixture.port), &Cancellation::default()).await;
    assert!(result.error.is_none());
    assert_eq!(result.model.as_deref(), Some("served-real-id"));
    assert!(fixture.next().await.starts_with("GET "));
    assert!(fixture.next().await.starts_with("GET "));
    assert_eq!(body(&fixture.next().await)["model"], "served-real-id");
}
#[tokio::test(flavor = "current_thread")]
async fn readiness_timeout_records_elapsed_without_measuring() {
    let mut fixture = Fixture::new(vec![Reply::Hold]).await;
    let mut input = input(fixture.port);
    input.readiness_timeout_ms = 20;
    let result = run(&input, &Cancellation::default()).await;
    assert!(result.error.unwrap().contains("deadline"));
    assert!(result.readiness_ms.is_some());
    assert!(result.model.is_none());
    assert!(result.warmup_ms.is_none());
    assert!(result.measurement.is_none());
    assert!(fixture.next().await.starts_with("GET "));
    assert!(fixture.empty());
}
#[tokio::test(flavor = "current_thread")]
async fn measured_timeout_keeps_warmup_and_model_evidence() {
    let fixture = Fixture::new(vec![Reply::models(), Reply::stream(5), Reply::Hold]).await;
    let mut input = input(fixture.port);
    input.request_timeout_ms = 20;
    let result = run(&input, &Cancellation::default()).await;
    assert!(result.error.unwrap().contains("deadline"));
    assert!(result.warmup_ms.is_some());
    assert_eq!(result.model.as_deref(), Some("served-real-id"));
    assert!(result.measurement.is_none());
}
#[tokio::test(flavor = "current_thread")]
async fn warmup_timeout_still_attempts_one_measurement() {
    let fixture = Fixture::new(vec![Reply::models(), Reply::Hold, Reply::stream(8)]).await;
    let mut input = input(fixture.port);
    input.request_timeout_ms = 30;
    let result = run(&input, &Cancellation::default()).await;
    assert!(result.error.is_none());
    assert!(result.warmup_error.unwrap().contains("deadline"));
    assert_eq!(result.measurement.unwrap().completion_tokens, Some(8));
}
#[tokio::test(flavor = "current_thread")]
async fn measured_malformed_usage_is_failed_with_null_metrics() {
    let fixture = Fixture::new(vec![
        Reply::models(),
        Reply::stream(2),
        Reply::Fragments(vec![
            b"data: {\"usage\":{\"completion_tokens\":\"bad\"}}\n\ndata: [DONE]\n\n".to_vec(),
        ]),
    ])
    .await;
    let result = run(&input(fixture.port), &Cancellation::default()).await;
    assert!(result.error.unwrap().contains("valid completion metrics"));
    let measured = result.measurement.unwrap();
    assert!(measured.malformed);
    assert_eq!(measured.completion_tokens, None);
    assert_eq!(measured.decode_tok_s, None);
}
#[tokio::test(flavor = "current_thread")]
async fn cancelled_before_start_never_sends_readiness() {
    let mut fixture = Fixture::new(vec![Reply::Hold]).await;
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let result = run(&input(fixture.port), &cancellation).await;
    assert!(result.error.unwrap().contains("interrupted"));
    assert!(fixture.empty());
    assert!(result.model.is_none());
}
#[tokio::test(flavor = "current_thread")]
async fn cancellation_during_warmup_does_not_start_measurement() {
    let mut fixture = Fixture::new(vec![Reply::models(), Reply::Hold]).await;
    let input = input(fixture.port);
    let cancellation = Cancellation::default();
    let cancel = async {
        assert!(fixture.next().await.starts_with("GET "));
        assert!(fixture.next().await.starts_with("POST "));
        cancellation.cancel();
    };
    let (result, ()) = tokio::join!(run(&input, &cancellation), cancel);
    assert!(result.error.unwrap().contains("interrupted"));
    assert!(result.measurement.is_none());
    assert!(fixture.empty());
}
#[tokio::test(flavor = "current_thread")]
async fn worker_evidence_serialization_preserves_bound_identity() {
    let fixture = Fixture::new(vec![Reply::models(), Reply::stream(3), Reply::stream(4)]).await;
    let input = input(fixture.port);
    let result = run(&input, &Cancellation::default()).await;
    let bytes = serde_json::to_vec(&result).unwrap();
    assert!(bytes.len() < 64 * 1024);
    let read: Evidence = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(read.schema_version, 1);
    assert_eq!(read.prompt_sha256, input.prompt_sha256);
    assert_eq!(read.measurement.unwrap().completion_tokens, Some(4));
}
#[test]
fn worker_refuses_invalid_identity_and_unbounded_budgets() {
    let mut value = input(1);
    assert!(value.validate().is_ok());
    value.prompt_sha256 = "0".repeat(64);
    assert!(value.validate().is_err());
    value = input(1);
    value.request_timeout_ms = u64::MAX;
    assert!(value.validate().is_err());
    value = input(1);
    value.schema_version = 2;
    assert!(value.validate().is_err());
    value = input(0);
    assert!(value.validate().is_err());
    value = input(1);
    value.prompt = "a".repeat(16 * 1024 + 1);
    value.prompt_sha256 = hex::encode(Sha256::digest(value.prompt.as_bytes()));
    assert!(value.validate().is_err());
}
#[test]
fn input_accepts_only_port_and_never_a_remote_network_endpoint() {
    let mut value = serde_json::to_value(input(1234)).unwrap();
    value["host"] = json!("remote.invalid");
    assert!(serde_json::from_value::<Input>(value).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_advertised_model_is_refused_before_warmup() {
    let models =
        serde_json::to_vec(&json!({"data":[{"id":"m".repeat(MAX_MODEL_BYTES + 1)}]})).unwrap();
    let mut fixture = Fixture::new(vec![Reply::Body(200, models)]).await;
    let mut input = input(fixture.port);
    input.readiness_timeout_ms = 30;
    let result = run(&input, &Cancellation::default()).await;
    assert!(result.error.is_some());
    assert!(result.model.is_none());
    assert!(result.warmup_ms.is_none());
    assert!(fixture.next().await.starts_with("GET "));
    assert!(fixture.empty());
}
#[test]
fn error_and_model_bounds_keep_serialized_receipt_below_64kib() {
    let message = bounded_error(&"\u{0001}".repeat(10_000));
    assert_eq!(message.chars().count(), 1024);
    let evidence = Evidence {
        schema_version: 1,
        model: Some("\u{0001}".repeat(MAX_MODEL_BYTES)),
        prompt_sha256: "0".repeat(64),
        warmup_error: Some(message.clone()),
        error: Some(message),
        ..Evidence::default()
    };
    assert!(serde_json::to_vec(&evidence).unwrap().len() < 64 * 1024);
}
