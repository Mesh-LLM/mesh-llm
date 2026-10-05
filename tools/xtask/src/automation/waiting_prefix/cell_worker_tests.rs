use super::*;
use serde_json::{Value, json};

fn input() -> Value {
    json!({"schema_version":1,"server_log":std::env::temp_dir().join("owned-server.log"),
        "startup_timeout_secs":1,"telemetry_timeout_secs":1,
        "phase":{"schema_version":1,"round":2,"version":"new",
            "base_url":"http://127.0.0.1:12345/v1","model":"fixture","output_tokens":2,
            "request_timeout_secs":1.0,"stagger_ms":0.0,
            "prompts":[{"family":"family-0","prompt":"measured task"}]},
        "cache_seed":{"families":2,"prefix_blocks":2,"output_tokens":1,"stagger_ms":1.0}})
}

#[test]
fn seed_phase_preserves_server_identity_and_uses_only_its_own_workload_shape() {
    let input: Input = serde_json::from_value(input()).unwrap();
    let seed = input.seed_phase().unwrap().unwrap();
    input.validate(Some(&seed)).unwrap();
    assert_eq!(seed.model, input.phase.model);
    assert_eq!(seed.base_url, input.phase.base_url);
    assert_eq!(seed.round, 2);
    assert_eq!(seed.version, super::super::acceptance::Version::New);
    assert_eq!(seed.prompts.len(), 2);
    assert_eq!(seed.output_tokens, 1);
    assert_eq!(seed.stagger_ms, 1.0);
    assert_eq!(input.phase.prompts.len(), 1);
    assert_eq!(input.phase.output_tokens, 2);
}

#[test]
fn cell_admission_bounds_combined_readiness_seed_request_and_telemetry_deadlines() {
    for (field, value) in [
        ("schema_version", json!(2)),
        ("startup_timeout_secs", json!(0)),
        ("telemetry_timeout_secs", json!(0)),
        ("telemetry_timeout_secs", json!(86400)),
        ("server_log", json!("relative.log")),
    ] {
        let mut document = input();
        document[field] = value;
        let input: Input = serde_json::from_value(document).unwrap();
        let seed = input.seed_phase().unwrap();
        assert!(input.validate(seed.as_ref()).is_err(), "{field}");
    }
    let mut document = input();
    document["cache_seed"]["families"] = json!(0);
    assert!(
        serde_json::from_value::<Input>(document)
            .unwrap()
            .seed_phase()
            .is_err()
    );
}

#[test]
fn preexisting_cancellation_cannot_issue_readiness_or_seed_requests() {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut document = input();
    document["phase"]["base_url"] = json!(format!("http://{}/v1", listener.local_addr().unwrap()));
    let input: Input = serde_json::from_value(document).unwrap();
    let seed = input.seed_phase().unwrap();
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut evidence = Evidence::new(&input);
    let error = runtime
        .block_on(measure(&input, seed, &cancellation, &mut evidence))
        .unwrap_err();
    assert!(error.to_string().contains("interrupted"));
    assert!(evidence.cache_seed.is_none());
    assert!(evidence.measurement.is_none());
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}
