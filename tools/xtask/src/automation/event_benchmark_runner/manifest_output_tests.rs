use super::super::{health_log::Observation, paired_execution, stream_metrics::Measurement};
use super::*;

fn manifests(progress: u64, diagnostic: u64, missing: bool) -> [Value; 3] {
    let spec = plan::Spec {
        seed: 42,
        pairs_primary: 20,
        pairs_scenario: 10,
        scenarios: vec!["fixture".into()],
    };
    let directory = tempfile::tempdir().unwrap();
    let sides = plan::sides(
        directory.path().join("current"),
        None,
        &[plan::Mode::Production, plan::Mode::EventDisabled],
    )
    .unwrap();
    let entries = plan::build(&spec, &sides).unwrap();
    let batch = paired_execution::run(&entries, &sides, |side, _| {
        let disabled = side.mode == plan::Mode::EventDisabled;
        Ok(paired_execution::Outcome {
            launched: true,
            measurement: Some(Measurement { completion_tokens: Some(2), ttft_ms: Some(10.0), elapsed_ms: 100.0, decode_tok_s: Some(20.0), decode_only_tok_s: Some(2.0 / 0.09), malformed: false }),
            health: if missing { Observation::default() } else { Observation { health: Some(serde_json::from_value(json!({"dropped_progress":if disabled { progress } else {0},"dropped_diagnostic":if disabled { diagnostic } else {0},"state_degraded":false,"rebuild_required":false})).unwrap()), ingress_p99_us: Some(10.0) } },
            ..paired_execution::Outcome::default()
        })
    }).unwrap();
    let host = Host::classify("Darwin".into(), "arm64".into());
    let context = Context {
        spec: &spec,
        model: "/fixture/model.gguf",
        source_model_sha256: &"a".repeat(64),
        attempt: 2,
        generated_at: "2026-10-05T00:00:00Z",
        host: &host,
        thermal_state: &json!({"available":false}),
    };
    let binary = Binary {
        path: sides[0].binary.clone(),
        sha256: "b".repeat(64),
        version: Some("fixture".into()),
    };
    let production = build(
        &context,
        &sides[0],
        &binary,
        &trial_environment::effective(BTreeMap::new(), sides[0].mode),
        &batch,
        0,
    )
    .unwrap();
    let disabled = build(
        &context,
        &sides[1],
        &binary,
        &trial_environment::effective(BTreeMap::new(), sides[1].mode),
        &batch,
        1,
    )
    .unwrap();
    [production.clone(), disabled, production]
}

#[test]
fn producer_preserves_single_trial_expectations_at_full_matrix_scale() {
    let inputs = manifests(1, 0, false);
    for input in &inputs {
        assert_eq!(input["trials"].as_array().unwrap().len(), 30);
        assert_eq!(input["attempt"], 2);
        assert_eq!(input["trial_plan_algorithm"], plan::ALGORITHM);
        assert_eq!(input["executed_order"], inputs[0]["executed_order"]);
        assert_eq!(input["expected_dropped_diagnostic"], 0);
    }
    assert_eq!(inputs[1]["expected_dropped_progress"], 1);
    assert_eq!(inputs[1]["health"]["dropped_progress"], 1);
    assert_eq!(inputs[0]["expected_dropped_progress"], 0);
}

fn compare(inputs: &[Value; 3]) -> (bool, Value) {
    let directory = tempfile::tempdir().unwrap();
    let mut args = Vec::new();
    for ((flag, name), input) in [
        ("--production", "production.json"),
        ("--event-disabled", "disabled.json"),
        ("--baseline", "baseline.json"),
    ]
    .into_iter()
    .zip(inputs)
    {
        let path = directory.path().join(name);
        std::fs::write(&path, serde_json::to_vec(input).unwrap()).unwrap();
        args.extend([flag.to_string(), path.to_str().unwrap().to_string()]);
    }
    let output = directory.path().join("report.json");
    args.extend(["--output".into(), output.to_str().unwrap().into()]);
    for (flag, value) in [
        ("--bootstrap-samples", "100"),
        ("--seed", "42"),
        ("--max-degradation-percent", "5"),
        ("--min-primary-pairs", "20"),
        ("--min-scenario-pairs", "10"),
        ("--max-mdd-percent", "5"),
    ] {
        args.extend([flag.into(), value.into()]);
    }
    let admitted = crate::automation::event_benchmark_comparison::run(&args).is_ok();
    let report = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    (admitted, report)
}

#[test]
fn actual_native_producer_consumer_accepts_one_progress_drop_and_rejects_divergence() {
    assert!(compare(&manifests(1, 0, false)).0);
    for (progress, diagnostic) in [(0, 0), (2, 0), (1, 2)] {
        let (admitted, report) = compare(&manifests(progress, diagnostic, false));
        assert!(!admitted);
        assert!(
            serde_json::to_string(&report)
                .unwrap()
                .contains("expected count")
        );
    }
}

#[test]
fn missing_health_is_null_and_blocks_actual_native_consumer() {
    let inputs = manifests(0, 0, true);
    assert!(inputs[1]["health"].is_null());
    assert!(inputs[1]["callback_ingress_p99_us"].is_null());
    assert!(!compare(&inputs).0);
}

#[test]
fn certification_host_classification_retains_frozen_gates() {
    for (system, machine, expected) in [
        ("Darwin", "arm64", Some("macos-arm64-metal")),
        ("Linux", "x86_64", Some("linux-x86_64-cuda")),
        ("Windows", "AMD64", None),
        ("Linux", "aarch64", None),
    ] {
        let host = Host::classify(system.into(), machine.into());
        assert_eq!(host.certification_host, expected);
        assert_eq!(
            host.p99_gate,
            if expected.is_some() {
                "enforced"
            } else {
                "informational"
            }
        );
    }
}

#[test]
fn actual_producer_manifest_preserves_schema_mode_status_and_component_trial_wording() {
    let values = manifests(1, 0, false);
    for (value, mode) in values[..2].iter().zip(["production", "event-disabled"]) {
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["metrics_schema"], "streaming_v1");
        assert_eq!(value["mode"], mode);
        assert_eq!(
            value["trial_unit"],
            serde_json::to_value(super::super::trial_contract::unit()).unwrap()
        );
        assert_eq!(value["trials"].as_array().unwrap().len(), 30);
        assert!(
            value["trials"]
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row["status"] == "succeeded")
        );
        assert_eq!(
            value["expected_dropped_progress"],
            u64::from(mode == "event-disabled")
        );
        assert_eq!(value["expected_dropped_diagnostic"], 0);
        let encoded = serde_json::to_vec(value).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&encoded).unwrap(), *value);
    }
    assert!(compare(&values).0);
}
