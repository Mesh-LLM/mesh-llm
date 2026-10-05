use super::{acceptance::*, *};

fn measured_input(version: &str) -> Value {
    serde_json::json!({
        "round": 1, "version": version, "makespan_ms": 10.0,
        "requests": [{"request_id": 0, "family": "one", "first_token_ms": 5.0,
            "ttft_ms": 5.0, "tokens_predicted": 2, "cached_tokens": 0}],
        "events": [{"attributes": {"skippy.kv.status": "miss", "skippy.kv.suffix_prefill_tokens": 10}}],
        "capacity_events": [{"attributes": {"skippy.kv.capacity_status": "evicted",
            "skippy.kv.capacity_evicted_tokens": 6, "skippy.kv.capacity_evicted_entries": 1,
            "skippy.kv.capacity_predicted_recompute_cost": 24}}],
        "record_events": [{"attributes": {"skippy.kv.decision": "proactive_eviction",
            "skippy.kv.proactive_evicted_tokens": 4, "skippy.kv.proactive_evicted_entries": 1}}],
    })
}

#[test]
fn request_summary_combines_capacity_and_proactive_evictions() {
    let input = serde_json::from_value(measured_input("new")).unwrap();
    let cell = telemetry::summarize(input).unwrap();
    assert_eq!(cell.summary.resident_evicted_tokens_total, Some(10.0));
    assert_eq!(cell.summary.resident_evicted_entries_total, Some(2.0));
    assert_eq!(cell.summary.predicted_recompute_cost_total, Some(24.0));
    assert_eq!(cell.summary.capacity_rejections, 0);
    assert_eq!(cell.summary.output_tokens_per_second, 200.0);
    assert_eq!(cell.summary.ttft_ms_p50, Some(5.0));
}

#[test]
fn absent_legacy_measurements_stay_unknown_and_bad_measurements_are_rejected() {
    let mut input = measured_input("old");
    input["capacity_events"] = serde_json::json!([]);
    input["events"] = serde_json::json!([]);
    let cell = telemetry::summarize(serde_json::from_value(input.clone()).unwrap()).unwrap();
    assert_eq!(cell.summary.predicted_recompute_cost_total, None);
    assert_eq!(cell.summary.suffix_prefill_tokens_total, None);
    assert_eq!(cell.summary.resident_evicted_tokens_total, Some(4.0));
    input["record_events"][0]["attributes"]["skippy.kv.proactive_evicted_tokens"] =
        serde_json::json!(-1);
    assert!(telemetry::summarize(serde_json::from_value(input).unwrap()).is_err());
}

#[test]
fn round_aggregation_preserves_failed_percentiles_and_refuses_duplicate_rounds() {
    let mut before = measured_input("old");
    before["requests"] =
        serde_json::json!([{"request_id": 0, "family": "one", "error": "no first token"}]);
    let before = telemetry::summarize(serde_json::from_value(before).unwrap()).unwrap();
    let after =
        telemetry::summarize(serde_json::from_value(measured_input("new")).unwrap()).unwrap();
    let input = serde_json::json!({"cells": [before, after]});
    let rows = aggregation::aggregate(serde_json::from_value(input.clone()).unwrap()).unwrap();
    assert_eq!(rows[0].successful, 0);
    assert_eq!(rows[0].ttft_ms_p50_median, None);
    assert_eq!(rows[1].ttft_ms_p50_median, Some(5.0));
    let acceptance = acceptance::evaluate(&rows, &fixture_contract("warm-affinity")).unwrap();
    assert!(!acceptance.passed);
    assert!(
        report::render(&rows, &acceptance)
            .unwrap()
            .contains("| Client TTFT p50 ms | n/a | 5.0 | n/a |")
    );
    let duplicated = serde_json::json!({"cells": [input["cells"][0], input["cells"][0]]});
    assert!(aggregation::aggregate(serde_json::from_value(duplicated).unwrap()).is_err());
}

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_contract(profile: &str) -> Contract {
    let input: Value = serde_json::from_slice(
        &std::fs::read(root().join("skippy/evals/skippy-scheduler-fixtures.json")).unwrap(),
    )
    .unwrap();
    serde_json::from_value(
        fixture_profile::resolve(&input, profile).unwrap()["hardware_acceptance"].clone(),
    )
    .unwrap()
}

fn row(version: Version, successful: u64, values: [f64; 6]) -> Aggregate {
    Aggregate {
        version,
        rounds: 4,
        requests: successful,
        successful,
        capacity_rejections: 0,
        cache_hits_median: Some(1.0),
        suffix_prefill_tokens_median: Some(values[0]),
        family_switches_median: Some(values[1]),
        ttft_ms_p50_median: Some(values[2]),
        ttft_ms_p95_median: Some(values[3]),
        makespan_ms_median: Some(values[4]),
        output_tokens_per_second_median: Some(values[5]),
        resident_evicted_tokens_median: Some(0.0),
        resident_evicted_entries_median: Some(0.0),
        predicted_recompute_cost_median: None,
    }
}

#[test]
fn warm_profile_preserves_neutral_work_and_bounded_user_metrics() {
    let contract = fixture_contract("warm-affinity");
    let mut rows = vec![
        row(
            Version::Old,
            24,
            [8237.0, 1.0, 3161.1, 3313.7, 5326.8, 71.34],
        ),
        row(
            Version::New,
            24,
            [8237.0, 1.0, 3190.0, 3305.4, 5331.8, 71.27],
        ),
    ];
    assert!(acceptance::evaluate(&rows, &contract).unwrap().passed);
    rows[1].suffix_prefill_tokens_median = Some(8238.0);
    assert!(!acceptance::evaluate(&rows, &contract).unwrap().passed);
    rows[1].suffix_prefill_tokens_median = rows[0].suffix_prefill_tokens_median;
    rows[1].ttft_ms_p95_median = Some(4000.0);
    assert!(!acceptance::evaluate(&rows, &contract).unwrap().passed);
}

#[test]
fn pressure_requires_measured_baseline_complete_success_and_all_gains() {
    let contract = fixture_contract("agentic-eviction-pressure");
    let mut rows = vec![
        row(
            Version::Old,
            64,
            [92061.5, 14.0, 30000.0, 38646.5, 41132.9, 12.02],
        ),
        row(
            Version::New,
            64,
            [66735.5, 10.0, 20000.0, 27901.1, 30328.4, 16.26],
        ),
    ];
    assert!(acceptance::evaluate(&rows, &contract).unwrap().passed);
    rows[0].suffix_prefill_tokens_median = Some(49999.0);
    assert!(!acceptance::evaluate(&rows, &contract).unwrap().passed);
    rows[0].suffix_prefill_tokens_median = Some(92061.5);
    rows[1].output_tokens_per_second_median = rows[0].output_tokens_per_second_median;
    assert!(!acceptance::evaluate(&rows, &contract).unwrap().passed);
    rows[1].output_tokens_per_second_median = Some(16.26);
    rows[1].requests += 1;
    assert!(!acceptance::evaluate(&rows, &contract).unwrap().passed);
}

#[test]
fn capacity_bounds_allow_unknown_legacy_cost_but_require_measured_after_cost() {
    let options = BTreeMap::from([("--contract", "skippy/evals/skippy-capacity-acceptance.json")]);
    let path = root().join(options["--contract"]);
    let path = path.to_str().unwrap();
    let contract = contract(&BTreeMap::from([("--contract", path)])).unwrap();
    let mut rows = vec![
        row(
            Version::Old,
            64,
            [66735.5, 10.0, 100.0, 27901.1, 30328.4, 16.26],
        ),
        row(
            Version::New,
            64,
            [65000.0, 10.0, 100.0, 27500.0, 30000.0, 16.5],
        ),
    ];
    rows[1].resident_evicted_tokens_median = Some(1500.0);
    rows[1].predicted_recompute_cost_median = Some(42000.0);
    assert!(acceptance::evaluate(&rows, &contract).unwrap().passed);
    rows[1].capacity_rejections = 1;
    assert!(!acceptance::evaluate(&rows, &contract).unwrap().passed);
    rows[1].capacity_rejections = 0;
    rows[1].predicted_recompute_cost_median = None;
    assert!(!acceptance::evaluate(&rows, &contract).unwrap().passed);
}

#[test]
fn missing_zero_or_nonfinite_comparisons_fail_closed() {
    assert_eq!(delta(Some(0.0), Some(0.0)), Some(0.0));
    for (old, new) in [
        (Some(0.0), Some(1.0)),
        (None, Some(5.0)),
        (Some(f64::NAN), Some(1.0)),
        (Some(1.0), Some(f64::INFINITY)),
    ] {
        assert_eq!(delta(old, new), None);
    }
    let contract = fixture_contract("warm-affinity");
    let rows = vec![
        row(Version::Old, 24, [0.0; 6]),
        row(Version::New, 24, [1.0; 6]),
    ];
    assert!(!acceptance::evaluate(&rows, &contract).unwrap().passed);
    assert!(pair(&rows[..1]).is_err());
    assert!(pair(&[rows[0].clone(), rows[0].clone()]).is_err());
    assert!(
        serde_json::from_value::<Contract>(
            serde_json::json!({"successful_requests_per_binary": 24, "unknown_bound": 1})
        )
        .is_err()
    );
}

#[test]
fn prompt_input_preserves_order_and_metadata_and_rejects_empty_or_malformed_entries() {
    let document = serde_json::json!({"metadata": {"dataset_revision": "abc123"}, "prompts": [{"family": "one", "prompt": "shared one"}, {"family": "two", "prompt": "shared two"}]});
    let decoded = prompt_manifest(&serde_json::to_vec(&document).unwrap()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), document);
    for rejected in [
        serde_json::json!({"prompts": []}),
        serde_json::json!({"prompts": [{"family": "one", "prompt": ""}]}),
        serde_json::json!({"metadata": [], "prompts": [{"family": "one", "prompt": "text"}]}),
    ] {
        assert!(prompt_manifest(&serde_json::to_vec(&rejected).unwrap()).is_err());
    }
}

#[test]
fn unavailable_measured_percentiles_remain_nullable_in_json_and_report() {
    let contract = fixture_contract("warm-affinity");
    let mut rows = vec![
        row(Version::Old, 24, [1.0; 6]),
        row(Version::New, 24, [1.0; 6]),
    ];
    rows[0].ttft_ms_p50_median = None;
    rows[1].ttft_ms_p50_median = Some(5.0);
    let acceptance = acceptance::evaluate(&rows, &contract).unwrap();
    assert!(!acceptance.passed);
    assert!(
        report::render(&rows, &acceptance)
            .unwrap()
            .contains("| Client TTFT p50 ms | n/a | 5.0 | n/a |")
    );
    assert!(serde_json::to_value(&rows).unwrap()[0]["ttft_ms_p50_median"].is_null());
}

#[test]
fn offline_command_writes_failure_evidence_and_preserves_output_on_invalid_input() {
    let temporary = tempfile::tempdir().unwrap();
    let comparison = temporary.path().join("comparison.json");
    let output = temporary.path().join("acceptance.json");
    let report = temporary.path().join("report.md");
    let rows = [
        row(Version::Old, 64, [1.0; 6]),
        row(Version::New, 64, [1.0; 6]),
    ];
    std::fs::write(
        &comparison,
        serde_json::to_vec(&serde_json::json!({"aggregate": rows})).unwrap(),
    )
    .unwrap();
    let args: Vec<String> = [
        "evaluate".into(),
        "--comparison".into(),
        comparison.to_str().unwrap().into(),
        "--output".into(),
        output.to_str().unwrap().into(),
        "--report".into(),
        report.to_str().unwrap().into(),
        "--contract".into(),
        root()
            .join("skippy/evals/skippy-capacity-acceptance.json")
            .to_str()
            .unwrap()
            .into(),
    ]
    .into();
    assert!(run(&args).is_err());
    let evidence: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(evidence["passed"], false);
    assert!(
        std::fs::read_to_string(&report)
            .unwrap()
            .contains("Fixture acceptance: **FAIL**")
    );
    std::fs::write(&comparison, b"{}").unwrap();
    std::fs::write(&output, b"preserve").unwrap();
    assert!(run(&args).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"preserve");
}
