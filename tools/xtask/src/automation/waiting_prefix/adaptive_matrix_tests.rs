use super::*;
fn input(root: &Path) -> Input {
    let arm = |version: &str| json!({"schema_version":1,"round":1,"version":version,"binary":root.join(version),"binary_sha256":"a".repeat(64),"commit":"b".repeat(40),"native_build":root.join(format!("native-{version}")),"native_build_sha256":"c".repeat(64),"native_profile":"standalone-static-skippy-server","model":root.join("model.gguf"),"model_sha256":"d".repeat(64),"model_id":"fixture-model","ctx_size":512,"split_layer":2,"layer_end":4,"n_gpu_layers":999,"adaptive_target_ms":10.0,"stage_ports":[12001,12002],"openai_port":12003});
    serde_json::from_value(json!({"schema_version":1,"old":arm("old"),"new":arm("new"),"worker":{"manifest":{"prompts":[{"family":"trace","prompt":"fixture"}]}},"rounds":4,"timeout_secs":300,"cell_timeout_secs":30,"startup_timeout_secs":2})).unwrap()
}
fn cell(scale: f64) -> Value {
    json!({"schema_version":1,"requests":{"makespan_ms":10.0*scale,"requests":[{"request_id":0,"ttft_ms":1.0*scale,"tokens_predicted":2,"prompt_sha256":"e".repeat(64),"prompt_provenance":{"family":"trace"},"content_sha256":"f".repeat(64)}]},"prefill_telemetry_available":true,"measured_prefill":[{"chunks":3,"minimum":128,"maximum":384,"elapsed_ms":5.0*scale}]})
}
#[test]
fn adaptive_matrix_injected_orchestration_alternates_exact_matched_arms_and_rounds() {
    let root = tempfile::tempdir().unwrap();
    let given = input(root.path());
    let mut seen = Vec::new();
    let output = execute_with(
        &given,
        root.path(),
        Instant::now() + Duration::from_secs(300),
        &Cancellation::default(),
        |assigned, path, _, _| {
            seen.push((assigned.arm.round, assigned.arm.version));
            assert_eq!(assigned.arm.model_sha256, given.old.model_sha256);
            assert_eq!(assigned.arm.ctx_size, 512);
            assert_eq!(
                assigned.arm.native_profile,
                "standalone-static-skippy-server"
            );
            assert_eq!(assigned.worker, given.worker);
            assert!(path.ends_with(format!(
                "round-{}-{}",
                assigned.arm.round,
                label(assigned.arm.version)
            )));
            Ok(cell(if assigned.arm.version == Version::Old {
                1.0
            } else {
                1.1
            }))
        },
    );
    assert!(output["error"].is_null(), "{output}");
    assert_eq!(
        seen,
        vec![
            (1, Version::Old),
            (1, Version::New),
            (2, Version::New),
            (2, Version::Old),
            (3, Version::Old),
            (3, Version::New),
            (4, Version::New),
            (4, Version::Old)
        ]
    );
    assert_eq!(output["cells"].as_array().unwrap().len(), 8);
    assert_eq!(output["comparison"]["output_parity"]["exact_matches"], 4);
    assert_eq!(
        output["metadata"]["discarded_calibration_requests_per_cell"],
        1
    );
    assert_eq!(output["metadata"]["old"]["commit"], "b".repeat(40));
    root.close().unwrap();
}
#[test]
fn adaptive_matrix_injected_failure_missing_prefill_cancel_and_budget_retain_partial_cells() {
    let root = tempfile::tempdir().unwrap();
    let given = input(root.path());
    for mode in ["failure", "missing-prefill", "cancel"] {
        let cancel = Cancellation::default();
        let mut count = 0;
        let output = execute_with(
            &given,
            root.path(),
            Instant::now() + Duration::from_secs(300),
            &cancel,
            |_, _, _, _| {
                count += 1;
                if count == 2 {
                    match mode {
                        "failure" => return Err("fixture cell refusal".into()),
                        "missing-prefill" => {
                            let mut value = cell(1.0);
                            value["measured_prefill"] = json!([]);
                            return Ok(value);
                        }
                        "cancel" => cancel.cancel(),
                        _ => unreachable!(),
                    }
                }
                Ok(cell(1.0))
            },
        );
        assert!(output["error"].is_string());
        assert!(!output["cells"].as_array().unwrap().is_empty());
        assert!(count <= 2);
        assert!(output["comparison"].is_null());
    }
    let output = execute_with(
        &given,
        root.path(),
        Instant::now() + Duration::from_secs(12),
        &Cancellation::default(),
        |_, _, _, _| panic!("expired arm must not launch"),
    );
    assert!(output["error"].is_string());
    assert!(output["cells"].as_array().unwrap().is_empty());
    let mut invalid = input(root.path());
    invalid.new.ctx_size = 513;
    assert!(invalid.validate().is_err());
    invalid = input(root.path());
    invalid.new.model_sha256 = "9".repeat(64);
    assert!(invalid.validate().is_err());
    root.close().unwrap();
}
#[test]
fn adaptive_paired_median_report_is_deterministic_nullable_and_refuses_incomplete_or_mismatched_pairs()
 {
    let mut cells = Vec::new();
    for round in 1..=4 {
        for version in ["old", "new"] {
            let mut row = cell(if version == "old" { 1.0 } else { 1.1 });
            row["round"] = json!(round);
            row["version"] = json!(version);
            row["summary"] = summary::summarize(&row).unwrap();
            cells.push(row);
        }
    }
    let first = summary::compare(&cells, 4).unwrap();
    assert_eq!(first, summary::compare(&cells, 4).unwrap());
    let interval = &first["paired_delta_percent"]["ttft_ms_p95"];
    assert!((interval["median"].as_f64().unwrap() - 10.0).abs() < 1e-9);
    assert_eq!(interval["round_deltas"].as_array().unwrap().len(), 4);
    let mut nullable = first.clone();
    nullable["aggregate"]["old"]["ttft_ms_p95"] = Value::Null;
    nullable["aggregate"]["new"]["prefill_elapsed_ms_p95"] = Value::Null;
    assert!(summary::render(&nullable).contains("n/a"));
    let mut bad = cells.clone();
    bad[1]["requests"]["requests"][0]["prompt_sha256"] = json!("0".repeat(64));
    assert!(summary::compare(&bad, 4).is_err());
    assert!(summary::compare(&cells[..7], 4).is_err());
    let mut bad = cells.clone();
    bad[1]["requests"]["requests"][0]["content_sha256"] = json!("0".repeat(64));
    let mismatched = summary::compare(&bad, 4).unwrap();
    assert_eq!(mismatched["output_parity"]["exact_matches"], 3);
    assert_eq!(mismatched["output_parity"]["mismatches"][0]["round"], 1);
    let mut invalid = cell(1.0);
    invalid["requests"]["makespan_ms"] = json!(0);
    assert!(summary::summarize(&invalid).is_err());
    invalid = cell(1.0);
    invalid["requests"]["requests"][0]["tokens_predicted"] = Value::Null;
    assert!(summary::summarize(&invalid).is_err());
}
#[test]
fn adaptive_optional_calibration_projects_only_finite_numeric_allowlist_and_keeps_latest() {
    let mut observation = super::super::adaptive_telemetry::Observation::default();
    for value in [1.0, 2.0] {
        observation.observe(serde_json::to_vec(&json!({"event":"stage.openai_prefill_calibration","attributes":{"llama_stage.prefill_bottleneck_compute_ms":value,"secret":"never-retained"}})).unwrap().as_slice());
    }
    let projected = serde_json::to_value(&observation.latest_calibration).unwrap();
    assert_eq!(projected["llama_stage.prefill_bottleneck_compute_ms"], 2.0);
    assert!(projected.get("secret").is_none());
    assert!(observation.error.is_none());
    observation.observe(serde_json::to_vec(&json!({"event":"stage.openai_prefill_calibration","attributes":{"llama_stage.prefill_bottleneck_compute_ms":-1.0}})).unwrap().as_slice());
    assert!(observation.error.is_some());
}
