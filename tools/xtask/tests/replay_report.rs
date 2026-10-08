mod replay_report_support;
use replay_report_support::{document, run};

#[test]
fn columns_and_values_survive_csv_quoting() {
    let artifact = tempfile::tempdir().unwrap();
    let input = document("arm,\"quoted\"\nline");
    std::fs::write(
        artifact.path().join("run.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    assert!(run(artifact.path()).status.success());
    let csv = std::fs::read_to_string(artifact.path().join("summary/comparison.csv")).unwrap();
    let records = replay_report_support::csv_records(&csv);
    let header = &records[0];
    assert_eq!(
        header.join(","),
        "label,ref,commit,concurrency,passes,trajectories_per_pass,trajectory_replays,requests,successful_requests,failed_requests,failed_request_ids,failure_identity_known,content_identity,content_identity_known,content_stable_across_passes,prompt_tokens_min,prompt_tokens_max,success_pct,budget_exhausted_pct,agent_steps_per_second,agent_steps_per_second_min,agent_steps_per_second_max,workload_output_tokens_per_second,workload_output_tokens_per_second_min,workload_output_tokens_per_second_max,decode_tokens_per_second,decode_tokens_per_second_min,decode_tokens_per_second_max,ttft_p50_seconds,ttft_p50_seconds_min,ttft_p50_seconds_max,ttft_p95_seconds,mean_in_flight,concurrency_utilization_pct,cache_pct,delta_comparable,agent_steps_per_second_delta_pct,workload_output_tokens_per_second_delta_pct,decode_tokens_per_second_delta_pct,ttft_p50_seconds_delta_pct"
    );
    let value = |name| &records[1][header.iter().position(|column| column == name).unwrap()];
    assert_eq!(value("label"), "arm,\"quoted\"\nline");
    assert_eq!(value("ref"), "refs/test");
    assert_eq!(value("decode_tokens_per_second"), "10.0");
    assert_eq!(value("prompt_tokens_min"), "40");
    assert_eq!(value("failed_request_ids"), "[]");
    assert_eq!(value("content_identity"), "{\"s:0\":[\"same\"]}");
    assert_eq!(value("delta_comparable"), "true");
}

#[test]
fn failed_gates_preserve_complete_artifacts_and_inventory() {
    let artifact = tempfile::tempdir().unwrap();
    let mut input = document("base");
    input["gates"] = serde_json::json!({"evaluated":true,"passed":false,
        "checks":[{"name":"cache_pct","passed":false,"detail":"cache 25 < 70"}],
        "session_acceptance_failures":[{"passed":false,"failures":["missing restore"]}]});
    std::fs::write(
        artifact.path().join("run.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    assert!(run(artifact.path()).status.success());
    let inventory = std::fs::read_to_string(artifact.path().join("artifact-sha256.txt")).unwrap();
    for path in [
        "run.json",
        "summary/REPORT.md",
        "summary/comparison.csv",
        "summary/comparison.json",
        "summary/charts/decode-throughput.svg",
        "summary/charts/workload-output-throughput.svg",
        "summary/charts/ttft-p50.svg",
    ] {
        assert!(
            inventory
                .lines()
                .any(|line| line.ends_with(&format!("  {path}")))
        );
    }
    assert!(!inventory.contains("artifact-sha256.txt"));
    let retained: serde_json::Value =
        serde_json::from_slice(&std::fs::read(artifact.path().join("run.json")).unwrap()).unwrap();
    assert_eq!(retained["gates"], input["gates"]);
    let markdown = std::fs::read_to_string(artifact.path().join("summary/REPORT.md")).unwrap();
    assert!(markdown.lines().any(|line| line.starts_with("- **FAIL**")
        && line.contains("<code>cache_pct</code>")
        && line.contains("25 < 70")));
    assert!(
        markdown
            .lines()
            .any(|line| line.contains("**FAIL**") && line.contains("missing restore"))
    );
}

#[test]
fn missing_metrics_remain_null_and_have_no_chart_points() {
    let artifact = tempfile::tempdir().unwrap();
    let mut input = document("base");
    let cell = &mut input["results"][0]["cells"][0];
    for metric in [
        "decode_tokens_per_second",
        "workload_output_tokens_per_second",
        "agent_steps_per_second",
        "ttft_p50_seconds",
        "ttft_p95_seconds",
        "mean_in_flight",
    ] {
        cell[metric] = serde_json::Value::Null;
    }
    for count in [
        "generation_seconds",
        "workload_window_seconds",
        "completion_tokens",
    ] {
        cell[count] = 0.into();
    }
    cell["ttft_samples"] = serde_json::json!([]);
    std::fs::write(
        artifact.path().join("run.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    assert!(run(artifact.path()).status.success());
    let rows: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifact.path().join("summary/comparison.json")).unwrap(),
    )
    .unwrap();
    assert!(rows[0]["decode_tokens_per_second"].is_null());
    let chart =
        std::fs::read_to_string(artifact.path().join("summary/charts/decode-throughput.svg"))
            .unwrap();
    assert!(chart.contains("points=\"\""));
    assert!(!chart.contains("<circle"));
}

#[test]
fn unicode_xml_labels_escape_delimiters_without_losing_text() {
    let artifact = tempfile::tempdir().unwrap();
    let input = document("雪<&\"'");
    std::fs::write(
        artifact.path().join("run.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    assert!(run(artifact.path()).status.success());
    let chart =
        std::fs::read_to_string(artifact.path().join("summary/charts/ttft-p50.svg")).unwrap();
    assert!(chart.contains("雪&lt;&amp;&quot;&#39;"));
    assert!(!chart.contains("雪<&"));
}

#[test]
fn output_divergence_suppresses_all_paired_deltas() {
    let artifact = tempfile::tempdir().unwrap();
    let mut input = document("base");
    let mut candidate = input["results"][0].clone();
    candidate["label"] = "candidate".into();
    candidate["cells"][0]["content_sha256_by_request"]["s:0"] = "different".into();
    candidate["cells"][0]["completion_tokens"] = 200.into();
    input["results"].as_array_mut().unwrap().push(candidate);
    input["builds"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"label":"candidate"}));
    std::fs::write(
        artifact.path().join("run.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    assert!(run(artifact.path()).status.success());
    let rows: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifact.path().join("summary/comparison.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(rows[1]["delta_comparable"], false);
    for metric in [
        "agent_steps_per_second_delta_pct",
        "decode_tokens_per_second_delta_pct",
        "workload_output_tokens_per_second_delta_pct",
        "ttft_p50_seconds_delta_pct",
    ] {
        assert!(rows[1][metric].is_null());
    }
    assert_eq!(rows[1]["decode_tokens_per_second"], 20.0);
}

#[test]
fn identities_and_cohort_counts_survive_markdown_rendering() {
    let artifact = tempfile::tempdir().unwrap();
    let mut input = document("base");
    input["builds"][0]["engine"] = "vllm".into();
    input["config"]["engine_config"] = serde_json::json!({});
    std::fs::write(
        artifact.path().join("run.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    assert!(run(artifact.path()).status.success());
    let markdown = std::fs::read_to_string(artifact.path().join("summary/REPORT.md")).unwrap();
    let tables = markdown
        .lines()
        .filter(|line| line.starts_with('|'))
        .collect::<Vec<_>>();
    assert!(tables.iter().any(|line| line.contains(
        "| base | vllm | <code>v&#96;1&#96;&#124;snow 雪</code> | <code>abcdef012345</code> |"
    )));
    assert!(
        tables
            .iter()
            .any(|line| line.contains("| 1 | 2 | 4 | 4 | harness 2 / 4 |"))
    );
}

#[test]
fn empty_results_render_a_headerless_csv_and_empty_json_array() {
    let artifact = tempfile::tempdir().unwrap();
    let mut input = document("base");
    input["results"] = serde_json::json!([]);
    std::fs::write(
        artifact.path().join("run.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    assert!(run(artifact.path()).status.success());
    assert_eq!(
        std::fs::read(artifact.path().join("summary/comparison.csv")).unwrap(),
        b"\r\n"
    );
    let rows: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifact.path().join("summary/comparison.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(rows, serde_json::json!([]));
}

#[test]
fn unevaluated_gates_render_without_inventing_an_outcome() {
    let artifact = tempfile::tempdir().unwrap();
    let mut input = document("base");
    input["gates"] = serde_json::json!({"evaluated":false,"passed":null,"checks":[]});
    let bytes = serde_json::to_vec(&input).unwrap();
    std::fs::write(artifact.path().join("run.json"), &bytes).unwrap();
    assert!(run(artifact.path()).status.success());
    let markdown = std::fs::read_to_string(artifact.path().join("summary/REPORT.md")).unwrap();
    assert!(markdown.contains("Overall: **NOT EVALUATED**"));
    assert!(!markdown.contains("Overall: **PASS**"));
    assert_eq!(
        std::fs::read(artifact.path().join("run.json")).unwrap(),
        bytes
    );
}

#[test]
fn evaluated_gates_without_an_outcome_reject_before_report_writes() {
    let artifact = tempfile::tempdir().unwrap();
    let mut input = document("base");
    input["gates"] = serde_json::json!({"evaluated":true,"passed":null,"checks":[]});
    let bytes = serde_json::to_vec(&input).unwrap();
    std::fs::write(artifact.path().join("run.json"), &bytes).unwrap();
    let output = run(artifact.path());
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("require a boolean outcome"));
    assert!(!artifact.path().join("summary").exists());
    assert_eq!(
        std::fs::read(artifact.path().join("run.json")).unwrap(),
        bytes
    );
}
