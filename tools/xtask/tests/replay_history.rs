#[path = "replay_history/mod.rs"]
mod replay_history;
use replay_history::support::{Fixture, read, write};
use serde_json::json;

#[test]
fn full_session_history_preserves_v3_shape_metrics_and_backend_identity() {
    let fixture = Fixture::new(false);
    let result = fixture.run(false, false);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let rows = fixture.rows();
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row["schema_version"], 3);
    assert_eq!(row["complete"], true);
    assert_eq!(row["prompt_count"], 12);
    assert_eq!(row["output_tokens"], 24);
    assert_eq!(row["decode_tokens_per_second"], 2.0);
    assert_eq!(row["end_to_end_tokens_per_second"], 1.0);
    assert_eq!(row["ttft_ms_mean"], 1000.0);
    assert_eq!(row["ttft_ms_p90"], 1000.0);
    assert_eq!(row["backend_binary_sha256"], "9".repeat(64));
    assert_eq!(row["model"]["revision"], "b".repeat(40));
    assert_eq!(row["session_cohort_sha256"].as_array().unwrap().len(), 1);
    assert!(row["recurrent_restores"].is_null());
    assert_eq!(
        std::fs::read_to_string(&fixture.github).unwrap().trim(),
        "repair_required=false"
    );
    assert_eq!(
        std::fs::read_to_string(&fixture.output)
            .unwrap()
            .lines()
            .count(),
        1
    );
}

#[test]
fn regression_is_informational_until_three_matching_complete_runs_then_gated() {
    let fixture = Fixture::new(false);
    fixture.seed(|row| {
        row["decode_tokens_per_second"] = 20.0.into();
        row["end_to_end_tokens_per_second"] = 10.0.into();
    });
    assert!(fixture.run(false, true).status.success());
    assert!(!fixture.run(true, true).status.success());
    assert!(
        std::fs::read_to_string(&fixture.github)
            .unwrap()
            .ends_with("repair_required=true\n")
    );
    assert_eq!(fixture.rows()[0]["complete"], true);
    let shards = std::fs::read_to_string(fixture.baseline.join("prior.jsonl")).unwrap();
    std::fs::write(
        fixture.baseline.join("prior.jsonl"),
        shards.lines().take(2).collect::<Vec<_>>().join("\n"),
    )
    .unwrap();
    assert!(fixture.run(true, true).status.success());
    assert!(
        std::fs::read_to_string(&fixture.github)
            .unwrap()
            .ends_with("repair_required=false\n")
    );
}

#[test]
fn hardware_dataset_model_and_cohort_drift_restart_the_baseline() {
    for field in ["hardware", "dataset", "model", "cohort"] {
        let fixture = Fixture::new(false);
        fixture.seed(|row| {
            row["decode_tokens_per_second"] = 100.0.into();
            match field {
                "hardware" => row["hardware_fingerprint"]["os_version"] = "different".into(),
                "dataset" => row["replay"]["dataset_sha256"] = "7".repeat(64).into(),
                "model" => row["model"]["sha256"] = "7".repeat(64).into(),
                "cohort" => row["session_cohort_sha256"] = json!(["7".repeat(64)]),
                _ => unreachable!(),
            }
        });
        assert!(fixture.run(true, true).status.success(), "{field}");
    }
}

#[test]
fn incomplete_raw_turns_and_failed_gates_never_request_repair() {
    for failure in ["missing", "duplicate", "order", "error", "gate"] {
        let fixture = Fixture::new(false);
        fixture.seed(|row| row["decode_tokens_per_second"] = 100.0.into());
        if failure == "gate" {
            let path = fixture.runs.join("fixture/run.json");
            let mut doc = read(&path);
            doc["gates"]["passed"] = false.into();
            write(&path, &doc);
        } else {
            let path = fixture
                .runs
                .join("fixture/data/pass-1/main/c-1-requests.jsonl");
            let mut raw = std::fs::read_to_string(&path)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                .collect::<Vec<_>>();
            match failure {
                "missing" => {
                    raw.pop();
                }
                "duplicate" => raw.push(raw[0].clone()),
                "order" => raw.swap(0, 1),
                "error" => raw[0]["error"] = "HTTP 500".into(),
                _ => unreachable!(),
            }
            std::fs::write(
                path,
                raw.iter()
                    .map(|row| serde_json::to_string(row).unwrap())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .unwrap();
        }
        assert!(!fixture.run(true, true).status.success(), "{failure}");
        assert!(
            std::fs::read_to_string(&fixture.github)
                .unwrap()
                .ends_with("repair_required=false\n")
        );
        assert_eq!(fixture.rows()[0]["complete"], false);
    }
}

#[test]
fn model_uri_manifest_context_prompt_and_backend_tampering_fail_closed() {
    for field in [
        "uri",
        "manifest",
        "context",
        "model-digest",
        "prompt",
        "backend",
        "metrics",
    ] {
        let fixture = Fixture::new(false);
        let run = fixture.runs.join("fixture/run.json");
        let mut doc = read(&run);
        match field {
            "uri" => doc["config"]["model"] = "/local/model.gguf".into(),
            "manifest" => doc["inputs"]["manifest_sha256"] = "0".repeat(64).into(),
            "context" => {
                doc["context_preflight"]["main"]["cohorts"]["1"]["context_tokens"] = 8192.into()
            }
            "model-digest" => {
                doc["context_preflight"]["main"]["model"]["sha256"] = "0".repeat(64).into()
            }
            "prompt" => {
                doc["context_preflight"]["main"]["prompt_tokens_by_cohort"]["1"]["s0:0"] =
                    123.into()
            }
            "backend" => doc["builds"][0]["binary_sha256"] = "mutable".into(),
            "metrics" => {
                let path = fixture.runs.join("fixture/data/pass-1/main/c-1.json");
                let mut cell = read(&path);
                cell["completion_tokens"] = 999999.0.into();
                write(&path, &cell);
            }
            _ => unreachable!(),
        }
        write(&run, &doc);
        assert!(!fixture.run(true, false).status.success(), "{field}");
        assert!(
            std::fs::read_to_string(&fixture.github)
                .unwrap()
                .ends_with("repair_required=false\n")
        );
    }
}

#[test]
fn every_pass_and_concurrency_is_required_and_sibling_rows_are_retained() {
    let fixture = Fixture::new(false);
    let mut matrix = read(&fixture.matrix);
    let mut missing = matrix["models"][0].clone();
    missing["family"] = "missing".into();
    matrix["models"].as_array_mut().unwrap().push(missing);
    write(&fixture.matrix, &matrix);
    assert!(!fixture.run(true, false).status.success());
    assert_eq!(fixture.rows().len(), 1);
    assert_eq!(fixture.rows()[0]["complete"], true);
    std::fs::remove_file(fixture.runs.join("fixture/data/pass-2/main/c-1.json")).unwrap();
    assert!(!fixture.run(true, false).status.success());
    assert_eq!(fixture.rows()[0]["complete"], false);
}

#[test]
fn different_session_cohorts_across_models_fail_without_repair() {
    let fixture = Fixture::new(false);
    let mut matrix = read(&fixture.matrix);
    let mut other = matrix["models"][0].clone();
    other["family"] = "other".into();
    matrix["models"].as_array_mut().unwrap().push(other);
    write(&fixture.matrix, &matrix);
    fixture.family("other", false, "other-session");
    assert!(!fixture.run(true, false).status.success());
    assert_eq!(fixture.rows().len(), 2);
    assert_eq!(fixture.rows()[1]["complete"], false);
}

#[test]
fn recurrent_history_requires_correlated_nontrivial_native_restores() {
    let fixture = Fixture::new(true);
    let result = fixture.run(true, false);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fixture.rows()[0]["recurrent_restores"], 6);
    std::fs::write(fixture.runs.join("fixture/data/pass-1/main/mesh.log"), "").unwrap();
    assert!(!fixture.run(true, false).status.success());
    assert_eq!(fixture.rows()[0]["complete"], false);
    assert!(
        std::fs::read_to_string(&fixture.github)
            .unwrap()
            .ends_with("repair_required=false\n")
    );
}

#[test]
fn malformed_baseline_and_input_never_request_repair_and_shards_survive() {
    let fixture = Fixture::new(false);
    fixture.seed(|_| {});
    std::fs::write(fixture.baseline.join("prior.jsonl"), "malformed\n").unwrap();
    assert!(!fixture.run(true, true).status.success());
    assert_eq!(fixture.rows()[0]["complete"], true);
    assert!(
        std::fs::read_to_string(&fixture.github)
            .unwrap()
            .ends_with("repair_required=false\n")
    );
    let mut matrix = read(&fixture.matrix);
    matrix["models"][0]["revision"] = "main".into();
    write(&fixture.matrix, &matrix);
    assert!(!fixture.run(true, false).status.success());
    assert!(
        std::fs::read_to_string(&fixture.github)
            .unwrap()
            .ends_with("repair_required=false\n")
    );
}

#[test]
fn within_tolerance_drift_passes_and_zero_baseline_clipping_is_gateable() {
    let fixture = Fixture::new(false);
    fixture.seed(|row| row["decode_tokens_per_second"] = 2.05.into());
    assert!(fixture.run(true, true).status.success());
    fixture.seed(|row| {
        row["ttft_ms_mean"] = 0.0.into();
        row["ttft_ms_p90"] = 0.0.into();
    });
    assert!(!fixture.run(true, true).status.success());
    assert!(
        std::fs::read_to_string(&fixture.github)
            .unwrap()
            .ends_with("repair_required=true\n")
    );
}

#[test]
fn repair_label_uses_its_source_and_backend_independently_of_the_base_arm() {
    let fixture = Fixture::new(false);
    for pass in 1..=2 {
        let parent = fixture.runs.join(format!("fixture/data/pass-{pass}"));
        std::fs::rename(parent.join("main"), parent.join("fixed")).unwrap();
    }
    let path = fixture.runs.join("fixture/run.json");
    let mut doc = read(&path);
    doc["builds"][0]["label"] = "fixed".into();
    let mut base = doc["builds"][0].clone();
    base["label"] = "base".into();
    base["commit"] = "8".repeat(40).into();
    base["binary_sha256"] = "8".repeat(64).into();
    doc["builds"].as_array_mut().unwrap().push(base);
    std::fs::rename(
        fixture.runs.join("fixture/context-preflight/main"),
        fixture.runs.join("fixture/context-preflight/fixed"),
    )
    .unwrap();
    let qualification = doc["context_preflight"]
        .as_object_mut()
        .unwrap()
        .remove("main")
        .unwrap();
    doc["context_preflight"]["fixed"] = qualification;
    write(&path, &doc);
    let result = fixture.run_label("fixed", true, false);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(fixture.rows()[0]["source_sha"], "a".repeat(40));
    assert_eq!(fixture.rows()[0]["backend_binary_sha256"], "9".repeat(64));
}

#[test]
fn terminal_length_finishes_are_preserved_and_gate_against_zero_baseline() {
    let fixture = Fixture::new(false);
    fixture.seed(|_| {});
    for pass in 1..=2 {
        let directory = fixture.runs.join(format!("fixture/data/pass-{pass}/main"));
        let rawpath = directory.join("c-1-requests.jsonl");
        let text = std::fs::read_to_string(&rawpath).unwrap();
        let mut records = text
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        records[0]["finish_reason"] = "length".into();
        std::fs::write(
            rawpath,
            records
                .iter()
                .map(|record| serde_json::to_string(record).unwrap())
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let path = directory.join("c-1.json");
        let mut cell = read(&path);
        cell["finish_reason_length_requests"] = 1.into();
        write(&path, &cell);
    }
    assert!(!fixture.run(true, true).status.success());
    assert!(
        (fixture.rows()[0]["finish_reason_length_pct"]
            .as_f64()
            .unwrap()
            - 100.0 / 6.0)
            .abs()
            < 1e-9
    );
    assert!(
        std::fs::read_to_string(&fixture.github)
            .unwrap()
            .ends_with("repair_required=true\n")
    );
}

#[test]
fn published_schema_describes_full_session_history_fields() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let schema = read(&root.join("ci/agentic-replay-nightly/schema.json"));
    assert_eq!(schema["properties"]["schema_version"]["const"], 3);
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("session_cohort_sha256")));
    assert!(required.contains(&json!("recurrent_restores")));
    let fixture = Fixture::new(false);
    assert!(fixture.run(false, false).status.success());
    let row = fixture.rows().remove(0);
    for property in required {
        assert!(row.get(property.as_str().unwrap()).is_some());
    }
    assert!(row["output_tokens"].is_u64());
    assert_eq!(row["replay"]["sessions_per_concurrency"], 3);
}

#[test]
fn historical_source_changes_are_allowed_and_only_latest_three_matches_gate() {
    let fixture = Fixture::new(false);
    fixture.seed(|_| {});
    let path = fixture.baseline.join("prior.jsonl");
    let rows = std::fs::read_to_string(&path).unwrap();
    let mut old: serde_json::Value = serde_json::from_str(rows.lines().next().unwrap()).unwrap();
    old["decode_tokens_per_second"] = 1000.0.into();
    old["source_sha"] = "0".repeat(40).into();
    std::fs::write(
        path,
        format!("{}\n{rows}", serde_json::to_string(&old).unwrap()),
    )
    .unwrap();
    assert!(fixture.run(true, true).status.success());
    assert!(
        std::fs::read_to_string(&fixture.github)
            .unwrap()
            .ends_with("repair_required=false\n")
    );
}
