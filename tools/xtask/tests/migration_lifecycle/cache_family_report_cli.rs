//! Register in migration_lifecycle after root wires the proposed report command.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

fn report(directory: &Path, inputs: &[Json], corpus: Json) -> process::ProcessReport {
    let mut args = vec![
        Value::Public("automation".into()),
        Value::Public("cache-family-report".into()),
    ];
    for (index, input) in inputs.iter().enumerate() {
        let path = directory.join(format!("input {index}.json"));
        fs::write(&path, serde_json::to_vec(input).unwrap()).unwrap();
        args.extend([Value::Public("--input".into()), Value::Public(path.into())]);
    }
    let corpus_path = directory.join("corpus.json");
    fs::write(&corpus_path, serde_json::to_vec(&corpus).unwrap()).unwrap();
    args.extend([
        Value::Public("--use-case-corpus".into()),
        Value::Public(corpus_path.into()),
        Value::Public("--output".into()),
        Value::Public(directory.join("output report.md").into()),
    ]);
    let spec = ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: args,
        cwd: directory.into(),
        environment: BTreeMap::new(),
    };
    let limits = Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let given = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(given.cleanup.complete, "{given:?}");
    assert!(
        !given.stdout.truncated && !given.stderr.truncated,
        "{given:?}"
    );
    given
}

fn row(family: &str) -> Json {
    json!({"family":family,"model_id":"org/model:q4","payload":"resident-kv","stage_load_mode":"runtime-slice","prefix_tokens":128,"benchmark_prompt_token_count":129,"notes":"measured | fixture","case":{"resident_kv_bytes_per_token":1024},"skippy":{"status":"pass","cache_hit_import_ms":[1,20,30],"cache_hit_decode_ms":[30,20,1]},"llama_server":{"status":"ok","warm_median_ms":62}})
}

#[test]
fn pairs_before_median_preserves_family_storage_and_package_fallback() {
    let directory = tempfile::tempdir().unwrap();
    let mut measured = row("Qwen3Next");
    measured["skippy"]["cache_storage_bytes"] = json!(0);
    let mut package = row("DeepSeek3");
    package["stage_load_mode"] = json!("layer-package");
    package["llama_server"] = json!({"status":"timeout","warm_median_ms":9999});
    package["skippy"] = json!({"status":"pass","cache_hit_total_ms":10,"recompute_total_ms":90});
    let output = report(
        directory.path(),
        &[json!([row("Llama"), measured, package])],
        json!({"use_cases":[]}),
    );
    assert!(output.success(), "{output:?}");
    let text = fs::read_to_string(directory.path().join("output report.md")).unwrap();
    assert!(text.find("| Qwen3Next |").unwrap() < text.find("| Llama |").unwrap());
    assert!(text.contains("| Llama | `org/model:q4` | `ResidentKv` | pass | 128 | 129 | 62.0 | 31.0 | **2.00x faster** | 128.0 KiB | metadata-derived | measured / fixture |"));
    assert!(text.contains("| 0 | measured |"));
    assert!(text.contains("| Skippy stage recompute | 10.0 | **9.00x faster** |"));
}

#[test]
fn package_baseline_label_identifies_the_successful_warm_measurement_used_for_speedup() {
    for (field, label) in [
        ("warm_median_ms", "llama-server warm median"),
        ("warm_mean_ms", "llama-server warm mean"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut package = row("DeepSeek3");
        package["stage_load_mode"] = json!("layer-package");
        package["llama_server"] = json!({"status":"ok",(field):100});
        package["skippy"] =
            json!({"status":"pass","cache_hit_total_ms":10,"recompute_total_ms":90});
        let output = report(
            directory.path(),
            &[json!([package])],
            json!({"use_cases":[]}),
        );
        assert!(output.success(), "{output:?}");
        let text = fs::read_to_string(directory.path().join("output report.md")).unwrap();
        assert!(
            text.contains(&format!("| {label} | 10.0 | **10.00x faster** |")),
            "{text}"
        );
        assert!(!text.contains("| Skippy stage recompute |"), "{text}");
    }
}

#[test]
fn combines_inputs_keeps_last_duplicate_matrix_cell_and_excludes_failed_baseline() {
    let directory = tempfile::tempdir().unwrap();
    let mut first = row("Llama");
    first["use_case"] = json!("coding_agent_loop");
    first["use_case_label"] = json!("Coding | loop");
    let mut last = first.clone();
    last["llama_server"]["warm_median_ms"] = json!(31);
    let mut failed = row("ExcludedFamily");
    failed["llama_server"]["status"] = json!("timeout");
    let output = report(
        directory.path(),
        &[json!([first, failed]), json!([last])],
        json!({"use_cases":[{"key":"coding_agent_loop","label":"Coding | loop","source":{"dataset":"org/data","config":"v1","split":"test","row_idx":7}}]}),
    );
    assert!(output.success(), "{output:?}");
    let text = fs::read_to_string(directory.path().join("output report.md")).unwrap();
    assert!(text.contains("| Coding / loop | 1.00x |"));
    assert!(text.contains("| Coding / loop | `org/data` | `v1` | `test` | 7 |"));
    assert!(!text.contains("ExcludedFamily"));
}

#[test]
fn malformed_negative_overflow_and_failed_correctness_cannot_publish_performance_claims() {
    for attack in [
        "shape",
        "negative",
        "paired-overflow",
        "storage-overflow",
        "unpaired",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("output report.md");
        fs::write(&destination, "preserved").unwrap();
        let mut bad = row("Llama");
        match attack {
            "shape" => bad["skippy"]["cache_hit_total_ms"] = json!("bad"),
            "unpaired" => bad["skippy"]["cache_hit_decode_ms"] = json!([1]),
            "negative" => bad["skippy"]["cache_hit_import_ms"] = json!([-1]),
            "paired-overflow" => {
                bad["skippy"]["cache_hit_import_ms"] = json!([1e308]);
                bad["skippy"]["cache_hit_decode_ms"] = json!([1e308]);
            }
            "storage-overflow" => {
                bad["case"]["resident_kv_bytes_per_token"] = json!(u64::MAX);
                bad["prefix_tokens"] = json!(2);
            }
            _ => unreachable!(),
        }
        let output = report(directory.path(), &[json!([bad])], json!({"use_cases":[]}));
        assert!(!output.success(), "{attack}: {output:?}");
        assert_eq!(fs::read_to_string(destination).unwrap(), "preserved");
    }
    let directory = tempfile::tempdir().unwrap();
    let mut failed = row("Llama");
    failed["skippy"]["status"] = json!("mismatch");
    let output = report(
        directory.path(),
        &[json!([failed])],
        json!({"use_cases":[]}),
    );
    assert!(output.success(), "{output:?}");
    let text = fs::read_to_string(directory.path().join("output report.md")).unwrap();
    assert!(text.contains("| mismatch |"));
    assert!(!text.contains("faster**"));
}

#[test]
fn actual_report_caller_preserves_two_inputs_and_configured_invalid_owner_fails_closed() {
    use std::os::unix::fs::PermissionsExt as _;
    let directory = tempfile::tempdir().unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let automation = directory.path().join("supplied automation");
    fs::write(&automation, "#!/bin/bash\nif [[ $1 == automation && $2 == cache-family-report ]]; then exec \"$XTASK_REPORT_BIN\" \"$@\"; fi\nexit 0\n").unwrap();
    fs::set_permissions(&automation, fs::Permissions::from_mode(0o700)).unwrap();
    let operator = directory.path().join("operator.json");
    fs::write(&operator, "{}").unwrap();
    let corpus = directory.path().join("corpus.json");
    fs::write(&corpus, r#"{"use_cases":[]}"#).unwrap();
    for (index, valid) in [true, false, false].into_iter().enumerate() {
        let output = directory.path().join(format!("results {index}"));
        let full = output.join("full-gguf");
        let cases = output.join("use-cases");
        fs::create_dir_all(&full).unwrap();
        fs::create_dir(&cases).unwrap();
        fs::write(
            full.join("production-cache-bench.json"),
            serde_json::to_vec(&json!([row("Llama")])).unwrap(),
        )
        .unwrap();
        fs::write(cases.join("production-cache-bench.json"), "[]").unwrap();
        let owner = if valid {
            automation.as_os_str().to_owned()
        } else if index == 1 {
            "".into()
        } else {
            "relative-owner".into()
        };
        let environment = BTreeMap::from([
            ("PATH".into(), Value::Public("/usr/bin:/bin".into())),
            (
                "XTASK_REPORT_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
            (
                "SKIPPY_CACHE_OPERATOR_INPUT".into(),
                Value::Public(operator.clone().into()),
            ),
            ("SKIPPY_CACHE_SKIP_BUILD".into(), Value::Public("1".into())),
            (
                "SKIPPY_CACHE_USECASE_CORPUS".into(),
                Value::Public(corpus.clone().into()),
            ),
            ("MESH_LLM_AUTOMATION_BIN".into(), Value::Public(owner)),
        ]);
        let spec = ProcessSpec {
            executable: "/bin/bash".into(),
            arguments: vec![
                Value::Public(root.join("evals/skippy-cache-family-bench.sh").into()),
                Value::Public(output.clone().into()),
            ],
            cwd: directory.path().into(),
            environment,
        };
        let limits = Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let given = process::supervise(
            &spec,
            &limits,
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert!(given.cleanup.complete, "{given:?}");
        assert_eq!(given.success(), valid, "{given:?}");
        let destination = output.join("readme-tables.md");
        assert_eq!(destination.exists(), valid);
        if valid {
            assert!(
                fs::read_to_string(destination)
                    .unwrap()
                    .contains("| Llama |")
            );
        }
    }
}
