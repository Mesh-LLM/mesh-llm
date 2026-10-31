use super::{
    contract::{self, Backend, Cell, Tool},
    evidence, execution, jy_plan,
};
use std::path::PathBuf;

#[test]
fn l3_restart_requires_distinct_processes_exact_state_and_typed_predictions() {
    let valid = serde_json::json!({"status":"passed","source":"l3","seed_process_id":1,
        "restore_process_id":2,"state_integrity":"sha256_exact","fixed_input_predictions":[1,2,3,4]});
    let check = |value| evidence::process_restart(&serde_json::to_vec(&value).unwrap());
    check(valid.clone()).unwrap();
    for (field, value) in [
        ("seed_process_id", serde_json::json!(0)),
        ("restore_process_id", serde_json::json!(1)),
        ("source", serde_json::json!("radix")),
        ("status", serde_json::json!("failed")),
        ("state_integrity", serde_json::json!("token_counts_only")),
        ("fixed_input_predictions", serde_json::json!([1, 2, 3])),
        (
            "fixed_input_predictions",
            serde_json::json!(["1", "2", "3", "4"]),
        ),
    ] {
        let mut invalid = valid.clone();
        invalid[field] = value;
        assert!(check(invalid).is_err(), "accepted invalid {field}");
    }
    let mut invalid = valid;
    invalid["arbitrary_headers"] = serde_json::json!({});
    assert!(check(invalid).is_err());
}

#[cfg(unix)]
#[test]
fn evidence_archive_receipt_binds_contents_and_excludes_runtime_symlinks() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("evidence");
    std::fs::create_dir_all(output.join("nested")).unwrap();
    std::fs::write(output.join("result.json"), br#"{"status":"passed"}"#).unwrap();
    std::fs::write(
        output.join("nested/response.txt"),
        b"exact generated content",
    )
    .unwrap();
    let outside = dir.path().join("external-runtime");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(
        outside.join("library"),
        b"must not enter evidence transport",
    )
    .unwrap();
    std::os::unix::fs::symlink(&outside, output.join("selected-runtime")).unwrap();
    execution::archive(&output).unwrap();

    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.with_extension("archive.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["archive"], "evidence.tar.gz");
    let bytes = std::fs::read(output.with_extension("tar.gz")).unwrap();
    assert_eq!(receipt["sha256"], super::hash(&bytes));
    let gzip = flate2::read::GzDecoder::new(bytes.as_slice());
    let mut tar = tar::Archive::new(gzip);
    let mut restored = std::collections::BTreeMap::new();
    for entry in tar.entries().unwrap() {
        let mut entry = entry.unwrap();
        assert!(entry.header().entry_type().is_file());
        let path = entry.path().unwrap().to_path_buf();
        let mut contents = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut contents).unwrap();
        assert!(restored.insert(path, contents).is_none());
    }
    assert_eq!(
        restored,
        std::collections::BTreeMap::from([
            (
                PathBuf::from("evidence/result.json"),
                br#"{"status":"passed"}"#.to_vec()
            ),
            (
                PathBuf::from("evidence/nested/response.txt"),
                b"exact generated content".to_vec()
            ),
        ])
    );
    assert!(
        execution::archive(&output).is_err(),
        "never overwrite existing evidence"
    );
}

#[cfg(unix)]
#[test]
fn archive_refuses_non_unicode_filename_before_creating_transport() {
    use std::os::unix::ffi::OsStringExt;
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join(std::ffi::OsString::from_vec(vec![0xff]));
    let error = execution::archive(&output).unwrap_err();
    assert!(error.to_string().contains("filename must be Unicode"));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn cache_projection_rejects_content_mismatch_invalid_reuse_and_extra_payload() {
    let valid = serde_json::json!({
        "status":"pass", "content_parity":"exact", "cold_content":"round",
        "repeat":{"cached_tokens":20,"prompt_tokens":22,"content":"round"},
        "growing_chat":{"cached_tokens":20,"prompt_tokens":28,"content":"sphere"},
        "growing_chat_repeat":{"cached_tokens":26,"prompt_tokens":28,"content":"sphere"}
    });
    let parse = |value: serde_json::Value| -> crate::command::DynResult<()> {
        serde_json::from_value::<evidence::CacheEvidence>(value)?.verify()
    };
    parse(valid.clone()).unwrap();
    for (pointer, value) in [
        ("/repeat/content", serde_json::json!("different")),
        (
            "/growing_chat_repeat/content",
            serde_json::json!("different"),
        ),
        ("/repeat/cached_tokens", serde_json::json!(0)),
        ("/repeat/cached_tokens", serde_json::json!(23)),
        ("/status", serde_json::json!("failed")),
    ] {
        let mut changed = valid.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        assert!(parse(changed).is_err());
    }
    let mut extra = valid;
    extra["arbitrary_protocol_headers"] = serde_json::json!({"header":"not retained"});
    assert!(parse(extra).is_err());
}

#[test]
fn doctor_requires_explicit_clean_issue_and_exact_selected_leaf() {
    let path = PathBuf::from("/runtime/exact-leaf");
    let clean = serde_json::json!({"native_runtime":{"path":path},"runtime_issue":null});
    evidence::doctor(&clean, &path).unwrap();
    for invalid in [
        serde_json::json!({"native_runtime":{"path":path}}),
        serde_json::json!({"native_runtime":{"path":path},"runtime_issue":"ABI mismatch"}),
        serde_json::json!({"native_runtime":{"path":"/runtime"},"runtime_issue":null}),
        serde_json::json!({"runtime_issue":null}),
    ] {
        assert!(evidence::doctor(&invalid, &path).is_err());
    }
}

#[test]
fn separately_pinned_headers_cannot_mismatch_the_runtime_build() {
    let source = "1".repeat(40);
    let manifest = serde_json::json!({
        "build":{"backend":"metal","llama_patched_sha":source},
        "runtime":{"backend":{"kind":"metal"},"libraries":["lib/libllama.dylib"]}
    });
    super::verify_manifest(&manifest, Backend::Metal, &source).unwrap();
    assert!(super::verify_manifest(&manifest, Backend::Metal, &"2".repeat(40)).is_err());
    assert!(super::verify_manifest(&manifest, Backend::Cuda, &source).is_err());
    let mut missing = manifest;
    missing["build"]
        .as_object_mut()
        .unwrap()
        .remove("llama_patched_sha");
    assert!(super::verify_manifest(&missing, Backend::Metal, &source).is_err());
}

#[test]
fn exact_four_agentx_model_bytes_and_pins_are_retained() {
    let models = contract::models().unwrap();
    assert_eq!(super::hash(contract::CATALOG), contract::CATALOG_SHA);
    assert_eq!(
        models.iter().map(|m| m.key.as_str()).collect::<Vec<_>>(),
        [
            "qwen3-4b-q4-k-m",
            "qwen3.5-4b-q4-k-m",
            "qwen3-coder-30b-a3b-q4-k-m",
            "qwen3.5-35b-a3b-q4-k-m"
        ]
    );
    assert_eq!(models.iter().filter(|m| m.recurrent_case()).count(), 2);
    assert!(
        models
            .iter()
            .all(|m| m.reference().contains("@") && m.reference().ends_with("Q4_K_M.gguf"))
    );
}

fn matrix(backend: Backend) -> String {
    let mut text = format!("MODEL backend={} loads=1\n", backend.device());
    for k in ["f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"] {
        for v in ["f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"] {
            if backend == Backend::Metal && k != v {
                text += &format!(
                    "SKIP backend=MTL0 K={k} V={v} reason=preexisting_Metal_requires_matching_KV_types\n"
                );
            } else {
                text += &format!(
                    "REAL_FA backend={} K={k} V={v} q=39 physical=256 logical=256 requests=1\nPASS backend={} K={k} V={v} prefill=39 decoded=16 attention_layers=8 seconds=0.5 completion_hex=6869\n",
                    backend.device(),
                    backend.device()
                );
            }
        }
    }
    text + &format!(
        "SUMMARY backend={} model_loads=1 passed={} failed=0 skipped={}\n",
        backend.device(),
        backend.pairs(),
        if backend == Backend::Metal { 56 } else { 0 }
    )
}

#[test]
fn gpu_evidence_requires_every_pair_actual_graph_and_finite_results() {
    for backend in [Backend::Cpu, Backend::Metal, Backend::Cuda] {
        let text = matrix(backend);
        evidence::matrix(&text, backend).unwrap();
        for broken in [
            text.replacen("q=39", "q=1", 1),
            text.replacen("seconds=0.5", "seconds=NaN", 1),
            text.replacen("decoded=16", "decoded=0", 1),
            text.replacen("completion_hex=6869", "completion_hex=", 1),
            text.replacen("failed=0", "failed=1", 1),
            text.replacen("REAL_FA ", "MISSING_FA ", 1),
        ] {
            assert!(evidence::matrix(&broken, backend).is_err());
        }
        assert!(
            evidence::matrix(&(text.clone() + "PASS backend=CPU K=f16 V=f16\n"), backend).is_err()
        );
    }
    assert!(
        evidence::matrix(&matrix(Backend::Cpu), Backend::Cuda).is_err(),
        "CPU cannot qualify requested CUDA"
    );
}

#[test]
fn rust_success_marker_cannot_hide_an_ignored_or_missing_test() {
    let name = "fixture::required_cache_test";
    evidence::rust_test(
        &format!("test {name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;"),
        name,
    )
    .unwrap();
    for text in [
        "test result: ok. 0 passed; 0 failed; 1 ignored;",
        "test result: ok. 1 passed; 0 failed; 0 ignored;",
    ] {
        assert!(evidence::rust_test(text, name).is_err());
    }
}

#[test]
fn changed_artifact_bytes_fail_identity_before_execution() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tool");
    std::fs::write(&path, b"first").unwrap();
    let tool = Tool {
        path: path.clone(),
        sha256: super::hash(b"first"),
    };
    tool.verify().unwrap();
    std::fs::write(path, b"other").unwrap();
    assert!(tool.verify().is_err());
}

#[test]
fn unknown_model_and_unbounded_cell_are_rejected() {
    let mut cell = Cell {
        schema_version: 1,
        prepared: Tool {
            path: "/prepared.json".into(),
            sha256: "0".repeat(64),
        },
        model_key: "qwen3.5-4b-q4-k-m".into(),
        model: "/model.gguf".into(),
        port: 19337,
        execution_seconds: 120,
    };
    cell.validate().unwrap();
    cell.model_key = "substituted-model".into();
    assert!(cell.validate().is_err());
    cell.model_key = "qwen3.5-4b-q4-k-m".into();
    cell.execution_seconds = u64::MAX;
    assert!(cell.validate().is_err());
}

#[test]
fn excluded_sparks_are_refused_by_id_and_real_hostname() {
    for (id, host) in [
        ("spark-000280", "different"),
        ("spark-000499", "different"),
        ("spark-000137", "spark-db08"),
        ("spark-000137", "spark-9f06"),
    ] {
        assert!(jy_plan::excluded(id, host));
    }
    assert!(!jy_plan::excluded("spark-000137", "spark-james"));
}

#[test]
fn accepted_jy_envelope_keeps_all_four_pins_and_never_submits() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("jobs");
    let input = jy_plan::Fleet {
        schema_version: 1,
        backend: Backend::Cuda,
        source_commit: "1".repeat(40),
        prepared_image: format!("sha256:{}", "2".repeat(64)),
        automation: "/opt/xtask".into(),
        prepared: Tool {
            path: "/opt/prepared.json".into(),
            sha256: "3".repeat(64),
        },
        nodes: vec![jy_plan::Node {
            id: "spark-000137".into(),
            hostname: "spark-james".into(),
        }],
        execution_seconds: 1800,
    };
    jy_plan::run(input, &output).unwrap();
    for model in contract::models().unwrap() {
        let value: serde_json::Value = serde_json::from_slice(
            &std::fs::read(output.join(format!("{}.json", model.key))).unwrap(),
        )
        .unwrap();
        let runtime = &value["spec"]["runtime"];
        assert_eq!(runtime["model_package"]["ref"], model.reference());
        assert_eq!(runtime["model_package"]["sha256"], model.files[0].sha256);
        assert_eq!(runtime["models"][0]["env"], "AGENTX_MODEL_PATH");
        assert_eq!(runtime["workspace"]["repos"][0]["rev"], "1".repeat(40));
        let command = runtime["command"].as_str().unwrap();
        assert!(command.contains("run-env"));
        assert!(
            !command.contains("cargo ")
                && !command.contains("base64")
                && !command.contains("jy submit")
        );
        assert!(command.contains("exit \"$worker_status\""));
        assert_eq!(
            value["spec"]["target"]["nodes"],
            serde_json::json!(["spark-000137"])
        );
    }
    assert_eq!(std::fs::read_dir(output).unwrap().count(), 5);
}

#[cfg(unix)]
#[test]
fn supervised_child_captures_success_and_refuses_failed_exit() {
    let dir = tempfile::tempdir().unwrap();
    let cancel = crate::process::Cancellation::default();
    let invoke = |label: &str, command: &str| {
        execution::child(
            &PathBuf::from("/bin/sh"),
            vec!["-c".into(), command.into()],
            (dir.path(), dir.path()),
            label,
            std::time::Duration::from_secs(2),
            &cancel,
            &[],
        )
    };
    assert_eq!(invoke("success", "printf observed").unwrap(), "observed");
    assert!(invoke("failed", "printf rejected; exit 7").is_err());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("failed.process.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["success"], false);
    assert_eq!(receipt["exit_code"], 7);
    assert_eq!(receipt["cleanup_complete"], true);
}

#[cfg(unix)]
#[test]
fn cancellation_refuses_child_and_preserves_failed_process_receipt() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("must-not-start");
    let cancel = crate::process::Cancellation::default();
    cancel.cancel();
    let result = execution::child(
        &PathBuf::from("/bin/sh"),
        vec![
            "-c".into(),
            "printf started > \"$1\"".into(),
            "fixture".into(),
            marker.clone().into_os_string(),
        ],
        (dir.path(), dir.path()),
        "cancelled",
        std::time::Duration::from_secs(2),
        &cancel,
        &[],
    );
    assert!(result.is_err());
    assert!(!marker.exists());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("cancelled.process.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["success"], false);
    assert!(receipt["supervision_failure"].as_str().is_some());
}
