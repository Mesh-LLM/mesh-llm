//! Actual native correctness CLI with supplied inert shard/package tool and product.
use super::cache_family_full_matrix_cli::Fixture;
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};
fn hash(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}
fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}
fn wrapper(f: &Fixture, name: &str, helper: &str) -> PathBuf {
    let path = f.root.join(name);
    let body = format!(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > {}\nexport CACHE_ARTIFACT_ROOT={}\n{} --ignored --exact cache_family_artifact_cli::{} --nocapture >/dev/null 2>&1 || exit $?\n{}\n",
        quote(f.root.join(format!("{name}-argv")).to_str().unwrap()),
        quote(f.root.to_str().unwrap()),
        quote(std::env::current_exe().unwrap().to_str().unwrap()),
        helper,
        if name == "artifact-tool" {
            format!(
                "/bin/cat {}",
                quote(f.root.join("artifact-receipt.json").to_str().unwrap())
            )
        } else {
            String::new()
        }
    );
    fs::write(&path, &body).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}
pub(super) fn input(f: &Fixture, package: bool) -> PathBuf {
    let matrix: Value = serde_json::from_slice(&fs::read(&f.input).unwrap()).unwrap();
    let mut v = matrix["profiles"]["qwen3_dense"]["correctness"].clone();
    let tool = wrapper(f, "artifact-tool", "inert_artifact_tool");
    let correctness = wrapper(f, "artifact-correctness", "inert_artifact_correctness");
    v["correctness"] = json!(correctness);
    v["correctness_sha256"] = json!(hash(&fs::read(correctness).unwrap()));
    let model = f.root.join(if package { "package" } else { "shards" });
    fs::create_dir(&model).unwrap();
    let (mut pins, primary) = (BTreeMap::<String, String>::new(), model.clone());
    if package {
        fs::write(model.join("model-package.json"), b"inert pinned manifest").unwrap();
        v["model_sha256"] = json!(hash(b"inert pinned manifest"));
    } else {
        for i in 1..=3 {
            let name = format!("MiniMax-M2.7-UD-Q2_K_XL-{i:05}-of-00003.gguf");
            let bytes = format!("inert supplied shard{i}");
            fs::write(model.join(&name), bytes.as_bytes()).unwrap();
            pins.insert(name, hash(bytes.as_bytes()));
        }
    }
    let primary = if package {
        primary
    } else {
        model.join("MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf")
    };
    v["model"] = json!(primary);
    if !package {
        v["model_sha256"] = json!(pins["MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf"]);
    }
    v["case_key"] = json!(if package { "deepseek3" } else { "minimax_m27" });
    v["model_id"] = json!(if package {
        "unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL"
    } else {
        "unsloth/MiniMax-M2.7-GGUF:UD-Q2_K_XL"
    });
    v["ctx_size"] = json!(if package { 32 } else { 512 });
    v["prefix_tokens"] = json!(if package { 4 } else { 16 });
    v["topologies"] = json!([if package {
        "package-stage1"
    } else {
        "one-stage"
    }]);
    v["cell_seconds"] = json!(20);
    v["execution_seconds"] = json!(90);
    v["artifact"] = json!({"kind":if package{"layer-package"}else{"complete-shards"},"tool":tool,"tool_sha256":hash(&fs::read(tool).unwrap()),"shard_pins":pins});
    let path = f.root.join("artifact-input.json");
    fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    path
}
fn invoke(f: &Fixture, path: PathBuf, cancel: Cancellation) -> process::ProcessReport {
    let result = process::supervise(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: [
                "automation".into(),
                "cache-family-correctness".into(),
                "--input".into(),
                path.into_os_string(),
                "--output".into(),
                f.root.join("artifact-output").into_os_string(),
            ]
            .into_iter()
            .map(Arg::Public)
            .collect(),
            cwd: f.root.clone(),
            environment: BTreeMap::new(),
        },
        &Limits {
            execution: Duration::from_secs(100),
            graceful_shutdown: Duration::from_secs(12),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &cancel,
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        result.failure.is_none()
            && result.cleanup.complete
            && !result.cleanup.forced
            && !result.cleanup.graceful_signal_failed
            && result.cleanup.failure.is_none()
    );
    assert!(
        [&result.stdout, &result.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
    );
    result
}
#[test]
fn artifact_actual_cli_routes_complete_minimax_shards_and_exact_deepseek_package_only_range() {
    for package in [false, true] {
        let f = Fixture::new();
        let path = input(&f, package);
        let r = invoke(&f, path, Cancellation::default());
        assert_eq!(r.outcome, process::Outcome::Exited);
        assert_eq!(r.status.and_then(|s| s.code()), Some(0));
        let receipt: Value = serde_json::from_slice(
            &fs::read(f.root.join("artifact-output/cache-correctness-stage.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["status"], "completed");
        if package {
            assert_eq!(receipt["baseline"], "n/a-package-only");
        }
        let report = &receipt["rows"][0]["evidence"]["skippy"];
        assert_eq!(report["layer_start"], if package { 3 } else { 0 });
        assert_eq!(report["layer_end"], if package { 4 } else { 62 });
        assert_eq!(report["stage_index"], i32::from(package));
        assert!(f.root.join("artifact-tool-admitted").exists());
        f.directory.close().unwrap();
    }
}
#[test]
fn artifact_actual_cli_refuses_missing_or_wrong_shard_before_correctness_and_preserves_partial() {
    for missing in [false, true] {
        let f = Fixture::new();
        let path = input(&f, false);
        fs::write(
            f.root.join("artifact-mode"),
            if missing {
                b"missing".as_slice()
            } else {
                b"wrong".as_slice()
            },
        )
        .unwrap();
        let r = invoke(&f, path, Cancellation::default());
        assert_eq!(r.status.and_then(|s| s.code()), Some(1));
        assert!(!f.root.join("artifact-correctness-argv").exists());
        let receipt: Value = serde_json::from_slice(
            &fs::read(f.root.join("artifact-output/cache-correctness-stage.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["status"], "failed-or-incomplete");
        assert_ne!(receipt["rows"][0]["evidence"]["status"], "pass");
        f.directory.close().unwrap();
    }
}
#[test]
fn artifact_actual_cli_causal_held_tool_cancellation_reaps_and_retains_nonpassing_receipt() {
    let f = Fixture::new();
    let path = input(&f, true);
    fs::write(f.root.join("artifact-mode"), b"hold").unwrap();
    let c = Cancellation::default();
    let child = c.clone();
    let root = f.root.clone();
    let task = thread::spawn(move || {
        // Borrowed artifact owner is represented only by the immutable spec here.
        process::supervise(
            &ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                arguments: [
                    "automation".into(),
                    "cache-family-correctness".into(),
                    "--input".into(),
                    path.into_os_string(),
                    "--output".into(),
                    root.join("artifact-output").into_os_string(),
                ]
                .into_iter()
                .map(Arg::Public)
                .collect(),
                cwd: root,
                environment: BTreeMap::new(),
            },
            &Limits {
                execution: Duration::from_secs(100),
                graceful_shutdown: Duration::from_secs(12),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &child,
            OutputFiles::default(),
        )
        .unwrap()
    });
    let until = Instant::now() + Duration::from_secs(15);
    while !f.root.join("artifact-tool-admitted").exists() && Instant::now() < until {
        thread::sleep(Duration::from_millis(5));
    }
    let admitted = f.root.join("artifact-tool-admitted").exists();
    c.cancel();
    let r = task.join().unwrap();
    assert!(admitted);
    assert_eq!(r.outcome, process::Outcome::Cancelled);
    assert!(
        r.cleanup.complete
            && !r.cleanup.forced
            && !r.cleanup.graceful_signal_failed
            && r.cleanup.failure.is_none()
            && r.failure.is_none()
    );
    assert!(
        [&r.stdout, &r.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
    );
    assert!(!f.root.join("artifact-correctness-argv").exists());
    let receipt: Value = serde_json::from_slice(
        &fs::read(f.root.join("artifact-output/cache-correctness-stage.json")).unwrap(),
    )
    .unwrap();
    assert_ne!(receipt["status"], "completed");
    f.directory.close().unwrap();
}
#[test]
#[ignore = "subprocess-only supplied inert artifact admission"]
fn inert_artifact_tool() {
    let root = PathBuf::from(std::env::var_os("CACHE_ARTIFACT_ROOT").unwrap());
    let v: Value =
        serde_json::from_slice(&fs::read(root.join("artifact-input.json")).unwrap()).unwrap();
    let argv = fs::read_to_string(root.join("artifact-tool-argv")).unwrap();
    let mode = fs::read_to_string(root.join("artifact-mode")).unwrap_or_default();
    fs::write(root.join("artifact-tool-admitted"), b"actual admitted tool").unwrap();
    if mode == "hold" {
        loop {
            thread::sleep(Duration::from_millis(10));
        }
    }
    let package = v["case_key"] == "deepseek3";
    assert!(argv.starts_with(if package {
        "admit-package\n"
    } else {
        "admit-source\n"
    }));
    assert!(argv.contains(v["model"].as_str().unwrap()));
    let shape = json!({"layer_count":if package{61}else{62},"activation_width":if package{7168}else{3072},"native_context_tokens":512});
    let receipt = if package {
        assert!(argv.contains("--layer-start\n3\n--layer-end\n4"));
        json!({"schema_version":1,"kind":"layer-package","root":v["model"],"manifest_sha256":v["model_sha256"],"model_id":v["model_id"],"dimensions":shape,"state_layer_start":3,"state_layer_end":4,"independent_full_source_verified":false,"baseline":"n/a-package-only","selected_files":[{"role":"metadata","sha256":"a".repeat(64)}]})
    } else {
        let parent = PathBuf::from(v["model"].as_str().unwrap())
            .parent()
            .unwrap()
            .to_owned();
        let mut files:Vec<_>=(1..=3).map(|i|{let name=format!("MiniMax-M2.7-UD-Q2_K_XL-{i:05}-of-00003.gguf");assert!(argv.contains(&format!("--pin\n{name}=")));json!({"logical_path":parent.join(&name),"path":parent.join(&name),"sha256":hash(&fs::read(parent.join(&name)).unwrap())})}).collect();
        if mode == "missing" {
            files.pop();
        }
        if mode == "wrong" {
            files[1]["sha256"] = json!("f".repeat(64));
        }
        json!({"schema_version":1,"kind":"gguf-source","primary":v["model"],"dimensions":shape,"ordered_files":files})
    };
    fs::write(
        root.join("artifact-receipt.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
}
#[test]
#[ignore = "subprocess-only supplied inert correctness product"]
fn inert_artifact_correctness() {
    let root = PathBuf::from(std::env::var_os("CACHE_ARTIFACT_ROOT").unwrap());
    let v: Value =
        serde_json::from_slice(&fs::read(root.join("artifact-input.json")).unwrap()).unwrap();
    let argv = fs::read_to_string(root.join("artifact-correctness-argv")).unwrap();
    let words: Vec<_> = argv.lines().collect();
    assert_eq!(words[0], "state-handoff");
    let mut flags = BTreeMap::new();
    let mut i = 1;
    while i < words.len() {
        if words[i] == "--n-gpu-layers=0" {
            i += 1;
            continue;
        }
        assert!(i + 1 < words.len());
        assert!(flags.insert(words[i], words[i + 1]).is_none());
        i += 2;
    }
    let package = v["case_key"] == "deepseek3";
    let known = [
        "--model",
        "--model-id",
        "--stage-server-bin",
        "--layer-end",
        "--ctx-size",
        "--activation-width",
        "--stage-load-mode",
        "--state-layer-start",
        "--state-layer-end",
        "--state-stage-index",
        "--state-payload-kind",
        "--prefix-token-count",
        "--cache-hit-repeats",
        "--runtime-lane-count",
        "--source-bind-addr",
        "--restore-bind-addr",
        "--report-out",
    ];
    assert!(flags.keys().all(|k| known.contains(k)));
    assert_eq!(flags["--layer-end"], if package { "61" } else { "62" });
    assert_eq!(
        flags["--activation-width"],
        if package { "7168" } else { "3072" }
    );
    assert_eq!(
        flags["--stage-load-mode"],
        if package {
            "layer-package"
        } else {
            "runtime-slice"
        }
    );
    assert_eq!(flags["--model"], v["model"].as_str().unwrap());
    assert_eq!(flags["--ctx-size"], if package { "32" } else { "512" });
    let start = flags["--state-layer-start"].parse::<u32>().unwrap();
    let end = flags["--state-layer-end"].parse::<u32>().unwrap();
    let index = flags["--state-stage-index"].parse::<u32>().unwrap();
    assert!(if package {
        (start, end, index) == (3, 4, 1)
    } else {
        [(0, 62, 0), (0, 20, 0), (20, 41, 1), (41, 62, 2)].contains(&(start, end, index))
    });
    assert_eq!(
        flags["--prefix-token-count"],
        if package { "4" } else { "16" }
    );
    let report = json!({"mode":"state-handoff","status":"pass","matches":true,"predicted_token_matches":true,"cache_hit_matches":true,"model_identity":{"model_id":v["model_id"]},"state_payload_kind":"resident-kv","stage_index":index,"layer_start":start,"layer_end":end,"requested_prefix_token_count":if package{4}else{16},"benchmark_prompt_token_count":if package{5}else{17},"benchmark_prompt_text":"fixed prompt","activation_width":if package{7168}else{3072},"cache_hit_repeats":3,"cache_hit_import_ms":[1.0,2.0,3.0],"cache_hit_decode_ms":[3.0,4.0,5.0],"recompute_total_ms":8.0,"cache_hit_total_ms":6.0});
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    fs::write(flags["--report-out"], &bytes).unwrap();
    println!("{}", String::from_utf8(bytes).unwrap());
}
