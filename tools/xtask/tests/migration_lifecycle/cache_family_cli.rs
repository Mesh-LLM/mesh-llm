//! Actual xtask route with an inert correctness child, never model inference.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn string(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend((s.len() as u64).to_le_bytes());
    bytes.extend(s.as_bytes());
}
fn model() -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3_u32.to_le_bytes());
    b.extend(0_u64.to_le_bytes());
    b.extend(4_u64.to_le_bytes());
    string(&mut b, "general.architecture");
    b.extend(8_u32.to_le_bytes());
    string(&mut b, "llama");
    for (k, v) in [
        ("llama.context_length", 512_u32),
        ("llama.block_count", 6),
        ("llama.embedding_length", 16),
    ] {
        string(&mut b, k);
        b.extend(4_u32.to_le_bytes());
        b.extend(v.to_le_bytes());
    }
    b
}
struct Fixture {
    _root: tempfile::TempDir,
    root: PathBuf,
    input: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        fs::create_dir(root.join("build")).unwrap();
        fs::write(root.join("build/libfixture.so"), b"inert").unwrap();
        let mut tree = Sha256::new();
        tree.update((13_u64).to_be_bytes());
        tree.update(b"libfixture.so");
        tree.update(Sha256::digest(b"inert"));
        let tree = hex::encode(tree.finalize());
        let executable = root.join("correctness");
        fs::write(&executable,r#"#!/bin/sh
[ "$1" = state-handoff ] || exit 64
shift
while [ "$#" -gt 0 ]; do
 case "$1" in
 --model) model="$2";; --report-out) output="$2";; --prompt) prompt="$2";;
 --model-id) [ "$2" = fixture ] || exit 65;;
 --state-layer-start) [ "$2" = 0 ] || exit 65;; --state-layer-end|--layer-end) [ "$2" = 6 ] || exit 65;;
 --activation-width) [ "$2" = 16 ] || exit 65;; --prefix-token-count) [ "$2" = 8 ] || exit 65;;
 --cache-hit-repeats) [ "$2" = 2 ] || exit 65;; --state-payload-kind) [ "$2" = resident-kv ] || exit 65;;
 --stage-server-bin|--ctx-size|--stage-load-mode|--state-stage-index|--runtime-lane-count|--source-bind-addr|--restore-bind-addr) :;;
 --n-gpu-layers=-1) shift; continue;; *) exit 66;; esac
 shift 2
done
[ -d "$LLAMA_STAGE_BUILD_DIR" ] || exit 67
[ "$OMP_NUM_THREADS" = 2 ] || exit 67
[ "$prompt" = fail ] && exit 23
/bin/cat "${model%/*}/report.json" > "$output"
/bin/cat "$output"
"#).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        let m = model();
        fs::write(root.join("model.gguf"), &m).unwrap();
        let digest = hash(&fs::read(&executable).unwrap());
        let input = root.join("request.json");
        let request = json!({"schema_version":1,"case_key":"llama","model_id":"fixture","correctness":executable,"correctness_sha256":digest,"stage_server":executable,"stage_server_sha256":digest,"model":root.join("model.gguf"),"model_sha256":hash(&m),"native_build":root.join("build"),"native_build_sha256":tree,"ctx_size":512,"prefix_tokens":8,"cache_hit_repeats":2,"runtime_lane_count":1,"source_port":19061,"restore_port":19062,"n_gpu_layers":-1,"prompt":"pass","topologies":["one-stage"],"borrow_resident_hits":false,"cache_decoded_result_hits":false,"execution_seconds":30,"cell_seconds":8,"settings":{"OMP_NUM_THREADS":"2"}});
        fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
        fs::write(root.join("report.json"),serde_json::to_vec_pretty(&json!({"mode":"state-handoff","status":"pass","matches":true,"predicted_token_matches":true,"cache_hit_matches":true,"model_identity":{"model_id":"fixture"},"state_payload_kind":"resident-kv","stage_index":0,"layer_start":0,"layer_end":6,"requested_prefix_token_count":8,"benchmark_prompt_token_count":9,"benchmark_prompt_text":"observed prompt","activation_width":16,"cache_hit_repeats":2,"cache_hit_import_ms":[1.0,2.0],"cache_hit_decode_ms":[3.0,4.0],"recompute_total_ms":8.0,"cache_hit_total_ms":5.0})).unwrap()).unwrap();
        Self {
            _root: tmp,
            root,
            input,
        }
    }
    fn change(&self, field: &str, value: Json) {
        let mut i: Json = serde_json::from_slice(&fs::read(&self.input).unwrap()).unwrap();
        i[field] = value;
        fs::write(&self.input, serde_json::to_vec(&i).unwrap()).unwrap();
    }
    fn invoke(&self, out: &Path) -> process::ProcessReport {
        let r = process::supervise(
            &ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                arguments: [
                    "automation".into(),
                    "cache-family-correctness".into(),
                    "--input".into(),
                    self.input.clone().into_os_string(),
                    "--output".into(),
                    out.to_path_buf().into_os_string(),
                ]
                .into_iter()
                .map(Value::Public)
                .collect(),
                cwd: self.root.clone(),
                environment: BTreeMap::new(),
            },
            &Limits {
                execution: Duration::from_secs(40),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 1024 * 1024,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert!(
            r.cleanup.complete
                && !r.cleanup.forced
                && r.cleanup.failure.is_none()
                && !r.cleanup.graceful_signal_failed
        );
        assert!(r.failure.is_none());
        assert_eq!(r.outcome, process::Outcome::Exited);
        assert!(
            [&r.stdout, &r.stderr]
                .iter()
                .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
        );
        r
    }
}
#[test]
fn cache_producer_actual_cli_observes_report_and_preserves_child_failure() {
    for failure in [false, true] {
        let f = Fixture::new();
        if failure {
            f.change("prompt", json!("fail"));
        }
        let out = f.root.join("output");
        let r = f.invoke(&out);
        assert_eq!(r.success(), !failure);
        let v: Json =
            serde_json::from_slice(&fs::read(out.join("cache-correctness-stage.json")).unwrap())
                .unwrap();
        assert_eq!(v["completed_topologies"], 1);
        assert_eq!(
            v["rows"][0]["evidence"]["status"],
            if failure { "failed-process" } else { "pass" }
        );
        if !failure {
            assert!(
                v["rows"][0]["evidence"]["lifecycle"]["stdout_suppressed_lines"]
                    .as_u64()
                    .unwrap()
                    > 0,
                "product token-named stdout must be sanitized independently of typed receipt"
            );
            assert_eq!(
                v["rows"][0]["evidence"]["skippy"]["cache_hit_total_ms"],
                5.0
            );
            assert!(out.join("topology-00/after-receipt.json").is_file());
        }
    }
}
#[test]
fn cache_producer_actual_cli_refuses_stale_pin_and_existing_output() {
    let f = Fixture::new();
    f.change("model_sha256", json!("b".repeat(64)));
    let out = f.root.join("stale");
    assert!(!f.invoke(&out).success());
    assert!(!out.join("topology-00/state-handoff.json").exists());
    assert!(out.join("cache-correctness-stage.json").is_file());
    let f = Fixture::new();
    let out = f.root.join("existing");
    fs::create_dir(&out).unwrap();
    fs::write(out.join("sentinel"), b"keep").unwrap();
    assert!(!f.invoke(&out).success());
    assert_eq!(fs::read(out.join("sentinel")).unwrap(), b"keep");
    assert!(!out.join("input.json").exists());
}
#[test]
fn cache_producer_actual_cli_refuses_mismatched_report_and_split_closure() {
    let f = Fixture::new();
    let mut report: Json =
        serde_json::from_slice(&fs::read(f.root.join("report.json")).unwrap()).unwrap();
    report["cache_hit_matches"] = json!(false);
    fs::write(
        f.root.join("report.json"),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();
    let output = f.root.join("mismatch");
    assert!(!f.invoke(&output).success());
    assert!(output.join("topology-00/state-handoff.json").is_file());
    assert!(!output.join("topology-00/after-receipt.json").exists());
    let f = Fixture::new();
    let mut bytes = model();
    bytes[16..24].copy_from_slice(&5_u64.to_le_bytes());
    string(&mut bytes, "split.count");
    bytes.extend(2_u32.to_le_bytes());
    bytes.extend(2_u16.to_le_bytes());
    fs::write(f.root.join("model.gguf"), &bytes).unwrap();
    f.change("model_sha256", json!(hash(&bytes)));
    let output = f.root.join("split");
    assert!(!f.invoke(&output).success());
    assert!(!output.join("topology-00/state-handoff.json").exists());
    assert!(output.join("topology-00/before-process.json").is_file());
}
#[test]
fn cache_producer_actual_cli_fifo_input_refuses_without_writer_or_publication() {
    use std::os::unix::ffi::OsStrExt as _;
    let f = Fixture::new();
    fs::remove_file(&f.input).unwrap();
    let path = std::ffi::CString::new(f.input.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let output = f.root.join("fifo");
    assert!(!f.invoke(&output).success());
    assert!(!output.exists());
}
