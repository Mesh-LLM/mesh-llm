use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Duration};
struct Fixture {
    directory: tempfile::TempDir,
    root: PathBuf,
    input: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        fs::create_dir(root.join("cache")).unwrap();
        let input = root.join("request.json");
        fs::write(&input,serde_json::to_vec(&json!({"schema_version":1,"cache_root":root.join("cache"),"cases":["llama"],"use_cases":[],"corpus":null,"prefix_sweep":[64,1024],"prefix_tokens":null,"n_gpu_layers":-1,"cache_hit_repeats":5,"runtime_lane_count":2,"serving_ctx_size":null,"concurrency":[1,4],"concurrent_requests":3,"concurrent_output_tokens":32,"llama_parallel":1,"llama_repeats":3,"ttft_slo_ms":2000,"tpot_slo_ms":100,"skip_llama_server":false,"old_server":null,"new_server":null,"model_sha256":{}})).unwrap()).unwrap();
        Self {
            directory,
            root,
            input,
        }
    }
    fn change(&self, key: &str, value: Json) {
        let mut i: Json = serde_json::from_slice(&fs::read(&self.input).unwrap()).unwrap();
        i[key] = value;
        fs::write(&self.input, serde_json::to_vec(&i).unwrap()).unwrap();
    }
    fn invoke(&self, out: &std::path::Path) -> process::ProcessReport {
        let report = process::supervise(
            &ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                arguments: [
                    "automation".into(),
                    "cache-family-plan".into(),
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
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert!(
            report.failure.is_none()
                && report.cleanup.complete
                && !report.cleanup.forced
                && report.cleanup.failure.is_none()
                && !report.cleanup.graceful_signal_failed
        );
        assert!(
            [&report.stdout, &report.stderr]
                .iter()
                .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
        );
        report
    }
}
#[test]
fn cache_family_plan_actual_cli_retains_missing_matrix_and_corpus_source() {
    let f = Fixture::new();
    let out = f.root.join("plan.json");
    assert!(f.invoke(&out).success());
    let plan: Json = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(plan["cell_count"], 2);
    assert_eq!(plan["cells"][1]["case"]["ctx_size"], 1152);
    assert_eq!(plan["cells"][0]["case"]["cache_hit_repeats"], 5);
    assert_eq!(plan["cells"][0]["case"]["n_gpu_layers"], -1);
    assert_eq!(
        plan["cells"][0]["model_observation"]["status"],
        "missing-model"
    );
    assert!(plan["execution"].is_null());
    assert_eq!(
        plan["cells"][0]["tasks"]["paired_serving"]["shared_ctx_size"],
        1024
    );
    let corpus = f.root.join("corpus.json");
    let bytes=serde_json::to_vec(&json!({"version":1,"use_cases":[{"key":"tool","label":"Tool","prompt":"owned fixture prompt","prefix_tokens":256,"source":{"dataset":"fixture/dataset","row_idx":11}}]})).unwrap();
    fs::write(&corpus, &bytes).unwrap();
    f.change("use_cases", json!(["all"]));
    f.change(
        "corpus",
        json!({"path":corpus,"sha256":hex::encode(Sha256::digest(&bytes))}),
    );
    f.change("prefix_sweep", json!([]));
    let out = f.root.join("corpus-plan.json");
    assert!(f.invoke(&out).success());
    let plan: Json = serde_json::from_slice(&fs::read(out).unwrap()).unwrap();
    assert_eq!(plan["cells"][0]["case"]["prefix_tokens"], 256);
    assert_eq!(plan["cells"][0]["use_case"]["source"]["row_idx"], 11);
    assert_eq!(plan["cells"][0]["output_relative"], "tool/llama-p256");
    f.directory.close().unwrap();
}
#[test]
fn cache_family_plan_actual_cli_refuses_corpus_pin_unknown_case_and_existing_output() {
    let f = Fixture::new();
    f.change("cases", json!(["qwen3moe"]));
    let output = f.root.join("unknown.json");
    assert!(!f.invoke(&output).success());
    assert!(!output.exists());
    f.change("cases", json!(["llama"]));
    let corpus = f.root.join("corpus.json");
    fs::write(&corpus, b"{\"version\":1,\"use_cases\":[]}").unwrap();
    f.change("use_cases", json!(["all"]));
    f.change("corpus", json!({"path":corpus,"sha256":"a".repeat(64)}));
    let output = f.root.join("corrupt.json");
    assert!(!f.invoke(&output).success());
    assert!(!output.exists());
    let output = f.root.join("existing.json");
    fs::write(&output, b"keep").unwrap();
    assert!(!f.invoke(&output).success());
    assert_eq!(fs::read(output).unwrap(), b"keep");
    f.directory.close().unwrap();
}
#[test]
fn cache_family_plan_actual_cli_refuses_fifo_and_external_model_link() {
    use std::os::unix::{ffi::OsStrExt as _, fs::symlink};
    let f = Fixture::new();
    let catalog: Vec<Json> = serde_json::from_slice(include_bytes!(
        "../../src/automation/cache_family_plan/catalog.json"
    ))
    .unwrap();
    let case = catalog.iter().find(|c| c["key"] == "llama").unwrap();
    let snapshot = f
        .root
        .join("cache")
        .join(case["snapshot_relative"].as_str().unwrap());
    fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
    let outside = f.root.join("outside.gguf");
    fs::write(&outside, b"outside bytes").unwrap();
    symlink(&outside, &snapshot).unwrap();
    let output = f.root.join("outside-plan.json");
    assert!(!f.invoke(&output).success());
    assert!(!output.exists());
    assert_eq!(fs::read(&outside).unwrap(), b"outside bytes");
    fs::remove_file(&f.input).unwrap();
    let path = std::ffi::CString::new(f.input.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let output = f.root.join("fifo-plan.json");
    assert!(!f.invoke(&output).success());
    assert!(!output.exists());
    f.directory.close().unwrap();
}
