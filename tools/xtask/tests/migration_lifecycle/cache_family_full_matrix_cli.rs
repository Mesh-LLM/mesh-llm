//! Actual full native producer with inert supplied catalog-model/protocol artifacts.
#[path = "cache_family_full_matrix_cli/matrix_correctness_fixture.rs"]
mod correctness;
#[path = "cache_family_full_matrix_cli/matrix_host_fixture.rs"]
mod host;
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
        ("llama.context_length", 4096_u32),
        ("llama.block_count", 28),
        ("llama.embedding_length", 1024),
    ] {
        string(&mut b, k);
        b.extend(4_u32.to_le_bytes());
        b.extend(v.to_le_bytes());
    }
    b
}
fn quote(v: &str) -> String {
    format!("'{}'", v.replace('\'', "'\\''"))
}
pub(super) struct Fixture {
    pub(super) directory: tempfile::TempDir,
    pub(super) root: PathBuf,
    pub(super) input: PathBuf,
    output: PathBuf,
}
impl Fixture {
    pub(super) fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let cache = root.join("cache");
        let model_path=cache.join("models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf");
        fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        fs::write(&model_path, model()).unwrap();
        let build = root.join("build");
        fs::create_dir(&build).unwrap();
        fs::write(build.join("libfixture.so"), b"inert").unwrap();
        let mut tree = Sha256::new();
        tree.update(13_u64.to_be_bytes());
        tree.update(b"libfixture.so");
        tree.update(Sha256::digest(b"inert"));
        let tree = hex::encode(tree.finalize());
        let helper = std::env::current_exe().unwrap();
        let mut artifacts = BTreeMap::new();
        for side in ["native", "old", "new", "correctness"] {
            let binary = root.join(side);
            let mut body = format!(
                "#!/bin/sh\nexport CACHE_FIXTURE_ROOT={}\nexport CACHE_MATRIX_MODEL={}\nexport CACHE_MATRIX_SIDE={}\nexport CACHE_MATRIX_ARGS={}/args-{}\nprintf '%s\\n' \"$@\" > \"$CACHE_MATRIX_ARGS\"\n",
                quote(root.to_str().unwrap()),
                quote(model_path.to_str().unwrap()),
                quote(side),
                quote(root.to_str().unwrap()),
                side
            );
            if side == "correctness" {
                body.push_str(&format!("exec {} --ignored --exact cache_family_full_matrix_cli::inert_cache_matrix_correctness --nocapture\n",quote(helper.to_str().unwrap())));
            } else {
                body.push_str(r#"if [ "$1" = serve-openai ]; then export CACHE_FIXTURE_NATIVE=0; shift; else export CACHE_FIXTURE_NATIVE=1; fi
while [ "$#" -gt 0 ]; do
 case "$1" in
 --model) export CACHE_FIXTURE_MODEL="$2"; shift 2;; --config) export CACHE_FIXTURE_CONFIG="$2"; shift 2;;
 --bind-addr) export CACHE_FIXTURE_BIND="$2"; shift 2;; --port) export CACHE_FIXTURE_BIND="127.0.0.1:$2"; shift 2;;
 --host) [ "$2" = 127.0.0.1 ] || exit 64; shift 2;; --ctx-size) [ "$2" = 512 ] || exit 64; shift 2;;
 --n-gpu-layers) [ "$2" = 0 ] || exit 64; shift 2;; --parallel|--generation-concurrency) [ "$2" = 1 ] || exit 64; shift 2;;
 --telemetry-level) [ "$2" = debug ] || exit 64; shift 2;; --no-webui) shift;; *) exit 64;; esac
done
"#);
                body.push_str(&format!("exec {} --ignored --exact cache_family_full_matrix_cli::inert_cache_matrix_host --nocapture\n",quote(helper.to_str().unwrap())));
            }
            if side == "new" {
                body = body.replace("serve-openai", "serve").replace(
                    "export CACHE_FIXTURE_NATIVE=0",
                    "export CACHE_FIXTURE_CURRENT=1; export CACHE_FIXTURE_NATIVE=0",
                );
            }
            if side == "old" || side == "new" {
                let verb = if side == "new" {
                    "serve"
                } else {
                    "serve-openai"
                };
                body = body.replace("#!/bin/sh\n", &format!("#!/bin/sh\nif [ \"$#\" = 2 ] && [ \"$2\" = --help ]; then [ \"$1\" = {verb} ] || exit 64; printf 'inert help\\n'; exit 0; fi\n"));
            }
            fs::write(&binary, &body).unwrap();
            fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
            artifacts.insert(side, json!({"path":binary,"sha256":hash(body.as_bytes())}));
        }
        let correctness = json!({"schema_version":1,"case_key":"qwen3_dense","model_id":"Qwen/Qwen3-0.6B:Q8_0","correctness":artifacts["correctness"]["path"],"correctness_sha256":artifacts["correctness"]["sha256"],"stage_server":artifacts["new"]["path"],"stage_server_sha256":artifacts["new"]["sha256"],"model":model_path,"model_sha256":hash(&model()),"native_build":build,"native_build_sha256":tree,"ctx_size":512,"prefix_tokens":64,"cache_hit_repeats":3,"runtime_lane_count":1,"source_port":1,"restore_port":2,"n_gpu_layers":0,"prompt":null,"topologies":["one-stage"],"borrow_resident_hits":false,"cache_decoded_result_hits":false,"execution_seconds":30,"cell_seconds":8,"settings":{"OMP_NUM_THREADS":"2"}});
        let profile = |side: &str| json!({"schema_version":1,"host":if side=="native"{"native-baseline"}else if side=="old"{"skippy-old"}else{"skippy-new"},"binary":artifacts[side]["path"],"binary_sha256":artifacts[side]["sha256"],"source_commit":"a".repeat(40),"native_build":build,"native_build_sha256":tree,"model":model_path,"model_sha256":hash(&model()),"model_id":"Qwen/Qwen3-0.6B:Q8_0","layer_end":28,"ctx_size":512,"lane_count":1,"n_gpu_layers":0,"port":1,"environment":{"OMP_NUM_THREADS":"2"},"worker":{},"startup_timeout_secs":5,"execution_timeout_secs":30});
        let input = root.join("input.json");
        let output = root.join("matrix");
        fs::write(&input,serde_json::to_vec(&json!({"schema_version":1,"plan":{"schema_version":1,"cache_root":cache,"cases":["qwen3_dense"],"use_cases":[],"corpus":null,"prefix_sweep":[64],"prefix_tokens":null,"n_gpu_layers":null,"cache_hit_repeats":null,"runtime_lane_count":1,"serving_ctx_size":512,"concurrency":[2],"concurrent_requests":2,"concurrent_output_tokens":32,"llama_parallel":1,"llama_repeats":1,"ttft_slo_ms":10000,"tpot_slo_ms":10000,"skip_llama_server":false,"old_server":artifacts["old"],"new_server":artifacts["new"],"model_sha256":{"qwen3_dense":hash(&model())}},"profiles":{"qwen3_dense":{"correctness":correctness,"native":profile("native"),"old":profile("old"),"new":profile("new")}},"execution_seconds":120,"cell_seconds":30,"request_timeout_ms":5000})).unwrap()).unwrap();
        Self {
            directory,
            root,
            input,
            output,
        }
    }
    fn spec(&self) -> ProcessSpec {
        ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: [
                "automation".into(),
                "cache-family-run".into(),
                "--input".into(),
                self.input.clone().into_os_string(),
                "--output".into(),
                self.output.clone().into_os_string(),
            ]
            .into_iter()
            .map(Arg::Public)
            .collect(),
            cwd: self.root.clone(),
            environment: BTreeMap::new(),
        }
    }
    fn receipt(&self) -> Value {
        serde_json::from_slice(&fs::read(self.output.join("cache-family-matrix.json")).unwrap())
            .unwrap()
    }
}
fn invoke(spec: &ProcessSpec, cancel: &Cancellation) -> process::ProcessReport {
    let r = process::supervise(
        spec,
        &Limits {
            execution: Duration::from_secs(130),
            graceful_shutdown: Duration::from_secs(15),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        r.failure.is_none()
            && r.cleanup.complete
            && !r.cleanup.forced
            && !r.cleanup.graceful_signal_failed
            && r.cleanup.failure.is_none()
    );
    assert!(
        [&r.stdout, &r.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
    );
    r
}
#[test]
fn cache_matrix_actual_full_correctness_native_pair_and_later_arm_failure() {
    for mode in ["success", "later-failure", "post-run-identity-failure"] {
        let failure = mode != "success";
        let fixture = Fixture::new();
        if failure {
            fs::write(fixture.root.join("mode"), mode).unwrap();
        }
        let report = invoke(&fixture.spec(), &Cancellation::default());
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert_eq!(
            report.status.and_then(|s| s.code()),
            Some(i32::from(failure)),
            "matrix failure: {:?}; row: {}; correctness: {}; receipt: {}",
            report.stderr,
            fs::read_to_string(fixture.output.join("cell-0000/row.json")).unwrap_or_default(),
            fs::read_to_string(
                fixture
                    .output
                    .join("cell-0000/correctness/child.stderr.log")
            )
            .unwrap_or_default(),
            fs::read_to_string(
                fixture
                    .output
                    .join("cell-0000/correctness/output/cache-correctness-stage.json")
            )
            .unwrap_or_default()
        );
        let receipt = fixture.receipt();
        assert_eq!(receipt["rows"].as_array().unwrap().len(), 1);
        assert_eq!(
            receipt["status"],
            if failure { "incomplete" } else { "completed" }
        );
        let row: Value =
            serde_json::from_slice(&fs::read(fixture.output.join("cell-0000/row.json")).unwrap())
                .unwrap();
        let observed = row["observations"].as_array().unwrap();
        assert_eq!(observed.len(), 5);
        assert_eq!(observed[0]["cohort"], "native-serial");
        assert!(observed[0]["warm_statistics"]["warm_mean_ms"].is_null());
        assert_eq!(observed[1]["cohort"], "native-concurrent");
        assert_eq!(observed[2]["cohort"], "skippy-old");
        assert_eq!(observed[2]["status"], "completed");
        assert_eq!(observed[3]["cohort"], "skippy-new");
        if failure {
            assert!(
                observed[4]["parity"]["matches"] == false
                    && observed[4]["parity"]["complete"] == false,
                "refused owning cell cannot produce accepted parity"
            );
        } else {
            assert_eq!(observed[4]["parity"]["matches"], true);
        }
        if mode == "post-run-identity-failure" {
            let new_cell: Value = serde_json::from_slice(
                &fs::read(
                    fixture
                        .output
                        .join("cell-0000/sweep-skippy-new/output/cell.json"),
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(new_cell["status"], "incomplete");
            assert_eq!(new_cell["measurement"]["status"], "completed");
            assert!(!new_cell["after_identity_error"].is_null());
            assert_eq!(observed[3]["client_metrics"]["complete"], true);
        }
        assert!(receipt["promotion"].is_null());
        assert!(fixture.output.join("production-cache-bench.json").is_file());
        let report_text =
            fs::read_to_string(fixture.output.join("production-cache-bench.md")).unwrap();
        assert!(report_text.contains("Qwen3 dense"));
        let report_rows: Value = serde_json::from_slice(
            &fs::read(fixture.output.join("production-cache-bench.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report_rows[0]["skippy"]["status"], "pass");
        assert!(report_rows[0]["llama_server"]["warm_mean_ms"].is_null());
        fixture.directory.close().unwrap();
    }
}
#[test]
fn cache_matrix_actual_full_causal_later_arm_cancellation_preserves_prior_evidence() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("mode"), b"hold").unwrap();
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let spec = fixture.spec();
    let task = thread::spawn(move || invoke(&spec, &child_cancel));
    let until = Instant::now() + Duration::from_secs(30);
    while !fixture.root.join("post-new").exists() && Instant::now() < until {
        thread::sleep(Duration::from_millis(5));
    }
    let admitted = fixture.root.join("post-new").exists();
    cancel.cancel();
    let report = task.join().unwrap();
    assert!(admitted, "new-arm POST not admitted before cancellation");
    assert_eq!(report.outcome, process::Outcome::Cancelled);
    assert_ne!(fixture.receipt()["status"], "completed");
    let old: Value = serde_json::from_slice(
        &fs::read(
            fixture
                .output
                .join("cell-0000/sweep-skippy-old/output/cell.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(old["status"], "completed");
    fixture.directory.close().unwrap();
}
#[test]
#[ignore = "subprocess-only inert cache matrix host"]
fn inert_cache_matrix_host() {
    host::run();
}
#[test]
#[ignore = "subprocess-only inert cache matrix correctness report"]
fn inert_cache_matrix_correctness() {
    correctness::run();
}

fn install_consumed_profile(fixture: &Fixture, input: &mut Value) {
    let toolkit = input["profiles"]["qwen3_dense"]["correctness"]["native_build"].clone();
    let pin = input["profiles"]["qwen3_dense"]["correctness"]["native_build_sha256"].clone();
    let settings = json!({"OMP_NUM_THREADS":"2","LLAMA_STAGE_BACKEND":"cpu","SKIPPY_LLAMA_BACKEND":"cpu","LLAMA_STAGE_LINK_MODE":"static","LLAMA_STAGE_CUDA_ARCHITECTURES":"80","LLAMA_STAGE_AMDGPU_TARGETS":"gfx1100","LLAMA_STAGE_GGML_NATIVE":"OFF","GGML_CUDA_NO_VMM":"1","CUDA_VISIBLE_DEVICES":"2","HIP_VISIBLE_DEVICES":"3","ROCR_VISIBLE_DEVICES":"4"});
    let toolkits: serde_json::Map<String, Value> = [
        "CUDA_PATH",
        "HIP_PATH",
        "ROCM_PATH",
        "LLVMInstallDir",
        "VULKAN_SDK",
    ]
    .into_iter()
    .map(|key| (key.into(), json!({"path":toolkit,"sha256":pin})))
    .collect();
    let profile = &mut input["profiles"]["qwen3_dense"];
    profile["correctness"]["settings"] = settings.clone();
    profile["correctness"]["toolkit_directories"] = json!(toolkits);
    for side in ["native", "old", "new"] {
        profile[side]["environment"] = settings.clone();
        profile[side]["toolkit_directories"] = json!(toolkits);
    }
    for side in ["correctness", "native", "old", "new"] {
        let path = fixture.root.join(side);
        let mut bytes = fs::read_to_string(&path).unwrap();
        let mut checks = String::new();
        for (key, value) in settings.as_object().unwrap() {
            checks.push_str(&format!(
                "[ \"${key}\" = {} ] || exit 63\n",
                quote(value.as_str().unwrap())
            ));
        }
        for key in toolkits.keys() {
            checks.push_str(&format!(
                "[ \"${key}\" = {} ] || exit 62\n",
                quote(toolkit.as_str().unwrap())
            ));
        }
        bytes = bytes.replacen("#!/bin/sh\n", &format!("#!/bin/sh\n{checks}"), 1);
        fs::write(&path, &bytes).unwrap();
        if side == "correctness" {
            profile["correctness"]["correctness_sha256"] = json!(hash(bytes.as_bytes()));
        } else {
            profile[side]["binary_sha256"] = json!(hash(bytes.as_bytes()));
        }
    }
    input["profiles"]["qwen3_dense"]["correctness"]["stage_server_sha256"] =
        input["profiles"]["qwen3_dense"]["new"]["binary_sha256"].clone();
    for (side, field) in [("old", "old_server"), ("new", "new_server")] {
        input["plan"][field]["sha256"] =
            input["profiles"]["qwen3_dense"][side]["binary_sha256"].clone();
    }
}
#[test]
fn cache_matrix_actual_persistent_warm_hosts_and_consumed_device_toolkit_profile() {
    let fixture = Fixture::new();
    let mut input: Value = serde_json::from_slice(&fs::read(&fixture.input).unwrap()).unwrap();
    input["plan"]["concurrency"] = json!([1, 2]);
    install_consumed_profile(&fixture, &mut input);
    fs::write(&fixture.input, serde_json::to_vec(&input).unwrap()).unwrap();
    let report = invoke(&fixture.spec(), &Cancellation::default());
    assert!(report.success());
    let receipt = fixture.receipt();
    assert_eq!(receipt["status"], "completed");
    let row: Value =
        serde_json::from_slice(&fs::read(fixture.output.join("cell-0000/row.json")).unwrap())
            .unwrap();
    let observations = row["observations"].as_array().unwrap();
    assert_eq!(observations.len(), 9);
    for side in ["old", "new"] {
        assert_eq!(
            fs::read_to_string(fixture.root.join(format!("host-launches-{side}")))
                .unwrap()
                .lines()
                .count(),
            1,
            "persistent host must span both rungs"
        );
        let cell: Value = serde_json::from_slice(
            &fs::read(
                fixture
                    .output
                    .join(format!("cell-0000/sweep-skippy-{side}/output/cell.json")),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(cell["measurement"]["sweep"].as_array().unwrap().len(), 2);
        for (index, concurrency) in [1, 2].into_iter().enumerate() {
            assert_eq!(
                cell["measurement"]["sweep"][index]["concurrency"],
                concurrency
            );
            assert_eq!(
                cell["measurement"]["sweep"][index]["measurement"]["status"],
                "completed"
            );
        }
    }
    assert_eq!(
        fs::read_to_string(fixture.root.join("host-launches-native"))
            .unwrap()
            .lines()
            .count(),
        2,
        "one serial baseline plus one full concurrent curve"
    );
    for index in [4, 8] {
        assert_eq!(observations[index]["parity"]["matches"], true);
    }
    fixture.directory.close().unwrap();
}

fn operator_input(fixture: &Fixture) -> Value {
    let input: Value = serde_json::from_slice(&fs::read(&fixture.input).unwrap()).unwrap();
    let profile = &input["profiles"]["qwen3_dense"];
    json!({"schema_version":1,"plan":input["plan"],"correctness":profile["correctness"]["correctness"],"stage_server":profile["correctness"]["stage_server"],"native_server":profile["native"]["binary"],"artifact_tool":null,"native_build":profile["correctness"]["native_build"],"old_source_commit":"a".repeat(40),"new_source_commit":"a".repeat(40),"native_source_commit":"a".repeat(40),"environment":{"OMP_NUM_THREADS":"2"},"toolkit_directories":{},"execution_seconds":120,"cell_seconds":30,"request_timeout_ms":5000,"preparation_seconds":60})
}
fn preparation_spec(
    fixture: &Fixture,
    input: &std::path::Path,
    output: &std::path::Path,
    mode: &str,
) -> ProcessSpec {
    let mut spec = fixture.spec();
    spec.arguments = [
        "automation".into(),
        "cache-family-run".into(),
        mode.into(),
        "--input".into(),
        input.to_path_buf().into_os_string(),
        "--output".into(),
        output.to_path_buf().into_os_string(),
    ]
    .into_iter()
    .map(Arg::Public)
    .collect();
    spec
}
#[test]
fn cache_operator_actual_preparation_observes_pins_and_runs_complete_inert_producer() {
    let mut fixture = Fixture::new();
    let operator = fixture.root.join("operator.json");
    let prepared = fixture.root.join("prepared");
    fs::write(
        &operator,
        serde_json::to_vec(&operator_input(&fixture)).unwrap(),
    )
    .unwrap();
    let report = invoke(
        &preparation_spec(&fixture, &operator, &prepared, "prepare-full"),
        &Cancellation::default(),
    );
    assert!(report.success());
    let observed: Value =
        serde_json::from_slice(&fs::read(prepared.join("observations.json")).unwrap()).unwrap();
    let request = fs::read(prepared.join("request.json")).unwrap();
    assert_eq!(observed["request_sha256"], hash(&request));
    fixture.input = prepared.join("cache-family-input.json");
    let input: Value = serde_json::from_slice(&fs::read(&fixture.input).unwrap()).unwrap();
    assert_eq!(input["profiles"].as_object().unwrap().len(), 1);
    assert_eq!(input["plan"]["use_cases"], json!([]));
    assert_eq!(
        input["profiles"]["qwen3_dense"]["correctness"]["model_sha256"],
        hash(&model())
    );
    assert_eq!(
        input["profiles"]["qwen3_dense"]["old"]["environment"]["OMP_NUM_THREADS"],
        "2"
    );
    let result = invoke(&fixture.spec(), &Cancellation::default());
    assert!(result.success());
    assert_eq!(fixture.receipt()["status"], "completed");
    fixture.directory.close().unwrap();
}
#[test]
fn cache_operator_actual_usecase_all_and_optional_pair_missing_model_selection() {
    let fixture = Fixture::new();
    let mut operator = operator_input(&fixture);
    let corpus = fixture.root.join("corpus.json");
    let bytes=serde_json::to_vec(&json!({"version":1,"use_cases":[{"key":"one","label":"One","prompt":"fixed prompt","prefix_tokens":64,"source":{}},{"key":"two","label":"Two","prompt":"fixed prompt","prefix_tokens":64,"source":{}}]})).unwrap();
    fs::write(&corpus, &bytes).unwrap();
    operator["plan"]["corpus"] = json!({"path":corpus,"sha256":hash(&bytes)});
    operator["plan"]["cases"] = json!(["qwen3_dense", "llama"]);
    operator["plan"]["old_server"] = Value::Null;
    operator["plan"]["new_server"] = Value::Null;
    let path = fixture.root.join("operator.json");
    fs::write(&path, serde_json::to_vec(&operator).unwrap()).unwrap();
    let prepared = fixture.root.join("prepared");
    let result = invoke(
        &preparation_spec(&fixture, &path, &prepared, "prepare-use-cases"),
        &Cancellation::default(),
    );
    assert!(result.success());
    let input: Value =
        serde_json::from_slice(&fs::read(prepared.join("cache-family-input.json")).unwrap())
            .unwrap();
    assert_eq!(input["plan"]["use_cases"], json!(["all"]));
    assert_eq!(input["plan"]["cases"], json!(["qwen3_dense", "llama"]));
    assert_eq!(input["profiles"].as_object().unwrap().len(), 1);
    assert!(input["profiles"]["qwen3_dense"]["old"].is_null());
    assert!(input["profiles"]["qwen3_dense"]["new"].is_null());
    assert_eq!(
        input["profiles"]["qwen3_dense"]["correctness"]["stage_server"],
        json!(fixture.root.join("new"))
    );
    fixture.directory.close().unwrap();
}
#[test]
fn cache_operator_actual_mismatched_pin_and_fifo_refuse_eligible_input() {
    for fifo in [false, true] {
        let fixture = Fixture::new();
        let mut operator = operator_input(&fixture);
        if fifo {
            use std::os::unix::ffi::OsStrExt as _;
            let path = fixture.root.join("tool-fifo");
            let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            operator["correctness"] = json!(path);
        } else {
            operator["plan"]["old_server"]["sha256"] = json!("0".repeat(64));
        }
        let path = fixture.root.join("operator.json");
        fs::write(&path, serde_json::to_vec(&operator).unwrap()).unwrap();
        let prepared = fixture.root.join("prepared");
        let result = invoke(
            &preparation_spec(&fixture, &path, &prepared, "prepare-full"),
            &Cancellation::default(),
        );
        assert_eq!(result.status.and_then(|s| s.code()), Some(1));
        assert!(!prepared.join("cache-family-input.json").exists());
        fixture.directory.close().unwrap();
    }
}

#[test]
fn cache_operator_actual_wrapper_full_usecase_report_and_failure_dispatch() {
    use std::os::unix::fs::PermissionsExt as _;
    for fail in [false, true] {
        let fixture = Fixture::new();
        let script = fixture
            .root
            .join("skippy/evals/skippy-cache-family-bench.sh");
        fs::create_dir_all(script.parent().unwrap()).unwrap();
        fs::write(
            &script,
            include_str!("../../../../skippy/evals/skippy-cache-family-bench.sh"),
        )
        .unwrap();
        let automation = fixture.root.join("automation");
        let record = fixture.root.join("dispatch.log");
        let body = format!(
            "#!/bin/bash\nprintf '%s\\n' \"$*\" >> {}\n{}\n",
            quote(record.to_str().unwrap()),
            if fail {
                "if [[ $3 == prepare-use-cases ]]; then exit 41; fi"
            } else {
                ":"
            }
        );
        fs::write(&automation, body).unwrap();
        fs::set_permissions(&automation, fs::Permissions::from_mode(0o700)).unwrap();
        let operator = fixture.root.join("operator.json");
        fs::write(&operator, b"{}").unwrap();
        let output = fixture.root.join("wrapper-output");
        let spec = ProcessSpec {
            executable: "/bin/bash".into(),
            arguments: [script.into_os_string(), output.into_os_string()]
                .into_iter()
                .map(Arg::Public)
                .collect(),
            cwd: fixture.root.clone(),
            environment: [
                ("PATH", "/usr/bin:/bin".into()),
                ("MESH_LLM_AUTOMATION_BIN", automation.into_os_string()),
                ("SKIPPY_CACHE_OPERATOR_INPUT", operator.into_os_string()),
                ("SKIPPY_CACHE_SKIP_BUILD", "1".into()),
            ]
            .into_iter()
            .map(|(k, v)| (k.into(), Arg::Public(v)))
            .collect(),
        };
        let result = invoke(&spec, &Cancellation::default());
        assert_eq!(
            result.status.and_then(|s| s.code()),
            Some(if fail { 41 } else { 0 })
        );
        let lines = fs::read_to_string(record).unwrap();
        let lines: Vec<_> = lines.lines().collect();
        assert_eq!(lines.len(), if fail { 3 } else { 5 });
        assert!(lines[0].starts_with("automation cache-family-run prepare-full --input "));
        assert!(lines[0].contains("--prefix-tokens 128 --runtime-lane-count 1 --llama-parallel 1 --llama-repeats 3 --cache-hit-repeats 3"));
        assert!(lines[2].starts_with("automation cache-family-run prepare-use-cases --input "));
        if !fail {
            assert!(lines[4].starts_with("automation cache-family-report --input "));
            assert!(lines[4].contains("/full-gguf/production-cache-bench.json --input "));
            assert!(lines[4].contains("/use-cases/production-cache-bench.json --use-case-corpus "));
        }
        fixture.directory.close().unwrap();
    }
}

#[test]
fn cache_operator_actual_minimax_snapshot_links_preserve_runtime_shard_and_sibling_pins() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    let mut operator = operator_input(&fixture);
    let catalog: Value = serde_json::from_str(include_str!(
        "../../src/automation/cache_family_plan/catalog.json"
    ))
    .unwrap();
    let case = catalog
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["key"] == "minimax_m27")
        .unwrap();
    let cache = PathBuf::from(operator["plan"]["cache_root"].as_str().unwrap());
    let first = cache.join(case["snapshot_relative"].as_str().unwrap());
    fs::create_dir_all(first.parent().unwrap()).unwrap();
    let blobs = cache.join("blobs");
    fs::create_dir(&blobs).unwrap();
    let mut pins = serde_json::Map::new();
    for i in 1..=3 {
        let bytes = format!("supplied inert MiniMax shard {i}");
        let blob = blobs.join(hash(bytes.as_bytes()));
        fs::write(&blob, bytes.as_bytes()).unwrap();
        let name = format!("MiniMax-M2.7-UD-Q2_K_XL-{i:05}-of-00003.gguf");
        symlink(&blob, first.parent().unwrap().join(&name)).unwrap();
        pins.insert(name, json!(hash(bytes.as_bytes())));
    }
    operator["plan"]["cases"] = json!(["minimax_m27"]);
    operator["plan"]["model_sha256"] =
        json!({"minimax_m27":pins[first.file_name().unwrap().to_str().unwrap()]});
    operator["artifact_tool"] = json!(fixture.root.join("native"));
    let input = fixture.root.join("operator.json");
    fs::write(&input, serde_json::to_vec(&operator).unwrap()).unwrap();
    let prepared = fixture.root.join("prepared");
    let result = invoke(
        &preparation_spec(&fixture, &input, &prepared, "prepare-full"),
        &Cancellation::default(),
    );
    assert!(result.success());
    let value: Value =
        serde_json::from_slice(&fs::read(prepared.join("cache-family-input.json")).unwrap())
            .unwrap();
    let profile = &value["profiles"]["minimax_m27"];
    assert_ne!(first, first.canonicalize().unwrap());
    assert_eq!(profile["correctness"]["model"], json!(first));
    assert_eq!(
        profile["correctness"]["artifact"]["shard_pins"],
        json!(pins)
    );
    for arm in ["native", "old", "new"] {
        assert_eq!(profile[arm]["model"], json!(first));
        assert_eq!(profile[arm]["artifact"], profile["correctness"]["artifact"]);
    }
    fixture.directory.close().unwrap();
}

#[test]
fn cache_operator_actual_source_tree_output_refusal_precedes_any_mutation() {
    for kind in ["build", "cache", "toolkit"] {
        let fixture = Fixture::new();
        let mut operator = operator_input(&fixture);
        let mut source = PathBuf::from(
            operator[if kind == "cache" {
                "plan"
            } else {
                "native_build"
            }]
            .as_str()
            .unwrap_or_else(|| operator["plan"]["cache_root"].as_str().unwrap()),
        );
        if kind == "toolkit" {
            source = fixture.root.join("independent-toolkit");
            fs::create_dir(&source).unwrap();
            fs::write(source.join("version.txt"), b"pinned inert toolkit").unwrap();
            operator["toolkit_directories"] =
                json!({"CUDA_PATH":{"path":source,"sha256":"a".repeat(64)}});
        }
        let output = source.join("must-not-create");
        let before = fs::read_dir(&source).unwrap().count();
        let input = fixture.root.join("operator.json");
        fs::write(&input, serde_json::to_vec(&operator).unwrap()).unwrap();
        let result = invoke(
            &preparation_spec(&fixture, &input, &output, "prepare-full"),
            &Cancellation::default(),
        );
        assert_eq!(result.status.and_then(|s| s.code()), Some(1));
        assert!(!output.exists());
        assert_eq!(fs::read_dir(&source).unwrap().count(), before);
        if kind == "toolkit" {
            assert_eq!(
                fs::read(source.join("version.txt")).unwrap(),
                b"pinned inert toolkit"
            );
        }
        fixture.directory.close().unwrap();
    }
}
