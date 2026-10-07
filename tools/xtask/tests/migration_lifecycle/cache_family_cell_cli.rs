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
#[path = "cache_family_cell_cli/host_fixture.rs"]
mod host_fixture;
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn string(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend((text.len() as u64).to_le_bytes());
    bytes.extend(text.as_bytes());
}
fn model() -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(4_u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, "llama");
    for (k, v) in [
        ("llama.context_length", 512_u32),
        ("llama.block_count", 6),
        ("llama.embedding_length", 16),
    ] {
        string(&mut bytes, k);
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(v.to_le_bytes());
    }
    bytes
}
fn quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
struct Fixture {
    directory: tempfile::TempDir,
    root: PathBuf,
    input: PathBuf,
    output: PathBuf,
    port: u16,
}
impl Fixture {
    fn new(host: &str, cohort: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        fs::write(root.join("model.gguf"), model()).unwrap();
        fs::create_dir(root.join("native-build")).unwrap();
        fs::write(root.join("native-build/libfixture.so"), b"inert").unwrap();
        let mut tree = Sha256::new();
        tree.update(13_u64.to_be_bytes());
        tree.update(b"libfixture.so");
        tree.update(Sha256::digest(b"inert"));
        let tree = hex::encode(tree.finalize());
        let executable = root.join("host");
        let body = format!(
            r#"#!/bin/sh
export CACHE_FIXTURE_ROOT={root}
printf '%s\n' "$@" > "$CACHE_FIXTURE_ROOT/argv.txt"
if [ "$1" = serve-openai ]; then
 export CACHE_FIXTURE_NATIVE=0
 shift
else
 export CACHE_FIXTURE_NATIVE=1
fi
while [ "$#" -gt 0 ]; do
 case "$1" in
 --model) export CACHE_FIXTURE_MODEL="$2"; shift 2;;
 --config) export CACHE_FIXTURE_CONFIG="$2"; shift 2;;
 --bind-addr) export CACHE_FIXTURE_BIND="$2"; shift 2;;
 --port) export CACHE_FIXTURE_BIND="127.0.0.1:$2"; shift 2;;
 --host) [ "$2" = 127.0.0.1 ] || exit 64; shift 2;;
 --ctx-size) [ "$2" = 128 ] || exit 64; shift 2;;
 --n-gpu-layers) [ "$2" = -1 ] || exit 64; shift 2;;
 --parallel|--generation-concurrency) [ "$2" = 1 ] || exit 64; shift 2;;
 --telemetry-level) [ "$2" = debug ] || exit 64; shift 2;;
 --no-webui) shift;;
 *) exit 64;;
 esac
done
exec {helper} --ignored --exact cache_family_cell_cli::inert_cache_host --nocapture
"#,
            root = quoted(root.to_str().unwrap()),
            helper = quoted(std::env::current_exe().unwrap().to_str().unwrap())
        );
        let mut body = body;
        if host == "skippy-new" {
            body = body.replace("serve-openai", "serve").replace(
                " export CACHE_FIXTURE_NATIVE=0",
                " export CACHE_FIXTURE_CURRENT=1\n export CACHE_FIXTURE_NATIVE=0",
            );
        }
        let verb = if host == "skippy-new" {
            "serve"
        } else {
            "serve-openai"
        };
        body = body.replace("#!/bin/sh\n", &format!("#!/bin/sh\nif [ \"$#\" = 2 ] && [ \"$2\" = --help ]; then [ \"$1\" = {verb} ] || exit 64; printf 'inert help\\n'; exit 0; fi\n"));
        fs::write(&executable, &body).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let native = host == "native-baseline";
        let serial = cohort == "native-serial";
        let input = root.join("input.json");
        let output = root.join("cell");
        fs::write(&input,serde_json::to_vec(&json!({"schema_version":1,"host":host,"binary":executable,"binary_sha256":hash(body.as_bytes()),"source_commit":"a".repeat(40),"native_build":root.join("native-build"),"native_build_sha256":tree,"model":root.join("model.gguf"),"model_sha256":hash(&model()),"model_id":"fixture","layer_end":6,"ctx_size":128,"lane_count":1,"n_gpu_layers":-1,"port":port,"environment":{"OMP_NUM_THREADS":"2"},"worker":{"schema_version":1,"cohort":cohort,"base_url":format!("http://127.0.0.1:{port}{}",if native{"/"}else{"/v1"}),"prompt":"fixed prompt","model_id":if native{None}else{Some("fixture")},"requests":if serial{3}else{2},"concurrency":if serial{1}else{2},"output_tokens":if serial{1}else if native{128}else{32},"request_timeout_ms":5000,"execution_timeout_ms":10000},"startup_timeout_secs":4,"execution_timeout_secs":30})).unwrap()).unwrap();
        Self {
            directory,
            root,
            input,
            output,
            port,
        }
    }
    fn spec(&self) -> ProcessSpec {
        ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: [
                "automation".into(),
                "cache-family-cell".into(),
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
        serde_json::from_slice(&fs::read(self.output.join("cell.json")).unwrap()).unwrap()
    }
}
fn invoke(spec: &ProcessSpec, cancel: &Cancellation) -> process::ProcessReport {
    let report = process::supervise(
        spec,
        &Limits {
            execution: Duration::from_secs(35),
            graceful_shutdown: Duration::from_secs(2),
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
        report.failure.is_none()
            && report.cleanup.complete
            && !report.cleanup.forced
            && !report.cleanup.graceful_signal_failed
            && report.cleanup.failure.is_none()
    );
    assert!(
        [&report.stdout, &report.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
    );
    report
}
#[test]
fn cache_cell_actual_cli_preserves_native_and_old_new_serving_profiles() {
    for (host, cohort) in [
        ("native-baseline", "native-serial"),
        ("native-baseline", "native-concurrent"),
        ("skippy-old", "openai-concurrent"),
        ("skippy-new", "openai-concurrent"),
    ] {
        let fixture = Fixture::new(host, cohort);
        let report = invoke(&fixture.spec(), &Cancellation::default());
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert_eq!(
            report
                .status
                .as_ref()
                .and_then(std::process::ExitStatus::code),
            Some(0)
        );
        let receipt = fixture.receipt();
        assert_eq!(receipt["status"], "completed");
        assert_eq!(receipt["measurement"]["status"], "completed");
        assert_eq!(receipt["readiness"]["ready"], true);
        assert_eq!(
            receipt["before_identity"]["admitted"]["model"],
            json!(fixture.root.join("model.gguf"))
        );
        assert_eq!(
            receipt["before_identity"]["model_identity"],
            receipt["after_identity"]["model_identity"]
        );
        assert_eq!(receipt["effective_environment"]["OMP_NUM_THREADS"], "2");
        assert_eq!(
            receipt["effective_environment"]["LLAMA_STAGE_BUILD_DIR"],
            json!(fixture.root.join("native-build"))
        );
        assert_eq!(receipt["lifecycle"]["members"].as_array().unwrap().len(), 3);
        for m in receipt["lifecycle"]["members"].as_array().unwrap() {
            assert_eq!(m["clean"], true);
            assert_eq!(m["status"], 0);
        }
        let argv = fs::read_to_string(fixture.root.join("argv.txt")).unwrap();
        assert!(argv.contains(if host == "native-baseline" {
            "--parallel\n1"
        } else {
            "--generation-concurrency\n1"
        }));
        let rebinding =
            std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, fixture.port)).unwrap();
        drop(rebinding);
        fixture.directory.close().unwrap();
    }
}
#[test]
fn cache_cell_actual_cli_refuses_occupied_port_and_bad_pin_before_host_publication() {
    for occupied in [true, false] {
        let fixture = Fixture::new("skippy-new", "openai-concurrent");
        let reservation = if occupied {
            Some(
                std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, fixture.port)).unwrap(),
            )
        } else {
            None
        };
        if !occupied {
            let mut input: Value =
                serde_json::from_slice(&fs::read(&fixture.input).unwrap()).unwrap();
            input["model_sha256"] = json!("d".repeat(64));
            fs::write(&fixture.input, serde_json::to_vec(&input).unwrap()).unwrap();
        }
        let report = invoke(&fixture.spec(), &Cancellation::default());
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert_eq!(
            report
                .status
                .as_ref()
                .and_then(std::process::ExitStatus::code),
            Some(1)
        );
        assert_eq!(fixture.receipt()["status"], "failed");
        assert!(!fixture.root.join("argv.txt").exists());
        assert!(!fixture.output.join("measurement.json").exists());
        drop(reservation);
        fixture.directory.close().unwrap();
    }
}
#[test]
fn cache_cell_actual_cli_readiness_refusal_has_no_measured_requests() {
    for mode in ["bad-readiness", "early-exit"] {
        let fixture = Fixture::new("skippy-new", "openai-concurrent");
        fs::write(fixture.root.join("mode"), mode.as_bytes()).unwrap();
        let report = invoke(&fixture.spec(), &Cancellation::default());
        assert_eq!(
            report
                .status
                .as_ref()
                .and_then(std::process::ExitStatus::code),
            Some(1)
        );
        assert!(!fixture.root.join("post-admitted").exists());
        assert!(!fixture.output.join("measurement.json").exists());
        assert_ne!(fixture.receipt()["status"], "completed");
        fixture.directory.close().unwrap();
    }
}

#[test]
fn cache_cell_actual_cli_causal_inflight_cancel_retains_partial_and_releases_host() {
    let fixture = Fixture::new("skippy-new", "openai-concurrent");
    fs::write(fixture.root.join("mode"), b"hold").unwrap();
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let spec = fixture.spec();
    let task = thread::spawn(move || invoke(&spec, &child_cancel));
    let until = Instant::now() + Duration::from_secs(10);
    while !fixture.root.join("post-admitted").exists() && Instant::now() < until {
        thread::sleep(Duration::from_millis(5));
    }
    let admitted = fixture.root.join("post-admitted").exists();
    cancel.cancel();
    let report = task.join().unwrap();
    assert!(admitted, "no actual POST before cancellation");
    assert_eq!(report.outcome, process::Outcome::Cancelled);
    assert_ne!(fixture.receipt()["status"], "completed");
    let rebound =
        std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, fixture.port)).unwrap();
    drop(rebound);
    fixture.directory.close().unwrap();
}
#[test]
#[ignore = "subprocess-only inert cache host"]
fn inert_cache_host() {
    host_fixture::run();
}

#[test]
fn cache_cell_actual_cli_owned_unforced_signal_stop_is_recorded_without_shutdown_attestation() {
    let fixture = Fixture::new("skippy-new", "openai-concurrent");
    fs::write(fixture.root.join("mode"), b"signal-stop").unwrap();
    let report = invoke(&fixture.spec(), &Cancellation::default());
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert_eq!(
        report
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code),
        Some(0)
    );
    let receipt = fixture.receipt();
    assert_eq!(receipt["status"], "completed");
    let host = receipt["lifecycle"]["members"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["member"] == "seed")
        .unwrap();
    assert_eq!(host["signal"], libc::SIGTERM);
    assert!(host["status"].is_null());
    assert_eq!(host["disposition"], "intentional-stop");
    assert_eq!(host["forced"], false);
    assert_eq!(host["clean"], true);
    fixture.directory.close().unwrap();
}
