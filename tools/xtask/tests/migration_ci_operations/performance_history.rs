//! Actual bounded native CLI proofs for all six original competitive history intents.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};
struct Fixture {
    temp: tempfile::TempDir,
    artifact: PathBuf,
    result: PathBuf,
    marker: PathBuf,
    output: PathBuf,
    report: PathBuf,
    baseline: PathBuf,
}
impl Fixture {
    fn new(arm: &str) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let artifact = root.join("competitive artifact");
        let result = artifact.join(format!("trace/cuda/qwen3/c-64/{arm}/result.json"));
        let marker = result.with_file_name("complete.json");
        let output = root.join("row.jsonl");
        let report = root.join("report.md");
        let baseline = root.join("baseline");
        fs::create_dir_all(result.parent().unwrap()).unwrap();
        fs::create_dir_all(artifact.join("provenance")).unwrap();
        fs::create_dir_all(&baseline).unwrap();
        let cell = json!({"platform":"cuda","model":"qwen3","arm":arm,"workload":"thoughtworks","concurrency":64,"config_sha256":"c".repeat(64),"manifest_sha256":"p".repeat(64),"binary_sha256":"b".repeat(64),"comparison_capacity_policy":"matched"});
        write(&marker, &json!({"cell_sha256":digest(&cell),"cell":cell}));
        write(
            &result,
            &json!({"prompt_count":64,"successful_requests":64,"failed_requests":0,"output_tokens":2048,"measured_wall_ms":20480.0,"output_tokens_per_second":100.0,"ttft_ms_mean":500.0}),
        );
        write(
            &artifact.join("provenance/cuda.json"),
            &json!({"created_utc":"2026-09-01T00:00:00Z","platform_details":"Linux-fixture","mesh_head":"a".repeat(40),"native_runtime_directory_sha256":"n".repeat(64),"models":{"qwen3":"m".repeat(64)}}),
        );
        fs::write(
            artifact.join("runner-gpu.csv"),
            "RTX 5080,GPU-fixture,12.0,999.1,0000:01:00.0,P0,42,2400,12000\n",
        )
        .unwrap();
        Self {
            temp,
            artifact,
            result,
            marker,
            output,
            report,
            baseline,
        }
    }
    fn run(&self, extra: &[&str]) -> process::RawProcessReport {
        let mut args = vec![
            "ci-ops".into(),
            "performance-history".into(),
            "--artifact".into(),
            self.artifact.to_string_lossy().into_owned(),
            "--output".into(),
            self.output.to_string_lossy().into_owned(),
            "--report".into(),
            self.report.to_string_lossy().into_owned(),
        ];
        args.extend(extra.iter().map(|arg| (*arg).to_owned()));
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                cwd: self.temp.path().to_owned(),
                environment: BTreeMap::new(),
                arguments: args
                    .into_iter()
                    .map(|s: String| Value::Public(s.into()))
                    .collect(),
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
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(report.process.failure.is_none(), "{:?}", report.process);
        assert!(report.process.cleanup.complete);
        report
    }
    fn row(&self) -> Json {
        serde_json::from_str(
            fs::read_to_string(&self.output)
                .unwrap()
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap()
    }
    fn prior(&self, rows: &[Json]) {
        let path = self.baseline.join("data/runs/immutable/1.jsonl");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            path,
            rows.iter()
                .map(|row| serde_json::to_string(row).unwrap() + "\n")
                .collect::<String>(),
        )
        .unwrap();
    }
    fn status(&self, extra: &[&str], code: i32) {
        let result = self.run(extra);
        assert_eq!(
            result.process.status.unwrap().code(),
            Some(code),
            "{}",
            String::from_utf8_lossy(result.stderr.unwrap().as_bytes())
        );
    }
    fn preserved_failure(&self, extra: &[&str]) {
        fs::write(&self.output, b"old shard").unwrap();
        fs::write(&self.report, b"old report").unwrap();
        self.status(extra, 1);
        assert_eq!(fs::read(&self.output).unwrap(), b"old shard");
        assert_eq!(fs::read(&self.report).unwrap(), b"old report");
    }
}
fn write(path: &Path, value: &Json) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn read(path: &Path) -> Json {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn digest(value: &Json) -> String {
    let sorted: BTreeMap<_, _> = value.as_object().unwrap().iter().collect();
    hex::encode(Sha256::digest(serde_json::to_vec(&sorted).unwrap()))
}
#[test]
fn performance_history_normalizes_consumed_schema1_privacy_and_observed_state() {
    let fixture = Fixture::new("mesh");
    fixture.status(&[], 0);
    let row = fixture.row();
    assert_eq!(row["schema_version"], 1);
    assert_eq!(row["complete"], true);
    assert_eq!(row["cohort"]["hardware"]["name"], "RTX 5080");
    assert!(row["cohort"]["hardware"].get("uuid").is_none());
    assert!(row["cohort"]["hardware"].get("pci_bus_id").is_none());
    assert_eq!(
        row["cohort"]["hardware"]["gpu_identity_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(row["observed_gpu_state"]["temperature_c"], "42");
    assert_eq!(row["cohort"]["concurrency"], 64);
    assert_eq!(row["cohort_key"].as_str().unwrap().len(), 64);
    assert_eq!(row["output_tokens_per_second"], 100.0);
    assert_eq!(
        row["cohort_key"],
        "6468340e022c7ac2293d1d2dc894927f996835ee208bdaaaabe33636c1a95a2e"
    );
    assert_eq!(
        row["cohort"]["hardware"]["gpu_identity_sha256"],
        "3ae8fd76506a7f2c04c2e3927435b3f920742db8f1850fcf47d085e7eab39bb2"
    );
    assert_eq!(
        row["artifact_result"],
        "trace/cuda/qwen3/c-64/mesh/result.json"
    );
    fs::write(
        fixture.artifact.join("runner-gpu.csv"),
        "\"RTX, 5080\",GPU-fixture,12.0,999.1,0000:01:00.0,P0,43,2400,12000\n",
    )
    .unwrap();
    fixture.status(&[], 0);
    assert_eq!(fixture.row()["cohort"]["hardware"]["name"], "RTX, 5080");
}
#[test]
fn performance_history_missing_incomplete_or_invalid_gpu_preserves_outputs() {
    let fixture = Fixture::new("mesh");
    let path = fixture.artifact.join("runner-gpu.csv");
    fs::remove_file(&path).unwrap();
    fixture.preserved_failure(&[]);
    for text in [
        "RTX 5080,,12.0,999.1,0000:01:00.0,P0,42,2400,12000\n",
        "a,b,c\n",
        "a,b,c,d,e,f,g,h,i\na,b,c,d,e,f,g,h,i\n",
    ] {
        fs::write(&path, text).unwrap();
        fixture.preserved_failure(&[]);
    }
}
#[test]
fn performance_history_external_upgrade_changes_exact_cohort() {
    let fixture = Fixture::new("vllm");
    fixture.status(&[], 0);
    let first = fixture.row();
    let mut marker = read(&fixture.marker);
    marker["cell"]["binary_sha256"] = "d".repeat(64).into();
    marker["cell_sha256"] = digest(&marker["cell"]).into();
    write(&fixture.marker, &marker);
    fixture.status(&[], 0);
    assert_ne!(first["cohort_key"], fixture.row()["cohort_key"]);
    marker["cell_sha256"] = "0".repeat(64).into();
    write(&fixture.marker, &marker);
    fixture.preserved_failure(&[]);
}
#[test]
fn performance_history_candidate_mesh_changes_stay_in_same_control_cohort() {
    for arm in ["mesh", "mesh-adaptive"] {
        let fixture = Fixture::new(arm);
        fixture.status(&[], 0);
        let first = fixture.row();
        let mut marker = read(&fixture.marker);
        marker["cell"]["binary_sha256"] = "d".repeat(64).into();
        marker["cell_sha256"] = digest(&marker["cell"]).into();
        write(&fixture.marker, &marker);
        fixture.status(&[], 0);
        assert_eq!(first["cohort_key"], fixture.row()["cohort_key"]);
        assert_ne!(
            first["backend_binary_sha256"],
            fixture.row()["backend_binary_sha256"]
        );
    }
}
#[test]
fn performance_history_three_exact_prior_sources_median_mad_and_gate_are_truthful() {
    let fixture = Fixture::new("mesh");
    fixture.status(&[], 0);
    let row = fixture.row();
    let baseline = (0..3)
        .map(|i| {
            let mut prior = row.clone();
            prior["source_sha"] = i.to_string().repeat(40).into();
            prior["output_tokens_per_second"] = json!([100.0, 101.0, 99.0][i]);
            prior
        })
        .collect::<Vec<_>>();
    fixture.prior(&baseline);
    let mut result = read(&fixture.result);
    result["output_tokens_per_second"] = 80.0.into();
    write(&fixture.result, &result);
    let baseline_path = fixture.baseline.to_str().unwrap();
    fixture.status(&["--baseline", baseline_path], 0);
    assert!(
        fs::read_to_string(&fixture.report)
            .unwrap()
            .contains("performance-regression")
    );
    fixture.status(&["--baseline", baseline_path, "--gate"], 1);
    assert_eq!(fixture.row()["output_tokens_per_second"], 80.0);
    let mut unrelated = baseline.clone();
    for prior in &mut unrelated {
        prior["cohort_key"] = "different".into();
    }
    fixture.prior(&unrelated);
    fixture.status(&["--baseline", baseline_path, "--gate"], 0);
    assert!(
        fs::read_to_string(&fixture.report)
            .unwrap()
            .contains("insufficient-baseline")
    );
    let mut noisy = baseline.clone();
    for (prior, rate) in noisy.iter_mut().zip([50.0, 100.0, 150.0]) {
        prior["output_tokens_per_second"] = rate.into();
    }
    fixture.prior(&noisy);
    fixture.status(&["--baseline", baseline_path, "--gate"], 0);
    assert!(
        fs::read_to_string(&fixture.report)
            .unwrap()
            .contains("| pass |")
    );
    fixture.prior(&baseline);
    result["output_tokens_per_second"] = 100.0.into();
    result["ttft_ms_mean"] = 600.0.into();
    write(&fixture.result, &result);
    fixture.status(&["--baseline", baseline_path, "--gate"], 1);
    let mut same = baseline.clone();
    for prior in &mut same {
        prior["source_sha"] = row["source_sha"].clone();
    }
    fixture.prior(&same);
    fixture.status(&["--baseline", baseline_path, "--gate"], 0);
    assert!(
        fs::read_to_string(&fixture.report)
            .unwrap()
            .contains("insufficient-baseline")
    );
    let mut zero = baseline.clone();
    for prior in &mut zero {
        prior["ttft_ms_mean"] = 0.0.into();
    }
    fixture.prior(&zero);
    result["output_tokens_per_second"] = 80.0.into();
    write(&fixture.result, &result);
    fixture.status(&["--baseline", baseline_path, "--gate"], 1);
}
#[test]
fn performance_history_immutable_shard_loading_and_invalid_input_preserve_both_outputs() {
    let fixture = Fixture::new("mesh");
    fixture.status(&[], 0);
    let row = fixture.row();
    let mut prior = row.clone();
    prior["source_sha"] = "1".repeat(40).into();
    fixture.prior(&[prior.clone(), prior.clone(), prior]);
    let baseline = fixture.baseline.to_str().unwrap();
    fixture.status(&["--baseline", baseline], 0);
    assert!(
        fs::read_to_string(&fixture.report)
            .unwrap()
            .contains("| 3 |")
    );
    fs::write(fixture.baseline.join("bad.jsonl"), "malformed\n").unwrap();
    fixture.preserved_failure(&["--baseline", baseline]);
    fs::remove_file(fixture.baseline.join("bad.jsonl")).unwrap();
    let mut result = read(&fixture.result);
    result["failed_requests"] = 1.into();
    result["successful_requests"] = 63.into();
    write(&fixture.result, &result);
    fixture.status(&["--gate"], 1);
    assert_eq!(fixture.row()["complete"], false);
    assert!(
        fs::read_to_string(&fixture.report)
            .unwrap()
            .contains("correctness-failure")
    );
    #[cfg(unix)]
    {
        fs::remove_file(fixture.artifact.join("runner-gpu.csv")).unwrap();
        std::os::unix::fs::symlink(&fixture.result, fixture.artifact.join("runner-gpu.csv"))
            .unwrap();
        fixture.preserved_failure(&[]);
    }
    #[cfg(unix)]
    {
        let gpu = fixture.artifact.join("runner-gpu.csv");
        fs::remove_file(&gpu).unwrap();
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(gpu.as_os_str().as_bytes()).unwrap();
        // SAFETY: NUL-terminated fixture-owned pathname; no pointer escapes this call.
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        fixture.preserved_failure(&[]);
    }
}
