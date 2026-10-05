//! Actual frontend/reader process contract, selected after automation bootstrap.
mod parquet_fixture;
#[path = "../../xtask/src/process/mod.rs"]
#[allow(dead_code)]
pub mod process;

use process::{Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};
use trajectory_reader::selection;

fn frontend() -> PathBuf {
    let path = std::env::var_os("MESH_LLM_TEST_XTASK_BIN")
        .expect("run this selected contract after just automation-bootstrap, with MESH_LLM_TEST_XTASK_BIN set to binary_path");
    let path = PathBuf::from(path);
    assert!(path.is_absolute());
    path.canonicalize().unwrap()
}

fn generate(directory: &Path, families: usize) -> process::ProcessReport {
    let reader = Path::new(env!("CARGO_BIN_EXE_trajectory-reader"))
        .canonicalize()
        .unwrap();
    let mut environment: BTreeMap<_, _> = [(
        "MESH_LLM_TRAJECTORY_READER_BIN".into(),
        Value::Public(reader.into_os_string()),
    )]
    .into_iter()
    .collect();
    for name in ["SYSTEMROOT", "WINDIR"] {
        if let Some(value) = std::env::var_os(name) {
            environment.insert(name.into(), Value::Public(value));
        }
    }
    let arguments = [
        "automation",
        "agentic-prompt-manifest",
        "--dataset-file",
        "sessions.parquet",
        "--dataset-revision",
        "abc",
        "--output",
        "manifest.json",
        "--source-dataset",
        "source",
        "--families",
        &families.to_string(),
        "--requests-per-family",
        "1",
    ]
    .into_iter()
    .map(|arg| Value::Public(arg.into()))
    .collect();
    process::supervise(
        &ProcessSpec {
            executable: frontend(),
            cwd: directory.to_path_buf(),
            arguments,
            environment,
        },
        &Limits {
            execution: Duration::from_secs(30),
            graceful_shutdown: Duration::from_secs(8),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap()
}

#[test]
fn native_prompt_command_preserves_consumed_bytes_and_existing_output_on_reader_refusal() {
    let temporary = tempfile::tempdir().unwrap();
    let directory = temporary.path().canonicalize().unwrap();
    let row = selection::Trajectory {
        session_id: "a".into(),
        source_dataset: "source".into(),
        messages_json: "[{\"role\":\"user\",\"content\":\"é\"}]".into(),
        n_turns: 20,
        max_isl: 9000,
        total_tokens: 9100,
    };
    parquet_fixture::write_parquet(
        &directory.join("sessions.parquet"),
        &[vec![row]],
        parquet::basic::Compression::ZSTD(Default::default()),
    );
    let result = generate(&directory, 1);
    assert!(result.success(), "{result:?}");
    let expected =
        include_bytes!("../../xtask/src/automation/agentic_prompt_manifest/fixture_expected.json");
    assert_eq!(
        std::fs::read(directory.join("manifest.json")).unwrap(),
        expected
    );
    std::fs::write(directory.join("manifest.json"), b"keep-me").unwrap();
    let rejected = generate(&directory, 2);
    assert!(!rejected.success());
    assert!(rejected.cleanup.complete, "{rejected:?}");
    assert!(
        String::from_utf8_lossy(&rejected.stderr.bytes_retained)
            .contains("selected 1 trajectories, expected 2")
    );
    assert_eq!(
        std::fs::read(directory.join("manifest.json")).unwrap(),
        b"keep-me"
    );
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 2);
}

#[test]
fn native_prompt_command_preserves_trajectory_data_larger_than_log_capture_limit() {
    let temporary = tempfile::tempdir().unwrap();
    let directory = temporary.path().canonicalize().unwrap();
    let content = "x".repeat(17 * 1024 * 1024);
    let row = selection::Trajectory {
        session_id: "large".into(),
        source_dataset: "source".into(),
        messages_json: serde_json::to_string(&serde_json::json!([{
            "role": "user", "content": content,
        }]))
        .unwrap(),
        n_turns: 20,
        max_isl: 9000,
        total_tokens: 9100,
    };
    parquet_fixture::write_parquet(
        &directory.join("sessions.parquet"),
        &[vec![row]],
        parquet::basic::Compression::ZSTD(Default::default()),
    );
    let result = generate(&directory, 1);
    assert!(result.success(), "{result:?}");
    assert!(result.cleanup.complete, "{result:?}");
    let manifest: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(directory.join("manifest.json")).unwrap())
            .unwrap();
    let prompt = manifest["prompts"][0]["prompt"].as_str().unwrap();
    assert!(prompt.contains(&content));
    assert!(
        prompt.ends_with(
            "Benchmark branch 0: summarize the latest repository state in one sentence."
        )
    );
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 2);
}
