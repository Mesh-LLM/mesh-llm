use super::{
    admission::{Admission, Candidate},
    options::Options,
};
use crate::{
    automation::command_interrupt::Interrupt,
    command::DynResult,
    process::{self, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
pub(super) const TEST: &str =
    "frontend::tests::tts_oracle::deterministic_tts_candidate_when_fixture_is_set";
fn environment() -> BTreeMap<OsString, Value> {
    std::env::vars_os()
        .map(|(key, value)| (key, Value::Public(value)))
        .collect()
}
fn logged(spec: &ProcessSpec, work: &Path, label: &str, interrupt: &Interrupt) -> DynResult<()> {
    let limits = Limits {
        execution: Duration::from_secs(900),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let output = process::supervise_raw(
        spec,
        &limits,
        &interrupt.cancellation(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(8 * 1024 * 1024),
            stderr: std::num::NonZeroUsize::new(8 * 1024 * 1024),
        },
    )?;
    // Raw complete per-stream logs are distinct from bounded redacted diagnostics.
    if let Some(bytes) = &output.stdout {
        fs::write(work.join(format!("{label}.log")), bytes.as_bytes())?;
    }
    if let Some(bytes) = &output.stderr {
        fs::write(work.join(format!("{label}.stderr.log")), bytes.as_bytes())?;
    }
    fs::write(
        work.join(format!("{label}.diagnostic.log")),
        format!("{:?}\n", output.process),
    )?;
    if !output.process.success() {
        return Err(format!(
            "TTS {label} failed; see {}: outcome={:?}, failure={:?}, status={:?}, cleanup_complete={}",
            work.display(),
            output.process.outcome,
            output.process.failure,
            output.process.status.as_ref().and_then(std::process::ExitStatus::code),
            output.process.cleanup.complete
        )
        .into());
    }
    Ok(())
}
pub(super) fn run(
    options: &Options,
    admission: &Admission,
    candidate_wav: &Path,
    oracle_wav: &Path,
) -> DynResult<()> {
    let interrupt = Interrupt::install()?;
    let result = execute(options, admission, candidate_wav, oracle_wav, &interrupt);
    interrupt.finish()?;
    result
}
fn execute(
    options: &Options,
    admission: &Admission,
    candidate_wav: &Path,
    oracle_wav: &Path,
    interrupt: &Interrupt,
) -> DynResult<()> {
    let (executable, arguments) = match &admission.candidate {
        Candidate::Prebuilt(binary) => (
            binary.clone(),
            vec![
                TEST.to_owned(),
                "--exact".into(),
                "--nocapture".into(),
                "--test-threads=1".into(),
            ],
        ),
        Candidate::Standalone(just) => (
            just.clone(),
            vec![
                "--justfile".into(),
                options.root.join("Justfile").to_string_lossy().into_owned(),
                "with-lld".into(),
                "cargo".into(),
                "test".into(),
                "--manifest-path".into(),
                options
                    .root
                    .join("Cargo.toml")
                    .to_string_lossy()
                    .into_owned(),
                "-p".into(),
                "skippy-serving".into(),
                "--lib".into(),
                TEST.into(),
                "--".into(),
                "--exact".into(),
                "--nocapture".into(),
                "--test-threads=1".into(),
            ],
        ),
    };
    let mut env = environment();
    for (key, value) in [
        ("LLAMA_STAGE_BACKEND", OsString::from("cpu")),
        (
            "SKIPPY_WORKLOAD_MODEL",
            options.model_path.clone().into_os_string(),
        ),
        (
            "SKIPPY_WORKLOAD_PROJECTOR",
            options.projector.clone().into_os_string(),
        ),
        ("SKIPPY_WORKLOAD_MODEL_ID", options.model.clone().into()),
        (
            "SKIPPY_WORKLOAD_LAYER_END",
            options.layer_end.to_string().into(),
        ),
        (
            "SKIPPY_TTS_ORACLE_CANDIDATE_WAV",
            candidate_wav.as_os_str().into(),
        ),
        ("SKIPPY_TTS_ORACLE_PROMPT", super::PROMPT.into()),
        ("SKIPPY_TTS_ORACLE_SEED", "7".into()),
        ("SKIPPY_TTS_ORACLE_TOP_K", "20".into()),
        ("SKIPPY_TTS_ORACLE_TOP_P", "0.8".into()),
        ("SKIPPY_TTS_ORACLE_MAX_FRAMES", "512".into()),
    ] {
        env.insert(key.into(), Value::Public(value));
    }
    logged(
        &ProcessSpec {
            executable,
            arguments: arguments
                .into_iter()
                .map(|value| Value::Public(value.into()))
                .collect(),
            cwd: options.root.clone(),
            environment: env,
        },
        &options.work,
        "tts-candidate-test",
        interrupt,
    )?;
    if !candidate_wav.is_file() {
        return Err("deterministic TTS candidate test did not write WAV output".into());
    }
    let arguments: Vec<OsString> = vec![
        "-m".into(),
        options.model_path.as_os_str().into(),
        "-mm".into(),
        options.projector.as_os_str().into(),
        "-p".into(),
        super::PROMPT.into(),
        "--output".into(),
        oracle_wav.as_os_str().into(),
        "-n".into(),
        "512".into(),
        "--seed".into(),
        "7".into(),
        "--top-k".into(),
        "20".into(),
        "--top-p".into(),
        "0.8".into(),
        "--temp".into(),
        "1".into(),
        "--min-p".into(),
        "0".into(),
        "--repeat-penalty".into(),
        "1".into(),
        "--no-repack".into(),
        "-c".into(),
        "2048".into(),
        "-b".into(),
        "2048".into(),
        "-ub".into(),
        "2048".into(),
        "-ngl".into(),
        "0".into(),
    ];
    logged(
        &ProcessSpec {
            executable: options.oracle.clone(),
            arguments: arguments.into_iter().map(Value::Public).collect(),
            cwd: options.root.clone(),
            environment: environment(),
        },
        &options.work,
        "tts-monolithic-oracle",
        interrupt,
    )
}
pub(super) fn fresh(work: &Path) -> DynResult<(PathBuf, PathBuf)> {
    fs::create_dir_all(work)?;
    for name in [
        "tts-candidate.wav",
        "tts-monolithic-oracle.wav",
        "tts-oracle-result.json",
        "tts-candidate-test.log",
        "tts-candidate-test.stderr.log",
        "tts-candidate-test.diagnostic.log",
        "tts-monolithic-oracle.log",
        "tts-monolithic-oracle.stderr.log",
        "tts-monolithic-oracle.diagnostic.log",
    ] {
        match fs::remove_file(work.join(name)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok((
        work.join("tts-candidate.wav"),
        work.join("tts-monolithic-oracle.wav"),
    ))
}
