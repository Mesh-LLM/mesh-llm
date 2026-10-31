use super::{
    contract::{Backend, Cell, Prepared},
    evidence, publish,
};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    },
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::Path,
    time::{Duration, Instant},
};
#[path = "server.rs"]
mod server;

pub(super) fn environment(extra: &[(&str, OsString)]) -> BTreeMap<OsString, Arg> {
    let mut env: BTreeMap<_, _> = [
        "PATH",
        "HOME",
        "SYSTEMROOT",
        "WINDIR",
        "CUDA_VISIBLE_DEVICES",
    ]
    .into_iter()
    .filter_map(|key| std::env::var_os(key).map(|v| (key.into(), Arg::Public(v))))
    .collect();
    env.insert("HF_HUB_OFFLINE".into(), Arg::Public("1".into()));
    env.extend(
        extra
            .iter()
            .map(|(k, v)| ((*k).into(), Arg::Public(v.clone()))),
    );
    env
}
pub(super) fn limits(execution: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: Duration::from_secs(10),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 16 * 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}
pub(super) fn child(
    binary: &Path,
    args: Vec<OsString>,
    paths: (&Path, &Path),
    label: &str,
    duration: Duration,
    cancel: &Cancellation,
    extra: &[(&str, OsString)],
) -> DynResult<String> {
    let (cwd, output) = paths;
    let stdout = output.join(format!("{label}.stdout.log"));
    let stderr = output.join(format!("{label}.stderr.log"));
    let result = process::supervise(
        &ProcessSpec {
            executable: binary.into(),
            arguments: args.into_iter().map(Arg::Public).collect(),
            cwd: cwd.into(),
            environment: environment(extra),
        },
        &limits(duration),
        cancel,
        OutputFiles {
            stdout: Some(stdout.clone()),
            stderr: Some(stderr),
        },
    );
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            publish(
                &output.join(format!("{label}.process.json")),
                &json!({"success":false,"supervision_failure":error.to_string()}),
            )?;
            return Err(error.into());
        }
    };
    publish(
        &output.join(format!("{label}.process.json")),
        &json!({"success":result.success(),"outcome":format!("{:?}",result.outcome),"cleanup_complete":result.cleanup.complete,"forced":result.cleanup.forced,"exit_code":result.status.and_then(|s|s.code())}),
    )?;
    if !result.success()
        || !result.cleanup.complete
        || result.cleanup.forced
        || result.cleanup.failure.is_some()
        || result.cleanup.graceful_signal_failed
    {
        return Err(format!("KV child {label} failed; raw output retained").into());
    }
    Ok(String::from_utf8(
        crate::automation::receipt_files::bounded(&stdout, 16 * 1024 * 1024)?,
    )?)
}
pub(super) fn run(input: Cell, output: &Path) -> DynResult<()> {
    input.validate()?;
    std::fs::create_dir(output)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let deadline = Instant::now() + Duration::from_secs(input.execution_seconds);
    let result = execute(&input, output, deadline, &interrupt.cancellation());
    let finish = interrupt.finish();
    let passed = result.is_ok() && finish.is_ok() && Instant::now() < deadline;
    publish(
        &output.join("result.json"),
        &json!({"schema_version":1,"status":if passed {"passed"} else {"failed"},"model_key":input.model_key,"request":input,"failure":result.as_ref().err().map(ToString::to_string),"interrupt_failure":finish.as_ref().err().map(ToString::to_string),"scope":"KVdtype_dispatch_generic_c1_cache_plus_recurrent_L3_process_restart_not_long_c4_HTTP_restart_or_performance"}),
    )?;
    archive(output)?;
    finish?;
    result?;
    if !passed {
        return Err("KV validation deadline exceeded".into());
    }
    Ok(())
}
pub(super) fn archive(output: &Path) -> DynResult<()> {
    let path = output.with_extension("tar.gz");
    let archive_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("KV archive filename must be Unicode")?;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    let gzip = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut tar = tar::Builder::new(gzip);
    let mut pending = vec![output.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() {
                tar.append_path_with_name(
                    entry.path(),
                    Path::new("evidence").join(entry.path().strip_prefix(output)?),
                )?;
            } else if !kind.is_symlink() {
                return Err("KV evidence contains a special file".into());
            }
        }
    }
    tar.into_inner()?.finish()?;
    let sha256 = crate::product::digest::file_sha256(&path).map_err(|e| e.error)?;
    publish(
        &output.with_extension("archive.json"),
        &json!({"archive":archive_name,"sha256":sha256,"symlinks":"excluded; selected runtime identity retained in JSON"}),
    )
}
fn remaining(deadline: Instant) -> DynResult<Duration> {
    deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(12))
        .filter(|d| !d.is_zero())
        .ok_or_else(|| "KV validation cleanup reserve exhausted".into())
}
fn execute(input: &Cell, output: &Path, deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    let model = input.validate()?;
    input.prepared.verify()?;
    let prepared: Prepared = serde_json::from_slice(&crate::automation::receipt_files::bounded(
        &input.prepared.path,
        1024 * 1024,
    )?)?;
    prepared.verify()?;
    crate::automation::replay_matrix::model_preflight::verify(
        &input.model,
        &model.files[0].sha256,
        2048,
    )?;
    let dimensions =
        crate::automation::replay_matrix::model_preflight::dimensions::inspect(&input.model)?
            .ok_or("model dimensions missing")?;
    publish(
        &output.join("identities.json"),
        &json!({"prepared":prepared,"model":model,"model_layers":dimensions.block_count}),
    )?;
    let producer = &prepared.producer;
    let doctor = child(
        &producer.cli.path,
        vec![
            "--output".into(),
            "json".into(),
            "--runtime-bundle".into(),
            producer.runtime_bundle.clone().into_os_string(),
            "--runtime-selection".into(),
            producer.backend.kind().into(),
            "doctor".into(),
        ],
        (output, output),
        "doctor",
        remaining(deadline)?,
        cancel,
        &[],
    )?;
    let doctor: serde_json::Value = serde_json::from_str(&doctor)?;
    evidence::doctor(&doctor, &producer.runtime_bundle)?;
    let text = child(
        &prepared.matrix.path,
        vec![
            input.model.clone().into_os_string(),
            producer.backend.device().into(),
        ],
        (output, output),
        "matrix",
        remaining(deadline)?,
        cancel,
        &[],
    )?;
    evidence::matrix(&text, producer.backend)?;
    server::run(
        input,
        &prepared,
        &model,
        output,
        remaining(deadline)?,
        cancel,
    )?;
    if model.recurrent_case() {
        recurrent(
            input,
            &prepared,
            dimensions.block_count,
            output,
            deadline,
            cancel,
        )?;
    }
    prepared.verify()?;
    Ok(())
}
fn recurrent(
    input: &Cell,
    prepared: &Prepared,
    layers: u64,
    output: &Path,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    // Test discovery's documented parent layout contains exactly this verified leaf.
    let parent = output.join("test-runtime");
    std::fs::create_dir(&parent)?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        &prepared.producer.runtime_bundle,
        parent.join("selected-runtime"),
    )?;
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(
        &prepared.producer.runtime_bundle,
        parent.join("selected-runtime"),
    )?;
    let model_id = format!("{}:Q4_K_M", input.validate()?.repo);
    let env = [
        (
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
            parent.into_os_string(),
        ),
        (
            "SKIPPY_RECURRENT_CACHE_TEST_MODEL",
            input.model.clone().into_os_string(),
        ),
        (
            "SKIPPY_RECURRENT_CACHE_TEST_MODEL_LAYERS",
            layers.to_string().into(),
        ),
        ("SKIPPY_RECURRENT_CACHE_TEST_MODEL_ID", model_id.into()),
        (
            "SKIPPY_RECURRENT_CACHE_TEST_GPU_LAYERS",
            if prepared.producer.backend == Backend::Cpu {
                "0"
            } else {
                "99"
            }
            .into(),
        ),
    ];
    for suffix in [
        "recurrent_chat_checkpoint_preserves_cached_output_parity",
        "recurrent_post_decode_checkpoint_reuses_a_growing_prompt",
        "durable_prefix::native_cold_prefix_survives_runtime_restart",
        "durable_prefix::native_cold_offload_refusal_keeps_inference_usable",
    ] {
        let test = format!("frontend::local_generation::tests::{suffix}");
        let text = child(
            &prepared.producer.serving_tests.path,
            vec![
                "--exact".into(),
                test.clone().into(),
                "--ignored".into(),
                "--nocapture".into(),
                "--test-threads=1".into(),
            ],
            (output, output),
            suffix,
            remaining(deadline)?,
            cancel,
            &env,
        )?;
        evidence::rust_test(&text, &test)?;
    }
    process_restart(prepared, &env, output, deadline, cancel)?;
    Ok(())
}

fn process_restart(
    prepared: &Prepared,
    env: &[(&str, OsString)],
    output: &Path,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    let root = output.join("l3-process-restart");
    std::fs::create_dir(&root)?;
    let test =
        "frontend::local_generation::tests::durable_prefix::native_cold_prefix_process_stage";
    for phase in ["seed", "restore"] {
        let mut phase_env = env.to_vec();
        phase_env.extend([
            ("SKIPPY_L3_TEST_ROOT", root.clone().into_os_string()),
            ("SKIPPY_L3_TEST_PHASE", phase.into()),
        ]);
        let text = child(
            &prepared.producer.serving_tests.path,
            vec![
                "--exact".into(),
                test.into(),
                "--ignored".into(),
                "--nocapture".into(),
                "--test-threads=1".into(),
            ],
            (output, output),
            &format!("l3-process-{phase}"),
            remaining(deadline)?,
            cancel,
            &phase_env,
        )?;
        evidence::rust_test(&text, test)?;
    }
    evidence::process_restart(&crate::automation::receipt_files::bounded(
        &root.join("restart-receipt.json"),
        4096,
    )?)
}
