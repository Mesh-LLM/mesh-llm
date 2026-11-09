use super::{Input, atomic_json, identity, plan::Step};
use crate::{
    command::DynResult,
    command_interrupt::Interrupt,
    process::{self, Completion, Limits, OutputFiles, Readiness, Value},
    repository::check_report::CheckReport,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const LOG_BYTES: usize = 16 * 1024 * 1024;
const REMOVED_ENVIRONMENT: &[&str] = &[
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "CANARY_REPAIR_TOKEN",
    "GITHUB_ENV",
    "GITHUB_OUTPUT",
    "GITHUB_STEP_SUMMARY",
    "GITHUB_PATH",
    "GITHUB_STATE",
    "CARGO_BUILD_TARGET",
];

struct Environment {
    process: BTreeMap<OsString, Value>,
    reproduction: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct ResultRow<'a> {
    step: &'a str,
    status: &'a str,
    failure_class: Option<&'a str>,
    exit_code: i32,
    elapsed_seconds: f64,
    log: &'a Path,
    stderr_log: &'a Path,
    working_directory: &'a Path,
    command: &'a [String],
    environment: &'a BTreeMap<String, String>,
    removed_environment: &'static [&'static str],
}

pub(super) fn execute(input: &Input, steps: &[Step]) -> DynResult<CheckReport> {
    let interrupt = Interrupt::install()?;
    let directory = tempfile::Builder::new()
        .prefix("check-")
        .tempdir_in(input.state.join("checks"))?
        .keep();
    let deadline = Instant::now() + Duration::from_secs(input.seconds);
    let mut identity = None;
    let mut result = Ok(CheckReport::success(String::new()));
    for step in steps {
        let unchanged = identity
            .as_ref()
            .map_or(Ok(()), |value| identity::Identity::verify(value, input));
        let outcome = if let Err(error) = unchanged {
            Err(error)
        } else if step.identity {
            static_identity(input, &directory, deadline, &interrupt.cancellation()).map(|value| {
                identity = Some(value);
                0
            })
        } else {
            command_with_retry(input, step, &directory, deadline, &interrupt.cancellation())
        };
        let outcome = outcome.and_then(|code| {
            if code == 0
                && let Some(identity) = &identity
            {
                identity.verify(input)?;
            }
            Ok(code)
        });
        match outcome {
            Ok(0) => (),
            Ok(code) => {
                result = Ok(CheckReport {
                    stdout: String::new(),
                    stderr: format!(
                        "repair check stopped at {}; evidence: {}\n",
                        step.label,
                        directory.display()
                    ),
                    code,
                });
                break;
            }
            Err(error) => {
                let candidate = error.downcast_ref::<identity::SourceMismatch>().is_some();
                let (class, code) = if candidate {
                    ("candidate", 1)
                } else {
                    ("infrastructure", 125)
                };
                let reproduction = environment(input, step)
                    .map(|value| value.reproduction)
                    .unwrap_or_default();
                atomic_json(
                    &input.state.join("last-check.json"),
                    &serde_json::json!({"step":step.label,"status":"fail","failure_class":class,"error":error.to_string(),"exit_code":code,
                        "working_directory":input.root,"command":step.arguments,"environment":reproduction,
                        "rerun_repair_check":rerun(input),
                        "removed_environment":REMOVED_ENVIRONMENT,
                        "log":directory.join(format!("{}.log",step.label)),
                        "stderr_log":directory.join(format!("{}.stderr.log",step.label))}),
                )?;
                result = Ok(CheckReport {
                    stdout: String::new(),
                    stderr: format!("repair {class} failure at {}: {error}\n", step.label),
                    code,
                });
                break;
            }
        }
    }
    let interrupted = interrupt.finish();
    if interrupted.is_err() {
        return Ok(CheckReport {
            stdout: String::new(),
            stderr: "repair checks interrupted; owned processes were stopped\n".into(),
            code: 125,
        });
    }
    result
}

fn rerun(input: &Input) -> Vec<String> {
    let phase = match input.phase {
        super::Phase::Environment => "environment",
        super::Phase::Fast => "fast",
        super::Phase::Full => "full",
    };
    [
        "cargo".into(),
        "xtool".into(),
        "automation".into(),
        "canary-receipts".into(),
        "repair-check".into(),
        "--root".into(),
        input.root.to_string_lossy().into_owned(),
        "--controller-root".into(),
        input.controller.to_string_lossy().into_owned(),
        "--state-dir".into(),
        input.state.to_string_lossy().into_owned(),
        "--build-dir".into(),
        input.build.to_string_lossy().into_owned(),
        "--backend".into(),
        input.backend.clone(),
        "--phase".into(),
        phase.into(),
        "--seconds".into(),
        input.seconds.to_string(),
    ]
    .into()
}

fn command_with_retry(
    input: &Input,
    step: &Step,
    directory: &Path,
    deadline: Instant,
    cancellation: &process::Cancellation,
) -> DynResult<i32> {
    let code = command(input, step, directory, deadline, cancellation)?;
    if code == 0 || code == 125 || step.label != "generated-family-and-native-suite" {
        return Ok(code);
    }
    let Some(retry) = super::retry::steps(input, &directory.join(format!("{}.log", step.label)))
    else {
        return Ok(code);
    };
    atomic_json(&directory.join("diagnostic-retry.json"), &retry)?;
    for step in retry {
        let code = command(input, &step, directory, deadline, cancellation)?;
        if code != 0 {
            if step.label == "metal-fixture-diagnostic-retry" {
                return Err("isolated Metal fixture failed its diagnostic retry; inspect the retained step logs before rerunning".into());
            }
            return Ok(code);
        }
    }
    Ok(0)
}

fn environment(input: &Input, step: &Step) -> DynResult<Environment> {
    let mut environment = inherited(std::env::vars_os());
    let mut reproduction = BTreeMap::new();
    if let Some(Value::Public(cache)) = environment.get(std::ffi::OsStr::new("HF_CACHE")) {
        reproduction.insert("HF_CACHE".into(), cache.to_string_lossy().into_owned());
    }
    let python = input.state.join("jinja-env");
    let mut path = vec![python.join("bin")];
    path.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    let values = [
        ("UV_PROJECT_ENVIRONMENT", python.into_os_string()),
        ("UV_PYTHON_DOWNLOADS", "never".into()),
        ("PATH", std::env::join_paths(path)?),
        ("LLAMA_STAGE_LINK_MODE", "static".into()),
        ("LLAMA_STAGE_BACKEND", input.backend.clone().into()),
        ("LLAMA_STAGE_BUILD_DIR", input.build.clone().into()),
        ("LLAMA_BUILD_DIR", input.build.clone().into()),
        ("LLAMA_STAGE_BUILD_TESTS", "OFF".into()),
        ("LLAMA_STAGE_UPSTREAM_TESTS", "OFF".into()),
        ("LLAMA_STAGE_FULL_REPLAY", "OFF".into()),
        ("LLAMA_STAGE_WORKLOAD_ORACLE", "OFF".into()),
        ("LLAMA_WORKDIR", input.root.join(".deps/llama.cpp").into()),
        ("CARGO_TARGET_DIR", input.root.join("target").into()),
        (
            "STAGE_SERVER_BIN",
            input.root.join("target/debug/skippy").into(),
        ),
        ("SKIPPY_SMOKE_FOCUSED_ONLY", "0".into()),
        ("SKIPPY_SMOKE_SKIP_BUILD", "1".into()),
        ("SKIPPY_SMOKE_SOURCE_ROOT", input.root.clone().into()),
        ("SMOKE_N_GPU_LAYERS", "0".into()),
        ("SMOKE_FLASH_ATTN", "disabled".into()),
        ("MESH_LLM_AUTOMATION_BIN", std::env::current_exe()?.into()),
    ];
    for (key, value) in values {
        reproduction.insert(key.into(), value.to_string_lossy().into_owned());
        environment.insert(key.into(), Value::Public(value));
    }
    for (key, value) in &step.environment {
        reproduction.insert(key.clone(), value.clone());
        environment.insert(key.into(), Value::Public(value.into()));
    }
    Ok(Environment {
        process: environment,
        reproduction,
    })
}

pub(super) fn inherited(
    values: impl IntoIterator<Item = (OsString, OsString)>,
) -> BTreeMap<OsString, Value> {
    values
        .into_iter()
        .filter(|(key, _)| !REMOVED_ENVIRONMENT.iter().any(|name| key == name))
        .map(|(key, value)| {
            let name = key.to_string_lossy();
            // Empty runner placeholders contain nothing to redact; the process
            // supervisor rejects them when marked as secret patterns.
            let secret = !value.is_empty()
                && ["_KEY", "_TOKEN", "_SECRET", "_PASSWORD"]
                    .iter()
                    .any(|suffix| name.ends_with(suffix));
            (
                key,
                if secret {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                },
            )
        })
        .collect()
}

fn executable(name: &str) -> DynResult<PathBuf> {
    if Path::new(name).is_absolute() {
        return Ok(name.into());
    }
    std::env::split_paths(&std::env::var_os("PATH").ok_or("PATH missing")?)
        .map(|p| p.join(name))
        .find(|p| p.is_file())
        .ok_or_else(|| format!("repair prerequisite missing: {name}").into())
}

fn command(
    input: &Input,
    step: &Step,
    directory: &Path,
    deadline: Instant,
    cancellation: &process::Cancellation,
) -> DynResult<i32> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .ok_or("repair check deadline exceeded")?;
    let log = directory.join(format!("{}.log", step.label));
    let stderr = directory.join(format!("{}.stderr.log", step.label));
    writeln!(
        crate::cli_output::stdout(),
        "repair-check: starting {}; log: {}",
        step.label,
        log.display()
    )?;
    crate::cli_output::stdout().flush()?;
    let started = Instant::now();
    let environment = environment(input, step)?;
    let report = supervise_progress(
        &process::ProcessSpec {
            executable: executable(step.arguments.first().ok_or("empty repair command")?)?,
            arguments: step.arguments[1..]
                .iter()
                .map(|s| Value::Public(s.into()))
                .collect(),
            cwd: input.root.clone(),
            environment: environment.process,
        },
        &Limits {
            execution: remaining,
            graceful_shutdown: Duration::from_secs(10),
            forced_shutdown: Duration::from_secs(5),
            retained_bytes_per_stream: LOG_BYTES,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        OutputFiles {
            stdout: Some(log.clone()),
            stderr: Some(stderr.clone()),
        },
        &step.label,
    )?;
    let exit_code = report
        .status
        .as_ref()
        .and_then(std::process::ExitStatus::code);
    let infrastructure = matches!(exit_code, None | Some(125..=127))
        || !matches!(report.outcome, process::Outcome::Exited)
        || !report.cleanup.complete
        || report.cleanup.forced
        || report.cleanup.graceful_signal_failed
        || report.cleanup.failure.is_some()
        || report.failure.is_some()
        || report.stdout.truncated
        || report.stderr.truncated;
    let code = if infrastructure {
        125
    } else {
        report
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code)
            .unwrap_or(125)
    };
    let diagnostics = format!("{}\n{}", tail(&log)?, tail(&stderr)?);
    let dependency = contains(&log, "ModuleNotFoundError:")?
        || contains(&stderr, "ModuleNotFoundError:")?
        || step.label == "native-test-environment";
    let class = if infrastructure {
        "infrastructure"
    } else if dependency {
        "dependency"
    } else {
        "candidate"
    };
    let code = if code != 0 && dependency { 125 } else { code };
    let row = ResultRow {
        step: &step.label,
        status: if code == 0 { "pass" } else { "fail" },
        failure_class: (code != 0).then_some(class),
        exit_code: code,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        log: &log,
        stderr_log: &stderr,
        working_directory: &input.root,
        command: &step.arguments,
        environment: &environment.reproduction,
        removed_environment: REMOVED_ENVIRONMENT,
    };
    atomic_json(&directory.join(format!("{}.json", step.label)), &row)?;
    atomic_json(&input.state.join("last-check.json"), &row)?;
    writeln!(
        crate::cli_output::stdout(),
        "repair-check: {} {} ({:.1}s)",
        row.step,
        row.status,
        row.elapsed_seconds
    )?;
    if code != 0 {
        writeln!(
            crate::cli_output::stderr(),
            "failure_class={class}; step={}\nreproduce arguments: {}\n{diagnostics}",
            step.label,
            serde_json::to_string(&step.arguments)?
        )?;
    }
    Ok(code)
}

fn supervise_progress(
    spec: &process::ProcessSpec,
    limits: &Limits,
    cancellation: &process::Cancellation,
    files: OutputFiles,
    label: &str,
) -> Result<process::ProcessReport, process::Failure> {
    std::thread::scope(|scope| {
        let (complete, receiver) = std::sync::mpsc::channel();
        let started = Instant::now();
        let progress = scope.spawn(move || -> std::io::Result<()> {
            while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                receiver.recv_timeout(Duration::from_secs(30))
            {
                let result = writeln!(
                    crate::cli_output::stdout(),
                    "repair-check: {label} running ({:.0}s); full output retained in step log",
                    started.elapsed().as_secs_f64()
                );
                if result.is_err() {
                    cancellation.cancel();
                }
                result?;
            }
            Ok(())
        });
        let report = process::supervise(spec, limits, cancellation, files);
        let _ = complete.send(());
        match progress.join() {
            Ok(Ok(())) => report,
            _ => Err(process::Failure::InvalidSpec(
                "repair progress output failed",
            )),
        }
    })
}

fn static_identity(
    input: &Input,
    directory: &Path,
    deadline: Instant,
    cancellation: &process::Cancellation,
) -> DynResult<identity::Identity> {
    let identity = identity::capture(input)?;
    if cfg!(target_os = "macos") {
        let step = Step {
            label: "native-archive-architecture".into(),
            arguments: vec![
                "lipo".into(),
                "-archs".into(),
                input
                    .build
                    .join("src/libllama.a")
                    .to_string_lossy()
                    .into_owned(),
            ],
            environment: BTreeMap::new(),
            identity: false,
        };
        if command(input, &step, directory, deadline, cancellation)? != 0
            || fs::read_to_string(directory.join("native-archive-architecture.log"))?.trim()
                != "arm64"
        {
            return Err("candidate native archive must be arm64".into());
        }
    }
    for binary in identity.binaries() {
        if binary
            .file_name()
            .is_some_and(|name| name == "skippy-package-builder")
        {
            continue;
        }
        super::symbols::require(binary)?;
    }
    identity.verify(input)?;
    atomic_json(&directory.join("runtime-identity.json"), &identity)?;
    writeln!(
        crate::cli_output::stdout(),
        "repair-check: static runtime identities verified"
    )?;
    Ok(identity)
}

fn tail(path: &Path) -> DynResult<String> {
    let mut file = fs::File::open(path)?;
    let length = file.metadata()?.len();
    file.seek(SeekFrom::Start(length.saturating_sub(16384)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes)
        .lines()
        .rev()
        .take(40)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n"))
}

pub(super) fn contains(path: &Path, needle: &str) -> DynResult<bool> {
    use std::io::BufRead;
    for line in std::io::BufReader::new(fs::File::open(path)?).lines() {
        if line?.contains(needle) {
            return Ok(true);
        }
    }
    Ok(false)
}
