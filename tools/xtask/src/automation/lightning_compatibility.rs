//! Three unfunded mixed-version compatibility cases over isolated local bundles.
mod http_checks;
mod identity;
mod owner;
mod setup;
mod terminal;
use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, ProcessSpec, Readiness, Value},
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
pub(crate) const USAGE: &str = "cargo xtool automation lightning-compatibility --current-binary PATH --released-binary PATH --released-dialect serve-client-bind-port --model PATH --output NEW_DIRECTORY [--timeout-secs 30..3600]";
struct Options {
    input: identity::Input,
    output: PathBuf,
    seconds: u64,
}
impl Options {
    fn parse(args: &[String]) -> DynResult<Self> {
        let mut flags = BTreeMap::new();
        if !args.len().is_multiple_of(2) {
            return Err(USAGE.into());
        }
        for pair in args.as_chunks::<2>().0 {
            if ![
                "--current-binary",
                "--released-binary",
                "--model",
                "--output",
                "--released-dialect",
                "--timeout-secs",
            ]
            .contains(&pair[0].as_str())
                || flags.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err("unknown/duplicate compatibility flag".into());
            }
        }
        let required = |flag| -> DynResult<&str> {
            flags
                .get(flag)
                .copied()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| USAGE.into())
        };
        if required("--released-dialect")? != "serve-client-bind-port" {
            return Err("unsupported declared released CLI dialect".into());
        }
        let seconds = flags.get("--timeout-secs").unwrap_or(&"600").parse()?;
        if !(30..=3600).contains(&seconds) {
            return Err("compatibility budget out of bounds".into());
        }
        let input = identity::Input {
            current: std::path::absolute(required("--current-binary")?)?,
            released: std::path::absolute(required("--released-binary")?)?,
            model: std::path::absolute(required("--model")?)?,
            evidence_logs: None,
        };
        let output = std::path::absolute(required("--output")?)?;
        match std::fs::symlink_metadata(&output) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            _ => return Err("compatibility output directory must be new".into()),
        }
        Ok(Self {
            input,
            output,
            seconds,
        })
    }
}
fn remaining(deadline: Instant, reserve: u64) -> DynResult<Duration> {
    deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(reserve))
        .filter(|d| !d.is_zero())
        .ok_or_else(|| "compatibility overall budget exhausted".into())
}
fn probe(
    spec: &ProcessSpec,
    directory: &Path,
    name: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<process::RawProcessReport> {
    let result = process::supervise_raw(
        spec,
        &Limits {
            execution: remaining(deadline, 3)?,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )?;
    let diagnostic = json!({"outcome":format!("{:?}",result.process.outcome),"exit_code":result.process.status.and_then(|s|s.code()),"clean":result.process.success(),"stdout_suppressed_lines":result.process.stdout.suppressed_lines,"stderr_suppressed_lines":result.process.stderr.suppressed_lines});
    identity::fresh(
        &directory.join(format!("{name}-process.json")),
        &serde_json::to_vec_pretty(&diagnostic)?,
    )?;
    if !result.process.success() {
        return Err("compatibility owned preflight probe failed".into());
    }
    Ok(result)
}
fn identity(
    input: &identity::Input,
    output: &Path,
    name: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<serde_json::Value> {
    let path = output.join(format!("{name}-input.json"));
    let receipt = output.join(format!("{name}.json"));
    let bytes = serde_json::to_vec(input)?;
    identity::fresh(&path, &bytes)?;
    probe(
        &ProcessSpec {
            executable: std::env::current_exe()?,
            cwd: output.into(),
            environment: BTreeMap::new(),
            arguments: vec![
                Value::Public("automation".into()),
                Value::Public("lightning-compatibility".into()),
                Value::Public("identity-worker".into()),
                Value::Public("--input".into()),
                Value::Public(path.into()),
                Value::Public("--output".into()),
                Value::Public(receipt.clone().into()),
            ],
        },
        output,
        name,
        deadline,
        cancel,
    )?;
    let value: serde_json::Value = serde_json::from_slice(&identity::bytes(&receipt, 65536)?)?;
    if value["schema_version"] != 1 || value["request_sha256"] != identity::hash(&bytes) {
        return Err("compatibility identity receipt correlation failed".into());
    }
    Ok(value)
}
fn versions(
    value: &serde_json::Value,
    output: &Path,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<serde_json::Value> {
    let mut versions = json!({});
    for side in ["current", "released"] {
        let raw = probe(
            &ProcessSpec {
                executable: value[side]["path"]
                    .as_str()
                    .ok_or("binary identity")?
                    .into(),
                arguments: vec![Value::Public("--version".into())],
                cwd: output.into(),
                environment: BTreeMap::new(),
            },
            output,
            &format!("{side}-version"),
            deadline,
            cancel,
        )?;
        let bytes = raw.stdout.ok_or("version stdout missing")?;
        if bytes.as_bytes().len() as u64 != raw.process.stdout.bytes_seen
            || raw.process.stdout.truncated
            || !raw.process.stdout.line_capture_complete
        {
            return Err("version stdout incomplete".into());
        }
        let text = std::str::from_utf8(bytes.as_bytes())?.trim();
        if text.is_empty() || text.len() > 4096 || text.contains(['\r', '\n']) {
            return Err("version output invalid".into());
        }
        versions[side] = text.into();
    }
    Ok(versions)
}
fn execute(
    options: &Options,
    cancel: &Cancellation,
    deadline: Instant,
) -> DynResult<serde_json::Value> {
    let admitted = identity(
        &options.input,
        &options.output,
        "identity-before",
        deadline,
        cancel,
    )?;
    let versions = versions(&admitted, &options.output, deadline, cancel)?;
    let execution = remaining(deadline, 14)?;
    let steps = setup::prepare(
        &options.output,
        &admitted,
        execution.min(Duration::from_secs(95)),
    )?;
    let (jobs, requests) = std::sync::mpsc::sync_channel(1);
    let (results, responses) = std::sync::mpsc::sync_channel(1);
    let mut owner = owner::Owner {
        steps,
        jobs,
        results: responses,
        waiting: false,
        token: String::new(),
        cases: Vec::new(),
    };
    let checks_deadline = Instant::now() + execution;
    let (session, joined, cases) = std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            while let Ok(check) = requests.recv() {
                if results
                    .send(http_checks::execute(check, checks_deadline, cancel))
                    .is_err()
                {
                    break;
                }
            }
        });
        let report = process::retained::run(
            &mut owner,
            &Limits {
                execution,
                graceful_shutdown: Duration::from_secs(2),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 1048576,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            cancel,
        );
        let cases = std::mem::take(&mut owner.cases);
        drop(owner.jobs);
        (report, worker.join().is_ok(), cases)
    });
    let successful = joined
        && session.as_ref().is_ok_and(|r| r.success())
        && cases.len() == 3
        && cases.iter().all(|row| row["passed"] == true);
    let members=session.as_ref().map(|r|r.members.iter().map(|m|json!({"name":String::from_utf8_lossy(m.member.name()),"pid":m.process.pid,"disposition":m.disposition.label(),"clean":m.process.success(),"cleanup_complete":m.process.cleanup.complete,"cleanup_forced":m.process.cleanup.forced,"stdout_suppressed_lines":m.process.stdout.suppressed_lines,"stderr_suppressed_lines":m.process.stderr.suppressed_lines,"stdout_bytes_seen":m.process.stdout.bytes_seen,"stderr_bytes_seen":m.process.stderr.bytes_seen})).collect::<Vec<_>>()).unwrap_or_default();
    let after_input = identity::Input {
        current: options.input.current.clone(),
        released: options.input.released.clone(),
        model: options.input.model.clone(),
        evidence_logs: Some(options.output.clone()),
    };
    let after = identity(
        &after_input,
        &options.output,
        "identity-after",
        deadline,
        cancel,
    );
    let unchanged = after.as_ref().is_ok_and(|v| {
        ["current", "released", "model"]
            .iter()
            .all(|key| v[*key] == admitted[*key])
    });
    Ok(
        json!({"schema_version":1,"passed":successful&&unchanged,"cases":cases,"current_binary_sha256":admitted["current"]["sha256"],"released_binary_sha256":admitted["released"]["sha256"],"model_sha256":admitted["model"]["sha256"],"released_version":versions["released"],"identity":admitted,"logs":after.as_ref().ok().map(|v|&v["logs"]),"host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH},"versions":versions,"declared_released_dialect":"serve-client-bind-port","released_cli_scope":"operator_declared_not_real_bundle_qualification","worker_joined":joined,"source_unchanged":unchanged,"members":members,"session_error":session.err().map(|_|"owned_session_failure"),"state_scope":"new_per_node_profiles_no_wallet_credentials_or_funding","response_scope":"typed_status_content_usage_and_hash_no_raw_invites_or_responses"}),
    )
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if let [verb, rest @ ..] = args
        && verb == "identity-worker"
    {
        return identity::worker(rest);
    }
    if args == ["--help"] || args == ["-h"] {
        println!("{USAGE}");
        return Ok(());
    }
    let options = Options::parse(args)?;
    let interrupt = super::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(options.seconds);
    std::fs::create_dir_all(options.output.parent().ok_or("output parent")?)?;
    std::fs::create_dir(&options.output)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&options.output, std::fs::Permissions::from_mode(0o700))?;
    }
    let result = execute(&options, &cancel, deadline);
    let restored = interrupt.finish();
    let report=result.unwrap_or_else(|_|json!({"schema_version":1,"passed":false,"cases":[],"error":"compatibility_preflight_or_orchestration_failed","evidence_scope":"inspect_owned_preflight_and_node_logs"}));
    let report = terminal::admit(report, restored.is_ok(), cancel.is_cancelled(), deadline);
    identity::fresh(
        &options.output.join("results.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}",
        json!({"output":options.output,"passed":report["passed"]})
    );
    restored?;
    if report["passed"] != true {
        return Err("compatibility failed; partial evidence retained".into());
    }
    Ok(())
}
