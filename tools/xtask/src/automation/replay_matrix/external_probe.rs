use super::external_config::{Arm, Config};
use crate::automation::command_interrupt::Interrupt;
use crate::command::DynResult;
use crate::process::{Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value, supervise};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Duration};

pub(super) const VERSION_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Deserialize, Serialize)]
pub(super) struct Verified {
    pub label: String,
    pub engine: super::external_config::Engine,
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: String,
    pub version: String,
    pub version_sha256: String,
    pub worktree: PathBuf,
    pub binary: String,
    pub runtime_root: String,
    pub backend: String,
    pub served_model: String,
    pub external_engine: Arm,
    pub provenance: Provenance,
}

#[derive(Deserialize, Serialize)]
pub(super) struct Provenance {
    #[serde(flatten)]
    pub arm: Arm,
    pub resolved_executable: PathBuf,
    pub version: String,
    pub version_sha256: String,
}

pub(super) fn verify(arm: &Arm) -> DynResult<Verified> {
    verify_with_budget(arm, VERSION_TIMEOUT)
}

pub(super) fn verify_with_budget(arm: &Arm, execution: Duration) -> DynResult<Verified> {
    if execution.is_zero() || execution > VERSION_TIMEOUT {
        return Err("external version budget must be positive and at most 30 seconds".into());
    }
    arm.validate_prepared()?;
    let executable = super::external_command::executable(arm)?;
    let spec = ProcessSpec {
        executable: executable.clone(),
        arguments: super::external_command::version(arm)
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        cwd: arm.cwd.clone(),
        environment: environment(),
    };
    #[cfg(not(test))]
    let interrupt = Interrupt::install()?;
    #[cfg(test)]
    let interrupt = fixture_interrupt()?;
    let limits = Limits {
        execution,
        graceful_shutdown: Duration::from_secs(2),
        forced_shutdown: Duration::from_secs(3),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = supervise(
        &spec,
        &limits,
        &interrupt.cancellation(),
        OutputFiles::default(),
    )?;
    interrupt.check()?;
    if !report.success() || report.stdout.truncated || report.stderr.truncated {
        return Err(format!("{} version probe failed: {:?}", arm.label, report.outcome).into());
    }
    let bytes = if report.stdout.bytes_retained.is_empty() {
        report.stderr.bytes_retained
    } else {
        report.stdout.bytes_retained
    };
    let version = String::from_utf8(bytes)?.trim().to_owned();
    if version.is_empty() {
        return Err("external version probe returned no version".into());
    }
    let digest = hex::encode(Sha256::digest(version.as_bytes()));
    Ok(Verified {
        label: arm.label.clone(),
        engine: arm.engine,
        reference: format!("{}@{version}", arm.engine.name()),
        commit: digest.clone(),
        version: version.clone(),
        version_sha256: digest.clone(),
        worktree: arm.cwd.clone(),
        binary: arm.executable.clone(),
        runtime_root: String::new(),
        backend: arm.engine.name().into(),
        served_model: arm.served_model()?.into(),
        external_engine: arm.clone(),
        provenance: Provenance {
            arm: arm.clone(),
            resolved_executable: executable,
            version,
            version_sha256: digest,
        },
    })
}

pub(super) fn environment() -> std::collections::BTreeMap<std::ffi::OsString, Value> {
    std::env::vars_os()
        .map(|(key, value)| (key, Value::Public(value)))
        .collect()
}

#[derive(Serialize)]
pub(super) struct Plan {
    pub engine_config: Config,
    pub builds: Vec<Verified>,
    pub external_server_commands: Vec<Vec<String>>,
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix external-config --config PATH --model ID --output PATH [--minimum-context TOKENS]",
        values: &["--config", "--model", "--output", "--minimum-context"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let config = super::external_config::load(std::path::Path::new(
        parsed.last("--config").ok_or("missing --config")?,
    ))?;
    config.admit(
        parsed.last("--model").ok_or("missing --model")?,
        &[],
        parsed.last("--minimum-context").unwrap_or("0").parse()?,
    )?;
    let output = std::path::absolute(parsed.last("--output").ok_or("missing --output")?)?;
    if output.try_exists()? {
        return Err("external plan output already exists".into());
    }
    let builds = config
        .arms
        .iter()
        .map(verify)
        .collect::<DynResult<Vec<_>>>()?;
    let commands = builds
        .iter()
        .map(|build| {
            super::external_command::server(
                &build.external_engine,
                &build.provenance.resolved_executable,
                9337,
            )
        })
        .collect::<DynResult<Vec<_>>>()?;
    crate::command::write_json_file(
        &output,
        &Plan {
            engine_config: config,
            builds,
            external_server_commands: commands,
        },
    )
}

// Unit fixture version probes share a process-global signal scope with unrelated
// parallel owner fixtures. Production continues rejecting overlap immediately.
#[cfg(test)]
fn fixture_interrupt() -> Result<Interrupt, crate::automation::command_interrupt::Reason> {
    use crate::automation::command_interrupt::Reason;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        match Interrupt::install() {
            Err(Reason::ScopeBusy) if std::time::Instant::now() < deadline => {
                std::thread::park_timeout(Duration::from_millis(1));
            }
            result => return result,
        }
    }
}
