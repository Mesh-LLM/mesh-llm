#[expect(
    dead_code,
    reason = "shared process owner includes HTTPS APIs unused by this client lifecycle driver"
)]
#[path = "../../src/process/mod.rs"]
pub mod process;
mod protocol;
mod qa_scenarios;

use process::{Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value};
use protocol::Audit;
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let [scenario, xtask, fixture, repository] = arguments.as_slice() else {
        return Err("usage: migration_client_qa <scenario> <xtask> <fixture> <repository>".into());
    };
    let plan = qa_scenarios::plan(scenario)?;
    let root = tempfile::tempdir()?;
    let native = root.path().join("native");
    let state = root.path().join("state");
    std::fs::create_dir(&native)?;
    std::fs::create_dir(&state)?;
    std::fs::write(native.join("fixture.json"), serde_json::to_vec(&plan)?)?;
    let mut sentinel = Sentinel(
        Command::new(fixture)
            .arg("--sentinel")
            .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &native)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?,
    );
    let spec = specification([xtask, fixture, repository], (&native, &state));
    let limits = Limits {
        execution: Duration::from_secs(12),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 256 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )?;
    let audit: Audit = serde_json::from_slice(&std::fs::read(native.join("audit.json"))?)?;
    let expected = if matches!(
        scenario.as_str(),
        "success" | "string" | "structured-object"
    ) {
        0
    } else {
        1
    };
    if report.status.and_then(|status| status.code()) != Some(expected)
        || !report.cleanup.complete
        || report.cleanup.forced
        || report.failure.is_some()
        || !absent(audit.pid)?
        || sentinel.0.try_wait()?.is_some()
    {
        return Err(format!("QA outcome/cleanup mismatch: {report:?}").into());
    }
    let stdout = String::from_utf8_lossy(&report.stdout.bytes_retained);
    let stderr = String::from_utf8_lossy(&report.stderr.bytes_retained);
    if stdout.contains("synthetic-never-print") || stderr.contains("synthetic-never-print") {
        return Err("QA diagnostic disclosed synthetic secret".into());
    }
    let remaining = std::fs::read_dir(&state)?.count();
    if (scenario == "deletion") != (remaining > 0) || (expected != 0 && !stdout.is_empty()) {
        return Err("QA state deletion or success output mismatch".into());
    }
    println!("{stdout}{stderr}");
    println!(
        "scenario={scenario} cli_pid={} client_pid={} status={expected} elapsed_ms={} state_entries={remaining} client_absent=true handler={} sentinel_pid={} sentinel_alive=true secret_disclosed=false",
        report.pid,
        audit.pid,
        report.elapsed.as_millis(),
        native.join("handler").is_file(),
        sentinel.0.id()
    );
    if native.join("leaf.pid").exists() {
        let leaf = std::fs::read_to_string(native.join("leaf.pid"))?.parse()?;
        if !absent(leaf)? {
            return Err("owned descendant survived".into());
        }
        println!("descendant_pid={leaf} descendant_absent=true");
    }
    drop(sentinel);
    root.close()?;
    Ok(())
}

fn specification(inputs: [&str; 3], roots: (&Path, &Path)) -> ProcessSpec {
    let [xtask, fixture, repository] = inputs;
    let (native, state) = roots;
    let mut environment = BTreeMap::new();
    for key in [
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "DYLD_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ] {
        if let Some(value) = std::env::var_os(key).filter(|value| !value.is_empty()) {
            environment.insert(key.into(), Value::Secret(value));
        }
    }
    ProcessSpec {
        executable: PathBuf::from(xtask),
        cwd: PathBuf::from(repository),
        environment,
        arguments: [
            "automation".into(),
            "client-readiness".into(),
            "--binary".into(),
            fixture.into(),
            "--native-runtime-root".into(),
            native.as_os_str().to_owned(),
            "--state-parent".into(),
            state.as_os_str().to_owned(),
            "--ready-max-wait".into(),
            "1".into(),
            "--shutdown-max-wait".into(),
            "1".into(),
        ]
        .into_iter()
        .map(Value::Public)
        .collect(),
    }
}

#[cfg(unix)]
fn absent(pid: u32) -> Result<bool> {
    let pid = i32::try_from(pid)?;
    // SAFETY: signal zero only queries the positive PID recorded by this driver's fixture.
    let result = unsafe { libc::kill(pid, 0) };
    Ok(result < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH))
}

#[cfg(windows)]
fn absent(_: u32) -> Result<bool> {
    Err("native Windows process QA is not implemented by this Unix driver".into())
}

struct Sentinel(Child);

impl Drop for Sentinel {
    fn drop(&mut self) {
        if let Err(error) = self.0.kill() {
            eprintln!("sentinel stop failed: {error}");
        }
        if let Err(error) = self.0.wait() {
            eprintln!("sentinel reap failed: {error}");
        }
    }
}
