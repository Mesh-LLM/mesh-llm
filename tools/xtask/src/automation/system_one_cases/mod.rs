//! Request qualification for an already running System One frontend, not readiness.
mod contract;
mod full_read;
mod requests;
mod transport;
mod validation;
use crate::{
    command::DynResult,
    command_interrupt::Interrupt,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Duration};
struct Options {
    endpoint: String,
    model: String,
    alias: String,
    mode: String,
    timeout: Duration,
    output: Option<PathBuf>,
}
impl Options {
    fn parse(args: &[String]) -> DynResult<Self> {
        const G: Grammar = Grammar {
            usage: "automation system-one-cases --base-url URL --model ID --mode contract|full-read [--alias ID --timeout SECONDS --json-out PATH]",
            values: &[
                "--base-url",
                "--model",
                "--mode",
                "--alias",
                "--timeout",
                "--json-out",
            ],
            flags: &[],
        };
        let parsed = G.parse(args).map_err(|_| "invalid System One arguments")?;
        if !parsed.positionals.is_empty() {
            return Err("System One accepts named inputs only".into());
        }
        for key in G.values {
            if parsed.all(key).len() > 1 {
                return Err("duplicate System One option".into());
            }
        }
        let required = |key| parsed.last(key).ok_or("missing System One option");
        let model = required("--model")?.to_owned();
        let alias = parsed
            .last("--alias")
            .unwrap_or("openjev-latest")
            .to_owned();
        for id in [&model, &alias] {
            if id.trim().is_empty() || id.len() > 4096 || id.contains(['\0', '\r', '\n']) {
                return Err("invalid System One model identity".into());
            }
        }
        let base = required("--base-url")?.trim_end_matches('/');
        let endpoint = format!("{base}/systemone");
        transport::validate_endpoint(&endpoint)?;
        let mode = required("--mode")?.to_owned();
        if !matches!(mode.as_str(), "contract" | "full-read") {
            return Err("invalid System One mode".into());
        }
        let seconds: f64 = parsed.last("--timeout").unwrap_or("600").parse()?;
        if !seconds.is_finite() || seconds <= 0.0 || seconds > 3600.0 {
            return Err("System One timeout must be finite in (0,3600] seconds".into());
        }
        let output = parsed.last("--json-out").map(PathBuf::from);
        if output.as_ref().is_some_and(|p| p.as_os_str().len() > 4096) {
            return Err("System One report path too long".into());
        }
        if output
            .as_ref()
            .is_some_and(|p| std::fs::symlink_metadata(p).is_ok_and(|m| !m.file_type().is_file()))
        {
            return Err("System One report must be regular".into());
        }
        Ok(Self {
            endpoint,
            model,
            alias,
            mode,
            timeout: Duration::from_secs_f64(seconds),
            output,
        })
    }
}
#[derive(Debug)]
enum Failure {
    Case(&'static str),
    Transport(&'static str),
}
type Result<T> = std::result::Result<T, Failure>;
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let options = match Options::parse(args) {
        Ok(value) => value,
        Err(_) => {
            return CheckReport {
                stdout: String::new(),
                stderr: "invalid System One inputs\n".into(),
                code: 2,
            }
            .emit();
        }
    };
    let interrupt = match Interrupt::install() {
        Ok(value) => value,
        Err(_) => return environment_error("System One signal scope unavailable"),
    };
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(value) => value,
        Err(_) => return environment_error("System One HTTP runtime unavailable"),
    };
    let mut cases = Vec::new();
    let client = transport::Client {
        options: &options,
        cancellation: interrupt.cancellation(),
    };
    let outcome = runtime.block_on(async {
        if options.mode == "contract" {
            contract::run(&client, &mut cases).await
        } else {
            full_read::run(&client, &mut cases).await
        }
    });
    // Dropped request futures cannot stop a blocking OS hostname resolver.
    // Bound runtime shutdown before restoring signals and emitting any report.
    shutdown_http_runtime(runtime);
    let outcome = match interrupt.finish() {
        Ok(()) => outcome,
        Err(_) => Err(Failure::Transport("System One operation cancelled")),
    };
    let (status, failures, code) = match outcome {
        Ok(()) => ("pass", vec![], 0),
        Err(Failure::Case(message)) => ("fail", vec![message], 1),
        Err(Failure::Transport(message)) => ("error", vec![message], 2),
    };
    let report = json!({"schema_version":1,"mode":options.mode,"model":options.model,"cases":cases,"status":status,"failures":failures});
    if let Some(path) = &options.output
        && write_report(path, &report).is_err()
    {
        return CheckReport {
            stdout: String::new(),
            stderr: "System One report destination failed\n".into(),
            code: 2,
        }
        .emit();
    }
    let stderr = if code == 0 {
        format!("system-one {}: matrix passed\n", options.mode)
    } else {
        format!("system-one {}: {status}\n", options.mode)
    };
    CheckReport {
        stdout: String::new(),
        stderr,
        code,
    }
    .emit()
}
fn write_report(path: &std::path::Path, report: &Value) -> DynResult<()> {
    use std::{fs::OpenOptions, io::Write};
    if std::fs::symlink_metadata(path).is_ok_and(|m| !m.file_type().is_file()) {
        return Err("System One report must be a regular file".into());
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    if bytes.len() > 65536 {
        return Err("System One report exceeds 64 KiB".into());
    }
    let mut options = OpenOptions::new();
    options.write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("System One report must be regular".into());
    }
    file.set_len(0)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    Ok(())
}
fn passed(cases: &mut Vec<Value>, name: &str) {
    cases.push(json!({"name":name,"status":"pass"}));
}

#[cfg(test)]
mod tests;

fn environment_error(message: &str) -> DynResult<()> {
    CheckReport {
        stdout: String::new(),
        stderr: format!("{message}\n"),
        code: 2,
    }
    .emit()
}

fn shutdown_http_runtime(runtime: tokio::runtime::Runtime) {
    runtime.shutdown_timeout(Duration::from_millis(100));
}
