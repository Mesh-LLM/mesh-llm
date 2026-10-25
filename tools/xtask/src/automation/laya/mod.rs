mod http;
mod parity;
mod product;
mod request;

use crate::command::DynResult;
use crate::process::{
    Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value, supervise_raw,
};
use crate::repository::check_args::{Grammar, ParsedArgs};
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};
use std::time::Duration;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation laya parity (--base-url URL --model ID | --cli PATH --gguf PATH) [--fixtures DIR] [--device NAME] [--timeout SECONDS] [--json-out PATH]",
    values: &[
        "--base-url",
        "--model",
        "--cli",
        "--gguf",
        "--fixtures",
        "--device",
        "--timeout",
        "--json-out",
    ],
    flags: &["--help"],
};

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    let [verb, rest @ ..] = args else {
        return GRAMMAR.error("expected parity").emit();
    };
    if verb == "product" {
        return product::run(root, rest);
    }
    if verb == "product-worker" {
        return product::worker(root, rest);
    }
    if verb != "parity" {
        return GRAMMAR.error("expected parity").emit();
    }
    let parsed = match GRAMMAR.parse(rest) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    let source = match Source::parse(&parsed) {
        Ok(source) => source,
        Err(error) => return GRAMMAR.error(&error.to_string()).emit(),
    };
    let timeout = budget(parsed.last("--timeout").unwrap_or("300"))?;
    let fixtures = parsed
        .last("--fixtures")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("ci/llama-canary/fixtures/laya-golden"));
    match battery(
        root,
        &fixtures,
        &source,
        timeout,
        parsed.last("--json-out").map(Path::new),
    ) {
        Ok(true) => CheckReport::success("laya parity: pass\n".into()).emit(),
        Ok(false) => CheckReport::failure(String::new(), "laya parity: failed\n".into()).emit(),
        Err(error) => CheckReport::usage(GRAMMAR.usage, &error.to_string()).emit(),
    }
}

pub(super) enum Source {
    Http {
        base_url: String,
        model: String,
    },
    Cli {
        executable: PathBuf,
        gguf: PathBuf,
        device: Option<String>,
    },
}

impl Source {
    fn parse(parsed: &ParsedArgs) -> DynResult<Self> {
        if !parsed.positionals.is_empty() {
            return Err("unexpected positional arguments".into());
        }
        match (parsed.last("--base-url"), parsed.last("--cli")) {
            (Some(base_url), None) => Ok(Self::Http {
                base_url: base_url.into(),
                model: parsed
                    .last("--model")
                    .ok_or("--base-url needs --model")?
                    .into(),
            }),
            (None, Some(cli)) => Ok(Self::Cli {
                executable: Path::new(cli).canonicalize()?,
                gguf: PathBuf::from(parsed.last("--gguf").ok_or("--cli needs --gguf")?),
                device: parsed.last("--device").map(str::to_owned),
            }),
            _ => Err("give exactly one of --base-url or --cli".into()),
        }
    }

    fn read(
        &self,
        root: &Path,
        fixture: &Path,
        golden: &parity::Golden,
        timeout: Duration,
    ) -> DynResult<parity::Response> {
        match self {
            Self::Http { base_url, model } => {
                let body = request::body(model, golden)?;
                let bytes = http::request(
                    &format!("{}/systemone", base_url.trim_end_matches('/')),
                    Some(body),
                    timeout,
                )?;
                let mut response: parity::Response = serde_json::from_slice(&bytes)?;
                response.per_question = None;
                Ok(response)
            }
            Self::Cli {
                executable,
                gguf,
                device,
            } => {
                let mut arguments = vec![
                    "-m".into(),
                    gguf.as_os_str().to_owned(),
                    "-f".into(),
                    fixture.as_os_str().to_owned(),
                ];
                if let Some(device) = device {
                    arguments.extend(["--device".into(), device.into()]);
                }
                let spec = ProcessSpec {
                    executable: executable.clone(),
                    arguments: arguments.into_iter().map(Value::Public).collect(),
                    cwd: root.canonicalize()?,
                    environment: std::env::vars_os()
                        .map(|(key, value)| (key, Value::Public(value)))
                        .collect(),
                };
                let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
                let result = supervise_raw(
                    &spec,
                    &limits(timeout),
                    &interrupt.cancellation(),
                    RawCaptureOptions {
                        stdout: std::num::NonZeroUsize::new(16 * 1024 * 1024),
                        stderr: None,
                    },
                )?;
                interrupt.finish()?;
                let report = result;
                if !report.process.success() {
                    return Err(format!("Laya CLI failed: {:?}", report.process.outcome).into());
                }
                let bytes = report.stdout.ok_or("missing CLI stdout")?;
                let response: parity::Response = serde_json::from_slice(bytes.as_bytes())?;
                if response.per_question.is_none() {
                    return Err("CLI response requires per_question token ids".into());
                }
                Ok(response)
            }
        }
    }
}

pub(super) fn budget(value: &str) -> DynResult<Duration> {
    let seconds: f64 = value.parse()?;
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 86400.0 {
        return Err("timeout must be positive and at most one day".into());
    }
    Ok(Duration::try_from_secs_f64(seconds)?)
}

pub(super) fn limits(execution: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: Duration::from_secs(10),
        forced_shutdown: Duration::from_secs(10),
        retained_bytes_per_stream: 12_000,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

pub(super) fn battery(
    root: &Path,
    fixtures: &Path,
    source: &Source,
    timeout: Duration,
    output: Option<&Path>,
) -> DynResult<bool> {
    let mut paths = std::fs::read_dir(fixtures)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
            && path.file_stem().is_some_and(|stem| stem != "manifest")
    });
    paths.sort();
    if paths.is_empty() {
        return Err("no Laya fixtures found".into());
    }
    let mut results = Vec::new();
    for path in paths {
        let golden = serde_json::from_slice(&std::fs::read(&path)?)?;
        let name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or("invalid fixture name")?;
        let response = source.read(root, &path, &golden, timeout)?;
        results.push(parity::compare(name, &golden, &response)?);
    }
    let passed = results.iter().all(|result| result.failures.is_empty());
    if let Some(output) = output {
        std::fs::write(
            output,
            serde_json::to_vec_pretty(&serde_json::json!({"results": results, "passed": passed}))?,
        )?;
    }
    Ok(passed)
}
