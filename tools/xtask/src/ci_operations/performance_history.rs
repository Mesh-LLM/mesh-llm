//! Schema1 competitive performance history. Agentic replay history remains separate.
mod gpu;
mod input;
mod normalize;
mod publication;
mod records;
use crate::{
    command::DynResult,
    process::Cancellation,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::path::Path;
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool ci-ops performance-history --artifact PATH --output PATH --report PATH [--baseline PATH] [--throughput-regression FRACTION] [--ttft-regression FRACTION] [--gate]",
    values: &[
        "--artifact",
        "--output",
        "--report",
        "--baseline",
        "--throughput-regression",
        "--ttft-regression",
    ],
    flags: &["--help", "--gate"],
};
pub(super) fn run(args: &[String]) -> CheckReport {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage));
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments");
    }
    let required = |key| {
        parsed
            .last(key)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("missing {key}"))
    };
    let values = (|| {
        Ok::<_, String>((
            required("--artifact")?,
            required("--output")?,
            required("--report")?,
            threshold(parsed.last("--throughput-regression"), 0.05)?,
            threshold(parsed.last("--ttft-regression"), 0.10)?,
        ))
    })();
    let (artifact, output, report, throughput, ttft) = match values {
        Ok(values) => values,
        Err(message) => return GRAMMAR.error(&message),
    };
    let options = Options {
        artifact: Path::new(artifact),
        output: Path::new(output),
        report: Path::new(report),
        baseline: parsed.last("--baseline").map(Path::new),
        throughput,
        ttft,
        gate: parsed.flag("--gate"),
    };
    let result = owned(&options);
    match result {
        Ok(code) => CheckReport {
            stdout: String::new(),
            stderr: String::new(),
            code,
        },
        Err(error) => CheckReport::failure(
            String::new(),
            format!("performance history error:{error}\n"),
        ),
    }
}
fn threshold(value: Option<&str>, fallback: f64) -> Result<f64, String> {
    let number = value
        .map_or(Ok(fallback), str::parse::<f64>)
        .map_err(|_| "invalid regression threshold")?;
    if !number.is_finite() || number < 0.0 {
        return Err("regression threshold must be finite and nonnegative".into());
    }
    Ok(number)
}
struct Options<'a> {
    artifact: &'a Path,
    output: &'a Path,
    report: &'a Path,
    baseline: Option<&'a Path>,
    throughput: f64,
    ttft: f64,
    gate: bool,
}
fn owned(options: &Options<'_>) -> DynResult<i32> {
    let interrupt = crate::command_interrupt::Interrupt::install()?;
    let result = execute(options, &interrupt.cancellation());
    interrupt.finish()?;
    result
}
fn execute(options: &Options<'_>, cancellation: &Cancellation) -> DynResult<i32> {
    let mut input = input::Input::new(cancellation);
    let current = normalize::normalize(&mut input, options.artifact)?;
    let history = input.history(options.baseline)?;
    let comparisons = records::compare(
        &current,
        &history,
        options.throughput,
        options.ttft,
        cancellation,
    )?;
    let text = records::report(&comparisons);
    let mut shard = Vec::new();
    for row in &current {
        input.check()?;
        serde_json::to_writer(&mut shard, row)?;
        shard.push(b'\n');
    }
    let failed = options.gate
        && comparisons.iter().any(|item| {
            matches!(
                item.classification,
                "correctness-failure" | "performance-regression"
            )
        });
    publication::publish(
        &mut input,
        [(options.output, &shard), (options.report, text.as_bytes())],
    )?;
    Ok(i32::from(failed))
}
