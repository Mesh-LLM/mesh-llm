//! Compare Google Benchmark's paired FFI/API samples only after correctness admission.
use crate::command::DynResult;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Write as _};
use std::path::Path;

const MAX_REPORT_BYTES: u64 = 16 * 1024 * 1024;
const USAGE: &str = "cargo xtool automation skippy-benchmark-compare <report.json>";

#[derive(Deserialize)]
struct Report {
    benchmarks: Vec<Sample>,
}

#[derive(Deserialize)]
struct Sample {
    name: String,
    #[serde(default)]
    error_occurred: bool,
    error_message: Option<String>,
    real_time: Option<f64>,
    time_unit: Option<String>,
    run_type: Option<String>,
    aggregate_name: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Side {
    Ffi,
    Api,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum Statistic {
    Median,
    Minimum,
}

#[derive(Default)]
struct Pair {
    ffi: Vec<f64>,
    api: Vec<f64>,
}

struct Comparison {
    family: String,
    argument: String,
    statistic: Statistic,
    ffi_ns: f64,
    api_ns: f64,
    ratio: f64,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let [path] = args else {
        return Err(USAGE.into());
    };
    let mut bytes = Vec::new();
    File::open(Path::new(path))?
        .take(MAX_REPORT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_REPORT_BYTES {
        return Err("benchmark report exceeds 16 MiB".into());
    }
    let text = compare(&bytes)?;
    crate::cli_output::stdout().write_all(text.as_bytes())?;
    Ok(())
}

fn compare(bytes: &[u8]) -> DynResult<String> {
    let report: Report = serde_json::from_slice(bytes)?;
    let comparisons = admit(report)?;
    render(&comparisons)
}

fn admit(report: Report) -> DynResult<Vec<Comparison>> {
    if report.benchmarks.is_empty() {
        return Err(
            "benchmark report contains no rows; the filter may have matched nothing".into(),
        );
    }
    let mut pairs: BTreeMap<(String, String, Statistic), Pair> = BTreeMap::new();
    for sample in report.benchmarks {
        // Google Benchmark reports correctness failures in JSON even when its
        // process exits zero. Never turn a rejected selected case into a skip.
        if sample.error_occurred {
            return Err(format!(
                "benchmark {} rejected: {}",
                sample.name,
                sample
                    .error_message
                    .as_deref()
                    .unwrap_or("unspecified error")
            )
            .into());
        }
        let Some(statistic) = statistic(&sample) else {
            continue;
        };
        let name = if statistic == Statistic::Minimum {
            sample.name.strip_suffix("_min").unwrap_or(&sample.name)
        } else {
            &sample.name
        };
        let Some((family, side, argument)) = split_name(name)? else {
            continue;
        };
        let duration = duration_ns(&sample)?;
        let pair = pairs.entry((family, argument, statistic)).or_default();
        match side {
            Side::Ffi => pair.ffi.push(duration),
            Side::Api => pair.api.push(duration),
        }
    }
    if pairs.is_empty() {
        return Err("benchmark report contains no comparable FFI/API samples".into());
    }
    derive_minima(&mut pairs);
    let mut comparisons = pairs
        .into_iter()
        .map(|((family, argument, statistic), pair)| comparison(family, argument, statistic, pair))
        .collect::<DynResult<Vec<_>>>()?;
    comparisons.sort_by(|left, right| {
        left.family
            .cmp(&right.family)
            .then_with(|| argument_key(&left.argument).cmp(&argument_key(&right.argument)))
            .then_with(|| left.statistic.cmp(&right.statistic))
    });
    Ok(comparisons)
}

fn derive_minima(pairs: &mut BTreeMap<(String, String, Statistic), Pair>) {
    // Google Benchmark does not emit a minimum aggregate by default. Preserve
    // the median comparison and derive one additional pair from repetitions.
    let derived = pairs
        .iter()
        .filter(|((_, _, statistic), pair)| {
            *statistic == Statistic::Median && (pair.ffi.len() > 1 || pair.api.len() > 1)
        })
        .filter_map(|((family, argument, _), pair)| {
            let key = (family.clone(), argument.clone(), Statistic::Minimum);
            if pairs.contains_key(&key) {
                return None;
            }
            let ffi = pair.ffi.iter().copied().min_by(f64::total_cmp)?;
            let api = pair.api.iter().copied().min_by(f64::total_cmp)?;
            Some((
                key,
                Pair {
                    ffi: vec![ffi],
                    api: vec![api],
                },
            ))
        })
        .collect::<Vec<_>>();
    pairs.extend(derived);
}

fn statistic(sample: &Sample) -> Option<Statistic> {
    if sample.aggregate_name.as_deref() == Some("min") || sample.name.ends_with("_min") {
        Some(Statistic::Minimum)
    } else if sample.run_type.as_deref() == Some("aggregate")
        || ["_mean", "_median", "_stddev", "_cv"]
            .iter()
            .any(|suffix| sample.name.ends_with(suffix))
    {
        None
    } else {
        Some(Statistic::Median)
    }
}

fn split_name(name: &str) -> DynResult<Option<(String, Side, String)>> {
    let parts = name
        .split('/')
        .filter(|part| {
            !matches!(*part, "real_time" | "cpu_time" | "manual_time")
                && !part.starts_with("iterations:")
                && !part.starts_with("repeats:")
        })
        .collect::<Vec<_>>();
    let Some(index) = parts.iter().position(|part| matches!(*part, "FFI" | "API")) else {
        return Ok(None);
    };
    if index == 0
        || parts.iter().any(|part| part.is_empty())
        || parts[index + 1..]
            .iter()
            .any(|part| matches!(*part, "FFI" | "API"))
    {
        return Err(format!("malformed paired benchmark name: {name}").into());
    }
    let side = if parts[index] == "FFI" {
        Side::Ffi
    } else {
        Side::Api
    };
    Ok(Some((
        parts[..index].join("/"),
        side,
        parts[index + 1..].join("/"),
    )))
}

fn duration_ns(sample: &Sample) -> DynResult<f64> {
    let multiplier = match sample.time_unit.as_deref() {
        Some("ns") => 1.0,
        Some("us") => 1_000.0,
        Some("ms") => 1_000_000.0,
        Some("s") => 1_000_000_000.0,
        _ => return Err(format!("benchmark {} has no supported time_unit", sample.name).into()),
    };
    let duration = sample
        .real_time
        .ok_or_else(|| format!("benchmark {} has no real_time", sample.name))?
        * multiplier;
    if !duration.is_finite() || duration <= 0.0 {
        return Err(format!(
            "benchmark {} has a nonpositive or nonfinite duration",
            sample.name
        )
        .into());
    }
    Ok(duration)
}

fn comparison(
    family: String,
    argument: String,
    statistic: Statistic,
    pair: Pair,
) -> DynResult<Comparison> {
    if pair.ffi.is_empty() || pair.api.is_empty() {
        return Err(
            format!("benchmark {family}/{argument} is missing its FFI or API counterpart").into(),
        );
    }
    let ffi_ns = median(pair.ffi);
    let api_ns = median(pair.api);
    let ratio = ffi_ns / api_ns;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(format!("benchmark {family}/{argument} has an invalid timing ratio").into());
    }
    Ok(Comparison {
        family,
        argument,
        statistic,
        ffi_ns,
        api_ns,
        ratio,
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        // Positive sorted measurements make this midpoint safe at both ends
        // of the finite domain, without overflowing their sum.
        values[middle - 1] + (values[middle] - values[middle - 1]) / 2.0
    } else {
        values[middle]
    }
}

fn argument_key(argument: &str) -> (i128, &str) {
    (argument.parse().unwrap_or(0), argument)
}

fn format_ns(value: f64) -> String {
    if value >= 1_000_000.0 {
        format!("{:.3} ms", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.3} us", value / 1_000.0)
    } else {
        format!("{value:.1} ns")
    }
}

fn render(comparisons: &[Comparison]) -> DynResult<String> {
    let mut output = String::new();
    writeln!(
        output,
        "{:<32} {:>12} {:>12} {:>12}  verdict",
        "benchmark", "FFI", "API", "API vs FFI"
    )?;
    writeln!(output, "{}", "-".repeat(86))?;
    for row in comparisons {
        let mut label = if row.argument.is_empty() {
            row.family.clone()
        } else {
            format!("{}/{}", row.family, row.argument)
        };
        if row.statistic == Statistic::Minimum {
            label.push_str("_min");
        }
        let verdict = if row.ratio >= 1.02 {
            "new faster"
        } else if row.ratio <= 0.98 {
            "old faster"
        } else {
            "same"
        };
        writeln!(
            output,
            "{label:<32} {:>12} {:>12} {:>11.3}x {verdict} ({:+.1}%)",
            format_ns(row.ffi_ns),
            format_ns(row.api_ns),
            row.ratio,
            (row.ratio - 1.0) * 100.0
        )?;
    }
    // Minimum rows supplement repetition medians; adding diagnostics must not
    // change the geomean of the original median comparisons. A report holding
    // only explicit minima retains its existing minimum-only summary.
    let minimum_only = !comparisons
        .iter()
        .any(|row| row.statistic == Statistic::Median);
    let (log_sum, count) = comparisons
        .iter()
        .filter(|row| minimum_only || row.statistic == Statistic::Median)
        .fold((0.0, 0usize), |(sum, count), row| {
            (sum + row.ratio.ln(), count + 1)
        });
    let geomean = (log_sum / count as f64).exp();
    writeln!(output, "{}", "-".repeat(86))?;
    writeln!(
        output,
        "{:<32} {:>12} {:>12} {geomean:>11.3}x",
        "geomean", "", ""
    )?;
    writeln!(
        output,
        "\n>1.0 means the skippy_api facade did the same work in less wall time."
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests;
