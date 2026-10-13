use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Upstream,
    Depot,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Observation {
    pub source: Source,
    pub sample: u32,
    pub elapsed_ms: u32,
    pub digest: String,
}

#[derive(Debug, Serialize)]
pub struct SampleCounts {
    pub upstream: usize,
    pub depot: usize,
}

#[derive(Debug, Serialize)]
pub struct Summary {
    pub digest: String,
    pub upstream_median_ms: f64,
    pub depot_median_ms: f64,
    pub improvement_ms: f64,
    pub improvement_percent: f64,
    pub eligible: bool,
    pub samples_per_source: SampleCounts,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: need the configured number of unique samples")]
    Samples(&'static str),
    #[error("upstream and Depot observations used different digests")]
    DigestMismatch,
    #[error("digest must be a sha256 digest")]
    Digest,
    #[error("minimum samples must be positive")]
    MinimumSamples,
    #[error("invalid registry-pulls arguments")]
    Arguments,
    #[error("observation I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("observation JSON failed: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn summarize(observations: &[Observation], minimum: usize) -> Result<Summary, Error> {
    if minimum == 0 {
        return Err(Error::MinimumSamples);
    }
    let upstream = select(observations, Source::Upstream, minimum)?;
    let depot = select(observations, Source::Depot, minimum)?;
    let digest = observations
        .first()
        .ok_or(Error::Samples("upstream"))?
        .digest
        .clone();
    if !digest.starts_with("sha256:") {
        return Err(Error::Digest);
    }
    if observations.iter().any(|item| item.digest != digest) {
        return Err(Error::DigestMismatch);
    }
    let upstream_median_ms = median(&upstream);
    let depot_median_ms = median(&depot);
    let improvement_ms = upstream_median_ms - depot_median_ms;
    let improvement_percent = if upstream_median_ms > 0.0 {
        improvement_ms / upstream_median_ms * 100.0
    } else {
        0.0
    };
    Ok(Summary {
        digest,
        upstream_median_ms,
        depot_median_ms,
        improvement_ms,
        improvement_percent,
        eligible: improvement_ms >= 10_000.0 && improvement_percent >= 20.0,
        samples_per_source: SampleCounts {
            upstream: upstream.len(),
            depot: depot.len(),
        },
    })
}

fn select(observations: &[Observation], source: Source, minimum: usize) -> Result<Vec<u32>, Error> {
    let mut identifiers = BTreeSet::new();
    let mut times = Vec::new();
    for observation in observations.iter().filter(|item| item.source == source) {
        if !identifiers.insert(observation.sample) {
            return Err(Error::Samples(match source {
                Source::Upstream => "upstream",
                Source::Depot => "depot",
            }));
        }
        times.push(observation.elapsed_ms);
    }
    if times.len() < minimum {
        return Err(Error::Samples(match source {
            Source::Upstream => "upstream",
            Source::Depot => "depot",
        }));
    }
    times.sort_unstable();
    Ok(times)
}

fn median(times: &[u32]) -> f64 {
    let middle = times.len() / 2;
    if times.len().is_multiple_of(2) {
        (f64::from(times[middle - 1]) + f64::from(times[middle])) / 2.0
    } else {
        f64::from(times[middle])
    }
}

pub fn markdown(summary: &Summary) -> String {
    format!(
        "## Depot Registry pull-through result\n\n| Signal | Value |\n| --- | ---: |\n| Upstream median | {:.3}s |\n| Depot median | {:.3}s |\n| Improvement | {:.3}s ({:.1}%) |\n| Adoption gate | {} |\n\nDigest: `{}`\n\nThe gate requires at least five fresh-runner samples per source, identical manifest digests, at least 20% improvement, and at least 10 seconds saved at the median.\n",
        summary.upstream_median_ms / 1000.0,
        summary.depot_median_ms / 1000.0,
        summary.improvement_ms / 1000.0,
        summary.improvement_percent,
        if summary.eligible { "pass" } else { "fail" },
        summary.digest
    )
}

pub fn load(root: &Path) -> Result<Vec<Observation>, Error> {
    let mut observations = Vec::new();
    let mut entries = std::fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::path);
    for entry in entries {
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            observations.extend(load(&path)?);
        } else if kind.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension == "json")
        {
            observations.push(serde_json::from_slice(&std::fs::read(path)?)?);
        }
    }
    Ok(observations)
}

pub struct CommandOutput {
    pub stdout: String,
    pub code: i32,
}

pub fn run(args: &[String]) -> Result<CommandOutput, Error> {
    if args == ["--help"] {
        return Ok(CommandOutput {
            stdout: "usage: ci-ops registry-pulls DIRECTORY [--minimum-samples N] [--json-out PATH] [--markdown-out PATH] [--enforce]\n".into(),
            code: 0,
        });
    }
    let mut directory = None;
    let mut minimum = 5;
    let mut json_output = None;
    let mut markdown_output = None;
    let mut enforce = false;
    let mut arguments = args.iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--minimum-samples" => {
                minimum = arguments
                    .next()
                    .ok_or(Error::Arguments)?
                    .parse()
                    .map_err(|_| Error::Arguments)?
            }
            "--json-out" => json_output = Some(arguments.next().ok_or(Error::Arguments)?),
            "--markdown-out" => markdown_output = Some(arguments.next().ok_or(Error::Arguments)?),
            "--enforce" => enforce = true,
            value if !value.starts_with('-') && directory.is_none() => {
                directory = Some(Path::new(value))
            }
            _ => return Err(Error::Arguments),
        }
    }
    let summary = summarize(&load(directory.ok_or(Error::Arguments)?)?, minimum)?;
    if let Some(path) = json_output {
        std::fs::write(path, serde_json::to_string_pretty(&summary)? + "\n")?;
    }
    let report = markdown(&summary);
    let stdout = match markdown_output {
        Some(path) => {
            std::fs::write(path, report)?;
            String::new()
        }
        None => report,
    };
    Ok(CommandOutput {
        stdout,
        code: i32::from(enforce && !summary.eligible),
    })
}
