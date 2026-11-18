//! Manual Windows ABI cold versus exact-warm measurement receipts.
//!
//! The diagnostic workflow restores (exact-warm only) and never saves, prunes or
//! publishes a cache. These checks refuse a sample rather than relabel it: a
//! warm miss is inconclusive, never cold, and a rebuild is never a warm sample.

use super::ci_metrics_time::{Instant, elapsed, isoformat, timestamp};
use super::windows_cache_identity::{Inputs, identity, segment};
use crate::repository::check_report::CheckReport;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write as _, path::Path};

const PHASES: [&str; 4] = ["lookup_start", "lookup_end", "prepare_start", "prepare_end"];
const MAX_SAMPLES: u32 = 5;
const STAMP: &str = ".mesh-llm-build-stamp";
const USAGE: &str = "usage: ci-ops windows-abi-measurement {row --backend B --condition C --sample N | mark --receipt-dir D --phase P | admit --receipt-dir D | finish --receipt-dir D}";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Condition {
    Cold,
    ExactWarm,
}

impl Condition {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cold" => Ok(Self::Cold),
            "exact-warm" => Ok(Self::ExactWarm),
            _ => Err(format!("unsupported measurement condition {value:?}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::ExactWarm => "exact-warm",
        }
    }
}

/// File metadata of the job-owned ABI build tree; mtimes expose a rebuild that
/// rewrites identical bytes.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Tree {
    files: usize,
    digest: String,
    stamp_sha256: Option<String>,
}

pub(super) fn run(args: &[String], root: &Path) -> CheckReport {
    let env = |name: &str| std::env::var(name).unwrap_or_default();
    let result = match args.split_first() {
        Some((command, rest)) if command == "row" => {
            options(rest, &["--backend", "--condition", "--sample"])
                .and_then(|values| row(root, values[0], values[1], values[2]))
        }
        Some((command, rest)) if command == "mark" => options(rest, &["--receipt-dir", "--phase"])
            .and_then(|values| mark(Path::new(values[0]), values[1], Instant::now())),
        Some((command, rest)) if command == "admit" => {
            options(rest, &["--receipt-dir"]).and_then(|values| admit(Path::new(values[0]), env))
        }
        Some((command, rest)) if command == "finish" => {
            options(rest, &["--receipt-dir"]).and_then(|values| finish(Path::new(values[0]), env))
        }
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(output) => CheckReport::success(output),
        Err(error) => {
            CheckReport::failure(String::new(), format!("windows-abi-measurement: {error}\n"))
        }
    }
}

fn options<'a>(args: &'a [String], names: &[&str]) -> Result<Vec<&'a str>, String> {
    if args.len() != names.len() * 2 {
        return Err(USAGE.into());
    }
    args.chunks(2)
        .zip(names)
        .map(|(pair, name)| {
            (pair[0] == *name)
                .then_some(pair[1].as_str())
                .ok_or_else(|| USAGE.to_owned())
        })
        .collect()
}

/// Closed row from the same `ci/slices.yml` Windows runtime rows the runtime
/// slice builds, so the diagnostic cannot introduce a cache family.
fn row(root: &Path, backend: &str, condition: &str, sample: &str) -> Result<String, String> {
    let condition = Condition::parse(condition)?;
    let sample = sample
        .parse::<u32>()
        .ok()
        .filter(|value| (1..=MAX_SAMPLES).contains(value))
        .ok_or("sample must be an integer from 1 to 5")?;
    let bytes = fs::read(root.join("ci/slices.yml")).map_err(|error| error.to_string())?;
    let catalog: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let rows = catalog["runtime_rows"]
        .as_array()
        .ok_or("runtime rows missing")?;
    let selected: Vec<_> = rows
        .iter()
        .filter(|row| row["platform"] == "windows" && row["backend"] == backend)
        .collect();
    let [row] = selected.as_slice() else {
        return Err(format!(
            "no unique Windows runtime row for backend {backend:?}"
        ));
    };
    let field = |name: &str| row[name].as_str().unwrap_or("").to_owned();
    let (cuda, rocm) = (field("cuda_architectures"), field("rocm_architectures"));
    let (build_dir, target) = (field("build_dir"), field("target"));
    for value in [&build_dir, &target, &cuda, &rocm] {
        if value.contains(['\r', '\n']) {
            return Err("Windows runtime row values must be single-line".into());
        }
    }
    if build_dir.is_empty() || target.is_empty() {
        return Err("Windows runtime row lacks build_dir or target".into());
    }
    let architectures = if cuda.is_empty() { &rocm } else { &cuda };
    Ok(format!(
        "backend={backend}\nbuild_dir={build_dir}\ntarget={target}\narchitecture_set={architectures}\ncuda_architectures={cuda}\nrocm_architectures={rocm}\ncondition={}\nsample={sample}\n",
        condition.name()
    ))
}

fn mark(directory: &Path, phase: &str, now: Instant) -> Result<String, String> {
    if !PHASES.contains(&phase) {
        return Err(format!("unsupported measurement phase {phase:?}"));
    }
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(format!("phase-{phase}.txt")))
        .map_err(|error| format!("phase {phase} was already recorded or is unwritable: {error}"))?;
    writeln!(file, "{}", isoformat(now)).map_err(|error| error.to_string())?;
    Ok(format!("{phase}={}\n", isoformat(now)))
}

fn snapshot(build_dir: &Path) -> Result<Tree, String> {
    let mut entries = Vec::new();
    collect(build_dir, build_dir, &mut entries)?;
    entries.sort();
    let mut hash = Sha256::new();
    for entry in &entries {
        hash.update(entry.as_bytes());
        hash.update(b"\n");
    }
    let stamp_sha256 = match fs::read(build_dir.join(STAMP)) {
        Ok(bytes) => Some(hex::encode(Sha256::digest(bytes))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.to_string()),
    };
    Ok(Tree {
        files: entries.len(),
        digest: hex::encode(hash.finalize()),
        stamp_sha256,
    })
}

fn collect(root: &Path, directory: &Path, entries: &mut Vec<String>) -> Result<(), String> {
    let read =
        fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in read {
        let path = entry.map_err(|error| error.to_string())?.path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if metadata.is_dir() {
            collect(root, &path, entries)?;
            continue;
        }
        let modified = metadata
            .modified()
            .map_err(|error| error.to_string())?
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let relative = path
            .strip_prefix(root)
            .map_err(|error| error.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        entries.push(format!("{relative}\0{}\0{modified}", metadata.len()));
    }
    Ok(())
}

fn admit_sample(
    condition: Condition,
    key: &str,
    cache_hit: &str,
    primary_key: &str,
    tree: &Tree,
) -> Result<(), String> {
    match condition {
        Condition::Cold if !cache_hit.is_empty() || !primary_key.is_empty() => {
            Err("cold sample attempted ABI retrieval".into())
        }
        Condition::Cold if tree.files != 0 => {
            Err("cold sample requires an empty job-owned ABI build directory".into())
        }
        Condition::ExactWarm if cache_hit != "true" => Err(
            "exact-warm sample requires an exact cache hit; a miss is inconclusive, never cold"
                .into(),
        ),
        Condition::ExactWarm if primary_key != key => {
            Err("restored key differs from the computed ABI identity".into())
        }
        Condition::ExactWarm if tree.stamp_sha256.is_none() => {
            Err("restored ABI tree has no build stamp".into())
        }
        _ => Ok(()),
    }
}

fn complete_sample(condition: Condition, before: &Tree, after: &Tree) -> Result<(), String> {
    if after.stamp_sha256.is_none() {
        return Err("prepared ABI tree has no build stamp".into());
    }
    if condition == Condition::ExactWarm && before != after {
        return Err(
            "exact-warm preparation modified the restored ABI tree; a rebuild is refused".into(),
        );
    }
    Ok(())
}

fn write_json(directory: &Path, name: &str, value: &Value) -> Result<(), String> {
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(directory.join(name), bytes).map_err(|error| error.to_string())
}

fn admit(directory: &Path, env: impl Fn(&str) -> String) -> Result<String, String> {
    if env("GITHUB_EVENT_NAME") != "workflow_dispatch" {
        return Err("measurement samples are admitted only from a manual dispatch".into());
    }
    let condition = Condition::parse(&env("MEASUREMENT_CONDITION"))?;
    let epoch = segment(&env("INPUT_TOOLCHAIN_EPOCH"), "toolchain_epoch")?;
    let exported = env("MESH_LLM_LLAMA_TOOLCHAIN_EPOCH");
    if epoch != segment(&exported, "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH")? {
        return Err("toolchain_epoch must match MESH_LLM_LLAMA_TOOLCHAIN_EPOCH".into());
    }
    let (backend, architectures) = (env("INPUT_BACKEND"), env("INPUT_ARCHITECTURE_SET"));
    let input_hash = env("CACHE_INPUT_HASH");
    let (cuda, vulkan, rocm) = (
        env("INPUT_CUDA_TOOLCHAIN_VERSION"),
        env("INPUT_VULKAN_TOOLCHAIN_VERSION"),
        env("INPUT_ROCM_TOOLCHAIN_VERSION"),
    );
    let identity = identity(&Inputs {
        backend: &backend,
        architectures: &architectures,
        epoch: &epoch,
        input_hash: &input_hash,
        cuda: &cuda,
        vulkan: &vulkan,
        rocm: &rocm,
    })?;
    let tree = snapshot(Path::new(&env("BUILD_DIR")))?;
    let (cache_hit, primary_key) = (env("CACHE_HIT"), env("CACHE_PRIMARY_KEY"));
    let outcome = admit_sample(condition, &identity.key, &cache_hit, &primary_key, &tree);
    let record = json!({
        "schema_version": 1,
        "condition": condition.name(),
        "sample": env("MEASUREMENT_SAMPLE"),
        "backend": backend,
        "architecture_set": architectures,
        "key": identity.key,
        "family": identity.family,
        "input_hash": input_hash,
        "cache_hit": cache_hit,
        "restored_key": primary_key,
        "tree": tree,
        "refusal": outcome.as_ref().err(),
    });
    let name = if outcome.is_ok() {
        "admission.json"
    } else {
        "refusal.json"
    };
    write_json(directory, name, &record)?;
    outcome?;
    Ok(format!(
        "admitted {} sample for {}\n",
        condition.name(),
        identity.key
    ))
}

fn phase_times(directory: &Path) -> Result<Vec<Instant>, String> {
    PHASES
        .iter()
        .map(|phase| {
            let text = fs::read_to_string(directory.join(format!("phase-{phase}.txt")))
                .map_err(|error| format!("phase {phase} missing: {error}"))?;
            timestamp(text.trim())
                .ok()
                .flatten()
                .ok_or_else(|| format!("phase {phase} has an invalid timestamp"))
        })
        .collect()
}

fn finish(directory: &Path, env: impl Fn(&str) -> String) -> Result<String, String> {
    let admission = fs::read(directory.join("admission.json"))
        .map_err(|_| "sample was not admitted; no receipt is written")?;
    let mut admission: Value =
        serde_json::from_slice(&admission).map_err(|error| error.to_string())?;
    let condition = Condition::parse(admission["condition"].as_str().unwrap_or(""))?;
    let before: Tree =
        serde_json::from_value(admission["tree"].take()).map_err(|error| error.to_string())?;
    let after = snapshot(Path::new(&env("BUILD_DIR")))?;
    let times = phase_times(directory)?;
    let seconds = |start: usize, end: usize| elapsed(Some(times[start]), Some(times[end]));
    let outcome = complete_sample(condition, &before, &after);
    let receipt = json!({
        "schema_version": 1,
        "admitted": outcome.is_ok(),
        "refusal": outcome.as_ref().err(),
        "admission": admission,
        "tree_before": before,
        "tree_after": after,
        "phases": PHASES.iter().zip(&times).map(|(phase, time)| (phase.to_string(), Value::from(isoformat(*time)))).collect::<serde_json::Map<_, _>>(),
        "seconds": {"lookup": seconds(0, 1), "prepare": seconds(2, 3), "primary": seconds(0, 3)},
        "run": {
            "repository": env("GITHUB_REPOSITORY"),
            "event": env("GITHUB_EVENT_NAME"),
            "ref": env("GITHUB_REF"),
            "source_sha": env("GITHUB_SHA"),
            "workflow_sha": env("GITHUB_WORKFLOW_SHA"),
            "run_id": env("GITHUB_RUN_ID"),
            "run_attempt": env("GITHUB_RUN_ATTEMPT"),
        },
        "runner": {
            "environment": env("RUNNER_ENVIRONMENT"),
            "os": env("RUNNER_OS"),
            "arch": env("RUNNER_ARCH"),
            "image_os": env("ImageOS"),
            "image_version": env("ImageVersion"),
        },
    });
    write_json(directory, "receipt.json", &receipt)?;
    outcome?;
    Ok(format!("{} sample receipt written\n", condition.name()))
}

#[cfg(test)]
#[path = "windows_abi_measurement_tests.rs"]
mod tests;
