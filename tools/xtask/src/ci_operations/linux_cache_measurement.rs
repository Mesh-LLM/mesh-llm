//! Manual Linux CPU runtime and hosted automation cache measurement receipts.
//!
//! The diagnostic restores existing exact production keys and never saves,
//! prunes or publishes a cache. A warm arm without an exact trusted entry is
//! BLOCKED, never relabelled cold, and an unrelated edit must leave the key
//! byte-identical before restoration.

use super::cache_inputs::{Inventory, PRESERVED_EDITS};
use super::ci_metrics_time::{Instant, elapsed, isoformat, timestamp};
use crate::repository::check_report::CheckReport;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write as _, path::Path};

const PHASES: [&str; 4] = ["lookup_start", "lookup_end", "prepare_start", "prepare_end"];
const EDIT_MARKER: &str = "\n// linux-cache-measurement unrelated edit\n";
const USAGE: &str = "usage: ci-ops linux-cache-measurement {key --receipt-dir D --cohort C --phase before|measured | edit --receipt-dir D --cohort C | mark --receipt-dir D --phase P | admit --receipt-dir D --cohort C | finish --receipt-dir D --cohort C}";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cohort {
    CpuRuntime,
    HostedXtask,
    HostedController,
}

impl Cohort {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu-runtime" => Ok(Self::CpuRuntime),
            "hosted-xtask" => Ok(Self::HostedXtask),
            "hosted-controller" => Ok(Self::HostedController),
            _ => Err(format!("unsupported measurement cohort {value:?}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::CpuRuntime => "cpu-runtime",
            Self::HostedXtask => "hosted-xtask",
            Self::HostedController => "hosted-controller",
        }
    }

    fn inventory(self) -> Inventory {
        match self {
            Self::CpuRuntime => Inventory::LinuxRuntime,
            Self::HostedXtask => Inventory::HostedXtask,
            Self::HostedController => Inventory::HostedController,
        }
    }

    fn output(self) -> &'static str {
        match self {
            Self::CpuRuntime => "runtime-input",
            Self::HostedXtask => "target/release/xtask",
            Self::HostedController => "target/release/ci-controller",
        }
    }

    fn unrelated_edit(self) -> Result<&'static str, String> {
        PRESERVED_EDITS
            .iter()
            .find(|(inventory, _)| *inventory == self.inventory())
            .map(|(_, path)| *path)
            .filter(|path| path.ends_with(".rs"))
            .ok_or_else(|| format!("{} has no admitted Rust-comment edit", self.name()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Condition {
    Cold,
    ExactWarm,
    UnrelatedEdit,
}

impl Condition {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cold" => Ok(Self::Cold),
            "exact-warm" => Ok(Self::ExactWarm),
            "unrelated-edit" => Ok(Self::UnrelatedEdit),
            _ => Err(format!("unsupported measurement condition {value:?}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::ExactWarm => "exact-warm",
            Self::UnrelatedEdit => "unrelated-edit",
        }
    }
}

/// Paths, sizes and mtimes of the measured output; mtimes expose a rebuild.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Tree {
    files: usize,
    digest: String,
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Admitted,
    Blocked(String),
    Refused(String),
}

struct Observation<'a> {
    key_before: &'a str,
    key: &'a str,
    edited: bool,
    cache_hit: &'a str,
    restored_key: &'a str,
    output: &'a Tree,
    stale_roots: &'a [String],
}

pub(super) fn run(args: &[String], root: &Path) -> CheckReport {
    let env = |name: &str| std::env::var(name).unwrap_or_default();
    let result = match args.split_first() {
        Some((command, rest)) => dispatch(command, rest, root, env),
        None => Err(USAGE.into()),
    };
    match result {
        Ok(output) => CheckReport::success(output),
        Err(error) => {
            CheckReport::failure(String::new(), format!("linux-cache-measurement: {error}\n"))
        }
    }
}

fn dispatch(
    command: &str,
    rest: &[String],
    root: &Path,
    env: impl Fn(&str) -> String,
) -> Result<String, String> {
    let cohort_options = ["--receipt-dir", "--cohort"];
    match command {
        "key" => options(rest, &["--receipt-dir", "--cohort", "--phase"]).and_then(|values| {
            key(
                Path::new(values[0]),
                Cohort::parse(values[1])?,
                values[2],
                &env("CACHE_KEY"),
            )
        }),
        "edit" => options(rest, &cohort_options)
            .and_then(|values| edit(root, Path::new(values[0]), Cohort::parse(values[1])?)),
        "mark" => options(rest, &["--receipt-dir", "--phase"])
            .and_then(|values| mark(Path::new(values[0]), values[1], Instant::now())),
        "admit" => options(rest, &cohort_options)
            .and_then(|values| admit(root, Path::new(values[0]), Cohort::parse(values[1])?, env)),
        "finish" => options(rest, &cohort_options)
            .and_then(|values| finish(root, Path::new(values[0]), Cohort::parse(values[1])?, env)),
        _ => Err(USAGE.into()),
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

fn create_new(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("{} was already recorded: {error}", path.display()))?;
    file.write_all(text.as_bytes())
        .map_err(|error| error.to_string())
}

/// The workflow binds the inventory expression; this refuses an unexpanded identity.
fn key(directory: &Path, cohort: Cohort, phase: &str, value: &str) -> Result<String, String> {
    if !matches!(phase, "before" | "measured") {
        return Err(format!("unsupported key phase {phase:?}"));
    }
    let family = format!("{}-", cohort.inventory().family());
    let complete = value.starts_with(&family)
        && !value.ends_with('-')
        && !value.contains("${{")
        && !value.contains("--")
        && !value.chars().any(char::is_whitespace);
    if !complete {
        return Err(format!(
            "{} key {value:?} is not a complete {family} identity",
            cohort.name()
        ));
    }
    create_new(&directory.join(format!("key-{phase}.txt")), value)?;
    Ok(format!("key={value}\n"))
}

fn edit(root: &Path, directory: &Path, cohort: Cohort) -> Result<String, String> {
    let path = cohort.unrelated_edit()?;
    create_new(&directory.join("edit.txt"), path)?;
    let target = root.join(path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&target)
        .and_then(|mut file| file.write_all(EDIT_MARKER.as_bytes()))
        .map_err(|error| format!("{path}: {error}"))?;
    Ok(format!("path={path}\n"))
}

fn mark(directory: &Path, phase: &str, now: Instant) -> Result<String, String> {
    if !PHASES.contains(&phase) {
        return Err(format!("unsupported measurement phase {phase:?}"));
    }
    create_new(
        &directory.join(format!("phase-{phase}.txt")),
        &format!("{}\n", isoformat(now)),
    )?;
    Ok(format!("{phase}={}\n", isoformat(now)))
}

fn snapshot(path: &Path) -> Result<Tree, String> {
    let mut entries = Vec::new();
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("{}: {error}", path.display())),
        Ok(metadata) if metadata.is_dir() => collect(path, path, &mut entries)?,
        Ok(metadata) => entries.push(entry(".", &metadata)?),
    }
    entries.sort();
    let mut hash = Sha256::new();
    for line in &entries {
        hash.update(line.as_bytes());
        hash.update(b"\n");
    }
    Ok(Tree {
        files: entries.len(),
        digest: hex::encode(hash.finalize()),
    })
}

fn entry(relative: &str, metadata: &fs::Metadata) -> Result<String, String> {
    let modified = metadata
        .modified()
        .map_err(|error| error.to_string())?
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    Ok(format!("{relative}\0{}\0{modified}", metadata.len()))
}

fn collect(root: &Path, directory: &Path, entries: &mut Vec<String>) -> Result<(), String> {
    let read =
        fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    for item in read {
        let path = item.map_err(|error| error.to_string())?.path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if metadata.is_dir() {
            collect(root, &path, entries)?;
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .map_err(|error| error.to_string())?
            .to_string_lossy()
            .into_owned();
        entries.push(entry(&relative, &metadata)?);
    }
    Ok(())
}

fn decide(condition: Condition, seen: &Observation) -> Verdict {
    if seen.key_before != seen.key {
        return Verdict::Refused(match condition {
            Condition::UnrelatedEdit => "the admitted unrelated edit moved the cache key".into(),
            _ => "the cache key changed between computations".into(),
        });
    }
    if seen.edited != (condition == Condition::UnrelatedEdit) {
        return Verdict::Refused("only the unrelated-edit arm applies the admitted edit".into());
    }
    if let Some(root) = seen.stale_roots.first() {
        return Verdict::Refused(format!("arm requires a fresh root; {root} already exists"));
    }
    match condition {
        Condition::Cold if !seen.cache_hit.is_empty() || !seen.restored_key.is_empty() => {
            Verdict::Refused("cold arm attempted retrieval".into())
        }
        Condition::Cold if seen.output.files != 0 => {
            Verdict::Refused("cold arm requires an absent output".into())
        }
        Condition::Cold => Verdict::Admitted,
        _ if seen.cache_hit != "true" => Verdict::Blocked(format!(
            "no exact trusted entry for {}; the cell is not manufactured",
            seen.key
        )),
        _ if seen.restored_key != seen.key => {
            Verdict::Refused("restored key differs from the computed identity".into())
        }
        _ if seen.output.files == 0 => Verdict::Refused("exact hit restored no output".into()),
        _ => Verdict::Admitted,
    }
}

fn read_text(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
}

/// Roots that must be absent in every arm: the native build tree (a warm
/// package must not rebuild it) and, for cold arms, the measured output.
fn stale_roots(
    root: &Path,
    cohort: Cohort,
    condition: Condition,
    env: &impl Fn(&str) -> String,
) -> Vec<String> {
    let mut roots = Vec::new();
    if cohort == Cohort::CpuRuntime {
        roots.push(env("LLAMA_STAGE_BUILD_DIR"));
    }
    if condition == Condition::Cold {
        roots.push(cohort.output().to_owned());
    }
    roots
        .into_iter()
        .filter(|path| path.is_empty() || root.join(path).exists())
        .map(|path| {
            if path.is_empty() {
                "an unset native build directory".into()
            } else {
                path
            }
        })
        .collect()
}

fn admit(
    root: &Path,
    directory: &Path,
    cohort: Cohort,
    env: impl Fn(&str) -> String,
) -> Result<String, String> {
    if env("GITHUB_EVENT_NAME") != "workflow_dispatch" {
        return Err("measurement samples are admitted only from a manual dispatch".into());
    }
    let condition = Condition::parse(&env("MEASUREMENT_CONDITION"))?;
    let key_before = read_text(&directory.join("key-before.txt"))?;
    let key = read_text(&directory.join("key-measured.txt"))?;
    let edit = fs::read_to_string(directory.join("edit.txt")).ok();
    let output = snapshot(&root.join(cohort.output()))?;
    let (cache_hit, restored_key) = (env("CACHE_HIT"), env("CACHE_PRIMARY_KEY"));
    let stale = stale_roots(root, cohort, condition, &env);
    let verdict = decide(
        condition,
        &Observation {
            key_before: &key_before,
            key: &key,
            edited: edit.is_some(),
            cache_hit: &cache_hit,
            restored_key: &restored_key,
            output: &output,
            stale_roots: &stale,
        },
    );
    let (name, outcome, reason) = match &verdict {
        Verdict::Admitted => ("admission.json", "admitted", None),
        Verdict::Blocked(reason) => ("blocked.json", "blocked", Some(reason)),
        Verdict::Refused(reason) => ("refusal.json", "refused", Some(reason)),
    };
    let record = json!({
        "schema_version": 1,
        "outcome": outcome,
        "reason": reason,
        "cohort": cohort.name(),
        "condition": condition.name(),
        "sample": env("MEASUREMENT_SAMPLE"),
        "family": cohort.inventory().family(),
        "key_before": key_before,
        "key": key,
        "unrelated_edit": edit,
        "cache_hit": cache_hit,
        "restored_key": restored_key,
        "output_path": cohort.output(),
        "output": output,
    });
    write_json(directory, name, &record)?;
    match verdict {
        Verdict::Admitted => Ok(format!(
            "admitted {} {} sample\n",
            cohort.name(),
            condition.name()
        )),
        Verdict::Blocked(reason) => Err(format!("BLOCKED: {reason}")),
        Verdict::Refused(reason) => Err(reason),
    }
}

fn write_json(directory: &Path, name: &str, value: &Value) -> Result<(), String> {
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(directory.join(name), bytes).map_err(|error| error.to_string())
}

/// Recorded instants in `PHASES` order; a missing or unparsable mark is `None`.
fn phase_times(directory: &Path) -> Vec<Option<Instant>> {
    PHASES
        .iter()
        .map(|phase| {
            fs::read_to_string(directory.join(format!("phase-{phase}.txt")))
                .ok()
                .and_then(|text| timestamp(text.trim()).ok().flatten())
        })
        .collect()
}

fn complete(
    condition: Condition,
    prepare_outcome: &str,
    times: &[Option<Instant>],
    before: &Tree,
    after: &Tree,
) -> Result<(), String> {
    if prepare_outcome != "success" {
        return Err(format!(
            "workload preparation did not succeed (outcome {prepare_outcome:?})"
        ));
    }
    if let Some((phase, _)) = PHASES.iter().zip(times).find(|(_, time)| time.is_none()) {
        return Err(format!("phase {phase} is missing or invalid"));
    }
    if times.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err("phase timestamps are out of order".into());
    }
    if after.files == 0 {
        return Err("the workload produced no measured output".into());
    }
    if condition != Condition::Cold && before != after {
        return Err("the workload modified the restored output; a rebuild is refused".into());
    }
    Ok(())
}

fn finish(
    root: &Path,
    directory: &Path,
    cohort: Cohort,
    env: impl Fn(&str) -> String,
) -> Result<String, String> {
    let admission = fs::read(directory.join("admission.json"))
        .map_err(|_| "sample was not admitted; no receipt is written")?;
    let mut admission: Value =
        serde_json::from_slice(&admission).map_err(|error| error.to_string())?;
    if admission["cohort"] != cohort.name() {
        return Err("admission belongs to a different cohort".into());
    }
    let condition = Condition::parse(admission["condition"].as_str().unwrap_or(""))?;
    let before: Tree =
        serde_json::from_value(admission["output"].take()).map_err(|error| error.to_string())?;
    // A failed workload can leave an unreadable output; that is refusal evidence.
    let after = snapshot(&root.join(cohort.output()));
    let times = phase_times(directory);
    let seconds = |start: usize, end: usize| elapsed(times[start], times[end]);
    let prepare_outcome = env("PREPARE_OUTCOME");
    let outcome = after
        .as_ref()
        .map_err(|error| format!("final output snapshot failed: {error}"))
        .and_then(|after| complete(condition, &prepare_outcome, &times, &before, after));
    let receipt = json!({
        "schema_version": 1,
        "admitted": outcome.is_ok(),
        "refusal": outcome.as_ref().err(),
        "admission": admission,
        "output_before": before,
        "output_after": after.as_ref().ok(),
        "prepare_outcome": prepare_outcome,
        "phases": PHASES.iter().zip(&times).map(|(phase, time)| ((*phase).to_owned(), time.map_or(Value::Null, |time| Value::from(isoformat(time))))).collect::<serde_json::Map<_, _>>(),
        "seconds": {"lookup": seconds(0, 1), "prepare": seconds(2, 3), "primary": seconds(0, 3)},
        "run": run_identity(&env),
    });
    write_json(directory, "receipt.json", &receipt)?;
    outcome?;
    Ok(format!(
        "{} {} receipt written\n",
        cohort.name(),
        condition.name()
    ))
}

fn run_identity(env: &impl Fn(&str) -> String) -> Value {
    let fields = [
        "GITHUB_REPOSITORY",
        "GITHUB_EVENT_NAME",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_WORKFLOW_SHA",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_JOB",
        "RUNNER_ENVIRONMENT",
        "RUNNER_OS",
        "RUNNER_ARCH",
        "ImageOS",
        "ImageVersion",
        "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
    ];
    Value::Object(
        fields
            .iter()
            .map(|name| ((*name).to_owned(), Value::from(env(name))))
            .collect(),
    )
}

#[cfg(test)]
#[path = "linux_cache_measurement_tests.rs"]
mod tests;
