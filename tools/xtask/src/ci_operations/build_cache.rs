//! `ci-ops build-cache`: the Rust owner of `scripts/manage-build-cache.py`.
//! Measures and safely prunes a repository-local Cargo target directory.
//! The target must be a strict child of the workspace; pruning removes only
//! the incremental directories and Cargo packages it selects, refuses while
//! a build holds the shared lock or a compiler is running, and previews by
//! default. Streams and exit statuses match the legacy tool byte for byte.

use crate::ci_operations::build_cache_argv::{Args, Mode, Parsed, parse};
use crate::ci_operations::build_cache_cargo::{
    Failure, LockMode, active_compilers, cache_lock, reject_separate_build_directory,
};
use crate::ci_operations::build_cache_prune::{
    Budget, float, int, prune_incremental, prune_packages,
};
use crate::ci_operations::build_cache_tree::{immediate_entries, io_text, tree_metrics};
use crate::ci_operations::build_cache_values::{human_size, resolve};
use crate::ci_operations::python_access::{object, string};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_json::dumps_indented;
use crate::repository::check_report::CheckReport;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn run(args: &[String]) -> CheckReport {
    let args = match parse(args) {
        Parsed::Run(args) => args,
        Parsed::Report(report) => return report,
    };
    execute(&args).unwrap_or_else(|message| {
        CheckReport::failure(String::new(), format!("ERROR: {message}\n"))
    })
}

fn locations(args: &Args) -> Result<(PathBuf, PathBuf), Failure> {
    let workspace = match &args.workspace {
        Some(text) => resolve(Path::new(text)),
        None => std::env::current_dir().map_err(|error| io_text(&error, Path::new(".")))?,
    };
    let target = args.target_dir.as_ref().map_or_else(
        || resolve(&workspace.join("target")),
        |text| resolve(Path::new(text)),
    );
    if target == workspace || !target.starts_with(&workspace) {
        return Err("target directory must be a child of the workspace".to_owned());
    }
    Ok((workspace, target))
}

fn execute(args: &Args) -> Result<CheckReport, Failure> {
    let (workspace, target) = locations(args)?;
    reject_separate_build_directory(&workspace, &target)?;
    if args.mode == Mode::Build {
        return build(args, &workspace, &target);
    }
    if args.max_age.0 < 0 {
        return Err("max age must be non-negative".to_owned());
    }
    if args.mode == Mode::Status {
        let _lock = cache_lock(&target, LockMode::SharedNow)?;
        let report = snapshot(&workspace, &target, args);
        let text = if args.json {
            dumps_indented(&report) + "\n"
        } else {
            render_status(&report, args)
        };
        return Ok(CheckReport::success(text));
    }
    if args.execute {
        let _lock = cache_lock(&target, LockMode::ExclusiveNow)?;
        if !active_compilers()?.is_empty() {
            return Err(
                "active Cargo/Rust compiler processes detected; refusing cleanup".to_owned(),
            );
        }
        return run_prune(args, &workspace, &target);
    }
    let _lock = cache_lock(&target, LockMode::SharedNow)?;
    run_prune(args, &workspace, &target)
}

/// `build`: run the command under the shared lock so pruning waits.
fn build(args: &Args, workspace: &Path, target: &Path) -> Result<CheckReport, Failure> {
    let command = match args.build_command.split_first() {
        Some((first, rest)) if first == "--" => rest,
        _ => &args.build_command[..],
    };
    let Some((program, arguments)) = command.split_first() else {
        return Err("build command is required".to_owned());
    };
    let _lock = cache_lock(target, LockMode::SharedWait)?;
    let status = Command::new(program)
        .args(arguments)
        .current_dir(workspace)
        .status()
        .map_err(|error| io_text(&error, Path::new(program)))?;
    let code = status.code().unwrap_or_else(|| {
        use std::os::unix::process::ExitStatusExt;
        -status.signal().unwrap_or(0)
    });
    Ok(CheckReport {
        stdout: String::new(),
        stderr: String::new(),
        code,
    })
}

fn number(report: &Json, key: &str) -> i128 {
    report.get(key).and_then(Json::as_int).unwrap_or_default()
}

fn snapshot(workspace: &Path, target: &Path, args: &Args) -> Json {
    let (total, newest) = tree_metrics(target);
    let entries = immediate_entries(target)
        .into_iter()
        .map(|entry| {
            object(&[
                ("path", string(&entry.path.to_string_lossy())),
                ("bytes", int(entry.bytes)),
                ("newest_mtime", float(entry.newest)),
            ])
        })
        .collect();
    object(&[
        ("schema", string("mesh-llm.local-build-cache")),
        ("schema_version", int(1)),
        ("workspace", string(&workspace.to_string_lossy())),
        ("target", string(&target.to_string_lossy())),
        ("target_bytes", int(total)),
        ("target_limit_bytes", int(args.max_size)),
        (
            "target_over_limit_bytes",
            int((total - args.max_size).max(0)),
        ),
        ("max_age_days", int(args.max_age.0)),
        ("newest_mtime", float(newest)),
        ("entries", Json::Array(entries)),
    ])
}

fn render_status(report: &Json, args: &Args) -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "Cargo target: {}",
        human_size(number(report, "target_bytes"))
    );
    let _ = writeln!(text, "Configured limit: {}", human_size(args.max_size));
    let _ = writeln!(text, "Configured maximum age: {} days", args.max_age.0);
    let over = number(report, "target_over_limit_bytes");
    if over != 0 {
        let _ = writeln!(text, "Over limit: {}", human_size(over));
    }
    text.push_str("Largest target entries:\n");
    let entries = report
        .get("entries")
        .and_then(Json::as_array)
        .unwrap_or(&[]);
    for entry in entries.iter().take(10) {
        let path = entry.get("path").and_then(Json::as_str).unwrap_or_default();
        let _ = writeln!(text, "  {:>10}  {path}", human_size(number(entry, "bytes")));
    }
    text
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

fn run_prune(args: &Args, workspace: &Path, target: &Path) -> Result<CheckReport, Failure> {
    let before = number(&snapshot(workspace, target, args), "target_bytes");
    let mut budget = Budget {
        current: before,
        max_bytes: args.max_size,
        cutoff: now() - (args.max_age.0.saturating_mul(86400)) as f64,
        execute: args.execute,
    };
    let mut actions = prune_incremental(target, &mut budget)?;
    actions.extend(prune_packages(workspace, target, &mut budget)?);
    let after = if args.execute {
        tree_metrics(target).0
    } else {
        budget.current
    };
    let mode = if args.execute { "execute" } else { "dry-run" };
    if args.json {
        let report = object(&[
            ("schema", string("mesh-llm.local-build-cache-prune")),
            ("schema_version", int(1)),
            ("mode", string(mode)),
            ("before_bytes", int(before)),
            ("after_bytes", int(after)),
            ("reclaimed_bytes", int(before - after)),
            ("actions", Json::Array(actions)),
        ]);
        return Ok(CheckReport::success(dumps_indented(&report) + "\n"));
    }
    let mut text = format!(
        "Mode: {mode}\nBefore: {}\nAfter: {}\nReclaimed: {}\n",
        human_size(before),
        human_size(after),
        human_size(before - after)
    );
    for action in &actions {
        let kind = action
            .get("kind")
            .and_then(Json::as_str)
            .unwrap_or_default();
        let identity = action
            .get("package")
            .or_else(|| action.get("path"))
            .and_then(Json::as_str)
            .unwrap_or_default();
        let size = action
            .get("estimated_bytes")
            .or_else(|| action.get("bytes"))
            .and_then(Json::as_int)
            .unwrap_or_default();
        let _ = writeln!(text, "  {kind}: {identity} ({})", human_size(size));
    }
    Ok(CheckReport::success(text))
}
