//! `automation parity --suite ci`: the transitional shadow comparison of
//! the legacy CI planner with `ci plan`. It is the only place xtask may run
//! the legacy planner, and task 28 deletes it with that planner. No required
//! caller is switched here and no digest field is added.
//!
//! Groups: `frozen-cases` (every fixture case against its golden, and the
//! legacy planner against Rust), `action-outputs` (every `$GITHUB_OUTPUT`
//! line of the protected action, including digest and lane projections),
//! `action-source` (the pinned action block those goldens were cut from),
//! `manifest-root` (inert PR catalogs against the protected default),
//! `protected-profiles` (real catalogs and Cargo), `command-surface` and
//! `runner-image-identity` (the planner rows its hidden import consumes).

mod action;
mod frozen;
mod identity;
mod jq_render;
mod legacy;
mod manifest_root;
mod options;
mod process;
mod projection;
mod protected;
mod report;
mod stage;

use crate::command::DynResult;
use legacy::Legacy;
use options::{Mode, Options};
use process::{Captured, process_difference, run_bounded};
use report::{Ledger, Outcome};
use serde_json::json;
use stage::Stage;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Everything one suite run shares.
struct Run<'a> {
    root: &'a Path,
    stage: &'a Stage,
    legacy: Option<Legacy>,
    ledger: Ledger,
    evidence: Option<PathBuf>,
}

impl Run<'_> {
    /// `xtask ci plan <args> < stdin` from the protected checkout.
    fn rust_plan(&self, args: &[&str], search_path: &str, stdin: &[u8]) -> DynResult<Captured> {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .current_dir(self.root)
            .args(["ci", "plan"])
            .args(args)
            .env("PATH", search_path);
        run_bounded(&mut command, stdin)
    }

    /// Runs the legacy planner (when enabled) on the same inputs and records
    /// the process comparison with `rust`.
    fn compare_legacy(
        &mut self,
        group: &'static str,
        label: &str,
        rust: &Captured,
        call: (&[&str], &str, &[u8]),
    ) -> DynResult<Option<Captured>> {
        let (args, search_path, stdin) = call;
        let Some(legacy) = &self.legacy else {
            return Ok(None);
        };
        let captured = legacy.plan(args, search_path, stdin)?;
        let difference = process_difference(&captured, rust);
        self.ledger
            .record(group, format!("{label} legacy-vs-rust"), difference);
        Ok(Some(captured))
    }

    /// Writes one evidence file when an evidence directory was requested.
    fn keep(&self, name: &str, bytes: &[u8]) -> DynResult<()> {
        if let Some(directory) = &self.evidence {
            let path = directory.join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, bytes)?;
        }
        Ok(())
    }
}

fn head(root: &Path) -> DynResult<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["rev-parse", "HEAD"])
        .output()?;
    if !output.status.success() {
        return Err("unable to resolve HEAD as the source revision; pass --source-sha".into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn legacy_side(root: &Path, options: &Options) -> DynResult<Option<Legacy>> {
    let planner = legacy::protected_planner(root, options.planner.as_deref())?;
    match &options.mode {
        Mode::RustOnly => Ok(None),
        Mode::Legacy { interpreter } => {
            if !interpreter.is_file() {
                return Err(format!(
                    "legacy interpreter {} does not exist",
                    interpreter.display()
                )
                .into());
            }
            let root = planner
                .parent()
                .and_then(Path::parent)
                .ok_or("planner has no checkout")?
                .to_path_buf();
            Ok(Some(Legacy {
                interpreter: interpreter.clone(),
                root,
            }))
        }
    }
}

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    let options = Options::parse(args)?;
    let legacy = legacy_side(root, &options)?;
    let fixtures = options
        .fixtures
        .clone()
        .unwrap_or_else(|| root.join("tools/xtask/tests/fixtures/ci_plan"));
    let source_repo = options
        .source_repo
        .clone()
        .unwrap_or_else(|| root.to_path_buf());
    let source_sha = match &options.source_sha {
        Some(sha) => sha.clone(),
        None => head(&source_repo)?,
    };
    let stage = Stage::new(&fixtures, options.bash.as_deref())?;
    let inert = stage.root().join("pr-manifests");
    let source = manifest_root::Source {
        repo: &source_repo,
        sha: &source_sha,
    };
    let inert =
        manifest_root::materialize(root, &source, &inert)?.map_err(|rejected| rejected.0)?;
    if let Some(evidence) = &options.evidence {
        std::fs::create_dir_all(evidence)?;
    }
    let mut run = Run {
        root,
        stage: &stage,
        legacy,
        ledger: Ledger::default(),
        evidence: options.evidence.clone(),
    };
    frozen::compare(&mut run)?;
    action::action_source(&mut run)?;
    protected::manifest_root(&mut run, &inert, &source_sha)?;
    protected::profiles(&mut run, &source_sha)?;
    identity::command_surface(&mut run)?;
    identity::runner_image(&mut run)?;
    let context = json!({
        "legacy": {
            "enabled": run.legacy.is_some(),
            "interpreter": run.legacy.as_ref().map(|side| side.interpreter.display().to_string()),
            "bash": options.bash.as_ref().map(|bash| bash.display().to_string()),
        },
        "source": {"repo": source_repo.display().to_string(), "sha": source_sha},
        "manifest_root_entries": manifest_root::entries(&inert)?,
    });
    let summary = run.ledger.summary(context);
    if let Some(evidence) = &run.evidence {
        report::write_summary(evidence, &summary)?;
    }
    for line in run.ledger.lines() {
        println!("{line}");
    }
    let unexplained = run.ledger.unexplained();
    if unexplained > 0 {
        return Err(format!("ci parity: {unexplained} unexplained differences").into());
    }
    let explained = run
        .ledger
        .rows
        .iter()
        .filter(|row| row.outcome == Outcome::Explained)
        .count();
    println!("ci parity: 0 unexplained differences ({explained} explained)");
    Ok(())
}
