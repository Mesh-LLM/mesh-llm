//! Shared local/CI repair checks; focused success never authorizes publication.
mod identity;
pub(crate) mod local;
mod plan;
mod retry;
mod runner;
mod symbols;
#[cfg(test)]
mod tests;

use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation canary-receipts repair-check --root ABS --controller-root ABS --state-dir ABS --build-dir ABS --backend metal|cuda|cpu --phase environment|fast|full [--seconds N] [--plan]",
    values: &[
        "--root",
        "--controller-root",
        "--state-dir",
        "--build-dir",
        "--backend",
        "--phase",
        "--seconds",
    ],
    flags: &["--help", "--plan"],
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Environment,
    Fast,
    Full,
}

struct Input {
    root: PathBuf,
    controller: PathBuf,
    state: PathBuf,
    build: PathBuf,
    backend: String,
    phase: Phase,
    seconds: u64,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let path = |key| -> DynResult<PathBuf> {
        let value = PathBuf::from(parsed.last(key).ok_or_else(|| format!("missing {key}"))?);
        if !value.is_absolute() {
            return Err(format!("{key} must be absolute").into());
        }
        Ok(value)
    };
    let input = Input {
        root: path("--root")?.canonicalize()?,
        controller: path("--controller-root")?.canonicalize()?,
        state: resolved_output(&path("--state-dir")?)?,
        build: resolved_output(&path("--build-dir")?)?,
        backend: match parsed.last("--backend") {
            Some(value @ ("metal" | "cuda" | "cpu")) => value.into(),
            _ => return Err("unsupported repair backend".into()),
        },
        phase: match parsed.last("--phase") {
            Some("environment") => Phase::Environment,
            Some("fast") => Phase::Fast,
            Some("full") => Phase::Full,
            _ => return Err("expected environment, fast or full phase".into()),
        },
        seconds: parsed.last("--seconds").unwrap_or("43200").parse()?,
    };
    if input.seconds == 0
        || input.seconds > 43200
        || input.state.starts_with(&input.build)
        || input.build.starts_with(&input.state)
        || !input.root.is_dir()
        || !input.controller.is_dir()
    {
        return Err("invalid repair deadline or overlapping output directories".into());
    }
    for output in [&input.state, &input.build] {
        if input.root.starts_with(output)
            || input.controller.starts_with(output)
            || output.starts_with(&input.root)
            || output.starts_with(&input.controller)
        {
            return Err("repair outputs must be outside both checkouts".into());
        }
    }
    let steps = plan::steps(&input);
    if parsed.flag("--plan") {
        return CheckReport::success(format!("{}\n", serde_json::to_string_pretty(&steps)?)).emit();
    }
    if !cfg!(any(target_os = "macos", target_os = "linux"))
        || (cfg!(target_os = "macos")
            && (input.backend == "cuda" || !cfg!(target_arch = "aarch64")))
        || (cfg!(target_os = "linux") && input.backend == "metal")
    {
        return Err(
            "repair checks require the matching native macOS arm64 or Linux producer backend"
                .into(),
        );
    }
    let report = fs::create_dir_all(&input.state)
        .and_then(|()| fs::create_dir_all(input.state.join("checks")))
        .map_err(Into::into)
        .and_then(|()| runner::execute(&input, &steps));
    match report {
        Ok(report) => report.emit(),
        Err(error) => CheckReport {
            stdout: String::new(),
            stderr: format!("repair infrastructure failure: {error}\n"),
            code: 125,
        }
        .emit(),
    }
}

fn resolved_output(path: &Path) -> DynResult<PathBuf> {
    let mut ancestor = path;
    let mut missing = Vec::new();
    loop {
        if ancestor.symlink_metadata().is_ok() {
            let mut resolved = ancestor.canonicalize()?;
            for name in missing.into_iter().rev() {
                resolved.push(name);
            }
            return Ok(resolved);
        }
        let name = ancestor
            .file_name()
            .ok_or("output path has no existing ancestor")?;
        missing.push(name.to_os_string());
        ancestor = ancestor.parent().ok_or("output path has no parent")?;
    }
}

fn atomic_json(path: &Path, value: &impl Serialize) -> DynResult<()> {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("output needs parent")?)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist(path)?;
    Ok(())
}
