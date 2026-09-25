//! The external surfaces of `ci-ops build-cache`: Cargo metadata through
//! `just cache-cargo-metadata`, package cleaning through
//! `just cache-cargo-clean`, the `ps` compiler census, and the advisory
//! `flock` on `<target>/.mesh-llm-cache-prune.lock`.

use crate::ci_operations::build_cache_tree::io_text;
use crate::ci_operations::build_cache_values::resolve;
use crate::ci_operations::python_json_decode::{DecodeError, Hooks, loads};
use crate::ci_plan::document::Json;
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub(crate) type Failure = String;

const LOCK_NAME: &str = ".mesh-llm-cache-prune.lock";
const COMPILERS: [&str; 4] = ["cargo", "rustc", "rustdoc", "clippy-driver"];

fn keep(pairs: Vec<(String, Json)>) -> Result<Json, String> {
    Ok(Json::Object(pairs))
}

fn constant(_: &str) -> Result<Json, String> {
    Err("unsupported JSON constant in cargo metadata".to_owned())
}

fn program_error(error: &std::io::Error, program: &str) -> Failure {
    io_text(error, Path::new(program))
}

pub(crate) fn cargo_metadata(workspace: &Path) -> Result<Json, Failure> {
    let output = Command::new("just")
        .arg("cache-cargo-metadata")
        .current_dir(workspace)
        .stdin(Stdio::inherit())
        .output()
        .map_err(|error| program_error(&error, "just"))?;
    if !output.status.success() {
        return Err("cargo metadata failed; refusing build-cache management".to_owned());
    }
    let hooks = Hooks {
        pairs: keep,
        constant,
    };
    loads(&output.stdout, &hooks).map_err(|error| match error {
        DecodeError::Value(text) => text,
        DecodeError::Recursion => "maximum recursion depth exceeded".to_owned(),
    })
}

fn key_error(key: &str) -> Failure {
    crate::repository::python_text::repr(key)
}

/// `cargo_packages`: the sorted, de-duplicated workspace package names.
pub(crate) fn cargo_packages(workspace: &Path) -> Result<Vec<String>, Failure> {
    let metadata = cargo_metadata(workspace)?;
    let packages = metadata
        .get_present("packages")
        .and_then(Json::as_array)
        .ok_or_else(|| key_error("packages"))?;
    let names = packages
        .iter()
        .map(|package| {
            package
                .get("name")
                .and_then(Json::as_str)
                .map(str::to_owned)
                .ok_or_else(|| key_error("name"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    Ok(names.into_iter().collect())
}

/// `reject_separate_build_directory`.
pub(crate) fn reject_separate_build_directory(
    workspace: &Path,
    managed_target: &Path,
) -> Result<(), Failure> {
    if std::env::var_os("CARGO_BUILD_BUILD_DIR").is_some_and(|value| !value.is_empty()) {
        return Err("CARGO_BUILD_BUILD_DIR is unsupported by build-cache management".to_owned());
    }
    if !workspace.join("Cargo.toml").is_file() {
        return Ok(());
    }
    let metadata = cargo_metadata(workspace)?;
    let target_directory = metadata
        .get_present("target_directory")
        .and_then(Json::as_str)
        .map(|text| resolve(Path::new(text)))
        .ok_or_else(|| key_error("target_directory"))?;
    let build_directory = metadata
        .get("build_directory")
        .and_then(Json::as_str)
        .filter(|text| !text.is_empty())
        .map_or_else(|| target_directory.clone(), |text| resolve(Path::new(text)));
    if build_directory != target_directory {
        return Err(format!(
            "Cargo build.build-dir outside target-dir is unsupported by build-cache management: {}",
            build_directory.display()
        ));
    }
    if managed_target != target_directory {
        return Err(format!(
            "managed target directory does not match Cargo's effective target directory: {}",
            target_directory.display()
        ));
    }
    Ok(())
}

/// `just cache-cargo-clean` for one package, output inherited.
pub(crate) fn clean_package(workspace: &Path, target: &Path, package: &str) -> Result<(), Failure> {
    let status = Command::new("just")
        .arg("cache-cargo-clean")
        .current_dir(workspace)
        .env("MESH_LLM_CACHE_TARGET_DIR", target)
        .env("MESH_LLM_CACHE_PACKAGE", package)
        .status()
        .map_err(|error| program_error(&error, "just"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("cargo clean failed for {package}"))
    }
}

/// `active_compilers`: `ps` rows whose command basename is a Rust tool.
pub(crate) fn active_compilers() -> Result<Vec<String>, Failure> {
    let output = Command::new("ps")
        .arg("-axo")
        .arg("pid=,comm=,args=")
        .stdin(Stdio::inherit())
        .output()
        .map_err(|error| program_error(&error, "ps"))?;
    if !output.status.success() {
        return Err(format!(
            "Command '['ps', '-axo', 'pid=,comm=,args=']' returned non-zero exit status {}.",
            output.status.code().unwrap_or(-1)
        ));
    }
    let own = std::process::id().to_string();
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let mut fields = line.split_whitespace();
            let pid = fields.next()?;
            let command = fields.next()?;
            let name = Path::new(command).file_name()?.to_str()?;
            (pid.trim_start_matches('0') != own.trim_start_matches('0')
                && COMPILERS.contains(&name))
            .then(|| line.to_owned())
        })
        .collect())
}

/// Held for the lifetime of a cache operation; dropping releases it.
pub(crate) struct CacheLock {
    _file: File,
}

#[derive(Clone, Copy)]
pub(crate) enum LockMode {
    SharedWait,
    SharedNow,
    ExclusiveNow,
}

pub(crate) fn cache_lock(target: &Path, mode: LockMode) -> Result<CacheLock, Failure> {
    std::fs::create_dir_all(target).map_err(|error| io_text(&error, target))?;
    let path: PathBuf = target.join(LOCK_NAME);
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(&path)
        .map_err(|error| io_text(&error, &path))?;
    let busy = || "build-cache cleanup or a local build is already running".to_owned();
    match mode {
        LockMode::SharedWait => file.lock_shared().map_err(|error| io_text(&error, &path))?,
        LockMode::SharedNow => file.try_lock_shared().map_err(|_| busy())?,
        LockMode::ExclusiveNow => file.try_lock().map_err(|_| busy())?,
    }
    Ok(CacheLock { _file: file })
}
