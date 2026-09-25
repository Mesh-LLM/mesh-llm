use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) const INTERPRETER_ENV: &str = "MIGRATION_CI_PLAN_LEGACY_PYTHON";
pub(crate) const BASH_ENV: &str = "MIGRATION_CI_PLAN_LEGACY_BASH";

pub(crate) fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

pub(crate) fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ci_plan")
}

pub(crate) fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// A per-test scratch directory, removed on drop.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-migration-ci-shadow-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path.canonicalize()?))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

/// `cargo xtool automation parity <args>` from the repository root, with
/// the legacy opt-in variables removed unless `inherit_legacy` is set.
pub(crate) fn parity(args: &[&str], inherit_legacy: bool) -> Result<Output, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(repository_root())
        .args(["automation", "parity"])
        .args(args);
    if !inherit_legacy {
        command.env_remove(INTERPRETER_ENV).env_remove(BASH_ENV);
    }
    Ok(command.output()?)
}

/// Prints both streams so `--nocapture` shows the evidence.
pub(crate) fn show(label: &str, output: &Output) {
    println!(
        "== {label}: status {:?}\n{}\n{}",
        output.status.code(),
        text(&output.stdout),
        text(&output.stderr)
    );
}

pub(crate) fn copy_tree(source: &Path, destination: &Path) -> TestResult {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

fn git(repo: &Path, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .current_dir(repo)
        .args([
            "-c",
            "user.name=shadow",
            "-c",
            "user.email=shadow@example.invalid",
        ])
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
        ])
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(format!("git {args:?}: {}", text(&output.stderr)).into());
    }
    Ok(text(&output.stdout).trim().to_owned())
}

/// A source repository holding the protected catalogs, then `mutate`
/// applied, committed; returns the commit SHA.
pub(crate) fn source_commit(
    repo: &Path,
    mutate: impl FnOnce(&Path) -> TestResult,
) -> Result<String, Box<dyn Error>> {
    fs::create_dir_all(repo.join("ci"))?;
    git(repo, &["init", "-q"])?;
    for catalog in ["ownership", "slices"] {
        let name = format!("ci/{catalog}.yml");
        fs::copy(repository_root().join(&name), repo.join(&name))?;
    }
    mutate(repo)?;
    git(repo, &["add", "-A"])?;
    git(repo, &["commit", "-q", "-m", "source"])?;
    git(repo, &["rev-parse", "HEAD"])
}
