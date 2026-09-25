use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

/// A scratch checkout for one test: manifest roots laid out as the planner
/// reads them (`<name>/ci/{ownership,slices}.yml`, copied from the census-
/// neutral `.json` fixtures) and a `bin/cargo` that replays the frozen
/// `cargo metadata` document the goldens were generated against.
pub(crate) struct Stage(PathBuf);

impl Stage {
    pub(crate) fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-migration-ci-plan-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(path.join("bin"))?;
        let stage = Self(path.canonicalize()?);
        stage.install_fake_cargo()?;
        Ok(stage)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }

    fn install_fake_cargo(&self) -> TestResult {
        let metadata = fixture_root().join("cargo-metadata.json");
        let script = self.0.join("bin/cargo");
        fs::write(
            &script,
            format!("#!/bin/sh\ncat '{}'\n", metadata.display()),
        )?;
        make_executable(&script)
    }

    /// Materializes fixture manifest set `name` and returns its root.
    pub(crate) fn manifest_root(&self, name: &str) -> Result<PathBuf, Box<dyn Error>> {
        let root = self.0.join("manifests").join(name);
        let source = fixture_root().join("manifests").join(name);
        fs::create_dir_all(root.join("ci"))?;
        for catalog in ["ownership", "slices"] {
            fs::copy(
                source.join(format!("{catalog}.json")),
                root.join("ci").join(format!("{catalog}.yml")),
            )?;
        }
        Ok(root)
    }

    /// `PATH` with the fake cargo (and an optional legacy bash) first.
    pub(crate) fn search_path(&self) -> Result<String, Box<dyn Error>> {
        if let Some(bash) = std::env::var_os("MIGRATION_CI_PLAN_LEGACY_BASH") {
            let link = self.0.join("bin/bash");
            if !link.exists() {
                symlink(Path::new(&bash), &link)?;
            }
        }
        let inherited = std::env::var("PATH").unwrap_or_default();
        Ok(format!("{}:{inherited}", self.0.join("bin").display()))
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

#[cfg(unix)]
fn make_executable(path: &Path) -> TestResult {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> TestResult {
    Ok(())
}

#[cfg(unix)]
fn symlink(target: &Path, link: &Path) -> TestResult {
    std::os::unix::fs::symlink(target, link)?;
    Ok(())
}

#[cfg(not(unix))]
fn symlink(_target: &Path, _link: &Path) -> TestResult {
    Err("legacy bash override requires a Unix host".into())
}

/// One planner process: program, argv, stdin and `PATH`.
pub(crate) struct Run<'a> {
    pub(crate) args: &'a [&'a str],
    pub(crate) stdin: &'a [u8],
    pub(crate) path: &'a str,
}

impl Run<'_> {
    /// `cargo xtool ci plan <args>` from the repository root.
    pub(crate) fn ported(&self) -> Result<Output, Box<dyn Error>> {
        let mut argv = vec!["ci", "plan"];
        argv.extend_from_slice(self.args);
        spawn(Path::new(env!("CARGO_BIN_EXE_xtask")), &argv, self)
    }

    /// `python3 scripts/plan-ci.py <args>` when the operator names an
    /// interpreter; `None` keeps default runs free of Python.
    pub(crate) fn legacy(&self) -> Result<Option<Output>, Box<dyn Error>> {
        let Some(python) = std::env::var_os("MIGRATION_CI_PLAN_LEGACY_PYTHON") else {
            return Ok(None);
        };
        let script = repository_root().join("scripts/plan-ci.py");
        let script = script.to_str().ok_or("non-UTF8 script path")?;
        let mut argv = vec![script];
        argv.extend_from_slice(self.args);
        spawn(Path::new(&python), &argv, self).map(Some)
    }
}

fn spawn(program: &Path, argv: &[&str], run: &Run<'_>) -> Result<Output, Box<dyn Error>> {
    let mut child = Command::new(program)
        .current_dir(repository_root())
        .args(argv)
        .env("PATH", run.path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().ok_or("missing stdin pipe")?;
    stdin.write_all(run.stdin)?;
    drop(stdin);
    Ok(child.wait_with_output()?)
}

/// Asserts that legacy and ported runs agree on status and both streams.
pub(crate) fn assert_same_process(legacy: &Output, ported: &Output, label: &str) {
    assert_eq!(
        legacy.status.code(),
        ported.status.code(),
        "{label}: status parity"
    );
    assert_eq!(
        text(&legacy.stdout),
        text(&ported.stdout),
        "{label}: stdout parity"
    );
    assert_eq!(
        text(&legacy.stderr),
        text(&ported.stderr),
        "{label}: stderr parity"
    );
}
