use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A per-test scratch directory removed on drop, unique per process and call.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-migration-repository-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path.canonicalize()?))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }

    pub(crate) fn write(&self, relative: &str, contents: &str) -> Result<PathBuf, Box<dyn Error>> {
        let path = self.0.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, contents)?;
        Ok(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

/// Runs a program with optional stdin and extra environment.
pub(crate) struct Invocation<'a> {
    pub(crate) cwd: &'a Path,
    pub(crate) args: &'a [&'a str],
    pub(crate) stdin: Option<&'a str>,
    pub(crate) env: &'a [(&'a str, &'a str)],
}

impl Invocation<'_> {
    pub(crate) fn run(&self) -> Result<Output, Box<dyn Error>> {
        self.run_program(Path::new(env!("CARGO_BIN_EXE_xtask")), &[])
    }

    fn run_program(&self, program: &Path, prefix: &[&str]) -> Result<Output, Box<dyn Error>> {
        let mut command = Command::new(program);
        command
            .current_dir(self.cwd)
            .args(prefix)
            .args(self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in self.env {
            command.env(key, value);
        }
        let mut child = command.spawn()?;
        let mut stdin = child.stdin.take().ok_or("missing stdin pipe")?;
        stdin.write_all(self.stdin.unwrap_or_default().as_bytes())?;
        drop(stdin);
        Ok(child.wait_with_output()?)
    }

    /// Runs the xtask command and, when the operator opts in, the legacy
    /// script with the same cwd/stdin/env, requiring byte-identical streams and
    /// status. `MIGRATION_REPOSITORY_LEGACY_BASH` / `_PYTHON` name the
    /// interpreters; without them only the recorded expectations apply.
    pub(crate) fn run_with_legacy(&self, legacy: Legacy<'_>) -> Result<Output, Box<dyn Error>> {
        let ported = self.run()?;
        let interpreter = match legacy.kind {
            LegacyKind::Bash => std::env::var_os("MIGRATION_REPOSITORY_LEGACY_BASH"),
            LegacyKind::Python => std::env::var_os("MIGRATION_REPOSITORY_LEGACY_PYTHON"),
        };
        if let Some(interpreter) = interpreter {
            let script = repository_root().join(legacy.script);
            let script = script.to_str().ok_or("non-UTF8 script path")?;
            let replay = Invocation {
                args: legacy.args,
                ..*self
            };
            let original = replay.run_program(Path::new(&interpreter), &[script])?;
            assert_eq!(
                text(&original.stdout),
                text(&ported.stdout),
                "stdout parity"
            );
            assert_eq!(
                text(&original.stderr),
                text(&ported.stderr),
                "stderr parity"
            );
            assert_eq!(
                original.status.code(),
                ported.status.code(),
                "status parity"
            );
        }
        Ok(ported)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum LegacyKind {
    Bash,
    Python,
}

/// The legacy script replaced by a ported command, and its argv.
#[derive(Clone, Copy)]
pub(crate) struct Legacy<'a> {
    pub(crate) kind: LegacyKind,
    pub(crate) script: &'a str,
    pub(crate) args: &'a [&'a str],
}

pub(crate) fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Asserts the exact status code and both streams of a command.
pub(crate) fn assert_output(output: &Output, code: i32, stdout: &str, stderr: &str) {
    assert_eq!(text(&output.stdout), stdout, "stdout");
    assert_eq!(text(&output.stderr), stderr, "stderr");
    assert_eq!(output.status.code(), Some(code), "status");
}

/// Runs git with hermetic identity and hook settings, returning trimmed stdout.
pub(crate) fn git(repo: &Path, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(format!("git {args:?} failed: {}", text(&output.stderr)).into());
    }
    Ok(text(&output.stdout).trim().to_owned())
}

/// Creates an empty git repository with a hermetic identity.
pub(crate) fn git_init(repo: &Path) -> TestResult {
    fs::create_dir_all(repo)?;
    git(repo, &["init", "--quiet"])?;
    Ok(())
}

/// Commits every change in `repo` with `message`, returning the new SHA.
pub(crate) fn commit_all(repo: &Path, message: &str) -> Result<String, Box<dyn Error>> {
    git(repo, &["add", "-A"])?;
    git(repo, &["commit", "--quiet", "--allow-empty", "-m", message])?;
    git(repo, &["rev-parse", "HEAD"])
}
