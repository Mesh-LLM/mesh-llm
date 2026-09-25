use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) const RESOLVER: &str = "scripts/resolve-test-model-manifest.py";
pub(crate) const GENERATOR: &str = "scripts/generate-test-model-manifests.py";

pub(crate) fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

pub(crate) fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models")
}

pub(crate) fn fixture(name: &str) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        fixture_dir().join(name),
    )?)?)
}

pub(crate) fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

pub(crate) fn field<'a>(value: &'a Value, key: &str) -> Result<&'a str, Box<dyn Error>> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("fixture field {key} is not a string").into())
}

pub(crate) fn code(value: &Value) -> Result<i32, Box<dyn Error>> {
    let code = value["code"]
        .as_i64()
        .ok_or("fixture code is not an integer")?;
    Ok(i32::try_from(code)?)
}

/// The legacy interpreter named by the operator, or `None` for default runs.
pub(crate) fn legacy(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from)
}

/// A per-test directory, canonical so legacy `{root}` substitutions match.
/// It carries the two workspace markers `--repo-root` requires.
pub(crate) struct Stage(PathBuf);

impl Stage {
    pub(crate) fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-migration-models-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(path.join("tools/xtask"))?;
        fs::write(path.join("Cargo.toml"), "")?;
        fs::write(path.join("tools/xtask/Cargo.toml"), "")?;
        Ok(Self(path.canonicalize()?))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }

    pub(crate) fn root_arg(&self) -> Result<&str, Box<dyn Error>> {
        self.0.to_str().ok_or_else(|| "non-UTF8 stage".into())
    }

    pub(crate) fn write(&self, relative: &str, bytes: &[u8]) -> TestResult {
        let path = self.0.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    pub(crate) fn read(&self, relative: &str) -> Result<String, Box<dyn Error>> {
        Ok(fs::read_to_string(self.0.join(relative))?)
    }

    /// Copies the frozen suite manifests to `manifests/`.
    pub(crate) fn frozen_manifests(&self) -> TestResult {
        for entry in fs::read_dir(fixture_dir().join("manifests"))? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_str().ok_or("non-UTF8 manifest name")?;
            self.write(&format!("manifests/{name}"), &fs::read(entry.path())?)?;
        }
        Ok(())
    }

    /// Copies a legacy script into the stage so its `ROOT` is the stage.
    pub(crate) fn legacy_script(&self, script: &str) -> Result<PathBuf, Box<dyn Error>> {
        self.write(script, &fs::read(repository_root().join(script))?)?;
        Ok(self.0.join(script))
    }

    /// Replaces `{root}` with this stage's canonical path.
    pub(crate) fn expand(&self, value: &str) -> Result<String, Box<dyn Error>> {
        Ok(value.replace("{root}", self.root_arg()?))
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

/// Runs `xtask <args>` in `cwd` with empty stdin.
pub(crate) fn xtask(cwd: &Path, args: &[&str]) -> Result<Output, Box<dyn Error>> {
    run(Path::new(env!("CARGO_BIN_EXE_xtask")), cwd, args)
}

/// Runs `program <args>` in `cwd` with empty stdin.
pub(crate) fn run(program: &Path, cwd: &Path, args: &[&str]) -> Result<Output, Box<dyn Error>> {
    Ok(Command::new(program)
        .current_dir(cwd)
        .args(args)
        .stdin(Stdio::null())
        .output()?)
}

/// Runs the legacy script under the operator's interpreter.
pub(crate) fn run_legacy(
    python: &Path,
    script: &Path,
    cwd: &Path,
    args: &[&str],
) -> Result<Output, Box<dyn Error>> {
    let script = script.to_str().ok_or("non-UTF8 script path")?;
    let mut argv = vec![script];
    argv.extend_from_slice(args);
    run(python, cwd, &argv)
}

/// Asserts status and both streams, labelled with the case name.
pub(crate) fn assert_streams(case: &str, output: &Output, code: i32, stdout: &str, stderr: &str) {
    assert_eq!(text(&output.stdout), stdout, "{case}: stdout");
    assert_eq!(text(&output.stderr), stderr, "{case}: stderr");
    assert_eq!(output.status.code(), Some(code), "{case}: status");
}

/// Asserts that two processes produced identical observable results.
pub(crate) fn assert_same(case: &str, legacy: &Output, ported: &Output) {
    assert_eq!(
        text(&legacy.stdout),
        text(&ported.stdout),
        "{case}: stdout parity"
    );
    assert_eq!(
        text(&legacy.stderr),
        text(&ported.stderr),
        "{case}: stderr parity"
    );
    assert_eq!(
        legacy.status.code(),
        ported.status.code(),
        "{case}: status parity"
    );
}
