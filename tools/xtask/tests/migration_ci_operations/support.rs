use serde_json::{Value, json};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub type TestResult = Result<(), Box<dyn Error>>;

pub const LEGACY_ENV: &str = "MIGRATION_CI_OPERATIONS_LEGACY_PYTHON";
/// Set with the legacy interpreter to (re)write goldens from legacy runs.
pub const CAPTURE_ENV: &str = "MIGRATION_CI_OPERATIONS_CAPTURE";
pub const SCRIPT: &str = "scripts/runner-image-identity.py";

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Checkout files the legacy tool reads through `--root`.
const STAGED_FILES: [&str; 5] = [
    "ci/runner-images.json",
    "ci/slices.yml",
    "ci/ownership.yml",
    "scripts/plan-ci.py",
    "just/ci.just",
];
const STAGED_DIRS: [&str; 2] = [".github/workflows", "ci/runner-image-evidence"];

#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ci_operations")
}

fn outcome(output: &std::process::Output) -> Outcome {
    Outcome {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Runs `xtask <args>` in `cwd` with empty stdin.
pub fn xtask_in(cwd: &Path, args: &[String]) -> Result<Outcome, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(cwd)
        .args(args)
        .stdin(Stdio::null())
        .output()?;
    Ok(outcome(&output))
}

/// A per-test copy of the checkout inputs, removed on drop.
pub struct Stage(PathBuf);

impl Stage {
    pub fn empty(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-ci-ops-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path.canonicalize()?))
    }

    pub fn checkout(label: &str) -> Result<Self, Box<dyn Error>> {
        let stage = Self::empty(label)?;
        let root = repo_root();
        for name in STAGED_FILES {
            stage.write(name, &fs::read(root.join(name))?)?;
        }
        for directory in STAGED_DIRS {
            for entry in fs::read_dir(root.join(directory))? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    let name = entry.file_name();
                    let name = name.to_str().ok_or("non-UTF8 file name")?;
                    stage.write(&format!("{directory}/{name}"), &fs::read(entry.path())?)?;
                }
            }
        }
        Ok(stage)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn root_arg(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }

    pub fn write(&self, relative: &str, bytes: &[u8]) -> TestResult {
        let path = self.0.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn read(&self, relative: &str) -> Result<String, Box<dyn Error>> {
        Ok(fs::read_to_string(self.0.join(relative))?)
    }

    /// Replaces the first `count` occurrences of `old`; the text must exist.
    pub fn replace(&self, relative: &str, old: &str, new: &str) -> TestResult {
        let text = self.read(relative)?;
        if !text.contains(old) {
            return Err(format!("{relative} lacks {old:?}").into());
        }
        self.write(relative, text.replacen(old, new, 1).as_bytes())
    }

    pub fn append(&self, relative: &str, addition: &str) -> TestResult {
        let text = self.read(relative).unwrap_or_default();
        self.write(relative, (text + addition).as_bytes())
    }

    pub fn catalog(&self) -> Result<Value, Box<dyn Error>> {
        Ok(serde_json::from_str(&self.read("ci/runner-images.json")?)?)
    }

    pub fn set_catalog(&self, catalog: &Value) -> TestResult {
        self.write(
            "ci/runner-images.json",
            serde_json::to_string_pretty(catalog)?.as_bytes(),
        )
    }

    pub fn edit_catalog(&self, edit: impl FnOnce(&mut Value)) -> TestResult {
        let mut catalog = self.catalog()?;
        edit(&mut catalog);
        self.set_catalog(&catalog)
    }

    pub fn image(&self, id: &str) -> Result<String, Box<dyn Error>> {
        let catalog = self.catalog()?;
        Ok(catalog["images"][id]["reference"]
            .as_str()
            .ok_or("no reference")?
            .to_owned())
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

fn golden_path(name: &str) -> PathBuf {
    fixture_dir()
        .join("runner_identity")
        .join(format!("{name}.json"))
}

fn expand(text: &str, stage: &Stage) -> String {
    text.replace("{root}", &stage.root_arg())
}

fn collapse(text: &str, stage: &Stage) -> String {
    text.replace(&stage.root_arg(), "{root}")
}

fn run_legacy(python: &Path, stage: &Stage, args: &[String]) -> Result<Outcome, Box<dyn Error>> {
    let output = Command::new(python)
        .current_dir(stage.path())
        .arg(repo_root().join(SCRIPT))
        .args(args)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env_remove("COLUMNS")
        .stdin(Stdio::null())
        .output()?;
    Ok(outcome(&output))
}

/// Runs the Rust port on `stage` with `args` (after the subcommand), checks
/// it against the captured golden `name`, and, when the legacy interpreter
/// is configured, against a side-by-side legacy run on the same inputs.
/// `reset` restores the stage between the two runs (for `bind`).
pub fn assert_case(
    name: &str,
    stage: &Stage,
    args: &[String],
    reset: &dyn Fn() -> TestResult,
) -> Result<Outcome, Box<dyn Error>> {
    let mut argv = vec!["ci-ops".to_owned(), "runner-identity".to_owned()];
    argv.extend(args.iter().cloned());
    let actual = xtask_in(stage.path(), &argv)?;
    if let Some(python) = std::env::var_os(LEGACY_ENV).map(PathBuf::from) {
        reset()?;
        let legacy = run_legacy(&python, stage, args)?;
        if std::env::var_os(CAPTURE_ENV).is_some() {
            let golden = json!({
                "code": legacy.code,
                "stdout": collapse(&legacy.stdout, stage),
                "stderr": collapse(&legacy.stderr, stage),
            });
            fs::create_dir_all(fixture_dir().join("runner_identity"))?;
            fs::write(
                golden_path(name),
                serde_json::to_string_pretty(&golden)? + "\n",
            )?;
        }
        assert_eq!(
            actual,
            legacy,
            "{name}: Rust port differs from legacy {}",
            python.display()
        );
    }
    let golden: Value = serde_json::from_slice(&fs::read(golden_path(name))?)?;
    let expected = Outcome {
        code: i32::try_from(golden["code"].as_i64().ok_or("golden code")?)?,
        stdout: expand(golden["stdout"].as_str().ok_or("golden stdout")?, stage),
        stderr: expand(golden["stderr"].as_str().ok_or("golden stderr")?, stage),
    };
    assert_eq!(
        actual, expected,
        "{name}: Rust port differs from captured golden"
    );
    Ok(actual)
}

/// `assert_case` with `--root <stage>` prepended and nothing to reset.
pub fn check_case(name: &str, stage: &Stage, args: &[&str]) -> Result<Outcome, Box<dyn Error>> {
    let mut argv = vec!["--root".to_owned(), stage.root_arg()];
    argv.extend(args.iter().map(|arg| (*arg).to_owned()));
    assert_case(name, stage, &argv, &|| Ok(()))
}

/// A case whose argv is used verbatim (usage errors, `--catalog`).
pub fn raw_case(name: &str, stage: &Stage, args: &[&str]) -> Result<Outcome, Box<dyn Error>> {
    let argv: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
    assert_case(name, stage, &argv, &|| Ok(()))
}
