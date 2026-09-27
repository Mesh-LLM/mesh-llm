//! Staging, stub `git`/`gh`, golden comparison and the opt-in legacy oracle
//! for the `release` parity tests.

use crate::failure_diagnostics::failure_snapshot;
use serde_json::{Value, json};
use std::error::Error;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub type TestResult = Result<(), Box<dyn Error>>;

/// Interpreter for side-by-side legacy runs; unset, no Python starts.
pub const LEGACY_ENV: &str = "MIGRATION_RELEASE_LEGACY_PYTHON";
/// Set with [`LEGACY_ENV`] to rewrite the goldens from legacy runs.
pub const CAPTURE_ENV: &str = "MIGRATION_RELEASE_CAPTURE";
const GIT_LOG: &str = "git-argv.log";
const GH_LOG: &str = "gh-argv.log";
const TRACEBACK: &str = "Traceback (most recent call last):\n";

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A legacy script and the `release` subcommand that replaces it.
#[derive(Clone, Copy)]
pub enum Tool {
    Base,
    Link,
    Classify,
    Regroup,
}

impl Tool {
    fn command(self) -> &'static str {
        match self {
            Self::Base => "notes-base",
            Self::Link => "notes-link",
            Self::Classify => "notes-classify",
            Self::Regroup => "notes-regroup",
        }
    }

    fn script(self) -> &'static str {
        match self {
            Self::Base => "scripts/select-release-notes-base.py",
            Self::Link => "scripts/release-notes-link.py",
            Self::Classify => "scripts/release-notes-classify.py",
            Self::Regroup => "scripts/release-notes-regroup.py",
        }
    }
}

/// A scratch directory removed on drop; `{root}` in compared text.
pub struct Stage(PathBuf);

impl Stage {
    pub fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-release-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(path.join("bin"))?;
        Ok(Self(path.canonicalize()?))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn write(&self, relative: &str, bytes: &[u8]) -> TestResult {
        let path = self.0.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn executable(&self, relative: &str, script: &str) -> TestResult {
        self.write(relative, script.as_bytes())?;
        #[cfg(unix)]
        fs::set_permissions(self.0.join(relative), fs::Permissions::from_mode(0o755))?;
        Ok(())
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("xtask lives under tools/")
        .to_path_buf()
}

/// Stub `git`: logs argv and cwd, then replays `git/{out,err,code}`.
fn git_stub(stage: &Path) -> String {
    let dir = stage.to_string_lossy();
    format!(
        "#!/bin/sh\ndir='{dir}'\n\
{{ printf 'cwd=%s\\n' \"$PWD\"; for arg in \"$@\"; do printf '%s\\n' \"$arg\"; done; printf '<end>\\n'; }} >> \"$dir/{GIT_LOG}\"\n\
if [ -f \"$dir/git/err\" ]; then cat \"$dir/git/err\" >&2; fi\n\
if [ -f \"$dir/git/out\" ]; then cat \"$dir/git/out\"; fi\n\
if [ -f \"$dir/git/signal\" ]; then kill -\"$(cat \"$dir/git/signal\")\" $$; fi\n\
if [ -f \"$dir/git/code\" ]; then exit \"$(cat \"$dir/git/code\")\"; fi\n"
    )
}

/// Stub `gh`: logs argv, then replays `gh/<key>.{out,err,code,signal}`
/// where `<key>` is `pulls_<sha>` or `pr_<number>`.
fn gh_stub(stage: &Path) -> String {
    let dir = stage.to_string_lossy();
    format!(
        "#!/bin/sh\ndir='{dir}'\n\
{{ for arg in \"$@\"; do printf '%s\\n' \"$arg\"; done; printf '<end>\\n'; }} >> \"$dir/{GH_LOG}\"\n\
case \"$1 $2\" in\n\
  'pr view') key=\"pr_$3\" ;;\n\
  api*) sha=${{2%/pulls}}; key=\"pulls_${{sha##*/}}\" ;;\n\
  *) key=unknown ;;\n\
esac\n\
reply=\"$dir/gh/$key\"\n\
if [ -f \"$reply.err\" ]; then cat \"$reply.err\" >&2; fi\n\
if [ -f \"$reply.out\" ]; then cat \"$reply.out\"; fi\n\
if [ -f \"$reply.signal\" ]; then kill -\"$(cat \"$reply.signal\")\" $$; fi\n\
if [ -f \"$reply.code\" ]; then exit \"$(cat \"$reply.code\")\"; fi\n\
if [ ! -f \"$reply.out\" ]; then echo \"stub gh: no reply for $key\" >&2; exit 1; fi\n"
    )
}

/// Everything a case sets up before a run.
#[derive(Default)]
pub struct Case {
    pub args: Vec<String>,
    pub stdin: Vec<u8>,
    /// Staged files, relative to the stage (e.g. `body.md`, `gh/pr_1.out`).
    pub files: Vec<(String, Vec<u8>)>,
    pub with_git: bool,
    pub with_gh: bool,
    /// Files whose contents after the run are part of the result.
    pub outputs: Vec<String>,
}

impl Case {
    pub fn new(args: &[&str]) -> Self {
        Self {
            args: args.iter().map(|arg| (*arg).to_owned()).collect(),
            with_git: true,
            with_gh: true,
            ..Self::default()
        }
    }

    pub fn file(mut self, relative: &str, text: &str) -> Self {
        self.files
            .push((relative.to_owned(), text.as_bytes().to_vec()));
        self
    }

    pub fn output(mut self, relative: &str) -> Self {
        self.outputs.push(relative.to_owned());
        self
    }
}

fn read_log(stage: &Path, name: &str) -> Vec<Vec<String>> {
    let root = stage.to_string_lossy();
    let log = fs::read_to_string(stage.join(name))
        .unwrap_or_default()
        .replace(root.as_ref(), "{root}");
    let mut calls = Vec::new();
    let mut current = Vec::new();
    for line in log.lines() {
        if line == "<end>" {
            calls.push(std::mem::take(&mut current));
        } else {
            current.push(line.to_owned());
        }
    }
    calls
}

/// A traceback keeps its header and final exception line; frames differ.
fn normalize(stderr: &str) -> String {
    match stderr.split_once(TRACEBACK) {
        Some((before, frames)) => {
            let last = frames
                .trim_end_matches('\n')
                .rsplit('\n')
                .next()
                .unwrap_or("");
            format!("{before}{TRACEBACK}{last}\n")
        }
        None => stderr.to_owned(),
    }
}

fn execute(tool: Tool, case: &Case, legacy: Option<&Path>) -> Result<Value, Box<dyn Error>> {
    let stage = Stage::new(tool.command())?;
    if case.with_git {
        stage.executable("bin/git", &git_stub(stage.path()))?;
    }
    if case.with_gh {
        stage.executable("bin/gh", &gh_stub(stage.path()))?;
    }
    for (relative, bytes) in &case.files {
        stage.write(relative, bytes)?;
    }
    let mut command = match legacy {
        Some(python) => {
            let mut command = Command::new(python);
            command.arg(repo_root().join(tool.script()));
            command
        }
        None => {
            let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
            command.args(["release", tool.command()]);
            command
        }
    };
    let root = stage.path().to_string_lossy().into_owned();
    let mut child = command
        .args(&case.args)
        .current_dir(stage.path())
        .env("PATH", format!("{root}/bin:/usr/bin:/bin"))
        .env("LANG", "en_US.UTF-8")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env_remove("LC_ALL")
        .env_remove("LC_CTYPE")
        .env_remove("PYTHONUTF8")
        .env_remove("COLUMNS")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write as _;
        stdin.write_all(&case.stdin)?;
    }
    let child_id = child.id();
    let output = child.wait_with_output()?;
    if output.status.code().is_none() {
        panic!(
            "child {} for {} exited unexpectedly: {}",
            child_id,
            tool.command(),
            failure_snapshot(stage.path(), &output, &["body.md", "body.linked.md"]),
        );
    }
    let clean = |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&root, "{root}");
    let mut outputs = serde_json::Map::new();
    for relative in &case.outputs {
        let text = fs::read(stage.path().join(relative))
            .ok()
            .map(|bytes| clean(&bytes));
        outputs.insert(relative.clone(), json!(text));
    }
    Ok(json!({
        "code": output.status.code(),
        "stdout": clean(&output.stdout),
        "stderr": normalize(&clean(&output.stderr)),
        "outputs": outputs,
        "git_argv": read_log(stage.path(), GIT_LOG),
        "gh_argv": read_log(stage.path(), GH_LOG),
    }))
}

fn golden_path(tool: Tool, name: &str) -> PathBuf {
    let dir = match tool {
        Tool::Base => "notes_base",
        Tool::Link => "notes_link",
        Tool::Classify => "notes_classify",
        Tool::Regroup => "notes_regroup",
    };
    repo_root()
        .join("tools/xtask/tests/fixtures/release")
        .join(dir)
        .join(format!("{name}.json"))
}

/// Runs the port and compares it with the golden captured from the legacy
/// script; with [`LEGACY_ENV`] set, also runs the legacy script on
/// identical inputs and requires identical results.
pub fn check(tool: Tool, name: &str, case: &Case) -> Result<Value, Box<dyn Error>> {
    let actual = execute(tool, case, None)?;
    let golden = golden_path(tool, name);
    if let Some(python) = std::env::var_os(LEGACY_ENV).map(PathBuf::from) {
        let legacy = execute(tool, case, Some(&python))?;
        if std::env::var_os(CAPTURE_ENV).is_some() {
            if let Some(parent) = golden.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut recorded = legacy.clone();
            recorded["args"] = json!(case.args);
            fs::write(&golden, serde_json::to_string_pretty(&recorded)? + "\n")?;
        }
        assert_eq!(
            actual,
            legacy,
            "{name}: port differs from legacy {}",
            python.display()
        );
    }
    let mut expected: Value = serde_json::from_slice(&fs::read(&golden)?)?;
    if let Some(map) = expected.as_object_mut() {
        map.remove("args");
    }
    assert_eq!(
        actual, expected,
        "{name}: port differs from captured golden"
    );
    Ok(actual)
}
