use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

/// One manifest edit for a table-driven rejection case.
pub(crate) type Edit = Box<dyn FnOnce(&mut serde_json::Value)>;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Opt-in interpreter for side-by-side legacy runs. Unset by default, so the
/// required suite never launches Python.
const LEGACY_PYTHON: &str = "MIGRATION_PREPARED_INPUTS_LEGACY_PYTHON";

/// A per-test scratch directory removed on drop.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-prepared-inputs-{label}-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path.canonicalize()?))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }

    pub(crate) fn join(&self, relative: &str) -> PathBuf {
        self.0.join(relative)
    }

    pub(crate) fn write(&self, relative: &str, contents: &[u8]) -> Result<PathBuf, Box<dyn Error>> {
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

pub(crate) fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

/// The legacy program a ported command replaces.
#[derive(Clone, Copy)]
pub(crate) enum Legacy {
    /// A standalone script run as `python <script> <args>`.
    Script(&'static str),
    /// The `occurrence`-th (1-based) `<<'PY'` heredoc in a checked-in file,
    /// run as `python - <args>` with the dedented body on stdin.
    Heredoc(&'static str, usize),
}

/// How closely the legacy result must match. Python tracebacks (JSON decode
/// errors, unpacking, missing files) are not reproduced byte for byte.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Parity {
    Exact,
    Status,
}

/// One ported invocation: xtask argv plus the equivalent legacy argv.
pub(crate) struct Case {
    pub(crate) args: Vec<String>,
    pub(crate) legacy: Legacy,
    pub(crate) legacy_args: Vec<String>,
    pub(crate) parity: Parity,
    /// A file the command writes; its legacy and ported bytes must agree.
    pub(crate) writes: Option<PathBuf>,
}

pub(crate) fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

impl Case {
    pub(crate) fn new(args: &[&str], legacy: Legacy, legacy_args: &[&str]) -> Self {
        Self {
            args: strings(args),
            legacy,
            legacy_args: strings(legacy_args),
            parity: Parity::Exact,
            writes: None,
        }
    }

    /// Same argv for xtask (after the command words) and the legacy program.
    pub(crate) fn same(command: &[&str], args: &[&str], legacy: Legacy) -> Self {
        Self::new(&[command, args].concat(), legacy, args)
    }

    pub(crate) fn status_only(mut self) -> Self {
        self.parity = Parity::Status;
        self
    }

    pub(crate) fn writing(mut self, path: &Path) -> Self {
        self.writes = Some(path.to_path_buf());
        self
    }

    /// Runs xtask. With `MIGRATION_PREPARED_INPUTS_LEGACY_PYTHON` set, the
    /// legacy program runs first with the same inputs; streams, status and
    /// any written file must then agree with the port.
    pub(crate) fn run(&self, cwd: &Path) -> Result<Output, Box<dyn Error>> {
        let legacy = match std::env::var_os(LEGACY_PYTHON) {
            Some(python) => {
                let output = self.run_legacy(Path::new(&python), cwd)?;
                let written = self.take_written()?;
                Some((output, written))
            }
            None => None,
        };
        let ported = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(cwd)
            .arg("prepared-input")
            .args(&self.args)
            .output()?;
        if let Some((original, written)) = legacy {
            assert_eq!(
                original.status.code(),
                ported.status.code(),
                "status parity"
            );
            if self.parity == Parity::Exact {
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
            }
            if let Some(path) = &self.writes {
                assert_eq!(written, fs::read(path).ok(), "written file parity");
            }
        }
        Ok(ported)
    }

    /// Reads and removes the legacy output so the port starts from the same
    /// state.
    fn take_written(&self) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
        let Some(path) = &self.writes else {
            return Ok(None);
        };
        let bytes = fs::read(path).ok();
        if bytes.is_some() {
            fs::remove_file(path)?;
        }
        Ok(bytes)
    }

    fn run_legacy(&self, python: &Path, cwd: &Path) -> Result<Output, Box<dyn Error>> {
        let mut command = Command::new(python);
        command
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let body = match self.legacy {
            Legacy::Script(script) => {
                command.arg(repository_root().join(script));
                None
            }
            Legacy::Heredoc(file, occurrence) => {
                command.arg("-");
                Some(heredoc(file, occurrence)?)
            }
        };
        command.args(&self.legacy_args);
        let mut child = command.spawn()?;
        let mut stdin = child.stdin.take().ok_or("missing stdin pipe")?;
        stdin.write_all(body.unwrap_or_default().as_bytes())?;
        drop(stdin);
        Ok(child.wait_with_output()?)
    }
}

/// Extracts the checked-in heredoc body, dedented like a YAML block scalar.
pub(crate) fn heredoc(file: &str, occurrence: usize) -> Result<String, Box<dyn Error>> {
    let source = fs::read_to_string(repository_root().join(file))?;
    let mut lines = source.lines();
    let mut seen = 0;
    for line in lines.by_ref() {
        if line.contains("<<'PY'") {
            seen += 1;
            if seen == occurrence {
                break;
            }
        }
    }
    let body: Vec<&str> = lines.take_while(|line| line.trim() != "PY").collect();
    if seen != occurrence || body.is_empty() {
        return Err(format!("{file} has no heredoc #{occurrence}").into());
    }
    let indent = body
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);
    Ok(body
        .iter()
        .map(|line| line.get(indent..).unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n")
}

pub(crate) fn assert_output(output: &Output, code: i32, stdout: &str, stderr: &str) {
    assert_eq!(text(&output.stderr), stderr, "stderr");
    assert_eq!(text(&output.stdout), stdout, "stdout");
    assert_eq!(output.status.code(), Some(code), "status");
}

/// Kind and bytes of every entry below `root`, symlinks included unfollowed.
pub(crate) fn snapshot(root: &Path) -> Result<BTreeMap<String, Vec<u8>>, Box<dyn Error>> {
    let mut entries = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
            let kind = fs::symlink_metadata(&path)?.file_type();
            let value = if kind.is_symlink() {
                [
                    b"link:".as_slice(),
                    fs::read_link(&path)?.to_string_lossy().as_bytes(),
                ]
                .concat()
            } else if kind.is_dir() {
                pending.push(path);
                b"dir".to_vec()
            } else {
                fs::read(&path)?
            };
            entries.insert(relative, value);
        }
    }
    Ok(entries)
}

#[cfg(unix)]
pub(crate) fn symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}
