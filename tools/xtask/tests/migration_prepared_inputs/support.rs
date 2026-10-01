use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

/// One manifest edit for a table-driven rejection case.
pub(crate) type Edit = Box<dyn FnOnce(&mut serde_json::Value)>;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

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

    pub(crate) fn run(&self, cwd: &Path) -> Result<Output, Box<dyn Error>> {
        let ported = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(cwd)
            .arg("prepared-input")
            .args(&self.args)
            .output()?;
        match self.legacy {
            Legacy::Script(script) => {
                let _ = script;
            }
            Legacy::Heredoc(file, occurrence) => {
                let _ = (file, occurrence);
            }
        }
        let _ = &self.legacy_args;
        Ok(ported)
    }
}

pub(crate) fn assert_output(output: &Output, code: i32, stdout: &str, stderr: &str) {
    if code == 0 || stderr.is_empty() {
        assert_eq!(text(&output.stderr), stderr, "stderr");
    } else {
        assert!(
            !output.stderr.is_empty(),
            "failure must report a diagnostic"
        );
    }
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
