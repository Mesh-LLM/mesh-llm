use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;
pub(crate) type Built = Result<(), Box<dyn Error>>;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Opt-in interpreter for side-by-side legacy runs. Unset by default, so the
/// required suite never launches Python.
const LEGACY_PYTHON: &str = "MIGRATION_ARCHIVES_LEGACY_PYTHON";

/// Placeholder for the per-run scratch directory in compared streams.
pub(crate) const SCRATCH: &str = "<SCRATCH>";

/// A per-test scratch directory removed on drop.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-archives-{label}-{}-{sequence}-{nanos}",
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

/// The legacy script an `artifact` command replaces.
#[derive(Clone, Copy)]
pub(crate) enum Tool {
    Checksum,
    Tar,
    Zip,
}

impl Tool {
    fn command(self) -> &'static str {
        match self {
            Self::Checksum => "verify-checksum",
            Self::Tar => "extract-tar",
            Self::Zip => "extract-zip",
        }
    }

    fn script(self) -> &'static str {
        match self {
            Self::Checksum => "scripts/verify-checksum-sidecar.py",
            Self::Tar => "scripts/safe-extract-tar.py",
            Self::Zip => "scripts/safe-extract-zip.py",
        }
    }
}

/// How closely a legacy run must match. Python tracebacks and argparse's
/// program name are not reproduced byte for byte; `LastLine` compares only
/// the final stderr line (the exception of an uncaught traceback).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Parity {
    Exact,
    Status,
    LastLine,
}

fn last_line(stream: &str) -> &str {
    stream
        .trim_end_matches('\n')
        .rsplit('\n')
        .next()
        .unwrap_or("")
}

/// The observable result of one run: streams with the scratch path replaced,
/// status, and every entry left in the scratch directory.
pub(crate) struct Outcome {
    pub(crate) code: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) tree: BTreeMap<String, String>,
    pub(crate) root: Scratch,
}

impl Outcome {
    pub(crate) fn assert(&self, code: i32, stderr: &str) {
        assert_eq!(self.stderr, stderr, "stderr");
        assert_eq!(self.stdout, "", "stdout");
        assert_eq!(self.code, Some(code), "status");
    }

    pub(crate) fn path(&self, relative: &str) -> PathBuf {
        self.root.path().join(relative)
    }

    pub(crate) fn entry(&self, relative: &str) -> Option<&str> {
        self.tree.get(relative).map(String::as_str)
    }
}

/// Builds the fixture into a fresh scratch directory, runs the port there with
/// `args` (relative to the scratch directory, which is also the working
/// directory) and, when `MIGRATION_ARCHIVES_LEGACY_PYTHON` is set, repeats
/// the run with the legacy script on an identical fixture and requires the
/// same status, streams and final tree.
pub(crate) fn run(
    tool: Tool,
    args: &[&str],
    parity: Parity,
    build: impl Fn(&Path) -> Built,
) -> Result<Outcome, Box<dyn Error>> {
    let ported = execute(tool, args, &build, None)?;
    if let Some(python) = std::env::var_os(LEGACY_PYTHON) {
        let legacy = execute(tool, args, &build, Some(Path::new(&python)))?;
        assert_eq!(legacy.code, ported.code, "status parity for {args:?}");
        if parity == Parity::Exact {
            assert_eq!(legacy.stdout, ported.stdout, "stdout parity for {args:?}");
            assert_eq!(legacy.stderr, ported.stderr, "stderr parity for {args:?}");
        }
        if parity == Parity::LastLine {
            assert_eq!(legacy.stdout, ported.stdout, "stdout parity for {args:?}");
            assert_eq!(
                last_line(&legacy.stderr),
                last_line(&ported.stderr),
                "stderr exception parity for {args:?}"
            );
        }
        assert_eq!(legacy.tree, ported.tree, "tree parity for {args:?}");
    }
    Ok(ported)
}

fn execute(
    tool: Tool,
    args: &[&str],
    build: &impl Fn(&Path) -> Built,
    legacy: Option<&Path>,
) -> Result<Outcome, Box<dyn Error>> {
    let root = Scratch::new(tool.command())?;
    build(root.path())?;
    let output: Output = match legacy {
        Some(python) => Command::new(python)
            .current_dir(root.path())
            .arg(repository_root().join(tool.script()))
            .args(args)
            .output()?,
        None => Command::new(env!("CARGO_BIN_EXE_xtask"))
            .current_dir(root.path())
            .args(["artifact", tool.command()])
            .args(args)
            .output()?,
    };
    let scratch = root.path().to_string_lossy().into_owned();
    Ok(Outcome {
        code: output.status.code(),
        stdout: text(&output.stdout).replace(&scratch, SCRATCH),
        stderr: text(&output.stderr).replace(&scratch, SCRATCH),
        tree: snapshot(root.path())?,
        root,
    })
}

/// Kind, permission bits, link count and bytes of every entry below `root`,
/// symlinks recorded by target and never followed.
pub(crate) fn snapshot(root: &Path) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut entries = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
            let metadata = fs::symlink_metadata(&path)?;
            let mode = metadata.permissions().mode() & 0o7777;
            let value = if metadata.file_type().is_symlink() {
                format!("link -> {}", fs::read_link(&path)?.to_string_lossy())
            } else if metadata.is_dir() {
                pending.push(path);
                format!("dir {mode:o}")
            } else if metadata.is_file() {
                let bytes = text(&fs::read(&path)?);
                format!("file {mode:o} nlink={} {bytes:?}", metadata.nlink())
            } else {
                format!("other {mode:o}")
            };
            entries.insert(relative, value);
        }
    }
    Ok(entries)
}

pub(crate) fn write(root: &Path, relative: &str, bytes: &[u8]) -> Built {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
