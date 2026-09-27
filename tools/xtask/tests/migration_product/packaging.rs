//! Shared runner for the packaging-snippet ports. The legacy oracle is the
//! inline `<<'PY'` heredoc, extracted from its checked-in caller at test
//! time and fed to `$MIGRATION_PRODUCT_LEGACY_PYTHON -` on stdin with the
//! caller's argument order; the port runs `xtask product <subcommand>`.
//! Each case records status, streams, every path left in the scratch tree
//! and the bytes of its `.json` files.

use crate::packaging_cases::Case;
use crate::support::{
    CAPTURE_ENV, LEGACY_ENV, Scratch, TestResult, fixture, normalize_stderr, read_json,
    repository_root,
};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

const SCRATCH: &str = "{scratch}";

/// One inline snippet: its caller, which heredoc, and the port.
pub(crate) struct Snippet {
    pub(crate) script: &'static str,
    pub(crate) heredoc: usize,
    pub(crate) subcommand: &'static str,
    /// The legacy argument index where the caller passes
    /// `scripts/safe-extract-tar.py`, which the port does not take.
    pub(crate) extractor_at: Option<usize>,
}

/// The body of the `heredoc`-th `<<'PY'` block in `script`.
fn snippet_source(snippet: &Snippet) -> Result<String, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(repository_root().join(snippet.script))?;
    let mut blocks = text.split("<<'PY'\n").skip(1);
    let block = blocks
        .nth(snippet.heredoc)
        .ok_or("missing heredoc in legacy caller")?;
    let end = block.find("\nPY\n").ok_or("unterminated heredoc")?;
    Ok(format!("{}\n", &block[..end]))
}

fn expand(args: &[&str], root: &Path) -> Vec<String> {
    let shown = root.to_string_lossy();
    args.iter()
        .map(|arg| arg.replace(SCRATCH, &shown))
        .collect()
}

fn run_port(
    snippet: &Snippet,
    root: &Path,
    args: &[String],
) -> std::io::Result<std::process::Output> {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args(["product", snippet.subcommand])
        .args(args)
        .output()
}

fn run_legacy(
    python: &Path,
    snippet: &Snippet,
    root: &Path,
    args: &[String],
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let mut args = args.to_vec();
    if let Some(index) = snippet.extractor_at {
        let extractor = repository_root().join("scripts/safe-extract-tar.py");
        args.insert(index, extractor.to_string_lossy().into_owned());
    }
    let mut child = Command::new(python)
        .current_dir(root)
        .arg("-")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(snippet_source(snippet)?.as_bytes())?;
    Ok(child.wait_with_output()?)
}

/// Every path below `root` (directories end in `/`) and each `.json` body.
fn tree(root: &Path) -> Result<(Value, Value), Box<dyn std::error::Error>> {
    let mut paths = Vec::new();
    let mut json = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
            if std::fs::symlink_metadata(&path)?.is_dir() {
                paths.push(format!("{relative}/"));
                pending.push(path);
                continue;
            }
            if relative.ends_with(".json") {
                let body = String::from_utf8_lossy(&std::fs::read(&path)?).into_owned();
                json.insert(relative.clone(), Value::String(body));
            }
            paths.push(relative);
        }
    }
    paths.sort();
    let paths = paths.into_iter().map(Value::String).collect();
    Ok((
        Value::Array(paths),
        Value::Object(json.into_iter().collect()),
    ))
}

fn observe(
    snippet: &Snippet,
    case: &Case,
    legacy: Option<&Path>,
) -> Result<Value, Box<dyn std::error::Error>> {
    let scratch = Scratch::new()?;
    let root = scratch.path();
    (case.setup)(root)?;
    let args = expand(case.args, root);
    let output = match legacy {
        Some(python) => run_legacy(python, snippet, root, &args)?,
        None => run_port(snippet, root, &args)?,
    };
    let shown = root.to_string_lossy().into_owned();
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&shown, SCRATCH);
    let (paths, json) = tree(root)?;
    let mut result = Map::new();
    result.insert("code".into(), output.status.code().into());
    result.insert("stdout".into(), text(&output.stdout).into());
    result.insert(
        "stderr".into(),
        normalize_stderr(&text(&output.stderr)).into(),
    );
    result.insert("paths".into(), paths);
    result.insert("json".into(), json);
    Ok(Value::Object(result))
}

/// Runs `cases` against the goldens and, when requested, the legacy snippet.
pub(crate) fn run_cases(snippet: &Snippet, goldens_name: &str, cases: &[Case]) -> TestResult {
    assert!(
        !cases.is_empty(),
        "empty case list for {}",
        snippet.subcommand
    );
    let golden_path = fixture(goldens_name);
    let legacy = std::env::var_os(LEGACY_ENV);
    let capture = legacy.is_some() && std::env::var_os(CAPTURE_ENV).is_some();
    let mut goldens = if golden_path.exists() {
        read_json(&golden_path)?
    } else {
        Value::Object(Map::new())
    };
    for case in cases {
        let ported = observe(snippet, case, None)?;
        if let Some(python) = &legacy {
            let observed = observe(snippet, case, Some(Path::new(python)))?;
            assert_eq!(observed, ported, "legacy parity for {}", case.name);
            if capture && let Some(object) = goldens.as_object_mut() {
                object.insert(case.name.to_owned(), observed);
                continue;
            }
        }
        let golden = goldens
            .get(case.name)
            .ok_or_else(|| format!("missing golden {}", case.name))?;
        assert_eq!(&ported, golden, "golden for {}", case.name);
    }
    if capture {
        let text = serde_json::to_string_pretty(&goldens)?;
        std::fs::write(golden_path, format!("{text}\n"))?;
    }
    Ok(())
}
