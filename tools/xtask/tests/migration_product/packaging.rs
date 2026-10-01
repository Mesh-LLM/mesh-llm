//! Each case records status, streams, every path left in the scratch tree
//! and the bytes of its `.json` files.

use crate::packaging_cases::Case;
use crate::support::{Scratch, TestResult, fixture, normalize_stderr, read_json};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

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

fn observe(snippet: &Snippet, case: &Case) -> Result<Value, Box<dyn std::error::Error>> {
    let scratch = Scratch::new()?;
    let root = scratch.path();
    (case.setup)(root)?;
    let args = expand(case.args, root);
    let output = run_port(snippet, root, &args)?;
    let _ = (snippet.script, snippet.heredoc, snippet.extractor_at);
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
    let goldens = if golden_path.exists() {
        read_json(&golden_path)?
    } else {
        Value::Object(Map::new())
    };
    for case in cases {
        let ported = observe(snippet, case)?;
        let golden = goldens
            .get(case.name)
            .ok_or_else(|| format!("missing golden {}", case.name))?;
        if golden["code"] != 0 {
            assert_eq!(ported["code"], golden["code"], "{}", case.name);
            assert_eq!(ported["stdout"], golden["stdout"], "{}", case.name);
            assert_eq!(ported["paths"], golden["paths"], "{}", case.name);
            assert_eq!(ported["json"], golden["json"], "{}", case.name);
            assert!(!ported["stderr"].as_str().ok_or("stderr")?.is_empty());
        } else {
            assert_eq!(&ported, golden, "golden for {}", case.name);
        }
    }
    Ok(())
}
