use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type TestResult = Result<(), Box<dyn Error>>;

/// Opt-in interpreter for side-by-side legacy runs; unset by default so the
/// required suite never launches Python.
/// With the interpreter set, rewrite the goldens from the legacy run.
const SCRATCH: &str = "{scratch}";

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A per-run scratch directory removed on drop.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn path(&self) -> &Path {
        &self.0
    }

    pub(crate) fn new() -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-product-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path.canonicalize()?))
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

pub(crate) fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/product")
        .join(name)
}

pub(crate) fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// A file body: a string, `{"hex": ...}`, or `{"prefix", "repeat", "count", "suffix"}`.
fn body(spec: &Value) -> Result<Vec<u8>, Box<dyn Error>> {
    if let Some(text) = spec.as_str() {
        return Ok(text.as_bytes().to_vec());
    }
    if let Some(hex_text) = spec.get("hex").and_then(Value::as_str) {
        return Ok(hex::decode(hex_text)?);
    }
    let part = |key: &str| spec.get(key).and_then(Value::as_str).unwrap_or("");
    let count = spec
        .get("count")
        .and_then(Value::as_u64)
        .ok_or("bad count")?;
    let middle = part("repeat").repeat(usize::try_from(count)?);
    Ok(format!("{}{middle}{}", part("prefix"), part("suffix")).into_bytes())
}

fn write_files(root: &Path, files: Option<&Value>) -> TestResult {
    let Some(files) = files.and_then(Value::as_object) else {
        return Ok(());
    };
    for (relative, spec) in files {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if relative.ends_with('/') {
            fs::create_dir_all(&path)?;
            continue;
        }
        fs::write(path, body(spec)?)?;
    }
    Ok(())
}

fn write_links(root: &Path, links: Option<&Value>) -> TestResult {
    let Some(links) = links.and_then(Value::as_object) else {
        return Ok(());
    };
    for (relative, target) in links {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        std::os::unix::fs::symlink(target.as_str().ok_or("bad link")?, path)?;
    }
    Ok(())
}

fn arguments(case: &Value, key: &str, root: &Path) -> Vec<String> {
    let scratch = root.to_string_lossy();
    case.get(key)
        .and_then(Value::as_array)
        .map(|args| {
            args.iter()
                .filter_map(Value::as_str)
                .map(|arg| arg.replace(SCRATCH, &scratch))
                .collect()
        })
        .unwrap_or_default()
}

/// Which implementation runs a case.
fn invoke(root: &Path, args: &[String]) -> std::io::Result<std::process::Output> {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args(["product", "compose"])
        .args(args)
        .output()
}

/// An uncaught Python exception is reduced to its traceback's last line,
/// the part the port reproduces.
pub(crate) fn normalize_stderr(stderr: &str) -> String {
    if !stderr.starts_with("Traceback (most recent call last):") {
        return stderr.to_owned();
    }
    let last = stderr
        .trim_end_matches('\n')
        .rsplit('\n')
        .next()
        .unwrap_or("");
    format!("{last}\n")
}

/// Status, streams and every regular file left below the scratch directory.
pub(crate) fn execute(case: &Value) -> Result<Value, Box<dyn Error>> {
    let scratch = Scratch::new()?;
    let root = scratch.0.as_path();
    write_files(root, case.get("files"))?;
    write_links(root, case.get("links"))?;
    let setup = arguments(case, "setup", root);
    if !setup.is_empty() {
        let output = invoke(root, &setup)?;
        assert!(output.status.success(), "setup failed: {output:?}");
    }
    remove(root, case.get("remove"))?;
    write_files(root, case.get("mutate"))?;
    let output = invoke(root, &arguments(case, "args", root))?;
    let shown = root.to_string_lossy().into_owned();
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&shown, SCRATCH);
    let mut result = Map::new();
    result.insert("code".into(), output.status.code().into());
    result.insert("stdout".into(), text(&output.stdout).into());
    result.insert(
        "stderr".into(),
        normalize_stderr(&text(&output.stderr)).into(),
    );
    result.insert("products".into(), products(root)?);
    Ok(Value::Object(result))
}

/// Every `product-manifest.json` below `root`, by relative path.
fn products(root: &Path) -> Result<Value, Box<dyn Error>> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.is_dir() {
                pending.push(path);
            } else if path
                .file_name()
                .is_some_and(|name| name == "product-manifest.json")
            {
                let relative = path.strip_prefix(root)?.to_string_lossy().into_owned();
                let bytes = String::from_utf8_lossy(&fs::read(&path)?).into_owned();
                found.insert(relative, Value::String(bytes));
            }
        }
    }
    Ok(Value::Object(found.into_iter().collect()))
}

fn remove(root: &Path, paths: Option<&Value>) -> TestResult {
    for relative in paths.and_then(Value::as_array).into_iter().flatten() {
        let path = root.join(relative.as_str().ok_or("bad remove path")?);
        if fs::symlink_metadata(&path)?.is_dir() {
            fs::remove_dir_all(path)?;
        } else {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}
