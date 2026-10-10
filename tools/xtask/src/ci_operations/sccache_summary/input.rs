use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

const FILE_BYTES: usize = 1024 * 1024;
const TOTAL_BYTES: usize = 64 * 1024 * 1024;
const MAX_FILES: usize = 4096;
const MAX_ENTRIES: usize = 32768;
const MAX_DEPTH: usize = 32;

pub(super) fn discover(paths: &[PathBuf]) -> Result<BTreeSet<PathBuf>, String> {
    let mut files = BTreeSet::new();
    let mut directories = BTreeSet::new();
    let mut pending = paths
        .iter()
        .map(|path| (path.clone(), 0_usize))
        .collect::<Vec<_>>();
    let mut entries = 0;
    while let Some((path, depth)) = pending.pop() {
        entries += 1;
        if entries > MAX_ENTRIES || depth > MAX_DEPTH {
            return Err("evidence discovery limit exceeded".into());
        }
        let metadata =
            fs::symlink_metadata(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "{}: symbolic links are not evidence inputs",
                path.display()
            ));
        }
        if metadata.is_dir() {
            let canonical = path.canonicalize().map_err(|error| error.to_string())?;
            if !directories.insert(canonical) {
                continue;
            }
            for entry in fs::read_dir(&path).map_err(|error| error.to_string())? {
                entries += 1;
                if entries > MAX_ENTRIES {
                    return Err("evidence discovery limit exceeded".into());
                }
                let entry = entry.map_err(|error| error.to_string())?;
                if pending.len() + entries >= MAX_ENTRIES {
                    return Err("evidence discovery limit exceeded".into());
                }
                let kind = entry.file_type().map_err(|error| error.to_string())?;
                let name = entry.file_name();
                let selected = name.to_str().is_some_and(|name| {
                    name.starts_with("sccache-stats") && name.ends_with(".json")
                });
                if kind.is_dir() || selected {
                    pending.push((entry.path(), depth + 1));
                }
            }
        } else if metadata.is_file() {
            files.insert(path.canonicalize().map_err(|error| error.to_string())?);
            if files.len() > MAX_FILES {
                return Err("too many evidence files".into());
            }
        } else {
            return Err(format!(
                "{}: evidence must be a regular file or directory",
                path.display()
            ));
        }
    }
    if files.is_empty() {
        return Err("no sccache-stats JSON evidence files found".into());
    }
    Ok(files)
}

pub(super) fn read(path: &Path, total: &mut usize) -> Result<(u64, u64), String> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.len() > FILE_BYTES as u64 {
        return Err("evidence requires a regular file of at most 1 MiB".into());
    }
    let mut bytes = Vec::new();
    file.take((FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    *total = total
        .checked_add(bytes.len())
        .ok_or("evidence byte total overflow")?;
    if bytes.len() > FILE_BYTES || *total > TOTAL_BYTES {
        return Err("evidence byte limit exceeded".into());
    }
    let document: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{}: invalid evidence JSON: {error}", path.display()))?;
    let stats = document
        .get("stats")
        .and_then(Value::as_object)
        .ok_or("evidence stats must be an object")?;
    let counter = |name: &str| {
        let counts = stats
            .get(name)
            .and_then(Value::as_object)
            .and_then(|map| map.get("counts"))
            .filter(|counts| counts.is_object())
            .ok_or_else(|| format!("stats.{name}.counts must be an object"))?;
        sum(counts, 0)
    };
    Ok((counter("cache_hits")?, counter("cache_misses")?))
}

fn sum(value: &Value, depth: usize) -> Result<u64, String> {
    if depth > MAX_DEPTH {
        return Err("counter nesting limit exceeded".into());
    }
    if let Some(count) = value.as_u64() {
        return Ok(count);
    }
    let map = value
        .as_object()
        .ok_or("counter trees require maps and nonnegative integer counters")?;
    map.values().try_fold(0_u64, |total, child| {
        total
            .checked_add(sum(child, depth + 1)?)
            .ok_or_else(|| "counter total overflow".into())
    })
}
