//! Bounded offline HF snapshot observation, without repository/provider authenticity claims.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Component, Path, PathBuf},
};
fn label(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
fn includes(row: &Value) -> DynResult<Vec<glob::Pattern>> {
    let names: Vec<_> = match row.get("include") {
        None | Some(Value::Null) => vec!["*.gguf".to_owned()],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| v.as_str().map(str::to_owned).ok_or("include must be text"))
            .collect::<Result<_, _>>()?,
        _ => return Err("include must be text/list".into()),
    };
    if names.is_empty()
        || names.iter().any(|s| {
            s.contains('\\')
                || s.chars().any(char::is_control)
                || Path::new(s)
                    .components()
                    .any(|c| !matches!(c, Component::Normal(_)))
        })
    {
        return Err("unsafe cache include pattern".into());
    }
    names
        .iter()
        .map(|s| glob::Pattern::new(s).map_err(Into::into))
        .collect()
}
pub(super) fn discover(
    cache: &Path,
    row: &Value,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> DynResult<Option<Value>> {
    guard()?;
    let Some(repo) = row["repo"].as_str().filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let parts: Vec<_> = repo.split('/').collect();
    if parts.len() != 2 || !parts.iter().all(|p| label(p)) {
        return Err("discovery repo must be owner/name".into());
    }
    let cache = cache.canonicalize()?;
    let snapshots = cache
        .join(format!("models--{}--{}", parts[0], parts[1]))
        .join("snapshots");
    if !snapshots.try_exists()? {
        return Ok(None);
    }
    if !fs::symlink_metadata(&snapshots)?.is_dir() || !snapshots.canonicalize()?.starts_with(&cache)
    {
        return Err("snapshot directory escapes selected cache".into());
    }
    let patterns = includes(row)?;
    let mut scan = Scan {
        patterns: &patterns,
        matches: Vec::new(),
        count: 0,
        unsupported_matches: 0,
    };
    for snapshot in fs::read_dir(&snapshots)? {
        guard()?;
        let snapshot = snapshot?;
        scan.count += 1;
        if scan.count > 16384 {
            return Err("offline cache census exceeds bound".into());
        }
        let revision = snapshot
            .file_name()
            .into_string()
            .map_err(|_| "snapshot revision must be UTF-8")?;
        if revision.len() != 40
            || !revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            continue;
        }
        if !snapshot.file_type()?.is_dir() {
            return Err("snapshot root must be regular directory".into());
        }
        scan.collect(&snapshot.path(), &snapshot.path(), &revision, 0, guard)?;
    }
    scan.matches
        .sort_by(|a: &(u8, u64, PathBuf, String, String), b| {
            (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2))
        });
    if scan.matches.is_empty() && scan.unsupported_matches > 0 {
        return Err("offline matches have no serving GGUF; later/projector/package-only candidates require owning admission".into());
    }
    let Some((_, size, named, revision, name)) = scan.matches.into_iter().next() else {
        return Ok(None);
    };
    let path = named.canonicalize()?;
    if !path.starts_with(&cache) || !fs::symlink_metadata(&path)?.is_file() {
        return Err("observed cache target escapes regular owned cache".into());
    }
    let digest = crate::product::digest::file_sha256_with_guard(&path, guard)
        .map_err(|error| error.error)?;
    Ok(Some(
        json!({"repo":repo,"revision":revision,"file":name,"blob_sha256":digest,"size_bytes":size}),
    ))
}
struct Scan<'a> {
    patterns: &'a [glob::Pattern],
    matches: Vec<(u8, u64, PathBuf, String, String)>,
    count: usize,
    unsupported_matches: usize,
}
impl Scan<'_> {
    fn collect(
        &mut self,
        directory: &Path,
        snapshot: &Path,
        revision: &str,
        depth: usize,
        guard: &mut impl FnMut() -> std::io::Result<()>,
    ) -> DynResult<()> {
        guard()?;
        if depth > 16 {
            return Err("offline cache depth exceeds bound".into());
        }
        for entry in fs::read_dir(directory)? {
            guard()?;
            let entry = entry?;
            self.count += 1;
            if self.count > 16384 {
                return Err("offline cache census exceeds bound".into());
            }
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                self.collect(&path, snapshot, revision, depth + 1, guard)?;
                continue;
            }
            let relative = path.strip_prefix(snapshot)?;
            let name = relative
                .to_str()
                .ok_or("cache filename must be UTF-8")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            if !self.patterns.iter().any(|p| {
                p.matches_path_with(
                    relative,
                    glob::MatchOptions {
                        case_sensitive: true,
                        require_literal_separator: true,
                        require_literal_leading_dot: true,
                    },
                )
            }) {
                continue;
            }
            let Some(rank) = crate::model_registry::serving_entry::rank(&name) else {
                self.unsupported_matches += 1;
                continue;
            };
            let metadata = fs::metadata(&path)?;
            if !metadata.is_file() {
                return Err("matching cache candidate must resolve to regular file".into());
            }
            self.matches
                .push((rank, metadata.len(), path, revision.to_owned(), name));
        }
        Ok(())
    }
}
