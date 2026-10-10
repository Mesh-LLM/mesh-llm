//! Bounded competitive artifact/shard discovery using the saved report file owner.
use crate::{command::DynResult, process::Cancellation};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
pub(super) struct Input<'a> {
    pub cancellation: &'a Cancellation,
    pub sources: BTreeSet<PathBuf>,
    bytes: usize,
}
impl<'a> Input<'a> {
    pub(super) fn new(cancellation: &'a Cancellation) -> Self {
        Self {
            cancellation,
            sources: BTreeSet::new(),
            bytes: 0,
        }
    }
    pub(super) fn check(&self) -> DynResult<()> {
        if self.cancellation.is_cancelled() {
            Err("performance history cancelled".into())
        } else {
            Ok(())
        }
    }
    pub(super) fn read(&mut self, path: &Path) -> DynResult<Vec<u8>> {
        self.check()?;
        let data = crate::ci_operations::ci_metrics_transport::read_file(path, self.cancellation)
            .map_err(|failure| match failure {
            crate::ci_operations::ci_metrics_normalize::Failure::Reported(s)
            | crate::ci_operations::ci_metrics_normalize::Failure::Uncaught(s) => s,
        })?;
        self.bytes = self
            .bytes
            .checked_add(data.len())
            .ok_or("history input size overflow")?;
        if self.bytes > 128 * 1024 * 1024 {
            return Err("history total input exceeds128MiB".into());
        }
        self.sources.insert(path.canonicalize()?);
        Ok(data)
    }
    pub(super) fn json(&mut self, path: &Path) -> DynResult<Value> {
        let value: Value = serde_json::from_slice(&self.read(path)?)?;
        if !value.is_object() {
            return Err("history artifact JSON must be an object".into());
        }
        Ok(value)
    }
    pub(super) fn files(&self, root: &Path) -> DynResult<Vec<PathBuf>> {
        let mut pending = vec![(root.to_owned(), 0)];
        let mut files = Vec::new();
        let mut entries = 0;
        while let Some((path, depth)) = pending.pop() {
            self.check()?;
            entries += 1;
            if entries > 32768 || depth > 16 {
                return Err("history discovery limit exceeded".into());
            }
            let meta = std::fs::symlink_metadata(&path)?;
            if meta.file_type().is_symlink() {
                return Err("history inputs must not be symlinked".into());
            }
            if meta.is_dir() {
                for entry in std::fs::read_dir(path)? {
                    self.check()?;
                    pending.push((entry?.path(), depth + 1));
                    if pending.len() + entries > 32768 {
                        return Err("history discovery limit exceeded".into());
                    }
                }
            } else if meta.is_file() {
                files.push(path);
                if files.len() > 4096 {
                    return Err("too many history files".into());
                }
            } else {
                return Err("history inputs must be regular files/directories".into());
            }
        }
        files.sort();
        Ok(files)
    }
    pub(super) fn history(&mut self, path: Option<&Path>) -> DynResult<Vec<super::records::Row>> {
        let Some(path) = path else {
            return Ok(Vec::new());
        };
        match std::fs::symlink_metadata(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
            Ok(_) => {}
        }
        let mut rows = Vec::new();
        for file in self.files(path)? {
            if path.is_dir() && file.extension().is_none_or(|e| e != "jsonl") {
                continue;
            }
            let data = self.read(&file)?;
            let text = std::str::from_utf8(&data)?;
            for line in text.lines().filter(|line| !line.trim().is_empty()) {
                self.check()?;
                let row: super::records::Row = serde_json::from_str(line)?;
                row.validate()?;
                rows.push(row);
                if rows.len() > 32768 {
                    return Err("too many history rows".into());
                }
            }
        }
        Ok(rows)
    }
}
