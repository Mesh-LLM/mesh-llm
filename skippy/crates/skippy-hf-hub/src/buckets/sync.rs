//! One-way directory synchronization between a local folder and a Hub bucket.
//!
//! [`HFBucket::sync`] compares a local directory with a bucket prefix, computes
//! the required uploads, downloads, deletions, and skips, executes those
//! operations, and returns the resulting [`BucketSyncPlan`].
//!
//! A few behaviors are worth knowing up front:
//!
//! - Sync is directional. Use [`BucketSyncDirection::Upload`] to push a local directory into a bucket, or
//!   [`BucketSyncDirection::Download`] to mirror a bucket prefix into a local directory.
//! - `prefix` scopes the remote side of the comparison. Returned operation paths are always relative to that prefix and
//!   to `local_path`.
//! - `include` and `exclude` glob patterns are evaluated against those relative paths. Excludes win over includes.
//! - The returned plan reflects the operations that were executed. Set the `verbose(true)` builder option on
//!   [`HFBucket::sync`] to keep explicit skip entries in the plan.
//! - By default, file comparisons consider both size and modification time, with a small timestamp tolerance to avoid
//!   unnecessary transfers.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use bon::bon;
use futures::StreamExt;
use globset::{Glob, GlobMatcher};

use crate::buckets::{BucketTreeEntry, BucketUpload, HFBucket};
use crate::error::{HFError, HFResult};
use crate::progress::{DownloadEvent, EmitEvent, Progress};

/// Direction for a bucket sync operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketSyncDirection {
    /// Local directory -> bucket (upload).
    Upload,
    /// Bucket -> local directory (download).
    Download,
}

/// Internal options struct used by [`HFBucket::sync`]'s helper functions. Public callers go
/// through the bon-generated `sync()` builder instead.
#[derive(Clone)]
pub(crate) struct BucketSyncParams {
    pub local_path: PathBuf,
    pub direction: BucketSyncDirection,
    pub prefix: Option<String>,
    pub delete: bool,
    pub ignore_times: bool,
    pub ignore_sizes: bool,
    pub existing: bool,
    pub ignore_existing: bool,
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub verbose: bool,
    pub progress: Option<Progress>,
}

/// Action to perform for a single file during sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketSyncAction {
    /// Upload a local file into the bucket.
    Upload,
    /// Download a bucket file into the local directory.
    Download,
    /// Remove a file from the receiving side because `delete` is enabled.
    Delete,
    /// Leave the file untouched.
    Skip,
}

/// A single operation within a sync plan.
#[derive(Debug, Clone)]
pub struct BucketSyncOperation {
    /// What action to take.
    pub action: BucketSyncAction,
    /// Relative file path (forward-slash separated).
    pub path: String,
    /// File size in bytes, if known.
    pub size: Option<u64>,
    /// Human-readable reason for this action (e.g., "new file", "size differs", "identical").
    pub reason: String,
}

/// The result of a sync — describes the operations that were executed.
///
/// Returned by [`HFBucket::sync`].
#[derive(Debug, Clone)]
pub struct BucketSyncPlan {
    /// Sync direction that produced this plan.
    pub direction: BucketSyncDirection,
    /// All operations in the plan.
    pub operations: Vec<BucketSyncOperation>,
    /// Bucket tree entries for download operations, keyed by relative path.
    /// Used internally during execution to avoid re-fetching metadata.
    pub(crate) download_entries: HashMap<String, BucketTreeEntry>,
}

impl BucketSyncPlan {
    /// Count upload operations in the plan.
    pub fn uploads(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Upload)
            .count()
    }

    /// Count download operations in the plan.
    pub fn downloads(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Download)
            .count()
    }

    /// Count delete operations in the plan.
    pub fn deletes(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Delete)
            .count()
    }

    /// Count skip operations in the plan.
    ///
    /// Skip entries are only included when the `verbose(true)` builder option is set on
    /// [`HFBucket::sync`].
    pub fn skips(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Skip)
            .count()
    }

    /// Total bytes to transfer (upload + download operations).
    pub fn transfer_bytes(&self) -> u64 {
        self.operations
            .iter()
            .filter(|op| {
                op.action == BucketSyncAction::Upload || op.action == BucketSyncAction::Download
            })
            .filter_map(|op| op.size)
            .sum()
    }
}

const SYNC_TIME_WINDOW_MS: f64 = 1000.0;

fn validate_params(params: &BucketSyncParams) -> HFResult<()> {
    if params.ignore_times && params.ignore_sizes {
        return Err(HFError::InvalidParameter(
            "cannot use both --ignore-times and --ignore-sizes".to_string(),
        ));
    }
    if params.existing && params.ignore_existing {
        return Err(HFError::InvalidParameter(
            "cannot use both --existing and --ignore-existing".to_string(),
        ));
    }
    if params.direction == BucketSyncDirection::Upload && !params.local_path.is_dir() {
        return Err(HFError::InvalidParameter(format!(
            "local path must be an existing directory for upload: {}",
            params.local_path.display()
        )));
    }
    Ok(())
}

fn compile_patterns(patterns: &[String]) -> HFResult<Vec<GlobMatcher>> {
    patterns
        .iter()
        .map(|p| {
            Glob::new(p)
                .map(|g| g.compile_matcher())
                .map_err(|e| HFError::InvalidParameter(format!("invalid glob pattern '{p}': {e}")))
        })
        .collect()
}

fn matches_filters(path: &str, include: &[GlobMatcher], exclude: &[GlobMatcher]) -> bool {
    for pat in exclude {
        if pat.is_match(path) {
            return false;
        }
    }
    if include.is_empty() {
        return true;
    }
    for pat in include {
        if pat.is_match(path) {
            return true;
        }
    }
    false
}

fn list_local_files(root: &Path) -> HFResult<HashMap<String, (u64, f64)>> {
    let mut result = HashMap::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)?;
        for entry in entries {
            let entry = entry?;
            let ft = entry.file_type()?;
            if ft.is_dir() {
                stack.push(entry.path());
            } else if ft.is_file() {
                let path = entry.path();
                let rel = path.strip_prefix(root).map_err(|e| {
                    HFError::InvalidParameter(format!("failed to strip prefix: {e}"))
                })?;
                let rel_str = rel
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                let metadata = std::fs::metadata(&path)?;
                let size = metadata.len();
                let mtime_ms = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as f64)
                    .unwrap_or(0.0);
                result.insert(rel_str, (size, mtime_ms));
            }
        }
    }

    Ok(result)
}

fn strip_prefix<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    if prefix.is_empty() {
        return Some(path);
    }
    let with_slash = if prefix.ends_with('/') {
        prefix.to_string()
    } else {
        format!("{prefix}/")
    };
    path.strip_prefix(&with_slash)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let m = if m <= 2 { m + 9 } else { m - 3 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * m as u64 + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

fn parse_iso_mtime(s: &str) -> f64 {
    // Expected format: "2024-01-15T10:30:00.123Z" or "2024-01-15T10:30:00Z"
    // or "2024-01-15T10:30:00+00:00" etc.
    let s = s.trim();
    if s.len() < 19 {
        return 0.0;
    }

    let parse = || -> Option<f64> {
        let year: i64 = s[0..4].parse().ok()?;
        if s.as_bytes().get(4)? != &b'-' {
            return None;
        }
        let month: i64 = s[5..7].parse().ok()?;
        if s.as_bytes().get(7)? != &b'-' {
            return None;
        }
        let day: i64 = s[8..10].parse().ok()?;
        if s.as_bytes().get(10)? != &b'T' {
            return None;
        }
        let hour: i64 = s[11..13].parse().ok()?;
        if s.as_bytes().get(13)? != &b':' {
            return None;
        }
        let minute: i64 = s[14..16].parse().ok()?;
        if s.as_bytes().get(16)? != &b':' {
            return None;
        }
        let second: i64 = s[17..19].parse().ok()?;

        let mut frac_ms: f64 = 0.0;
        let mut rest_start = 19;
        if s.as_bytes().get(19) == Some(&b'.') {
            let frac_start = 20;
            let mut frac_end = frac_start;
            while frac_end < s.len() && s.as_bytes()[frac_end].is_ascii_digit() {
                frac_end += 1;
            }
            if frac_end > frac_start {
                let frac_str = &s[frac_start..frac_end];
                let frac_val: f64 = frac_str.parse().ok()?;
                let divisor = 10f64.powi(frac_str.len() as i32);
                frac_ms = (frac_val / divisor) * 1000.0;
            }
            rest_start = frac_end;
        }

        // We ignore timezone offset and assume UTC
        let _ = rest_start;

        let days = days_from_civil(year, month, day);
        let total_ms = days as f64 * 86_400_000.0
            + hour as f64 * 3_600_000.0
            + minute as f64 * 60_000.0
            + second as f64 * 1000.0
            + frac_ms;

        Some(total_ms)
    };

    parse().unwrap_or(0.0)
}

#[derive(Debug, Clone, Copy)]
enum CompareRole {
    Upload,
    Download,
}

fn compare_files(
    path: String,
    role: CompareRole,
    source_size: u64,
    source_mtime: f64,
    dest_size: u64,
    dest_mtime: f64,
    params: &BucketSyncParams,
) -> Option<BucketSyncOperation> {
    let action = match role {
        CompareRole::Upload => BucketSyncAction::Upload,
        CompareRole::Download => BucketSyncAction::Download,
    };

    let source_newer = (source_mtime - dest_mtime) > SYNC_TIME_WINDOW_MS;
    let dest_newer = (dest_mtime - source_mtime) > SYNC_TIME_WINDOW_MS;
    let size_differs = source_size != dest_size;

    let source_label = match role {
        CompareRole::Upload => "local newer",
        CompareRole::Download => "remote newer",
    };
    let dest_newer_label = match role {
        CompareRole::Upload => "remote newer",
        CompareRole::Download => "local newer",
    };

    if params.ignore_existing {
        if params.verbose {
            return Some(BucketSyncOperation {
                action: BucketSyncAction::Skip,
                path,
                size: Some(source_size),
                reason: "exists on receiver (--ignore-existing)".to_string(),
            });
        }
        return None;
    }

    if params.ignore_sizes {
        if source_newer {
            return Some(BucketSyncOperation {
                action,
                path,
                size: Some(source_size),
                reason: source_label.to_string(),
            });
        }
        let reason = if dest_newer {
            dest_newer_label.to_string()
        } else {
            "same mtime".to_string()
        };
        if params.verbose {
            return Some(BucketSyncOperation {
                action: BucketSyncAction::Skip,
                path,
                size: Some(source_size),
                reason,
            });
        }
        return None;
    }

    if params.ignore_times {
        if size_differs {
            return Some(BucketSyncOperation {
                action,
                path,
                size: Some(source_size),
                reason: "size differs".to_string(),
            });
        }
        if params.verbose {
            return Some(BucketSyncOperation {
                action: BucketSyncAction::Skip,
                path,
                size: Some(source_size),
                reason: "same size".to_string(),
            });
        }
        return None;
    }

    if size_differs || source_newer {
        let reason = if size_differs {
            "size differs".to_string()
        } else {
            source_label.to_string()
        };
        return Some(BucketSyncOperation {
            action,
            path,
            size: Some(source_size),
            reason,
        });
    }

    if params.verbose {
        return Some(BucketSyncOperation {
            action: BucketSyncAction::Skip,
            path,
            size: Some(source_size),
            reason: "identical".to_string(),
        });
    }
    None
}

impl HFBucket {
    async fn list_remote_files(
        &self,
        prefix: &Option<String>,
        include: &[GlobMatcher],
        exclude: &[GlobMatcher],
    ) -> HFResult<(
        HashMap<String, (u64, f64)>,
        HashMap<String, BucketTreeEntry>,
    )> {
        let stream = self
            .list_tree()
            .maybe_prefix(prefix.clone())
            .recursive(true)
            .send()?;
        futures::pin_mut!(stream);

        let mut files: HashMap<String, (u64, f64)> = HashMap::new();
        let mut entries: HashMap<String, BucketTreeEntry> = HashMap::new();

        let prefix_str = prefix.as_deref().unwrap_or("");

        while let Some(item) = stream.next().await {
            let entry = item?;
            match &entry {
                BucketTreeEntry::File {
                    path, size, mtime, ..
                } => {
                    let rel = match strip_prefix(path, prefix_str) {
                        Some(r) => r.to_string(),
                        None => continue,
                    };
                    if !matches_filters(&rel, include, exclude) {
                        continue;
                    }
                    let mtime_ms = mtime.as_deref().map(parse_iso_mtime).unwrap_or(0.0);
                    files.insert(rel.clone(), (*size, mtime_ms));
                    entries.insert(rel, entry);
                }
                BucketTreeEntry::Directory { .. } => continue,
            }
        }

        Ok((files, entries))
    }

    fn compute_upload_plan(
        &self,
        local_files: &HashMap<String, (u64, f64)>,
        remote_files: &HashMap<String, (u64, f64)>,
        params: &BucketSyncParams,
    ) -> BucketSyncPlan {
        let mut all_keys = BTreeSet::new();
        all_keys.extend(local_files.keys().cloned());
        all_keys.extend(remote_files.keys().cloned());

        let mut operations = Vec::new();

        for key in &all_keys {
            let in_local = local_files.contains_key(key);
            let in_remote = remote_files.contains_key(key);

            if in_local && !in_remote {
                if params.existing {
                    if params.verbose {
                        operations.push(BucketSyncOperation {
                            action: BucketSyncAction::Skip,
                            path: key.clone(),
                            size: local_files.get(key).map(|(s, _)| *s),
                            reason: "not on receiver (--existing)".to_string(),
                        });
                    }
                } else {
                    let (size, _) = local_files[key];
                    operations.push(BucketSyncOperation {
                        action: BucketSyncAction::Upload,
                        path: key.clone(),
                        size: Some(size),
                        reason: "new file".to_string(),
                    });
                }
            } else if in_local && in_remote {
                let (local_size, local_mtime) = local_files[key];
                let (remote_size, remote_mtime) = remote_files[key];
                if let Some(op) = compare_files(
                    key.clone(),
                    CompareRole::Upload,
                    local_size,
                    local_mtime,
                    remote_size,
                    remote_mtime,
                    params,
                ) {
                    operations.push(op);
                }
            } else if !in_local && in_remote && params.delete {
                let (size, _) = remote_files[key];
                operations.push(BucketSyncOperation {
                    action: BucketSyncAction::Delete,
                    path: key.clone(),
                    size: Some(size),
                    reason: "not in source (--delete)".to_string(),
                });
            }
        }

        BucketSyncPlan {
            direction: BucketSyncDirection::Upload,
            operations,
            download_entries: HashMap::new(),
        }
    }

    fn compute_download_plan(
        &self,
        local_files: &HashMap<String, (u64, f64)>,
        remote_files: &HashMap<String, (u64, f64)>,
        remote_entries: &HashMap<String, BucketTreeEntry>,
        params: &BucketSyncParams,
    ) -> BucketSyncPlan {
        let mut all_keys = BTreeSet::new();
        all_keys.extend(local_files.keys().cloned());
        all_keys.extend(remote_files.keys().cloned());

        let mut operations = Vec::new();
        let mut download_entries = HashMap::new();

        for key in &all_keys {
            let in_local = local_files.contains_key(key);
            let in_remote = remote_files.contains_key(key);

            if in_remote && !in_local {
                if params.existing {
                    if params.verbose {
                        operations.push(BucketSyncOperation {
                            action: BucketSyncAction::Skip,
                            path: key.clone(),
                            size: remote_files.get(key).map(|(s, _)| *s),
                            reason: "not on receiver (--existing)".to_string(),
                        });
                    }
                } else {
                    let (size, _) = remote_files[key];
                    operations.push(BucketSyncOperation {
                        action: BucketSyncAction::Download,
                        path: key.clone(),
                        size: Some(size),
                        reason: "new file".to_string(),
                    });
                    if let Some(entry) = remote_entries.get(key) {
                        download_entries.insert(key.clone(), entry.clone());
                    }
                }
            } else if in_local && in_remote {
                let (remote_size, remote_mtime) = remote_files[key];
                let (local_size, local_mtime) = local_files[key];
                if let Some(op) = compare_files(
                    key.clone(),
                    CompareRole::Download,
                    remote_size,
                    remote_mtime,
                    local_size,
                    local_mtime,
                    params,
                ) {
                    if op.action == BucketSyncAction::Download
                        && let Some(entry) = remote_entries.get(key)
                    {
                        download_entries.insert(key.clone(), entry.clone());
                    }
                    operations.push(op);
                }
            } else if in_local && !in_remote && params.delete {
                let (size, _) = local_files[key];
                operations.push(BucketSyncOperation {
                    action: BucketSyncAction::Delete,
                    path: key.clone(),
                    size: Some(size),
                    reason: "not in source (--delete)".to_string(),
                });
            }
        }

        BucketSyncPlan {
            direction: BucketSyncDirection::Download,
            operations,
            download_entries,
        }
    }

    async fn execute_upload_plan(
        &self,
        plan: &BucketSyncPlan,
        params: &BucketSyncParams,
    ) -> HFResult<()> {
        let upload_files: Vec<BucketUpload> = plan
            .operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Upload)
            .map(|op| {
                let local_path = params.local_path.join(&op.path);
                let remote_path = match &params.prefix {
                    Some(prefix) => format!("{prefix}/{}", op.path),
                    None => op.path.clone(),
                };
                BucketUpload::new(local_path, remote_path)
            })
            .collect();

        let delete_paths: Vec<String> = plan
            .operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Delete)
            .map(|op| match &params.prefix {
                Some(prefix) => format!("{prefix}/{}", op.path),
                None => op.path.clone(),
            })
            .collect();

        if !upload_files.is_empty() {
            self.upload_files()
                .files(upload_files)
                .maybe_progress(params.progress.clone())
                .send()
                .await?;
        }
        if !delete_paths.is_empty() {
            self.delete_files().paths(delete_paths).send().await?;
        }

        Ok(())
    }

    async fn execute_download_plan(
        &self,
        plan: &BucketSyncPlan,
        params: &BucketSyncParams,
    ) -> HFResult<()> {
        let mut xet_batch_files = Vec::new();
        let mut total_bytes: u64 = 0;

        for op in plan
            .operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Download)
        {
            let entry =
                plan.download_entries
                    .get(&op.path)
                    .ok_or_else(|| HFError::EntryNotFound {
                        path: op.path.clone(),
                        repo_id: self.bucket_id(),
                        context: None,
                    })?;

            let local_path = params.local_path.join(&op.path);
            if let Some(parent) = local_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            match entry {
                BucketTreeEntry::File { xet_hash, size, .. } => {
                    let remote_path = match &params.prefix {
                        Some(prefix) => format!("{prefix}/{}", op.path),
                        None => op.path.clone(),
                    };
                    total_bytes += size;
                    xet_batch_files.push(crate::xet::XetBatchFile {
                        hash: xet_hash.clone(),
                        file_size: *size,
                        path: local_path,
                        filename: remote_path,
                    });
                }
                BucketTreeEntry::Directory { path, .. } => {
                    return Err(HFError::InvalidParameter(format!(
                        "Cannot download directory entry: {path}"
                    )));
                }
            }
        }

        if !xet_batch_files.is_empty() {
            params.progress.emit(DownloadEvent::Start {
                total_files: xet_batch_files.len(),
                total_bytes,
            });
            self.xet_download_batch(&xet_batch_files, &params.progress)
                .await?;
            params.progress.emit(DownloadEvent::Complete);
        }

        let delete_paths: Vec<&str> = plan
            .operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Delete)
            .map(|op| op.path.as_str())
            .collect();

        for rel_path in &delete_paths {
            let local_path = params.local_path.join(rel_path);
            if local_path.is_file() {
                std::fs::remove_file(&local_path)?;
            }
        }

        // Clean up empty parent directories (leaf-first)
        let mut dirs_to_check: BTreeSet<PathBuf> = BTreeSet::new();
        for rel_path in &delete_paths {
            let local_path = params.local_path.join(rel_path);
            let mut parent = local_path.parent().map(Path::to_path_buf);
            while let Some(p) = parent {
                if p == params.local_path || !p.starts_with(&params.local_path) {
                    break;
                }
                dirs_to_check.insert(p.clone());
                parent = p.parent().map(Path::to_path_buf);
            }
        }

        let mut dirs_sorted: Vec<PathBuf> = dirs_to_check.into_iter().collect();
        dirs_sorted.sort_by_key(|b| std::cmp::Reverse(b.components().count()));
        for dir in dirs_sorted {
            if dir.is_dir() {
                let is_empty = std::fs::read_dir(&dir)
                    .map(|mut d| d.next().is_none())
                    .unwrap_or(false);
                if is_empty {
                    let _ = std::fs::remove_dir(&dir);
                }
            }
        }

        Ok(())
    }

    async fn sync_impl(&self, params: BucketSyncParams) -> HFResult<BucketSyncPlan> {
        validate_params(&params)?;

        let include = compile_patterns(&params.include)?;
        let exclude = compile_patterns(&params.exclude)?;

        let (remote_files, remote_entries) = self
            .list_remote_files(&params.prefix, &include, &exclude)
            .await?;

        match params.direction {
            BucketSyncDirection::Upload => {
                let all_local = list_local_files(&params.local_path)?;
                let local_files: HashMap<String, (u64, f64)> = all_local
                    .into_iter()
                    .filter(|(k, _)| matches_filters(k, &include, &exclude))
                    .collect();

                let plan = self.compute_upload_plan(&local_files, &remote_files, &params);
                self.execute_upload_plan(&plan, &params).await?;
                Ok(plan)
            }
            BucketSyncDirection::Download => {
                std::fs::create_dir_all(&params.local_path)?;

                let all_local = list_local_files(&params.local_path)?;
                let local_files: HashMap<String, (u64, f64)> = all_local
                    .into_iter()
                    .filter(|(k, _)| matches_filters(k, &include, &exclude))
                    .collect();

                let plan = self.compute_download_plan(
                    &local_files,
                    &remote_files,
                    &remote_entries,
                    &params,
                );
                self.execute_download_plan(&plan, &params).await?;
                Ok(plan)
            }
        }
    }
}

#[bon]
impl HFBucket {
    /// Synchronize a local directory with this bucket.
    ///
    /// The sync is one-way and controlled by `direction`:
    ///
    /// - [`BucketSyncDirection::Upload`] compares `local_path` against the bucket (optionally scoped by `prefix`) and
    ///   uploads changed or missing files.
    /// - [`BucketSyncDirection::Download`] compares the bucket against `local_path` and downloads changed or missing
    ///   files.
    ///
    /// When `delete` is enabled, files that only exist on the receiving side are removed as part
    /// of the sync. The returned [`BucketSyncPlan`] describes the operations that were executed;
    /// set `verbose(true)` if you also want explicit `Skip` entries explaining why untouched files
    /// were not transferred.
    ///
    /// # Comparison behavior
    ///
    /// By default, files are considered different when their sizes differ or when the source file
    /// is newer than the destination file. Modification times are compared with a small tolerance
    /// to avoid unnecessary transfers caused by filesystem timestamp precision. `ignore_times`
    /// switches the comparison to size-only; `ignore_sizes` switches it to mtime-only. `existing`
    /// and `ignore_existing` further limit which files are eligible to transfer.
    ///
    /// # Parameters
    ///
    /// - `local_path` (required): local directory path.
    /// - `direction` (required): sync direction (upload or download).
    /// - `prefix`: optional prefix within the bucket (subdirectory).
    /// - `delete`: delete destination files not present in source.
    /// - `ignore_times`: only compare sizes, ignore modification times.
    /// - `ignore_sizes`: only compare modification times, ignore sizes.
    /// - `existing`: only sync files that already exist at destination.
    /// - `ignore_existing`: skip files that already exist at destination.
    /// - `include`: glob-style include patterns.
    /// - `exclude`: glob-style exclude patterns. Exclusions take precedence over inclusions.
    /// - `verbose`: include skip operations in the returned plan.
    /// - `progress`: progress handler for upload/download tracking.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn sync(
        &self,
        /// Local directory path.
        #[builder(into)]
        local_path: PathBuf,
        /// Sync direction (upload or download).
        direction: BucketSyncDirection,
        /// Prefix within the bucket (subdirectory).
        #[builder(into)]
        prefix: Option<String>,
        /// Delete destination files not present in source.
        #[builder(default)]
        delete: bool,
        /// Only compare sizes, ignore modification times.
        #[builder(default)]
        ignore_times: bool,
        /// Only compare modification times, ignore sizes.
        #[builder(default)]
        ignore_sizes: bool,
        /// Only sync files that already exist at destination.
        #[builder(default)]
        existing: bool,
        /// Skip files that already exist at destination.
        #[builder(default)]
        ignore_existing: bool,
        /// Glob-style include patterns.
        #[builder(default)]
        include: Vec<String>,
        /// Glob-style exclude patterns. Exclusions take precedence over inclusions.
        #[builder(default)]
        exclude: Vec<String>,
        /// Include skip operations in the returned plan.
        #[builder(default)]
        verbose: bool,
        /// Progress handler for upload/download tracking.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<BucketSyncPlan> {
        Box::pin(self.sync_impl(BucketSyncParams {
            local_path,
            direction,
            prefix,
            delete,
            ignore_times,
            ignore_sizes,
            existing,
            ignore_existing,
            include,
            exclude,
            verbose,
            progress,
        }))
        .await
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFBucketSync {
    /// Blocking counterpart of [`HFBucket::sync`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn sync(
        &self,
        #[builder(into)] local_path: PathBuf,
        direction: BucketSyncDirection,
        #[builder(into)] prefix: Option<String>,
        #[builder(default)] delete: bool,
        #[builder(default)] ignore_times: bool,
        #[builder(default)] ignore_sizes: bool,
        #[builder(default)] existing: bool,
        #[builder(default)] ignore_existing: bool,
        #[builder(default)] include: Vec<String>,
        #[builder(default)] exclude: Vec<String>,
        #[builder(default)] verbose: bool,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<BucketSyncPlan> {
        self.runtime.block_on(
            self.inner
                .sync()
                .local_path(local_path)
                .direction(direction)
                .maybe_prefix(prefix)
                .delete(delete)
                .ignore_times(ignore_times)
                .ignore_sizes(ignore_sizes)
                .existing(existing)
                .ignore_existing(ignore_existing)
                .include(include)
                .exclude(exclude)
                .verbose(verbose)
                .maybe_progress(progress)
                .send(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(direction: BucketSyncDirection) -> BucketSyncParams {
        BucketSyncParams {
            local_path: PathBuf::from("/tmp"),
            direction,
            prefix: None,
            delete: false,
            ignore_times: false,
            ignore_sizes: false,
            existing: false,
            ignore_existing: false,
            include: Vec::new(),
            exclude: Vec::new(),
            verbose: false,
            progress: None,
        }
    }

    fn make_plan(ops: Vec<(BucketSyncAction, Option<u64>)>) -> BucketSyncPlan {
        BucketSyncPlan {
            direction: BucketSyncDirection::Upload,
            operations: ops
                .into_iter()
                .enumerate()
                .map(|(i, (action, size))| BucketSyncOperation {
                    action,
                    path: format!("file_{i}.txt"),
                    size,
                    reason: "test".to_string(),
                })
                .collect(),
            download_entries: HashMap::new(),
        }
    }

    #[test]
    fn test_plan_counts() {
        let plan = make_plan(vec![
            (BucketSyncAction::Upload, Some(100)),
            (BucketSyncAction::Upload, Some(200)),
            (BucketSyncAction::Download, Some(300)),
            (BucketSyncAction::Delete, Some(50)),
            (BucketSyncAction::Skip, None),
        ]);
        assert_eq!(plan.uploads(), 2);
        assert_eq!(plan.downloads(), 1);
        assert_eq!(plan.deletes(), 1);
        assert_eq!(plan.skips(), 1);
    }

    #[test]
    fn test_transfer_bytes() {
        let plan = make_plan(vec![
            (BucketSyncAction::Upload, Some(100)),
            (BucketSyncAction::Download, Some(300)),
            (BucketSyncAction::Delete, Some(50)),
            (BucketSyncAction::Skip, None),
        ]);
        assert_eq!(plan.transfer_bytes(), 400);
    }

    #[test]
    fn test_empty_plan() {
        let plan = make_plan(vec![]);
        assert_eq!(plan.uploads(), 0);
        assert_eq!(plan.downloads(), 0);
        assert_eq!(plan.deletes(), 0);
        assert_eq!(plan.skips(), 0);
        assert_eq!(plan.transfer_bytes(), 0);
    }

    #[test]
    fn test_parse_iso_mtime_basic() {
        let ms = parse_iso_mtime("2024-01-15T10:30:00Z");
        assert!(ms > 0.0);

        let expected_days = days_from_civil(2024, 1, 15);
        let expected_ms =
            expected_days as f64 * 86_400_000.0 + 10.0 * 3_600_000.0 + 30.0 * 60_000.0;
        assert!((ms - expected_ms).abs() < 1.0);
    }

    #[test]
    fn test_parse_iso_mtime_with_frac() {
        let ms = parse_iso_mtime("2024-01-15T10:30:00.500Z");
        let expected_days = days_from_civil(2024, 1, 15);
        let expected_ms =
            expected_days as f64 * 86_400_000.0 + 10.0 * 3_600_000.0 + 30.0 * 60_000.0 + 500.0;
        assert!((ms - expected_ms).abs() < 1.0);
    }

    #[test]
    fn test_parse_iso_mtime_invalid() {
        assert_eq!(parse_iso_mtime(""), 0.0);
        assert_eq!(parse_iso_mtime("not a date"), 0.0);
        assert_eq!(parse_iso_mtime("short"), 0.0);
    }

    #[test]
    fn test_days_from_civil_epoch() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
    }

    #[test]
    fn test_days_from_civil_known_date() {
        // 2024-01-01 is 19723 days after epoch
        assert_eq!(days_from_civil(2024, 1, 1), 19723);
    }

    #[test]
    fn test_strip_prefix_empty() {
        assert_eq!(strip_prefix("foo/bar.txt", ""), Some("foo/bar.txt"));
    }

    #[test]
    fn test_strip_prefix_match() {
        assert_eq!(
            strip_prefix("data/foo/bar.txt", "data"),
            Some("foo/bar.txt")
        );
        assert_eq!(
            strip_prefix("data/foo/bar.txt", "data/"),
            Some("foo/bar.txt")
        );
    }

    #[test]
    fn test_strip_prefix_no_match() {
        assert_eq!(strip_prefix("other/bar.txt", "data"), None);
    }

    #[test]
    fn test_strip_prefix_partial_no_match() {
        assert_eq!(strip_prefix("submarine.txt", "sub"), None);
    }

    #[test]
    fn test_strip_prefix_exact_file() {
        assert_eq!(strip_prefix("data/file.txt", "data"), Some("file.txt"));
    }

    #[test]
    fn test_matches_filters_no_patterns() {
        assert!(matches_filters("foo.txt", &[], &[]));
    }

    #[test]
    fn test_matches_filters_include() {
        let include = compile_patterns(&["*.txt".to_string()]).unwrap();
        assert!(matches_filters("foo.txt", &include, &[]));
        assert!(!matches_filters("foo.bin", &include, &[]));
    }

    #[test]
    fn test_matches_filters_exclude() {
        let exclude = compile_patterns(&["*.log".to_string()]).unwrap();
        assert!(matches_filters("foo.txt", &[], &exclude));
        assert!(!matches_filters("foo.log", &[], &exclude));
    }

    #[test]
    fn test_matches_filters_exclude_takes_priority() {
        let include = compile_patterns(&["*.txt".to_string()]).unwrap();
        let exclude = compile_patterns(&["secret*".to_string()]).unwrap();
        assert!(matches_filters("foo.txt", &include, &exclude));
        assert!(!matches_filters("secret.txt", &include, &exclude));
    }

    #[test]
    fn test_validate_params_conflicting_times_sizes() {
        let params = BucketSyncParams {
            ignore_times: true,
            ignore_sizes: true,
            ..make_params(BucketSyncDirection::Download)
        };
        assert!(validate_params(&params).is_err());
    }

    #[test]
    fn test_validate_params_conflicting_existing() {
        let params = BucketSyncParams {
            existing: true,
            ignore_existing: true,
            ..make_params(BucketSyncDirection::Download)
        };
        assert!(validate_params(&params).is_err());
    }

    #[test]
    fn test_compare_files_identical() {
        let params = BucketSyncParams {
            verbose: true,
            ..make_params(BucketSyncDirection::Upload)
        };

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            100,
            5000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Skip);
        assert_eq!(op.reason, "identical");
    }

    #[test]
    fn test_compare_files_identical_not_verbose() {
        let params = make_params(BucketSyncDirection::Upload);

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            100,
            5000.0,
            100,
            5000.0,
            &params,
        );
        assert!(op.is_none());
    }

    #[test]
    fn test_compare_files_size_differs() {
        let params = make_params(BucketSyncDirection::Upload);

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            200,
            5000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Upload);
        assert_eq!(op.reason, "size differs");
    }

    #[test]
    fn test_compare_files_source_newer() {
        let params = make_params(BucketSyncDirection::Upload);

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            100,
            7000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Upload);
        assert_eq!(op.reason, "local newer");
    }

    #[test]
    fn test_compare_files_download_source_newer() {
        let params = make_params(BucketSyncDirection::Download);

        let op = compare_files(
            String::new(),
            CompareRole::Download,
            100,
            7000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Download);
        assert_eq!(op.reason, "remote newer");
    }

    #[test]
    fn test_compare_files_within_safety_window() {
        let params = BucketSyncParams {
            verbose: true,
            ..make_params(BucketSyncDirection::Upload)
        };

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            100,
            5500.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Skip);
        assert_eq!(op.reason, "identical");
    }

    #[test]
    fn test_compare_files_ignore_times() {
        let params = BucketSyncParams {
            ignore_times: true,
            verbose: true,
            ..make_params(BucketSyncDirection::Upload)
        };

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            100,
            9000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Skip);
        assert_eq!(op.reason, "same size");

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            200,
            5000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Upload);
        assert_eq!(op.reason, "size differs");
    }

    #[test]
    fn test_compare_files_ignore_sizes() {
        let params = BucketSyncParams {
            ignore_sizes: true,
            verbose: true,
            ..make_params(BucketSyncDirection::Upload)
        };

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            200,
            5000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Skip);
        assert_eq!(op.reason, "same mtime");

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            100,
            3000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Skip);
        assert_eq!(op.reason, "remote newer");
    }

    #[test]
    fn test_compare_files_ignore_sizes_download() {
        let params = BucketSyncParams {
            ignore_sizes: true,
            verbose: true,
            ..make_params(BucketSyncDirection::Download)
        };

        let op = compare_files(
            String::new(),
            CompareRole::Download,
            100,
            3000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Skip);
        assert_eq!(op.reason, "local newer");
    }

    #[test]
    fn test_compare_files_ignore_existing() {
        let params = BucketSyncParams {
            ignore_existing: true,
            verbose: true,
            ..make_params(BucketSyncDirection::Upload)
        };

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            200,
            9000.0,
            100,
            5000.0,
            &params,
        )
        .unwrap();
        assert_eq!(op.action, BucketSyncAction::Skip);
        assert_eq!(op.reason, "exists on receiver (--ignore-existing)");
    }

    #[test]
    fn test_compare_files_ignore_existing_not_verbose() {
        let params = BucketSyncParams {
            ignore_existing: true,
            ..make_params(BucketSyncDirection::Upload)
        };

        let op = compare_files(
            String::new(),
            CompareRole::Upload,
            200,
            9000.0,
            100,
            5000.0,
            &params,
        );
        assert!(op.is_none());
    }
}
