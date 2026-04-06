use super::catalog;
use super::local::legacy_models_dir;
use super::resolve::{catalog_hf_asset_ref, catalog_hf_match, catalog_match, parse_hf_resolve_url};
use super::{
    build_hf_api, format_size_bytes, huggingface_hub_cache, huggingface_hub_cache_dir,
    short_revision,
};
use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use hf_hub::api::RepoInfo;
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MigrationStatus {
    Rehydratable,
    LegacyOnly,
}

struct MigrationEntry {
    path: PathBuf,
    status: MigrationStatus,
    detail: String,
    catalog: Option<&'static catalog::CatalogModel>,
}

impl MigrationEntry {
    fn file_name(&self) -> String {
        self.path
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| self.path.display().to_string())
    }
}

struct CachedRepo {
    repo_id: String,
    ref_name: String,
    local_revision: String,
}

#[derive(Default)]
struct MigrationCounts {
    adopted: usize,
    downloaded: usize,
    ambiguous: usize,
    historical: usize,
}

#[derive(Default)]
struct UpdateCounts {
    refreshed: usize,
    missing_meta: usize,
}

enum AdoptionResult {
    Adopted(PathBuf),
    AdoptedHistorical { path: PathBuf, commit_hash: String },
    DownloadRequired(DownloadReason),
}

enum DownloadReason {
    VerificationUnavailable(String),
    AmbiguousLegacyCandidates {
        file: String,
        paths: Vec<PathBuf>,
    },
    SizeMismatch {
        legacy_path: PathBuf,
        legacy_size: u64,
        remote_size: u64,
    },
    ChecksumMismatch {
        legacy_path: PathBuf,
    },
}

#[derive(Deserialize)]
struct RemoteRepoInfo {
    sha: String,
    siblings: Vec<RemoteSibling>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteSibling {
    rfilename: String,
    size: Option<u64>,
    blob_id: Option<String>,
    lfs: Option<RemoteLfsInfo>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteLfsInfo {
    sha256: String,
    size: u64,
}

struct RemoteFileMetadata {
    commit_hash: String,
    size: u64,
    blob_id: String,
    sha256: Option<String>,
}

#[derive(Deserialize)]
struct RepoCommit {
    id: String,
}

struct LegacyCandidates<'a> {
    by_name: BTreeMap<String, Vec<&'a MigrationEntry>>,
    has_flat_layout: bool,
    has_nested_layout: bool,
}

impl DownloadReason {
    fn describe(&self) -> String {
        match self {
            Self::VerificationUnavailable(err) => {
                format!("could not verify remote metadata ({err})")
            }
            Self::AmbiguousLegacyCandidates { file, paths } => format!(
                "multiple legacy candidates found for {file}: {}",
                paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::SizeMismatch {
                legacy_path,
                legacy_size,
                remote_size,
            } => format!(
                "size mismatch for {} (legacy {}, remote {})",
                legacy_path.display(),
                format_size_bytes(*legacy_size),
                format_size_bytes(*remote_size)
            ),
            Self::ChecksumMismatch { legacy_path } => {
                format!("checksum mismatch for {}", legacy_path.display())
            }
        }
    }
}

pub fn run_migrate(apply: bool, prune: bool) -> Result<()> {
    let entries = migration_entries();
    let legacy_dir = legacy_models_dir();
    if entries.is_empty() {
        eprintln!("📦 No legacy GGUF files found");
        eprintln!("   {}", legacy_dir.display());
        return Ok(());
    }

    eprintln!("🧳 Legacy model scan");
    eprintln!("📁 Source: {}", legacy_dir.display());
    eprintln!();
    for entry in &entries {
        let (label, icon) = match entry.status {
            MigrationStatus::Rehydratable => ("Rehydratable", "✅"),
            MigrationStatus::LegacyOnly => ("Legacy-only", "⚠️"),
        };
        eprintln!("{icon} {label}: {}", entry.file_name());
        eprintln!("   path: {}", entry.path.display());
        eprintln!("   info: {}", entry.detail);
    }
    eprintln!();

    let rehydratable = entries
        .iter()
        .filter(|entry| entry.status == MigrationStatus::Rehydratable)
        .count();
    let legacy_only = entries.len() - rehydratable;
    eprintln!("📊 Summary");
    eprintln!("   ✅ rehydratable: {rehydratable}");
    eprintln!("   ⚠️ legacy-only: {legacy_only}");

    if !apply && !prune {
        eprintln!();
        eprintln!("➡️ Next steps");
        if rehydratable > 0 {
            eprintln!("   mesh-llm models migrate --apply");
            eprintln!(
                "   Rehydrate recognized Hugging Face-backed models into {}",
                huggingface_hub_cache_dir().display()
            );
            eprintln!("   mesh-llm models migrate --prune");
            eprintln!("   Remove rehydrated legacy GGUF files after you verify the HF cache copy");
        }
        if legacy_only > 0 {
            eprintln!("   mesh-llm --gguf /path/to/model.gguf");
            eprintln!("   Keep using custom local GGUF files explicitly");
        }
        return Ok(());
    }

    if prune && !apply {
        return run_prune(&entries);
    }

    let api = build_hf_api(true)?;
    let mut migrated = 0usize;
    let mut totals = MigrationCounts::default();
    let mut grouped = BTreeMap::<String, Vec<&MigrationEntry>>::new();
    eprintln!("🚚 Migrating recognized models");
    eprintln!("📁 Destination: {}", huggingface_hub_cache_dir().display());
    eprintln!();
    for entry in entries
        .iter()
        .filter(|entry| entry.status == MigrationStatus::Rehydratable)
    {
        let Some(model) = entry.catalog else {
            continue;
        };
        grouped.entry(model.name.clone()).or_default().push(entry);
    }
    let total_groups = grouped.len();
    for (index, grouped_entries) in grouped.values().enumerate() {
        let Some(model) = grouped_entries.first().and_then(|entry| entry.catalog) else {
            continue;
        };
        eprintln!("🧭 [{}/{}] {}", index + 1, total_groups, model.name);
        let counts = migrate_catalog_model(&api, model, grouped_entries)?;
        totals.adopted += counts.adopted;
        totals.downloaded += counts.downloaded;
        totals.ambiguous += counts.ambiguous;
        totals.historical += counts.historical;
        migrated += 1;
    }

    eprintln!();
    eprintln!("✅ Migration complete");
    eprintln!("   model groups migrated: {migrated}");
    eprintln!("   adopted local files: {}", totals.adopted);
    eprintln!("   historical checksum matches: {}", totals.historical);
    eprintln!("   ambiguous legacy files: {}", totals.ambiguous);
    eprintln!("   downloaded files: {}", totals.downloaded);
    eprintln!("   destination: {}", huggingface_hub_cache_dir().display());
    eprintln!("   legacy files were left in place");
    eprintln!("   next: mesh-llm models migrate --prune");
    eprintln!("   custom local GGUFs still work via `mesh-llm --gguf /path/to/model.gguf`");
    Ok(())
}

pub fn run_update(repo: Option<&str>, all: bool, check: bool) -> Result<()> {
    let api = build_hf_api(!check)?;
    let repos = cached_repos()?;
    if repos.is_empty() {
        eprintln!("📦 No cached Hugging Face model repos found");
        eprintln!("   {}", huggingface_hub_cache_dir().display());
        return Ok(());
    }

    let selected: Vec<CachedRepo> = if check {
        if all {
            repos
        } else if let Some(repo_id) = repo {
            let repo_id = repo_id.trim();
            let Some(found) = repos.into_iter().find(|entry| entry.repo_id == repo_id) else {
                anyhow::bail!("Cached repo not found: {repo_id}");
            };
            vec![found]
        } else {
            repos
        }
    } else if all {
        repos
    } else {
        let Some(repo_id) = repo else {
            anyhow::bail!("Pass a repo id or --all. Use `mesh-llm models updates --check` to inspect updates without downloading.");
        };
        let repo_id = repo_id.trim();
        let Some(found) = repos.into_iter().find(|entry| entry.repo_id == repo_id) else {
            anyhow::bail!("Cached repo not found: {repo_id}");
        };
        vec![found]
    };

    if !check {
        eprintln!("🔄 Updating cached Hugging Face repos");
        eprintln!("📁 Cache: {}", huggingface_hub_cache_dir().display());
        eprintln!("📦 Selected: {}", selected.len());
        eprintln!();
    }
    let mut updates = 0usize;
    let total_selected = selected.len();
    let mut refresh_totals = UpdateCounts::default();
    for (index, repo) in selected.into_iter().enumerate() {
        if check {
            print_update_check_progress(index + 1, total_selected, &repo.repo_id)?;
            if let Some(remote_revision) = check_repo_update(&api, &repo)? {
                updates += 1;
                clear_progress_line()?;
                eprintln!("🆕 [{}/{}] {}", index + 1, total_selected, repo.repo_id);
                eprintln!("   ref: {}", repo.ref_name);
                eprintln!("   local: {}", short_revision(&repo.local_revision));
                eprintln!("   latest: {}", short_revision(&remote_revision));
                eprintln!("   update: mesh-llm models updates {}", repo.repo_id);
            }
        } else {
            eprintln!("🧭 [{}/{}] {}", index + 1, total_selected, repo.repo_id);
            let counts = update_cached_repo(&api, &repo)?;
            refresh_totals.refreshed += counts.refreshed;
            refresh_totals.missing_meta += counts.missing_meta;
        }
    }
    if check {
        clear_progress_line()?;
        if updates > 0 {
            eprintln!("📬 Update summary");
            eprintln!("   repos with updates: {updates}");
            eprintln!("   update one: mesh-llm models updates <repo>");
            eprintln!("   update all: mesh-llm models updates --all");
        }
    } else {
        eprintln!();
        eprintln!("✅ Update complete");
        eprintln!("   refreshed files: {}", refresh_totals.refreshed);
        if refresh_totals.missing_meta > 0 {
            eprintln!("   missing config.json: {}", refresh_totals.missing_meta);
        }
    }
    Ok(())
}

pub fn warn_about_updates_for_paths(paths: &[PathBuf]) {
    let mut cache_models = Vec::new();
    let mut seen = BTreeSet::new();
    for path in paths {
        let Some(repo) = (match cached_repo_for_path(path) {
            Ok(repo) => repo,
            Err(err) => {
                eprintln!(
                    "Warning: could not inspect cached Hugging Face repo for {}: {err}",
                    path.display()
                );
                continue;
            }
        }) else {
            continue;
        };
        if seen.insert((repo.repo_id.clone(), repo.local_revision.clone())) {
            cache_models.push(repo);
        }
    }
    if cache_models.is_empty() {
        return;
    }

    let api = match build_hf_api(false) {
        Ok(api) => api,
        Err(err) => {
            eprintln!("Warning: could not initialize Hugging Face update checks: {err}");
            return;
        }
    };
    for repo in cache_models {
        match check_repo_update(&api, &repo) {
            Ok(Some(remote_revision)) => {
                eprintln!("🆕 Update available for {}", repo.repo_id);
                eprintln!("   local: {}", short_revision(&repo.local_revision));
                eprintln!("   latest: {}", short_revision(&remote_revision));
                eprintln!("   continuing with pinned local snapshot");
                eprintln!("   update: mesh-llm models updates {}", repo.repo_id);
            }
            Ok(None) => {}
            Err(err) => {
                eprintln!(
                    "Warning: could not check for updates for {}: {err}",
                    repo.repo_id
                );
            }
        }
    }
}

fn migration_entries() -> Vec<MigrationEntry> {
    let mut paths = legacy_gguf_files(legacy_models_dir());
    paths.sort();
    paths.into_iter().map(classify_legacy_path).collect()
}

fn legacy_gguf_files(root: PathBuf) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !root.exists() {
        return files;
    }
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
                files.push(path);
            }
        }
    }
    files
}

fn classify_legacy_path(path: PathBuf) -> MigrationEntry {
    if let Some((model, repo, revision, file)) = catalog_hf_match(&path) {
        let detail = match revision {
            Some(revision) => format!(
                "recognized as {} from Hugging Face repo {} at revision {} ({})",
                model.name, repo, revision, file
            ),
            None => format!(
                "recognized as {} from Hugging Face repo {} ({})",
                model.name, repo, file
            ),
        };
        return MigrationEntry {
            path,
            status: MigrationStatus::Rehydratable,
            detail,
            catalog: Some(model),
        };
    }

    if let Some(model) = catalog_match(&path) {
        return MigrationEntry {
            path,
            status: MigrationStatus::LegacyOnly,
            detail: format!(
                "recognized as {}, but its catalog source is not a Hugging Face repo",
                model.name
            ),
            catalog: None,
        };
    }

    MigrationEntry {
        path,
        status: MigrationStatus::LegacyOnly,
        detail: "no canonical Hugging Face source is known for this GGUF".to_string(),
        catalog: None,
    }
}

fn migrate_catalog_model(
    api: &Api,
    model: &catalog::CatalogModel,
    entries: &[&MigrationEntry],
) -> Result<MigrationCounts> {
    let mut counts = MigrationCounts::default();
    let mut adopted_files = Vec::new();
    let mut historical_files = Vec::new();
    let mut downloaded_files = Vec::new();
    let mut ambiguous_files = Vec::new();
    let mut split_downloads = Vec::new();
    let legacy_root = legacy_models_dir();
    let legacy_files = collect_legacy_candidates(entries, &legacy_root);
    let duplicate_legacy_files = duplicate_legacy_files(&legacy_files);
    let expected_split_files = expected_split_gguf_files(model);
    let present_split_files: Vec<String> = expected_split_files
        .iter()
        .filter(|file| legacy_files.by_name.contains_key(&file.to_lowercase()))
        .cloned()
        .collect();
    let missing_split_files: Vec<String> = expected_split_files
        .iter()
        .filter(|file| !legacy_files.by_name.contains_key(&file.to_lowercase()))
        .cloned()
        .collect();

    if legacy_files.has_flat_layout && legacy_files.has_nested_layout {
        eprintln!(
            "   ⚠️ mixed legacy layout detected; both flat ~/.models files and nested subdirectories exist for this model"
        );
    }
    if !duplicate_legacy_files.is_empty() {
        eprintln!(
            "   ⚠️ duplicate legacy basenames detected; automatic adoption will be skipped for ambiguous files"
        );
        for (file, paths) in &duplicate_legacy_files {
            eprintln!("      {file}:");
            for path in paths {
                eprintln!("         {}", path.display());
            }
        }
    }

    if !expected_split_files.is_empty() {
        eprintln!(
            "   📦 legacy split parts present: {}/{}",
            present_split_files.len(),
            expected_split_files.len()
        );
        if !present_split_files.is_empty() && !missing_split_files.is_empty() {
            eprintln!(
                "   ⚠️ partial split set detected; missing legacy parts: {}",
                missing_split_files.join(", ")
            );
        }
    }

    let mut config_downloaded = BTreeSet::new();
    for url in model
        .extra_files
        .iter()
        .map(|asset| asset.url.as_str())
        .chain(std::iter::once(model.url.as_str()))
        .chain(model.mmproj.iter().map(|asset| asset.url.as_str()))
    {
        let Some((repo_id, revision, file)) = parse_hf_resolve_url(url) else {
            continue;
        };
        let repo = Repo::with_revision(
            repo_id.clone(),
            RepoType::Model,
            revision.clone().unwrap_or_else(|| "main".to_string()),
        );
        let api_repo = api.repo(repo.clone());
        match legacy_candidate_for_file(&legacy_files, &file) {
            LegacyCandidateSelection::Unique(legacy_path) => {
                match adopt_legacy_asset_into_hf_cache(api, &repo, &file, legacy_path)? {
                    AdoptionResult::Adopted(path) => {
                        eprintln!("   🔁 adopted {}", path.display());
                        counts.adopted += 1;
                        adopted_files.push(file.clone());
                    }
                    AdoptionResult::AdoptedHistorical { path, commit_hash } => {
                        eprintln!(
                            "   🔁 adopted {} from historical revision {}",
                            path.display(),
                            short_revision(&commit_hash)
                        );
                        counts.adopted += 1;
                        counts.historical += 1;
                        adopted_files.push(file.clone());
                        historical_files.push(format!("{}@{}", file, short_revision(&commit_hash)));
                    }
                    AdoptionResult::DownloadRequired(reason) => {
                        eprintln!("   ↪️ downloading {file} because {}", reason.describe());
                        if is_split_gguf_file(&file) {
                            split_downloads.push(file.clone());
                            eprintln!(
                                "   ⚠️ split GGUF part {file} did not match the expected Hugging Face asset"
                            );
                        }
                        let path = api_repo
                            .download(&file)
                            .with_context(|| format!("Download {repo_id}/{file}"))?;
                        eprintln!("   ✅ downloaded {}", path.display());
                        counts.downloaded += 1;
                        downloaded_files.push(file.clone());
                    }
                }
            }
            LegacyCandidateSelection::Ambiguous(paths) => {
                let reason = DownloadReason::AmbiguousLegacyCandidates {
                    file: file.clone(),
                    paths,
                };
                eprintln!("   ↪️ downloading {file} because {}", reason.describe());
                if is_split_gguf_file(&file) {
                    split_downloads.push(file.clone());
                }
                counts.ambiguous += 1;
                ambiguous_files.push(file.clone());
                let path = api_repo
                    .download(&file)
                    .with_context(|| format!("Download {repo_id}/{file}"))?;
                eprintln!("   ✅ downloaded {}", path.display());
                counts.downloaded += 1;
                downloaded_files.push(file.clone());
            }
            LegacyCandidateSelection::Missing => {
                if is_split_gguf_file(&file) {
                    eprintln!("   ↪️ downloading missing split GGUF part {file}");
                }
                let path = api_repo
                    .download(&file)
                    .with_context(|| format!("Download {repo_id}/{file}"))?;
                eprintln!("   ✅ downloaded {}", path.display());
                counts.downloaded += 1;
                downloaded_files.push(file.clone());
            }
        }

        if config_downloaded.insert((repo_id.clone(), revision.clone())) {
            match api_repo.get("config.json") {
                Ok(config_path) => eprintln!("   🧾 config {}", config_path.display()),
                Err(err) => {
                    if is_not_found_error(&err.to_string()) {
                        eprintln!("   ℹ️ no config.json published for {repo_id}");
                    } else {
                        eprintln!("   ⚠️ config {repo_id}: {err}");
                    }
                }
            }
        }
    }

    if !split_downloads.is_empty() {
        split_downloads.sort();
        split_downloads.dedup();
        eprintln!(
            "   ⚠️ split migration required downloading {} part(s): {}",
            split_downloads.len(),
            split_downloads.join(", ")
        );
    }

    eprintln!("   📌 model outcome");
    eprintln!("      adopted local files: {}", counts.adopted);
    if !adopted_files.is_empty() {
        eprintln!("      adopted files: {}", adopted_files.join(", "));
    }
    if !historical_files.is_empty() {
        eprintln!(
            "      historical checksum matches adopted: {}",
            historical_files.join(", ")
        );
    }
    eprintln!("      downloaded files: {}", counts.downloaded);
    if !downloaded_files.is_empty() {
        eprintln!(
            "      downloaded file names: {}",
            downloaded_files.join(", ")
        );
    }
    if !ambiguous_files.is_empty() {
        eprintln!(
            "      ambiguous legacy files: {}",
            ambiguous_files.join(", ")
        );
    }
    if !expected_split_files.is_empty() {
        eprintln!(
            "      split parts in legacy storage: {}/{}",
            present_split_files.len(),
            expected_split_files.len()
        );
        if !missing_split_files.is_empty() {
            eprintln!(
                "      missing split parts: {}",
                missing_split_files.join(", ")
            );
        }
    }

    Ok(counts)
}

enum LegacyCandidateSelection<'a> {
    Missing,
    Unique(&'a Path),
    Ambiguous(Vec<PathBuf>),
}

fn collect_legacy_candidates<'a>(
    entries: &[&'a MigrationEntry],
    legacy_root: &Path,
) -> LegacyCandidates<'a> {
    let mut by_name = BTreeMap::<String, Vec<&MigrationEntry>>::new();
    let mut has_flat_layout = false;
    let mut has_nested_layout = false;

    for entry in entries {
        by_name
            .entry(entry.file_name().to_lowercase())
            .or_default()
            .push(entry);
        if is_flat_legacy_path(&entry.path, legacy_root) {
            has_flat_layout = true;
        } else {
            has_nested_layout = true;
        }
    }

    LegacyCandidates {
        by_name,
        has_flat_layout,
        has_nested_layout,
    }
}

fn duplicate_legacy_files(candidates: &LegacyCandidates<'_>) -> Vec<(String, Vec<PathBuf>)> {
    let mut duplicates = Vec::new();
    for (file, entries) in &candidates.by_name {
        if entries.len() <= 1 {
            continue;
        }
        let mut paths: Vec<PathBuf> = entries.iter().map(|entry| entry.path.clone()).collect();
        paths.sort();
        duplicates.push((file.clone(), paths));
    }
    duplicates.sort_by(|left, right| left.0.cmp(&right.0));
    duplicates
}

fn legacy_candidate_for_file<'a>(
    candidates: &LegacyCandidates<'a>,
    file: &str,
) -> LegacyCandidateSelection<'a> {
    match candidates.by_name.get(&file.to_lowercase()) {
        Some(entries) if entries.len() == 1 => LegacyCandidateSelection::Unique(&entries[0].path),
        Some(entries) => LegacyCandidateSelection::Ambiguous(
            entries.iter().map(|entry| entry.path.clone()).collect(),
        ),
        None => LegacyCandidateSelection::Missing,
    }
}

fn is_flat_legacy_path(path: &Path, legacy_root: &Path) -> bool {
    path.parent() == Some(legacy_root)
}

fn run_prune(entries: &[MigrationEntry]) -> Result<()> {
    eprintln!();
    eprintln!("🧹 Pruning migrated legacy files");
    eprintln!("📁 Source: {}", legacy_models_dir().display());
    eprintln!();

    let mut pruned = 0usize;
    let mut kept = 0usize;
    for entry in entries
        .iter()
        .filter(|entry| entry.status == MigrationStatus::Rehydratable)
    {
        if pruneable_legacy_path(entry)? {
            std::fs::remove_file(&entry.path)
                .with_context(|| format!("Remove {}", entry.path.display()))?;
            eprintln!("   🗑️ {}", entry.path.display());
            pruned += 1;
        } else {
            eprintln!("   ⏭️ {}", entry.path.display());
            kept += 1;
        }
    }

    eprintln!();
    eprintln!("✅ Prune complete");
    eprintln!("   removed legacy files: {pruned}");
    eprintln!("   kept legacy files: {kept}");
    Ok(())
}

fn pruneable_legacy_path(entry: &MigrationEntry) -> Result<bool> {
    let Some(model) = entry.catalog else {
        return Ok(false);
    };
    let file_name = entry.file_name();
    let Some((repo_id, revision, cached_file)) = catalog_hf_asset_ref(model, &file_name) else {
        return Ok(false);
    };
    let cache = huggingface_hub_cache();
    let repo = Repo::with_revision(
        repo_id,
        RepoType::Model,
        revision.unwrap_or_else(|| "main".to_string()),
    );
    let cache_repo = cache.repo(repo);
    Ok(cache_repo.get(&cached_file).is_some())
}

fn adopt_legacy_asset_into_hf_cache(
    api: &Api,
    repo: &Repo,
    file: &str,
    legacy_path: &Path,
) -> Result<AdoptionResult> {
    let remote = match remote_file_metadata(api, repo, file) {
        Ok(remote) => remote,
        Err(err) => {
            return Ok(AdoptionResult::DownloadRequired(
                DownloadReason::VerificationUnavailable(err.to_string()),
            ))
        }
    };

    let cache = huggingface_hub_cache();
    let cache_repo = cache.repo(repo.clone());
    let blob_path = cache_repo.blob_path(&remote.blob_id);
    if blob_path.exists() {
        let path = materialize_cached_snapshot_pointer(
            &cache_repo,
            &remote.commit_hash,
            file,
            &blob_path,
        )?;
        return Ok(AdoptionResult::Adopted(path));
    }

    let legacy_size = std::fs::metadata(legacy_path)
        .with_context(|| format!("Read {}", legacy_path.display()))?
        .len();
    if legacy_size != remote.size {
        return Ok(AdoptionResult::DownloadRequired(
            DownloadReason::SizeMismatch {
                legacy_path: legacy_path.to_path_buf(),
                legacy_size,
                remote_size: remote.size,
            },
        ));
    }

    let Some(remote_sha256) = remote.sha256.as_deref() else {
        return Ok(AdoptionResult::DownloadRequired(
            DownloadReason::VerificationUnavailable(
                "remote file is not LFS-backed, so no SHA-256 is available".to_string(),
            ),
        ));
    };

    let display_name = legacy_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(file);
    let digest = sha256_file_hex(legacy_path, display_name, legacy_size)?;
    eprintln!(
        "   ✅ verified {} ({})",
        display_name,
        format_size_bytes(legacy_size)
    );
    if !digest.eq_ignore_ascii_case(remote_sha256) {
        if let Some(historical) = find_historical_remote_match(
            api,
            repo,
            file,
            legacy_size,
            &digest,
            &remote.commit_hash,
        )? {
            let historical_blob_path = cache_repo.blob_path(&historical.blob_id);
            if let Some(parent) = historical_blob_path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Create {}", parent.display()))?;
            }
            link_or_copy_file(legacy_path, &historical_blob_path)?;

            let path = materialize_cached_snapshot_pointer(
                &cache_repo,
                &historical.commit_hash,
                file,
                &historical_blob_path,
            )?;
            return Ok(AdoptionResult::AdoptedHistorical {
                path,
                commit_hash: historical.commit_hash,
            });
        }
        return Ok(AdoptionResult::DownloadRequired(
            DownloadReason::ChecksumMismatch {
                legacy_path: legacy_path.to_path_buf(),
            },
        ));
    }

    if let Some(parent) = blob_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    link_or_copy_file(legacy_path, &blob_path)?;

    let path =
        materialize_cached_snapshot_pointer(&cache_repo, &remote.commit_hash, file, &blob_path)?;
    Ok(AdoptionResult::Adopted(path))
}

fn remote_file_metadata(api: &Api, repo: &Repo, file: &str) -> Result<RemoteFileMetadata> {
    let mut response = api
        .repo(repo.clone())
        .info_request()
        .query("blobs", "true")
        .call()
        .map_err(Box::new)
        .with_context(|| format!("Fetch repo info for {}", repo.url()))?;
    let info: RemoteRepoInfo = response
        .body_mut()
        .read_json()
        .map_err(Box::new)
        .with_context(|| format!("Parse repo info for {}", repo.url()))?;
    let sibling = info
        .siblings
        .into_iter()
        .find(|sibling| sibling.rfilename == file)
        .with_context(|| format!("Remote file metadata missing for {}/{}", repo.url(), file))?;
    let size = sibling
        .lfs
        .as_ref()
        .map(|lfs| lfs.size)
        .or(sibling.size)
        .with_context(|| format!("Remote file size missing for {}/{}", repo.url(), file))?;
    let blob_id = sibling
        .lfs
        .as_ref()
        .map(|lfs| lfs.sha256.clone())
        .or(sibling.blob_id)
        .with_context(|| format!("Remote blob id missing for {}/{}", repo.url(), file))?;
    let sha256 = sibling.lfs.map(|lfs| lfs.sha256);

    Ok(RemoteFileMetadata {
        commit_hash: info.sha,
        size,
        blob_id,
        sha256,
    })
}

fn find_historical_remote_match(
    api: &Api,
    repo: &Repo,
    file: &str,
    legacy_size: u64,
    digest: &str,
    current_commit_hash: &str,
) -> Result<Option<RemoteFileMetadata>> {
    let commits = fetch_recent_repo_commits(&repo.url(), repo.revision(), 20)?;
    for commit in commits {
        if commit.id == current_commit_hash {
            continue;
        }
        let historical_repo = Repo::with_revision(repo.url(), RepoType::Model, commit.id);
        let Ok(remote) = remote_file_metadata(api, &historical_repo, file) else {
            continue;
        };
        if remote_metadata_matches_checksum(&remote, legacy_size, digest) {
            return Ok(Some(remote));
        }
    }
    Ok(None)
}

fn remote_metadata_matches_checksum(
    remote: &RemoteFileMetadata,
    legacy_size: u64,
    digest: &str,
) -> bool {
    remote.size == legacy_size
        && remote
            .sha256
            .as_deref()
            .map(|sha256| sha256.eq_ignore_ascii_case(digest))
            .unwrap_or(false)
}

fn fetch_recent_repo_commits(
    repo_id: &str,
    revision: &str,
    limit: usize,
) -> Result<Vec<RepoCommit>> {
    let endpoint = hf_endpoint();
    let token = hf_token_override();
    let repo_id = repo_id.to_string();
    let revision = revision.to_string();

    let join = std::thread::spawn(move || -> Result<Vec<RepoCommit>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("Build Tokio runtime for Hugging Face commit lookup")?;
        runtime.block_on(async move {
            let client = reqwest::Client::new();
            let url = format!(
                "{}/api/models/{}/commits/{}?limit={limit}",
                endpoint,
                urlencoding::encode(&repo_id),
                urlencoding::encode(&revision)
            );
            let mut request = client.get(url);
            if let Some(token) = token {
                request = request.bearer_auth(token);
            }
            request
                .send()
                .await
                .context("Fetch Hugging Face commit history")?
                .error_for_status()
                .context("Hugging Face commit history request failed")?
                .json::<Vec<RepoCommit>>()
                .await
                .context("Parse Hugging Face commit history")
        })
    });

    join.join()
        .map_err(|_| anyhow::anyhow!("Hugging Face commit lookup thread panicked"))?
}

fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

fn hf_token_override() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = std::env::var(key) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

fn materialize_cached_snapshot_pointer(
    cache_repo: &hf_hub::CacheRepo,
    commit_hash: &str,
    file: &str,
    blob_path: &Path,
) -> Result<PathBuf> {
    let mut pointer_path = cache_repo.pointer_path(commit_hash);
    pointer_path.push(file);
    if let Some(parent) = pointer_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    link_or_copy_file(blob_path, &pointer_path)?;
    cache_repo
        .create_ref(commit_hash)
        .context("Write cache ref")?;

    Ok(pointer_path)
}

fn link_or_copy_file(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        return Ok(());
    }
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(src, dst)
                .with_context(|| format!("Copy {} -> {}", src.display(), dst.display()))?;
            Ok(())
        }
    }
}

fn is_split_gguf_file(file: &str) -> bool {
    regex_lite::Regex::new(r"-\d{5}-of-\d{5}\.gguf$")
        .unwrap()
        .is_match(file)
}

fn expected_split_gguf_files(model: &catalog::CatalogModel) -> Vec<String> {
    let mut files: Vec<String> = std::iter::once(model.file.as_str())
        .chain(model.extra_files.iter().map(|asset| asset.file.as_str()))
        .filter(|file| is_split_gguf_file(file))
        .map(str::to_string)
        .collect();
    files.sort();
    files.dedup();
    files
}

fn sha256_file_hex(path: &Path, label: &str, total_bytes: u64) -> Result<String> {
    let file = std::fs::File::open(path).with_context(|| format!("Open {}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    let mut processed = 0u64;
    let mut last_progress = std::time::Instant::now();
    print_verify_progress(label, processed, total_bytes)?;
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("Read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        processed += read as u64;
        if total_bytes > 0 && last_progress.elapsed() >= std::time::Duration::from_millis(500) {
            print_verify_progress(label, processed, total_bytes)?;
            last_progress = std::time::Instant::now();
        }
    }
    print_verify_progress(label, total_bytes, total_bytes)?;
    eprintln!();
    Ok(format!("{:x}", hasher.finalize()))
}

fn print_verify_progress(label: &str, processed: u64, total_bytes: u64) -> Result<()> {
    let pct = if total_bytes > 0 {
        (processed as f64 / total_bytes as f64) * 100.0
    } else {
        0.0
    };
    eprint!(
        "\r   🔍 Verifying {}  {:>5.1}%  {}/{}",
        label,
        pct,
        format_size_bytes(processed),
        format_size_bytes(total_bytes)
    );
    std::io::stderr().flush().context("Flush verify progress")?;
    Ok(())
}

fn print_update_check_progress(current: usize, total: usize, repo_id: &str) -> Result<()> {
    let pct = if total > 0 {
        (current as f64 / total as f64) * 100.0
    } else {
        100.0
    };
    eprint!(
        "\r🔄 Checking updates {:>5.1}%  [{}/{}] {}",
        pct, current, total, repo_id
    );
    std::io::stderr()
        .flush()
        .context("Flush update check progress")?;
    Ok(())
}

fn clear_progress_line() -> Result<()> {
    eprint!("\r{: <140}\r", "");
    std::io::stderr().flush().context("Flush progress clear")?;
    Ok(())
}

fn cached_repos() -> Result<Vec<CachedRepo>> {
    let root = huggingface_hub_cache_dir();
    let mut repos = Vec::new();
    if !root.exists() {
        return Ok(repos);
    }

    for entry in std::fs::read_dir(&root).with_context(|| format!("Read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("models--") {
            continue;
        }
        let Some(repo_id) = cache_repo_id_from_dir(name) else {
            continue;
        };
        let refs_dir = path.join("refs");
        if !refs_dir.is_dir() {
            continue;
        }
        if let Some((ref_name, local_revision)) = first_cache_ref(&refs_dir)? {
            repos.push(CachedRepo {
                repo_id,
                ref_name,
                local_revision,
            });
        }
    }

    repos.sort_by(|left, right| left.repo_id.cmp(&right.repo_id));
    Ok(repos)
}

fn cached_repo_for_path(path: &Path) -> Result<Option<CachedRepo>> {
    let root = huggingface_hub_cache_dir();
    let rel = match path.strip_prefix(&root) {
        Ok(rel) => rel,
        Err(_) => return Ok(None),
    };
    let mut components = rel.components();
    let Some(repo_component) = components.next() else {
        return Ok(None);
    };
    let Some(repo_dir_name) = repo_component.as_os_str().to_str() else {
        return Ok(None);
    };
    if !repo_dir_name.starts_with("models--") {
        return Ok(None);
    }
    let Some(snapshot_component) = components.next() else {
        return Ok(None);
    };
    if snapshot_component.as_os_str() != "snapshots" {
        return Ok(None);
    }
    let Some(revision_component) = components.next() else {
        return Ok(None);
    };
    let Some(local_revision) = revision_component.as_os_str().to_str() else {
        return Ok(None);
    };
    let Some(repo_id) = cache_repo_id_from_dir(repo_dir_name) else {
        return Ok(None);
    };
    let repo_dir = root.join(repo_dir_name);
    let ref_name =
        matching_ref_name(&repo_dir, local_revision)?.unwrap_or_else(|| "main".to_string());
    Ok(Some(CachedRepo {
        repo_id,
        ref_name,
        local_revision: local_revision.to_string(),
    }))
}

pub(super) fn cache_repo_id_from_dir(name: &str) -> Option<String> {
    Some(name.strip_prefix("models--")?.replace("--", "/"))
}

fn first_cache_ref(refs_dir: &Path) -> Result<Option<(String, String)>> {
    let main = refs_dir.join("main");
    if main.is_file() {
        let value = std::fs::read_to_string(&main)
            .with_context(|| format!("Read {}", main.display()))?
            .trim()
            .to_string();
        if !value.is_empty() {
            return Ok(Some(("main".to_string(), value)));
        }
    }

    let mut refs = Vec::new();
    collect_ref_files(refs_dir, refs_dir, &mut refs)?;
    refs.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(refs.into_iter().next())
}

fn matching_ref_name(repo_dir: &Path, revision: &str) -> Result<Option<String>> {
    let refs_dir = repo_dir.join("refs");
    if !refs_dir.is_dir() {
        return Ok(None);
    }
    let mut refs = Vec::new();
    collect_ref_files(&refs_dir, &refs_dir, &mut refs)?;
    refs.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(refs
        .into_iter()
        .find(|(_, value)| value == revision)
        .map(|(name, _)| name))
}

fn collect_ref_files(root: &Path, dir: &Path, refs: &mut Vec<(String, String)>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("Read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_ref_files(root, &path, refs)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let ref_name = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let revision = std::fs::read_to_string(&path)
            .with_context(|| format!("Read {}", path.display()))?
            .trim()
            .to_string();
        if !revision.is_empty() {
            refs.push((ref_name, revision));
        }
    }
    Ok(())
}

fn remote_repo_info(api: &Api, repo_id: &str, ref_name: &str) -> Result<RepoInfo> {
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, ref_name.to_string());
    api.repo(repo)
        .info()
        .with_context(|| format!("Fetch repo info for {repo_id}@{ref_name}"))
}

fn repo_info_sha(info: &RepoInfo) -> String {
    info.sha.clone()
}

fn check_repo_update(api: &Api, repo: &CachedRepo) -> Result<Option<String>> {
    let remote = remote_repo_info(api, &repo.repo_id, &repo.ref_name)?;
    let remote_revision = repo_info_sha(&remote);
    if remote_revision == repo.local_revision {
        Ok(None)
    } else {
        Ok(Some(remote_revision))
    }
}

fn update_cached_repo(api: &Api, repo: &CachedRepo) -> Result<UpdateCounts> {
    let repo_handle =
        Repo::with_revision(repo.repo_id.clone(), RepoType::Model, repo.ref_name.clone());
    let api_repo = api.repo(repo_handle);
    let files = cached_repo_files(repo)?;
    if files.is_empty() {
        eprintln!("⚠️ {} has no cached files to refresh", repo.repo_id);
        return Ok(UpdateCounts::default());
    }

    eprintln!("   ref: {}", repo.ref_name);
    eprintln!("   current: {}", short_revision(&repo.local_revision));
    let mut counts = UpdateCounts::default();
    let mut downloaded = BTreeSet::new();
    let total_files = files.len() + 1;
    let mut position = 0usize;
    for file in files
        .into_iter()
        .chain(std::iter::once("config.json".to_string()))
    {
        if !downloaded.insert(file.clone()) {
            continue;
        }
        position += 1;
        eprintln!("   ↻ [{}/{}] {}", position, total_files, file);
        match api_repo.download(&file) {
            Ok(path) => {
                eprintln!("   ✅ {}", path.display());
                counts.refreshed += 1;
            }
            Err(err) if file == "config.json" => {
                if is_not_found_error(&err.to_string()) {
                    eprintln!("   ℹ️ no config.json published for {}", repo.repo_id);
                } else {
                    eprintln!("   ⚠️ config.json: {err}");
                }
                counts.missing_meta += 1;
            }
            Err(err) => {
                return Err(err).with_context(|| format!("Download {}/{}", repo.repo_id, file))
            }
        }
    }

    Ok(counts)
}

fn is_not_found_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("404") || message.contains("not found")
}

fn cached_repo_files(repo: &CachedRepo) -> Result<Vec<String>> {
    let cache = huggingface_hub_cache();
    let repo_handle =
        Repo::with_revision(repo.repo_id.clone(), RepoType::Model, repo.ref_name.clone());
    let root = cache.repo(repo_handle).pointer_path(&repo.local_revision);
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    collect_snapshot_files(&root, &root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_snapshot_files(root: &Path, dir: &Path, files: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("Read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_snapshot_files(root, &path, files)?;
            continue;
        }
        if !file_type.is_file() && !file_type.is_symlink() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push(rel);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_entry(path: PathBuf) -> MigrationEntry {
        MigrationEntry {
            path,
            status: MigrationStatus::Rehydratable,
            detail: String::new(),
            catalog: None,
        }
    }

    #[test]
    fn collect_legacy_candidates_detects_duplicates_and_mixed_layout() {
        let root = PathBuf::from("/tmp/.models");
        let flat = test_entry(root.join("MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf"));
        let nested = test_entry(
            root.join("minimax-hf")
                .join("Q4_K_M")
                .join("MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf"),
        );
        let entries = vec![&flat, &nested];

        let candidates = collect_legacy_candidates(&entries, &root);
        let duplicates = duplicate_legacy_files(&candidates);

        assert!(candidates.has_flat_layout);
        assert!(candidates.has_nested_layout);
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].0, "minimax-m2.5-q4_k_m-00001-of-00004.gguf");
        assert_eq!(duplicates[0].1.len(), 2);
    }

    #[test]
    fn legacy_candidate_for_file_returns_ambiguous_for_duplicate_basename() {
        let root = PathBuf::from("/tmp/.models");
        let flat = test_entry(root.join("foo.gguf"));
        let nested = test_entry(root.join("nested").join("foo.gguf"));
        let entries = vec![&flat, &nested];
        let candidates = collect_legacy_candidates(&entries, &root);

        match legacy_candidate_for_file(&candidates, "foo.gguf") {
            LegacyCandidateSelection::Ambiguous(paths) => {
                assert_eq!(paths.len(), 2);
            }
            _ => panic!("expected ambiguous legacy selection"),
        }
    }

    #[test]
    fn remote_metadata_checksum_match_requires_size_and_sha256() {
        let remote = RemoteFileMetadata {
            commit_hash: "abc123".to_string(),
            size: 42,
            blob_id: "deadbeef".to_string(),
            sha256: Some("0123456789abcdef".to_string()),
        };

        assert!(remote_metadata_matches_checksum(
            &remote,
            42,
            "0123456789ABCDEF"
        ));
        assert!(!remote_metadata_matches_checksum(
            &remote,
            41,
            "0123456789abcdef"
        ));
        assert!(!remote_metadata_matches_checksum(
            &remote,
            42,
            "ffffffffffffffff"
        ));
    }

    #[test]
    fn expected_split_gguf_files_includes_all_catalog_parts() {
        let model = catalog::MODEL_CATALOG
            .iter()
            .find(|model| model.name == "MiniMax-M2.5-Q4_K_M")
            .unwrap();
        let files = expected_split_gguf_files(model);

        assert_eq!(files.len(), 4);
        assert!(files.contains(&"MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf".to_string()));
        assert!(files.contains(&"MiniMax-M2.5-Q4_K_M-00004-of-00004.gguf".to_string()));
    }
}
