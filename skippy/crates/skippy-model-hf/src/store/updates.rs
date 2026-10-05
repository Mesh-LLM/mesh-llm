//! Cached Hugging Face repository update policy shared by both CLIs.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use hf_hub::{HFClientSync, RepoTypeModel, repository::ModelInfo};
use serde::Serialize;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CachedRepo {
    pub repo_id: String,
    pub ref_name: String,
    pub local_revision: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct UpdateCounts {
    pub refreshed: usize,
    pub missing_meta: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct UpdateReport {
    pub checked_repos: usize,
    pub updates_available: usize,
    pub selected_repos: usize,
    pub refreshed_files: usize,
    pub missing_meta: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UpdateEvent {
    Empty {
        cache_dir: PathBuf,
    },
    Checking {
        current: usize,
        total: usize,
        repo: CachedRepo,
    },
    Available {
        current: usize,
        total: usize,
        repo: CachedRepo,
        remote_revision: String,
    },
    Refreshing {
        current: usize,
        total: usize,
        repo: CachedRepo,
    },
    NoCachedFiles {
        repo_id: String,
    },
    FileStarted {
        current: usize,
        total: usize,
        file: String,
    },
    FileRefreshed {
        path: PathBuf,
    },
    ConfigMissing {
        repo_id: String,
        error: Option<String>,
    },
}

/// Check revisions or refresh the same files Mesh previously selected from its Hub cache.
pub fn run_update_in(
    cache_root: &Path,
    repo: Option<&str>,
    all: bool,
    check: bool,
    mut emit: impl FnMut(UpdateEvent) -> Result<()>,
) -> Result<UpdateReport> {
    let repos = cached_repos_in(cache_root)?;
    if repos.is_empty() {
        emit(UpdateEvent::Empty {
            cache_dir: cache_root.to_path_buf(),
        })?;
        return Ok(UpdateReport {
            checked_repos: 0,
            updates_available: 0,
            selected_repos: 0,
            refreshed_files: 0,
            missing_meta: 0,
        });
    }
    let selected = select_cached_repos(repos, repo, all, check)?;
    let total = selected.len();
    let api = crate::build_hf_sync_api_in(cache_root)?;
    let mut report = UpdateReport {
        checked_repos: 0,
        updates_available: 0,
        selected_repos: total,
        refreshed_files: 0,
        missing_meta: 0,
    };
    for (index, cached) in selected.into_iter().enumerate() {
        let current = index + 1;
        if check {
            emit(UpdateEvent::Checking {
                current,
                total,
                repo: cached.clone(),
            })?;
            report.checked_repos += 1;
            if let Some(remote_revision) = check_repo_update(&api, &cached)? {
                report.updates_available += 1;
                emit(UpdateEvent::Available {
                    current,
                    total,
                    repo: cached,
                    remote_revision,
                })?;
            }
        } else {
            emit(UpdateEvent::Refreshing {
                current,
                total,
                repo: cached.clone(),
            })?;
            let counts = refresh_cached_repo_in(&api, cache_root, &cached, &mut emit)?;
            report.refreshed_files += counts.refreshed;
            report.missing_meta += counts.missing_meta;
        }
    }
    Ok(report)
}

pub fn select_cached_repos(
    repos: Vec<CachedRepo>,
    repo: Option<&str>,
    all: bool,
    check: bool,
) -> Result<Vec<CachedRepo>> {
    if all || check && repo.is_none() {
        return Ok(repos);
    }
    let Some(repo_id) = repo else {
        bail!(
            "Pass a repo id or --all. Use `models updates --check` to inspect updates without downloading."
        );
    };
    let repo_id = repo_id.trim();
    let Some(found) = repos.into_iter().find(|entry| entry.repo_id == repo_id) else {
        bail!("Cached repo not found: {repo_id}");
    };
    Ok(vec![found])
}

pub fn cached_repos_in(cache_root: &Path) -> Result<Vec<CachedRepo>> {
    let mut repos = Vec::new();
    if !cache_root.exists() {
        return Ok(repos);
    }
    for entry in
        std::fs::read_dir(cache_root).with_context(|| format!("Read {}", cache_root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
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

pub fn cached_repo_for_path_in(path: &Path, cache_root: &Path) -> Result<Option<CachedRepo>> {
    let rel = match path.strip_prefix(cache_root) {
        Ok(rel) => rel,
        Err(_) => return Ok(None),
    };
    let mut components = rel.components();
    let Some(repo_dir_name) = components.next().and_then(|part| part.as_os_str().to_str()) else {
        return Ok(None);
    };
    let Some(repo_id) = cache_repo_id_from_dir(repo_dir_name) else {
        return Ok(None);
    };
    if components
        .next()
        .is_none_or(|part| part.as_os_str() != "snapshots")
    {
        return Ok(None);
    }
    let Some(local_revision) = components.next().and_then(|part| part.as_os_str().to_str()) else {
        return Ok(None);
    };
    let repo_dir = cache_root.join(repo_dir_name);
    let ref_name =
        matching_ref_name(&repo_dir, local_revision)?.unwrap_or_else(|| "main".to_string());
    Ok(Some(CachedRepo {
        repo_id,
        ref_name,
        local_revision: local_revision.to_string(),
    }))
}

pub fn cache_repo_id_from_dir(name: &str) -> Option<String> {
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

fn remote_repo_info(api: &HFClientSync, repo: &CachedRepo) -> Result<ModelInfo> {
    let (owner, name) = repo.repo_id.split_once('/').unwrap_or(("", &repo.repo_id));
    api.model(owner, name)
        .info()
        .revision(repo.ref_name.clone())
        .send()
        .with_context(|| format!("Fetch repo info for {}@{}", repo.repo_id, repo.ref_name))
}

pub fn check_repo_update(api: &HFClientSync, repo: &CachedRepo) -> Result<Option<String>> {
    let remote_revision = remote_repo_info(api, repo)?.sha.unwrap_or_default();
    Ok((remote_revision != repo.local_revision).then_some(remote_revision))
}

fn refresh_cached_repo_in(
    api: &HFClientSync,
    cache_root: &Path,
    repo: &CachedRepo,
    emit: &mut impl FnMut(UpdateEvent) -> Result<()>,
) -> Result<UpdateCounts> {
    let (owner, name) = repo.repo_id.split_once('/').unwrap_or(("", &repo.repo_id));
    let api_repo = api.model(owner, name);
    let files = cached_repo_files_in(cache_root, repo)?;
    if files.is_empty() {
        emit(UpdateEvent::NoCachedFiles {
            repo_id: repo.repo_id.clone(),
        })?;
        return Ok(UpdateCounts::default());
    }
    let total = files.len() + 1;
    let mut seen = BTreeSet::new();
    let mut counts = UpdateCounts::default();
    for file in files
        .into_iter()
        .chain(std::iter::once("config.json".to_string()))
    {
        if !seen.insert(file.clone()) {
            continue;
        }
        emit(UpdateEvent::FileStarted {
            current: seen.len(),
            total,
            file: file.clone(),
        })?;
        match api_repo
            .download_file()
            .filename(file.clone())
            .revision(repo.ref_name.clone())
            .send()
        {
            Ok(path) => {
                counts.refreshed += 1;
                emit(UpdateEvent::FileRefreshed { path })?;
            }
            Err(error) if file == "config.json" => {
                counts.missing_meta += 1;
                let message = error.to_string();
                emit(UpdateEvent::ConfigMissing {
                    repo_id: repo.repo_id.clone(),
                    error: (!is_not_found_error(&message)).then_some(message),
                })?;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("Download {}/{}", repo.repo_id, file));
            }
        }
    }
    Ok(counts)
}

fn is_not_found_error(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("404") || message.contains("not found")
}

pub fn cached_repo_files_in(cache_root: &Path, repo: &CachedRepo) -> Result<Vec<String>> {
    let snapshots_dir = cache_root
        .join(super::local::huggingface_repo_folder_name(
            &repo.repo_id,
            RepoTypeModel,
        ))
        .join("snapshots");
    if !snapshots_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(&snapshots_dir)
        .with_context(|| format!("Read {}", snapshots_dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            entries.push((
                entry.file_name().to_string_lossy().to_string(),
                entry.path(),
            ));
        }
    }
    let exact = snapshots_dir.join(&repo.local_revision);
    let mut roots = if exact.is_dir() {
        vec![exact]
    } else {
        let mut matches: Vec<PathBuf> = entries
            .iter()
            .filter(|(name, _)| {
                name.starts_with(&repo.local_revision) || repo.local_revision.starts_with(name)
            })
            .map(|(_, path)| path.clone())
            .collect();
        matches.sort();
        matches
    };
    if roots.is_empty() {
        roots = entries.into_iter().map(|(_, path)| path).collect();
        roots.sort();
    }
    let mut files = BTreeSet::new();
    for root in roots {
        let mut collected = Vec::new();
        collect_snapshot_files(&root, &root, &mut collected)?;
        files.extend(collected);
    }
    Ok(files.into_iter().collect())
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
        files.push(
            path.strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/"),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_repo_files_fall_back_to_matching_snapshot_prefix() {
        let base = tempfile::tempdir().unwrap();
        let snapshot = base.path().join("models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/9280dd353ab5cafebabedeadbeef123456789abc");
        std::fs::create_dir_all(snapshot.join("BF16")).unwrap();
        std::fs::write(snapshot.join("BF16/model.gguf"), b"gguf").unwrap();
        let repo = CachedRepo {
            repo_id: "unsloth/Qwen3.6-35B-A3B-GGUF".into(),
            ref_name: "main".into(),
            local_revision: "9280dd353ab5".into(),
        };
        assert_eq!(
            cached_repo_files_in(base.path(), &repo).unwrap(),
            ["BF16/model.gguf"]
        );
    }

    #[test]
    fn selection_matches_mesh_check_and_refresh_modes() {
        let repos = vec![CachedRepo {
            repo_id: "Qwen/model".into(),
            ref_name: "main".into(),
            local_revision: "abc".into(),
        }];
        assert_eq!(
            select_cached_repos(repos.clone(), None, false, true).unwrap(),
            repos
        );
        assert!(select_cached_repos(repos.clone(), None, false, false).is_err());
        assert_eq!(
            select_cached_repos(repos.clone(), Some("Qwen/model"), false, false).unwrap(),
            repos
        );
        assert!(select_cached_repos(repos, Some("other/model"), false, false).is_err());
    }
}
