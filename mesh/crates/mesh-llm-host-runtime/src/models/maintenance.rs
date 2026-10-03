//! Mesh presentation for the shared Hugging Face cache update workflow.

use super::{huggingface_hub_cache_dir, run_hf_sync, short_revision};
use anyhow::Result;
use mesh_llm_events::terminal_progress::{DeterminateProgressLine, clear_stderr_line};
use skippy_model_hf::store::updates::{self, UpdateEvent, UpdateReport};
use std::{collections::BTreeSet, io::Write, path::PathBuf};

#[cfg(test)]
pub(super) use updates::cache_repo_id_from_dir;

pub fn run_update(repo: Option<&str>, all: bool, check: bool) -> Result<()> {
    let repo = repo.map(ToOwned::to_owned);
    run_hf_sync(move || run_update_sync(repo.as_deref(), all, check))
}

fn run_update_sync(repo: Option<&str>, all: bool, check: bool) -> Result<()> {
    let cache_root = huggingface_hub_cache_dir();
    let mut announced = false;
    let report = updates::run_update_in(&cache_root, repo, all, check, |event| {
        if !check && !announced && matches!(event, UpdateEvent::Refreshing { .. }) {
            let mut out = mesh_llm_events::console_err();
            writeln!(out, "🔄 Updating cached Hugging Face repos")?;
            writeln!(out, "📁 Cache: {}", cache_root.display())?;
            if let UpdateEvent::Refreshing { total, .. } = &event {
                writeln!(out, "📦 Selected: {total}")?;
            }
            writeln!(out)?;
            announced = true;
        }
        render_update_event(event)
    })?;
    render_update_summary(check, &report)
}

fn render_update_event(event: UpdateEvent) -> Result<()> {
    let mut out = mesh_llm_events::console_err();
    match event {
        UpdateEvent::Empty { cache_dir } => {
            writeln!(out, "📦 No cached Hugging Face model repos found")?;
            writeln!(out, "   {}", cache_dir.display())?;
        }
        UpdateEvent::Checking {
            current,
            total,
            repo,
        } => {
            DeterminateProgressLine::new("🔄").draw_counts(
                "Checking updates",
                current,
                total,
                Some(&format!(" {}", repo.repo_id)),
            )?;
        }
        UpdateEvent::Available {
            current,
            total,
            repo,
            remote_revision,
        } => {
            clear_stderr_line()?;
            writeln!(out, "🆕 [{current}/{total}] {}", repo.repo_id)?;
            writeln!(out, "   ref: {}", repo.ref_name)?;
            writeln!(out, "   local: {}", short_revision(&repo.local_revision))?;
            writeln!(out, "   latest: {}", short_revision(&remote_revision))?;
            writeln!(out, "   update: mesh-llm models updates {}", repo.repo_id)?;
            writeln!(out)?;
        }
        UpdateEvent::Refreshing {
            current,
            total,
            repo,
        } => {
            writeln!(out, "🧭 [{current}/{total}] {}", repo.repo_id)?;
            writeln!(out, "   ref: {}", repo.ref_name)?;
            writeln!(out, "   current: {}", short_revision(&repo.local_revision))?;
        }
        UpdateEvent::NoCachedFiles { repo_id } => {
            writeln!(out, "⚠️ {repo_id} has no cached files to refresh")?;
        }
        UpdateEvent::FileStarted {
            current,
            total,
            file,
        } => {
            writeln!(out, "   ↻ [{current}/{total}] {file}")?;
        }
        UpdateEvent::FileRefreshed { path } => {
            writeln!(out, "   ✅ {}", path.display())?;
        }
        UpdateEvent::ConfigMissing { repo_id, error } => {
            if let Some(error) = error {
                writeln!(out, "   ⚠️ config.json: {error}")?;
            } else {
                writeln!(out, "   ℹ️ no config.json published for {repo_id}")?;
            }
        }
    }
    Ok(())
}

fn render_update_summary(check: bool, report: &UpdateReport) -> Result<()> {
    let mut out = mesh_llm_events::console_err();
    if report.selected_repos == 0 {
        return Ok(());
    }
    if check {
        clear_stderr_line()?;
        if report.updates_available > 0 {
            writeln!(out, "📬 Update summary")?;
            writeln!(out, "   repos with updates: {}", report.updates_available)?;
            writeln!(out, "   update one: mesh-llm models updates <repo>")?;
            writeln!(out, "   update all: mesh-llm models updates --all")?;
        }
    } else {
        writeln!(out)?;
        writeln!(out, "✅ Update complete")?;
        writeln!(out, "   refreshed files: {}", report.refreshed_files)?;
        if report.missing_meta > 0 {
            writeln!(out, "   missing config.json: {}", report.missing_meta)?;
        }
    }
    Ok(())
}

pub fn warn_about_updates_for_paths(paths: &[PathBuf]) {
    let mut console = mesh_llm_events::console_err();
    let cache_root = huggingface_hub_cache_dir();
    let mut cache_models = Vec::new();
    let mut seen = BTreeSet::new();
    for path in paths {
        let Some(repo) = (match updates::cached_repo_for_path_in(path, &cache_root) {
            Ok(repo) => repo,
            Err(error) => {
                let _ = writeln!(
                    console,
                    "Warning: could not inspect cached Hugging Face repo for {}: {error}",
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

    let result = run_hf_sync(move || {
        let mut console = mesh_llm_events::console_err();
        let api = skippy_model_hf::build_hf_sync_api()?;
        for repo in cache_models {
            match updates::check_repo_update(&api, &repo) {
                Ok(Some(remote_revision)) => {
                    let _ = writeln!(console, "🆕 Update available for {}", repo.repo_id);
                    let _ = writeln!(
                        console,
                        "   local: {}",
                        short_revision(&repo.local_revision)
                    );
                    let _ = writeln!(console, "   latest: {}", short_revision(&remote_revision));
                    let _ = writeln!(console, "   continuing with pinned local snapshot");
                    let _ = writeln!(
                        console,
                        "   update: mesh-llm models updates {}",
                        repo.repo_id
                    );
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = writeln!(
                        console,
                        "Warning: could not check for updates for {}: {error}",
                        repo.repo_id
                    );
                }
            }
        }
        Ok(())
    });
    if let Err(error) = result {
        let _ = writeln!(
            console,
            "Warning: could not initialize Hugging Face update checks: {error}"
        );
    }
}
