//! Shared presentation for the Hugging Face cache update workflow.

use skippy_model_hf::{blocking::run_hf_sync, huggingface_hub_cache_dir};
fn short_revision(value: &str) -> &str {
    value.get(..12).unwrap_or(value)
}
use crate::models::output::{DeterminateProgressLine, clear_stderr_line};
use anyhow::Result;
use skippy_model_hf::store::updates::{self, UpdateEvent, UpdateReport};
use std::io::Write;

pub fn run_update(repo: Option<&str>, all: bool, check: bool) -> Result<()> {
    let repo = repo.map(ToOwned::to_owned);
    let context = super::output::context();
    run_hf_sync(move || {
        super::output::sync_scope(context, || run_update_sync(repo.as_deref(), all, check))
    })
}

fn run_update_sync(repo: Option<&str>, all: bool, check: bool) -> Result<()> {
    let cache_root = huggingface_hub_cache_dir();
    let mut announced = false;
    let report = updates::run_update_in(&cache_root, repo, all, check, |event| {
        if !check && !announced && matches!(event, UpdateEvent::Refreshing { .. }) {
            let mut out = crate::models::output::console_err();
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
    let mut out = crate::models::output::console_err();
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
            writeln!(
                out,
                "   update: {program} models updates {}",
                repo.repo_id,
                program = crate::models::output::program()
            )?;
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
    let mut out = crate::models::output::console_err();
    if report.selected_repos == 0 {
        return Ok(());
    }
    if check {
        clear_stderr_line()?;
        if report.updates_available > 0 {
            writeln!(out, "📬 Update summary")?;
            writeln!(out, "   repos with updates: {}", report.updates_available)?;
            writeln!(
                out,
                "   update one: {program} models updates <repo>",
                program = crate::models::output::program()
            )?;
            writeln!(
                out,
                "   update all: {program} models updates --all",
                program = crate::models::output::program()
            )?;
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
