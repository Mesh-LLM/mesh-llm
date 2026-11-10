//! The catalog plugin-set surface shared by `plugins sets`, `setup`, and
//! `update`: fetching one catalog snapshot per operation, rendering sets for
//! humans, offering the recommended set when attended, and installing a set
//! with per-plugin results.
//!
//! Machine-readable output never sees any of this rendering; set operations
//! today are human-facing commands.

use std::io::{IsTerminal, Write};

use anyhow::{Context, Result};
use mesh_llm_plugin_manager::{
    MemberAvailability, PluginCatalog, PluginInstallOptions, PluginProgressEvent,
    PluginProgressReporter, PluginStore, SetInstallResult, set_offer_summary,
};
use reqwest::Client;

use crate::plugin::CliPluginProgress;

/// The catalog set setup and update offer by name.
pub const RECOMMENDED_SET: &str = "recommended";

/// One catalog snapshot for one operation: the same fetched copy backs set
/// resolution and installation so membership cannot shift mid-operation.
/// The request carries a timeout so a stalled catalog server cannot hang a
/// set command, setup's optional phase, or an upgrade.
pub async fn fetch_catalog(options: &PluginInstallOptions) -> Result<PluginCatalog> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("build plugin catalog client")?;
    let sets_url = mesh_llm_plugin_manager::sets_url_for(&options.catalog_url);
    PluginCatalog::fetch_with_sets(&client, &options.catalog_url, sets_url.as_deref()).await
}

/// An operator is attended only on an interactive terminal without
/// `--no-interactive`; automation and missing input must never be prompted.
pub fn attended(no_interactive: bool) -> bool {
    !no_interactive && std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// Render `mesh-llm plugins sets list`.
pub fn render_sets_list(catalog: &PluginCatalog) -> Vec<String> {
    let mut lines = vec!["ℹ️  Plugin sets".to_string()];
    if catalog.sets().is_empty() {
        lines.push("ℹ️  The catalog publishes no plugin sets.".to_string());
    }
    lines.push("NAME\tDESCRIPTION".to_string());
    for set in catalog.sets() {
        lines.push(format!("{}\t{}", set.name, set.description));
    }
    lines
}

/// Render `mesh-llm plugins sets show NAME`, including per-member platform
/// availability. Returns `Err` for an unknown set so the caller can fail
/// with guidance.
pub fn render_set_show(
    catalog: &PluginCatalog,
    name: &str,
    target: &str,
    host_version: &str,
) -> Result<Vec<String>> {
    let resolution = mesh_llm_plugin_manager::resolve_set(catalog, name, target, host_version)?;
    let mut lines = vec![format!(
        "ℹ️  Set: {} — {}",
        resolution.name, resolution.description
    )];
    if resolution.members.is_empty() {
        lines.push("ℹ️  This set is empty.".to_string());
    }
    lines.push("NAME\tVERSION\tDESCRIPTION\tAVAILABILITY".to_string());
    for member in &resolution.members {
        lines.push(format!(
            "{}\t{}\t{}\t{}",
            member.name,
            member.version.as_deref().unwrap_or("-"),
            member.description,
            match &member.availability {
                MemberAvailability::Available => "Available".to_string(),
                MemberAvailability::Unavailable(reason) => reason.clone(),
            }
        ));
    }
    let installable = resolution
        .members
        .iter()
        .any(|member| member.availability == MemberAvailability::Available);
    if installable {
        lines.push(format!(
            "Install this set:\n  mesh-llm plugins sets install {name}"
        ));
    } else {
        lines.push("ℹ️  No members of this set can be installed on this platform.".to_string());
    }
    Ok(lines)
}

/// The prompt line an attended offer prints before asking, describing the
/// set and its resolvable members. `None` when the set is absent or empty,
/// so callers skip an unnecessary prompt.
pub fn offer_lines(catalog: &PluginCatalog, set_name: &str) -> Option<Vec<String>> {
    let (description, members) = set_offer_summary(catalog, set_name)?;
    if members.is_empty() {
        return None;
    }
    let mut lines = vec![format!("ℹ️  {set_name} plugin set")];
    let mut joined = String::new();
    for (index, (name, version)) in members.iter().enumerate() {
        let member = format!("  {name} {version}");
        joined.push_str(&member);
        if index + 1 < members.len() {
            joined.push('\n');
        }
    }
    let _ = description;
    lines.push(joined);
    Some(lines)
}

/// Ask whether to install `set_name`, defaulting to No. An EOF, a read
/// error, or a missing terminal declines; unattended callers never reach
/// here.
pub fn prompt_install_set(set_name: &str) -> bool {
    crate::terminal::confirm_yes_no(
        &format!("Install plugin set '{set_name}'?"),
        crate::terminal::ConfirmDefault::No,
    )
    .unwrap_or(Some(false))
    .unwrap_or(false)
}

/// Install the named set against one catalog snapshot, with the CLI progress
/// rendering and the summary line an explicit `plugins sets install NAME`
/// prints. Fails (returns Err) when any member could not be installed, after
/// reporting every per-plugin result.
pub async fn install_set_command(
    catalog: &PluginCatalog,
    set_name: &str,
    options: &PluginInstallOptions,
) -> Result<SetInstallResult> {
    let mut err = mesh_llm_events::console_err();
    if let Some(lines) = offer_lines(catalog, set_name) {
        for line in lines {
            let _ = writeln!(err, "{line}");
        }
    }
    let _ = writeln!(err, "ℹ️  Installing plugin set: {set_name}");
    let store = PluginStore::new(&options.store_root);
    store.migrate_legacy_defaults()?;
    let result = mesh_llm_plugin_manager::install_named_set(
        catalog,
        set_name,
        options.target.triple(),
        &options.host_version,
        options,
        &mut VerifyingPluginProgress::default(),
    )
    .await?;
    report_set_result(&result);
    if !result.is_complete() {
        let mut err = mesh_llm_events::console_err();
        for (name, reason) in &result.failed {
            let _ = writeln!(err, "❌ Could not install {name}: {reason}");
        }
        anyhow::bail!(
            "plugin set '{}' installed {} of {} members; retry with mesh-llm plugins sets install {set_name}",
            set_name,
            result.installed_count(),
            result.installed_count() + result.failed.len()
        );
    }
    Ok(result)
}

/// What a set installation prints when it finishes, per outcome.
pub fn report_set_result(result: &SetInstallResult) {
    let mut err = mesh_llm_events::console_err();
    for (name, reason) in &result.skipped {
        let _ = writeln!(err, "ℹ️  {name} skipped: {reason}");
    }
    if result.failed.is_empty() {
        let _ = writeln!(
            err,
            "✅ Set installation complete: {} installed.",
            result.installed_count()
        );
    }
}

/// The plugin progress rendering for set installs: the CLI spinner/gauge
/// plus the verification line each verified download prints.
#[derive(Default)]
pub struct VerifyingPluginProgress {
    inner: CliPluginProgress,
}

impl PluginProgressReporter for VerifyingPluginProgress {
    fn report(&mut self, event: PluginProgressEvent) {
        match event {
            PluginProgressEvent::DownloadFinished { asset } => {
                self.inner.report(PluginProgressEvent::DownloadFinished {
                    asset: asset.clone(),
                });
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(err, "✅ Archive verified.");
            }
            PluginProgressEvent::Warning { message } => {
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(err, "⚠️  {message}");
            }
            other => self.inner.report(other),
        }
    }
}

/// What optional plugin installation should do, decided before it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionalPluginPlan {
    /// Install this catalog set without prompting (explicit `--plugin-set`).
    InstallSet(String),
    /// Offer this set when attended; default No.
    Offer(String),
    /// Skip optional plugin installation and say so.
    Skip(Option<String>),
}

/// Decide the optional-plugin plan for setup and update from the explicit
/// flags and the environment. `--skip-plugin-install` (or its deprecated
/// alias, or the legacy `MESH_LLM_NO_DEFAULT_PLUGINS` setting) skips optional
/// installation only; required repairs are never suppressed.
pub fn optional_plugin_plan(
    plugin_set: Option<&str>,
    skip_plugin_install: bool,
    legacy_no_default_plugins: bool,
    attended: bool,
) -> OptionalPluginPlan {
    if let Some(name) = plugin_set {
        return OptionalPluginPlan::InstallSet(name.to_string());
    }
    if skip_plugin_install || legacy_no_default_plugins {
        return OptionalPluginPlan::Skip(Some("Optional plugin installation skipped.".to_string()));
    }
    if !attended {
        return OptionalPluginPlan::Skip(None);
    }
    OptionalPluginPlan::Offer(RECOMMENDED_SET.to_string())
}

/// What the optional-plugin phase did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionalPluginResult {
    /// Nothing optional was installed: skipped, declined, or nothing to do.
    Skipped,
    /// Every eligible member landed.
    Installed(usize),
    /// Some members landed, some could not; a retry is guided.
    Incomplete(usize),
}

/// Run the optional-plugin phase of setup/update after the core work
/// succeeded. Optional failure never fails the host result: it warns, gives
/// retry guidance, and reports the installation as incomplete. `prompt`
/// answers "install set NAME?" for an attended offer; an explicit
/// `--plugin-set` selection never asks it.
pub async fn run_optional_plugin_phase(
    plan: &OptionalPluginPlan,
    prompt: &mut dyn FnMut(&str) -> bool,
) -> Result<OptionalPluginResult> {
    let mut err = mesh_llm_events::console_err();
    match plan {
        OptionalPluginPlan::Skip(reason) => {
            if let Some(reason) = reason {
                let _ = writeln!(err, "ℹ️  {reason}");
            } else {
                let _ = writeln!(err, "ℹ️  No optional plugin set selected.");
            }
            Ok(OptionalPluginResult::Skipped)
        }
        OptionalPluginPlan::InstallSet(name) | OptionalPluginPlan::Offer(name) => {
            let options = match PluginInstallOptions::from_env() {
                Ok(options) => options,
                Err(error) => {
                    // No usable plugin store on this machine (for example an
                    // unsupported platform): an incomplete optional install,
                    // never a failed core setup or upgrade.
                    let _ = writeln!(
                        err,
                        "⚠️  Optional plugin installation needs a plugin store ({error:#}). \
                         Optional plugin installation incomplete."
                    );
                    let _ = writeln!(err, "ℹ️  Retry: mesh-llm plugins sets install {name}");
                    return Ok(OptionalPluginResult::Incomplete(0));
                }
            };
            let catalog = match fetch_catalog(&options).await {
                Ok(catalog) => catalog,
                Err(error) => {
                    let _ = writeln!(
                        err,
                        "⚠️  Could not read the plugin catalog ({error:#}). Optional plugin \
                         installation incomplete."
                    );
                    let _ = writeln!(err, "ℹ️  Retry: mesh-llm plugins sets install {name}");
                    return Ok(OptionalPluginResult::Incomplete(0));
                }
            };
            if matches!(plan, OptionalPluginPlan::Offer(_)) {
                let Some(lines) = offer_lines(&catalog, name) else {
                    let _ = writeln!(
                        err,
                        "ℹ️  The catalog publishes no '{name}' plugin set with members; nothing \
                         to install."
                    );
                    return Ok(OptionalPluginResult::Skipped);
                };
                for line in lines {
                    let _ = writeln!(err, "{line}");
                }
                if !prompt(name) {
                    let _ = writeln!(err, "ℹ️  Optional plugin installation skipped.");
                    return Ok(OptionalPluginResult::Skipped);
                }
            } else {
                let _ = writeln!(err, "ℹ️  Installing plugin set: {name}");
            }
            match install_set_quiet(&catalog, name, &options).await {
                Ok(result) => {
                    let mut err = mesh_llm_events::console_err();
                    if result.failed.is_empty() {
                        let _ = writeln!(
                            err,
                            "✅ Plugin set installed: {} installed.",
                            result.installed_count()
                        );
                        Ok(OptionalPluginResult::Installed(result.installed_count()))
                    } else {
                        for (failed_name, reason) in &result.failed {
                            let _ = writeln!(
                                err,
                                "⚠️  Could not install {failed_name}: {reason}. Optional plugin \
                                 installation incomplete."
                            );
                        }
                        let _ = writeln!(err, "ℹ️  Retry: mesh-llm plugins sets install {name}");
                        Ok(OptionalPluginResult::Incomplete(result.installed_count()))
                    }
                }
                Err(error) => {
                    // An unknown explicitly-selected set is a usage error,
                    // never a successful empty selection: surface it.
                    if format!("{error:#}").contains("Unknown plugin set") {
                        return Err(error);
                    }
                    // Any other resolution failure (a download, a vanished
                    // entry) is an incomplete optional install, not a failed
                    // host operation.
                    let mut err = mesh_llm_events::console_err();
                    let _ = writeln!(
                        err,
                        "⚠️  {error:#}. Optional plugin installation incomplete."
                    );
                    let _ = writeln!(err, "ℹ️  Retry: mesh-llm plugins sets install {name}");
                    Ok(OptionalPluginResult::Incomplete(0))
                }
            }
        }
    }
}

/// Install a set without the command's own summary lines; per-member progress
/// still renders.
async fn install_set_quiet(
    catalog: &PluginCatalog,
    name: &str,
    options: &PluginInstallOptions,
) -> Result<SetInstallResult> {
    mesh_llm_plugin_manager::install_named_set(
        catalog,
        name,
        options.target.triple(),
        &options.host_version,
        options,
        &mut VerifyingPluginProgress::default(),
    )
    .await
}

/// The attended update notices: every compatible installed plugin with an
/// available update, showing installed and available versions and the exact
/// command to run. Plugins inside and outside every set are listed alike.
pub fn render_update_notices(plan: &mesh_llm_plugin_manager::PluginUpgradePlan) -> Vec<String> {
    let notices = plan.update_notices();
    if notices.is_empty() {
        return Vec::new();
    }
    let notices = &notices[..];
    let mut lines = vec!["⚠️  Plugin updates available:".to_string()];
    for check in notices {
        if let mesh_llm_plugin_manager::PluginUpgradeVerdict::UpdateAvailable { to } =
            &check.verdict
        {
            lines.push(format!(
                "  {} {} -> {}",
                check.name, check.installed_version, to
            ));
        }
    }
    lines.push("Update when ready:".to_string());
    for check in notices {
        lines.push(format!("  mesh-llm plugins update {}", check.name));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENTRIES: &str = r#"{"name":"capsules","description":"Audit and provenance records","github_url":"https://github.com/Mesh-LLM/capsules","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v0.2.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"}}
"#;

    const SETS: &str = r#"{"name":"recommended","description":"Recommended plugins for MeshLLM","plugins":["capsules"]}"#;

    fn catalog() -> PluginCatalog {
        PluginCatalog::parse_jsonl(ENTRIES)
            .unwrap()
            .with_sets(PluginCatalog::parse_sets_jsonl(SETS).unwrap())
            .unwrap()
    }

    #[test]
    fn sets_list_renders_one_row_per_set() {
        let lines = render_sets_list(&catalog());
        assert_eq!(lines[0], "ℹ️  Plugin sets");
        assert_eq!(lines[1], "NAME\tDESCRIPTION");
        assert_eq!(lines[2], "recommended\tRecommended plugins for MeshLLM");
    }

    #[test]
    fn sets_show_renders_availability_and_the_install_command() {
        let lines = render_set_show(
            &catalog(),
            "recommended",
            "x86_64-unknown-linux-gnu",
            "0.90.0",
        )
        .unwrap();
        assert!(lines[0].contains("Set: recommended"), "{lines:?}");
        assert!(lines[1].starts_with("NAME\tVERSION"), "{lines:?}");
        assert!(lines[2].contains("capsules"), "{lines:?}");
        assert!(
            lines
                .last()
                .unwrap()
                .contains("plugins sets install recommended"),
            "{lines:?}"
        );

        let error = render_set_show(&catalog(), "missing", "x86_64-unknown-linux-gnu", "0.90.0")
            .unwrap_err();
        assert!(format!("{error:#}").contains("Unknown plugin set 'missing'"));
    }

    #[test]
    fn the_offer_names_the_set_and_its_members() {
        let lines = offer_lines(&catalog(), "recommended").unwrap();
        assert_eq!(lines[0], "ℹ️  recommended plugin set");
        assert!(lines[1].contains("capsules v0.2.0"), "{lines:?}");
        assert!(offer_lines(&catalog(), "missing").is_none());
    }

    #[test]
    fn an_empty_set_is_not_offered() {
        let catalog = PluginCatalog::parse_jsonl(ENTRIES)
            .unwrap()
            .with_sets(
                PluginCatalog::parse_sets_jsonl(
                    r#"{"name":"empty","description":"Nothing here","plugins":[]}"#,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(offer_lines(&catalog, "empty").is_none());
    }

    #[test]
    fn explicit_selection_wins_and_unattended_defaults_to_no_optional_plugins() {
        assert_eq!(
            optional_plugin_plan(Some("extra"), false, false, false),
            OptionalPluginPlan::InstallSet("extra".into())
        );
        assert_eq!(
            optional_plugin_plan(Some("recommended"), true, false, true),
            OptionalPluginPlan::InstallSet("recommended".into()),
            "an explicit set with --skip-plugin-install is rejected at parse time"
        );
        assert_eq!(
            optional_plugin_plan(None, true, false, true),
            OptionalPluginPlan::Skip(Some("Optional plugin installation skipped.".into()))
        );
        assert_eq!(
            optional_plugin_plan(None, false, true, false),
            OptionalPluginPlan::Skip(Some("Optional plugin installation skipped.".into())),
            "the legacy environment default skips optional installation"
        );
        assert_eq!(
            optional_plugin_plan(None, false, false, false),
            OptionalPluginPlan::Skip(None),
            "unattended installs no optional plugin by default"
        );
        assert_eq!(
            optional_plugin_plan(None, false, false, true),
            OptionalPluginPlan::Offer(RECOMMENDED_SET.into())
        );
    }
}
