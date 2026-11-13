//! The catalog plugin-set surface shared by `plugins sets`, `setup`, and
//! `update`: fetching one catalog snapshot per operation, rendering sets for
//! humans, offering the recommended set when attended, and installing a set
//! with per-plugin results.
//!
//! Machine-readable output never sees any of this rendering; set operations
//! today are human-facing commands.

use std::io::{IsTerminal, Write};

use anyhow::Result;
use mesh_llm_plugin_manager::{
    MemberAvailability, PluginCatalog, PluginInstallOptions, PluginProgressEvent,
    PluginProgressReporter, PluginStore, SetInstallResult,
};

use crate::plugin::CliPluginProgress;
use crate::plugin_set_offers::{RecordLock, SetOffers, added_members_notice, members_added_since};

/// The catalog set setup and update offer by name.
pub const RECOMMENDED_SET: &str = "recommended";

pub use mesh_llm_plugin_manager::upgrade_gate::fetch_catalog;

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

/// Describe an attended offer's set and eligible pinned members. Skip the
/// prompt when the set is absent or nothing supports this target and host.
pub fn offer_lines(
    catalog: &PluginCatalog,
    set_name: &str,
    target: &str,
    host_version: &str,
) -> Option<Vec<String>> {
    let resolution =
        mesh_llm_plugin_manager::resolve_set(catalog, set_name, target, host_version).ok()?;
    let members: Vec<_> = resolution
        .members
        .iter()
        .filter(|member| member.availability == MemberAvailability::Available)
        .map(|member| {
            format!(
                "  {} {}",
                member.name,
                member.version.as_deref().unwrap_or("-")
            )
        })
        .collect();
    if members.is_empty() {
        return None;
    }
    Some(vec![
        format!("ℹ️  {set_name} plugin set — {}", resolution.description),
        members.join("\n"),
    ])
}

/// Ask whether to install `set_name`, with install preselected (Enter
/// accepts). Answering `n`, an EOF, or a terminal read error declines;
/// unattended callers never reach here.
pub fn prompt_install_set(set_name: &str) -> bool {
    crate::terminal::confirm_yes_no(
        &format!("Install plugin set '{set_name}'?"),
        crate::terminal::ConfirmDefault::Yes,
    )
    .unwrap_or(Some(false))
    .unwrap_or(true)
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
    if let Some(lines) = offer_lines(
        catalog,
        set_name,
        options.target.triple(),
        &options.host_version,
    ) {
        for line in lines {
            let _ = writeln!(err, "{line}");
        }
    }
    let _ = writeln!(err, "ℹ️  Installing plugin set: {set_name}");
    crate::plugin::migrate_legacy_records_best_effort(&options.store_root);
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
    // An explicit install also settles the offer: a later update on this
    // machine must not offer or install the set again. A damaged record is
    // replaced here, since the operator just chose the set.
    let mut offers = SetOffers::load(&options.store_root).unwrap_or_default();
    record_offer(&mut offers, None, options, catalog, set_name);
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
    /// Offer this set when attended; install is preselected (default Yes).
    Offer(String),
    /// Install this set without prompting because no one chose either way
    /// (unattended runs): the catalog's default. A catalog that publishes no
    /// such set (or an empty one) makes this a quiet skip, never an error.
    InstallDefault(String),
    /// Skip optional plugin installation and say so.
    Skip,
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
        return OptionalPluginPlan::Skip;
    }
    if !attended {
        return OptionalPluginPlan::InstallDefault(RECOMMENDED_SET.to_string());
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

/// Windows finishes the host install after this process exits, so it cannot
/// reach the normal optional phase. Never silently discard a selected set.
pub fn handoff_optional_plugin_lines(plan: &OptionalPluginPlan) -> Vec<String> {
    let (name, action) = match plan {
        OptionalPluginPlan::InstallSet(name) | OptionalPluginPlan::InstallDefault(name) => {
            (name, "was not installed")
        }
        OptionalPluginPlan::Offer(name) => (name, "was not offered"),
        OptionalPluginPlan::Skip => return Vec::new(),
    };
    vec![
        format!(
            "⚠️  Optional plugin set '{name}' {action}; the Windows host installer must finish first."
        ),
        format!("After the installer finishes: mesh-llm plugins sets install {name}"),
    ]
}

/// Run the optional-plugin phase of setup/update after the core work
/// succeeded. Optional failure never fails the host result: it warns, gives
/// retry guidance, and reports the installation as incomplete. `prompt`
/// answers "install set NAME?" for an attended offer; an explicit
/// `--plugin-set` selection never asks it.
///
/// A defaulted or offered set installs once per machine. The per-machine
/// record (`SetOffers`) says whether this machine was already offered it:
/// when it was, nothing installs or prompts (a member the operator removed
/// stays removed) and an attended run only learns of members the catalog
/// added since. An explicit `--plugin-set` always installs, and records.
pub async fn run_optional_plugin_phase(
    plan: &OptionalPluginPlan,
    host_version: &str,
    prompt: &mut dyn FnMut(&str) -> bool,
) -> Result<OptionalPluginResult> {
    let mut err = mesh_llm_events::console_err();
    let name = match plan {
        OptionalPluginPlan::Skip => {
            let _ = writeln!(err, "ℹ️  Optional plugin installation skipped.");
            return Ok(OptionalPluginResult::Skipped);
        }
        OptionalPluginPlan::InstallSet(name)
        | OptionalPluginPlan::Offer(name)
        | OptionalPluginPlan::InstallDefault(name) => name,
    };
    let mut options = match PluginInstallOptions::from_env() {
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
    // An update may already have replaced the executable on disk; this
    // process still contains the old release's build information.
    options.host_version = host_version.to_string();
    run_phase_with(
        plan,
        name,
        &options,
        async || fetch_catalog(&options).await,
        prompt,
    )
    .await
}

/// The phase against explicit options and a catalog source, so tests drive it
/// with a temporary store and a local catalog.
async fn run_phase_with(
    plan: &OptionalPluginPlan,
    name: &str,
    options: &PluginInstallOptions,
    fetch: impl AsyncFn() -> Result<PluginCatalog>,
    prompt: &mut dyn FnMut(&str) -> bool,
) -> Result<OptionalPluginResult> {
    let mut err = mesh_llm_events::console_err();
    let defaulted = !matches!(plan, OptionalPluginPlan::InstallSet(_));
    let mut offers = match SetOffers::load(&options.store_root) {
        Ok(offers) => offers,
        // The operator chose this set explicitly: replace a damaged record.
        Err(_) if !defaulted => SetOffers::default(),
        Err(error) => {
            // A damaged record must never cause a reinstall: skip and say why.
            let path = SetOffers::path(&options.store_root);
            let _ = writeln!(
                err,
                "⚠️  Could not read the plugin set record ({error:#}); not installing plugin sets. \
                 `mesh-llm plugins sets install {name}` rewrites it, or delete {} to be offered again.",
                path.display()
            );
            return Ok(OptionalPluginResult::Skipped);
        }
    };
    if defaulted && let Some(offered) = offers.offered(name) {
        let offered = offered.to_vec();
        if matches!(plan, OptionalPluginPlan::Offer(_)) {
            notify_added_members(options, &mut offers, name, &offered, &fetch).await;
        }
        return Ok(OptionalPluginResult::Skipped);
    }
    let catalog = match fetch().await {
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
    if defaulted && let Some(member) = legacy_refused_member(options, &catalog, name) {
        // The old bundled-default installer recorded the operator's refusal as
        // a `default-off` marker. Honor it as this machine's answer, and
        // migrate it into the record so it needs no marker lookup again.
        let _ = writeln!(
            err,
            "ℹ️  Plugin set '{name}' skipped: {member} was turned off or removed under the \
             previous installer. `mesh-llm plugins sets install {name}` installs it anyway."
        );
        record_offer(&mut offers, None, options, &catalog, name);
        return Ok(OptionalPluginResult::Skipped);
    }
    match offer_gate(plan, &catalog, name, options, prompt) {
        OfferGate::Absent => Ok(OptionalPluginResult::Skipped),
        OfferGate::Declined => {
            // Declining is an answer: this machine was offered the set.
            record_offer(&mut offers, None, options, &catalog, name);
            Ok(OptionalPluginResult::Skipped)
        }
        OfferGate::Proceed => {
            Ok(install_and_settle(defaulted, offers, &catalog, name, options).await)
        }
    }
}

/// Install the set and settle the offer. A default install is only valid
/// while nobody has recorded an answer: it takes the record lock, reloads, and
/// keeps the lock through the install and its record, so a refusal recorded
/// while this run waited for the catalog is honored and a concurrent run
/// waits for this one's answer. An explicit install needs no such check.
async fn install_and_settle(
    defaulted: bool,
    mut offers: SetOffers,
    catalog: &PluginCatalog,
    name: &str,
    options: &PluginInstallOptions,
) -> OptionalPluginResult {
    let guard = if defaulted {
        match SetOffers::lock_and_reload(&options.store_root) {
            Ok((_, latest)) if latest.offered(name).is_some() => {
                return OptionalPluginResult::Skipped;
            }
            Ok((guard, latest)) => {
                offers = latest;
                Some(guard)
            }
            Err(error) => {
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(
                    err,
                    "⚠️  Could not read the plugin set record ({error:#}); not installing plugin sets."
                );
                return OptionalPluginResult::Skipped;
            }
        }
    } else {
        None
    };
    let result = install_and_report(catalog, name, options).await;
    // A complete install (or one with nothing left to install) settles the
    // offer; an incomplete one stays unrecorded so a later run retries.
    if settles_offer(&result) {
        record_offer(&mut offers, guard.as_ref(), options, catalog, name);
    }
    result
}

/// A member of `set` the operator turned off or removed under the retired
/// bundled-default installer (its `default-off` marker in the plugin store).
fn legacy_refused_member(
    options: &PluginInstallOptions,
    catalog: &PluginCatalog,
    set: &str,
) -> Option<String> {
    let store = PluginStore::new(&options.store_root);
    set_members(catalog, set)
        .into_iter()
        .find(|member| store.default_turned_off(member))
}

/// Whether a finished install settles the offer: complete or nothing left to
/// install does; an incomplete one stays unrecorded so a later run retries.
fn settles_offer(result: &OptionalPluginResult) -> bool {
    !matches!(result, OptionalPluginResult::Incomplete(_))
}

/// An already-offered set on an attended run: show members the catalog added
/// since, once, and record them as seen. A catalog that cannot be read
/// changes nothing (there is no install to report as incomplete).
async fn notify_added_members(
    options: &PluginInstallOptions,
    offers: &mut SetOffers,
    name: &str,
    offered: &[String],
    fetch: &impl AsyncFn() -> Result<PluginCatalog>,
) {
    let Ok(catalog) = fetch().await else {
        return;
    };
    let added = members_added_since(offered, &installable_members(options, &catalog, name));
    if added.is_empty() {
        return;
    }
    let mut err = mesh_llm_events::console_err();
    for line in added_members_notice(name, &added) {
        let _ = writeln!(err, "{line}");
    }
    record_offer(offers, None, options, &catalog, name);
}

/// Record every current member of `set` as offered; a failed write only warns,
/// because the install itself already happened.
pub(crate) fn record_offer(
    offers: &mut SetOffers,
    guard: Option<&RecordLock>,
    options: &PluginInstallOptions,
    catalog: &PluginCatalog,
    set: &str,
) {
    let members = installable_members(options, catalog, set);
    let written = match guard {
        Some(guard) => offers.record_held(guard, &options.store_root, set, &members),
        None => offers.record(&options.store_root, set, &members),
    };
    if let Err(error) = written {
        let mut err = mesh_llm_events::console_err();
        let _ = writeln!(
            err,
            "⚠️  Could not record the {set} plugin set offer ({error:#}); a later update may offer it again."
        );
    }
}

/// The members of `set` this machine could install: pinned for its platform
/// and compatible with its host. Only these count as offered, so a member
/// with no archive here yet (a Windows build still to ship) stays un-offered
/// and shows as a new member once it can install.
fn installable_members(
    options: &PluginInstallOptions,
    catalog: &PluginCatalog,
    set: &str,
) -> Vec<String> {
    mesh_llm_plugin_manager::resolve_set(
        catalog,
        set,
        options.target.triple(),
        &options.host_version,
    )
    .map(|resolution| {
        resolution
            .members
            .into_iter()
            .filter(|member| member.availability == MemberAvailability::Available)
            .map(|member| member.name)
            .collect()
    })
    .unwrap_or_default()
}

/// The member names the catalog publishes for `set`; empty when it has none.
fn set_members(catalog: &PluginCatalog, set: &str) -> Vec<String> {
    catalog
        .find_set(set)
        .map(|set| set.plugins.clone())
        .unwrap_or_default()
}

/// Install the set and print what happened to every member.
async fn install_and_report(
    catalog: &PluginCatalog,
    name: &str,
    options: &PluginInstallOptions,
) -> OptionalPluginResult {
    report_optional_set_result(name, install_set_quiet(catalog, name, options).await)
}

/// Core setup/update has already succeeded: even an invalid optional
/// selection must be reported without discarding that result or its summary.
fn report_optional_set_result(
    name: &str,
    result: Result<SetInstallResult>,
) -> OptionalPluginResult {
    let mut err = mesh_llm_events::console_err();
    match result {
        Ok(result) => {
            // Every member is accounted for: what landed, what was left as it
            // is (and why), and what failed.
            for (skipped_name, reason) in &result.skipped {
                let _ = writeln!(err, "ℹ️  {skipped_name} skipped: {reason}");
            }
            if !result.failed.is_empty() {
                for (failed_name, reason) in &result.failed {
                    let _ = writeln!(
                        err,
                        "⚠️  Could not install {failed_name}: {reason}. Optional plugin \
                         installation incomplete."
                    );
                }
                let _ = writeln!(err, "ℹ️  Retry: mesh-llm plugins sets install {name}");
                return OptionalPluginResult::Incomplete(result.installed_count());
            }
            if result.installed_count() == 0 {
                let _ = writeln!(err, "ℹ️  No member of this set needed installing.");
                return OptionalPluginResult::Skipped;
            }
            let _ = writeln!(
                err,
                "✅ Plugin set installed: {} installed.",
                result.installed_count()
            );
            OptionalPluginResult::Installed(result.installed_count())
        }
        Err(error) => {
            let _ = writeln!(
                err,
                "⚠️  {error:#}. Optional plugin installation incomplete."
            );
            let _ = writeln!(err, "ℹ️  Retry: mesh-llm plugins sets install {name}");
            OptionalPluginResult::Incomplete(0)
        }
    }
}

/// How an optional install gate resolved.
#[derive(Debug, Eq, PartialEq)]
enum OfferGate {
    /// Install the set.
    Proceed,
    /// The operator answered no to an attended offer.
    Declined,
    /// The catalog publishes no such set with members; nothing to offer.
    Absent,
}

/// Decide whether an optional install proceeds, printing what the operator
/// should see first. An explicit set proceeds; a defaulted or offered set
/// proceeds only when the catalog publishes it with members, and an offer also
/// needs the prompt's yes. `InstallDefault` never asks, so a missing or empty
/// set is a quiet skip, never an error.
fn offer_gate(
    plan: &OptionalPluginPlan,
    catalog: &PluginCatalog,
    name: &str,
    options: &PluginInstallOptions,
    prompt: &mut dyn FnMut(&str) -> bool,
) -> OfferGate {
    let mut err = mesh_llm_events::console_err();
    if !matches!(
        plan,
        OptionalPluginPlan::Offer(_) | OptionalPluginPlan::InstallDefault(_)
    ) {
        let _ = writeln!(err, "ℹ️  Installing plugin set: {name}");
        return OfferGate::Proceed;
    }
    let Some(lines) = offer_lines(
        catalog,
        name,
        options.target.triple(),
        &options.host_version,
    ) else {
        let _ = writeln!(
            err,
            "ℹ️  No members of plugin set '{name}' can be installed for this platform and host; \
             nothing to offer."
        );
        return OfferGate::Absent;
    };
    for line in lines {
        let _ = writeln!(err, "{line}");
    }
    if matches!(plan, OptionalPluginPlan::InstallDefault(_)) || prompt(name) {
        return OfferGate::Proceed;
    }
    let _ = writeln!(err, "ℹ️  Optional plugin installation skipped.");
    OfferGate::Declined
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
    fn the_offer_names_the_set_description_and_eligible_members() {
        let lines = offer_lines(
            &catalog(),
            "recommended",
            "x86_64-unknown-linux-gnu",
            "0.90.0",
        )
        .unwrap();
        assert_eq!(
            lines[0],
            "ℹ️  recommended plugin set — Recommended plugins for MeshLLM"
        );
        assert!(lines[1].contains("capsules v0.2.0"), "{lines:?}");
        assert!(offer_lines(&catalog(), "missing", "x86_64-unknown-linux-gnu", "0.90.0").is_none());
    }

    #[test]
    fn sets_with_no_platform_or_host_compatible_members_are_not_offered() {
        assert!(offer_lines(&catalog(), "recommended", "aarch64-apple-darwin", "0.90.0").is_none());
        let entries = ENTRIES.replace(
            "\"version\":\"v0.2.0\"",
            "\"version\":\"v0.2.0\",\"compatibility\":{\"max\":\"0.89.0\"}",
        );
        let catalog = PluginCatalog::parse_jsonl(&entries)
            .unwrap()
            .with_sets(PluginCatalog::parse_sets_jsonl(SETS).unwrap())
            .unwrap();
        assert!(
            offer_lines(
                &catalog,
                "recommended",
                "x86_64-unknown-linux-gnu",
                "0.90.0"
            )
            .is_none()
        );
        assert!(
            offer_lines(
                &catalog,
                "recommended",
                "x86_64-unknown-linux-gnu",
                "0.89.0"
            )
            .is_some()
        );
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
        assert!(offer_lines(&catalog, "empty", "x86_64-unknown-linux-gnu", "0.90.0").is_none());
    }

    #[test]
    fn the_default_install_never_prompts_and_skips_a_missing_set_quietly() {
        let root = tempfile::tempdir().unwrap();
        let options = phase_options(root.path());
        let mut never = |_: &str| -> bool { panic!("InstallDefault must not prompt") };
        let default = OptionalPluginPlan::InstallDefault("recommended".into());
        assert_eq!(
            offer_gate(&default, &catalog(), "recommended", &options, &mut never),
            OfferGate::Proceed
        );
        let missing = OptionalPluginPlan::InstallDefault("missing".into());
        assert_eq!(
            offer_gate(&missing, &catalog(), "missing", &options, &mut never),
            OfferGate::Absent
        );
    }

    #[test]
    fn an_offer_installs_only_on_yes() {
        let root = tempfile::tempdir().unwrap();
        let options = phase_options(root.path());
        let offer = OptionalPluginPlan::Offer("recommended".into());
        assert_eq!(
            offer_gate(&offer, &catalog(), "recommended", &options, &mut |_| true),
            OfferGate::Proceed
        );
        assert_eq!(
            offer_gate(&offer, &catalog(), "recommended", &options, &mut |_| false),
            OfferGate::Declined
        );
    }

    #[test]
    fn explicit_selection_wins_and_unattended_defaults_to_the_recommended_set() {
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
            OptionalPluginPlan::Skip
        );
        assert_eq!(
            optional_plugin_plan(None, false, true, false),
            OptionalPluginPlan::Skip,
            "the legacy environment default skips optional installation"
        );
        assert_eq!(
            optional_plugin_plan(None, false, false, false),
            OptionalPluginPlan::InstallDefault(RECOMMENDED_SET.into()),
            "unattended installs the recommended set by default"
        );
        assert_eq!(
            optional_plugin_plan(None, false, false, true),
            OptionalPluginPlan::Offer(RECOMMENDED_SET.into())
        );
    }

    #[tokio::test]
    async fn an_unrelated_corrupt_record_does_not_block_set_installation() {
        use mesh_llm_plugin_manager::{InstalledPluginMetadata, PluginStore, PluginTarget};

        let root = tempfile::tempdir().unwrap();
        let options = PluginInstallOptions {
            store_root: root.path().join("store"),
            install_root: root.path().join("installed"),
            catalog_url: "unused".into(),
            target: PluginTarget::from_os_arch("linux", "x86_64").unwrap(),
            host_version: "0.90.0".into(),
        };
        let store = PluginStore::new(&options.store_root);
        let installed = InstalledPluginMetadata {
            name: "capsules".into(),
            source_repository: "https://github.com/Mesh-LLM/capsules".into(),
            installed_version: "v0.2.0".into(),
            target_triple: options.target.triple().into(),
            downloaded_asset_name: "capsules.tar.gz".into(),
            install_path: options.install_root.join("capsules"),
            enabled: false,
            default_managed: false,
            host_compatibility: None,
            allowed_surfaces: Some(Vec::new()),
            manifest: None,
            last_protocol_version: None,
            last_status: None,
            last_error: None,
            verified_sha256: None,
        };
        store.save(&installed).unwrap();
        let corrupt = options.store_root.join("broken/plugin-install.json");
        std::fs::create_dir_all(corrupt.parent().unwrap()).unwrap();
        std::fs::write(&corrupt, b"invalid metadata").unwrap();
        assert!(store.migrate_legacy_defaults().is_err());

        let result = install_set_command(&catalog(), "recommended", &options)
            .await
            .unwrap();

        assert!(result.is_complete());
        assert_eq!(result.installed_count(), 0);
        assert_eq!(result.skipped.len(), 1);
        assert!(result.skipped[0].1.contains("disabled"));
        assert_eq!(store.load("capsules").unwrap(), installed);
        assert_eq!(std::fs::read(corrupt).unwrap(), b"invalid metadata");
    }

    #[tokio::test]
    async fn an_unknown_optional_set_is_incomplete_but_the_standalone_command_fails() {
        let root = tempfile::tempdir().unwrap();
        let options = PluginInstallOptions {
            store_root: root.path().join("store"),
            install_root: root.path().join("installed"),
            catalog_url: "unused".into(),
            target: mesh_llm_plugin_manager::PluginTarget::from_os_arch("linux", "x86_64").unwrap(),
            host_version: "0.90.0".into(),
        };
        let error = install_set_command(&catalog(), "missing", &options)
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("Unknown plugin set 'missing'"));
        assert_eq!(
            report_optional_set_result("missing", Err(error)),
            OptionalPluginResult::Incomplete(0)
        );
    }

    #[test]
    fn windows_handoff_reports_selected_sets_and_the_post_installer_command() {
        let explicit =
            handoff_optional_plugin_lines(&OptionalPluginPlan::InstallSet("extra".into()));
        assert!(explicit[0].contains("'extra' was not installed"));
        assert_eq!(
            explicit[1],
            "After the installer finishes: mesh-llm plugins sets install extra"
        );
        let offer = handoff_optional_plugin_lines(&OptionalPluginPlan::Offer("recommended".into()));
        assert!(offer[0].contains("was not offered"));
        assert!(offer[1].ends_with("mesh-llm plugins sets install recommended"));
        let default = handoff_optional_plugin_lines(&OptionalPluginPlan::InstallDefault(
            "recommended".into(),
        ));
        assert!(default[0].contains("was not installed"));
        assert!(handoff_optional_plugin_lines(&OptionalPluginPlan::Skip).is_empty());
    }

    // The install-once phase, driven against a temporary store and a local
    // catalog so no test needs the network.

    const PHASE_ENTRIES: &str = r#"{"name":"capsules","description":"Audit","github_url":"https://github.com/Mesh-LLM/capsules","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v0.2.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"}}
{"name":"notes","description":"Notes","github_url":"https://github.com/Mesh-LLM/notes","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v1.0.0","sha256":{"x86_64-unknown-linux-gnu":"abababababababababababababababababababababababababababababababab"}}
"#;

    fn phase_catalog(members: &[&str]) -> PluginCatalog {
        let sets = format!(
            r#"{{"name":"recommended","description":"Recommended","plugins":{}}}"#,
            serde_json::to_string(members).unwrap()
        );
        PluginCatalog::parse_jsonl(PHASE_ENTRIES)
            .unwrap()
            .with_sets(PluginCatalog::parse_sets_jsonl(&sets).unwrap())
            .unwrap()
    }

    fn phase_options(root: &std::path::Path) -> PluginInstallOptions {
        PluginInstallOptions {
            store_root: root.to_path_buf(),
            install_root: root.join("installed"),
            catalog_url: "http://127.0.0.1:9/unreachable".into(),
            target: mesh_llm_plugin_manager::PluginTarget::from_os_arch("linux", "x86_64").unwrap(),
            host_version: "0.90.0".into(),
        }
    }

    fn install_marker(root: &std::path::Path, name: &str) {
        PluginStore::new(root)
            .save(&mesh_llm_plugin_manager::InstalledPluginMetadata {
                name: name.into(),
                source_repository: format!("https://github.com/mesh-llm/{name}"),
                installed_version: "v1.0.0".into(),
                target_triple: "x86_64-unknown-linux-gnu".into(),
                downloaded_asset_name: format!("{name}.tar.gz"),
                install_path: root.join("installed").join(name),
                enabled: true,
                default_managed: false,
                host_compatibility: None,
                allowed_surfaces: None,
                manifest: None,
                last_protocol_version: None,
                last_status: None,
                last_error: None,
                verified_sha256: None,
            })
            .unwrap();
    }

    fn offered(root: &std::path::Path) -> Option<Vec<String>> {
        SetOffers::load(root)
            .unwrap()
            .offered("recommended")
            .map(<[String]>::to_vec)
    }

    async fn run_phase(
        plan: &OptionalPluginPlan,
        root: &std::path::Path,
        catalog: &PluginCatalog,
        fetches: &std::cell::Cell<u32>,
        answer: bool,
    ) -> OptionalPluginResult {
        run_phase_with(
            plan,
            "recommended",
            &phase_options(root),
            async || {
                fetches.set(fetches.get() + 1);
                Ok(catalog.clone())
            },
            &mut |_| answer,
        )
        .await
        .unwrap()
    }

    fn default_plan() -> OptionalPluginPlan {
        OptionalPluginPlan::InstallDefault("recommended".into())
    }

    fn offer_plan() -> OptionalPluginPlan {
        OptionalPluginPlan::Offer("recommended".into())
    }

    #[tokio::test]
    async fn the_first_default_run_settles_the_offer_for_this_machine() {
        let temp = tempfile::tempdir().unwrap();
        install_marker(temp.path(), "capsules");
        let fetches = std::cell::Cell::new(0);
        let catalog = phase_catalog(&["capsules"]);
        run_phase(&default_plan(), temp.path(), &catalog, &fetches, true).await;
        assert_eq!(offered(temp.path()), Some(vec!["capsules".to_string()]));
    }

    #[tokio::test]
    async fn a_refusal_recorded_while_a_default_run_waits_for_the_catalog_wins() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = phase_catalog(&["capsules"]);
        let root = temp.path().to_path_buf();
        // The unattended run loaded an empty record; while it waits for the
        // catalog, an attended setup declines the set and records the refusal.
        let result = run_phase_with(
            &default_plan(),
            "recommended",
            &phase_options(temp.path()),
            async || {
                SetOffers::default()
                    .record(&root, "recommended", &["capsules".to_string()])
                    .unwrap();
                Ok(catalog.clone())
            },
            &mut |_| true,
        )
        .await
        .unwrap();
        assert!(
            matches!(result, OptionalPluginResult::Skipped),
            "the recorded refusal must stop the install, got {result:?}"
        );
        assert!(
            PluginStore::new(temp.path()).load("capsules").is_err(),
            "nothing may be installed after a refusal"
        );
    }

    #[tokio::test]
    async fn a_legacy_default_off_marker_is_a_refusal_for_default_and_offer_runs() {
        for plan in [default_plan(), offer_plan()] {
            let temp = tempfile::tempdir().unwrap();
            PluginStore::new(temp.path())
                .set_default_turned_off("capsules", true)
                .unwrap();
            let fetches = std::cell::Cell::new(0);
            let catalog = phase_catalog(&["capsules"]);
            let result = run_phase(&plan, temp.path(), &catalog, &fetches, true).await;
            assert_eq!(result, OptionalPluginResult::Skipped);
            assert!(
                PluginStore::new(temp.path())
                    .load_optional("capsules")
                    .unwrap()
                    .is_none(),
                "an operator who deleted capsules under the old installer is not reinstalled"
            );
            assert_eq!(offered(temp.path()), Some(vec!["capsules".to_string()]));
        }
    }

    #[tokio::test]
    async fn an_explicit_set_installs_despite_a_legacy_default_off_marker() {
        let temp = tempfile::tempdir().unwrap();
        install_marker(temp.path(), "capsules");
        PluginStore::new(temp.path())
            .set_default_turned_off("capsules", true)
            .unwrap();
        let fetches = std::cell::Cell::new(0);
        let catalog = phase_catalog(&["capsules"]);
        let plan = OptionalPluginPlan::InstallSet("recommended".into());
        run_phase(&plan, temp.path(), &catalog, &fetches, true).await;
        assert_eq!(
            fetches.get(),
            1,
            "the explicit set is not blocked by the marker"
        );
    }

    #[tokio::test]
    async fn a_member_with_no_archive_for_this_machine_stays_un_offered() {
        let temp = tempfile::tempdir().unwrap();
        // `notes` has no archive for this platform (as capsules on Windows
        // before its build ships); `capsules` does.
        let entries = PHASE_ENTRIES.replace(
            r#""name":"notes","description":"Notes","github_url":"https://github.com/Mesh-LLM/notes","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v1.0.0","sha256":{"x86_64-unknown-linux-gnu""#,
            r#""name":"notes","description":"Notes","github_url":"https://github.com/Mesh-LLM/notes","author_email":"m@meshllm.cloud","author_name":"Mesh LLM","version":"v1.0.0","sha256":{"x86_64-pc-windows-msvc""#,
        );
        assert_ne!(entries, PHASE_ENTRIES);
        let sets =
            r#"{"name":"recommended","description":"Recommended","plugins":["capsules","notes"]}"#;
        let catalog = PluginCatalog::parse_jsonl(&entries)
            .unwrap()
            .with_sets(PluginCatalog::parse_sets_jsonl(sets).unwrap())
            .unwrap();
        install_marker(temp.path(), "capsules");
        let fetches = std::cell::Cell::new(0);
        run_phase(&default_plan(), temp.path(), &catalog, &fetches, true).await;
        assert_eq!(
            offered(temp.path()),
            Some(vec!["capsules".to_string()]),
            "only the member this machine could install is recorded"
        );
        // Once the member can install here, an attended run shows it as new.
        let shipped = phase_catalog(&["capsules", "notes"]);
        run_phase(&offer_plan(), temp.path(), &shipped, &fetches, true).await;
        assert_eq!(
            offered(temp.path()),
            Some(vec!["capsules".to_string(), "notes".to_string()])
        );
    }

    #[tokio::test]
    async fn an_uninstalled_member_is_not_reinstalled_by_a_later_update() {
        let temp = tempfile::tempdir().unwrap();
        SetOffers::default()
            .record(temp.path(), "recommended", &["capsules".to_string()])
            .unwrap();
        let fetches = std::cell::Cell::new(0);
        let catalog = phase_catalog(&["capsules"]);
        for plan in [default_plan(), offer_plan()] {
            let result = run_phase(&plan, temp.path(), &catalog, &fetches, true).await;
            assert_eq!(result, OptionalPluginResult::Skipped);
        }
        assert!(
            PluginStore::new(temp.path())
                .load_optional("capsules")
                .unwrap()
                .is_none(),
            "a removed member stays removed"
        );
    }

    #[tokio::test]
    async fn an_offered_set_never_prompts_again() {
        let temp = tempfile::tempdir().unwrap();
        let fetches = std::cell::Cell::new(0);
        let catalog = phase_catalog(&["capsules"]);
        // Declining the first offer still counts as having been offered.
        run_phase(&offer_plan(), temp.path(), &catalog, &fetches, false).await;
        assert_eq!(offered(temp.path()), Some(vec!["capsules".to_string()]));
        let result = run_phase_with(
            &offer_plan(),
            "recommended",
            &phase_options(temp.path()),
            async || Ok(catalog.clone()),
            &mut |_| panic!("an offered set must not prompt again"),
        )
        .await
        .unwrap();
        assert_eq!(result, OptionalPluginResult::Skipped);
    }

    #[tokio::test]
    async fn members_added_later_are_a_notice_once_and_never_an_unattended_install() {
        let temp = tempfile::tempdir().unwrap();
        SetOffers::default()
            .record(temp.path(), "recommended", &["capsules".to_string()])
            .unwrap();
        let grown = phase_catalog(&["capsules", "notes"]);
        let fetches = std::cell::Cell::new(0);

        // Unattended: no catalog read, no install, and the new member stays
        // unseen so a later attended run can still show it.
        run_phase(&default_plan(), temp.path(), &grown, &fetches, true).await;
        assert_eq!(fetches.get(), 0);
        assert_eq!(offered(temp.path()), Some(vec!["capsules".to_string()]));

        // Attended: the notice is shown, nothing installs, and the member is
        // recorded as seen.
        let result = run_phase(&offer_plan(), temp.path(), &grown, &fetches, true).await;
        assert_eq!(result, OptionalPluginResult::Skipped);
        assert_eq!(fetches.get(), 1);
        assert!(
            PluginStore::new(temp.path())
                .load_optional("notes")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            offered(temp.path()),
            Some(vec!["capsules".to_string(), "notes".to_string()])
        );
    }

    #[tokio::test]
    async fn an_explicit_plugin_set_installs_even_when_the_machine_was_offered() {
        let temp = tempfile::tempdir().unwrap();
        install_marker(temp.path(), "capsules");
        SetOffers::default()
            .record(temp.path(), "recommended", &["capsules".to_string()])
            .unwrap();
        let fetches = std::cell::Cell::new(0);
        let catalog = phase_catalog(&["capsules"]);
        let plan = OptionalPluginPlan::InstallSet("recommended".into());
        run_phase(&plan, temp.path(), &catalog, &fetches, true).await;
        assert_eq!(
            fetches.get(),
            1,
            "the explicit selection reaches the catalog"
        );
    }

    #[tokio::test]
    async fn a_damaged_record_skips_instead_of_installing() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("set-offers.json"), b"not json").unwrap();
        let fetches = std::cell::Cell::new(0);
        let catalog = phase_catalog(&["capsules"]);
        let result = run_phase(&default_plan(), temp.path(), &catalog, &fetches, true).await;
        assert_eq!(result, OptionalPluginResult::Skipped);
        assert_eq!(fetches.get(), 0);
    }

    #[tokio::test]
    async fn an_explicit_plugin_set_replaces_a_damaged_record() {
        let temp = tempfile::tempdir().unwrap();
        install_marker(temp.path(), "capsules");
        std::fs::write(temp.path().join("set-offers.json"), b"not json").unwrap();
        let fetches = std::cell::Cell::new(0);
        let catalog = phase_catalog(&["capsules"]);
        let plan = OptionalPluginPlan::InstallSet("recommended".into());
        run_phase(&plan, temp.path(), &catalog, &fetches, true).await;
        assert_eq!(fetches.get(), 1, "the explicit selection still installs");
        assert_eq!(offered(temp.path()), Some(vec!["capsules".to_string()]));
    }

    #[test]
    fn only_an_incomplete_install_leaves_the_offer_unrecorded() {
        assert!(settles_offer(&OptionalPluginResult::Skipped));
        assert!(settles_offer(&OptionalPluginResult::Installed(1)));
        assert!(!settles_offer(&OptionalPluginResult::Incomplete(0)));
    }

    #[tokio::test]
    async fn the_explicit_set_command_records_the_offer() {
        let temp = tempfile::tempdir().unwrap();
        install_marker(temp.path(), "capsules");
        install_set_command(
            &phase_catalog(&["capsules"]),
            "recommended",
            &phase_options(temp.path()),
        )
        .await
        .unwrap();
        assert_eq!(offered(temp.path()), Some(vec!["capsules".to_string()]));
    }
}
