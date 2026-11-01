use std::io::Write;
use std::path::Path;

use anyhow::{Result, bail};
use mesh_llm_plugin_manager::InstalledPluginMetadata;
use mesh_llm_plugin_manager::defaults::{
    DEFAULT_PLUGINS, DefaultPluginOutcome, NO_DEFAULT_PLUGINS_ENV, default_plugin,
    default_plugins_opted_out, provision_bundled_defaults, turn_off_defaults,
};
use mesh_llm_plugin_manager::install::install_plugin_archive;
use mesh_llm_plugin_manager::{
    PluginCatalog, PluginInstallOptions, PluginProgressEvent, PluginProgressReporter, PluginStore,
    PluginTarget, default_store_root, install_plugin, update_plugin,
};
use reqwest::Client;
use std::collections::BTreeSet;

use mesh_llm_cli::PluginCommand;
use mesh_llm_tui::terminal_progress::{
    SpinnerHandle, clear_stderr_line, ratio_complete_u64, render_inline_gauge, start_spinner,
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PluginListRows {
    pub externals: Vec<RuntimePluginRow>,
    pub inactive: Vec<InactivePluginRow>,
    /// Capability providers reported by the running node
    /// (`GET /api/plugins/providers`); `None` when no node answered, in which
    /// case binding readiness cannot be judged and is not reported.
    pub providers: Option<Vec<CapabilityProviderRow>>,
}

/// One row of `GET /api/plugins/providers`.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize)]
pub struct CapabilityProviderRow {
    pub capability: String,
    pub plugin_name: String,
    pub available: bool,
}

/// Fetch the running node's capability providers; `None` when it is not
/// running or does not answer.
pub async fn fetch_capability_providers(port: u16) -> Option<Vec<CapabilityProviderRow>> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;
    let response = client
        .get(format!("http://127.0.0.1:{port}/api/plugins/providers"))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.json().await.ok()
}

/// Installed, enabled plugins whose manifest asks for a provider by capability
/// but for which the running node has no provider, or more than one. Such a
/// binding is refused on every call, so the plugin is not ready; the node says
/// the same once at start, this is the persistent view.
fn unresolvable_capability_grants(
    installed: &[InstalledPluginMetadata],
    providers: Option<&[CapabilityProviderRow]>,
) -> Vec<String> {
    let Some(providers) = providers else {
        return Vec::new();
    };
    let mut warnings = Vec::new();
    for plugin in installed.iter().filter(|plugin| plugin.enabled) {
        let Some(manifest) = plugin.manifest.as_ref() else {
            continue;
        };
        for binding in &manifest.operation_bindings {
            let mut candidates: Vec<&str> = providers
                .iter()
                .filter(|provider| {
                    provider.capability == binding.capability
                        && provider.available
                        && provider.plugin_name != plugin.name
                })
                .map(|provider| provider.plugin_name.as_str())
                .collect();
            candidates.sort_unstable();
            candidates.dedup();
            match candidates.as_slice() {
                [_] => {}
                [] => warnings.push(format!(
                    "{} binding {} is not ready: no running plugin serves {}",
                    plugin.name, binding.name, binding.capability
                )),
                several => warnings.push(format!(
                    "{} binding {} is not ready: several plugins serve {} ({}); name one with [[plugin.operation_bindings]]",
                    plugin.name,
                    binding.name,
                    binding.capability,
                    several.join(", ")
                )),
            }
        }
    }
    warnings
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimePluginRow {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InactivePluginRow {
    pub name: String,
    pub kind: String,
    pub status: String,
    pub error: Option<String>,
}

/// Forward one published operation call to the running node, which dispatches
/// it to the plugin and returns the plugin's answer unchanged. The node links
/// nothing plugin-specific: the operation name and input are the plugin's own.
async fn call(plugin: &str, operation: &str, input: &str, port: u16) -> Result<()> {
    let input = if input.trim() == "-" {
        let mut buffer = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buffer)?;
        buffer
    } else {
        input.to_owned()
    };
    let input: serde_json::Value = serde_json::from_str(&input)
        .map_err(|error| anyhow::anyhow!("--input is not JSON: {error}"))?;
    let client = Client::builder().no_proxy().build()?;
    let url = format!("http://127.0.0.1:{port}/api/plugins/{plugin}/tools/{operation}");
    let response = match client.post(&url).json(&input).send().await {
        Ok(response) => response,
        // Never repeat a request after it may have reached the node.
        Err(error) if error.is_connect() => bail!(
            "mesh-llm is not running on port {port}; plugin operations are served by the              running node. Start `mesh-llm` and retry."
        ),
        Err(error) => return Err(error.into()),
    };
    let status = response.status();
    let body = response.text().await?;
    if status.as_u16() == 404 {
        bail!(
            "the node on port {port} has no running plugin '{plugin}' publishing '{operation}': {}",
            body.trim()
        );
    }
    if !status.is_success() {
        bail!("plugin '{plugin}' rejected '{operation}': {}", body.trim());
    }
    let value: serde_json::Value =
        serde_json::from_str(&body).unwrap_or(serde_json::Value::String(body));
    writeln!(
        mesh_llm_events::machine_out(),
        "{}",
        serde_json::to_string_pretty(&value)?
    )?;
    Ok(())
}

pub async fn run_plugin_command(
    command: &PluginCommand,
    runtime_rows: Option<&PluginListRows>,
    config_path: Option<&Path>,
) -> Result<bool> {
    match command {
        PluginCommand::InstallDefaults { off: false } => install_defaults(config_path)?,
        PluginCommand::InstallDefaults { off: true } => turn_off_default_plugins()?,
        PluginCommand::Install {
            reference,
            archive,
            name,
            version,
        } => {
            install(
                reference.as_deref(),
                archive.as_deref(),
                name.as_deref(),
                version.as_deref().unwrap_or("dev"),
            )
            .await?
        }
        PluginCommand::Update { name } => update(name).await?,
        PluginCommand::Enable { name } => set_enabled(name, true)?,
        PluginCommand::Disable { name } => set_enabled(name, false)?,
        PluginCommand::Delete { name } => delete(name)?,
        PluginCommand::Info { name } => return info(name, runtime_rows),
        PluginCommand::Search { query } => search(query.as_deref()).await?,
        PluginCommand::Call {
            plugin,
            operation,
            input,
            port,
        } => call(plugin, operation, input, *port).await?,
        PluginCommand::List => {
            let Some(runtime_rows) = runtime_rows else {
                return Ok(false);
            };
            list(runtime_rows)?;
        }
    }
    Ok(true)
}

/// The plugins the operator runs from config: an entry that says how to start
/// the plugin (`command` or `url`). The console writes a bare `[[plugin]]`
/// entry (a name and its settings) when an operator saves a plugin setting;
/// such an entry configures the installed plugin and does not take it out of
/// default management, so its reviewed pin still moves on update.
fn operator_run_plugins(entries: &[mesh_llm_config::PluginConfigEntry]) -> BTreeSet<String> {
    entries
        .iter()
        .filter(|entry| entry.command.is_some() || entry.url.is_some())
        .map(|entry| entry.name.clone())
        .collect()
}

/// What `plugins install-defaults` says when the environment turns the
/// defaults off, instead of a silent exit 0.
fn turned_off_by_environment_line() -> String {
    format!("ℹ️  Default plugins not installed: {NO_DEFAULT_PLUGINS_ENV} turns them off here")
}

fn configured_operator_plugins(config_path: Option<&Path>) -> Result<BTreeSet<String>> {
    let config = mesh_llm_config::load_config(config_path)?;
    Ok(operator_run_plugins(&config.plugins))
}

/// Install the defaults from this release's bundled copy; never downloads.
fn install_defaults(config_path: Option<&Path>) -> Result<()> {
    if default_plugins_opted_out() {
        writeln!(
            mesh_llm_events::console_err(),
            "{}",
            turned_off_by_environment_line()
        )?;
        return Ok(());
    }
    if DEFAULT_PLUGINS.is_empty() {
        return Ok(());
    }
    let options = PluginInstallOptions::from_env()?;
    let configured = configured_operator_plugins(config_path)?;
    let mut progress = CliPluginProgress::default();
    let outcomes =
        provision_bundled_defaults(DEFAULT_PLUGINS, &configured, &options, &mut progress);
    progress.finish();
    let mut err = mesh_llm_events::console_err();
    let mut failed = false;
    for (name, outcome) in outcomes {
        match outcome {
            DefaultPluginOutcome::Installed(installed) => writeln!(
                err,
                "✅ Installed default {name} {}",
                installed.metadata.installed_version
            )?,
            DefaultPluginOutcome::AlreadyCurrent => {}
            other @ (DefaultPluginOutcome::OperatorManaged
            | DefaultPluginOutcome::Disabled
            | DefaultPluginOutcome::TurnedOff
            | DefaultPluginOutcome::UnsupportedPlatform
            | DefaultPluginOutcome::NotBundled) => {
                if let Some(line) = left_alone_line(name, &other) {
                    writeln!(err, "{line}")?;
                }
            }
            DefaultPluginOutcome::NotInstalled(reason) => {
                writeln!(err, "⚠️ Default {name} not installed: {reason}")?;
                failed = true;
            }
        }
    }
    if failed {
        bail!("one or more default plugins could not be provisioned");
    }
    Ok(())
}

/// `plugins install-defaults --off`, and `mesh-llm update --no-default-plugins`:
/// turn every default off, so neither a node start nor `install-defaults`
/// installs it from the bundled copy until `plugins enable NAME`.
pub fn turn_off_default_plugins() -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    let mut err = mesh_llm_events::console_err();
    for name in turn_off_defaults(DEFAULT_PLUGINS, &store)? {
        writeln!(
            err,
            "⏸️  Default {name} turned off: it is not installed or loaded until \
             `mesh-llm plugins enable {name}`"
        )?;
    }
    Ok(())
}

async fn install(
    reference: Option<&str>,
    archive: Option<&std::path::Path>,
    name: Option<&str>,
    version: &str,
) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut progress = CliPluginProgress::default();
    let outcome = match (reference, archive, name) {
        (Some(reference), None, None) => install_plugin(reference, &options, &mut progress).await?,
        (None, Some(archive), Some(name)) => {
            install_plugin_archive(name, version, archive, &options, &mut progress)?
        }
        _ => bail!("provide either a plugin reference or --archive with --name"),
    };
    // The package's granted access was disclosed as it landed
    // (`CliPluginProgress`).
    progress.finish();
    if outcome.changed {
        let mut err = mesh_llm_events::console_err();
        writeln!(
            err,
            "✅ Installed {} {}",
            outcome.metadata.name, outcome.metadata.installed_version
        )?;
    }
    Ok(())
}

async fn update(name: &str) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut progress = CliPluginProgress::default();
    let outcome = update_plugin(name, &options, &mut progress).await?;
    progress.finish();
    if outcome.changed {
        let mut err = mesh_llm_events::console_err();
        writeln!(
            err,
            "✅ Updated {} to {}",
            outcome.metadata.name, outcome.metadata.installed_version
        )?;
    } else {
        // Nothing landed, so nothing was disclosed on the way: say what the
        // installed version has.
        report_exchange_access(&outcome.metadata)?;
    }
    Ok(())
}

fn set_enabled(name: &str, enabled: bool) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    // The target a node start installs defaults for; none on a platform the
    // plugin manager does not support.
    let target = PluginTarget::current().ok();
    set_enabled_in(
        &store,
        name,
        enabled,
        target.as_ref().map(PluginTarget::triple),
        &mut mesh_llm_events::console_err(),
    )
}

fn set_enabled_in(
    store: &PluginStore,
    name: &str,
    enabled: bool,
    target: Option<&str>,
    err: &mut impl Write,
) -> Result<()> {
    // A default that is not installed (for example right after a delete) is
    // turned off, or back on, by a record a node start respects.
    // Enabling a default always removes that record, installed or not, so a
    // stale one never outlives an install.
    let default = default_plugin(name);
    let is_default = default.is_some();
    if is_default && enabled {
        store.set_default_turned_off(name, false)?;
    }
    if is_default && store.load_optional(name)?.is_none() {
        // A start installs a default only on a target this build pins it for.
        let pinned = default
            .zip(target)
            .is_some_and(|(default, target)| default.has_pin_for(target));
        if enabled && pinned {
            writeln!(
                err,
                "✅ Enabled {name}: the node installs it from its bundled copy when it next starts"
            )?;
        } else if enabled {
            // Say what `plugins install-defaults` says: the next start installs nothing.
            writeln!(err, "✅ Enabled {name}")?;
            writeln!(err, "{}", no_reviewed_release_line(name))?;
        } else {
            store.set_default_turned_off(name, true)?;
            writeln!(
                err,
                "⏸️  Disabled {name}: the node will not install it from its bundled copy again"
            )?;
        }
        return Ok(());
    }
    let metadata = store.set_enabled(name, enabled)?;
    if metadata.enabled {
        writeln!(err, "✅ Enabled {}", metadata.name)?;
    } else {
        writeln!(err, "⏸️  Disabled {}", metadata.name)?;
    }
    Ok(())
}

fn report_exchange_access(
    metadata: &mesh_llm_plugin_manager::InstalledPluginMetadata,
) -> Result<()> {
    if let Some(summary) = exchange_access_summary(metadata) {
        writeln!(mesh_llm_events::console_err(), "{summary}")?;
    }
    Ok(())
}

/// The installation notice for what `metadata`'s package was granted; none
/// for a package that requests no OpenAI lifecycle access.
fn exchange_access_summary(
    metadata: &mesh_llm_plugin_manager::InstalledPluginMetadata,
) -> Option<String> {
    metadata
        .manifest
        .as_ref()
        .and_then(|manifest| manifest.openai_exchange_hook.as_ref())
        .map(|access| access.access_summary())
}

/// What an update asks for that the version it replaces did not. The store
/// keeps the manifest each version was installed with, which is the record
/// of what the operator accepted; an update that widens it is called out,
/// including for a dependency pulled in by `requires`, instead of being
/// granted quietly because the earlier install was.
fn access_increase_lines(
    previous: &mesh_llm_plugin_manager::InstalledPluginMetadata,
    installed: &mesh_llm_plugin_manager::InstalledPluginMetadata,
) -> Vec<String> {
    let mut lines = Vec::new();
    let hook = |metadata: &mesh_llm_plugin_manager::InstalledPluginMetadata| {
        metadata
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.openai_exchange_hook.clone())
    };
    let increases = match (hook(previous), hook(installed)) {
        (Some(before), Some(now)) => now.increases_over(&before),
        (None, Some(now)) => now.increases_over(
            &mesh_llm_plugin_manager::store::InstalledOpenAiExchangeAccess::default(),
        ),
        _ => Vec::new(),
    };
    if !increases.is_empty() {
        lines.push(format!(
            "⚠️  {} now asks for OpenAI lifecycle access it did not have at version {}: {}. Narrow or replace it with a [[plugin]] entry if that is not wanted.",
            installed.name,
            previous.installed_version,
            increases.join(", ")
        ));
    }
    let bindings = |metadata: &mesh_llm_plugin_manager::InstalledPluginMetadata| {
        metadata
            .manifest
            .as_ref()
            .map(|manifest| manifest.operation_bindings.clone())
            .unwrap_or_default()
    };
    let before = bindings(previous);
    let new_bindings: Vec<String> = bindings(installed)
        .into_iter()
        .filter(|binding| {
            !before
                .iter()
                .any(|b| b.capability == binding.capability && b.operations == binding.operations)
        })
        .map(|binding| format!("{} ({})", binding.capability, binding.operations.join(", ")))
        .collect();
    if !new_bindings.is_empty() {
        lines.push(format!(
            "⚠️  {} now asks for provider bindings it did not have at version {}: {}.",
            installed.name,
            previous.installed_version,
            new_bindings.join("; ")
        ));
    }
    lines
}

/// What is said when a package lands (`PluginProgressEvent::Installed` /
/// `Updated`): the landing line, then the disclosure of what it was granted.
/// Printed as the package lands, before anything else is attempted.
fn landed_lines(
    landing: String,
    metadata: &mesh_llm_plugin_manager::InstalledPluginMetadata,
) -> Vec<String> {
    let mut lines = vec![landing];
    lines.extend(exchange_access_summary(metadata));
    lines
}

fn delete(name: &str) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    delete_in(&store, name, &mut mesh_llm_events::console_err())
}

fn delete_in(store: &PluginStore, name: &str, err: &mut impl Write) -> Result<()> {
    store.delete(name)?;
    writeln!(err, "🗑️  Deleted {name}")?;
    // A deleted default stays removed: delete leaves the record that `plugins
    // disable` leaves for a default that is not installed, which a node start
    // and `plugins install-defaults` respect. `plugins enable` removes it.
    if default_plugin(name).is_some() {
        store.set_default_turned_off(name, true)?;
        writeln!(err, "{}", deleted_default_note(name))?;
    }
    Ok(())
}

/// What a deleted default plugin's operator should know: it stays removed.
fn deleted_default_note(name: &str) -> String {
    format!(
        "{name} is a default plugin: it stays removed, and the node will not install it from its \
         bundled copy again. To have it installed again, run mesh-llm plugins enable {name}."
    )
}

/// A default with no pin for this platform: nothing is installed.
fn no_reviewed_release_line(name: &str) -> String {
    format!("ℹ️  No reviewed {name} release for this platform; not installed")
}

/// One line for a default that provisioning left alone; none for one already current.
fn left_alone_line(name: &str, outcome: &DefaultPluginOutcome) -> Option<String> {
    match outcome {
        DefaultPluginOutcome::OperatorManaged => Some(format!(
            "ℹ️  Default {name} left alone: you run it from your own config or install"
        )),
        DefaultPluginOutcome::Disabled => Some(format!(
            "ℹ️  Default {name} left at its installed version: it is disabled \
             (enable it, then run `mesh-llm plugins update {name}` to move to the reviewed pin)"
        )),
        DefaultPluginOutcome::TurnedOff => Some(format!(
            "ℹ️  Default {name} not installed: you turned it off \
             (run `mesh-llm plugins enable {name}` to have it installed again)"
        )),
        DefaultPluginOutcome::UnsupportedPlatform => Some(no_reviewed_release_line(name)),
        DefaultPluginOutcome::NotBundled => Some(format!(
            "ℹ️  Default {name} not installed: this install carries no bundled copy of it, \
             and a default plugin is never downloaded"
        )),
        _ => None,
    }
}

fn info(name: &str, runtime_rows: Option<&PluginListRows>) -> Result<bool> {
    let store = PluginStore::new(default_store_root()?);
    let mut out = mesh_llm_events::console_out();
    if let Some(metadata) = store.load_optional(name)? {
        report_exchange_access(&metadata)?;
        writeln!(out, "name\t{}", metadata.name)?;
        writeln!(out, "version\t{}", metadata.installed_version)?;
        writeln!(out, "enabled\t{}", metadata.enabled)?;
        writeln!(out, "source\t{}", metadata.source_repository)?;
        writeln!(out, "target\t{}", metadata.target_triple)?;
        writeln!(out, "asset\t{}", metadata.downloaded_asset_name)?;
        writeln!(out, "path\t{}", metadata.install_path.display())?;
        if let Some(protocol) = metadata.last_protocol_version {
            writeln!(out, "protocol\t{protocol}")?;
        }
        if let Some(status) = metadata.last_status {
            writeln!(out, "status\t{status}")?;
        }
        if let Some(error) = metadata.last_error {
            writeln!(out, "error\t{error}")?;
        }
        return Ok(true);
    }
    let Some(runtime_rows) = runtime_rows else {
        return Ok(false);
    };
    if let Some(row) = runtime_rows.externals.iter().find(|row| row.name == name) {
        for line in runtime_plugin_info_lines(row) {
            writeln!(out, "{line}")?;
        }
        return Ok(true);
    }
    if let Some(row) = runtime_rows.inactive.iter().find(|row| row.name == name) {
        for line in inactive_plugin_info_lines(row) {
            writeln!(out, "{line}")?;
        }
        return Ok(true);
    }
    bail!("plugin '{name}' is not installed")
}

fn runtime_plugin_info_lines(row: &RuntimePluginRow) -> Vec<String> {
    vec![
        format!("name\t{}", row.name),
        "kind\truntime".to_string(),
        format!("command\t{}", row.command),
        format!("args\t{}", row.args.join(" ")),
        "source\tbuilt-in/runtime".to_string(),
    ]
}

fn inactive_plugin_info_lines(row: &InactivePluginRow) -> Vec<String> {
    vec![
        format!("name\t{}", row.name),
        format!("kind\t{}", row.kind),
        format!("status\t{}", row.status),
        format!("error\t{}", row.error.clone().unwrap_or_default()),
    ]
}

async fn search(query: Option<&str>) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut spinner = start_spinner("Searching plugin catalog");
    let catalog = PluginCatalog::fetch(&Client::new(), &options.catalog_url).await;
    spinner.finish();
    let catalog = catalog?;
    let hits = catalog.search(query);
    if hits.is_empty() {
        let mut err = mesh_llm_events::console_err();
        writeln!(err, "🔎 No plugins found")?;
        return Ok(());
    }
    let mut out = mesh_llm_events::console_out();
    for entry in hits {
        writeln!(
            out,
            "{}\t{}\t{}\t{} <{}>",
            entry.name, entry.description, entry.github_url, entry.author_name, entry.author_email
        )?;
    }
    Ok(())
}

fn list(runtime_rows: &PluginListRows) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    let mut out = mesh_llm_events::console_out();
    let installed = store.list()?;
    for metadata in &installed {
        let state = if metadata.enabled {
            "enabled"
        } else {
            "disabled"
        };
        writeln!(
            out,
            "{}\tversion={}\tstate={}\tsource={}",
            metadata.name, metadata.installed_version, state, metadata.source_repository
        )?;
    }
    let mut err = mesh_llm_events::console_err();
    for warning in unresolvable_capability_grants(&installed, runtime_rows.providers.as_deref()) {
        writeln!(err, "⚠️  {warning}")?;
    }

    for spec in &runtime_rows.externals {
        writeln!(
            out,
            "{}\tkind=runtime\tcommand={}\targs={}",
            spec.name,
            spec.command,
            spec.args.join(" ")
        )?;
    }
    for summary in &runtime_rows.inactive {
        writeln!(
            out,
            "{}\tkind={}\tstate={}\terror={}",
            summary.name,
            summary.kind,
            summary.status,
            summary.error.clone().unwrap_or_default()
        )?;
    }
    Ok(())
}

#[derive(Default)]
struct CliPluginProgress {
    spinner: Option<SpinnerHandle>,
    active_download: Option<String>,
    last_percent: Option<u64>,
}

impl CliPluginProgress {
    fn finish(&mut self) {
        if let Some(mut spinner) = self.spinner.take() {
            spinner.finish();
        }
        if self.active_download.take().is_some() {
            let _ = clear_stderr_line();
        }
    }

    fn spinner(&mut self, message: String) {
        self.finish();
        self.spinner = Some(start_spinner(&message));
    }

    /// A package landed: print `landing` and disclose what it was granted.
    fn landed(
        &mut self,
        landing: String,
        metadata: &mesh_llm_plugin_manager::InstalledPluginMetadata,
    ) {
        self.finish();
        let mut err = mesh_llm_events::console_err();
        for line in landed_lines(landing, metadata) {
            let _ = writeln!(err, "{line}");
        }
    }

    fn started_download(&mut self, asset: String, total_bytes: Option<u64>) {
        self.finish();
        self.active_download = Some(asset.clone());
        self.last_percent = None;
        let mut err = mesh_llm_events::console_err();
        let _ = writeln!(err, "⬇️  Downloading {asset}");
        if let Some(total) = total_bytes {
            let _ = writeln!(err, "   size: {}", format_bytes(total));
        }
    }

    fn download_progress(&mut self, downloaded: u64, total: Option<u64>) {
        let Some(asset) = self.active_download.as_deref() else {
            return;
        };
        if let Some(total) = total.filter(|total| *total > 0) {
            let percent = downloaded.saturating_mul(100) / total;
            if self.last_percent == Some(percent) {
                return;
            }
            self.last_percent = Some(percent);
            let gauge = render_inline_gauge(
                ratio_complete_u64(downloaded, total),
                &format!(
                    "⬇️  {} {} / {} ({}%)",
                    asset,
                    format_bytes(downloaded),
                    format_bytes(total),
                    percent
                ),
            );
            let mut err = mesh_llm_events::console_err();
            let _ = write!(err, "\r\x1b[2K{gauge}");
            let _ = err.flush();
        }
    }
}

impl PluginProgressReporter for CliPluginProgress {
    fn report(&mut self, event: PluginProgressEvent) {
        match event {
            PluginProgressEvent::ResolvingCatalog { name } => {
                self.spinner(format!("Looking up {name} in the plugin catalog"));
            }
            PluginProgressEvent::ResolvingGitHub { repo } => {
                self.spinner(format!("Checking GitHub releases for {repo}"));
            }
            PluginProgressEvent::SelectingAsset { target } => {
                self.spinner(format!("Finding compatible plugin asset for {target}"));
            }
            PluginProgressEvent::DownloadStarted { asset, total_bytes } => {
                self.started_download(asset, total_bytes);
            }
            PluginProgressEvent::DownloadProgress {
                downloaded_bytes,
                total_bytes,
            } => self.download_progress(downloaded_bytes, total_bytes),
            PluginProgressEvent::DownloadFinished { asset } => {
                self.finish();
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(err, "✅ Downloaded {asset}");
            }
            PluginProgressEvent::Extracting { asset } => {
                self.spinner(format!("Installing {asset}"));
            }
            PluginProgressEvent::Installed { installed } => {
                self.landed(
                    format!(
                        "📦 Installed {} {}",
                        installed.name, installed.installed_version
                    ),
                    &installed,
                );
            }
            PluginProgressEvent::Updated {
                installed,
                from,
                previous,
            } => {
                self.landed(
                    format!(
                        "⬆️  Updated {} {from} -> {}",
                        installed.name, installed.installed_version
                    ),
                    &installed,
                );
                let mut err = mesh_llm_events::console_err();
                for line in access_increase_lines(&previous, &installed) {
                    let _ = writeln!(err, "{line}");
                }
            }
            PluginProgressEvent::AlreadyCurrent { name, version } => {
                self.finish();
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(err, "✅ {name} is up to date ({version})");
            }
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for candidate in &UNITS[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = candidate;
    }
    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{value:.1} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_plugin_info_lines_describe_builtin_runtime_plugin() {
        let row = RuntimePluginRow {
            name: "blobstore".to_string(),
            command: "/tmp/mesh-llm".to_string(),
            args: vec![
                "--log-format".to_string(),
                "json".to_string(),
                "--plugin".to_string(),
                "blobstore".to_string(),
            ],
        };

        assert_eq!(
            runtime_plugin_info_lines(&row),
            vec![
                "name\tblobstore".to_string(),
                "kind\truntime".to_string(),
                "command\t/tmp/mesh-llm".to_string(),
                "args\t--log-format json --plugin blobstore".to_string(),
                "source\tbuilt-in/runtime".to_string(),
            ]
        );
    }

    #[test]
    fn inactive_plugin_info_lines_describe_startup_failure() {
        let row = InactivePluginRow {
            name: "image-tools".to_string(),
            kind: "external".to_string(),
            status: "inactive".to_string(),
            error: Some("command not found".to_string()),
        };

        assert_eq!(
            inactive_plugin_info_lines(&row),
            vec![
                "name\timage-tools".to_string(),
                "kind\texternal".to_string(),
                "status\tinactive".to_string(),
                "error\tcommand not found".to_string(),
            ]
        );
    }

    #[test]
    fn a_console_saved_setting_does_not_take_a_default_out_of_default_management() {
        #[derive(serde::Deserialize)]
        struct Config {
            plugin: Vec<mesh_llm_config::PluginConfigEntry>,
        }
        let config: Config = toml::from_str(
            r#"
[[plugin]]
name = "capsules"
[plugin.settings]
share_history_segments = "off"

[[plugin]]
name = "operator-run"
command = "/opt/plugins/operator-run"

[[plugin]]
name = "remote"
url = "unix:///run/remote.sock"
"#,
        )
        .unwrap();
        let configured = operator_run_plugins(&config.plugin);
        assert!(!configured.contains("capsules"));
        assert!(configured.contains("operator-run"));
        assert!(configured.contains("remote"));
    }

    #[test]
    fn install_defaults_uses_the_selected_config_for_operator_plugins() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("alternate.toml");
        std::fs::write(
            &path,
            "[[plugin]]\nname = \"capsules\"\ncommand = \"/opt/capsules\"\n",
        )
        .unwrap();

        let configured = configured_operator_plugins(Some(&path)).unwrap();

        assert!(configured.contains("capsules"));
    }

    fn installed_default(store: &PluginStore, enabled: bool) {
        store
            .save(&mesh_llm_plugin_manager::InstalledPluginMetadata {
                name: "capsules".into(),
                source_repository: "https://github.com/Mesh-LLM/capsules".into(),
                installed_version: "v0.1.3".into(),
                target_triple: "x86_64-unknown-linux-gnu".into(),
                downloaded_asset_name: "capsules-0.1.3-x86_64-unknown-linux-gnu.tar.gz".into(),
                install_path: store.root().join("installed").join("capsules"),
                enabled,
                default_managed: false,
                manifest: None,
                last_protocol_version: None,
                last_status: None,
                last_error: None,
                verified_sha256: None,
            })
            .unwrap();
    }

    /// A target this build pins capsules for.
    const PINNED: Option<&str> = Some("x86_64-unknown-linux-gnu");

    #[test]
    fn enabling_an_uninstalled_default_says_the_next_start_installs_it() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut err = Vec::new();
        set_enabled_in(&store, "capsules", true, PINNED, &mut err).unwrap();
        let err = String::from_utf8(err).unwrap();
        assert_eq!(
            err,
            "✅ Enabled capsules: the node installs it from its bundled copy when it next starts\n"
        );
    }

    #[test]
    fn enabling_a_default_with_no_pin_for_the_target_says_what_install_defaults_says() {
        for target in [Some("x86_64-pc-windows-msvc"), None] {
            let temp = tempfile::tempdir().unwrap();
            let store = PluginStore::new(temp.path());
            store.set_default_turned_off("capsules", true).unwrap();
            let mut err = Vec::new();
            set_enabled_in(&store, "capsules", true, target, &mut err).unwrap();
            let err = String::from_utf8(err).unwrap();
            assert_eq!(
                err,
                format!(
                    "✅ Enabled capsules\n{}\n",
                    left_alone_line("capsules", &DefaultPluginOutcome::UnsupportedPlatform)
                        .unwrap()
                ),
                "{target:?}"
            );
            assert!(!err.contains("bundled copy"), "{err}");
            assert!(
                !store.default_turned_off("capsules"),
                "enabling still removes the turned-off record"
            );
        }
    }

    #[test]
    fn disabling_a_plugin_that_is_not_installed_and_not_a_default_fails() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let error =
            set_enabled_in(&store, "not-a-default", false, PINNED, &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("not installed"), "{error:#}");
        assert!(!store.default_turned_off("not-a-default"));
    }

    #[test]
    fn disabling_an_uninstalled_default_records_it_and_enabling_removes_it() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        set_enabled_in(&store, "capsules", false, PINNED, &mut Vec::new()).unwrap();
        assert!(store.default_turned_off("capsules"));
        set_enabled_in(&store, "capsules", true, PINNED, &mut Vec::new()).unwrap();
        assert!(!store.default_turned_off("capsules"));
    }

    #[test]
    fn enabling_an_installed_default_removes_a_stale_turned_off_record() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        // Turned off while not installed, then installed explicitly anyway.
        store.set_default_turned_off("capsules", true).unwrap();
        installed_default(&store, false);
        set_enabled_in(&store, "capsules", true, PINNED, &mut Vec::new()).unwrap();
        assert!(!store.default_turned_off("capsules"));
        assert!(store.load("capsules").unwrap().enabled);
    }

    #[test]
    fn disabling_an_installed_default_keeps_the_install_record() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        installed_default(&store, true);
        set_enabled_in(&store, "capsules", false, PINNED, &mut Vec::new()).unwrap();
        assert!(!store.load("capsules").unwrap().enabled);
        assert!(!store.default_turned_off("capsules"));
    }

    #[test]
    fn deleting_a_default_keeps_it_removed_until_it_is_enabled() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        installed_default(&store, true);
        let mut err = Vec::new();
        delete_in(&store, "capsules", &mut err).unwrap();
        assert!(store.load_optional("capsules").unwrap().is_none());
        assert!(
            store.default_turned_off("capsules"),
            "a deleted default is recorded as turned off"
        );
        let err = String::from_utf8(err).unwrap();
        assert!(err.contains("it stays removed"), "{err}");
        assert!(
            err.contains("run mesh-llm plugins enable capsules"),
            "{err}"
        );

        set_enabled_in(&store, "capsules", true, PINNED, &mut Vec::new()).unwrap();
        assert!(!store.default_turned_off("capsules"));
    }

    #[test]
    fn deleting_a_plugin_that_is_not_a_default_records_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut err = Vec::new();
        delete_in(&store, "mine", &mut err).unwrap();
        assert!(!store.default_turned_off("mine"));
        assert!(!String::from_utf8(err).unwrap().contains("default plugin"));
    }

    fn installed_plugin(
        store: &PluginStore,
        name: &str,
    ) -> mesh_llm_plugin_manager::InstalledPluginMetadata {
        let metadata = mesh_llm_plugin_manager::InstalledPluginMetadata {
            name: name.into(),
            source_repository: format!("https://github.com/Mesh-LLM/{name}"),
            installed_version: "v0.1.0".into(),
            target_triple: "x86_64-unknown-linux-gnu".into(),
            downloaded_asset_name: format!("{name}-0.1.0-x86_64-unknown-linux-gnu.tar.gz"),
            install_path: store.root().join("installed").join(name),
            enabled: true,
            default_managed: false,
            verified_sha256: None,
            manifest: None,
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        };
        store.save(&metadata).unwrap();
        metadata
    }

    #[test]
    fn a_landing_discloses_what_the_package_was_granted() {
        // The lines printed on `Installed`/`Updated`, i.e. when a package
        // lands and before anything else is attempted, carry the access
        // disclosure.
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        let mut root = installed_plugin(&store, "root");
        root.manifest = Some(mesh_llm_plugin_manager::InstalledPluginManifestMetadata {
            openai_exchange_hook: Some(
                mesh_llm_plugin_manager::store::InstalledOpenAiExchangeAccess {
                    contract_version: 1,
                    request_body: true,
                    ..Default::default()
                },
            ),
            operation_bindings: Vec::new(),
            peer_blocks: false,
            config_schema: None,
            web_ui: None,
        });
        let lines = landed_lines("📦 Installed root v0.1.0".to_string(), &root);
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0], "📦 Installed root v0.1.0");
        assert!(
            lines[1].starts_with("OpenAI lifecycle v1 access: original prompts."),
            "{}",
            lines[1]
        );
        assert!(lines[1].contains("Installing grants these requests"));

        // A package requesting nothing gets the landing line only.
        let plain = installed_plugin(&store, "needed");
        assert_eq!(
            landed_lines("📦 Installed needed v0.1.0".to_string(), &plain),
            vec!["📦 Installed needed v0.1.0".to_string()]
        );
    }

    #[test]
    fn install_defaults_turned_off_by_the_environment_says_so() {
        let line = turned_off_by_environment_line();
        assert!(line.contains("MESH_LLM_NO_DEFAULT_PLUGINS"), "{line}");
        assert!(line.contains("not installed"), "{line}");
    }

    #[test]
    fn a_default_left_alone_says_why() {
        let line = |outcome| left_alone_line("capsules", &outcome);
        assert!(
            line(DefaultPluginOutcome::OperatorManaged)
                .unwrap()
                .contains("left alone")
        );
        assert!(
            line(DefaultPluginOutcome::Disabled)
                .unwrap()
                .contains("disabled")
        );
        assert!(
            line(DefaultPluginOutcome::UnsupportedPlatform)
                .unwrap()
                .contains("this platform")
        );
        assert!(
            line(DefaultPluginOutcome::TurnedOff)
                .unwrap()
                .contains("plugins enable capsules")
        );
        assert!(line(DefaultPluginOutcome::AlreadyCurrent).is_none());
    }

    #[test]
    fn update_takes_no_default_plugins() {
        use clap::Parser;
        let cli = mesh_llm_cli::Cli::try_parse_from(["mesh-llm", "update", "--no-default-plugins"])
            .unwrap();
        assert!(matches!(
            cli.command,
            Some(mesh_llm_cli::Command::Update {
                no_default_plugins: true,
                ..
            })
        ));
    }
}
