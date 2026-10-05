use std::io::Write;

use anyhow::{Result, bail};
use mesh_llm_plugin_manager::defaults::{
    DEFAULT_PLUGINS, DefaultPluginOutcome, default_plugin, default_plugins_opted_out,
    install_default_plugins,
};
use mesh_llm_plugin_manager::install::install_plugin_archive;
use mesh_llm_plugin_manager::{
    PluginCatalog, PluginInstallOptions, PluginProgressEvent, PluginProgressReporter, PluginStore,
    default_store_root, install_plugin, update_plugin,
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

pub async fn run_plugin_command(
    command: &PluginCommand,
    runtime_rows: Option<&PluginListRows>,
) -> Result<bool> {
    match command {
        PluginCommand::InstallDefaults => install_defaults().await?,
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

async fn install_defaults() -> Result<()> {
    if default_plugins_opted_out() || DEFAULT_PLUGINS.is_empty() {
        return Ok(());
    }
    let options = PluginInstallOptions::from_env()?;
    let config = mesh_llm_config::load_config(None)?;
    let configured = operator_run_plugins(&config.plugins);
    let mut progress = CliPluginProgress::default();
    let outcomes =
        install_default_plugins(DEFAULT_PLUGINS, &configured, &options, &mut progress).await;
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
            | DefaultPluginOutcome::UnsupportedPlatform) => {
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
    }
    Ok(())
}

fn set_enabled(name: &str, enabled: bool) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    let mut err = mesh_llm_events::console_err();
    // A default that is not installed (for example right after a delete) is
    // turned off, or back on, by a record the installers and update respect.
    if default_plugin(name).is_some() && store.load_optional(name)?.is_none() {
        store.set_default_turned_off(name, !enabled)?;
        if enabled {
            writeln!(
                err,
                "✅ Enabled {name}: the next installer or mesh-llm update run installs it"
            )?;
        } else {
            writeln!(
                err,
                "⏸️  Disabled {name}: the installers and mesh-llm update will not install it again"
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

fn delete(name: &str) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    store.delete(name)?;
    let mut err = mesh_llm_events::console_err();
    writeln!(err, "🗑️  Deleted {name}")?;
    Ok(())
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
        DefaultPluginOutcome::UnsupportedPlatform => Some(format!(
            "ℹ️  No reviewed {name} release for this platform; not installed"
        )),
        _ => None,
    }
}

fn info(name: &str, runtime_rows: Option<&PluginListRows>) -> Result<bool> {
    let store = PluginStore::new(default_store_root()?);
    let mut out = mesh_llm_events::console_out();
    if let Some(metadata) = store.load_optional(name)? {
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
    for metadata in store.list()? {
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
            PluginProgressEvent::Installed { name, version } => {
                self.finish();
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(err, "📦 Installed {name} {version}");
            }
            PluginProgressEvent::Updated { name, from, to } => {
                self.finish();
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(err, "⬆️  Updated {name} {from} -> {to}");
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
name = "capsule-emit-mesh"
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
        assert!(!configured.contains("capsule-emit-mesh"));
        assert!(configured.contains("operator-run"));
        assert!(configured.contains("remote"));
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
