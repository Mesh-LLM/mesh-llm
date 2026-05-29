use std::io::Write;

use anyhow::Result;
use mesh_llm_plugin_manager::{
    PluginCatalog, PluginInstallOptions, PluginProgressEvent, PluginProgressReporter, PluginStore,
    default_store_root, install_plugin, update_plugin,
};
use reqwest::Client;
use serde_json::{Value, json};

use crate::cli::terminal_progress::{SpinnerHandle, clear_stderr_line, start_spinner};
use crate::cli::{Cli, PluginCommand};
use crate::runtime;

pub(crate) async fn run_plugin_command(command: &PluginCommand, cli: &Cli) -> Result<()> {
    match command {
        PluginCommand::Install { reference } => install(reference).await?,
        PluginCommand::Update { name } => update(name).await?,
        PluginCommand::Enable { name } => set_enabled(name, true)?,
        PluginCommand::Disable { name } => set_enabled(name, false)?,
        PluginCommand::Delete { name } => delete(name)?,
        PluginCommand::Info { name, json } => info(name, *json)?,
        PluginCommand::Search { query } => search(query.as_deref()).await?,
        PluginCommand::List { json } => list(cli, *json)?,
    }
    Ok(())
}

async fn install(reference: &str) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut progress = CliPluginProgress::default();
    let outcome = install_plugin(reference, &options, &mut progress).await?;
    progress.finish();
    if outcome.changed {
        eprintln!(
            "✅ Installed {} {}",
            outcome.metadata.name, outcome.metadata.installed_version
        );
    }
    Ok(())
}

async fn update(name: &str) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut progress = CliPluginProgress::default();
    let outcome = update_plugin(name, &options, &mut progress).await?;
    progress.finish();
    if outcome.changed {
        eprintln!(
            "✅ Updated {} to {}",
            outcome.metadata.name, outcome.metadata.installed_version
        );
    }
    Ok(())
}

fn set_enabled(name: &str, enabled: bool) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    let metadata = store.set_enabled(name, enabled)?;
    if metadata.enabled {
        eprintln!("✅ Enabled {}", metadata.name);
    } else {
        eprintln!("⏸️  Disabled {}", metadata.name);
    }
    Ok(())
}

fn delete(name: &str) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    store.delete(name)?;
    eprintln!("🗑️  Deleted {name}");
    Ok(())
}

fn info(name: &str, json_output: bool) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    let metadata = store.load(name)?;
    if json_output {
        return print_json(&json!({ "plugin": metadata }));
    }

    println!("🔌 Plugin");
    println!();
    println!("Name: {}", metadata.name);
    println!("Version: {}", metadata.installed_version);
    println!("State: {}", enabled_label(metadata.enabled));
    println!("Source: {}", metadata.source_repository);
    println!("Target: {}", metadata.target_triple);
    println!("Asset: {}", metadata.downloaded_asset_name);
    println!("Path: {}", metadata.install_path.display());
    if let Some(protocol) = metadata.last_protocol_version {
        println!("Protocol: {protocol}");
    }
    if let Some(status) = metadata.last_status {
        println!("Last status: {status}");
    }
    if let Some(error) = metadata.last_error {
        println!("Last error: {error}");
    }
    Ok(())
}

async fn search(query: Option<&str>) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut spinner = start_spinner("Searching plugin catalog");
    let catalog = PluginCatalog::fetch(&Client::new(), &options.catalog_url).await;
    spinner.finish();
    let catalog = catalog?;
    let hits = catalog.search(query);
    if hits.is_empty() {
        eprintln!("🔎 No plugins found");
        return Ok(());
    }
    for entry in hits {
        println!(
            "{}\t{}\t{}\t{} <{}>",
            entry.name, entry.description, entry.github_url, entry.author_name, entry.author_email
        );
    }
    Ok(())
}

fn list(cli: &Cli, json_output: bool) -> Result<()> {
    if json_output {
        return print_json(&plugin_inventory_json(cli)?);
    }

    let store = PluginStore::new(default_store_root()?);
    let installed = store.list()?;
    let resolved = runtime::load_resolved_plugins(cli)?;

    println!("🔌 Plugins");
    println!();

    println!("📦 Installed: {}", installed.len());
    for metadata in installed {
        println!(
            "  {}  version={}  state={}  source={}",
            metadata.name,
            metadata.installed_version,
            enabled_label(metadata.enabled),
            metadata.source_repository
        );
    }

    println!();
    println!("⚙️  Runtime: {}", resolved.externals.len());
    for spec in resolved.externals {
        println!(
            "  {}  command={}  args={}",
            spec.name,
            spec.command,
            spec.args.join(" ")
        );
    }

    if !resolved.inactive.is_empty() {
        println!();
        println!("⚠️  Inactive: {}", resolved.inactive.len());
    }
    for summary in resolved.inactive {
        println!(
            "  {}  kind={}  state={}  error={}",
            summary.name,
            summary.kind,
            summary.status,
            summary.error.unwrap_or_default()
        );
    }
    Ok(())
}

fn enabled_label(enabled: bool) -> &'static str {
    if enabled {
        "✅ enabled"
    } else {
        "⏸️ disabled"
    }
}

pub(crate) fn plugin_inventory_json(cli: &Cli) -> Result<Value> {
    let store_root = default_store_root()?;
    let store = PluginStore::new(&store_root);
    let installed = store.list()?;
    let resolved = runtime::load_resolved_plugins(cli)?;
    let active = resolved
        .externals
        .into_iter()
        .map(|spec| {
            let env_keys = spec.env.keys().cloned().collect::<Vec<_>>();
            json!({
                "name": spec.name,
                "kind": "runtime",
                "command": spec.command,
                "args": spec.args,
                "url": spec.url,
                "env_keys": env_keys,
            })
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "store_root": store_root,
        "installed": installed,
        "resolved": {
            "active": active,
            "inactive": resolved.inactive,
        },
    }))
}

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
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
        eprintln!("⬇️  Downloading {asset}");
        if let Some(total) = total_bytes {
            eprintln!("   size: {}", format_bytes(total));
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
            eprint!(
                "\r\x1b[2K⬇️  {} {} / {} ({}%)",
                asset,
                format_bytes(downloaded),
                format_bytes(total),
                percent
            );
            let _ = std::io::stderr().flush();
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
                eprintln!("✅ Downloaded {asset}");
            }
            PluginProgressEvent::Extracting { asset } => {
                self.spinner(format!("Installing {asset}"));
            }
            PluginProgressEvent::Installed { name, version } => {
                self.finish();
                eprintln!("📦 Installed {name} {version}");
            }
            PluginProgressEvent::Updated { name, from, to } => {
                self.finish();
                eprintln!("⬆️  Updated {name} {from} -> {to}");
            }
            PluginProgressEvent::AlreadyCurrent { name, version } => {
                self.finish();
                eprintln!("✅ {name} is up to date ({version})");
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
