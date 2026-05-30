use std::ffi::OsString;
use std::io::Write;

use anyhow::{Context, Result, bail};
use mesh_llm_plugin_manager::{PluginStore, default_store_root};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::cli::Cli;
use crate::{plugin, runtime};

const CLI_RUN_OPERATION: &str = "cli.run";

#[derive(Debug, Serialize)]
struct PluginCliRunRequest {
    command: String,
    args: Vec<String>,
    argv: Vec<String>,
    cwd: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PluginCliRunResponse {
    #[serde(default)]
    exit_code: Option<i32>,
    #[serde(default)]
    stdout: Option<String>,
    #[serde(default)]
    stderr: Option<String>,
}

pub(crate) async fn run_external_plugin_command(cli: &Cli, raw_args: &[OsString]) -> Result<()> {
    let request = build_cli_run_request(raw_args)?;
    let command = request.command.clone();
    let spec = resolve_required_cli_plugin(cli, &command)?;
    run_plugin_cli_request(cli, spec, request).await
}

async fn run_plugin_cli_request(
    cli: &Cli,
    spec: plugin::ExternalPluginSpec,
    request: PluginCliRunRequest,
) -> Result<()> {
    let command = request.command.clone();
    let input_json = serde_json::to_string(&request).context("Serialize plugin CLI request")?;
    let resolved = plugin::ResolvedPlugins {
        externals: vec![spec],
        inactive: Vec::new(),
    };
    let (plugin_mesh_tx, _plugin_mesh_rx) = mpsc::channel(1);
    let manager =
        plugin::PluginManager::start(&resolved, plugin_host_mode(cli), plugin_mesh_tx).await?;
    let result = manager
        .invoke_operation_without_timeout(&command, CLI_RUN_OPERATION, &input_json)
        .await;
    manager.shutdown().await;

    let result = result?;
    handle_plugin_cli_result(&command, result)
}

fn build_cli_run_request(raw_args: &[OsString]) -> Result<PluginCliRunRequest> {
    let (command, args) = raw_args
        .split_first()
        .context("Missing external plugin command")?;
    let command = os_to_string(command, "plugin command")?;
    let args = args
        .iter()
        .map(|arg| os_to_string(arg, "plugin command argument"))
        .collect::<Result<Vec<_>>>()?;
    let argv = std::iter::once(command.clone())
        .chain(args.iter().cloned())
        .collect();
    let cwd = std::env::current_dir()
        .ok()
        .map(|path| path.display().to_string());

    Ok(PluginCliRunRequest {
        command,
        args,
        argv,
        cwd,
    })
}

fn os_to_string(value: &OsString, label: &str) -> Result<String> {
    value
        .clone()
        .into_string()
        .map_err(|_| anyhow::anyhow!("{label} must be valid UTF-8"))
}

fn resolve_required_cli_plugin(cli: &Cli, command: &str) -> Result<plugin::ExternalPluginSpec> {
    if let Some(spec) = resolve_configured_cli_plugin(cli, command)? {
        return Ok(spec);
    }
    if let Some(spec) = resolve_installed_cli_plugin(command)? {
        return Ok(spec);
    }
    bail!("Unknown command '{command}'. Run `mesh-llm --help` to see available commands.");
}

fn resolve_configured_cli_plugin(
    cli: &Cli,
    command: &str,
) -> Result<Option<plugin::ExternalPluginSpec>> {
    let config = plugin::load_config(cli.config.as_deref())?;
    if let Some(entry) = config.plugins.iter().find(|entry| entry.name == command) {
        if !entry.enabled.unwrap_or(true) {
            return plugin::bundled_cli_plugin_spec(command);
        }
        let resolved = runtime::load_resolved_plugins(cli)?;
        let spec = resolved
            .externals
            .into_iter()
            .find(|spec| spec.name == command)
            .with_context(|| {
                format!("Plugin command '{command}' is configured but not runnable")
            })?;
        return Ok(Some(spec));
    }

    if let Some(spec) = resolve_installed_cli_plugin(command)? {
        return Ok(Some(spec));
    }

    plugin::bundled_cli_plugin_spec(command)
}

fn resolve_installed_cli_plugin(command: &str) -> Result<Option<plugin::ExternalPluginSpec>> {
    let root = match default_store_root() {
        Ok(root) => root,
        Err(_) => return Ok(None),
    };
    let store = PluginStore::new(root);
    let Some(metadata) = store.load_optional(command)? else {
        return Ok(None);
    };
    if !metadata.enabled {
        return Ok(None);
    }
    let executable = metadata.executable_path();
    if !executable.exists() {
        bail!(
            "Plugin command '{}' is installed but not runnable: missing executable {}",
            metadata.name,
            executable.display()
        );
    }
    Ok(Some(plugin::ExternalPluginSpec {
        name: metadata.name,
        command: executable.display().to_string(),
        args: Vec::new(),
        url: None,
        env: Default::default(),
        startup: Default::default(),
    }))
}

fn plugin_host_mode(cli: &Cli) -> plugin::PluginHostMode {
    plugin::PluginHostMode {
        mesh_visibility: if cli.publish || cli.nostr_discovery {
            mesh_llm_plugin::MeshVisibility::Public
        } else {
            mesh_llm_plugin::MeshVisibility::Private
        },
    }
}

fn handle_plugin_cli_result(command: &str, result: plugin::ToolCallResult) -> Result<()> {
    if result.is_error {
        bail!("Plugin command '{command}' failed: {}", result.content_json);
    }

    let value = parse_result_value(&result.content_json)?;
    match value {
        serde_json::Value::Null => Ok(()),
        serde_json::Value::String(text) => {
            print!("{text}");
            std::io::stdout().flush().ok();
            Ok(())
        }
        serde_json::Value::Object(_) => handle_structured_result(command, value),
        other => {
            println!("{}", serde_json::to_string_pretty(&other)?);
            Ok(())
        }
    }
}

fn parse_result_value(content_json: &str) -> Result<serde_json::Value> {
    if content_json.trim().is_empty() {
        return Ok(serde_json::Value::Null);
    }
    match serde_json::from_str(content_json) {
        Ok(value) => Ok(value),
        Err(_) => Ok(serde_json::Value::String(content_json.to_string())),
    }
}

fn handle_structured_result(command: &str, value: serde_json::Value) -> Result<()> {
    let response: PluginCliRunResponse =
        serde_json::from_value(value).context("Decode plugin CLI response")?;
    if let Some(stderr) = response.stderr {
        eprint!("{stderr}");
        std::io::stderr().flush().ok();
    }
    if let Some(stdout) = response.stdout {
        print!("{stdout}");
        std::io::stdout().flush().ok();
    }
    if let Some(code) = response.exit_code
        && code != 0
    {
        bail!("Plugin command '{command}' exited with status {code}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{env, ffi::OsString, path::PathBuf};

    use clap::Parser;
    use mesh_llm_plugin_manager::InstalledPluginMetadata;
    use serial_test::serial;
    use tempfile::TempDir;

    use super::*;

    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &std::path::Path) -> Self {
            let previous = env::var(key).ok();
            unsafe { env::set_var(key, value) };
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => unsafe { env::set_var(self.key, value) },
                None => unsafe { env::remove_var(self.key) },
            }
        }
    }

    fn cli_with_missing_config(temp: &TempDir) -> Cli {
        Cli::parse_from(vec![
            OsString::from("mesh-llm"),
            OsString::from("--config"),
            temp.path().join("missing-config.toml").into_os_string(),
        ])
    }

    fn installed_metadata(name: &str, install_path: PathBuf) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: name.to_string(),
            source_repository: "https://github.com/mesh-llm/demo".to_string(),
            installed_version: "v1.0.0".to_string(),
            target_triple: "test-target".to_string(),
            downloaded_asset_name: "demo.tar.gz".to_string(),
            install_path,
            enabled: true,
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        }
    }

    #[test]
    #[serial]
    fn resolves_installed_cli_plugin_when_not_configured() {
        let temp = TempDir::new().unwrap();
        let _plugin_dir = EnvGuard::set("MESH_LLM_PLUGIN_DIR", temp.path());
        let install_path = temp.path().join("installed").join("demo");
        std::fs::create_dir_all(&install_path).unwrap();
        let executable = install_path.join(format!("demo{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&executable, "").unwrap();
        PluginStore::new(temp.path())
            .save(&installed_metadata("demo", install_path))
            .unwrap();

        let spec = resolve_required_cli_plugin(&cli_with_missing_config(&temp), "demo")
            .expect("installed plugin should resolve");

        assert_eq!(spec.name, "demo");
        assert_eq!(spec.command, executable.display().to_string());
        assert!(spec.args.is_empty());
    }

    #[test]
    #[serial]
    fn installed_cli_plugin_missing_executable_is_reported() {
        let temp = TempDir::new().unwrap();
        let _plugin_dir = EnvGuard::set("MESH_LLM_PLUGIN_DIR", temp.path());
        let install_path = temp.path().join("installed").join("demo");
        std::fs::create_dir_all(&install_path).unwrap();
        PluginStore::new(temp.path())
            .save(&installed_metadata("demo", install_path))
            .unwrap();

        let err = resolve_required_cli_plugin(&cli_with_missing_config(&temp), "demo")
            .expect_err("missing executable should be reported");

        assert!(err.to_string().contains("installed but not runnable"));
    }
}
