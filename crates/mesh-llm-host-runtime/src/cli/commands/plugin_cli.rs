use std::ffi::OsString;
use std::io::Write;

use anyhow::{Context, Result, bail};
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
    bail!(
        "Unknown command '{command}'. To provide it from a plugin, add \
         [[plugin]] name = \"{command}\" command = \"mesh-llm-plugin-{command}\" \
         to your mesh-llm config."
    );
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

    plugin::bundled_cli_plugin_spec(command)
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
