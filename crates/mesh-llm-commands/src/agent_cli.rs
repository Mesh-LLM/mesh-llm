use anyhow::{Context, Result};
use mesh_llm_plugin_manager::SkillAgent;
use std::process::{Command, Stdio};

use crate::skills::install_skills_for_agent;
use mesh_llm_cli::shell;
use url::Url;

const OPENCODE_PROVIDER_ID: &str = "mesh";
const OPENCODE_API_KEY_ENV: &str = "OPENAI_API_KEY";
const OPENCODE_API_KEY_VALUE: &str = "dummy";
const OPENCODE_INSTALL_HINT: &str = "curl -fsSL https://opencode.ai/install | bash";
const OPENCODE_DEFAULT_CONTEXT_LIMIT: u32 = 32_768;
const OPENCODE_OUTPUT_LIMIT: u32 = 4_096;
const MESH_MCP_SERVER_ID: &str = "mesh";
const MESH_MCP_DISPLAY_NAME: &str = "Mesh LLM";
const DEFAULT_MESH_MCP_URL: &str = "http://127.0.0.1:3131/mcp";

fn configure_interactive_stdio(command: &mut Command) {
    #[cfg(unix)]
    if let Ok(tty) = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
    {
        if let Ok(stdin) = tty.try_clone() {
            command.stdin(Stdio::from(stdin));
        }
        if let Ok(stdout) = tty.try_clone() {
            command.stdout(Stdio::from(stdout));
        }
        command.stderr(Stdio::from(tty));
        return;
    }

    command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
}

fn configure_opencode_launch_command(command: &mut Command, spec: &OpenCodeLaunchSpec) {
    command
        .args(["-m", &spec.model])
        .env(spec.api_key_env, spec.api_key_value);
    // OpenCode runs on Bun, which expects the original terminal file
    // descriptors. Reopening /dev/tty here can make Bun fail while
    // initializing its TTY write streams.
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpenCodeLaunchSpec {
    provider_id: &'static str,
    model: String,
    config_content: String,
    api_key_env: &'static str,
    api_key_value: &'static str,
    install_hint: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpenCodeTarget {
    input: String,
    api_base_url: String,
    api_models_url: String,
    management_models_url: String,
    mcp_url: String,
    auto_start_local_mesh: bool,
    local_port: Option<u16>,
}

fn mesh_mcp_opencode_config(mcp_url: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "remote",
        "url": mcp_url,
        "enabled": true,
        "timeout": 300000,
    })
}

fn mesh_mcp_claude_config_json(mcp_url: &str) -> Result<String> {
    serde_json::to_string(&serde_json::json!({
        "mcpServers": {
            MESH_MCP_SERVER_ID: {
                "type": "http",
                "url": mcp_url,
            }
        }
    }))
    .context("serialize Claude MCP config")
}

fn mesh_mcp_goose_extension(mcp_url: &str) -> Result<serde_yaml::Value> {
    serde_yaml::to_value(serde_json::json!({
        "enabled": true,
        "type": "streamable_http",
        "name": MESH_MCP_DISPLAY_NAME,
        "description": "Expose mesh-llm plugin MCP tools.",
        "uri": mcp_url,
        "timeout": 300,
        "bundled": null,
        "available_tools": [],
    }))
    .context("build Goose MCP extension config")
}

fn yaml_key(key: &str) -> serde_yaml::Value {
    serde_yaml::Value::String(key.to_string())
}

fn empty_yaml_mapping() -> serde_yaml::Value {
    serde_yaml::Value::Mapping(serde_yaml::Mapping::new())
}

fn ensure_yaml_mapping<'a>(
    parent: &'a mut serde_yaml::Mapping,
    key: &str,
    path: &std::path::Path,
) -> Result<&'a mut serde_yaml::Mapping> {
    let key_value = yaml_key(key);
    parent
        .entry(key_value.clone())
        .or_insert_with(empty_yaml_mapping);
    parent
        .get_mut(&key_value)
        .and_then(serde_yaml::Value::as_mapping_mut)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Expected '{}' in {} to be a YAML mapping",
                key,
                path.display()
            )
        })
}

fn read_goose_config(path: &std::path::Path) -> Result<serde_yaml::Value> {
    if !path.exists() {
        return Ok(empty_yaml_mapping());
    }
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    if content.trim().is_empty() {
        return Ok(empty_yaml_mapping());
    }
    let value: serde_yaml::Value = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse {} as YAML", path.display()))?;
    if value.as_mapping().is_none() {
        anyhow::bail!("Expected {} to contain a YAML mapping", path.display());
    }
    Ok(value)
}

fn merge_goose_mcp_config(
    config: &mut serde_yaml::Value,
    mcp_url: &str,
    path: &std::path::Path,
) -> Result<()> {
    let root = config
        .as_mapping_mut()
        .ok_or_else(|| anyhow::anyhow!("Expected {} to contain a YAML mapping", path.display()))?;
    let extensions = ensure_yaml_mapping(root, "extensions", path)?;
    extensions.insert(
        yaml_key(MESH_MCP_SERVER_ID),
        mesh_mcp_goose_extension(mcp_url)?,
    );
    Ok(())
}

fn write_goose_mcp_config_to_path(path: &std::path::Path, mcp_url: &str) -> Result<()> {
    std::fs::create_dir_all(path.parent().expect("Goose config path must have parent"))?;
    let mut config = read_goose_config(path)?;
    merge_goose_mcp_config(&mut config, mcp_url, path)?;
    std::fs::write(path, serde_yaml::to_string(&config)?)?;
    eprintln!("✅ Wrote mesh MCP extension to {}", path.display());
    Ok(())
}

fn write_goose_mcp_config(mcp_url: &str) -> Result<()> {
    let config_path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".config")
        .join("goose")
        .join("config.yaml");
    write_goose_mcp_config_to_path(&config_path, mcp_url)
}

fn is_loopback_or_localhost(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

fn normalize_mesh_host(host: &str) -> Result<OpenCodeTarget> {
    normalize_mesh_host_with_label(host, "mesh host")
}

fn normalize_mesh_host_with_label(host: &str, label: &str) -> Result<OpenCodeTarget> {
    const DEFAULT_API_PORT: u16 = 9337;
    const DEFAULT_MANAGEMENT_PORT: u16 = 3131;

    let trimmed = host.trim();
    if trimmed.is_empty() {
        anyhow::bail!("{label} cannot be empty");
    }

    let has_scheme = trimmed.contains("://");
    let normalized_host = if has_scheme {
        trimmed.to_string()
    } else if trimmed.parse::<u16>().is_ok() {
        format!("127.0.0.1:{trimmed}")
    } else {
        trimmed.to_string()
    };
    let mut parsed = if has_scheme {
        Url::parse(&normalized_host).with_context(|| format!("Invalid {label} URL '{trimmed}'"))?
    } else {
        Url::parse(&format!("http://{normalized_host}"))
            .with_context(|| format!("Invalid {label} '{trimmed}'"))?
    };

    let host_name = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("{label} '{trimmed}' is missing a hostname"))?
        .to_string();

    let is_local_host = is_loopback_or_localhost(&host_name);
    let should_default_api_port =
        parsed.port().is_none() && (!has_scheme || (is_local_host && parsed.scheme() == "http"));
    if should_default_api_port {
        parsed
            .set_port(Some(DEFAULT_API_PORT))
            .map_err(|_| anyhow::anyhow!("Invalid {label} '{trimmed}'"))?;
    }

    parsed.set_query(None);
    parsed.set_fragment(None);

    let mut api_base = parsed.clone();
    api_base.set_path("/v1");

    let mut api_models = api_base.clone();
    api_models.set_path("/v1/models");

    let mut management = parsed.clone();
    if !has_scheme || should_default_api_port || (is_local_host && parsed.scheme() == "http") {
        management
            .set_port(Some(DEFAULT_MANAGEMENT_PORT))
            .map_err(|_| anyhow::anyhow!("Invalid {label} '{trimmed}'"))?;
    }
    management.set_path("/api/models");

    let mut mcp = management.clone();
    mcp.set_path("/mcp");

    let auto_start_local_mesh = is_local_host && parsed.scheme() == "http";

    Ok(OpenCodeTarget {
        input: trimmed.to_string(),
        api_base_url: api_base.to_string(),
        api_models_url: api_models.to_string(),
        management_models_url: management.to_string(),
        mcp_url: mcp.to_string(),
        auto_start_local_mesh,
        local_port: api_base.port_or_known_default(),
    })
}

fn normalize_opencode_host(host: &str) -> Result<OpenCodeTarget> {
    normalize_mesh_host_with_label(host, "OpenCode host")
}

#[cfg(test)]
fn build_opencode_launch_spec(
    model_names: &[String],
    resolved_model: &str,
    api_base_url: &str,
) -> OpenCodeLaunchSpec {
    build_opencode_launch_spec_with_mcp(
        model_names,
        resolved_model,
        api_base_url,
        DEFAULT_MESH_MCP_URL,
    )
}

#[cfg(test)]
fn build_opencode_launch_spec_with_mcp(
    model_names: &[String],
    resolved_model: &str,
    api_base_url: &str,
    mcp_url: &str,
) -> OpenCodeLaunchSpec {
    build_opencode_launch_spec_with_limits(
        model_names,
        resolved_model,
        api_base_url,
        mcp_url,
        &std::collections::HashMap::new(),
    )
}

fn build_opencode_launch_spec_with_limits(
    model_names: &[String],
    resolved_model: &str,
    api_base_url: &str,
    mcp_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
) -> OpenCodeLaunchSpec {
    let mut models = serde_json::Map::new();
    for model in model_names {
        let mut model_obj = serde_json::Map::new();
        model_obj.insert("name".to_string(), serde_json::json!(model));

        let ctx_len = context_lengths
            .get(model)
            .and_then(|ctx_len| *ctx_len)
            .unwrap_or(OPENCODE_DEFAULT_CONTEXT_LIMIT);
        let limit = serde_json::json!({
            "context": ctx_len,
            "output": OPENCODE_OUTPUT_LIMIT.min(ctx_len),
        });
        model_obj.insert("limit".to_string(), limit);

        models.insert(model.clone(), serde_json::Value::Object(model_obj));
    }

    // Build provider object with explicit field order: name, npm, options, then models
    let mut mesh_provider = serde_json::Map::new();
    mesh_provider.insert("name".to_string(), serde_json::json!("mesh-llm"));
    mesh_provider.insert(
        "npm".to_string(),
        serde_json::json!("@ai-sdk/openai-compatible"),
    );
    mesh_provider.insert(
        "options".to_string(),
        serde_json::json!({
            "baseURL": api_base_url,
        }),
    );
    mesh_provider.insert("models".to_string(), serde_json::Value::Object(models));

    let config = serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            OPENCODE_PROVIDER_ID: serde_json::Value::Object(mesh_provider),
        },
        "mcp": {
            MESH_MCP_SERVER_ID: mesh_mcp_opencode_config(mcp_url),
        }
    });

    OpenCodeLaunchSpec {
        provider_id: OPENCODE_PROVIDER_ID,
        model: format!("{OPENCODE_PROVIDER_ID}/{resolved_model}"),
        config_content: config.to_string(),
        api_key_env: OPENCODE_API_KEY_ENV,
        api_key_value: OPENCODE_API_KEY_VALUE,
        install_hint: OPENCODE_INSTALL_HINT,
    }
}

fn opencode_missing_binary_guidance(
    chosen: &str,
    host: &str,
    spec: &OpenCodeLaunchSpec,
) -> Vec<String> {
    vec![
        "opencode not found in PATH".to_string(),
        spec.install_hint.to_string(),
        "Then rerun through mesh-llm:".to_string(),
        format!("  mesh-llm opencode --host {host} --model {chosen}"),
        "mesh-llm writes the mesh provider into your OpenCode config before launching.".to_string(),
    ]
}

fn pi_missing_binary_guidance(model_arg: &str) -> Vec<String> {
    vec![
        "pi not found in PATH.".to_string(),
        "Install: npm install -g @mariozechner/pi-coding-agent".to_string(),
        "Or run manually:".to_string(),
        format!("  pi --model {}", shell::single_quote(model_arg)),
    ]
}

fn cleanup_mesh_child(mesh_child: &mut Option<std::process::Child>) {
    if let Some(child) = mesh_child {
        eprintln!("🧹 Stopping mesh-llm node we started...");
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Ensure mesh-llm is running on `port`, then return available models, chosen model, spawned child.
async fn check_mesh(
    client: &reqwest::Client,
    port: u16,
    model: &Option<String>,
) -> Result<(Vec<String>, String, Option<std::process::Child>)> {
    let url = format!("http://127.0.0.1:{port}/v1/models");

    let mut child: Option<std::process::Child> = None;
    if client.get(&url).send().await.is_err() {
        eprintln!("🚀 No mesh-llm on port {port}; starting background auto-join node");
        let exe = std::env::current_exe().unwrap_or_else(|_| "mesh-llm".into());
        child = Some(
            std::process::Command::new(&exe)
                .args(["client", "--auto", "--port", &port.to_string()])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .context("Failed to start mesh-llm node")?,
        );
    }

    let models_url = format!("http://127.0.0.1:{port}/v1/models");
    let mut models = Vec::new();
    for attempt in 0..40 {
        if let Ok(resp) = client.get(&models_url).send().await
            && let Ok(body) = resp.json::<serde_json::Value>().await
        {
            models = body["data"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|model| model["id"].as_str().map(String::from))
                .collect();
            if !models.is_empty() {
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        if attempt % 5 == 4 {
            eprintln!(
                "⏳ Waiting for mesh/models... ({:.0}s)",
                (attempt + 1) as f64 * 3.0
            );
        }
    }

    if models.is_empty() {
        if let Some(mut child) = child {
            let _ = child.kill();
            let _ = child.wait();
        }
        anyhow::bail!(
            "mesh-llm on port {port} has no models yet (or could not be reached).\n\
             Ensure at least one serving peer is available on the mesh."
        );
    }

    let chosen = choose_requested_or_agent_model(&models, model, &mut child)?;
    eprintln!("   Models: {}", models.join(", "));
    eprintln!("   Using: {chosen}");
    Ok((models, chosen, child))
}

fn choose_requested_or_agent_model(
    models: &[String],
    requested_model: &Option<String>,
    mesh_child: &mut Option<std::process::Child>,
) -> Result<String> {
    if let Some(model) = requested_model {
        if models.iter().any(|name| name == model) {
            return Ok(model.clone());
        }
        if let Some(mut child) = mesh_child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        anyhow::bail!(
            "Model '{}' not available. Available: {}",
            model,
            models.join(", ")
        );
    }

    Ok(choose_agent_model(models))
}

fn choose_agent_model(models: &[String]) -> String {
    models
        .iter()
        .find(|name| {
            let lower = name.to_ascii_lowercase();
            lower.contains("coder") || lower.contains("code") || lower.contains("qwen")
        })
        .cloned()
        .unwrap_or_else(|| models[0].clone())
}

async fn fetch_mesh_models(
    client: &reqwest::Client,
    models_url: &str,
    requested_model: &Option<String>,
) -> Result<(Vec<String>, String)> {
    let resp = client
        .get(models_url)
        .send()
        .await
        .with_context(|| format!("Failed to reach mesh target at {models_url}"))?;

    let body = resp
        .error_for_status()
        .with_context(|| format!("mesh target returned an error for {models_url}"))?
        .json::<serde_json::Value>()
        .await
        .with_context(|| format!("Failed to parse model list from {models_url}"))?;

    let models: Vec<String> = body["data"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|m| m["id"].as_str().map(String::from))
        .collect();

    if models.is_empty() {
        anyhow::bail!(
            "mesh target at {models_url} has no models yet (or could not be reached).\n\
             Ensure at least one serving peer is available on the mesh."
        );
    }

    let chosen = if let Some(model) = requested_model {
        if !models.iter().any(|name| name == model) {
            anyhow::bail!(
                "Model '{}' not available. Available: {}",
                model,
                models.join(", ")
            );
        }
        model.clone()
    } else {
        // Pre-startup path: no live routing metrics yet, so candidates
        // are scored as cold (uniform weight).
        choose_agent_model(&models)
    };

    eprintln!("   Models: {}", models.join(", "));
    eprintln!("   Using: {chosen}");

    Ok((models, chosen))
}

pub async fn run_goose(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (models, chosen, mut mesh_child) = check_mesh(&client, port, &model).await?;

    let goose_config_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".config")
        .join("goose")
        .join("custom_providers");
    std::fs::create_dir_all(&goose_config_dir)?;

    let provider_models: Vec<serde_json::Value> = models
        .iter()
        .map(|name| serde_json::json!({"name": name, "context_limit": 65536}))
        .collect();

    let provider = serde_json::json!({
        "name": "mesh",
        "engine": "openai",
        "display_name": "mesh-llm",
        "description": "Distributed LLM inference via mesh-llm",
        "api_key_env": "",
        "base_url": format!("http://localhost:{port}"),
        "models": provider_models,
        "timeout_seconds": 600,
        "supports_streaming": true,
        "requires_auth": false
    });

    let provider_path = goose_config_dir.join("mesh.json");
    std::fs::write(&provider_path, serde_json::to_string_pretty(&provider)?)?;
    eprintln!("✅ Wrote {}", provider_path.display());
    write_goose_mcp_config(DEFAULT_MESH_MCP_URL)?;
    install_skills_for_agent(SkillAgent::Goose);

    let goose_app = std::path::Path::new("/Applications/Goose.app");
    if goose_app.exists() {
        eprintln!("🪿 Launching Goose.app...");
        std::process::Command::new("open")
            .arg("-a")
            .arg(goose_app)
            .env("GOOSE_PROVIDER", "mesh")
            .env("GOOSE_MODEL", &chosen)
            .spawn()?;
        if mesh_child.is_some() {
            eprintln!(
                "ℹ️  mesh-llm node running in background (kill manually or use `mesh-llm stop`)"
            );
        }
    } else {
        eprintln!("🪿 Launching goose session...");
        let mut command = Command::new("goose");
        command
            .arg("session")
            .env("GOOSE_PROVIDER", "mesh")
            .env("GOOSE_MODEL", &chosen);
        configure_interactive_stdio(&mut command);
        let status = command.status();
        match status {
            Ok(s) if s.success() => {}
            Ok(s) => eprintln!("goose exited with {s}"),
            Err(_) => {
                eprintln!("goose not found. Install: https://github.com/block/goose");
                eprintln!("Or run manually:");
                eprintln!("  GOOSE_PROVIDER=mesh GOOSE_MODEL={chosen} goose session");
            }
        }
        if let Some(ref mut c) = mesh_child {
            eprintln!("🧹 Stopping mesh-llm node we started...");
            let _ = c.kill();
            let _ = c.wait();
        }
    }
    Ok(())
}

pub async fn run_claude(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (_models, chosen, mut mesh_child) = check_mesh(&client, port, &model).await?;

    let base_url = format!("http://127.0.0.1:{port}");
    let settings = serde_json::json!({
        "env": {
            "ANTHROPIC_BASE_URL": &base_url,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": &chosen,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": &chosen,
            "CLAUDE_CODE_SUBAGENT_MODEL": &chosen,
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "DISABLE_PROMPT_CACHING": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1"
        },
        "attribution": {
            "commit": "",
            "pr": ""
        },
        "prefersReducedMotion": true,
        "terminalProgressBarEnabled": false
    });
    let settings_json = serde_json::to_string(&settings)?;
    let mcp_config_json = mesh_mcp_claude_config_json(DEFAULT_MESH_MCP_URL)?;
    install_skills_for_agent(SkillAgent::Claude);

    eprintln!("🚀 Launching Claude Code with {chosen} → {base_url}\n");
    let mut command = Command::new("claude");
    command.args([
        "--model",
        &chosen,
        "--settings",
        &settings_json,
        "--mcp-config",
        &mcp_config_json,
    ]);
    configure_interactive_stdio(&mut command);
    let status = command.status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("claude exited with {s}"),
        Err(_) => {
            eprintln!("claude not found. Install: https://docs.anthropic.com/en/docs/claude-code");
            eprintln!("Or run manually:");
            eprintln!("  ANTHROPIC_BASE_URL={base_url} ANTHROPIC_API_KEY= claude --model {chosen}");
        }
    }
    if let Some(ref mut c) = mesh_child {
        eprintln!("🧹 Stopping mesh-llm node we started...");
        let _ = c.kill();
        let _ = c.wait();
    }
    Ok(())
}

fn resolve_pi_models_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".pi")
        .join("agent")
        .join("models.json")
}

#[cfg(test)]
fn build_pi_provider_config(model_names: &[String], api_base_url: &str) -> serde_json::Value {
    build_pi_provider_config_with_limits(
        model_names,
        api_base_url,
        &std::collections::HashMap::new(),
    )
}

fn build_pi_provider_config_with_limits(
    model_names: &[String],
    api_base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
) -> serde_json::Value {
    let models: Vec<serde_json::Value> = model_names
        .iter()
        .map(|name| {
            let mut model = serde_json::Map::new();
            model.insert("id".to_string(), serde_json::json!(name));
            model.insert("name".to_string(), serde_json::json!(name));

            if let Some(&Some(ctx_len)) = context_lengths.get(name) {
                model.insert("contextWindow".to_string(), serde_json::json!(ctx_len));
                model.insert("maxTokens".to_string(), serde_json::json!(ctx_len));
            }

            serde_json::Value::Object(model)
        })
        .collect();

    let mut provider = serde_json::Map::new();
    provider.insert("api".to_string(), serde_json::json!("openai-completions"));
    provider.insert("apiKey".to_string(), serde_json::json!("mesh"));
    provider.insert("baseUrl".to_string(), serde_json::json!(api_base_url));
    provider.insert(
        "compat".to_string(),
        serde_json::json!({
            "supportsStore": false,
            "supportsDeveloperRole": false,
            "supportsUsageInStreaming": true,
        }),
    );
    provider.insert("models".to_string(), serde_json::Value::Array(models));

    serde_json::Value::Object(provider)
}

fn load_existing_config(path: &std::path::Path) -> Result<serde_json::Value> {
    if !path.exists() {
        return Ok(serde_json::json!({}));
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let config: serde_json::Value = parse_config_content(path, &content)?;

    if !config.is_object() {
        anyhow::bail!("Expected {} to contain a JSON object", path.display());
    }

    Ok(config)
}

fn parse_config_content(path: &std::path::Path, content: &str) -> Result<serde_json::Value> {
    if path.extension().and_then(|ext| ext.to_str()) == Some("jsonc") {
        json5::from_str(content).with_context(|| {
            format!(
                "Failed to parse {} as JSONC-compatible OpenCode config",
                path.display()
            )
        })
    } else {
        serde_json::from_str(content)
            .with_context(|| format!("Failed to parse {} as JSON", path.display()))
    }
}

fn provider_map_mut<'a>(
    config: &'a mut serde_json::Value,
    field_name: &str,
    path: &std::path::Path,
) -> Result<&'a mut serde_json::Map<String, serde_json::Value>> {
    let config_object = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Expected {} to contain a JSON object", path.display()))?;
    let providers = config_object
        .entry(field_name.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));

    providers.as_object_mut().ok_or_else(|| {
        anyhow::anyhow!(
            "Expected '{}' in {} to be a JSON object",
            field_name,
            path.display()
        )
    })
}

fn merge_provider(
    config: &mut serde_json::Value,
    field_name: &str,
    provider_id: &str,
    provider: serde_json::Value,
    path: &std::path::Path,
) -> Result<()> {
    provider_map_mut(config, field_name, path)?.insert(provider_id.to_string(), provider);
    Ok(())
}

fn write_pi_config_with_limits(
    model_names: &[String],
    api_base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
) -> Result<()> {
    let models_path = resolve_pi_models_path();
    write_pi_config_to_path_with_limits(&models_path, model_names, api_base_url, context_lengths)
}

#[cfg(test)]
fn write_pi_config_to_path(
    models_path: &std::path::Path,
    model_names: &[String],
    api_base_url: &str,
) -> Result<()> {
    write_pi_config_to_path_with_limits(
        models_path,
        model_names,
        api_base_url,
        &std::collections::HashMap::new(),
    )
}

fn write_pi_config_to_path_with_limits(
    models_path: &std::path::Path,
    model_names: &[String],
    api_base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
) -> Result<()> {
    std::fs::create_dir_all(models_path.parent().expect("models path must have parent"))?;

    let mut config = load_existing_config(models_path)?;
    let provider = build_pi_provider_config_with_limits(model_names, api_base_url, context_lengths);
    merge_provider(&mut config, "providers", "mesh", provider, models_path)?;

    std::fs::write(models_path, serde_json::to_string_pretty(&config)?)?;
    eprintln!(
        "✅ Wrote mesh provider to {} ({} models)",
        models_path.display(),
        model_names.len()
    );

    Ok(())
}

#[cfg(test)]
fn write_pi_config_for_test(
    models_path: &std::path::Path,
    model_names: &[String],
    host: &str,
) -> Result<()> {
    let target = normalize_mesh_host(host)?;
    write_pi_config_to_path(models_path, model_names, &target.api_base_url)
}

pub async fn run_pi(model: Option<String>, host: &str, write: bool) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let target = normalize_mesh_host(host)?;

    let (models, chosen, mut mesh_child) = if target.auto_start_local_mesh {
        let port = target
            .local_port
            .ok_or_else(|| anyhow::anyhow!("Pi host '{}' is missing a usable port", host))?;
        let (models, chosen, child) = check_mesh(&client, port, &model).await?;
        (models, chosen, child)
    } else {
        let (models, chosen) = fetch_mesh_models(&client, &target.api_models_url, &model).await?;
        (models, chosen, None)
    };

    let context_lengths = fetch_model_context_lengths(&client, &target.management_models_url).await;
    let result = run_pi_with_mesh(
        &models,
        &chosen,
        &target.api_base_url,
        &context_lengths,
        write,
    );

    cleanup_mesh_child(&mut mesh_child);

    result
}

fn run_pi_with_mesh(
    model_names: &[String],
    chosen: &str,
    base_url: &str,
    context_lengths: &std::collections::HashMap<String, Option<u32>>,
    write: bool,
) -> Result<()> {
    write_pi_config_with_limits(model_names, base_url, context_lengths)?;
    install_skills_for_agent(SkillAgent::Pi);

    if write {
        return Ok(());
    }

    let model_arg = format!("mesh/{chosen}");
    eprintln!("🚀 Launching pi with {chosen} → {base_url}\n");
    let mut command = Command::new("pi");
    command.args(["--model", &model_arg]);
    configure_interactive_stdio(&mut command);
    let status = command.status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("pi exited with {s}"),
        Err(_) => {
            for line in pi_missing_binary_guidance(&model_arg) {
                eprintln!("{line}");
            }
        }
    }

    Ok(())
}

pub async fn run_opencode(model: Option<String>, host: &str, write: bool) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let target = normalize_opencode_host(host)?;

    let (models, chosen, mut mesh_child) = if target.auto_start_local_mesh {
        let port = target
            .local_port
            .ok_or_else(|| anyhow::anyhow!("OpenCode host '{}' is missing a usable port", host))?;
        let (models, chosen, child) = check_mesh(&client, port, &model).await?;
        (models, chosen, child)
    } else {
        let (models, chosen) = fetch_mesh_models(&client, &target.api_models_url, &model).await?;
        (models, chosen, None)
    };

    let result = if write {
        install_skills_for_agent(SkillAgent::Opencode);
        write_opencode_config(&client, &models, &chosen, &target).await
    } else {
        let context_lengths =
            fetch_model_context_lengths(&client, &target.management_models_url).await;
        match write_opencode_config(&client, &models, &chosen, &target).await {
            Ok(()) => {
                let spec = build_opencode_launch_spec_with_limits(
                    &models,
                    &chosen,
                    &target.api_base_url,
                    &target.mcp_url,
                    &context_lengths,
                );

                eprintln!(
                    "🚀 Launching OpenCode with {} → {}\n",
                    chosen, target.api_base_url
                );
                install_skills_for_agent(SkillAgent::Opencode);
                let mut command = Command::new("opencode");
                configure_opencode_launch_command(&mut command, &spec);
                let status = command.status();
                match status {
                    Ok(s) if s.success() => {}
                    Ok(s) => eprintln!("opencode exited with {s}"),
                    Err(_) => {
                        for line in opencode_missing_binary_guidance(&chosen, &target.input, &spec)
                        {
                            eprintln!("{line}");
                        }
                    }
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    };

    cleanup_mesh_child(&mut mesh_child);

    result
}

fn resolve_opencode_config_path() -> Result<std::path::PathBuf> {
    let home_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .to_path_buf();
    resolve_opencode_config_path_from_home(&home_dir)
}

fn resolve_opencode_config_path_from_home(
    home_dir: &std::path::Path,
) -> Result<std::path::PathBuf> {
    let config_dir = home_dir.join(".config").join("opencode");

    std::fs::create_dir_all(&config_dir)?;

    let json_path = config_dir.join("opencode.json");
    let jsonc_path = config_dir.join("opencode.jsonc");

    if json_path.exists() {
        return Ok(json_path);
    }
    if jsonc_path.exists() {
        return Ok(jsonc_path);
    }

    Ok(json_path)
}

fn merge_mesh_provider(
    config: &mut serde_json::Value,
    mesh_provider: serde_json::Value,
    config_path: &std::path::Path,
) -> Result<()> {
    merge_provider(config, "provider", "mesh", mesh_provider, config_path)
}

async fn fetch_model_context_lengths(
    client: &reqwest::Client,
    management_models_url: &str,
) -> std::collections::HashMap<String, Option<u32>> {
    let models_json = fetch_json(client, management_models_url).await;

    // Query /api/runtime/processes for the actual running context_lengths.
    let processes_url = management_models_url.replace("/api/models", "/api/runtime/processes");
    let processes_json = fetch_json(client, &processes_url).await;

    merge_context_lengths(&models_json, &processes_json)
}

async fn fetch_json(client: &reqwest::Client, url: &str) -> serde_json::Value {
    match client.get(url).send().await {
        Ok(resp) => resp.json::<serde_json::Value>().await.unwrap_or_default(),
        Err(_) => serde_json::Value::Null,
    }
}

fn merge_context_lengths(
    models_json: &serde_json::Value,
    processes_json: &serde_json::Value,
) -> std::collections::HashMap<String, Option<u32>> {
    let mut context_map = std::collections::HashMap::new();

    // Primary source: runtime process data — the actual context_length the
    // model is running with (from CLI --ctx-size, config.toml, or auto-computed
    // from VRAM by plan_runtime_resources).
    if let Some(processes) = processes_json["processes"].as_array() {
        for process in processes {
            let name = process["name"].as_str().map(String::from);
            let ctx_len = process["context_length"].as_u64().map(|v| v as u32);
            if let (Some(n), Some(ctx_len)) = (name, ctx_len) {
                context_map.insert(n, Some(ctx_len));
            }
        }
    }

    // Fallback: GGUF metadata / peer metadata for any model whose runtime
    // context_length is unknown (e.g. remote models or stopped instances).
    if let Some(mesh_models) = models_json["mesh_models"].as_array() {
        for model in mesh_models {
            let name = model["name"].as_str().map(String::from);
            let ctx_len = model["context_length"].as_u64().map(|v| v as u32);
            if let Some(n) = name {
                context_map.entry(n).or_insert(ctx_len);
            }
        }
    }

    context_map
}

async fn write_opencode_config(
    client: &reqwest::Client,
    model_names: &[String],
    resolved_model: &str,
    target: &OpenCodeTarget,
) -> Result<()> {
    let config_path = resolve_opencode_config_path()?;
    write_opencode_config_to_path(client, model_names, resolved_model, target, &config_path).await
}

async fn write_opencode_config_to_path(
    client: &reqwest::Client,
    model_names: &[String],
    resolved_model: &str,
    target: &OpenCodeTarget,
    config_path: &std::path::Path,
) -> Result<()> {
    std::fs::create_dir_all(config_path.parent().expect("config path must have parent"))?;

    let existing_config = load_existing_config(config_path)?;

    let context_lengths = fetch_model_context_lengths(client, &target.management_models_url).await;

    let spec = build_opencode_launch_spec_with_limits(
        model_names,
        resolved_model,
        &target.api_base_url,
        &target.mcp_url,
        &context_lengths,
    );
    let config_value: serde_json::Value = serde_json::from_str(&spec.config_content)?;
    let mesh_provider = config_value["provider"]["mesh"].clone();
    let mesh_mcp = config_value["mcp"]["mesh"].clone();

    // Merge schema if needed (for display in ordered format)
    let mut merged_config = existing_config.clone();
    let schema = config_value
        .get("$schema")
        .filter(|_| merged_config.get("$schema").is_none());
    if let Some(schema) = schema {
        merged_config
            .as_object_mut()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Expected {} to contain a JSON object",
                    config_path.display()
                )
            })?
            .insert("$schema".to_string(), schema.clone());
    }

    merge_mesh_provider(&mut merged_config, mesh_provider.clone(), config_path)?;
    merge_provider(&mut merged_config, "mcp", "mesh", mesh_mcp, config_path)?;

    let formatted_json = serde_json::to_string_pretty(&merged_config)?;
    std::fs::write(config_path, &formatted_json)?;

    eprintln!(
        "✅ Wrote {} ({} models)",
        config_path.display(),
        model_names.len()
    );

    Ok(())
}

#[cfg(test)]
pub(crate) async fn write_opencode_config_for_test(
    config_path: &std::path::Path,
    models: &[String],
    host: &str,
) -> Result<(), anyhow::Error> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let target = normalize_opencode_host(host)?;
    write_opencode_config_to_path(
        &client,
        models,
        &models.first().cloned().unwrap_or_default(),
        &target,
        config_path,
    )
    .await
}

#[cfg(test)]
pub(crate) fn build_mesh_provider_spec_for_test(
    models: &[String],
    host: &str,
) -> serde_json::Value {
    let target = normalize_opencode_host(host).expect("valid OpenCode host");
    let spec = build_opencode_launch_spec(
        models,
        &models.first().cloned().unwrap_or_default(),
        &target.api_base_url,
    );
    let config_value: serde_json::Value =
        serde_json::from_str(&spec.config_content).expect("valid JSON");
    config_value["provider"]["mesh"].clone()
}

#[cfg(test)]
mod tests;
