use anyhow::Result;

use crate::runtime;

pub(crate) async fn run_goose(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (models, chosen, mut mesh_child) = runtime::check_mesh(&client, port, &model).await?;

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
        let status = std::process::Command::new("goose")
            .arg("session")
            .env("GOOSE_PROVIDER", "mesh")
            .env("GOOSE_MODEL", &chosen)
            .status();
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

pub(crate) async fn run_claude(model: Option<String>, port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let (_models, chosen, mut mesh_child) = runtime::check_mesh(&client, port, &model).await?;

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

    eprintln!("🚀 Launching Claude Code with {chosen} → {base_url}\n");
    let status = std::process::Command::new("claude")
        .args(["--model", &chosen, "--settings", &settings_json])
        .status();
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
