//! `mesh-llm agent -t "..."`: a one-shot agent on the mesh.

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use mesh_llm_agent::{AgentOptions, AgentUpdate};

use super::model_inventory::ModelInventory;

pub struct MeshAgentArgs {
    pub text: String,
    pub model: Option<String>,
    pub port: u16,
    pub max_turns: u32,
    pub cwd: Option<PathBuf>,
    pub quiet: bool,
}

const PREVIEW_LINES: usize = 8;

pub async fn run(args: MeshAgentArgs) -> Result<()> {
    let instructions = if args.text == "-" {
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .context("reading instructions from stdin")?;
        text
    } else {
        args.text.clone()
    };
    let cwd = match args.cwd {
        Some(cwd) => cwd,
        None => std::env::current_dir()?,
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let model = pick_model(&client, args.port, args.model.as_deref()).await?;

    let options = AgentOptions {
        base_url: format!("http://127.0.0.1:{}", args.port),
        model,
        cwd,
        max_turns: args.max_turns,
    };
    let quiet = args.quiet;
    let outcome = mesh_llm_agent::run(options, &instructions, |update| {
        if !quiet {
            progress(update);
        }
    })
    .await;

    let outcome = outcome?;
    let mut out = mesh_llm_events::console_out();
    writeln!(out, "{}", outcome.final_text.trim())?;
    Ok(())
}

/// Use the mesh-llm node already running on `port`; never start one.
/// With no `--model`, ask for the `mesh` virtual model so routing picks a
/// target that can fit the request.
async fn pick_model(
    client: &reqwest::Client,
    port: u16,
    requested: Option<&str>,
) -> Result<String> {
    let url = format!("http://127.0.0.1:{port}/v1/models");
    let body: serde_json::Value = client
        .get(&url)
        .send()
        .await
        .and_then(|resp| resp.error_for_status())
        .map_err(|_| {
            anyhow::anyhow!(
                "no mesh-llm API on port {port}. Start one first \
                 (e.g. `mesh-llm client --auto`), or pass --port"
            )
        })?
        .json()
        .await
        .context("reading /v1/models")?;
    let names = ModelInventory::from_response(&body).names;
    let model = match requested {
        Some(model) if names.iter().any(|name| name == model) => model.to_string(),
        Some(model) => anyhow::bail!(
            "model '{model}' not available on port {port}. Available: {}",
            names.join(", ")
        ),
        None if names.iter().any(|name| name == "mesh") => "mesh".to_string(),
        None => names
            .first()
            .cloned()
            .with_context(|| format!("mesh-llm on port {port} has no models yet"))?,
    };
    let _ = writeln!(mesh_llm_events::console_err(), "   Using: {model}");
    Ok(model)
}

fn progress(update: AgentUpdate) {
    let mut err = mesh_llm_events::console_err();
    let _ = match update {
        AgentUpdate::ToolCall { name, arguments } => writeln!(err, "▶ {name} {arguments}"),
        AgentUpdate::ToolResult { error, text } => {
            let mark = if error { "✗" } else { "✓" };
            let mut lines = text.lines();
            let preview: Vec<&str> = lines.by_ref().take(PREVIEW_LINES).collect();
            let more = lines.count();
            let tail = if more > 0 {
                format!("\n  … {more} more lines")
            } else {
                String::new()
            };
            writeln!(err, "{mark} {}{tail}", preview.join("\n  "))
        }
        AgentUpdate::Text(_) => Ok(()),
    };
}
