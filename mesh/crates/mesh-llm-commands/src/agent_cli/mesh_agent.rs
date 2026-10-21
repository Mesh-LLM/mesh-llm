//! `mesh-llm agent -t "..."`: a one-shot agent on the mesh.

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use mesh_llm_agent::{AgentOptions, AgentUpdate};

use super::check_mesh;

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
    let (_models, model, mut mesh_child) = check_mesh(&client, args.port, &args.model).await?;

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

    if let Some(child) = mesh_child.as_mut() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let outcome = outcome?;
    let mut out = mesh_llm_events::console_out();
    writeln!(out, "{}", outcome.final_text.trim())?;
    Ok(())
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
