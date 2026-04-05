//! CLI handlers for `mesh-llm agent` subcommands.

use anyhow::Result;

use crate::agent;
use crate::cli::AgentCommand;

pub(crate) async fn dispatch_agent_command(command: &AgentCommand) -> Result<()> {
    match command {
        AgentCommand::Serve {
            agent_type,
            args,
            max_sessions,
        } => run_serve(agent_type, args, *max_sessions).await,
        AgentCommand::Connect { port } => run_connect(*port).await,
        AgentCommand::List { port } => run_list(*port).await,
    }
}

async fn run_serve(agent_type: &str, args: &[String], max_sessions: Option<u32>) -> Result<()> {
    let (cmd, resolved_args) = agent::provider::resolve_agent_command(agent_type, args)?;
    eprintln!("🤖 Agent provider: {agent_type}");
    eprintln!("   Command: {cmd} {}", resolved_args.join(" "));
    if let Some(max) = max_sessions {
        eprintln!("   Max sessions: {max}");
    }
    eprintln!();
    eprintln!("⚠️  Agent serving is not yet wired into the mesh runtime.");
    eprintln!("   This will be connected in the next iteration.");
    eprintln!("   The agent provider needs to:");
    eprintln!("   1. Join the mesh (use --join or --discover flags on the parent command)");
    eprintln!("   2. Advertise agent capability via gossip");
    eprintln!("   3. Accept incoming STREAM_ACP sessions");
    Ok(())
}

async fn run_connect(_port: u16) -> Result<()> {
    eprintln!("🤖 ACP agent client (stdio mode)");
    eprintln!("   Reads JSON-RPC from stdin, tunnels to mesh agent provider via QUIC");
    eprintln!();
    eprintln!("⚠️  Agent connect is not yet wired into the mesh runtime.");
    eprintln!("   This will be connected in the next iteration.");
    eprintln!("   The client needs a running mesh-llm node to tunnel through.");
    eprintln!();
    eprintln!("   Editor config (Zed):");
    eprintln!(
        r#"   {{"agent_servers": {{"mesh-llm": {{"command": "mesh-llm", "args": ["agent", "connect", "--join", "<token>"]}}}}}}"#
    );
    Ok(())
}

async fn run_list(port: u16) -> Result<()> {
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    let resp = client
        .get(format!("{base}/api/status"))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    // Look for peers advertising agent capability.
    let peers = resp
        .get("peers")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();

    let mut found = false;
    for peer in &peers {
        if let Some(agent) = peer.get("agent") {
            found = true;
            let agent_type = agent
                .get("agent_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let status = agent
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let active = agent
                .get("active_sessions")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hostname = peer.get("hostname").and_then(|v| v.as_str()).unwrap_or("?");
            eprintln!("  🤖 {agent_type} on {hostname} — {status} ({active} active sessions)");
        }
    }

    if !found {
        eprintln!("No agents found on the mesh.");
        eprintln!("To share an agent: mesh-llm agent serve goose");
    }

    Ok(())
}
