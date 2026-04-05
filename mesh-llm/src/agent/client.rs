//! ACP agent client — stdio proxy that tunnels sessions to mesh providers.
//!
//! When used as an editor's ACP agent command (e.g. in Zed's `agent_servers`),
//! this bridges stdio JSON-RPC ↔ QUIC to a remote agent provider on the mesh.
//!
//! Editor config (Zed):
//! ```json
//! {
//!   "agent_servers": {
//!     "mesh-llm": {
//!       "command": "mesh-llm",
//!       "args": ["agent", "connect", "--join", "<token>"]
//!     }
//!   }
//! }
//! ```

use anyhow::{bail, Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{info, warn};

use crate::agent::relay;
use crate::mesh;

/// Run the stdio ACP relay: read JSON-RPC from stdin, send to agent via QUIC,
/// relay responses back to stdout.
pub async fn run_stdio_relay(node: &mesh::Node) -> Result<()> {
    let (mut quic_send, mut quic_recv) = open_agent_tunnel(node).await?;

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let mut stdin_lines = BufReader::new(stdin);
    let mut stdout = stdout;

    // Forward: stdin → QUIC (agent)
    let forward = async {
        let mut line = String::new();
        loop {
            line.clear();
            match stdin_lines.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if relay::write_message(&mut quic_send, trimmed).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("stdin read error: {e}");
                    break;
                }
            }
        }
    };

    // Backward: QUIC (agent) → stdout
    let backward = async {
        loop {
            match relay::read_message(&mut quic_recv).await {
                Ok(Some(msg)) => {
                    // Write as a JSON line to stdout (ACP stdio protocol).
                    if stdout.write_all(msg.as_bytes()).await.is_err() {
                        break;
                    }
                    if stdout.write_all(b"\n").await.is_err() {
                        break;
                    }
                    if stdout.flush().await.is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    warn!("QUIC relay error: {e}");
                    break;
                }
            }
        }
    };

    tokio::select! {
        _ = forward => info!("stdin closed, ending ACP relay"),
        _ = backward => info!("agent closed, ending ACP relay"),
    }

    Ok(())
}

/// Open a QUIC bi-stream tagged STREAM_ACP to a peer that advertises agent capability.
///
/// Phase 0: tries each connected peer. Phase 1 will use gossip-advertised
/// AgentCapability for load-aware routing.
async fn open_agent_tunnel(
    node: &mesh::Node,
) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
    let peers = node.peers().await;
    if peers.is_empty() {
        bail!("no mesh peers connected — cannot find an agent provider");
    }

    // Phase 0: try the first peer. Phase 1 will filter by agent capability.
    let peer = &peers[0];
    node.open_acp_tunnel(peer.id)
        .await
        .context("open ACP tunnel to agent provider")
}
