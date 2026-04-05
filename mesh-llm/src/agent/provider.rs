//! ACP agent provider — spawns an ACP subprocess and relays sessions over QUIC.
//!
//! The provider runs on a machine that has an ACP-compatible agent installed
//! (e.g. `goose acp`, `claude-agent-acp`, `gemini --acp`). It joins the mesh,
//! advertises agent capability via gossip, and accepts incoming ACP sessions
//! tunneled from clients over QUIC STREAM_ACP (0x0A).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Notify;
use tracing::{error, info, warn};

use crate::agent::relay;

/// Known agent type shortcuts.
pub const AGENT_GOOSE: &str = "goose";
pub const AGENT_CLAUDE: &str = "claude-acp";
pub const AGENT_GEMINI: &str = "gemini";

/// Resolve an agent type name to the command + args needed to spawn it in ACP stdio mode.
pub fn resolve_agent_command(
    agent_type: &str,
    extra_args: &[String],
) -> Result<(String, Vec<String>)> {
    match agent_type {
        AGENT_GOOSE => Ok(("goose".into(), vec!["acp".into()])),
        AGENT_CLAUDE => Ok(("claude-agent-acp".into(), vec![])),
        AGENT_GEMINI => Ok(("gemini".into(), vec!["--acp".into()])),
        other => {
            // Treat as a raw command path. Extra args are appended.
            if extra_args.is_empty() {
                Ok((other.into(), vec![]))
            } else {
                Ok((other.into(), extra_args.to_vec()))
            }
        }
    }
}

/// State for a running agent provider.
pub struct AgentProvider {
    pub agent_type: String,
    pub active_sessions: AtomicU32,
    pub max_sessions: Option<u32>,
    shutdown: Notify,
}

impl AgentProvider {
    pub fn new(agent_type: String, max_sessions: Option<u32>) -> Arc<Self> {
        Arc::new(Self {
            agent_type,
            active_sessions: AtomicU32::new(0),
            max_sessions,
            shutdown: Notify::new(),
        })
    }

    /// Whether this provider can accept another session.
    pub fn can_accept(&self) -> bool {
        match self.max_sessions {
            Some(max) => self.active_sessions.load(Ordering::Relaxed) < max,
            None => true,
        }
    }

    /// Gossip-ready status.
    pub fn status(&self) -> crate::proto::node::AgentStatus {
        use crate::proto::node::AgentStatus;
        let active = self.active_sessions.load(Ordering::Relaxed);
        match self.max_sessions {
            Some(max) if active >= max => AgentStatus::AgentFull,
            _ if active > 0 => AgentStatus::AgentBusy,
            _ => AgentStatus::AgentIdle,
        }
    }

    /// Signal shutdown.
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    /// Handle one ACP session arriving over a QUIC bidirectional stream.
    ///
    /// Spawns the agent subprocess, relays messages between the QUIC stream
    /// and the subprocess stdio, and cleans up when either side closes.
    pub async fn handle_session(
        self: &Arc<Self>,
        mut quic_send: iroh::endpoint::SendStream,
        mut quic_recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        if !self.can_accept() {
            let err_msg = r#"{"jsonrpc":"2.0","error":{"code":-32000,"message":"agent at capacity"},"id":null}"#;
            relay::write_message(&mut quic_send, err_msg).await.ok();
            bail!("agent provider at capacity");
        }

        self.active_sessions.fetch_add(1, Ordering::Relaxed);
        let _guard = SessionGuard(self.clone());

        let (cmd, args) = resolve_agent_command(&self.agent_type, &[])?;
        info!(agent = %self.agent_type, cmd = %cmd, "spawning ACP agent for session");

        let mut child = Command::new(&cmd)
            .args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .with_context(|| format!("spawn ACP agent: {cmd}"))?;

        let mut child_stdin = child.stdin.take().context("agent stdin")?;
        let mut child_stdout = child.stdout.take().context("agent stdout")?;

        // Log stderr in background.
        if let Some(stderr) = child.stderr.take() {
            let agent_type = self.agent_type.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    info!(agent = %agent_type, "[agent stderr] {line}");
                }
            });
        }

        // Relay: QUIC → agent stdin
        let forward = tokio::spawn(async move {
            loop {
                match relay::read_message(&mut quic_recv).await {
                    Ok(Some(msg)) => {
                        // ACP over stdio: each message is a JSON line terminated by \n.
                        if child_stdin.write_all(msg.as_bytes()).await.is_err() {
                            break;
                        }
                        if !msg.ends_with('\n') {
                            if child_stdin.write_all(b"\n").await.is_err() {
                                break;
                            }
                        }
                        if child_stdin.flush().await.is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        warn!("ACP relay QUIC→agent error: {e}");
                        break;
                    }
                }
            }
        });

        // Relay: agent stdout → QUIC
        let backward = tokio::spawn(async move {
            let mut lines = BufReader::new(child_stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match lines.read_line(&mut line).await {
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
                        warn!("ACP relay agent→QUIC error: {e}");
                        break;
                    }
                }
            }
        });

        // Wait for either direction to finish, then clean up.
        tokio::select! {
            _ = forward => {}
            _ = backward => {}
            _ = self.shutdown.notified() => {
                info!("agent provider shutting down, killing session");
            }
        }

        kill_child(&mut child).await;
        Ok(())
    }
}

use tokio::io::AsyncWriteExt;

/// RAII guard to decrement active_sessions on drop.
struct SessionGuard(Arc<AgentProvider>);

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.0.active_sessions.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn kill_child(child: &mut Child) {
    if let Err(e) = child.kill().await {
        // Process may have already exited — that's fine.
        if e.kind() != std::io::ErrorKind::InvalidInput {
            warn!("failed to kill ACP agent subprocess: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_known_agents() {
        let (cmd, args) = resolve_agent_command(AGENT_GOOSE, &[]).unwrap();
        assert_eq!(cmd, "goose");
        assert_eq!(args, vec!["acp"]);

        let (cmd, args) = resolve_agent_command(AGENT_CLAUDE, &[]).unwrap();
        assert_eq!(cmd, "claude-agent-acp");
        assert!(args.is_empty());

        let (cmd, args) = resolve_agent_command(AGENT_GEMINI, &[]).unwrap();
        assert_eq!(cmd, "gemini");
        assert_eq!(args, vec!["--acp"]);
    }

    #[test]
    fn resolve_custom_agent() {
        let (cmd, args) =
            resolve_agent_command("/usr/local/bin/my-agent", &["--mode".into(), "acp".into()])
                .unwrap();
        assert_eq!(cmd, "/usr/local/bin/my-agent");
        assert_eq!(args, vec!["--mode", "acp"]);
    }

    #[test]
    fn provider_capacity() {
        let provider = AgentProvider::new("goose".into(), Some(2));
        assert!(provider.can_accept());
        provider.active_sessions.store(1, Ordering::Relaxed);
        assert!(provider.can_accept());
        provider.active_sessions.store(2, Ordering::Relaxed);
        assert!(!provider.can_accept());
    }

    #[test]
    fn provider_status() {
        use crate::proto::node::AgentStatus;
        let provider = AgentProvider::new("goose".into(), Some(2));
        assert_eq!(provider.status(), AgentStatus::AgentIdle);
        provider.active_sessions.store(1, Ordering::Relaxed);
        assert_eq!(provider.status(), AgentStatus::AgentBusy);
        provider.active_sessions.store(2, Ordering::Relaxed);
        assert_eq!(provider.status(), AgentStatus::AgentFull);
    }
}
