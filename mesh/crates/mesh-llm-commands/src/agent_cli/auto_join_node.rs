//! The background node a launcher starts when nothing serves its port.
//!
//! The launcher runs `mesh-llm client --auto` as a child process. The child
//! joins the mesh the user asked for: the top-level joining flags
//! (`--mesh-discovery-mode`, `--join`, `--join-file`) are passed on to it.
use std::ffi::OsString;
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result};
use mesh_llm_cli::JoiningOptions;

/// Start the background node with [`client_args`].
pub(super) fn spawn(
    port: u16,
    console_port: Option<u16>,
    joining: &JoiningOptions,
) -> Result<Child> {
    let exe = std::env::current_exe().unwrap_or_else(|_| "mesh-llm".into());
    Command::new(&exe)
        .args(client_args(port, console_port, joining))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to start mesh-llm node")
}

/// Arguments for the `mesh-llm client` child.
///
/// The discovery mode is always passed, so the child never falls back to its
/// own default. `--auto` stays next to a join token: with a token configured,
/// the runtime skips discovery and joins with the token.
pub(super) fn client_args(
    port: u16,
    console_port: Option<u16>,
    joining: &JoiningOptions,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![
        "client".into(),
        "--auto".into(),
        "--port".into(),
        port.to_string().into(),
    ];
    if let Some(console_port) = console_port {
        args.push("--console".into());
        args.push(console_port.to_string().into());
    }
    args.push("--mesh-discovery-mode".into());
    args.push(joining.mesh_discovery_mode.as_str().into());
    for token in &joining.join {
        args.push("--join".into());
        args.push(token.into());
    }
    for path in &joining.join_file {
        args.push("--join-file".into());
        args.push(path.into());
    }
    args
}

#[cfg(test)]
mod tests;
