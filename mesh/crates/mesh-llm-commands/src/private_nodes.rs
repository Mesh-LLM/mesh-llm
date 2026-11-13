//! `mesh-llm private` — choose the other nodes this node's private requests
//! may go to.
//!
//! The list is `[private] nodes` in the config file, so it survives restarts
//! and can be edited by hand. A running node notices the file change and reads
//! the list again. Only node ids are written: a label or hostname is resolved
//! to the id of the connected node it names when it is added, because routing
//! trusts the id a node proves on its connection, not a name it announces.

use anyhow::{Context, Result, bail};
use mesh_llm_cli::PrivateCommand;
use mesh_llm_config::{ConfigStore, config_path, is_node_id, load_config};
use mesh_llm_events::{console_out, machine_out};
use serde_json::json;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

/// How many connected nodes that are not private `list` names; a public mesh
/// can have hundreds.
const SHOWN_OTHERS: usize = 20;

pub async fn dispatch_private_command(
    command: &PrivateCommand,
    config_override: Option<&Path>,
) -> Result<()> {
    match command {
        PrivateCommand::Add { node, port } => {
            let peers = connected_nodes(*port).await;
            add(config_override, node, peers.as_deref(), *port)
        }
        PrivateCommand::Remove { node, port } => {
            let peers = connected_nodes(*port).await;
            remove(config_override, node, peers.as_deref(), *port)
        }
        PrivateCommand::List { port, json } => {
            let peers = connected_nodes(*port).await;
            list(config_override, peers.as_deref(), *port, *json)
        }
    }
}

/// A node the running node is connected to, as its status API reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ConnectedNode {
    node_id: String,
    label: Option<String>,
    hostname: Option<String>,
}

impl ConnectedNode {
    fn name(&self) -> Option<&str> {
        self.label.as_deref().or(self.hostname.as_deref())
    }

    /// Whether `name` is this node's label or hostname, ignoring case and a
    /// hostname's domain (`studio` names `studio.local`).
    fn is_called(&self, name: &str) -> bool {
        let label = self.label.iter().map(String::as_str);
        let hostname = self
            .hostname
            .iter()
            .flat_map(|host| [host.as_str(), host.split('.').next().unwrap_or(host)]);
        label
            .chain(hostname)
            .any(|candidate| candidate.eq_ignore_ascii_case(name))
    }
}

/// The nodes the mesh-llm on `port` is connected to; `None` when it is not
/// running or does not answer.
async fn connected_nodes(port: u16) -> Option<Vec<ConnectedNode>> {
    let client = mesh_llm_system::local_http::local_http_client_builder()
        .timeout(Duration::from_secs(3))
        .build()
        .ok()?;
    let status: serde_json::Value = client
        .get(format!("http://127.0.0.1:{port}/api/status"))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()?;
    Some(connected_nodes_in(&status))
}

fn connected_nodes_in(status: &serde_json::Value) -> Vec<ConnectedNode> {
    let text = |value: &serde_json::Value| {
        value
            .as_str()
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    };
    status["peers"]
        .as_array()
        .into_iter()
        .flatten()
        // `peers` also holds nodes this one knows of but has no connection to.
        .filter(|peer| peer["connected"].as_bool() == Some(true))
        .filter_map(|peer| {
            Some(ConnectedNode {
                // Older nodes report only a short id, which cannot be listed.
                node_id: text(&peer["node_id"]).filter(|id| is_node_id(id))?,
                label: text(&peer["owner"]["node_label"]),
                hostname: text(&peer["hostname"]),
            })
        })
        .collect()
}

/// The node id `node` names: itself when it is one, else the one connected
/// node with that label or hostname.
fn resolve(node: &str, peers: Option<&[ConnectedNode]>, port: u16) -> Result<String> {
    if is_node_id(node) {
        return Ok(node.to_ascii_lowercase());
    }
    let Some(peers) = peers else {
        bail!(
            "{node:?} is not a node id, and no mesh-llm is answering on port {port} to look it up. \
             Pass the node id (64 hex characters) instead; `mesh-llm auth status` on that node prints it."
        );
    };
    let matches: Vec<&ConnectedNode> = peers.iter().filter(|peer| peer.is_called(node)).collect();
    match matches.as_slice() {
        [peer] => Ok(peer.node_id.clone()),
        [] => bail!(
            "no connected node is called {node:?}. `mesh-llm private list` shows the connected nodes; \
             you can also pass a node id (64 hex characters)."
        ),
        _ => bail!(
            "more than one connected node is called {node:?}: {}. Pass the node id instead.",
            matches
                .iter()
                .map(|peer| peer.node_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn listed_nodes(config_override: Option<&Path>) -> Result<Vec<String>> {
    Ok(load_config(config_override)?
        .private
        .nodes
        .into_iter()
        .map(|id| id.trim().to_ascii_lowercase())
        .collect())
}

/// Change `[private] nodes` in the file at `path`, keeping the rest of the
/// file. The read and the rewrite happen under the config file's lock, which a
/// running node's config applies take too, so `change` gets the list as the
/// file holds it and no other edit lands before it is written; nothing is
/// written when it fails.
fn update_nodes<T>(path: &Path, change: impl FnOnce(&mut Vec<String>) -> Result<T>) -> Result<T> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }
    let mut outcome = None;
    ConfigStore::open(path)
        .edit_preserving(|document| {
            let mut nodes = nodes_in(document);
            outcome = Some(change(&mut nodes)?);
            let private = document
                .entry("private")
                .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()));
            if private.as_table().is_none() {
                *private = toml_edit::Item::Table(toml_edit::Table::new());
            }
            private["nodes"] = toml_edit::value(nodes.iter().collect::<toml_edit::Array>());
            Ok(())
        })
        .with_context(|| format!("failed to update {}", path.display()))?;
    outcome.context("the config edit did not run")
}

/// `[private] nodes` as `document` holds it, normalized.
fn nodes_in(document: &toml_edit::DocumentMut) -> Vec<String> {
    document
        .get("private")
        .and_then(|private| private.get("nodes"))
        .and_then(toml_edit::Item::as_array)
        .into_iter()
        .flatten()
        .filter_map(|id| id.as_str())
        .map(|id| id.trim().to_ascii_lowercase())
        .collect()
}

fn described(id: &str, peers: Option<&[ConnectedNode]>) -> String {
    match peers
        .into_iter()
        .flatten()
        .find(|peer| peer.node_id == id)
        .and_then(ConnectedNode::name)
    {
        Some(name) => format!("{name} ({id})"),
        None => id.to_string(),
    }
}

fn add(
    config_override: Option<&Path>,
    node: &str,
    peers: Option<&[ConnectedNode]>,
    port: u16,
) -> Result<()> {
    let id = resolve(node, peers, port)?;
    let path = config_path(config_override)?;
    let added = update_nodes(&path, |nodes| {
        let added = !nodes.contains(&id);
        if added {
            nodes.push(id.clone());
        }
        Ok(added)
    })?;
    let mut out = console_out();
    if !added {
        writeln!(
            out,
            "{} is already a private node in {}",
            described(&id, peers),
            path.display()
        )?;
        return Ok(());
    }
    writeln!(
        out,
        "Private requests may now go to {}. Written to {}",
        described(&id, peers),
        path.display()
    )?;
    writeln!(out, "A running node picks this up within a few seconds.")?;
    Ok(())
}

/// The listed id `node` names: an id, a unique prefix of one, or the label or
/// hostname of a connected node that is listed.
fn listed_match(nodes: &[String], node: &str, peers: Option<&[ConnectedNode]>) -> Result<String> {
    let wanted = node.trim().to_ascii_lowercase();
    if nodes.contains(&wanted) {
        return Ok(wanted);
    }
    // A prefix and a name can both match: take every listed node named
    // either way, so two different nodes are refused rather than guessed.
    let is_prefix = !wanted.is_empty() && wanted.chars().all(|ch| ch.is_ascii_hexdigit());
    let matches: Vec<&String> = nodes
        .iter()
        .filter(|id| {
            (is_prefix && id.starts_with(&wanted))
                || peers
                    .into_iter()
                    .flatten()
                    .any(|peer| &peer.node_id == *id && peer.is_called(node))
        })
        .collect();
    match matches.as_slice() {
        [id] => Ok((*id).clone()),
        [] => bail!(
            "{node:?} is not a private node. `mesh-llm private list` shows the ones there are."
        ),
        _ => bail!("{node:?} names more than one private node. Pass more of the node id."),
    }
}

fn remove(
    config_override: Option<&Path>,
    node: &str,
    peers: Option<&[ConnectedNode]>,
    port: u16,
) -> Result<()> {
    let path = config_path(config_override)?;
    let (id, left) = update_nodes(&path, |nodes| {
        let id = match listed_match(nodes, node, peers) {
            Ok(id) => id,
            Err(error) if peers.is_none() && !is_node_id(node) => {
                return Err(error.context(format!(
                    "no mesh-llm is answering on port {port} to look up connected node names"
                )));
            }
            Err(error) => return Err(error),
        };
        nodes.retain(|listed| listed != &id);
        Ok((id, nodes.len()))
    })?;
    let mut out = console_out();
    writeln!(
        out,
        "Private requests no longer go to {}. Written to {}",
        described(&id, peers),
        path.display()
    )?;
    if left == 0 {
        writeln!(
            out,
            "No private nodes are left: private requests stay on this node."
        )?;
    }
    writeln!(out, "A running node picks this up within a few seconds.")?;
    Ok(())
}

fn list(
    config_override: Option<&Path>,
    peers: Option<&[ConnectedNode]>,
    port: u16,
    json: bool,
) -> Result<()> {
    let path = config_path(config_override)?;
    let nodes = listed_nodes(config_override)?;
    let connected = |id: &str| peers.map(|peers| peers.iter().any(|peer| peer.node_id == id));
    let name = |id: &str| {
        peers
            .into_iter()
            .flatten()
            .find(|peer| peer.node_id == id)
            .and_then(ConnectedNode::name)
            .map(str::to_string)
    };

    if json {
        let payload = json!({
            "config_path": path.display().to_string(),
            "nodes": nodes.iter().map(|id| json!({
                "node_id": id,
                "name": name(id),
                "connected": connected(id),
            })).collect::<Vec<_>>(),
        });
        writeln!(machine_out(), "{}", serde_json::to_string_pretty(&payload)?)?;
        return Ok(());
    }

    let mut out = console_out();
    if nodes.is_empty() {
        writeln!(out, "No private nodes: private requests stay on this node.")?;
        writeln!(
            out,
            "Add one with `mesh-llm private add <node id or name>`."
        )?;
    } else {
        writeln!(out, "Private requests may go to this node and:")?;
        for id in &nodes {
            let state = match connected(id) {
                Some(true) => "connected",
                Some(false) => "not connected",
                None => "unknown",
            };
            writeln!(
                out,
                "  {:<16} {id}  {state}",
                name(id).unwrap_or_else(|| "-".to_string())
            )?;
        }
    }
    if peers.is_none() {
        writeln!(
            out,
            "\nNo mesh-llm is answering on port {port}, so connection state is unknown."
        )?;
    } else if let Some(others) = peers
        .map(|peers| {
            peers
                .iter()
                .filter(|peer| !nodes.contains(&peer.node_id))
                .collect::<Vec<_>>()
        })
        .filter(|others| !others.is_empty())
    {
        writeln!(out, "\nConnected, not private nodes:")?;
        for peer in others.iter().take(SHOWN_OTHERS) {
            writeln!(out, "  {:<16} {}", peer.name().unwrap_or("-"), peer.node_id)?;
        }
        if others.len() > SHOWN_OTHERS {
            writeln!(out, "  and {} more", others.len() - SHOWN_OTHERS)?;
        }
    }
    writeln!(out, "\nConfig: {}", path.display())?;
    Ok(())
}

#[cfg(test)]
#[path = "private_nodes/tests.rs"]
mod tests;
